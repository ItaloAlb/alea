// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Validation gate L2: the joint e+h
// exchange move samples a Gaussian electron-hole model exactly at finite M.
//
// Model: electrons (m_e = 1) and holes (m_h = 2) in a 2D harmonic trap
// V_trap = sum_i m_i w^2 r_i^2 / 2 plus an e-h attraction
// V_eh = (kappa/2) sum_{e,h} |r_e - r_h|^2.  The primitive action is then
// quadratic for every pair of permutations P = (P_e, P_h): per dimension
// S_P = x^T A_P x / 2 with A_P = K_P / tau + tau U, K_P the spring Laplacian of
// the permuted links and U the potential Hessian on each slice, so
//   ln Z_P = (d/2) [sum_i M ln(m_i / 2 pi tau) + N M ln 2 pi - ln det A_P],
//   d ln Z_P / d tau = (d/2) [-N M / tau - tr(A_P^{-1} dA_P/dtau)],
//   dA_P/dtau = -K_P / tau^2 + U.
// E = -(1/M) d ln Z / d tau at fixed M for Z = sum_P s_P Z_P (s_P = 1 for the
// bosonic |weight| ensemble, sgn P_e sgn P_h for fermions), and the cycle
// fractions follow from the sector weights.  Joint moves alone preserve
// parity(P_e) = parity(P_h), so with no single-species move the chain samples
// the equal-parity sectors only — an exact restricted target that isolates
// the joint move's own detailed balance.
//
// The all-pairs harmonic attraction couples only the e and h centroids: it
// does not bind individual excitons, so these tests check exactness, not the
// efficiency gain for bound pairs (that needs an interacting system).
// Energies and signs compare within 4 sigma + 1e-3 of the driver's error;
// cycle fractions within an absolute 0.03.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <cmath>
#include <functional>
#include <memory>
#include <numbers>
#include <numeric>
#include <string>
#include <vector>

#include "hamiltonian/harmonic_trap.h"
#include "hamiltonian/hamiltonian_component.h"
#include "pimc/move.h"
#include "pimc/pimc.h"

namespace {
using namespace alea;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;

double tolerance(double error) { return 4.0 * error + 1e-3; }

// V = (kappa/2) sum_{a in A, b in B} |r_a - r_b|^2 (test-only potential term).
class HarmonicPair final : public HamiltonianComponent {
public:
    HarmonicPair(ParticleGroup a, ParticleGroup b, double kappa)
        : a_(std::move(a)), b_(std::move(b)), kappa_(kappa) {}
    std::string name() const override { return "harmonic_pair"; }
    void evaluate_potential(const ParticleBatch& configs, std::span<double> v_out,
                            std::size_t w_begin, std::size_t w_end) const override {
        const std::size_t nw = configs.num_walkers(), dim = configs.dim();
        const auto pos = configs.positions();  // [np * dim * nw]
        w_end = std::min(w_end, nw);
        for (std::size_t w = w_begin; w < w_end; ++w) {
            double sum = 0.0;
            for (std::size_t i = a_.first; i < a_.last; ++i)
                for (std::size_t j = b_.first; j < b_.last; ++j)
                    for (std::size_t d = 0; d < dim; ++d) {
                        const double diff = pos[i * dim * nw + d * nw + w] -
                                            pos[j * dim * nw + d * nw + w];
                        sum += diff * diff;
                    }
            v_out[w] += 0.5 * kappa_ * sum;
        }
    }
    std::unique_ptr<HamiltonianComponent> clone() const override {
        return std::make_unique<HarmonicPair>(*this);
    }
private:
    ParticleGroup a_, b_;
    double kappa_;
};

ParticleGroup group(std::string name, std::size_t first, std::size_t last,
                    std::size_t dim, double mass) {
    ParticleGroup g;
    g.name = std::move(name);
    g.first = first;
    g.last = last;
    g.count = last - first;
    g.charge = 0.0;
    g.mass.assign(dim, mass);
    g.inv_mass.assign(dim, 1.0 / mass);
    return g;
}

struct GaussianModel {
    std::size_t ne = 2, nh = 2, M = 8, dim = 2;
    double beta = 1.0, me = 1.0, mh = 2.0, omega2 = 1.0, kappa = 1.0;
    // Exchange classes: members of each class (all electrons share one class
    // unless split into flavours, likewise the holes).
    std::vector<std::vector<std::size_t>> classes;
};

struct Sector {
    std::vector<std::size_t> pi;  // global permutation (class preserving)
    double log_z = 0.0, dlog_z_dtau = 0.0, sign = 1.0;
    bool equal_parity = true;     // parity(electron part) == parity(hole part)
};

double parity_of(const std::vector<std::size_t>& pi, const std::vector<std::size_t>& members) {
    double sign = 1.0;
    std::vector<char> seen(pi.size(), 0);
    for (std::size_t p : members) {
        if (seen[p]) continue;
        std::size_t length = 0;
        for (std::size_t q = p; !seen[q]; q = pi[q]) { seen[q] = 1; ++length; }
        if (length % 2 == 0) sign = -sign;
    }
    return sign;
}

void sector_integral(const GaussianModel& g, Sector& s) {
    const std::size_t N = g.ne + g.nh, size = N * g.M;
    const double tau = g.beta / static_cast<double>(g.M);
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(size, size), U = Eigen::MatrixXd::Zero(size, size);
    const auto idx = [&](std::size_t i, std::size_t k) { return i * g.M + k; };
    const auto mass = [&](std::size_t i) { return i < g.ne ? g.me : g.mh; };
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t k = 0; k < g.M; ++k) {
            const std::size_t a = idx(i, k), b = k + 1 < g.M ? idx(i, k + 1) : idx(s.pi[i], 0);
            K(a, a) += mass(i); K(b, b) += mass(i); K(a, b) -= mass(i); K(b, a) -= mass(i);
            U(a, a) += mass(i) * g.omega2 + g.kappa * static_cast<double>(i < g.ne ? g.nh : g.ne);
        }
    for (std::size_t k = 0; k < g.M; ++k)
        for (std::size_t e = 0; e < g.ne; ++e)
            for (std::size_t h = g.ne; h < N; ++h) {
                U(idx(e, k), idx(h, k)) -= g.kappa;
                U(idx(h, k), idx(e, k)) -= g.kappa;
            }
    const Eigen::MatrixXd A = K / tau + tau * U;
    const Eigen::LLT<Eigen::MatrixXd> chol(A);
    REQUIRE(chol.info() == Eigen::Success);
    double log_det = 0.0;
    for (Eigen::Index i = 0; i < A.rows(); ++i) log_det += 2.0 * std::log(chol.matrixL()(i, i));
    const Eigen::MatrixXd dA = -K / (tau * tau) + U;
    const double trace = chol.solve(dA).trace();
    double log_norm = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        log_norm += static_cast<double>(g.M) * std::log(mass(i) / (2.0 * std::numbers::pi * tau));
    const double d = static_cast<double>(g.dim);
    s.log_z = 0.5 * d * (log_norm + static_cast<double>(size) * std::log(2.0 * std::numbers::pi) - log_det);
    s.dlog_z_dtau = 0.5 * d * (-static_cast<double>(size) / tau - trace);
}

// Every class-preserving permutation (product of permutations of each class).
std::vector<Sector> sectors(const GaussianModel& g) {
    const std::size_t N = g.ne + g.nh;
    std::vector<Sector> out;
    std::vector<std::vector<std::size_t>> orders;
    for (const auto& c : g.classes) orders.push_back(c);
    std::vector<std::size_t> pi(N);
    std::iota(pi.begin(), pi.end(), std::size_t{0});
    // Odometer over the classes' permutations.
    std::function<void(std::size_t)> recurse = [&](std::size_t c) {
        if (c == orders.size()) {
            Sector s;
            s.pi = pi;
            std::vector<std::size_t> electrons, holes;
            for (std::size_t p = 0; p < N; ++p) (p < g.ne ? electrons : holes).push_back(p);
            const double pe = parity_of(pi, electrons), ph = parity_of(pi, holes);
            s.sign = pe * ph;
            s.equal_parity = pe == ph;
            sector_integral(g, s);
            out.push_back(s);
            return;
        }
        auto perm = orders[c];
        std::sort(perm.begin(), perm.end());
        do {
            for (std::size_t m = 0; m < perm.size(); ++m) pi[orders[c][m]] = perm[m];
            recurse(c + 1);
        } while (std::next_permutation(perm.begin(), perm.end()));
        for (std::size_t p : orders[c]) pi[p] = p;
    };
    recurse(0);
    return out;
}

struct Ensemble {
    double energy = 0.0, partition = 0.0;
    // Fraction of class-c particles in cycles of length L ([c][L-1]), |weight|.
    std::vector<std::vector<double>> cycle_fraction;
};

Ensemble ensemble(const GaussianModel& g, const std::vector<Sector>& all,
                  bool equal_parity_only, bool signed_weights) {
    double ref = -1e300;
    for (const auto& s : all) ref = std::max(ref, s.log_z);
    Ensemble e;
    double num = 0.0, den = 0.0, abs_den = 0.0;
    e.cycle_fraction.assign(g.classes.size(), {});
    for (std::size_t c = 0; c < g.classes.size(); ++c)
        e.cycle_fraction[c].assign(g.classes[c].size(), 0.0);
    for (const auto& s : all) {
        if (equal_parity_only && !s.equal_parity) continue;
        const double z = std::exp(s.log_z - ref);
        const double w = signed_weights ? s.sign * z : z;
        num += w * s.dlog_z_dtau;
        den += w;
        abs_den += z;
        for (std::size_t c = 0; c < g.classes.size(); ++c) {
            std::vector<char> seen(s.pi.size(), 0);
            for (std::size_t p : g.classes[c]) {
                if (seen[p]) continue;
                std::size_t length = 0;
                for (std::size_t q = p; !seen[q]; q = s.pi[q]) { seen[q] = 1; ++length; }
                e.cycle_fraction[c][length - 1] += z * static_cast<double>(length) /
                                                   static_cast<double>(g.classes[c].size());
            }
        }
    }
    e.energy = -num / den / static_cast<double>(g.M);
    e.partition = den / abs_den;  // average sign of this ensemble
    for (auto& fractions : e.cycle_fraction)
        for (double& f : fractions) f /= abs_den;
    return e;
}

struct Run {
    std::vector<ParticleGroup> groups;
    Hamiltonian ham;
};

Run gaussian_system(const GaussianModel& g) {
    Run run;
    run.groups = {group("e", 0, g.ne, g.dim, g.me), group("h", g.ne, g.ne + g.nh, g.dim, g.mh)};
    run.ham.add_component(std::make_unique<HarmonicTrap>(run.groups, g.omega2));
    run.ham.add_component(std::make_unique<HarmonicPair>(run.groups[0], run.groups[1], g.kappa));
    return run;
}

PimcConfig gaussian_config(const GaussianModel& g, std::uint64_t seed, bool single_species) {
    PimcConfig cfg;
    cfg.threads = 2;
    cfg.samples = 32;
    cfg.beta = g.beta;
    cfg.time_step = g.beta / static_cast<double>(g.M);
    cfg.equilibration_blocks = 200;
    cfg.accumulation_blocks = 800;
    cfg.steps_per_block = 20;
    cfg.sub_steps = 4;
    cfg.seed = seed;
    PimcConfig::MoveSpec bisection;
    bisection.type = "bisection";
    bisection.levels = 2;
    cfg.moves.push_back(bisection);
    PimcConfig::MoveSpec centroid;
    centroid.type = "centroid";
    centroid.step = 0.5;
    cfg.moves.push_back(centroid);
    PimcConfig::MoveSpec joint;
    joint.type = "joint_permutation";
    joint.levels = 2;
    joint.max_cycle = 3;
    joint.pair = {"e", "h"};
    joint.pair_radius = 1.0;
    cfg.moves.push_back(joint);
    if (single_species) {
        PimcConfig::MoveSpec permutation;
        permutation.type = "permutation";
        permutation.levels = 2;
        permutation.max_cycle = 3;
        cfg.moves.push_back(permutation);
    }
    return cfg;
}

std::vector<ExchangeClass> classes_of(const GaussianModel& g, const std::string& statistics) {
    std::vector<ExchangeClass> out;
    for (std::size_t c = 0; c < g.classes.size(); ++c)
        out.push_back(ExchangeClass{g.classes[c].front() < g.ne ? "e" : "h", c, statistics,
                                    g.classes[c]});
    return out;
}

double scalar(const RunResult& result, const std::string& name) {
    for (const auto& [key, value] : result.scalars)
        if (key == name) return value;
    FAIL("missing run scalar " << name);
    return 0.0;
}

GaussianModel pair_model(std::size_t n) {
    GaussianModel g;
    g.ne = g.nh = n;
    std::vector<std::size_t> e(n), h(n);
    std::iota(e.begin(), e.end(), std::size_t{0});
    std::iota(h.begin(), h.end(), n);
    g.classes = {e, h};
    return g;
}
} // namespace

TEST_CASE("The Gaussian e-h reference matches an independent numpy evaluation",
          "[pimc][joint]") {
    // numpy slogdet/solve evaluation of the same sums (M = 8, beta = 1,
    // kappa = 1, w^2 = 1, 2D, m_e = 1, m_h = 2).
    const GaussianModel g = pair_model(2);
    const auto all = sectors(g);
    REQUIRE(all.size() == 4);
    CHECK(ensemble(g, all, false, true).energy == Approx(11.61136).margin(1e-4));
    CHECK(ensemble(g, all, false, false).energy == Approx(8.74364).margin(1e-4));
    CHECK(ensemble(g, all, false, true).partition == Approx(0.1583).margin(1e-4));
    CHECK(ensemble(g, all, true, false).energy == Approx(9.13550).margin(1e-4));
}

TEST_CASE("Joint moves alone sample the equal-parity sectors exactly",
          "[pimc][joint][physics]") {
    for (std::size_t n : {2u, 3u}) {
        CAPTURE(n);
        const GaussianModel g = pair_model(n);
        const Ensemble exact = ensemble(g, sectors(g), true, false);
        Run run = gaussian_system(g);
        PimcConfig cfg = gaussian_config(g, 600 + n, false);
        cfg.exchange_classes = classes_of(g, "fermions");
        Pimc pimc(cfg, run.groups, run.ham, SimulationCell{}, g.dim, 2 * n);
        pimc.run();
        INFO(pimc.summary());
        INFO("E = " << pimc.energy() << " +/- " << pimc.energy_error() << " exact " << exact.energy);
        // Equal parities: every sampled fermion sign is +1.
        CHECK(pimc.average_sign() == 1.0);
        CHECK(std::abs(pimc.energy() - exact.energy) <= tolerance(pimc.energy_error()));
        REQUIRE(pimc.cycle_fractions().size() == 2);
        for (std::size_t c = 0; c < 2; ++c)
            for (std::size_t k = 0; k < n; ++k) {
                INFO("class " << c << " cycle " << k + 1 << ": " << pimc.cycle_fractions()[c][k]
                     << " exact " << exact.cycle_fraction[c][k]);
                CHECK(std::abs(pimc.cycle_fractions()[c][k] - exact.cycle_fraction[c][k]) < 0.03);
            }
        const RunResult result = pimc.run_result();
        CHECK(scalar(result, "move_2_cycle2_accepts") > 0);
    }
}

TEST_CASE("Joint and single-species moves sample the full fermionic model",
          "[pimc][joint][physics]") {
    const GaussianModel g = pair_model(2);
    const auto all = sectors(g);
    const Ensemble fermions = ensemble(g, all, false, true);
    const Ensemble bosons = ensemble(g, all, false, false);
    Run run = gaussian_system(g);
    PimcConfig cfg = gaussian_config(g, 2025, true);
    cfg.exchange_classes = classes_of(g, "fermions");
    Pimc pimc(cfg, run.groups, run.ham, SimulationCell{}, g.dim, 4);
    pimc.run();
    INFO(pimc.summary());
    INFO("E = " << pimc.energy() << " +/- " << pimc.energy_error() << " exact " << fermions.energy);
    INFO("<s> = " << pimc.average_sign() << " +/- " << pimc.average_sign_error()
         << " exact " << fermions.partition);
    CHECK(std::abs(pimc.average_sign() - fermions.partition) <= tolerance(pimc.average_sign_error()));
    CHECK(std::abs(pimc.energy() - fermions.energy) <= tolerance(pimc.energy_error()));
    for (std::size_t c = 0; c < 2; ++c)
        for (std::size_t k = 0; k < 2; ++k)
            CHECK(std::abs(pimc.cycle_fractions()[c][k] - bosons.cycle_fraction[c][k]) < 0.03);
}

TEST_CASE("Joint and single-species moves sample bosons with two flavours per group",
          "[pimc][joint][physics]") {
    // Electrons 0..3 (flavours {0,1}, {2,3}), holes 4..7 ({4,5}, {6,7}): the
    // joint move pairs every electron class with every hole class.
    GaussianModel g;
    g.ne = g.nh = 4;
    g.beta = 0.75;
    g.classes = {{0, 1}, {2, 3}, {4, 5}, {6, 7}};
    const auto all = sectors(g);
    REQUIRE(all.size() == 16);
    const Ensemble bosons = ensemble(g, all, false, false);
    Run run = gaussian_system(g);
    PimcConfig cfg = gaussian_config(g, 31, true);
    cfg.exchange_classes = classes_of(g, "bosons");
    Pimc pimc(cfg, run.groups, run.ham, SimulationCell{}, g.dim, 8);
    pimc.run();
    INFO(pimc.summary());
    INFO("E = " << pimc.energy() << " +/- " << pimc.energy_error() << " exact " << bosons.energy);
    CHECK(std::abs(pimc.energy() - bosons.energy) <= tolerance(pimc.energy_error()));
    REQUIRE(pimc.cycle_fractions().size() == 4);
    for (std::size_t c = 0; c < 4; ++c)
        for (std::size_t k = 0; k < 2; ++k)
            CHECK(std::abs(pimc.cycle_fractions()[c][k] - bosons.cycle_fraction[c][k]) < 0.03);
    CHECK(scalar(pimc.run_result(), "move_2_cycle2_accepts") > 0);
}

TEST_CASE("Joint moves keep the path, tables and both permutations consistent",
          "[pimc][joint]") {
    const GaussianModel g = pair_model(3);
    Run run = gaussian_system(g);
    // levels 3 on M = 8 stages 2 * 3 * 7 = 42 beads, more than one class's
    // whole cycle (3 * 8): the staging capacity must cover two classes.
    for (const int levels : {2, 3})
    for (bool periodic : {false, true}) {
        CAPTURE(levels, periodic);
        const SimulationCell cell = periodic
            ? SimulationCell(2, std::vector<double>{4.0, 0.0, 0.0, 4.0}) : SimulationCell{};
        PathSet path(8, 6, 2, 0.125, cell);
        path.set_exchange_classes(std::vector<std::size_t>{10, 10, 10, 20, 20, 20},
                                  std::vector<char>{1, 1, 1, 1, 1, 1});
        for (std::size_t p = 0; p < 6; ++p)
            for (std::size_t k = 0; k < 8; ++k) {
                path.pos(p, 0, k) = 0.4 * std::sin(1.7 * p) + 0.05 * std::cos(0.9 * k + p);
                path.pos(p, 1, k) = 0.4 * std::cos(1.1 * p) + 0.05 * std::sin(0.7 * k);
            }
        path.wrap_positions();
        path.rebuild_tables();
        Action action(run.groups, run.ham, 2);
        JointPermutationMove joint(levels, 3, 1.0, {{{0, 1, 2}, {3, 4, 5}}});
        BisectionMove bisection(2);
        numerics::Random rng(8080);
        std::size_t accepted = 0;
        for (int attempt = 0; attempt < 3000; ++attempt) {
            Move& move = attempt % 2 == 0 ? static_cast<Move&>(joint) : bisection;
            if (move.attempt(path, action, rng) && attempt % 2 == 0) {
                ++accepted;
                double se = 1.0, sh = 1.0;
                std::vector<char> seen(6, 0);
                for (std::size_t p = 0; p < 6; ++p) {
                    if (seen[p]) continue;
                    std::size_t length = 0;
                    for (std::size_t q = p; !seen[q]; q = path.next_particle(q)) {
                        seen[q] = 1;
                        ++length;
                    }
                    if (length % 2 == 0) (p < 3 ? se : sh) *= -1.0;
                }
                REQUIRE(se == sh);  // joint moves keep equal parities
                REQUIRE(path.permutation_sign() == se * sh);
                for (std::size_t p = 0; p < 6; ++p)
                    REQUIRE((path.next_particle(p) < 3) == (p < 3));
            }
            const auto tables = path.dist_tables().data();
            const std::vector<double> kept(tables.begin(), tables.end());
            path.rebuild_tables();
            const auto fresh = path.dist_tables().data();
            REQUIRE(std::equal(kept.begin(), kept.end(), fresh.begin()));
            REQUIRE(std::isfinite(action.total_action(path)));
        }
        CHECK(accepted > 0);
    }
}

TEST_CASE("The driver validates joint_permutation moves", "[pimc][joint]") {
    const GaussianModel g = pair_model(2);
    Run run = gaussian_system(g);
    PimcConfig cfg = gaussian_config(g, 5, false);
    cfg.exchange_classes = classes_of(g, "fermions");
    CHECK_NOTHROW(Pimc(cfg, run.groups, run.ham, SimulationCell{}, 2, 4));

    PimcConfig same = cfg;
    same.moves.back().pair = {"e", "e"};
    CHECK_THROWS_WITH(Pimc(same, run.groups, run.ham, SimulationCell{}, 2, 4),
                      ContainsSubstring("two different groups"));
    PimcConfig missing = cfg;
    missing.moves.back().pair = {"e", "x"};
    CHECK_THROWS_WITH(Pimc(missing, run.groups, run.ham, SimulationCell{}, 2, 4),
                      ContainsSubstring("both groups"));
    PimcConfig radius = cfg;
    radius.moves.back().pair_radius = 0.0;
    CHECK_THROWS_WITH(Pimc(radius, run.groups, run.ham, SimulationCell{}, 2, 4),
                      ContainsSubstring("pair_radius"));
    PimcConfig boltzmannon = cfg;
    boltzmannon.statistics = "boltzmannon";
    CHECK_THROWS_WITH(Pimc(boltzmannon, run.groups, run.ham, SimulationCell{}, 2, 4),
                      ContainsSubstring("needs quantum statistics"));
}

// Speed report G4 for the joint move: time per attempt against a bisection
// attempt, N electrons + N holes (free, unit density each, L = sqrt(N)),
// M = 32, tau = 1/32, levels 3, max_cycle 3, pair_radius 0.5.
// Hidden: test_pimc_joint "[pimc-joint-bench]" -s
TEST_CASE("G4: joint permutation attempt cost against bisection as N grows",
          "[.][pimc-joint-bench]") {
    for (std::size_t N : {4u, 8u, 16u, 32u}) {
        const std::size_t M = 32, dim = 2;
        const double L = std::sqrt(static_cast<double>(N));
        const SimulationCell cell(2, std::vector<double>{L, 0.0, 0.0, L});
        const std::vector groups{group("e", 0, N, dim, 1.0), group("h", N, 2 * N, dim, 2.0)};
        Action action(groups, Hamiltonian{}, dim);
        PathSet path(M, 2 * N, dim, 1.0 / 32, cell);
        std::vector<std::size_t> ids(2 * N);
        for (std::size_t p = 0; p < 2 * N; ++p) ids[p] = p < N ? 1 : 2;
        path.set_exchange_classes(ids, std::vector<char>(2 * N, 1));
        numerics::Random rng(N);
        for (std::size_t p = 0; p < 2 * N; ++p) {
            const double x = rng.uniform(0.0, L), y = rng.uniform(0.0, L);
            for (std::size_t k = 0; k < M; ++k) { path.pos(p, 0, k) = x; path.pos(p, 1, k) = y; }
        }
        path.wrap_positions();
        path.rebuild_tables();
        std::vector<std::size_t> e(N), h(N);
        std::iota(e.begin(), e.end(), std::size_t{0});
        std::iota(h.begin(), h.end(), N);
        JointPermutationMove joint(3, 3, 0.5, {{e, h}});
        BisectionMove bisection(3);
        for (int i = 0; i < 20000; ++i) bisection.attempt(path, action, rng);
        const auto time = [&](Move& move, int attempts, std::size_t& accepted) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < attempts; ++i) accepted += move.attempt(path, action, rng);
            return std::chrono::duration<double, std::micro>(
                       std::chrono::steady_clock::now() - start).count() / attempts;
        };
        std::size_t acc_j = 0, acc_b = 0;
        const int attempts = 2000;
        const double t_j = time(joint, attempts, acc_j);
        const double t_b = time(bisection, attempts, acc_b);
        std::cout << "N=" << N << "+" << N << "  joint " << t_j << " us (acc "
                  << static_cast<double>(acc_j) / attempts << ")  bisection " << t_b
                  << " us  ratio " << t_j / t_b << std::endl;
        CHECK(std::isfinite(t_j));
    }
}
