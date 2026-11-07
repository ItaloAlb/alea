// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "estimators/density_matrix.h"
#include "numerics/random.h"
#include "population/simulation_cell.h"
#include "population/walker.h"
#include "population/walker_set.h"
#include "qmc/run_result.h"
#include "wavefunction/drummond_jastrow.h"
#include "wavefunction/orbital_set.h"
#include "wavefunction/pair_orbital.h"
#include "wavefunction/pairing_determinant.h"
#include "wavefunction/slater_determinant.h"
#include "wavefunction/trial_wavefunction.h"

using namespace alea;

// Checks of the density-matrix estimator (López Ríos et al., PRL 120, 177701
// (2018), Eqs. S6–S8) and of the ratios of a pair moved together that it
// takes from the trial wavefunction, against references that do not use the
// code under test: determinants recomputed with Eigen at the shifted
// configurations, Jastrow factors from fresh evaluations, the exact
// translation trace of a plane-wave determinant, and a single bound pair.

namespace {

constexpr double kSide = 6.0;   // square cell; Wigner–Seitz inradius 3
constexpr double kTwoPi = 6.283185307179586;

ParticleGroup group(const char* name, std::size_t first, std::size_t last) {
    ParticleGroup g;
    g.name = name;
    g.charge = 0.0;
    g.first = first;
    g.last = last;
    g.count = last - first;
    return g;
}

SimulationCell square_cell() {
    const std::array<double, 4> v{kSide, 0.0, 0.0, kSide};
    return SimulationCell(2, v);
}

// The 9 closed-shell plane waves of the square cell (|n|² ≤ 2): 1, then cos
// and sin at (1,0), (0,1), (1,1), (1,−1), in units of b = 2π/side.
std::vector<std::array<int, 2>> star_vectors() { return {{1, 0}, {0, 1}, {1, 1}, {1, -1}}; }
std::unique_ptr<PlaneWaveOrbitalSet> nine_plane_waves() {
    const double b = kTwoPi / kSide;
    std::vector<double> k{0.0, 0.0};
    std::vector<char> is_sin{0};
    for (const std::array<int, 2> n : star_vectors())
        for (char s : {char(0), char(1)}) {
            k.push_back(b * n[0]);
            k.push_back(b * n[1]);
            is_sin.push_back(s);
        }
    return std::make_unique<PlaneWaveOrbitalSet>(2, std::move(k), std::move(is_sin));
}

// Walkers with every particle uniform in the cell.
struct Walkers {
    std::vector<Walker> walkers;
    WalkerSet set;
    Walkers(std::size_t nw, std::size_t np, const SimulationCell& cell, std::uint64_t seed)
        : set(nw, np, 2, cell) {
        numerics::Random rng(seed);
        walkers.reserve(nw);
        for (std::size_t w = 0; w < nw; ++w) {
            std::vector<double> pos(np * 2);
            for (double& x : pos) x = rng.uniform(-0.5 * kSide, 0.5 * kSide);
            walkers.emplace_back(np, 2);
            walkers.back().set_positions(pos);
        }
        for (auto& w : walkers) set.add_walker(w);
        set.load_all();
    }
};

// The bilayer of the paper in miniature: e_up [0,3), e_dn [3,6), h_up [6,9),
// h_dn [9,12); blocks (e_up, h_dn) and (e_dn, h_up), φ with 3 stars and a
// quartic polynomial of cutoff 2.8; an e–h and a same-spin Drummond Jastrow.
const std::vector<double> kPairParams{0.3, 0.2, 0.1, 1.0, 0.2, -0.05, 0.01, 2.8};

TrialWavefunction bilayer_psi(const SimulationCell& cell) {
    std::vector<Parameter> params;
    for (std::size_t i = 0; i < kPairParams.size(); ++i)
        params.push_back({"p" + std::to_string(i), kPairParams[i], i == 3, false});
    std::vector<PairingDeterminant::Block> blocks;
    blocks.push_back({group("e_up", 0, 3), group("h_dn", 9, 12), nullptr});
    blocks.push_back({group("e_dn", 3, 6), group("h_up", 6, 9), nullptr});
    TrialWavefunction psi;
    psi.add_component(std::make_unique<PairingDeterminant>(PairOrbital(cell, 3, 4),
                                                           std::move(blocks), std::move(params)));
    const std::vector<Parameter> eh{{"a0", -0.2, false, false}, {"a2", 0.01, false, false},
                                    {"L", 2.9, false, false}};
    psi.add_component(std::make_unique<DrummondJastrow>(
        3, 0.0,
        std::vector<JastrowGroupPair>{{group("e_up", 0, 3), group("h_up", 6, 9)},
                                      {group("e_up", 0, 3), group("h_dn", 9, 12)},
                                      {group("e_dn", 3, 6), group("h_up", 6, 9)},
                                      {group("e_dn", 3, 6), group("h_dn", 9, 12)}},
        eh));
    const std::vector<Parameter> same{{"a0", 0.3, false, false}, {"a2", -0.02, false, false},
                                      {"L", 2.7, false, false}};
    psi.add_component(std::make_unique<DrummondJastrow>(3, -0.5, group("e_up", 0, 3),
                                                        group("e_up", 0, 3), same));
    return psi;
}

// ψ(R′)/ψ(R) of the bilayer trial with its sign: the determinants of both
// blocks recomputed with Eigen from fresh evaluations, the Jastrow factors
// from their fresh log ψ_c.
std::vector<double> bilayer_ratio(const SimulationCell& cell, const WalkerSet& base,
                                  const std::vector<double>& moved_positions) {
    const std::size_t nw = base.num_walkers(), np = base.n_particles();
    auto signed_parts = [&](std::span<const double> positions) {
        WalkerSet s(nw, np, 2, cell);
        std::copy(positions.begin(), positions.end(), s.positions().begin());
        for (std::size_t p = 0; p < np; ++p)
            for (std::size_t w = 0; w < nw; ++w) {
                double r[2] = {s.positions()[(p * 2) * nw + w], s.positions()[(p * 2 + 1) * nw + w]};
                cell.wrap(r);
                s.positions()[(p * 2) * nw + w] = r[0];
                s.positions()[(p * 2 + 1) * nw + w] = r[1];
            }
        s.dist_tables().build_all(s.positions());
        TrialWavefunction psi = bilayer_psi(cell);
        psi.evaluate_batch(s);
        const auto& det = dynamic_cast<const PairingDeterminant&>(*psi.components()[0]);
        std::vector<std::pair<double, double>> out(nw);   // (determinant product, Jastrow log)
        for (std::size_t w = 0; w < nw; ++w) {
            double product = 1.0;
            for (std::size_t b = 0; b < det.n_blocks(); ++b) {
                const auto m = det.matrix(b, w);
                const std::size_t n = 3;
                Eigen::MatrixXd d(n, n);
                for (std::size_t r = 0; r < n; ++r)
                    for (std::size_t c = 0; c < n; ++c) d(r, c) = m[r * n + c];
                product *= d.determinant();
            }
            double jastrow = 0.0;
            for (std::size_t c = 1; c < psi.components().size(); ++c)
                jastrow += psi.components()[c]->log_psi_c()[w];
            out[w] = {product, jastrow};
        }
        return out;
    };
    const auto before = signed_parts(base.positions());
    const auto after = signed_parts(moved_positions);
    std::vector<double> ratio(nw);
    for (std::size_t w = 0; w < nw; ++w)
        ratio[w] = after[w].first / before[w].first * std::exp(after[w].second - before[w].second);
    return ratio;
}

// Shifts of three walkers: short, beyond the inradius, beyond the side.
std::vector<double> three_shifts() {
    const std::array<double, 3> length{0.4, 3.7, 7.3}, angle{0.3, 2.1, 4.4};
    std::vector<double> s(2 * 3);
    for (std::size_t w = 0; w < 3; ++w) {
        s[0 * 3 + w] = length[w] * std::cos(angle[w]);
        s[1 * 3 + w] = length[w] * std::sin(angle[w]);
    }
    return s;
}

// The fraction of an estimator series at a key.
const std::vector<double>& series(const RunResult& out, const std::string& key) {
    for (const auto& [name, values] : out.series)
        if (name == key) return values;
    throw std::runtime_error("no series " + key);
}

}  // namespace

TEST_CASE("Ratios of a shifted pair equal recomputed determinants and Jastrow factors",
          "[density_matrix][shifted_pairs]") {
    const SimulationCell cell = square_cell();
    const std::size_t np = 12, nw = 3;
    Walkers base(nw, np, cell, 17);
    TrialWavefunction psi = bilayer_psi(cell);
    psi.evaluate_batch(base.set);
    REQUIRE(psi.components_without_shifted_pairs().empty());
    const std::vector<double> shift = three_shifts();

    // Paired block (rank 2, either order), different blocks, both electrons.
    struct Channel { std::size_t a0, a1, b0, b1; };
    for (const Channel ch : {Channel{0, 3, 9, 12}, Channel{9, 12, 0, 3}, Channel{0, 3, 6, 9},
                             Channel{0, 3, 3, 6}}) {
        const std::size_t na = ch.a1 - ch.a0, nb = ch.b1 - ch.b0;
        std::vector<double> logs(na * nb * nw), phases(na * nb * nw);
        psi.shifted_pair_ratios_batch(base.set, ch.a0, ch.a1, ch.b0, ch.b1, shift, logs, phases);
        for (std::size_t i = 0; i < na; ++i)
            for (std::size_t j = 0; j < nb; ++j) {
                std::vector<double> moved(base.set.positions().begin(),
                                          base.set.positions().end());
                for (const std::size_t p : {ch.a0 + i, ch.b0 + j})
                    for (std::size_t d = 0; d < 2; ++d)
                        for (std::size_t w = 0; w < nw; ++w)
                            moved[(p * 2 + d) * nw + w] += shift[d * nw + w];
                const std::vector<double> ref = bilayer_ratio(cell, base.set, moved);
                for (std::size_t w = 0; w < nw; ++w) {
                    const std::size_t k = (i * nb + j) * nw + w;
                    const double ratio = std::exp(logs[k]) * std::cos(phases[k]);
                    CHECK(ratio == Catch::Approx(ref[w]).epsilon(1e-9).margin(1e-12));
                }
            }
    }
}

TEST_CASE("A pair whose electron alone would leave every hole out of range still gets its ratio",
          "[density_matrix][shifted_pairs]") {
    // One pair with a compact φ (polynomial only, cutoff 1.2): the electron
    // shifted alone has no hole in range, so ψ of that intermediate is exactly
    // 0, yet the pair moved together keeps ψ.  The ratio must be exactly 1.
    const SimulationCell cell = square_cell();
    const std::size_t np = 2, nw = 3;
    WalkerSet set(nw, np, 2, cell);
    const std::array<double, 3> ex{0.3, -1.0, 2.0}, ey{0.2, 1.4, -2.5}, dx{0.5, -0.3, 0.2},
        dy{-0.3, 0.4, 0.6};
    for (std::size_t w = 0; w < nw; ++w) {
        set.positions()[0 * nw + w] = ex[w];
        set.positions()[1 * nw + w] = ey[w];
        set.positions()[2 * nw + w] = ex[w] + dx[w];
        set.positions()[3 * nw + w] = ey[w] + dy[w];
    }
    set.dist_tables().build_all(set.positions());
    std::vector<PairingDeterminant::Block> blocks;
    blocks.push_back({group("e", 0, 1), group("h", 1, 2), nullptr});
    std::vector<Parameter> params{{"c0", 1.0, true, false}, {"c2", 0.3, false, false},
                                  {"c3", -0.1, false, false}, {"c4", 0.02, false, false},
                                  {"L", 1.2, false, false}};
    TrialWavefunction psi;
    psi.add_component(std::make_unique<PairingDeterminant>(PairOrbital(cell, 0, 4),
                                                           std::move(blocks), params));
    psi.evaluate_batch(set);
    const std::vector<double> shift = three_shifts();
    std::vector<double> logs(nw), phases(nw);
    psi.shifted_pair_ratios_batch(set, 0, 1, 1, 2, shift, logs, phases);
    for (std::size_t w = 0; w < nw; ++w) {
        CHECK(logs[w] == Catch::Approx(0.0).margin(1e-12));
        CHECK(phases[w] == 0.0);
    }
}

TEST_CASE("One-body ratios of a plane-wave determinant sum to the trace of the translation",
          "[density_matrix][one_body]") {
    // Shifting one particle by s maps the closed-shell plane waves into their
    // own span: cos/sin(k·(r+s)) is a rotation by k·s of (cos, sin)(k·r).  So
    // Σ_i ψ(…, r_i + s, …)/ψ(R) = tr T(s) = 1 + Σ_k 2 cos(k·s) for EVERY R.
    const SimulationCell cell = square_cell();
    const std::size_t np = 9, nw = 3;
    Walkers base(nw, np, cell, 23);
    std::vector<SlaterDeterminant::Block> blocks;
    blocks.push_back({0, 9, nine_plane_waves()});
    TrialWavefunction psi;
    psi.add_component(std::make_unique<SlaterDeterminant>(std::move(blocks)));
    psi.evaluate_batch(base.set);

    VirtualMovePlan plan;
    for (std::size_t p = 0; p < np; ++p) plan.singles.push_back(p);
    const std::vector<double> shift = three_shifts();
    std::vector<double> one(np * nw), two, scratch;
    virtual_move_ratios(psi, base.set, cell, plan, shift, one, two, scratch);
    const double b = kTwoPi / kSide;
    for (std::size_t w = 0; w < nw; ++w) {
        double sum = 0.0;
        for (std::size_t q = 0; q < np; ++q) sum += one[q * nw + w];
        double trace = 1.0;
        for (const std::array<int, 2> n : star_vectors())
            trace += 2.0 * std::cos(b * (n[0] * shift[0 * nw + w] + n[1] * shift[1 * nw + w]));
        CHECK(sum == Catch::Approx(trace).margin(1e-10));
    }
}

TEST_CASE("Ratios of a shifted pair in one Slater block equal recomputed determinants",
          "[density_matrix][shifted_pairs]") {
    const SimulationCell cell = square_cell();
    const std::size_t np = 18, nw = 3;
    Walkers base(nw, np, cell, 29);
    auto make = [] {
        std::vector<SlaterDeterminant::Block> blocks;
        blocks.push_back({0, 9, nine_plane_waves()});
        blocks.push_back({9, 18, nine_plane_waves()});
        TrialWavefunction psi;
        psi.add_component(std::make_unique<SlaterDeterminant>(std::move(blocks)));
        return psi;
    };
    TrialWavefunction psi = make();
    psi.evaluate_batch(base.set);
    // det of block [first, first + 9) at `positions` [p * 2 * nw + d * nw + w].
    const auto orbitals = nine_plane_waves();
    auto det = [&](std::span<const double> positions, std::size_t first, std::size_t w) {
        Eigen::MatrixXd d(9, 9);
        std::vector<double> val(9), grad(18), lap(18);
        for (std::size_t c = 0; c < 9; ++c) {
            const double xy[2] = {positions[((first + c) * 2 + 0) * nw + w],
                                  positions[((first + c) * 2 + 1) * nw + w]};
            orbitals->evaluate(std::span<const double>(xy, 2), 2, val, grad, lap, 0, 1);
            for (std::size_t m = 0; m < 9; ++m) d(c, m) = val[m];
        }
        return d.determinant();
    };
    const std::vector<double> shift = three_shifts();
    struct Channel { std::size_t a0, a1, b0, b1; };
    for (const Channel ch : {Channel{0, 3, 3, 9}, Channel{0, 3, 9, 12}}) {
        const std::size_t na = ch.a1 - ch.a0, nb = ch.b1 - ch.b0;
        std::vector<double> logs(na * nb * nw), phases(na * nb * nw);
        psi.shifted_pair_ratios_batch(base.set, ch.a0, ch.a1, ch.b0, ch.b1, shift, logs, phases);
        for (std::size_t i = 0; i < na; ++i)
            for (std::size_t j = 0; j < nb; ++j) {
                std::vector<double> moved(base.set.positions().begin(),
                                          base.set.positions().end());
                for (const std::size_t p : {ch.a0 + i, ch.b0 + j})
                    for (std::size_t d = 0; d < 2; ++d)
                        for (std::size_t w = 0; w < nw; ++w)
                            moved[(p * 2 + d) * nw + w] += shift[d * nw + w];
                for (std::size_t w = 0; w < nw; ++w) {
                    const double ref = det(moved, 0, w) / det(base.set.positions(), 0, w) *
                                       det(moved, 9, w) / det(base.set.positions(), 9, w);
                    const std::size_t k = (i * nb + j) * nw + w;
                    CHECK(std::exp(logs[k]) * std::cos(phases[k]) ==
                          Catch::Approx(ref).epsilon(1e-9).margin(1e-12));
                }
            }
    }
}

TEST_CASE("A single bound pair has rho2 = 1 and a pair condensate fraction of 1 beyond its size",
          "[density_matrix][estimator]") {
    // ψ = φ(r_e − r_h) depends on the separation only, so moving both by r′
    // leaves it: rho2(r) = 1 at every r.  φ vanishes beyond L = 1.2 and the
    // separations are below 0.6, so for r > 1.8 one particle moved alone sees
    // φ = 0: rho1 = 0 (the 1e-300 floor of the determinant's log) and
    // n_c = (1·1/1)(1 − 0) = 1.
    const SimulationCell cell = square_cell();
    const std::size_t np = 2, nw = 5;
    std::vector<ParticleGroup> groups{group("e", 0, 1), group("h", 1, 2)};
    WalkerSet set(nw, np, 2, cell);
    numerics::Random rng(41);
    for (std::size_t w = 0; w < nw; ++w) {
        const double x = rng.uniform(-2.0, 2.0), y = rng.uniform(-2.0, 2.0);
        set.positions()[0 * nw + w] = x;
        set.positions()[1 * nw + w] = y;
        set.positions()[2 * nw + w] = x + rng.uniform(-0.4, 0.4);
        set.positions()[3 * nw + w] = y + rng.uniform(-0.4, 0.4);
    }
    set.dist_tables().build_all(set.positions());
    std::vector<PairingDeterminant::Block> blocks;
    blocks.push_back({groups[0], groups[1], nullptr});
    std::vector<Parameter> params{{"c0", 1.0, true, false}, {"c2", 0.3, false, false},
                                  {"c3", -0.1, false, false}, {"c4", 0.02, false, false},
                                  {"L", 1.2, false, false}};
    TrialWavefunction psi;
    psi.add_component(std::make_unique<PairingDeterminant>(PairOrbital(cell, 0, 4),
                                                           std::move(blocks), params));

    DensityMatrix est(groups, {}, {{"e", "h"}}, 2.9, 29, 1, cell, 2);
    est.bind_wavefunction(psi, 7, 1);
    const std::vector<double> weights(nw, 1.0);
    for (int block = 0; block < 4; ++block) {
        est.start_block();
        for (int step = 0; step < 12; ++step) est.accumulate(set, weights);
        est.stop_block();
    }
    RunResult out;
    est.finalize(out);
    const auto& r = series(out, "density_matrix_r");
    const auto& rho2 = series(out, "density_matrix_rho2_e_h");
    const auto& frac = series(out, "density_matrix_pair_condensate_fraction_e_h");
    const auto& rho1 = series(out, "density_matrix_rho1_e");
    REQUIRE(r.size() == 29);
    for (std::size_t b = 0; b < r.size(); ++b) {
        CHECK(rho2[b] == Catch::Approx(1.0).margin(1e-12));
        if (r[b] > 1.9) {
            CHECK(std::abs(rho1[b]) < 1e-200);
            CHECK(frac[b] == Catch::Approx(1.0).margin(1e-12));
        }
    }
}

TEST_CASE("The density matrix does not depend on how its walkers are split among workers",
          "[density_matrix][estimator]") {
    const SimulationCell cell = square_cell();
    const std::size_t np = 12, nw = 70;   // three chunks of up to 32
    std::vector<ParticleGroup> groups{group("e_up", 0, 3), group("e_dn", 3, 6),
                                      group("h_up", 6, 9), group("h_dn", 9, 12)};
    Walkers base(nw, np, cell, 53);
    TrialWavefunction psi = bilayer_psi(cell);
    std::vector<double> weights(nw);
    for (std::size_t w = 0; w < nw; ++w) weights[w] = 0.5 + 0.01 * static_cast<double>(w);
    auto run = [&](int workers) {
        DensityMatrix est(groups, {"e_dn"}, {{"e_up", "h_dn"}, {"e_up", "h_up"}}, 4.0, 8, 2,
                          cell, 2);
        est.bind_wavefunction(psi, 99, workers);
        for (int block = 0; block < 3; ++block) {
            est.start_block();
            for (int step = 0; step < 4; ++step) est.accumulate(base.set, weights);
            est.stop_block();
        }
        RunResult out;
        est.finalize(out);
        return out;
    };
    const RunResult one = run(1), three = run(3);
    REQUIRE(one.series.size() == three.series.size());
    for (std::size_t k = 0; k < one.series.size(); ++k) {
        REQUIRE(one.series[k].first == three.series[k].first);
        for (std::size_t b = 0; b < one.series[k].second.size(); ++b)
            CHECK(one.series[k].second[b] ==
                  Catch::Approx(three.series[k].second[b]).epsilon(1e-12).margin(1e-14));
    }
}

TEST_CASE("density_matrix refuses impossible requests at construction",
          "[density_matrix][validation]") {
    const SimulationCell cell = square_cell();
    std::vector<ParticleGroup> groups{group("e", 0, 3), group("h", 3, 6)};
    using Catch::Matchers::ContainsSubstring;
    CHECK_THROWS_WITH(DensityMatrix(groups, {"x"}, {}, 2.0, 10, 1, cell, 2),
                      ContainsSubstring("unknown or empty group 'x'"));
    CHECK_THROWS_WITH(DensityMatrix(groups, {}, {{"e", "e"}}, 2.0, 10, 1, cell, 2),
                      ContainsSubstring("two different groups"));
    CHECK_THROWS_WITH(DensityMatrix(groups, {}, {{"e", "h"}, {"h", "e"}}, 2.0, 10, 1, cell, 2),
                      ContainsSubstring("listed twice"));
    CHECK_THROWS_WITH(DensityMatrix(groups, {"e", "e"}, {}, 2.0, 10, 1, cell, 2),
                      ContainsSubstring("listed twice"));
    CHECK_THROWS_WITH(DensityMatrix(groups, {}, {}, 2.0, 10, 1, cell, 2),
                      ContainsSubstring("give groups, pairs or both"));
    CHECK_THROWS(DensityMatrix(groups, {"e"}, {}, 0.0, 10, 1, cell, 2));
    CHECK_THROWS(DensityMatrix(groups, {"e"}, {}, 2.0, 0, 1, cell, 2));
    CHECK_THROWS(DensityMatrix(groups, {"e"}, {}, 2.0, 10, 0, cell, 2));
    DensityMatrix unbound(groups, {"e"}, {}, 2.0, 10, 1, cell, 2);
    WalkerSet set(1, 6, 2, cell);
    const std::vector<double> weights{1.0};
    CHECK_THROWS_WITH(unbound.accumulate(set, weights), ContainsSubstring("bind_wavefunction"));
}
