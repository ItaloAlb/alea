// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The canonical worm with one cut link (Boninsegni, Prokof'ev & Svistunov,
// Phys. Rev. E 74, 036701 (2006)).
//
// Bookkeeping: PathSet::exchange_successors reconnects exactly two links and
// keeps the in-array links the identity; worm moves keep coordinates, tables,
// the open chain and the sign consistent.  Physics (validation gate L4) is in
// the second half: worm-only exchange against the exact canonical recursion,
// and against the Ceperley moves on the same systems.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include "hamiltonian/coulomb.h"
#include "hamiltonian/harmonic_trap.h"
#include "pimc/move.h"
#include "pimc/pimc.h"
#include "numerics/quadrature.h"

namespace {
using namespace alea;
using Catch::Matchers::ContainsSubstring;

ParticleGroup group(std::string name, std::size_t first, std::size_t last,
                    std::size_t dim, double charge = 0.0, double mass = 1.0) {
    ParticleGroup g;
    g.name = std::move(name);
    g.first = first;
    g.last = last;
    g.count = last - first;
    g.charge = charge;
    g.mass.assign(dim, mass);
    g.inv_mass.assign(dim, 1.0 / mass);
    return g;
}

// Parity of pi restricted to fermionic particles, computed independently.
double parity_sign(const PathSet& path, std::size_t np, const std::vector<char>& fermionic) {
    std::vector<char> seen(np, 0);
    double sign = 1.0;
    for (std::size_t p = 0; p < np; ++p) {
        if (seen[p]) continue;
        std::size_t length = 0;
        for (std::size_t q = p; !seen[q]; q = path.next_particle(q)) {
            seen[q] = 1;
            ++length;
        }
        if (fermionic[p] && length % 2 == 0) sign = -sign;
    }
    return sign;
}

using Point = std::pair<double, double>;
Point point(const PathSet& path, PathSet::Bead b) {
    return {path.pos(b.particle, 0, b.slice), path.pos(b.particle, 1, b.slice)};
}
// Successor of every bead, keyed by the bead's (unique) coordinates.
std::map<Point, Point> successor_map(const PathSet& path) {
    std::map<Point, Point> next;
    for (std::size_t p = 0; p < path.n_particles(); ++p)
        for (std::size_t k = 0; k < path.num_slices(); ++k)
            next[point(path, {p, k})] = point(path, path.bead_after(p, k));
    return next;
}

void require_tables_exact(PathSet& path) {
    const auto tables = path.dist_tables().data();
    const std::vector<double> kept(tables.begin(), tables.end());
    path.rebuild_tables();
    const auto fresh = path.dist_tables().data();
    REQUIRE(std::equal(kept.begin(), kept.end(), fresh.begin()));
}

double extra(const Move& move, const std::string& name) {
    for (const auto& [key, value] : move.extra_diagnostics())
        if (key == name) return value;
    FAIL("missing diagnostic " << name);
    return 0.0;
}
} // namespace

TEST_CASE("exchange_successors reconnects exactly two links and keeps tables exact",
          "[pimc][worm]") {
    const std::size_t M = 8, np = 4, dim = 2;
    for (bool periodic : {false, true}) {
        const SimulationCell cell = periodic
            ? SimulationCell(2, std::vector<double>{5.0, 0.0, 0.0, 5.0}) : SimulationCell{};
        // PathSet is move-only: build a fresh copy of the same path each time.
        const auto make_path = [&] {
            PathSet path(M, np, dim, 0.1, cell);
            path.set_exchange_classes(std::vector<std::size_t>{7, 7, 7, 7},
                                      std::vector<char>{1, 1, 1, 1});
            path.set_permutation(std::vector<std::size_t>{1, 0, 2, 3});
            for (std::size_t p = 0; p < np; ++p)
                for (std::size_t k = 0; k < M; ++k) {
                    path.pos(p, 0, k) = 0.37 * p + 0.011 * k + 0.001 * p * k - 1.0;
                    path.pos(p, 1, k) = -0.23 * p + 0.017 * k + 0.4;
                }
            path.wrap_positions();
            path.rebuild_tables();
            return path;
        };
        const PathSet base = make_path();
        for (const std::size_t k : {std::size_t{0}, std::size_t{2}, std::size_t{5}, M - 1})
            for (const auto& [p, s] : {std::pair<std::size_t, std::size_t>{0, 2}, {3, 1}, {1, 0}}) {
                CAPTURE(periodic, k, p, s);
                PathSet path = make_path();
                const auto before = successor_map(path);
                const Point bead_p = point(path, {p, k}), bead_s = point(path, {s, k});
                const Point next_p = point(path, path.bead_after(p, k));
                const Point next_s = point(path, path.bead_after(s, k));
                const std::size_t row = path.exchange_successors(p, s, k);
                // The bead that was at (p, k) is in the returned row.
                CHECK(point(path, {row, k}) == bead_p);
                auto expected = before;
                expected[bead_p] = next_s;
                expected[bead_s] = next_p;
                CHECK(successor_map(path) == expected);
                // Every slice keeps its set of beads.
                for (std::size_t j = 0; j < M; ++j) {
                    std::vector<Point> was, now;
                    for (std::size_t q = 0; q < np; ++q) {
                        was.push_back(point(base, {q, j}));
                        now.push_back(point(path, {q, j}));
                    }
                    std::sort(was.begin(), was.end());
                    std::sort(now.begin(), now.end());
                    CHECK(was == now);
                }
                CHECK(path.permutation_sign() == parity_sign(path, np, {1, 1, 1, 1}));
                require_tables_exact(path);
            }
    }
    PathSet path(4, 3, 2, 0.1);
    path.set_exchange_classes(std::vector<std::size_t>{0, 0, 1}, std::vector<char>{0, 0, 0});
    CHECK_THROWS_WITH(path.exchange_successors(0, 2, 1), ContainsSubstring("one exchange class"));
    CHECK_THROWS_WITH(path.exchange_successors(0, 0, 1), ContainsSubstring("distinct"));
}

TEST_CASE("An open path cuts the link leaving the head", "[pimc][worm]") {
    PathSet path(6, 3, 1, 0.1);
    path.set_exchange_classes(std::vector<std::size_t>{0, 0, 0}, std::vector<char>{0, 0, 0});
    path.set_permutation(std::vector<std::size_t>{2, 0, 1});
    CHECK_FALSE(path.is_open());
    path.open_at({1, 5});
    CHECK(path.is_open());
    CHECK(path.tail().particle == 0);  // pi(1) = 0 at slice 0
    CHECK(path.tail().slice == 0);
    CHECK_THROWS_WITH(path.open_at({0, 2}), ContainsSubstring("already open"));
    path.set_head({0, 2});
    CHECK(path.tail().particle == 0);
    CHECK(path.tail().slice == 3);
    path.close();
    CHECK_FALSE(path.is_open());
    CHECK_THROWS_WITH(path.open_at({3, 0}), ContainsSubstring("outside"));
}

TEST_CASE("Worm moves keep the path, tables, open chain and sign consistent",
          "[pimc][worm]") {
    // Interacting e-h bilayer: a worm on the 3 electrons, bisection while closed.
    const std::size_t M = 16, dim = 2, np = 5;
    const std::vector groups{group("e", 0, 3, dim, -1.0, 1.0), group("h", 3, 5, dim, 1.0, 2.0)};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(groups, 0.5));
    ham.add_component(std::make_unique<CoulombTerm>(groups[0], groups[0], 2.0));
    ham.add_component(std::make_unique<CoulombTerm>(groups[1], groups[1], 2.0));
    ham.add_component(std::make_unique<CoulombTerm>(groups[0], groups[1], 2.0, 1.0));
    const std::vector<char> fermionic{1, 1, 1, 1, 1};
    for (bool periodic : {false, true}) {
        CAPTURE(periodic);
        const SimulationCell cell = periodic
            ? SimulationCell(2, std::vector<double>{4.0, 0.0, 0.0, 4.0}) : SimulationCell{};
        PathSet path(M, np, dim, 0.0625, cell);
        path.set_exchange_classes(std::vector<std::size_t>{10, 10, 10, 20, 20}, fermionic);
        for (std::size_t p = 0; p < np; ++p)
            for (std::size_t k = 0; k < M; ++k) {
                path.pos(p, 0, k) = 0.25 * std::sin(1.7 * p) + 0.03 * std::cos(0.9 * k + p);
                path.pos(p, 1, k) = 0.25 * std::cos(1.1 * p) + 0.03 * std::sin(0.7 * k);
            }
        path.wrap_positions();
        path.rebuild_tables();
        Action action(groups, ham, dim);
        // C near p_sel rho0 / (kClose N_b) balances open and closed paths.
        WormMove worm({{0, 1, 2}}, 4, 3, 0.05);
        worm.set_selection_probability(0.5);
        BisectionMove bisection(2);
        numerics::Random rng(2718);
        std::size_t open_measurements = 0;
        std::vector<double> displacement(4 * dim), weight(4);
        for (int attempt = 0; attempt < 6000; ++attempt) {
            if (!path.is_open() && attempt % 2 == 1)
                bisection.attempt(path, action, rng);
            else
                worm.attempt(path, action, rng);
            REQUIRE(path.permutation_sign() == parity_sign(path, np, fermionic));
            for (std::size_t p = 0; p < np; ++p)
                REQUIRE(path.exchange_class(path.next_particle(p)) == path.exchange_class(p));
            require_tables_exact(path);
            if (path.is_open()) {
                REQUIRE(worm.owns(path));
                REQUIRE(path.head().particle < 3);
                // The open chain runs from the tail to the head without
                // revisiting a bead.
                PathSet::Bead b = path.tail();
                std::size_t steps = 0;
                while (!(b.particle == path.head().particle && b.slice == path.head().slice)) {
                    b = path.bead_after(b.particle, b.slice);
                    REQUIRE(++steps < np * M);
                }
                REQUIRE((steps + 1) % M == 0);
                worm.sample_open_ends(path, action, rng, displacement, weight);
                for (const double w : weight) REQUIRE(std::isfinite(w));
                ++open_measurements;
            } else {
                REQUIRE(std::isfinite(action.total_action(path)));
            }
        }
        CHECK(open_measurements > 1000);
        for (const char* kind : {"open", "close", "swap", "head", "tail"}) {
            CAPTURE(kind);
            CHECK(extra(worm, std::string(kind) + "_accepts") > 0);
        }
        CHECK(extra(worm, "relabel_slices") > 0);
    }
}

// ── Validation gate L4: worm-only exchange against exact references ─────────
//
// Same systems and references as gate L1 (test_pimc_permutation.cpp, exact
// canonical recursion at finite M), with the Ceperley move replaced by the
// worm: diagonal observables are measured on closed paths only.  Energies and
// signs compare within 4 sigma + 1e-3 of the driver's error; cycle fractions
// within an absolute 0.03.

#include "pimc_ideal_reference.h"

namespace {
using namespace pimc_reference;

double tolerance(double error) { return 4.0 * error + 1e-3; }

PimcConfig worm_config(double beta, double tau, std::uint64_t seed, int levels,
                       int max_length, int swap_length) {
    PimcConfig cfg;
    cfg.threads = 2;
    cfg.samples = 32;
    cfg.beta = beta;
    cfg.time_step = tau;
    cfg.equilibration_blocks = 200;
    cfg.accumulation_blocks = 800;
    cfg.steps_per_block = 20;
    cfg.sub_steps = 4;
    cfg.seed = seed;
    PimcConfig::MoveSpec bisection, centroid, worm;
    bisection.type = "bisection";
    bisection.levels = levels;
    centroid.type = "centroid";
    centroid.step = 0.5;
    worm.type = "worm";
    worm.group = "g";
    worm.max_length = max_length;
    worm.swap_length = swap_length;
    cfg.moves = {bisection, centroid, worm};
    return cfg;
}

double scalar(const RunResult& result, const std::string& name) {
    for (const auto& [key, value] : result.scalars)
        if (key == name) return value;
    FAIL("missing run scalar " << name);
    return 0.0;
}
} // namespace

TEST_CASE("Worm-only exchange: ideal bosons in a 1D trap follow the canonical recursion",
          "[pimc][worm][physics]") {
    const std::size_t N = 3, M = 16, dim = 1;
    const double beta = 2.0, tau = beta / M;
    const auto ring = [&](std::size_t K) { return harmonic_ring(K, tau, 1.0, dim); };
    const Canonical bosons = recursion(N, M, +1, ring);

    const std::vector groups{group("g", 0, N, dim)};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(groups, 1.0));
    PimcConfig cfg = worm_config(beta, tau, 20261006, 3, 6, 4);
    cfg.exchange_classes = {ExchangeClass{"g", 0, "bosons", {0, 1, 2}}};
    Pimc pimc(cfg, groups, ham, SimulationCell{}, dim, N);
    pimc.run();
    INFO(pimc.summary());
    INFO("E = " << pimc.energy() << " +/- " << pimc.energy_error() << " exact " << bosons.energy);
    CHECK(pimc.worm_run());
    CHECK_FALSE(pimc.signed_run());
    CHECK(pimc.average_sign() == 1.0);
    CHECK(pimc.closed_fraction() > 0.2);
    CHECK(pimc.closed_fraction() < 0.8);
    CHECK(std::abs(pimc.energy() - bosons.energy) <= tolerance(pimc.energy_error()));
    CHECK(pimc.energy_error() < 0.02 * bosons.energy);
    for (std::size_t k = 0; k < N; ++k) {
        INFO("cycle length " << k + 1 << ": " << pimc.cycle_fractions()[0][k]
             << " exact " << bosons.cycle_fraction[k]);
        CHECK(std::abs(pimc.cycle_fractions()[0][k] - bosons.cycle_fraction[k]) < 0.03);
    }
    const RunResult result = pimc.run_result();
    CHECK(scalar(result, "move_2_swap_accepts") > 0);
    CHECK(scalar(result, "move_2_close_accepts") > 0);
    CHECK(scalar(result, "move_2_constant_mean") > 0);
}

TEST_CASE("Worm-only exchange: ideal fermions in a 2D trap give the exact signed energy and sign",
          "[pimc][worm][physics]") {
    const std::size_t N = 3, M = 16, dim = 2;
    const double beta = 1.0, tau = beta / M;
    const auto ring = [&](std::size_t K) { return harmonic_ring(K, tau, 1.0, dim); };
    const Canonical fermions = recursion(N, M, -1, ring);
    const Canonical bosons = recursion(N, M, +1, ring);
    const double sign_exact = fermions.partition / bosons.partition;

    const std::vector groups{group("g", 0, N, dim)};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(groups, 1.0));
    PimcConfig cfg = worm_config(beta, tau, 4711, 3, 6, 4);
    cfg.exchange_classes = {ExchangeClass{"g", 0, "fermions", {0, 1, 2}}};
    Pimc pimc(cfg, groups, ham, SimulationCell{}, dim, N);
    pimc.run();
    INFO(pimc.summary());
    INFO("E = " << pimc.energy() << " +/- " << pimc.energy_error() << " exact " << fermions.energy);
    INFO("<s> = " << pimc.average_sign() << " +/- " << pimc.average_sign_error()
         << " exact " << sign_exact);
    CHECK(pimc.signed_run());
    CHECK_FALSE(pimc.sign_unreliable());
    CHECK(std::abs(pimc.average_sign() - sign_exact) <= tolerance(pimc.average_sign_error()));
    CHECK(std::abs(pimc.energy() - fermions.energy) <= tolerance(pimc.energy_error()));
    CHECK(std::abs(bosons.energy - fermions.energy) > 10.0 * pimc.energy_error());
    for (std::size_t k = 0; k < N; ++k)
        CHECK(std::abs(pimc.cycle_fractions()[0][k] - bosons.cycle_fraction[k]) < 0.03);
}

TEST_CASE("Worm-only exchange: ideal bosons on a periodic ring give the exact energy and windings",
          "[pimc][worm][physics]") {
    const std::size_t N = 3, M = 32, dim = 1;
    const double L = 3.0, beta = 1.0, tau = beta / M;
    const auto ring = [&](std::size_t K) { return periodic_ring(K, tau, L, dim); };
    const Canonical bosons = recursion(N, M, +1, ring);
    const auto ring_winding = [&](double b) {
        double z = 0.0, w2 = 0.0;
        for (int w = -30; w <= 30; ++w) {
            const double weight = std::exp(-L * L * w * w / (2.0 * b));
            z += weight;
            w2 += static_cast<double>(w) * w * weight;
        }
        return w2 / z;
    };
    double w2_exact = 0.0;
    for (std::size_t k = 1; k <= N; ++k)
        w2_exact += static_cast<double>(N) * bosons.cycle_fraction[k - 1] /
                    static_cast<double>(k) * ring_winding(static_cast<double>(k) * beta);
    w2_exact /= static_cast<double>(N * dim);

    const std::vector groups{group("g", 0, N, dim)};
    PimcConfig cfg = worm_config(beta, tau, 314, 4, 8, 6);
    cfg.exchange_classes = {ExchangeClass{"g", 0, "bosons", {0, 1, 2}}};
    Pimc pimc(cfg, groups, Hamiltonian{}, SimulationCell(1, std::vector<double>{L}), dim, N);
    pimc.run();
    INFO(pimc.summary());
    INFO("E = " << pimc.energy() << " +/- " << pimc.energy_error() << " exact " << bosons.energy);
    INFO("W2 = " << pimc.winding2() << " +/- " << pimc.winding2_error() << " exact " << w2_exact);
    CHECK(std::abs(pimc.energy() - bosons.energy) <= tolerance(pimc.energy_error()));
    CHECK(std::abs(pimc.winding2() - w2_exact) <= tolerance(pimc.winding2_error()));
    for (std::size_t k = 0; k < N; ++k)
        CHECK(std::abs(pimc.cycle_fractions()[0][k] - bosons.cycle_fraction[k]) < 0.03);
}

TEST_CASE("Worm and Ceperley moves agree on interacting fermions",
          "[pimc][worm][physics]") {
    // Three same-flavour fermions with Coulomb repulsion in a 2D trap: no
    // closed form, so the worm-only, Ceperley-only and mixed move sets must
    // agree with each other.
    const std::size_t N = 3, M = 16, dim = 2;
    const double beta = 1.0, tau = beta / M;
    const std::vector groups{group("g", 0, N, dim, 1.0)};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(groups, 1.0));
    ham.add_component(std::make_unique<CoulombTerm>(groups[0], groups[0], 2.0));
    struct Result { double e, e_err, s, s_err; };
    const auto run = [&](bool worm, bool permutation, std::uint64_t seed) {
        PimcConfig cfg = worm_config(beta, tau, seed, 3, 6, 4);
        if (!worm) cfg.moves.pop_back();
        if (permutation) {
            PimcConfig::MoveSpec move;
            move.type = "permutation";
            move.levels = 3;
            move.max_cycle = 3;
            cfg.moves.push_back(move);
        }
        cfg.exchange_classes = {ExchangeClass{"g", 0, "fermions", {0, 1, 2}}};
        Pimc pimc(cfg, groups, ham, SimulationCell{}, dim, N);
        pimc.run();
        INFO(pimc.summary());
        CHECK_FALSE(pimc.sign_unreliable());
        return Result{pimc.energy(), pimc.energy_error(), pimc.average_sign(),
                      pimc.average_sign_error()};
    };
    const Result ceperley = run(false, true, 11), worm = run(true, false, 12),
                 mixed = run(true, true, 13);
    for (const Result& other : {worm, mixed}) {
        INFO("E = " << other.e << " +/- " << other.e_err << " vs Ceperley "
             << ceperley.e << " +/- " << ceperley.e_err);
        INFO("<s> = " << other.s << " +/- " << other.s_err << " vs Ceperley "
             << ceperley.s << " +/- " << ceperley.s_err);
        CHECK(std::abs(other.e - ceperley.e) <=
              4.0 * std::hypot(other.e_err, ceperley.e_err) + 1e-3);
        CHECK(std::abs(other.s - ceperley.s) <=
              4.0 * std::hypot(other.s_err, ceperley.s_err) + 1e-3);
    }
}

TEST_CASE("The driver validates worm moves", "[pimc][worm]") {
    const std::size_t N = 3, dim = 2;
    const std::vector groups{group("g", 0, N, dim)};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(groups, 1.0));
    const auto make = [&] {
        PimcConfig cfg = worm_config(1.0, 1.0 / 16, 1, 3, 6, 4);
        cfg.exchange_classes = {ExchangeClass{"g", 0, "fermions", {0, 1, 2}}};
        return cfg;
    };
    CHECK_NOTHROW(Pimc(make(), groups, ham, SimulationCell{}, dim, N));
    PimcConfig boltzmannon = make();
    boltzmannon.statistics = "boltzmannon";
    CHECK_THROWS_WITH(Pimc(boltzmannon, groups, ham, SimulationCell{}, dim, N),
                      ContainsSubstring("needs quantum statistics"));
    PimcConfig unknown = make();
    unknown.moves.back().group = "h";
    CHECK_THROWS_WITH(Pimc(unknown, groups, ham, SimulationCell{}, dim, N),
                      ContainsSubstring("group 'h'"));
    PimcConfig long_regrowth = make();
    long_regrowth.moves.back().max_length = 15;
    CHECK_THROWS_WITH(Pimc(long_regrowth, groups, ham, SimulationCell{}, dim, N),
                      ContainsSubstring("max_length"));
    PimcConfig short_swap = make();
    short_swap.moves.back().swap_length = 1;
    CHECK_THROWS_WITH(Pimc(short_swap, groups, ham, SimulationCell{}, dim, N),
                      ContainsSubstring("swap_length"));
    PimcConfig negative_c = make();
    negative_c.moves.back().worm_constant = -1.0;
    CHECK_THROWS_WITH(Pimc(negative_c, groups, ham, SimulationCell{}, dim, N),
                      ContainsSubstring("worm C"));
    PimcConfig twice = make();
    twice.moves.push_back(twice.moves.back());
    CHECK_THROWS_WITH(Pimc(twice, groups, ham, SimulationCell{}, dim, N),
                      ContainsSubstring("two worm moves"));
}

// ── One-body density matrix and n(k) from the worm's open paths ─────────────
//
// Ideal canonical gases (eta = +1 bosons, -1 fermions; Z_n from the recursion):
//   n(r)  = (1/N) sum_j eta^{j+1} (Z_{N-j}/Z_N) Int dR rho1(R + r, R; j beta),
//   n~(k) = Fourier transform of n(r),  <n_k> = (N/V) n~(k) under PBC.
// Periodic ring: Int dR rho1(R + r, R; b) = sum_k cos(k r) e^{-b k^2/2}, so
//   <n_k> = sum_j eta^{j+1} e^{-j beta k^2/2} Z_{N-j}/Z_N, exact at any M.
// Harmonic trap (Mehler kernel, per dimension, m = omega = 1):
//   Int dR rho(R + r, R; b) = z(b) exp(-(r^2/4) coth(b/2)),
// the continuum value: the primitive action's finite-M bias at tau = 1/16 is
// O(tau^2) ~ 1e-3, below these statistical errors.

#include "estimators/momentum_distribution.h"

namespace {
// Z_0..Z_N of the canonical recursion from per-cycle partition functions
// z[j] (j = 1..N, z[0] unused).
std::vector<double> canonical_partitions(std::size_t N, int eta, const std::vector<double>& z) {
    std::vector<double> Z(N + 1, 0.0);
    Z[0] = 1.0;
    for (std::size_t n = 1; n <= N; ++n) {
        for (std::size_t j = 1; j <= n; ++j)
            Z[n] += ((eta < 0 && j % 2 == 0) ? -1.0 : 1.0) * z[j] * Z[n - j];
        Z[n] /= static_cast<double>(n);
    }
    return Z;
}

std::vector<double> series(const RunResult& result, const std::string& name) {
    for (const auto& [key, values] : result.series)
        if (key == name) return values;
    FAIL("missing run series " << name);
    return {};
}
} // namespace

TEST_CASE("Worm n(k) and n(r) of ideal gases on a periodic ring are exact",
          "[pimc][worm][physics]") {
    // Bosons: L = 3, beta = 1 (degenerate, n_0 = 2.75).  Fermions: L = 6,
    // beta = 0.5, where <s> = 0.27 resolves a smeared Fermi edge
    // (n_k = 0.70, 0.63, 0.38, 0.12 on the first four shells).
    const std::size_t N = 3, dim = 1;
    struct Case { int eta; double L, beta; std::size_t M; std::uint64_t seed; };
    for (const Case c : {Case{+1, 3.0, 1.0, 32, 515}, Case{-1, 6.0, 0.5, 16, 616}}) {
        CAPTURE(c.eta);
        const double tau = c.beta / static_cast<double>(c.M);
        std::vector<double> z(N + 1, 0.0);
        for (std::size_t j = 1; j <= N; ++j)
            z[j] = std::exp(periodic_ring(j * c.M, tau, c.L, dim).log_z);
        const std::vector<double> Z = canonical_partitions(N, c.eta, z);
        const auto occupation = [&](double k) {
            double n = 0.0;
            for (std::size_t j = 1; j <= N; ++j)
                n += ((c.eta < 0 && j % 2 == 0) ? -1.0 : 1.0) *
                     std::exp(-static_cast<double>(j) * c.beta * k * k / 2.0) * Z[N - j] / Z[N];
            return n;
        };
        const std::vector groups{group("g", 0, N, dim)};
        const SimulationCell cell(1, std::vector<double>{c.L});
        PimcConfig cfg = worm_config(c.beta, tau, c.seed, 3, 6, 4);
        cfg.exchange_classes = {ExchangeClass{"g", 0, c.eta > 0 ? "bosons" : "fermions", {0, 1, 2}}};
        Pimc pimc(cfg, groups, Hamiltonian{}, cell, dim, N);
        const double rmax = c.L / 2.0, kmax = 3.0 * 2.0 * std::numbers::pi / c.L + 1e-9;
        std::vector<std::unique_ptr<Estimator>> ests;
        ests.push_back(std::make_unique<MomentumDistribution>(groups, "g", rmax, 6, kmax, 0, 4,
                                                              cell, dim));
        pimc.set_estimators(std::move(ests));
        pimc.run();
        INFO(pimc.summary());
        CHECK_FALSE(pimc.sign_unreliable());
        const RunResult result = pimc.run_result();
        const auto k = series(result, "nk_g_k"), nk = series(result, "nk_g"),
                   nk_err = series(result, "nk_g_error");
        REQUIRE(k.size() == 4);  // shells |k| = 0, 2 pi/L, 4 pi/L, 6 pi/L
        for (std::size_t s = 0; s < k.size(); ++s) {
            const double exact = occupation(k[s]);
            INFO("|k| = " << k[s] << ": <n_k> = " << nk[s] << " +/- " << nk_err[s]
                 << " exact " << exact);
            CHECK(std::abs(nk[s] - exact) <= tolerance(nk_err[s]));
            CHECK(nk_err[s] < 0.05);  // resolved, not passing on noise
            if (c.eta < 0) CHECK(nk[s] <= 1.0 + 4.0 * nk_err[s]);  // Pauli
        }
        // Finite-cutoff sums have an independent canonical value; tails
        // beyond kmax are not silently treated as zero in the reference.
        double exact_number = 0.0, exact_kinetic = 0.0;
        for (int q = -3; q <= 3; ++q) {
            const double wave = 2.0 * std::numbers::pi * static_cast<double>(q) / c.L;
            exact_number += occupation(wave);
            exact_kinetic += 0.5 * wave * wave * occupation(wave);
        }
        CHECK(std::abs(series(result, "nk_g_number")[0] - exact_number) <=
              tolerance(series(result, "nk_g_number_error")[0]));
        CHECK(std::abs(series(result, "nk_g_kinetic")[0] - exact_kinetic) <=
              tolerance(series(result, "nk_g_kinetic_error")[0]));
        const auto r = series(result, "obdm_g_r"), n_r = series(result, "obdm_g"),
                   n_r_err = series(result, "obdm_g_error");
        const double delta = rmax / 6.0;
        for (std::size_t b = 0; b < r.size(); ++b) {
            // Bin average of n(r) = (1/N) sum_k <n_k> cos(k r) over |r| in the bin.
            const double r0 = static_cast<double>(b) * delta, r1 = r0 + delta;
            double exact = 0.0;
            for (int n = -60; n <= 60; ++n) {
                const double kn = 2.0 * std::numbers::pi * n / c.L;
                const double average = n == 0 ? 1.0
                    : (std::sin(kn * r1) - std::sin(kn * r0)) / (kn * delta);
                exact += occupation(kn) * average / static_cast<double>(N);
            }
            INFO("r = " << r[b] << ": n(r) = " << n_r[b] << " +/- " << n_r_err[b]
                 << " exact " << exact);
            CHECK(std::abs(n_r[b] - exact) <= tolerance(n_r_err[b]));
        }
    }
}

TEST_CASE("Momentum cutoff sums retain shell covariance and anisotropic masses through cloning",
          "[pimc][worm][momentum][contract]") {
    const double pi = std::numbers::pi;
    const SimulationCell cell(2, std::vector<double>{2*pi, 0.0, 0.0, 2*pi});
    auto g = group("g", 0, 2, 2);
    g.mass = {2.0, 4.0};
    g.inv_mass = {0.5, 0.25};
    const std::vector groups{g};
    MomentumDistribution estimator(groups, "g", 1.0, 2, 1.1, 0, 1, cell, 2);
    estimator.configure_signed_weights();
    estimator.configure_open_paths(2);
    estimator.configure_reblocking(64);
    auto clone = estimator.clone();
    PathSet path(1, 2, 2, 1.0, cell);
    const std::array<double, 1> closed{1.0};
    const std::array<double, 2> displacement{pi/2, pi};
    for (std::size_t block = 0; block < 64; ++block) {
        // Five vectors: 0, +/-x, +/-y. Their cosines are 1,0,0,-1,-1.
        // Number kernel=-1; kinetic kernel=-2/(2*m_y)=-1/4.
        const double c = block % 2 == 0 ? 1.0 : 2.0;
        for (auto* target : {static_cast<Estimator*>(&estimator), clone.get()}) {
            target->start_block();
            target->accumulate(path, closed);
            const std::array<double, 1> weight{target == &estimator ? c : 2*c};
            target->accumulate_open(displacement, weight);
            target->stop_block();
        }
    }
    estimator.collect(*clone);
    RunResult result;
    estimator.finalize(result);
    const double factor = 2.0 / cell.volume();
    CHECK(series(result, "nk_g_number")[0] == Catch::Approx(-2.25*factor));
    CHECK(series(result, "nk_g_kinetic")[0] == Catch::Approx(-0.5625*factor));
    CHECK(series(result, "nk_g_degeneracy") == std::vector<double>{1.0,4.0});
    const double zero_error = series(result, "nk_g_error")[0];
    REQUIRE(zero_error > 0);
    CHECK(series(result, "nk_g_number_error")[0] == Catch::Approx(zero_error));
    CHECK(series(result, "nk_g_kinetic_error")[0] == Catch::Approx(0.25*zero_error));
    const auto numerator = series(result,"nk_g_number_numerator_blocks");
    const auto denominator = series(result,"nk_g_closed_denominator_blocks");
    REQUIRE(numerator.size() == 64);
    REQUIRE(denominator.size() == 64);
    CHECK(numerator[0] == Catch::Approx(-0.75*factor));
    CHECK(numerator[1] == Catch::Approx(-1.5*factor));
    CHECK(std::all_of(denominator.begin(),denominator.end(),[](double d){return d==0.5;}));
}

TEST_CASE("Momentum kinetic sums reject missing or nonpositive periodic masses",
          "[pimc][worm][momentum][contract]") {
    const SimulationCell cell(2, std::vector<double>{4.0,0.0,0.0,4.0});
    auto g = group("g",0,2,2);
    for (const auto& invalid : {std::vector<double>{}, std::vector<double>{1.0},
                              std::vector<double>{1.0,0.0}}) {
        g.inv_mass = invalid;
        const std::vector groups{g};
        CHECK_THROWS_WITH(MomentumDistribution(groups,"g",1.0,2,1.0,0,1,cell,2),
                          ContainsSubstring("positive finite inverse masses"));
    }
}

TEST_CASE("Open-cell momentum kinetic quadrature agrees with independent angular integration",
          "[pimc][worm][momentum][contract]") {
    auto g=group("g",0,2,2);
    g.mass={2.0,4.0};
    g.inv_mass={0.5,0.25};
    const std::vector groups{g};
    for (const std::array<double,2> r : {std::array<double,2>{0.0,0.0},
                                       std::array<double,2>{0.75,1.25}}) {
        MomentumDistribution estimator(groups,"g",2.0,4,1.0,2,1,SimulationCell{},2);
        estimator.configure_signed_weights();
        estimator.configure_open_paths(2);
        estimator.configure_reblocking(64);
        PathSet path(1,2,2,1.0);
        const std::array<double,1> weight{1.0};
        for (std::size_t b=0; b<64; ++b) {
            estimator.start_block();
            estimator.accumulate(path,weight);
            estimator.accumulate_open(r,weight);
            estimator.stop_block();
        }
        RunResult result;
        estimator.finalize(result);
        double number=0.0,kinetic=0.0;
        constexpr std::size_t angles=4096;
        for (std::size_t i=0; i<angles; ++i) {
            const double theta=2*std::numbers::pi*static_cast<double>(i)/static_cast<double>(angles);
            const double x=std::cos(theta),y=std::sin(theta);
            const double phase=std::cos(x*r[0]+y*r[1]);
            number+=phase;
            kinetic+=0.5*(x*x/2+y*y/4)*phase;
        }
        // Two-point radial trapezoid at k=0,1: only k=1 has nonzero
        // 2D measure, 0.5/(2pi), multiplied by N=2.
        const double factor=1/(2*std::numbers::pi*static_cast<double>(angles));
        CHECK(series(result,"nk_g_number")[0] == Catch::Approx(number*factor).margin(1e-12));
        CHECK(series(result,"nk_g_kinetic")[0] == Catch::Approx(kinetic*factor).margin(1e-12));
    }
}

TEST_CASE("Three-dimensional open momentum moments agree with sphere integration",
          "[pimc][worm][momentum][contract]") {
    auto g=group("g",0,2,3);
    g.mass={2.0,4.0,8.0};
    g.inv_mass={0.5,0.25,0.125};
    const std::vector groups{g};
    MomentumDistribution estimator(groups,"g",2.0,4,1.0,2,1,SimulationCell{},3);
    estimator.configure_signed_weights();
    estimator.configure_open_paths(2);
    estimator.configure_reblocking(64);
    PathSet path(1,2,3,1.0);
    const std::array<double,1> weight{1.0};
    const std::array<double,3> r{0.3,0.4,0.5};
    for (std::size_t b=0; b<64; ++b) {
        estimator.start_block();
        estimator.accumulate(path,weight);
        estimator.accumulate_open(r,weight);
        estimator.stop_block();
    }
    const auto angular=[&](double z,bool kinetic) {
        double sum=0;
        for (int i=0; i<512; ++i) {
            const double phi=2*std::numbers::pi*static_cast<double>(i)/512;
            const double x=std::sqrt(1-z*z)*std::cos(phi),y=std::sqrt(1-z*z)*std::sin(phi);
            const double energy=kinetic ? 0.5*(x*x/2+y*y/4+z*z/8) : 1;
            sum+=energy*std::cos(x*r[0]+y*r[1]+z*r[2]);
        }
        return sum/512;
    };
    const double number=0.5*numerics::gauss20([&](double z){return angular(z,false);},-1,1);
    const double kinetic=0.5*numerics::gauss20([&](double z){return angular(z,true);},-1,1);
    RunResult result;
    estimator.finalize(result);
    const double factor=1/(2*std::numbers::pi*std::numbers::pi);
    CHECK(series(result,"nk_g_number")[0] == Catch::Approx(factor*number).margin(1e-12));
    CHECK(series(result,"nk_g_kinetic")[0] == Catch::Approx(factor*kinetic).margin(1e-12));
}

TEST_CASE("Anisotropic 3D momentum kinetic moments remain accurate near zero displacement",
          "[pimc][worm][momentum][contract]") {
    auto g=group("g",0,2,3);
    g.mass={2.0,4.0,8.0};
    g.inv_mass={0.5,0.25,0.125};
    const std::vector groups{g};
    for (double radius : {0.0,1e-5,1e-4,1e-3,1e-2}) {
        MomentumDistribution estimator(groups,"g",1.0,4,1.0,2,1,SimulationCell{},3);
        estimator.configure_signed_weights();
        estimator.configure_open_paths(2);
        estimator.configure_reblocking(64);
        PathSet path(1,2,3,1.0);
        const std::array<double,1> weight{1.0};
        const std::array<double,3> displacement{0.0,0.0,radius};
        for (std::size_t b=0; b<64; ++b) {
            estimator.start_block();
            estimator.accumulate(path,weight);
            estimator.accumulate_open(displacement,weight);
            estimator.stop_block();
        }
        // Integrate the polar direction directly. The azimuth average of
        // k_x²/m_x+k_y²/m_y is (1-z²)*(1/m_x+1/m_y)/2, k=1.
        // This integral avoids the cancellation in x*cos(x)-sin(x).
        const double angular=numerics::gauss20([&](double z) {
            return 0.25*((1-z*z)*(0.5+0.25)/2+z*z*0.125)*std::cos(radius*z);
        },-1,1);
        RunResult result;
        estimator.finalize(result);
        const double expected=angular/(2*std::numbers::pi*std::numbers::pi);
        INFO("radius=" << radius);
        CHECK_THAT(series(result,"nk_g_kinetic")[0],Catch::Matchers::WithinAbs(expected,1e-12));
    }
}

TEST_CASE("Worm n(r) and n~(k) of ideal fermions in a 2D trap match the Mehler kernel",
          "[pimc][worm][physics]") {
    const std::size_t N = 3, M = 16, dim = 2;
    const double beta = 1.0, tau = beta / M;
    std::vector<double> z(N + 1, 0.0);
    for (std::size_t j = 1; j <= N; ++j)
        z[j] = std::pow(2.0 * std::sinh(static_cast<double>(j) * beta / 2.0), -2.0);
    const std::vector<double> Z = canonical_partitions(N, -1, z);
    // Cycle j contributes c_j z_j exp(-alpha_j r^2), alpha_j = coth(j beta/2)/4.
    const auto weight = [&](std::size_t j) {
        return ((j % 2 == 0) ? -1.0 : 1.0) * z[j] * Z[N - j] / (static_cast<double>(N) * Z[N]);
    };
    const auto alpha = [&](std::size_t j) {
        return 0.25 / std::tanh(static_cast<double>(j) * beta / 2.0);
    };

    const std::vector groups{group("g", 0, N, dim)};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(groups, 1.0));
    PimcConfig cfg = worm_config(beta, tau, 8642, 3, 6, 4);
    cfg.exchange_classes = {ExchangeClass{"g", 0, "fermions", {0, 1, 2}}};
    Pimc pimc(cfg, groups, ham, SimulationCell{}, dim, N);
    std::vector<std::unique_ptr<Estimator>> ests;
    ests.push_back(std::make_unique<MomentumDistribution>(groups, "g", 3.0, 6, 3.0, 4, 4,
                                                          SimulationCell{}, dim));
    pimc.set_estimators(std::move(ests));
    pimc.run();
    INFO(pimc.summary());
    CHECK_FALSE(pimc.sign_unreliable());
    const RunResult result = pimc.run_result();
    const auto r = series(result, "obdm_g_r"), n_r = series(result, "obdm_g"),
               n_r_err = series(result, "obdm_g_error");
    const double delta = 0.5;
    for (std::size_t b = 0; b < r.size(); ++b) {
        // Shell average of sum_j weight_j exp(-alpha_j r^2) over [r0, r1).
        const double r0 = static_cast<double>(b) * delta, r1 = r0 + delta;
        const double area = std::numbers::pi * (r1 * r1 - r0 * r0);
        double exact = 0.0;
        for (std::size_t j = 1; j <= N; ++j)
            exact += weight(j) * std::numbers::pi / alpha(j) *
                     (std::exp(-alpha(j) * r0 * r0) - std::exp(-alpha(j) * r1 * r1)) / area;
        INFO("r = " << r[b] << ": n(r) = " << n_r[b] << " +/- " << n_r_err[b]
             << " exact " << exact);
        CHECK(std::abs(n_r[b] - exact) <= tolerance(n_r_err[b]) + 0.01);
    }
    const auto k = series(result, "nk_g_k"), nk = series(result, "nk_g"),
               nk_err = series(result, "nk_g_error");
    REQUIRE(k.size() == 4);
    for (std::size_t s = 0; s < k.size(); ++s) {
        // n~(k) = sum_j weight_j (pi / alpha_j) exp(-k^2 / (4 alpha_j)).
        double exact = 0.0;
        for (std::size_t j = 1; j <= N; ++j)
            exact += weight(j) * std::numbers::pi / alpha(j) *
                     std::exp(-k[s] * k[s] / (4.0 * alpha(j)));
        INFO("k = " << k[s] << ": n~(k) = " << nk[s] << " +/- " << nk_err[s]
             << " exact " << exact);
        CHECK(std::abs(nk[s] - exact) <= tolerance(nk_err[s]) + 0.02);
    }
    // Exact finite-M harmonic transfer eigenvalues and characteristic
    // widths; independent Gaussian/Mehler algebra, not the estimator kernels.
    const double angle=std::asinh(tau/2.0),scale=std::sqrt(1+tau*tau/4);
    std::vector<double> finite_z(N+1,0.0);
    for (std::size_t j=1; j<=N; ++j)
        finite_z[j]=std::pow(2*std::sinh(static_cast<double>(j*M)*angle),-2.0);
    const auto finite_Z=canonical_partitions(N,-1,finite_z);
    double exact_number=0,exact_kinetic=0;
    for (std::size_t q=0; q<k.size(); ++q) {
        double exact=0;
        for (std::size_t j=1; j<=N; ++j) {
            const double a=0.25/(scale*std::tanh(static_cast<double>(j*M)*angle));
            const double sign=j%2==0 ? -1.0 : 1.0;
            exact+=sign*finite_z[j]*finite_Z[N-j]/(static_cast<double>(N)*finite_Z[N]) *
                   std::numbers::pi/a*std::exp(-k[q]*k[q]/(4*a));
        }
        const double radial=static_cast<double>(N)*(k[1]-k[0])*k[q]/(2*std::numbers::pi) *
                            (q==0 || q+1==k.size() ? 0.5 : 1.0);
        exact_number+=radial*exact;
        exact_kinetic+=radial*0.5*k[q]*k[q]*exact;
    }
    CHECK(std::abs(series(result,"nk_g_number")[0]-exact_number)<=
          4*series(result,"nk_g_number_error")[0]);
    CHECK(std::abs(series(result,"nk_g_kinetic")[0]-exact_kinetic)<=
          4*series(result,"nk_g_kinetic_error")[0]);
}
