// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Validation gate L1: the Ceperley
// permutation move samples ideal Bose and Fermi gases exactly at finite M.
//
// Reference: a k-cycle with M slices is one ring of kM beads, so the
// canonical partition function obeys the recursion (Borrmann & Franke,
// J. Chem. Phys. 98, 2484 (1993))
//     Z_N = (1/N) sum_{k=1}^{N} eta^{k+1} z(kM) Z_{N-k},  Z_0 = 1,
// eta = +1 for bosons, -1 for fermions, with the ring partition function z(K)
// of the SAME primitive action the code samples:
//   harmonic trap:  z(K) = prod_{j<K} (4 sin^2(pi j/K) + tau^2 w^2)^{-d/2}
//                   (normal modes of the ring, normalization included);
//   periodic box:   z(K) = [sum_n exp(-K tau eps_n)]^d, eps_n = (2 pi n/L)^2/2
//                   (the minimum-image Gaussian ring, exact up to discarded
//                   images ~ exp(-L^2 / (8 tau)) for the widths used here).
// E = -d ln Z_N / d beta at fixed M (d/d beta = (1/M) d/d tau), the average
// sign is Z_F / Z_B, and in the |weight| (bosonic) ensemble a particle sits
// in a k-cycle with probability z(kM) Z_{N-k} / (N Z_N).  Under PBC the cycle
// windings are independent, so <sum_c W_c^2> = sum_k (N P_k / k) <W^2>_{k beta}
// with <W^2>_{b} = sum_W W^2 exp(-L^2 W^2 / 2b) / sum_W exp(-L^2 W^2 / 2b).
//
// Energies and signs compare within the driver's reblocked / jackknife error
// (4 sigma + 1e-3, as test_pimc_physics.cpp).  Cycle fractions carry no error
// bar of their own; they are compared within an absolute 0.03, about three
// standard deviations for these run lengths.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <functional>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include "hamiltonian/coulomb.h"
#include "hamiltonian/harmonic_trap.h"
#include "estimators/density.h"
#include "pimc/move.h"
#include "pimc/pimc.h"
#include "pimc_ideal_reference.h"

namespace {
using namespace alea;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;

double tolerance(double error) { return 4.0 * error + 1e-3; }

using namespace pimc_reference;

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

void add_move(PimcConfig& cfg, const std::string& type, int levels, double step,
              int max_cycle = 4) {
    PimcConfig::MoveSpec move;
    move.type = type;
    move.levels = levels;
    move.step = step;
    move.max_cycle = max_cycle;
    cfg.moves.push_back(move);
}

PimcConfig ideal_config(double beta, double tau, std::uint64_t seed, int levels) {
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
    add_move(cfg, "bisection", levels, 0.0);
    add_move(cfg, "centroid", 0, 0.5);
    add_move(cfg, "permutation", levels, 0.0, 3);
    return cfg;
}

double scalar(const RunResult& result, const std::string& name) {
    for (const auto& [key, value] : result.scalars)
        if (key == name) return value;
    FAIL("missing run scalar " << name);
    return 0.0;
}
} // namespace

TEST_CASE("Ideal bosons in a 1D trap: energy and cycle lengths follow the canonical recursion",
          "[pimc][permutation][physics]") {
    const std::size_t N = 3, M = 16, dim = 1;
    const double beta = 2.0, tau = beta / M;
    const auto ring = [&](std::size_t K) { return harmonic_ring(K, tau, 1.0, dim); };
    const Canonical bosons = recursion(N, M, +1, ring);

    const std::vector groups{group("g", 0, N, dim)};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(groups, 1.0));
    PimcConfig cfg = ideal_config(beta, tau, 20261005, 3);
    cfg.exchange_classes = {ExchangeClass{"g", 0, "bosons", {0, 1, 2}}};
    Pimc pimc(cfg, groups, ham, SimulationCell{}, dim, N);
    pimc.run();
    INFO(pimc.summary());
    INFO("E = " << pimc.energy() << " +/- " << pimc.energy_error() << " exact " << bosons.energy);
    CHECK(pimc.quantum_statistics());
    CHECK_FALSE(pimc.signed_run());
    CHECK(std::abs(pimc.energy() - bosons.energy) <= tolerance(pimc.energy_error()));
    CHECK(pimc.energy_error() < 0.02 * bosons.energy);
    REQUIRE(pimc.cycle_fractions().size() == 1);
    for (std::size_t k = 0; k < N; ++k) {
        INFO("cycle length " << k + 1 << ": " << pimc.cycle_fractions()[0][k]
             << " exact " << bosons.cycle_fraction[k]);
        CHECK(std::abs(pimc.cycle_fractions()[0][k] - bosons.cycle_fraction[k]) < 0.03);
    }
    const RunResult result = pimc.run_result();
    CHECK(scalar(result, "move_2_cycle2_accepts") > 0);
    CHECK(scalar(result, "move_2_cycle3_accepts") > 0);
}

TEST_CASE("Ideal fermions in a 2D trap: signed energy and average sign follow the recursion",
          "[pimc][permutation][physics]") {
    const std::size_t N = 3, M = 16, dim = 2;
    const double beta = 1.0, tau = beta / M;
    const auto ring = [&](std::size_t K) { return harmonic_ring(K, tau, 1.0, dim); };
    const Canonical fermions = recursion(N, M, -1, ring);
    const Canonical bosons = recursion(N, M, +1, ring);
    const double sign_exact = fermions.partition / bosons.partition;

    const std::vector groups{group("g", 0, N, dim)};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(groups, 1.0));
    PimcConfig cfg = ideal_config(beta, tau, 777, 3);
    cfg.exchange_classes = {ExchangeClass{"g", 0, "fermions", {0, 1, 2}}};
    Pimc pimc(cfg, groups, ham, SimulationCell{}, dim, N);
    pimc.run();
    INFO(pimc.summary());
    INFO("E = " << pimc.energy() << " +/- " << pimc.energy_error() << " exact " << fermions.energy
         << " (bosons " << bosons.energy << ")");
    INFO("<s> = " << pimc.average_sign() << " +/- " << pimc.average_sign_error()
         << " exact " << sign_exact);
    CHECK(pimc.signed_run());
    CHECK_FALSE(pimc.sign_unreliable());
    CHECK(std::abs(pimc.average_sign() - sign_exact) <= tolerance(pimc.average_sign_error()));
    CHECK(std::abs(pimc.energy() - fermions.energy) <= tolerance(pimc.energy_error()));
    // The sign matters: the bosonic energy is many error bars away.
    CHECK(std::abs(bosons.energy - fermions.energy) > 10.0 * pimc.energy_error());
    // The |weight| chain is the bosonic one.
    for (std::size_t k = 0; k < N; ++k)
        CHECK(std::abs(pimc.cycle_fractions()[0][k] - bosons.cycle_fraction[k]) < 0.03);
}

TEST_CASE("Two flavours are independent exchange classes",
          "[pimc][permutation][physics]") {
    // Spin-up and spin-down fermions (2 + 2) in a 2D trap: Z = Z_2^F squared.
    const std::size_t M = 16, dim = 2;
    const double beta = 1.0, tau = beta / M;
    const auto ring = [&](std::size_t K) { return harmonic_ring(K, tau, 1.0, dim); };
    const Canonical pair_f = recursion(2, M, -1, ring), pair_b = recursion(2, M, +1, ring);
    const double sign_exact = std::pow(pair_f.partition / pair_b.partition, 2);

    const std::vector groups{group("e", 0, 4, dim)};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(groups, 1.0));
    PimcConfig cfg = ideal_config(beta, tau, 4242, 3);
    cfg.exchange_classes = {ExchangeClass{"e", 0, "fermions", {0, 1}},
                            ExchangeClass{"e", 1, "fermions", {2, 3}}};
    Pimc pimc(cfg, groups, ham, SimulationCell{}, dim, 4);
    pimc.run();
    INFO(pimc.summary());
    INFO("E = " << pimc.energy() << " +/- " << pimc.energy_error() << " exact " << 2 * pair_f.energy);
    CHECK(std::abs(pimc.energy() - 2.0 * pair_f.energy) <= tolerance(pimc.energy_error()));
    CHECK(std::abs(pimc.average_sign() - sign_exact) <= tolerance(pimc.average_sign_error()));
    REQUIRE(pimc.cycle_fractions().size() == 2);
    for (std::size_t c = 0; c < 2; ++c)
        for (std::size_t k = 0; k < 2; ++k)
            CHECK(std::abs(pimc.cycle_fractions()[c][k] - pair_b.cycle_fraction[k]) < 0.03);
}

TEST_CASE("Ideal bosons on a periodic ring: energy and cycle windings follow the recursion",
          "[pimc][permutation][physics]") {
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
    double w2_exact = 0.0;  // sum_c W_c^2 / (N d)
    for (std::size_t k = 1; k <= N; ++k)
        w2_exact += static_cast<double>(N) * bosons.cycle_fraction[k - 1] /
                    static_cast<double>(k) * ring_winding(static_cast<double>(k) * beta);
    w2_exact /= static_cast<double>(N * dim);

    const std::vector groups{group("g", 0, N, dim)};
    PimcConfig cfg = ideal_config(beta, tau, 99, 4);
    cfg.exchange_classes = {ExchangeClass{"g", 0, "bosons", {0, 1, 2}}};
    Pimc pimc(cfg, groups, Hamiltonian{}, SimulationCell(1, std::vector<double>{L}), dim, N);
    pimc.run();
    INFO(pimc.summary());
    INFO("E = " << pimc.energy() << " +/- " << pimc.energy_error() << " exact " << bosons.energy);
    INFO("W2 = " << pimc.winding2() << " +/- " << pimc.winding2_error() << " exact " << w2_exact);
    CHECK(std::abs(pimc.energy() - bosons.energy) <= tolerance(pimc.energy_error()));
    CHECK(std::abs(pimc.winding2() - w2_exact) <= tolerance(pimc.winding2_error()));
    CHECK(pimc.winding2_error() < 0.25 * w2_exact);
    for (std::size_t k = 0; k < N; ++k)
        CHECK(std::abs(pimc.cycle_fractions()[0][k] - bosons.cycle_fraction[k]) < 0.03);
}

TEST_CASE("Permutation moves keep the path, tables and sign consistent",
          "[pimc][permutation]") {
    // Interacting e-h bilayer: 3 electrons and 2 holes, both fermionic classes.
    const std::size_t M = 8, dim = 2;
    const std::vector groups{group("e", 0, 3, dim, -1.0, 1.0), group("h", 3, 5, dim, 1.0, 2.0)};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(groups, 0.5));
    ham.add_component(std::make_unique<CoulombTerm>(groups[0], groups[0], 2.0));
    ham.add_component(std::make_unique<CoulombTerm>(groups[1], groups[1], 2.0));
    ham.add_component(std::make_unique<CoulombTerm>(groups[0], groups[1], 2.0, 1.0));
    for (bool periodic : {false, true}) {
        CAPTURE(periodic);
        const SimulationCell cell = periodic
            ? SimulationCell(2, std::vector<double>{4.0, 0.0, 0.0, 4.0}) : SimulationCell{};
        PathSet path(M, 5, dim, 0.125, cell);
        path.set_exchange_classes(std::vector<std::size_t>{10, 10, 10, 20, 20},
                                  std::vector<char>{1, 1, 1, 1, 1});
        for (std::size_t p = 0; p < 5; ++p)
            for (std::size_t k = 0; k < M; ++k) {
                path.pos(p, 0, k) = 0.3 * std::sin(1.7 * p) + 0.05 * std::cos(0.9 * k + p);
                path.pos(p, 1, k) = 0.3 * std::cos(1.1 * p) + 0.05 * std::sin(0.7 * k);
            }
        path.wrap_positions();
        path.rebuild_tables();
        Action action(groups, ham, dim);
        PermutationMove permutation(2, 3);
        BisectionMove bisection(2);
        numerics::Random rng(31337);
        std::size_t accepted = 0;
        for (int attempt = 0; attempt < 4000; ++attempt) {
            Move& move = attempt % 2 == 0 ? static_cast<Move&>(permutation) : bisection;
            const bool accept = move.attempt(path, action, rng);
            if (accept && attempt % 2 == 0) {
                ++accepted;
                // The sign is the parity of pi, computed independently here.
                std::vector<char> seen(5, 0);
                double sign = 1.0;
                for (std::size_t p = 0; p < 5; ++p) {
                    if (seen[p]) continue;
                    std::size_t length = 0;
                    for (std::size_t q = p; !seen[q]; q = path.next_particle(q)) {
                        seen[q] = 1;
                        ++length;
                    }
                    if (length % 2 == 0) sign = -sign;
                }
                REQUIRE(path.permutation_sign() == sign);
                for (std::size_t p = 0; p < 5; ++p)
                    REQUIRE((path.next_particle(p) < 3) == (p < 3));  // classes kept
            }
            const auto tables = path.dist_tables().data();
            const std::vector<double> kept(tables.begin(), tables.end());
            path.rebuild_tables();
            const auto fresh = path.dist_tables().data();
            REQUIRE(std::equal(kept.begin(), kept.end(), fresh.begin()));
            REQUIRE(std::isfinite(action.total_action(path)));
        }
        CHECK(accepted > 0);
        const auto diagnostics = permutation.extra_diagnostics();
        double proposals = 0.0;
        for (const auto& [name, value] : diagnostics)
            if (name.ends_with("_proposals")) proposals += value;
        CHECK(proposals > 0.0);
    }
}

TEST_CASE("Multi-particle bead deltas equal full action differences",
          "[pimc][permutation]") {
    // Beads of several particles on the same slices, as a permutation move
    // stages them; telescoping must still give the exact difference.
    const std::size_t M = 6, dim = 2;
    const std::vector groups{group("e", 0, 3, dim, -1.0), group("h", 3, 5, dim, 1.0, 2.0)};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(groups, 0.5));
    ham.add_component(std::make_unique<CoulombTerm>(groups[0], groups[0], 2.0));
    ham.add_component(std::make_unique<CoulombTerm>(groups[0], groups[1], 2.0, 1.0));
    PathSet path(M, 5, dim, 0.125);
    path.set_exchange_classes(std::vector<std::size_t>{10, 10, 10, 20, 20},
                              std::vector<char>{1, 1, 1, 1, 1});
    for (std::size_t p = 0; p < 5; ++p)
        for (std::size_t k = 0; k < M; ++k) {
            path.pos(p, 0, k) = 0.8 * std::sin(1.3 * p + 0.37 * k);
            path.pos(p, 1, k) = 0.7 * std::cos(0.9 * p + 0.51 * k);
        }
    path.rebuild_tables();
    Action potential(groups, ham, dim, false, true);
    const std::vector<PathSet::Bead> beads{{0, 4}, {1, 4}, {2, 5}, {0, 5}, {1, 0}, {3, 4}};
    std::vector<double> proposed;
    PathSet changed(M, 5, dim, 0.125);
    std::copy(path.positions().begin(), path.positions().end(), changed.positions().begin());
    for (std::size_t i = 0; i < beads.size(); ++i)
        for (std::size_t d = 0; d < dim; ++d) {
            const double value = path.pos(beads[i].particle, d, beads[i].slice) + 0.1 * std::sin(2.0 * i + d);
            proposed.push_back(value);
            changed.pos(beads[i].particle, d, beads[i].slice) = value;
        }
    const auto saved = path.all_positions();
    const double full = potential.total_action(changed) - potential.total_action(path);
    CHECK(potential.delta_potential_beads(path, beads, proposed) == Approx(full).margin(1e-11));
    CHECK(path.all_positions() == saved);
}

TEST_CASE("The driver pairs permutation moves with quantum statistics",
          "[pimc][permutation]") {
    const std::vector groups{group("g", 0, 2, 1)};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(groups, 1.0));
    PimcConfig cfg = ideal_config(1.0, 0.125, 5, 2);
    cfg.exchange_classes = {ExchangeClass{"g", 0, "fermions", {0, 1}}};
    CHECK_NOTHROW(Pimc(cfg, groups, ham, SimulationCell{}, 1, 2));

    PimcConfig boltzmannon = cfg;
    boltzmannon.statistics = "boltzmannon";
    CHECK_THROWS_WITH(Pimc(boltzmannon, groups, ham, SimulationCell{}, 1, 2),
                      ContainsSubstring("needs quantum statistics"));

    PimcConfig no_permutation = cfg;
    no_permutation.moves.pop_back();
    CHECK_THROWS_WITH(Pimc(no_permutation, groups, ham, SimulationCell{}, 1, 2),
                      ContainsSubstring("permutation"));

    PimcConfig too_wide = cfg;
    too_wide.moves.back().levels = 4;  // 16 > M = 8
    CHECK_THROWS_WITH(Pimc(too_wide, groups, ham, SimulationCell{}, 1, 2),
                      ContainsSubstring("permutation levels"));
    PimcConfig bad_cycle = cfg;
    bad_cycle.moves.back().max_cycle = 1;
    CHECK_THROWS_WITH(Pimc(bad_cycle, groups, ham, SimulationCell{}, 1, 2),
                      ContainsSubstring("max_cycle"));
}

TEST_CASE("Rooted cycle normalizers equal independently enumerated free endpoint weights",
          "[pimc][permutation][contract]") {
    // The free-bridge proposal's endpoint weight is the product of its
    // propagators. Enumerate all 4! target assignments instead
    // of reproducing CycleTable's adjacency/DFS traversal. This checks every
    // current permutation, root and cycle cap, including symmetric pruning.
    // Reference: Ceperley, Rev. Mod. Phys. 67, 279 (1995), Sec. V.F.
    constexpr std::size_t N = 4, M = 8, dim = 2, window = 4;
    constexpr double tau = 0.25, cell_length = 5.0;
    const std::array<double, dim> mass{1.0, 1.7}; // [dimension]
    const std::array<std::array<double, dim>, N> left{{ // [particle][dimension]
        {-2.2, -0.7}, {-0.4, 0.5}, {1.3, -0.2}, {2.2, 1.1}}};
    const std::array<std::array<double, dim>, N> right{{ // [particle][dimension]
        {2.2, 0.6}, {0.5, -0.8}, {-1.2, 0.1}, {-2.3, -1.0}}};
    const std::array<std::size_t, N> members{0, 1, 2, 3}; // [particle]
    for (const bool periodic : {false, true}) {
        const SimulationCell cell = periodic
            ? SimulationCell(dim, std::vector<double>{cell_length, 0.0, 0.0, cell_length})
            : SimulationCell{};
        PathSet path(M, N, dim, tau, cell);
        path.set_exchange_classes(std::vector<std::size_t>(N, 0), std::vector<char>(N, 0));
        std::array<double, N * N> propagator{}; // [left member, right member]
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t d = 0; d < dim; ++d) {
                path.pos(i, d, M - window) = left[i][d];
                path.pos(i, d, 0) = right[i][d];
            }
            for (std::size_t j = 0; j < N; ++j) {
                double exponent = 0.0;
                for (std::size_t d = 0; d < dim; ++d) {
                    double distance = right[j][d] - left[i][d];
                    if (periodic)
                        distance -= cell_length * std::floor(distance / cell_length + 0.5);
                    exponent += mass[d] * distance * distance / (2.0 * window * tau);
                }
                propagator[i * N + j] = std::exp(-exponent);
            }
        }
        const auto endpoint_weight = [&](const std::array<std::size_t, N>& sigma) {
            double weight = 1.0;
            for (std::size_t i = 0; i < N; ++i) weight *= propagator[i * N + sigma[i]];
            return weight;
        };
        for (const std::size_t cap : {2u, 3u, 4u}) {
            CycleTable table;
            table.prepare(N, cap);
            auto old = members;
            do {
                path.set_permutation(old);
                table.build(path, members, M - window, 0, window, mass.data(), cap);
                const double old_weight = endpoint_weight(old);
                for (std::size_t root = 0; root < N; ++root) {
                    double expected = 1.0; // identity proposal
                    auto proposed = members;
                    do {
                        std::size_t changed = 0, first = N;
                        bool admissible = true;
                        for (std::size_t i = 0; i < N; ++i) {
                            if (proposed[i] == old[i]) continue;
                            ++changed;
                            first = std::min(first, i);
                            admissible = admissible &&
                                propagator[i * N + old[i]] >= CycleTable::kEpsilon &&
                                propagator[i * N + proposed[i]] >= CycleTable::kEpsilon;
                        }
                        if (!admissible || first != root || changed < 2 || changed > cap) continue;
                        // sigma' must differ by one cycle, not disjoint swaps.
                        std::size_t visited = 0, member = root;
                        do {
                            member = static_cast<std::size_t>(
                                std::find(old.begin(), old.end(), proposed[member]) - old.begin());
                            ++visited;
                        } while (member != root);
                        if (visited == changed) expected += endpoint_weight(proposed) / old_weight;
                    } while (std::next_permutation(proposed.begin(), proposed.end()));
                    CAPTURE(periodic, cap, old, root);
                    CHECK(table.cycle_sum_from(root) == Approx(expected).epsilon(2e-12));
                }
            } while (std::next_permutation(old.begin(), old.end()));
        }
    }
}

// Speed report G4: time per permutation
// attempt against a bisection attempt of the same window, for free 2D
// bosons at unit density (L = sqrt(N)), M = 32, tau = 1/32, levels 3.  The
// endpoint table is O(N^2) per attempt, so the ratio is expected to grow
// with N.  Hidden: test_pimc_permutation "[pimc-permutation-bench]" -s
TEST_CASE("G4: permutation attempt cost against bisection as N grows",
          "[.][pimc-permutation-bench]") {
    for (std::size_t N : {4u, 8u, 16u, 32u}) {
        const std::size_t M = 32, dim = 2;
        const double L = std::sqrt(static_cast<double>(N));
        const SimulationCell cell(2, std::vector<double>{L, 0.0, 0.0, L});
        const std::vector groups{group("g", 0, N, dim)};
        Action action(groups, Hamiltonian{}, dim);
        PathSet path(M, N, dim, 1.0 / 32, cell);
        path.set_exchange_classes(std::vector<std::size_t>(N, 7), std::vector<char>(N, 0));
        numerics::Random rng(N);
        for (std::size_t p = 0; p < N; ++p) {
            const double x = rng.uniform(0.0, L), y = rng.uniform(0.0, L);
            for (std::size_t k = 0; k < M; ++k) { path.pos(p, 0, k) = x; path.pos(p, 1, k) = y; }
        }
        path.wrap_positions();
        path.rebuild_tables();
        PermutationMove permutation(3, 4);
        BisectionMove bisection(3);
        for (int i = 0; i < 20000; ++i) bisection.attempt(path, action, rng);  // thermalize rings
        const auto time = [&](Move& move, int attempts, std::size_t& accepted) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < attempts; ++i) accepted += move.attempt(path, action, rng);
            return std::chrono::duration<double, std::micro>(
                       std::chrono::steady_clock::now() - start).count() / attempts;
        };
        std::size_t acc_p = 0, acc_b = 0;
        const int attempts = 2000;
        const double t_p = time(permutation, attempts, acc_p);
        const double t_b = time(bisection, attempts, acc_b);
        std::cout << "N=" << N << "  permutation " << t_p << " us (acc "
                  << static_cast<double>(acc_p) / attempts << ")  bisection " << t_b
                  << " us (acc " << static_cast<double>(acc_b) / attempts
                  << ")  ratio " << t_p / t_b << std::endl;
        CHECK(std::isfinite(t_p));
    }
}

// ── Cycle-centroid virial and signed estimators ─────────────────────────────

TEST_CASE("The cycle-centroid virial reproduces exact ideal-gas energies with exchange",
          "[pimc][permutation][physics][virial]") {
    // Bosons in a 1D trap and fermions in a 2D trap (the L1 systems), now with
    // energy: centroid_virial.  With exchange each loop is a cycle: the free
    // term is d n_cycles / 2 beta and the virial uses cycle centroids.
    struct Case { std::size_t dim; double beta; int eta; const char* statistics; };
    for (const Case c : {Case{1, 2.0, +1, "bosons"}, Case{2, 1.0, -1, "fermions"}}) {
        CAPTURE(c.statistics);
        const std::size_t N = 3, M = 16;
        const double tau = c.beta / M;
        const auto ring = [&](std::size_t K) { return harmonic_ring(K, tau, 1.0, c.dim); };
        const Canonical exact = recursion(N, M, c.eta, ring);
        const std::vector groups{group("g", 0, N, c.dim)};
        Hamiltonian ham;
        ham.add_component(std::make_unique<HarmonicTrap>(groups, 1.0));
        PimcConfig cfg = ideal_config(c.beta, tau, 4096, 3);
        cfg.energy = "centroid_virial";
        cfg.compare_thermodynamic = true;
        cfg.exchange_classes = {ExchangeClass{"g", 0, c.statistics, {0, 1, 2}}};
        Pimc pimc(cfg, groups, ham, SimulationCell{}, c.dim, N);
        pimc.run();
        INFO(pimc.summary());
        INFO("E_cv = " << pimc.energy() << " +/- " << pimc.energy_error() << " exact " << exact.energy);
        CHECK(std::abs(pimc.energy() - exact.energy) <= tolerance(pimc.energy_error()));
        CHECK(pimc.energy_error() < 0.02 * exact.energy);
        // The paired virial-minus-thermodynamic difference is measured (also
        // in signed runs) and vanishes on average: both estimate the same E.
        const RunResult result = pimc.run_result();
        const double difference = scalar(result, "energy_difference");
        const double difference_error = scalar(result, "energy_difference_error");
        INFO("E_cv - E_th = " << difference << " +/- " << difference_error);
        CHECK(difference_error > 0.0);
        CHECK(std::abs(difference) <= tolerance(difference_error));
    }
}

TEST_CASE("The signed one-body density of two fermions matches the exact Gaussian mixture",
          "[pimc][permutation][physics][signed-estimators]") {
    // 2 fermions in a 1D trap, beta = 1, M = 16 (<s> = 0.368).  At finite M a
    // ring of K = kM beads has a Gaussian one-bead marginal with variance
    // sigma_k^2 = (1/K) sum_j 1/lambda_j, lambda_j = (4 sin^2(pi j/K) + tau^2 w^2)/tau,
    // so the slice density is n(x) = sum_k eta^{k+1} z(kM) Z_{N-k} G_k(x) / Z_N.
    const std::size_t N = 2, M = 16, dim = 1;
    const double beta = 1.0, tau = beta / M, rmax = 3.0;
    const int n_bins = 24;
    const auto ring = [&](std::size_t K) { return harmonic_ring(K, tau, 1.0, dim); };
    const auto sigma2 = [&](std::size_t K) {
        double sum = 0.0;
        for (std::size_t j = 0; j < K; ++j) {
            const double s = std::sin(std::numbers::pi * static_cast<double>(j) / static_cast<double>(K));
            sum += tau / (4.0 * s * s + tau * tau);
        }
        return sum / static_cast<double>(K);
    };
    const auto exact_density = [&](int eta) {
        std::vector<double> z(N + 1), Z(N + 1, 0.0);
        for (std::size_t k = 1; k <= N; ++k) z[k] = std::exp(ring(k * M).log_z);
        Z[0] = 1.0;
        for (std::size_t n = 1; n <= N; ++n) {
            for (std::size_t k = 1; k <= n; ++k)
                Z[n] += ((eta < 0 && k % 2 == 0) ? -1.0 : 1.0) * z[k] * Z[n - k];
            Z[n] /= static_cast<double>(n);
        }
        const double width = 2.0 * rmax / n_bins;
        std::vector<double> density(n_bins, 0.0);
        for (std::size_t k = 1; k <= N; ++k) {
            // Particles in k-cycles: z(kM) Z_{N-k} / Z_N (the cycle fractions
            // times N); the sum over k is N by the recursion.
            const double c = ((eta < 0 && k % 2 == 0) ? -1.0 : 1.0) * z[k] * Z[N - k] / Z[N];
            const double s = std::sqrt(sigma2(k * M));
            for (int b = 0; b < n_bins; ++b) {
                const double lo = -rmax + b * width, hi = lo + width;
                density[b] += c * 0.5 * (std::erf(hi / (std::sqrt(2.0) * s)) -
                                         std::erf(lo / (std::sqrt(2.0) * s))) / width;
            }
        }
        return density;
    };
    const auto fermion_density = exact_density(-1), boson_density = exact_density(+1);

    const std::vector groups{group("g", 0, N, dim)};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(groups, 1.0));
    PimcConfig cfg = ideal_config(beta, tau, 777, 3);
    cfg.exchange_classes = {ExchangeClass{"g", 0, "fermions", {0, 1}}};
    Pimc pimc(cfg, groups, ham, SimulationCell{}, dim, N);
    std::vector<std::unique_ptr<Estimator>> estimators;
    estimators.push_back(std::make_unique<Density>(groups, dim, rmax, n_bins));
    pimc.set_estimators(std::move(estimators));
    pimc.run();
    INFO(pimc.summary());
    const RunResult result = pimc.run_result();
    const auto series = [&](const std::string& name) {
        for (const auto& [key, values] : result.series)
            if (key == name) return values;
        FAIL("missing series " << name);
        return std::vector<double>{};
    };
    const auto density = series("density_g"), error = series("density_g_error");
    REQUIRE(density.size() == static_cast<std::size_t>(n_bins));
    double distance_to_bosons = 0.0;
    for (int b = 0; b < n_bins; ++b) {
        INFO("bin " << b << ": " << density[b] << " +/- " << error[b]
             << " exact " << fermion_density[b] << " (bosons " << boson_density[b] << ")");
        CHECK(std::abs(density[b] - fermion_density[b]) <= 4.0 * error[b] + 2e-3);
        distance_to_bosons = std::max(distance_to_bosons,
            std::abs(fermion_density[b] - boson_density[b]) / (error[b] + 1e-12));
    }
    // The signed estimator resolves the Pauli difference from the bosons.
    CHECK(distance_to_bosons > 10.0);
}
