// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Exchange infrastructure: the
// permuted closing link (p, M-1) -> (pi(p), 0) wired through PathSet, the
// action, the moves and the driver's signed estimators.
//
// Every check uses a reference that does not trust the code under test:
//  * hand-computed springs of a permuted two-particle path;
//  * moving the permutation cut (relabeling imaginary time) leaves every
//    action component unchanged, and a path read with the wrong closing link
//    does not — a sensitive test of the wiring with no re-implemented formula;
//  * worldline window and whole-cycle deltas against complete before/after
//    action differences;
//  * the fixed-permutation-sector driver against exact sector partition
//    functions.  A 2-cycle of two particles is ONE ring of 2M beads (Ceperley,
//    Rev. Mod. Phys. 67, 279 (1995), Sec. V), so Z_P(beta) = Z_1^{(2M)}(2 beta):
//      harmonic trap (primitive action, normal modes of the ring):
//        E_P = (tau w^2 / M) sum_{k=0}^{2M-1} 1/(4 sin^2(pi k/2M) + tau^2 w^2);
//      free particles in a periodic box: E_P = 2 E_1(2 beta), and the cycle
//        winding is distributed as exp(-m L^2 W^2 / (2 * 2 beta)).
// Statistical comparisons use the driver's own error bar at a fixed seed with
// the same 4 sigma + 1e-3 tolerance as test_pimc_physics.cpp.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "estimators/estimator.h"
#include "hamiltonian/coulomb.h"
#include "hamiltonian/harmonic_trap.h"
#include "pimc/move.h"
#include "pimc/pair_product_action.h"
#include "pimc/pimc.h"
#include "pimc/primitive_action.h"

namespace {
using namespace alea;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;

std::vector<std::size_t> ids(std::initializer_list<std::size_t> values) {
    return std::vector<std::size_t>(values);
}

ParticleGroup group(std::string name, std::size_t first, std::size_t last,
                    double charge, std::size_t dim, double mass = 1.0) {
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

double tolerance(double error) { return 4.0 * error + 1e-3; }

std::vector<double> table_snapshot(const PathSet& path) {
    const auto values = path.dist_tables().data();
    return {values.begin(), values.end()};
}

// Electrons 0..ne-1 (charge -1, mass 1) and holes ne..ne+nh-1 (charge +1,
// mass 2) in 2D: a trap, same-species Coulomb and a layer-separated e-h term
// (finite, so the primitive action admits it).
struct EhSystem {
    std::vector<ParticleGroup> groups;
    Hamiltonian ham, coulomb;
    std::vector<std::size_t> class_id;
    std::vector<char> fermionic;
    EhSystem(std::size_t ne, std::size_t nh) {
        groups = {group("electrons", 0, ne, -1.0, 2, 1.0),
                  group("holes", ne, ne + nh, 1.0, 2, 2.0)};
        ham.add_component(std::make_unique<HarmonicTrap>(groups, 0.5));
        for (Hamiltonian* h : {&ham, &coulomb}) {
            h->add_component(std::make_unique<CoulombTerm>(groups[0], groups[0], 2.0));
            h->add_component(std::make_unique<CoulombTerm>(groups[1], groups[1], 2.0));
            h->add_component(std::make_unique<CoulombTerm>(groups[0], groups[1], 2.0, 1.0));
        }
        for (std::size_t p = 0; p < ne + nh; ++p) {
            class_id.push_back(p < ne ? 100 : 200);
            fermionic.push_back(1);
        }
    }
    // Spread beads deterministically, without coincidences.
    void fill(PathSet& path) const {
        for (std::size_t p = 0; p < path.n_particles(); ++p)
            for (std::size_t k = 0; k < path.num_slices(); ++k) {
                path.pos(p, 0, k) = 0.8 * std::sin(1.3 * p + 0.37 * k) + 0.1 * p;
                path.pos(p, 1, k) = 0.7 * std::cos(0.9 * p + 0.51 * k) - 0.05 * p;
            }
        path.rebuild_tables();
    }
};

// Relabel imaginary time by `shift` slices: new bead (p, k) is old bead
// (p, k+shift) while k+shift < M and old (pi(p), k+shift-M) afterwards.  The
// worldlines, every link and every slice configuration are the same sets and
// pi is unchanged; only the place of the permuted closing link moves.
PathSet rotate_time_origin(const PathSet& path, std::size_t shift,
                           const EhSystem& system) {
    const auto M = path.num_slices();
    PathSet out(M, path.n_particles(), path.dim(), path.tau(), path.cell());
    out.set_exchange_classes(system.class_id, system.fermionic);
    out.set_permutation(path.permutation());
    for (std::size_t p = 0; p < path.n_particles(); ++p)
        for (std::size_t d = 0; d < path.dim(); ++d)
            for (std::size_t k = 0; k < M; ++k) {
                const auto old_k = k + shift;
                out.pos(p, d, k) = old_k < M ? path.pos(p, d, old_k)
                                             : path.pos(path.next_particle(p), d, old_k - M);
            }
    out.rebuild_tables();
    return out;
}

// A slice-local potential component without a delta override: every move
// falls back to evaluating complete slices (incremental-move test pattern).
class FullSliceAction final : public ActionComponent {
public:
    bool slice_local_deltas() const noexcept override { return true; }
    explicit FullSliceAction(const Hamiltonian& ham) : ham_(ham.clone()) {}
    FullSliceAction(const FullSliceAction& other)
        : ham_(other.ham_.clone()), scratch_(other.scratch_) {}
    void prepare(std::size_t slices) override { scratch_.resize(slices); }
    double potential_action_slice(const PathSet& path, std::size_t slice) override {
        ham_.evaluate_potential(path, scratch_, slice, slice + 1);
        return path.tau() * scratch_[slice];
    }
    double total_action(const PathSet& path) override {
        double value = 0.0;
        for (std::size_t slice = 0; slice < path.num_slices(); ++slice)
            value += potential_action_slice(path, slice);
        return value;
    }
    bool supports_free_bridge() const noexcept override { return true; }
    std::unique_ptr<ActionComponent> clone() const override {
        return std::make_unique<FullSliceAction>(*this);
    }
private:
    Hamiltonian ham_;
    std::vector<double> scratch_; // [M]
};

// Minimal estimator: lets the driver's estimator gates be exercised.
class NullEstimator final : public Estimator {
public:
    void start_block() override {}
    void accumulate(const ParticleBatch&, std::span<const double>) override {}
    void stop_block() override {}
    void collect(const Estimator&) override {}
    void finalize(RunResult&) const override {}
    void configure_reblocking(std::size_t) override {}
    std::size_t reblocking_storage_bytes(std::size_t) const override { return 0; }
    std::unique_ptr<Estimator> clone() const override {
        return std::make_unique<NullEstimator>(*this);
    }
    std::string type() const override { return "null"; }
    std::vector<std::string> output_namespaces() const override { return {"null"}; }
};

PimcConfig sector_config(double beta, double tau, std::uint64_t seed) {
    PimcConfig cfg;
    cfg.threads = 2;
    cfg.samples = 32;
    cfg.beta = beta;
    cfg.time_step = tau;
    cfg.equilibration_blocks = 200;
    cfg.accumulation_blocks = 800;
    cfg.steps_per_block = 20;
    cfg.sub_steps = 2;
    cfg.seed = seed;
    return cfg;
}

void add_move(PimcConfig& cfg, const std::string& type, int levels, double step) {
    PimcConfig::MoveSpec move;
    move.type = type;
    move.levels = levels;
    move.step = step;
    cfg.moves.push_back(move);
}

// Every replica's paths with the permutation `pi` (beads as exported).
io::PathCheckpoint with_permutation(const Pimc& pimc, std::span<const std::size_t> pi) {
    io::PathCheckpoint checkpoint = pimc.export_paths();
    for (std::size_t r = 0; r < checkpoint.replicas; ++r)
        std::copy(pi.begin(), pi.end(),
                  checkpoint.permutation.begin() + static_cast<std::ptrdiff_t>(r * checkpoint.np));
    return checkpoint;
}

// Exact sector energy of a 2-cycle in a 1D harmonic trap, M slices each.
double harmonic_two_cycle_energy(double tau, std::size_t M, double omega) {
    const std::size_t beads = 2 * M;
    double sum = 0.0;
    for (std::size_t k = 0; k < beads; ++k) {
        const double s = std::sin(std::numbers::pi * static_cast<double>(k) /
                                  static_cast<double>(beads));
        sum += 1.0 / (4.0 * s * s + tau * tau * omega * omega);
    }
    return tau * omega * omega / static_cast<double>(M) * sum;
}
} // namespace

TEST_CASE("PathSet admits only class-preserving bijections and caches cycles and sign",
          "[pimc][exchange]") {
    PathSet path(4, 5, 1, 0.25);
    CHECK(path.identity_permutation());
    CHECK(path.permutation_sign() == 1.0);
    CHECK(path.largest_exchange_class() == 1);
    // By default every particle is its own class, so nothing may exchange.
    CHECK_THROWS_AS(path.set_permutation(ids({1, 0, 2, 3, 4})), std::invalid_argument);
    CHECK(path.identity_permutation());

    path.set_exchange_classes(ids({7, 7, 7, 9, 9}), std::vector<char>{1, 1, 1, 0, 0});
    CHECK(path.largest_exchange_class() == 3);

    // A fermion 3-cycle (two transpositions) and a boson 2-cycle: sign +1.
    path.set_permutation(ids({1, 2, 0, 4, 3}));
    CHECK_FALSE(path.identity_permutation());
    CHECK(path.permutation_sign() == 1.0);
    CHECK(std::vector<std::size_t>(path.cycle_leaders().begin(), path.cycle_leaders().end()) ==
          ids({0, 3}));
    CHECK(path.next_particle(2) == 0);
    CHECK(path.prev_particle(0) == 2);
    const auto closing = path.bead_after(1, 3);
    CHECK(closing.particle == 2);
    CHECK(closing.slice == 0);
    const auto inner = path.bead_after(1, 1);
    CHECK(inner.particle == 1);
    CHECK(inner.slice == 2);
    const auto incoming = path.bead_before(0, 0);
    CHECK(incoming.particle == 2);
    CHECK(incoming.slice == 3);

    // One fermion transposition: sign -1.
    path.set_permutation(ids({1, 0, 2, 3, 4}));
    CHECK(path.permutation_sign() == -1.0);
    CHECK(std::vector<std::size_t>(path.cycle_leaders().begin(), path.cycle_leaders().end()) ==
          ids({0, 2, 3, 4}));

    // Every rejection leaves the path as it was.
    CHECK_THROWS_AS(path.set_permutation(ids({3, 1, 2, 0, 4})), std::invalid_argument); // crosses classes
    CHECK_THROWS_AS(path.set_permutation(ids({0, 0, 2, 3, 4})), std::invalid_argument); // not a bijection
    CHECK_THROWS_AS(path.set_permutation(ids({0, 1, 2, 3})), std::invalid_argument);    // wrong size
    CHECK_THROWS_AS(path.set_permutation(ids({0, 1, 2, 3, 5})), std::invalid_argument); // out of range
    CHECK(std::vector<std::size_t>(path.permutation().begin(), path.permutation().end()) ==
          ids({1, 0, 2, 3, 4}));
    CHECK(path.permutation_sign() == -1.0);
    // A class may not mix statistics, and new classes may not split the current pi.
    CHECK_THROWS_AS(path.set_exchange_classes(ids({7, 7, 7, 9, 9}), std::vector<char>{1, 0, 1, 0, 0}),
                    std::invalid_argument);
    CHECK_THROWS_AS(path.set_exchange_classes(ids({7, 8, 7, 9, 9}), std::vector<char>{1, 1, 1, 0, 0}),
                    std::invalid_argument);
}

TEST_CASE("Kinetic action of a permuted path matches hand-computed springs",
          "[pimc][exchange]") {
    const std::vector groups{group("g", 0, 2, 0.0, 1)};
    Action action(groups, Hamiltonian{}, 1, true, false);
    PathSet path(2, 2, 1, 0.5);  // m/(2 tau) = 1
    path.pos(0, 0, 0) = 0.0;
    path.pos(0, 0, 1) = 1.0;
    path.pos(1, 0, 0) = 3.0;
    path.pos(1, 0, 1) = 5.0;
    path.rebuild_tables();
    // Identity: rings 0 -> 1 -> 0 and 3 -> 5 -> 3, springs 1 + 1 + 4 + 4.
    CHECK(action.total_action(path) == Approx(10.0));

    path.set_exchange_classes(ids({4, 4}), std::vector<char>{1, 1});
    path.set_permutation(ids({1, 0}));
    // One worldline 0 -> 1 -> [pi(0) = 1] 3 -> 5 -> [pi(1) = 0] 0:
    // springs 1 + 4 + 4 + 25.
    CHECK(action.total_action(path) == Approx(34.0));
    CHECK(action.link_action(path, 0, 1) == Approx(4.0));
    CHECK(action.link_action(path, 1, 1) == Approx(25.0));
    // The same-particle spring remains an explicit pair of beads.
    CHECK(action.spring_action(path, 0, 1, 0) == Approx(1.0));
}

TEST_CASE("Moving the permutation cut leaves every action component unchanged",
          "[pimc][exchange]") {
    const EhSystem system(3, 2);
    PathSet path(6, 5, 2, 0.125);
    path.set_exchange_classes(system.class_id, system.fermionic);
    system.fill(path);
    path.set_permutation(ids({1, 2, 0, 4, 3}));  // electron 3-cycle, hole swap

    Action primitive(system.groups, system.ham, 2);
    Action pair(system.groups, Hamiltonian{}, 2, true, false);
    pair.add_component(std::make_unique<PairProductAction>(system.groups, system.coulomb, 2, 6.0, 9));
    pair.prepare_time_step(path.tau());

    for (Action* action : {&primitive, &pair}) {
        const double reference = action->total_action(path);
        REQUIRE(std::isfinite(reference));
        for (std::size_t shift : {1u, 2u, 5u}) {
            CAPTURE(shift);
            PathSet moved = rotate_time_origin(path, shift, system);
            CHECK(action->total_action(moved) == Approx(reference).epsilon(1e-11));
        }
        // Sensitivity: the same beads read as identity rings are another path.
        PathSet rings = rotate_time_origin(path, 0, system);
        rings.set_permutation(ids({0, 1, 2, 3, 4}));
        CHECK(std::abs(action->total_action(rings) - reference) > 1e-3);
    }
}

TEST_CASE("Worldline window and whole-cycle deltas equal full action differences",
          "[pimc][exchange]") {
    const EhSystem system(3, 2);
    const std::size_t M = 6, dim = 2;
    PathSet path(M, 5, dim, 0.125);
    path.set_exchange_classes(system.class_id, system.fermionic);
    system.fill(path);
    path.set_permutation(ids({1, 2, 0, 4, 3}));

    Action primitive(system.groups, system.ham, dim, false, true);
    Action pair(system.groups, Hamiltonian{}, dim, false, false);
    pair.add_component(std::make_unique<PairProductAction>(system.groups, system.coulomb, dim, 6.0, 9));
    pair.prepare_time_step(path.tau());

    struct Window { std::size_t p, first, count; };
    const std::vector<Window> windows{
        {0, 4, 4},   // crosses the boundary onto pi(0) = 1
        {1, 0, 3},   // starts on slice 0 (incoming permuted link)
        {3, 5, M},   // a full M-bead window across the boundary
        {2, 2, 2},   // interior only
        {0, 0, 3 * M},  // the whole electron 3-cycle
        {4, 3, 2 * M},  // the whole hole 2-cycle, starting mid-ring
    };
    for (Action* action : {&primitive, &pair}) {
        for (const Window& w : windows) {
            CAPTURE(w.p, w.first, w.count);
            std::vector<double> proposed(w.count * dim);
            PathSet changed = rotate_time_origin(path, 0, system);
            // Owners follow the worldline: next slice, and pi at the boundary.
            for (std::size_t i = 0, owner = w.p; i < w.count; ++i) {
                const auto k = (w.first + i) % M;
                if (i > 0 && k == 0) owner = path.next_particle(owner);
                for (std::size_t d = 0; d < dim; ++d) {
                    proposed[i * dim + d] = path.pos(owner, d, k) + 0.07 * std::sin(3.1 * i + d);
                    changed.pos(owner, d, k) = proposed[i * dim + d];
                }
            }
            changed.rebuild_tables();
            const auto saved = path.all_positions();
            const auto saved_tables = table_snapshot(path);
            const double before = action->total_action(path);
            const double full = action->total_action(changed) - before;
            const double delta = action->delta_potential_window(path, w.p, w.first, proposed);
            CHECK(delta == Approx(full).margin(1e-10));
            CHECK(path.all_positions() == saved);
            CHECK(table_snapshot(path) == saved_tables);
        }
    }
    // A window longer than M must be exactly one whole cycle.
    std::vector<double> too_long((3 * M - 1) * dim, 0.0);
    CHECK_THROWS_AS(primitive.delta_potential_window(path, 0, 0, too_long), std::invalid_argument);
}

TEST_CASE("Window scratch follows a change of M before a whole-cycle window",
          "[pimc][exchange]") {
    // One action prepared for M = 12, then used on an M = 6 path: the cycle
    // window (3 * 6 beads) must still find enough staging scratch.
    const EhSystem system(3, 2);
    Action primitive(system.groups, system.ham, 2, false, true);
    for (std::size_t M : {12u, 6u}) {
        CAPTURE(M);
        PathSet path(M, 5, 2, 0.125);
        path.set_exchange_classes(system.class_id, system.fermionic);
        system.fill(path);
        path.set_permutation(ids({1, 2, 0, 4, 3}));
        std::vector<double> proposed(3 * M * 2);
        PathSet changed = rotate_time_origin(path, 0, system);
        for (std::size_t i = 0, owner = 0; i < 3 * M; ++i) {
            const auto k = i % M;
            if (i > 0 && k == 0) owner = path.next_particle(owner);
            for (std::size_t d = 0; d < 2; ++d) {
                proposed[i * 2 + d] = path.pos(owner, d, k) + 0.05;
                changed.pos(owner, d, k) = proposed[i * 2 + d];
            }
        }
        const double full = primitive.total_action(changed) - primitive.total_action(path);
        CHECK(primitive.delta_potential_window(path, 0, 0, proposed) == Approx(full).margin(1e-10));
    }
}

TEST_CASE("Moves on permuted paths match full-slice evaluation and keep tables exact",
          "[pimc][exchange]") {
    const EhSystem system(3, 2);
    for (bool periodic : {false, true}) {
        const SimulationCell cell = periodic
            ? SimulationCell(2, std::array{12.0, 0.0, 0.0, 12.0}) : SimulationCell{};
        for (int kind : {0, 1, 2}) {
            CAPTURE(periodic, kind);
            PathSet fast(8, 5, 2, 0.0625, cell), reference(8, 5, 2, 0.0625, cell);
            for (PathSet* path : {&fast, &reference}) {
                path->set_exchange_classes(system.class_id, system.fermionic);
                system.fill(*path);
                path->set_permutation(ids({1, 2, 0, 4, 3}));
            }
            Action fast_action(system.groups, system.ham, 2);
            Action full_action(system.groups, system.ham, 2, true, false);
            full_action.add_component(std::make_unique<FullSliceAction>(system.ham));
            std::unique_ptr<Move> fast_move, full_move;
            if (kind == 0) {
                fast_move = std::make_unique<SingleSliceMove>(0.3);
                full_move = std::make_unique<SingleSliceMove>(0.3);
            } else if (kind == 1) {
                fast_move = std::make_unique<BisectionMove>(2);
                full_move = std::make_unique<BisectionMove>(2);
            } else {
                fast_move = std::make_unique<CentroidMove>(0.3);
                full_move = std::make_unique<CentroidMove>(0.3);
            }
            numerics::Random fast_rng(4127), full_rng(4127);
            std::size_t accepted = 0, rejected = 0;
            for (int attempt = 0; attempt < 256; ++attempt) {
                const auto saved = fast.all_positions();
                const bool accept_fast = fast_move->attempt(fast, fast_action, fast_rng);
                const bool accept_full = full_move->attempt(reference, full_action, full_rng);
                REQUIRE(accept_fast == accept_full);
                REQUIRE(fast.all_positions() == reference.all_positions());
                if (accept_fast) ++accepted;
                else {
                    ++rejected;
                    REQUIRE(fast.all_positions() == saved);
                }
                // Incremental tables agree with a fresh all-pair build.
                const auto tables = table_snapshot(fast);
                fast.rebuild_tables();
                REQUIRE(table_snapshot(fast) == tables);
            }
            CHECK(accepted > 0);
            CHECK(rejected > 0);
        }
    }
}

TEST_CASE("A fixed fermion 2-cycle in a harmonic trap is one ring of 2M beads, sign -1",
          "[pimc][exchange][physics]") {
    // d = 1, w = 1, beta = 2, tau = 0.25 (M = 8): exact sector energy.
    const double tau = 0.25, beta = 2.0;
    const std::size_t M = 8;
    const double e_exact = harmonic_two_cycle_energy(tau, M, 1.0);
    const std::vector groups{group("g", 0, 2, 0.0, 1)};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(groups, 1.0));

    double fermion_energy = 0.0;
    for (const std::string statistics : {"fermions", "bosons"}) {
        CAPTURE(statistics);
        PimcConfig cfg = sector_config(beta, tau, 8123);
        add_move(cfg, "bisection", 3, 0.0);
        add_move(cfg, "centroid", 0, 0.6);
        cfg.exchange_classes = {ExchangeClass{"g", 0, statistics, {0, 1}}};
        cfg.fixed_permutation_sector = true;
        Pimc pimc(cfg, groups, ham, SimulationCell{}, 1, 2);
        pimc.import_paths(with_permutation(pimc, ids({1, 0})));
        pimc.run();
        INFO(pimc.summary());
        INFO("E = " << pimc.energy() << " +/- " << pimc.energy_error() << " exact " << e_exact);
        CHECK(pimc.quantum_statistics());
        CHECK(std::abs(pimc.energy() - e_exact) <= tolerance(pimc.energy_error()));
        CHECK(pimc.energy_error() < e_exact / 20.0);
        // A quadratic potential splits the finite-M energy equally.
        CHECK(std::abs(pimc.potential() - e_exact / 2) <= tolerance(pimc.potential_error()));
        REQUIRE(pimc.cycle_fractions().size() == 1);
        CHECK(pimc.cycle_fractions()[0] == std::vector<double>{0.0, 1.0});
        if (statistics == "fermions") {
            CHECK(pimc.signed_run());
            CHECK(pimc.average_sign() == -1.0);
            CHECK(pimc.average_sign_error() == 0.0);
            CHECK_FALSE(pimc.sign_unreliable());
            fermion_energy = pimc.energy();
        } else {
            CHECK_FALSE(pimc.signed_run());
            CHECK(pimc.average_sign() == 1.0);
            // Same chain: the signed ratio with s = -1 is the plain mean.
            CHECK(pimc.energy() == Approx(fermion_energy).epsilon(1e-12));
        }
    }
}

TEST_CASE("A fixed 2-cycle of free particles winds like one ring at 2 beta",
          "[pimc][exchange][physics]") {
    // d = 1, L = 2, m = 1, beta = 1, tau = 1/32 (M = 32): the cycle is a ring
    // of 64 beads at 2 beta = 2.  Reference sums over momenta and windings.
    const double L = 2.0, beta = 1.0, tau = 0.03125, beta_ring = 2.0 * beta;
    double z = 0.0, e_sum = 0.0, zw = 0.0, w2_sum = 0.0;
    for (int n = -60; n <= 60; ++n) {
        const double eps = 0.5 * std::pow(2.0 * std::numbers::pi * n / L, 2);
        z += std::exp(-beta_ring * eps);
        e_sum += eps * std::exp(-beta_ring * eps);
        const double weight = std::exp(-L * L * n * n / (2.0 * beta_ring));
        zw += weight;
        w2_sum += static_cast<double>(n) * n * weight;
    }
    const double e_exact = 2.0 * e_sum / z;            // E_P = 2 E_1(2 beta)
    const double w2_exact = (w2_sum / zw) / 2.0;        // sum_c W_c^2 / (np * dim)

    const std::vector groups{group("g", 0, 2, 0.0, 1)};
    const SimulationCell cell(1, std::vector<double>{L});
    PimcConfig cfg = sector_config(beta, tau, 5521);
    cfg.accumulation_blocks = 1000;
    add_move(cfg, "bisection", 5, 0.0);
    add_move(cfg, "bisection", 3, 0.0);
    add_move(cfg, "centroid", 0, 0.5);
    cfg.exchange_classes = {ExchangeClass{"g", 0, "fermions", {0, 1}}};
    cfg.fixed_permutation_sector = true;
    Pimc pimc(cfg, groups, Hamiltonian{}, cell, 1, 2);
    pimc.import_paths(with_permutation(pimc, ids({1, 0})));
    pimc.run();
    INFO(pimc.summary());
    INFO("E = " << pimc.energy() << " +/- " << pimc.energy_error() << " exact " << e_exact);
    INFO("W2 = " << pimc.winding2() << " +/- " << pimc.winding2_error() << " exact " << w2_exact);
    CHECK(std::abs(pimc.energy() - e_exact) <= tolerance(pimc.energy_error()));
    CHECK(std::abs(pimc.winding2() - w2_exact) <= tolerance(pimc.winding2_error()));
    CHECK(pimc.winding2() > 0.05);  // windings are actually sampled
    CHECK(pimc.potential() == 0.0);
    CHECK(pimc.average_sign() == -1.0);
}

TEST_CASE("Fermions in the identity sector reproduce the boltzmannon chain exactly",
          "[pimc][exchange]") {
    const std::vector groups{group("g", 0, 2, 0.0, 1)};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(groups, 1.0));
    PimcConfig cfg = sector_config(1.0, 0.125, 77);
    cfg.accumulation_blocks = 64;
    cfg.equilibration_blocks = 16;
    add_move(cfg, "bisection", 2, 0.0);
    add_move(cfg, "centroid", 0, 0.5);
    add_move(cfg, "single_slice", 0, 0.3);
    cfg.exchange_classes = {ExchangeClass{"g", 0, "fermions", {0, 1}}};

    PimcConfig distinguishable = cfg;
    distinguishable.statistics = "boltzmannon";
    Pimc reference(distinguishable, groups, ham, SimulationCell{}, 1, 2);
    reference.run();

    cfg.fixed_permutation_sector = true;
    Pimc fermions(cfg, groups, ham, SimulationCell{}, 1, 2);
    fermions.run();

    CHECK(fermions.signed_run());
    CHECK(fermions.average_sign() == 1.0);
    REQUIRE(fermions.block_energies().size() == reference.block_energies().size());
    CHECK(std::equal(fermions.block_energies().begin(), fermions.block_energies().end(),
                     reference.block_energies().begin()));
    CHECK(fermions.energy() == Approx(reference.energy()).epsilon(1e-12));
    CHECK(fermions.energy_error() == Approx(reference.energy_error()).epsilon(1e-10));
    CHECK(fermions.cycle_fractions()[0] == std::vector<double>{1.0, 0.0});
}

TEST_CASE("The driver refuses exchange it cannot sample or measure yet",
          "[pimc][exchange]") {
    const std::vector groups{group("g", 0, 2, 0.0, 1)};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(groups, 1.0));
    PimcConfig cfg = sector_config(1.0, 0.25, 3);
    add_move(cfg, "bisection", 2, 0.0);
    cfg.exchange_classes = {ExchangeClass{"g", 0, "fermions", {0, 1}}};

    // Quantum statistics without permutation moves would sample one sector.
    CHECK_THROWS_WITH(Pimc(cfg, groups, ham, SimulationCell{}, 1, 2),
                      ContainsSubstring("boltzmannon"));

    cfg.fixed_permutation_sector = true;
    PimcConfig virial = cfg;
    virial.energy = "periodic_virial";  // the open centroid virial is supported
    CHECK_THROWS_WITH(Pimc(virial, groups, ham, SimulationCell{}, 1, 2),
                      ContainsSubstring("virial"));

    // Signed runs refuse an estimator without signed-weight support (the five
    // built-in families have it); bosonic runs accept any estimator.
    {
        Pimc fermions(cfg, groups, ham, SimulationCell{}, 1, 2);
        std::vector<std::unique_ptr<Estimator>> estimators;
        estimators.push_back(std::make_unique<NullEstimator>());
        CHECK_THROWS_WITH(fermions.set_estimators(std::move(estimators)),
                          ContainsSubstring("signed weights"));
    }
    {
        PimcConfig bosonic = cfg;
        bosonic.exchange_classes[0].statistics = "bosons";
        Pimc bosons(bosonic, groups, ham, SimulationCell{}, 1, 2);
        std::vector<std::unique_ptr<Estimator>> estimators;
        estimators.push_back(std::make_unique<NullEstimator>());
        CHECK_NOTHROW(bosons.set_estimators(std::move(estimators)));
    }

    // A distinguishable run admits only identity checkpoints.
    {
        PimcConfig distinguishable = cfg;
        distinguishable.statistics = "boltzmannon";
        Pimc boltzmannons(distinguishable, groups, ham, SimulationCell{}, 1, 2);
        CHECK_THROWS_WITH(boltzmannons.import_paths(with_permutation(boltzmannons, ids({1, 0}))),
                          ContainsSubstring("quantum statistics"));
    }

    // Classes must be mass-uniform and disjoint; a checkpoint may not cross them.
    {
        const std::vector two{group("a", 0, 1, 0.0, 1, 1.0), group("b", 1, 2, 0.0, 1, 2.0)};
        PimcConfig mixed = cfg;
        mixed.exchange_classes = {ExchangeClass{"a", 0, "fermions", {0, 1}}};
        CHECK_THROWS_WITH(Pimc(mixed, two, Hamiltonian{}, SimulationCell{}, 1, 2),
                          ContainsSubstring("mass"));
        PimcConfig overlapping = cfg;
        overlapping.exchange_classes = {ExchangeClass{"g", 0, "fermions", {0, 1}},
                                        ExchangeClass{"g", 1, "fermions", {1}}};
        CHECK_THROWS_WITH(Pimc(overlapping, groups, ham, SimulationCell{}, 1, 2),
                          ContainsSubstring("disjoint"));
    }
    {
        const std::vector four{group("e", 0, 2, 0.0, 1), group("h", 2, 4, 0.0, 1)};
        PimcConfig separate = cfg;
        separate.exchange_classes = {ExchangeClass{"e", 0, "fermions", {0, 1}},
                                     ExchangeClass{"h", 0, "fermions", {2, 3}}};
        Pimc pimc(separate, four, Hamiltonian{}, SimulationCell{}, 1, 4);
        CHECK_THROWS_AS(pimc.import_paths(with_permutation(pimc, ids({2, 1, 0, 3}))),
                        std::invalid_argument);
        CHECK_NOTHROW(pimc.import_paths(with_permutation(pimc, ids({1, 0, 3, 2}))));
    }
}

// Speed gate G2: the exchange bookkeeping with
// pi = identity — permutation-aware links, class labels, signed accumulation
// and cycle counting — must cost at most 5% over the boltzmannon run of the
// same workload.  Hidden: run explicitly on an idle machine,
//   test_pimc_exchange "[pimc-exchange-bench]" -s
// Single thread, five alternating repeats, medians compared.
TEST_CASE("G2: exchange bookkeeping with pi = identity costs at most 5 percent",
          "[.][pimc-exchange-bench]") {
    const EhSystem system(3, 3);
    PimcConfig cfg;
    cfg.threads = 1;
    cfg.samples = 2;
    cfg.beta = 2.0;
    cfg.time_step = 0.0625;  // M = 32
    cfg.equilibration_blocks = 10;
    cfg.accumulation_blocks = 1500;
    cfg.steps_per_block = 20;
    cfg.sub_steps = 6;
    cfg.seed = 1234;
    add_move(cfg, "bisection", 3, 0.0);
    add_move(cfg, "centroid", 0, 0.4);
    add_move(cfg, "single_slice", 0, 0.3);
    cfg.exchange_classes = {ExchangeClass{"electrons", 0, "fermions", {0, 1, 2}},
                            ExchangeClass{"holes", 0, "fermions", {3, 4, 5}}};
    PimcConfig boltzmannon = cfg;
    boltzmannon.statistics = "boltzmannon";
    PimcConfig fermions = cfg;
    fermions.fixed_permutation_sector = true;

    const auto seconds = [&](const PimcConfig& config, std::vector<double>& blocks) {
        Pimc pimc(config, system.groups, system.ham, SimulationCell{}, 2, 6);
        const auto start = std::chrono::steady_clock::now();
        pimc.run();
        const auto stop = std::chrono::steady_clock::now();
        blocks.assign(pimc.block_energies().begin(), pimc.block_energies().end());
        return std::chrono::duration<double>(stop - start).count();
    };
    std::vector<double> t_boltzmannon, t_fermions, blocks_b, blocks_f;
    for (int repeat = 0; repeat < 5; ++repeat) {
        if (repeat % 2 == 0) {
            t_boltzmannon.push_back(seconds(boltzmannon, blocks_b));
            t_fermions.push_back(seconds(fermions, blocks_f));
        } else {
            t_fermions.push_back(seconds(fermions, blocks_f));
            t_boltzmannon.push_back(seconds(boltzmannon, blocks_b));
        }
        REQUIRE(blocks_b == blocks_f);  // same chain, s = +1
    }
    std::sort(t_boltzmannon.begin(), t_boltzmannon.end());
    std::sort(t_fermions.begin(), t_fermions.end());
    const double ratio = t_fermions[2] / t_boltzmannon[2];
    std::cout << "G2 median seconds: boltzmannon " << t_boltzmannon[2]
              << "  fermions(pi=identity) " << t_fermions[2]
              << "  ratio " << ratio << '\n';
    CHECK(ratio <= 1.05);
}
