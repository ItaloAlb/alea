// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Grand-canonical worm PIMC (Boninsegni, Prokof'ev & Svistunov, Phys. Rev. E
// 74, 036701 (2006)).
//
// Bookkeeping first: a PathSet with spare rows keeps its active particles
// contiguous, pi a bijection and its distance tables exact through row
// insertions, removals and gaps of missing beads.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

#include <memory>
#include <numbers>
#include <string>
#include <utility>

#include <catch2/catch_approx.hpp>

#include "hamiltonian/coulomb.h"
#include "hamiltonian/hamiltonian.h"
#include "hamiltonian/harmonic_trap.h"
#include "hamiltonian/inverse_power.h"
#include "hamiltonian/moire.h"
#include "pimc/path_set.h"
#include "population/simulation_cell.h"

namespace {
using namespace alea;

ParticleGroup group(std::size_t first, std::size_t last, std::size_t dim,
                    double charge = 0.0, double mass = 1.0) {
    ParticleGroup g;
    g.name = "bosons";
    g.first = first;
    g.last = last;
    g.count = last - first;
    g.charge = charge;
    g.mass.assign(dim, mass);
    g.inv_mass.assign(dim, 1.0 / mass);
    return g;
}

// A small hexagonal cell: 3 x 3 moire cells of period a_M = 1.
SimulationCell hexagonal_cell() {
    return SimulationCell(2, std::vector<double>{3.0, 0.0, 1.5, 1.5 * std::numbers::sqrt3});
}

// moire + intra-group inverse_power p = 3 (the dipolar exciton terms) acting
// on rows [0, n): the Hamiltonian of an n-particle configuration.
Hamiltonian dipolar_moire(std::size_t n, const SimulationCell& cell) {
    Hamiltonian ham;
    std::vector<std::pair<std::size_t, std::size_t>> ranges{{0, n}};
    ham.add_component(std::make_unique<MoireTerm>(
        2, 0.3, 4.296264364086565, 0.2617993877991494,
        moire_reciprocal_vectors(1.0, 0.5 * std::numbers::pi, 2), ranges));
    ham.add_component(std::make_unique<InversePowerTerm>(
        group(0, n, 2), group(0, n, 2), 5.439295845011993, 3.0, cell));
    return ham;
}

// harmonic_trap + Coulomb on rows [0, n), open cell.
Hamiltonian trapped_coulomb(std::size_t n) {
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(
        std::vector<ParticleGroup>{group(0, n, 2, 1.0)}, 1.3, std::vector<double>{0.1, -0.2}));
    ham.add_component(std::make_unique<CoulombTerm>(group(0, n, 2, 1.0), group(0, n, 2, 1.0), 2.0));
    return ham;
}

// V of rows [0, n) at slice w, from the full evaluation.
double slice_potential(const Hamiltonian& ham, const PathSet& path, std::size_t w) {
    std::vector<double> v(path.num_slices(), 0.0);
    ham.evaluate_potential(path, v, w, w + 1);
    return v[w];
}

// Distinct finite coordinates for every row and slice.
void fill_distinct(PathSet& path) {
    for (std::size_t p = 0; p < path.capacity(); ++p)
        for (std::size_t d = 0; d < path.dim(); ++d)
            for (std::size_t k = 0; k < path.num_slices(); ++k)
                path.pos(p, d, k) = 0.37 * static_cast<double>(p) + 0.11 * static_cast<double>(d) +
                                    0.013 * static_cast<double>(k * (p + 1));
    path.rebuild_tables();
}

// Every table entry against the coordinates, recomputed from scratch.
bool tables_exact(const PathSet& path) {
    const auto& dt = path.dist_tables();
    for (std::size_t k = 0; k < path.num_slices(); ++k)
        for (std::size_t i = 0; i < path.capacity(); ++i)
            for (std::size_t j = 0; j < path.capacity(); ++j) {
                if (i == j) continue;
                double r2 = 0.0;
                for (std::size_t d = 0; d < path.dim(); ++d) {
                    const double x = path.pos(i, d, k) - path.pos(j, d, k);
                    r2 += x * x;
                }
                if (std::abs(dt(k, i, j) - std::sqrt(r2)) > 1e-12) return false;
            }
    return true;
}

void one_class(PathSet& path) {
    const std::vector<std::size_t> class_id(path.capacity(), 0);
    const std::vector<char> fermionic(path.capacity(), 0);
    path.set_exchange_classes(class_id, fermionic);
}
}  // namespace

TEST_CASE("Spare rows are fixed points outside every cycle and class",
          "[pimc][grand_canonical]") {
    PathSet path(6, 4, 2, 0.1);
    fill_distinct(path);
    one_class(path);
    path.set_active_particles(2);
    CHECK(path.capacity() == 4);
    CHECK(path.active_particles() == 2);
    CHECK(path.cycle_leaders().size() == 2);
    REQUIRE(path.exchangeable_classes().size() == 1);
    CHECK(path.exchangeable_classes()[0] == std::vector<std::size_t>{0, 1});
    CHECK(path.largest_exchange_class() == 4);  // the bound that sizes scratch

    // A permutation that moves an unused row is refused; the path is unchanged.
    CHECK_THROWS_AS(path.set_permutation(std::vector<std::size_t>{0, 1, 3, 2}),
                    std::invalid_argument);
    path.set_permutation(std::vector<std::size_t>{1, 0, 2, 3});
    CHECK(path.cycle_leaders().size() == 1);
    // Shrinking the active set may not cut a cycle.
    CHECK_THROWS_AS(path.set_active_particles(1), std::invalid_argument);
}

TEST_CASE("append_row adds a fixed point and remove_row keeps rows contiguous",
          "[pimc][grand_canonical]") {
    const std::size_t M = 5;
    PathSet path(M, 4, 2, 0.1);
    fill_distinct(path);
    one_class(path);
    path.set_active_particles(3);
    path.set_permutation(std::vector<std::size_t>{2, 1, 0, 3});  // (0 2), 1 alone

    const std::size_t q = path.append_row();
    CHECK(q == 3);
    CHECK(path.active_particles() == 4);
    CHECK(path.next_particle(3) == 3);
    CHECK(path.exchangeable_classes()[0] == std::vector<std::size_t>{0, 1, 2, 3});
    CHECK_THROWS_AS(path.append_row(), std::logic_error);  // at capacity

    // Put the last row into the cycle of 0, then delete the lone row 1: row 3
    // moves into slot 1 and the cycle (0 2 3) becomes (0 2 1).
    path.set_permutation(std::vector<std::size_t>{2, 1, 3, 0});
    std::vector<double> last_row(2 * M);
    for (std::size_t d = 0; d < 2; ++d)
        for (std::size_t k = 0; k < M; ++k) last_row[d * M + k] = path.pos(3, d, k);
    CHECK_THROWS_AS(path.remove_row(0), std::logic_error);  // not a fixed point
    path.remove_row(1);
    CHECK(path.active_particles() == 3);
    CHECK(path.next_particle(0) == 2);
    CHECK(path.next_particle(2) == 1);
    CHECK(path.next_particle(1) == 0);
    CHECK(path.next_particle(3) == 3);
    CHECK(path.prev_particle(1) == 2);
    for (std::size_t d = 0; d < 2; ++d)
        for (std::size_t k = 0; k < M; ++k) CHECK(path.pos(1, d, k) == last_row[d * M + k]);
    CHECK(path.exchangeable_classes()[0] == std::vector<std::size_t>{0, 1, 2});
    CHECK(path.cycle_leaders().size() == 1);
    CHECK(tables_exact(path));

    // Removing the last row itself, a fixed point, only shrinks the count.
    path.set_permutation(std::vector<std::size_t>{1, 0, 2, 3});
    path.remove_row(2);
    CHECK(path.active_particles() == 2);
    CHECK(path.exchangeable_classes()[0] == std::vector<std::size_t>{0, 1});
    CHECK_THROWS_AS(path.remove_row(0), std::logic_error);  // (0 1): no fixed point left
}

TEST_CASE("A gap of g beads after the head misses one bead per slice at most",
          "[pimc][grand_canonical]") {
    const std::size_t M = 8;
    PathSet path(M, 3, 1, 0.1);
    fill_distinct(path);
    one_class(path);
    path.set_permutation(std::vector<std::size_t>{1, 2, 0});  // one 3-cycle

    // The gap's beads, walked along the worldline from the head.
    const auto check_gap = [&](PathSet::Bead head, std::size_t g) {
        path.open_with_gap(head, g);
        std::vector<std::size_t> expected(M, PathSet::npos);
        PathSet::Bead bead = head;
        for (std::size_t i = 0; i < g; ++i) {
            bead = path.bead_after(bead.particle, bead.slice);
            REQUIRE(expected[bead.slice] == PathSet::npos);  // one per slice
            expected[bead.slice] = bead.particle;
        }
        bead = path.bead_after(bead.particle, bead.slice);
        CHECK(path.tail().particle == bead.particle);
        CHECK(path.tail().slice == bead.slice);
        for (std::size_t k = 0; k < M; ++k) CHECK(path.missing_row(k) == expected[k]);
        path.close();
        CHECK(path.gap() == 0);
        for (std::size_t k = 0; k < M; ++k) CHECK(path.missing_row(k) == PathSet::npos);
    };
    check_gap({0, 2}, 0);  // canonical: tail = bead_after(head)
    check_gap({0, 2}, 3);  // inside row 0
    check_gap({0, 5}, 4);  // crosses the closing link: rows 0 and pi(0) = 1
    check_gap({2, 7}, 7);  // the longest gap, starting at the closing link
    CHECK_THROWS_AS(path.open_with_gap({0, 0}, M), std::invalid_argument);
    CHECK_FALSE(path.is_open());

    // Appending or removing rows needs a closed path.
    path.open_with_gap({0, 1}, 2);
    CHECK_THROWS_AS(path.remove_row(0), std::logic_error);
    path.set_gap(5);
    CHECK(path.gap() == 5);
    path.close();
}

// ── Bead energy: e = V(R + r) - V(R) against the full evaluation ────────────

TEST_CASE("bead_energy is the potential difference of adding one particle",
          "[pimc][grand_canonical][bead_energy]") {
    const SimulationCell hex = hexagonal_cell(), open;
    struct Case {
        std::string name;
        const SimulationCell* cell;
        Hamiltonian (*build)(std::size_t, const SimulationCell&);
    };
    const auto trap = [](std::size_t n, const SimulationCell&) { return trapped_coulomb(n); };
    const std::vector<Case> cases{{"moire + inverse_power, hexagonal", &hex, dipolar_moire},
                                  {"harmonic_trap + coulomb, open", &open, +trap}};
    for (const auto& c : cases) {
        SECTION(c.name) {
            const std::size_t capacity = 4, M = 3;
            PathSet path(M, capacity, 2, 0.05, *c.cell);
            fill_distinct(path);
            path.wrap_positions();
            path.rebuild_tables();
            const Hamiltonian full = c.build(capacity, *c.cell);
            REQUIRE(full.supports_bead_energy());
            for (std::size_t w = 0; w < M; ++w)
                for (std::size_t n = 0; n < capacity; ++n) {
                    // Row n joins rows [0, n): the n-particle Hamiltonian's
                    // V (tail(n) included) against the (n+1)-particle one.
                    const double r[2] = {path.pos(n, 0, w), path.pos(n, 1, w)};
                    const double expected =
                        slice_potential(c.build(n + 1, *c.cell), path, w) -
                        (n == 0 ? 0.0 : slice_potential(c.build(n, *c.cell), path, w));
                    const double e = full.bead_energy(path, w, n, std::span<const double>(r, 2),
                                                      n, PathSet::npos);
                    CHECK(e == Catch::Approx(expected).epsilon(1e-12).margin(1e-12));
                }
        }
    }
}

TEST_CASE("bead_energy skips the missing row and cancels the tail of a moved bead",
          "[pimc][grand_canonical][bead_energy]") {
    const SimulationCell cell = hexagonal_cell();
    const std::size_t M = 2;
    PathSet path(M, 4, 2, 0.05, cell);
    fill_distinct(path);
    path.wrap_positions();
    path.rebuild_tables();
    const Hamiltonian full = dipolar_moire(4, cell);
    const std::size_t w = 1;
    const double r3[2] = {path.pos(3, 0, w), path.pos(3, 1, w)};

    // Row 3 joins {0, 2} while row 1 is missing: compare with a path whose
    // rows are reordered so that {r0, r2, r3} are contiguous.
    PathSet reordered(M, 4, 2, 0.05, cell);
    const std::size_t order[4] = {0, 2, 3, 1};
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t d = 0; d < 2; ++d)
            for (std::size_t k = 0; k < M; ++k) reordered.pos(i, d, k) = path.pos(order[i], d, k);
    reordered.rebuild_tables();
    const double expected = slice_potential(dipolar_moire(3, cell), reordered, w) -
                            slice_potential(dipolar_moire(2, cell), reordered, w);
    CHECK(full.bead_energy(path, w, 3, std::span<const double>(r3, 2), 4, 1) ==
          Catch::Approx(expected).epsilon(1e-12));

    // Moving bead (1, w) with nothing missing: e(new) - e(old) is the existing
    // one-particle delta, and the tail increments cancel.
    const double old_r[2] = {path.pos(1, 0, w), path.pos(1, 1, w)};
    const double new_r[2] = {old_r[0] + 0.41, old_r[1] - 0.27};
    std::vector<double> scratch(M);
    const double delta = full.delta_potential_particle(path, 1, w,
                                                       std::span<const double>(new_r, 2), scratch);
    const double from_beads =
        full.bead_energy(path, w, 1, std::span<const double>(new_r, 2), 4, PathSet::npos) -
        full.bead_energy(path, w, 1, std::span<const double>(old_r, 2), 4, PathSet::npos);
    CHECK(from_beads == Catch::Approx(delta).epsilon(1e-11));
}

TEST_CASE("The inverse_power tail increment is tail(n + 1) - tail(n)",
          "[pimc][grand_canonical][bead_energy]") {
    // Far-apart partners carry no information beyond the tail; checking the
    // increment against tail_energy() of n- and (n+1)-particle terms ties the
    // bead energy to the same Wigner-Seitz integral.
    const SimulationCell cell = hexagonal_cell();
    for (std::size_t n = 1; n <= 13; ++n) {
        const InversePowerTerm small(group(0, n, 2), group(0, n, 2), 5.44, 3.0, cell);
        const InversePowerTerm large(group(0, n + 1, 2), group(0, n + 1, 2), 5.44, 3.0, cell);
        const double increment = large.tail_energy() - small.tail_energy();
        // c (n + 1/2) with c = 2 tail(1): the formula the bead energy uses.
        const InversePowerTerm one(group(0, 1, 2), group(0, 1, 2), 5.44, 3.0, cell);
        CHECK(increment == Catch::Approx(2.0 * one.tail_energy() *
                                         (static_cast<double>(n) + 0.5)).epsilon(1e-13));
    }
}

// ── GrandCanonicalWormMove: bookkeeping against an independent S_V ──────────

#include "numerics/random.h"
#include "pimc/action.h"
#include "pimc/grand_canonical_worm.h"
#include "pimc/move.h"

namespace {

using HamiltonianFactory = Hamiltonian (*)(std::size_t, const SimulationCell&);

// The n-particle Hamiltonians [n], n = 0..capacity (entry 0 unused): built
// once, since a periodic inverse_power term integrates its tail at setup.
std::vector<Hamiltonian> by_count(HamiltonianFactory build, std::size_t capacity,
                                  const SimulationCell& cell) {
    std::vector<Hamiltonian> hams(capacity + 1);
    for (std::size_t n = 1; n <= capacity; ++n) hams[n] = build(n, cell);
    return hams;
}

// tau sum_k V(beads present at slice k), each slice evaluated from scratch on
// a one-slice path holding exactly its present beads: an oracle that shares
// no code with the bead energies or the gap bookkeeping.
double oracle_potential_action(const PathSet& path, const SimulationCell& cell,
                               const std::vector<Hamiltonian>& hams) {
    double action = 0.0;
    for (std::size_t k = 0; k < path.num_slices(); ++k) {
        std::vector<std::size_t> rows;
        for (std::size_t j = 0; j < path.active_particles(); ++j)
            if (path.missing_row(k) != j) rows.push_back(j);
        if (rows.empty()) continue;
        PathSet one(1, rows.size(), path.dim(), path.tau(), cell);
        for (std::size_t i = 0; i < rows.size(); ++i)
            for (std::size_t d = 0; d < path.dim(); ++d) one.pos(i, d, 0) = path.pos(rows[i], d, k);
        one.rebuild_tables();
        action += path.tau() * slice_potential(hams[rows.size()], one, 0);
    }
    return action;
}

// The open chain runs from the tail to the head over present beads only, and
// it is L_c M - g beads long (L_c rows in its permutation cycle).
void require_open_chain(const PathSet& path) {
    std::size_t cycle = 1;
    for (std::size_t q = path.next_particle(path.head().particle); q != path.head().particle;
         q = path.next_particle(q))
        ++cycle;
    PathSet::Bead b = path.tail();
    std::size_t beads = 1;
    while (!(b.particle == path.head().particle && b.slice == path.head().slice)) {
        REQUIRE(path.missing_row(b.slice) != b.particle);
        b = path.bead_after(b.particle, b.slice);
        ++beads;
        REQUIRE(beads <= cycle * path.num_slices());
    }
    REQUIRE(beads == cycle * path.num_slices() - path.gap());
}

// Tables between present beads equal the minimum-image distances.
void require_present_tables(const PathSet& path) {
    const auto& dt = path.dist_tables();
    for (std::size_t k = 0; k < path.num_slices(); ++k)
        for (std::size_t i = 0; i < path.active_particles(); ++i)
            for (std::size_t j = 0; j < path.active_particles(); ++j) {
                if (i == j || path.missing_row(k) == i || path.missing_row(k) == j) continue;
                double del[SimulationCell::kMaxDim];
                for (std::size_t d = 0; d < path.dim(); ++d)
                    del[d] = path.pos(i, d, k) - path.pos(j, d, k);
                REQUIRE(dt(k, i, j) == Catch::Approx(std::sqrt(path.cell().min_image_dist2(del)))
                                           .epsilon(1e-12).margin(1e-12));
            }
}

std::vector<ParticleGroup> one_group(std::size_t n, std::size_t dim, double charge = 0.0) {
    return {group(0, n, dim, charge)};
}

double extra(const Move& move, const std::string& name) {
    for (const auto& [key, value] : move.extra_diagnostics())
        if (key == name) return value;
    FAIL("missing diagnostic " << name);
    return 0.0;
}

}  // namespace

TEST_CASE("Grand-canonical worm moves change S_V by exactly their bead actions",
          "[pimc][grand_canonical][worm]") {
    struct Case {
        std::string name;
        SimulationCell cell;
        HamiltonianFactory build;
        GrandCanonicalSpec spec;
        double charge;
    };
    GrandCanonicalSpec periodic_spec;
    periodic_spec.chemical_potential = 8.0;
    periodic_spec.n_min = 1;
    periodic_spec.n_max = 4;
    GrandCanonicalSpec open_spec;
    open_spec.chemical_potential = 2.0;
    open_spec.n_min = 0;
    open_spec.n_max = 3;
    open_spec.insertion_width = 0.8;
    const auto trap = [](std::size_t n, const SimulationCell&) { return trapped_coulomb(n); };
    const std::vector<Case> cases{
        {"moire + inverse_power, hexagonal L = 3", hexagonal_cell(), dipolar_moire, periodic_spec, 0.0},
        {"harmonic_trap + coulomb, open", SimulationCell{}, +trap, open_spec, 1.0}};
    for (const auto& c : cases) {
        SECTION(c.name) {
            const std::size_t M = 12, dim = 2, capacity = c.spec.n_max;
            const double tau = 0.05;
            PathSet path(M, capacity, dim, tau, c.cell);
            fill_distinct(path);
            path.wrap_positions();
            path.rebuild_tables();
            one_class(path);
            path.set_active_particles(2);
            // The worm's bead actions come from the capacity Hamiltonian; the
            // closed-path bisection uses the Hamiltonian of the current N.
            const Hamiltonian full_ham = c.build(capacity, c.cell);
            Action worm_action(one_group(capacity, dim, c.charge), full_ham, dim);
            std::vector<Action> sector;
            for (std::size_t n = 1; n <= capacity; ++n)
                sector.emplace_back(one_group(n, dim, c.charge), c.build(n, c.cell), dim);
            GrandCanonicalWormMove worm(4, 3, 0.03, c.spec);
            worm.set_selection_probability(0.5);
            BisectionMove bisection(2);
            numerics::Random rng(20261119);
            const std::vector<Hamiltonian> hams = by_count(c.build, capacity, c.cell);
            double s_v = oracle_potential_action(path, c.cell, hams);
            std::size_t open_steps = 0, sizes_seen = 0;
            std::vector<char> seen(capacity + 1, 0);
            for (int attempt = 0; attempt < 20000; ++attempt) {
                bool accepted = false, worm_move = true;
                if (!path.is_open() && attempt % 2 == 1 && path.active_particles() > 0) {
                    worm_move = false;
                    accepted = bisection.attempt(path, sector[path.active_particles() - 1], rng);
                } else {
                    accepted = worm.attempt(path, worm_action, rng);
                }
                const double s_new = oracle_potential_action(path, c.cell, hams);
                if (accepted && worm_move) {
                    REQUIRE(s_new - s_v == Catch::Approx(worm.last_delta_action())
                                               .epsilon(1e-9).margin(1e-9));
                }
                if (!accepted) REQUIRE(s_new == Catch::Approx(s_v).epsilon(1e-12).margin(1e-12));
                s_v = s_new;
                require_present_tables(path);
                const std::size_t R = path.active_particles();
                REQUIRE(R <= c.spec.n_max);
                if (path.is_open()) {
                    ++open_steps;
                    REQUIRE(R >= std::max<std::size_t>(c.spec.n_min, 1));
                    require_open_chain(path);
                } else {
                    REQUIRE(R >= c.spec.n_min);
                    if (!seen[R]) ++sizes_seen;
                    seen[R] = 1;
                }
                for (std::size_t p = R; p < capacity; ++p) REQUIRE(path.next_particle(p) == p);
            }
            CHECK(open_steps > 2000);
            for (std::size_t n = 0; n <= capacity; ++n) UNSCOPED_INFO("N = " << n << " seen " << int(seen[n]));
            for (const auto& [key, value] : worm.extra_diagnostics()) UNSCOPED_INFO(key << " = " << value);
            CHECK(sizes_seen >= 3);  // the chain changes N
            for (const char* kind : {"open", "close", "insert", "remove", "advance", "recede", "swap"}) {
                CAPTURE(kind);
                CHECK(extra(worm, std::string(kind) + "_accepts") > 0);
            }
        }
    }
}

// ── Exact sector populations of ideal bosons (validation, exact at finite M) ─
//
// P(N) = e^{beta mu N} Z_N / sum_N' e^{beta mu N'} Z_N' over the window, with
// Z_N the absolute finite-M partition function from the canonical recursion
// (pimc_ideal_reference.h).  Absolute: the area or insertion density, the
// rho0 normalizations and the e^{tau mu} bead counting all enter.

#include <functional>

#include "pimc_ideal_reference.h"

namespace {

// Per-block closed-path measurements of one grand-canonical chain, [block][N].
struct SectorBlocks {
    std::vector<std::vector<double>> weight;  // closed measurements
    std::vector<std::vector<double>> energy;  // sum of the sector-N energy over them
    std::size_t transitions = 0;              // closed N -> N +- 1 changes
    std::size_t longest_open = 0;             // longest run of attempts in G
    std::string longest_state;                // R, gap, cycle at its 1000th attempt
};

// The driver rule by hand: in Z draw bisection / centroid / worm with
// weights 2 / 1 / 2 (p_sel = 0.4), in G call the worm.  Closed-path moves and
// energies use the action of the current N; the worm uses `worm_action`.
// Equilibration tunes C towards a 50 % closed fraction.
SectorBlocks run_grand_canonical(PathSet& path, std::vector<Action>& actions,
                                 Action& worm_action, GrandCanonicalWormMove& worm,
                                 int levels, std::uint64_t seed, int eq_blocks, int blocks,
                                 int attempts, int measure_energy_every) {
    worm.set_selection_probability(0.4);
    BisectionMove bisection(levels);
    CentroidMove centroid(0.5);
    numerics::Random rng(seed);
    const auto step = [&]() {
        if (path.is_open()) return worm.attempt(path, worm_action, rng);
        const double u = rng.uniform();
        const std::size_t n = path.active_particles();
        if (u < 0.6)
            return n > 0 && (u < 0.4 ? bisection.attempt(path, actions[n], rng)
                                     : centroid.attempt(path, actions[n], rng));
        return worm.attempt(path, worm_action, rng);
    };
    for (int b = 0; b < eq_blocks; ++b) {
        std::size_t closed = 0;
        for (int a = 0; a < attempts; ++a) {
            closed += !path.is_open();
            step();
        }
        const double total = attempts;
        worm.tune_constant(static_cast<double>(closed) / total,
                           1.0 - static_cast<double>(closed) / total, 1.0);
    }
    SectorBlocks out;
    const std::size_t sectors = path.capacity() + 1;
    out.weight.assign(blocks, std::vector<double>(sectors, 0.0));
    out.energy.assign(blocks, std::vector<double>(sectors, 0.0));
    std::size_t last_closed_n = path.active_particles(), open_run = 0;
    for (int b = 0; b < blocks; ++b)
        for (int a = 0; a < attempts; ++a) {
            step();
            if (path.is_open()) {
                ++open_run;
                if (open_run > out.longest_open) {
                    out.longest_open = open_run;
                    if (open_run == 1000 || open_run % 5000 == 0) {
                        std::size_t cycle = 1;
                        for (std::size_t q = path.next_particle(path.head().particle);
                             q != path.head().particle; q = path.next_particle(q)) ++cycle;
                        out.longest_state += " [run " + std::to_string(open_run) + ": R " +
                            std::to_string(path.active_particles()) + " gap " +
                            std::to_string(path.gap()) + " cycle " + std::to_string(cycle) +
                            " head (" + std::to_string(path.pos(path.head().particle, 0, path.head().slice)) +
                            ", " + std::to_string(path.pos(path.head().particle, 1, path.head().slice)) + ")]";
                    }
                }
                continue;
            }
            open_run = 0;
            const std::size_t n = path.active_particles();
            out.transitions += n != last_closed_n;
            last_closed_n = n;
            if (measure_energy_every > 0 && a % measure_energy_every != 0) continue;
            out.weight[b][n] += 1.0;
            if (measure_energy_every > 0 && n > 0)
                out.energy[b][n] += actions[n].thermodynamic_energy(path);
        }
    return out;
}

// Sector fractions P(N) = sum_b w_b(N) / sum_b W_b and their jackknife errors
// over blocks: a ratio estimator, so a block spent entirely in G (W_b = 0,
// e.g. a worm whose tail sits where the insertion density is tiny and that
// must regrow to close) carries no weight instead of an undefined fraction.
void sector_fractions(const SectorBlocks& run, std::vector<double>& mean,
                      std::vector<double>& error) {
    const std::size_t blocks = run.weight.size(), sectors = run.weight.front().size();
    std::vector<double> block_total(blocks, 0.0), sector_total(sectors, 0.0);
    double total = 0.0;
    for (std::size_t b = 0; b < blocks; ++b)
        for (std::size_t n = 0; n < sectors; ++n) {
            block_total[b] += run.weight[b][n];
            sector_total[n] += run.weight[b][n];
            total += run.weight[b][n];
        }
    mean.assign(sectors, 0.0);
    error.assign(sectors, 0.0);
    for (std::size_t n = 0; n < sectors; ++n) {
        mean[n] = sector_total[n] / total;
        double var = 0.0;
        for (std::size_t b = 0; b < blocks; ++b) {
            const double jack = (sector_total[n] - run.weight[b][n]) / (total - block_total[b]);
            var += (jack - mean[n]) * (jack - mean[n]);
        }
        error[n] = std::sqrt(var * static_cast<double>(blocks - 1) / static_cast<double>(blocks));
    }
}

// Exact P(N) over [n_min, n_max] from the absolute Z_N.
std::vector<double> exact_populations(double beta, double mu, std::size_t n_min,
                                      std::size_t n_max, std::size_t M,
                                      const std::function<pimc_reference::Ring(std::size_t)>& ring) {
    std::vector<double> p(n_max + 1, 0.0);
    double total = 0.0;
    for (std::size_t n = n_min; n <= n_max; ++n) {
        const double z = n == 0 ? 1.0 : pimc_reference::recursion(n, M, +1, ring).partition;
        p[n] = std::exp(beta * mu * static_cast<double>(n)) * z;
        total += p[n];
    }
    for (double& x : p) x /= total;
    return p;
}

// A collapsed start of n particles at distinct points, capacity rows, one class.
void start_path(PathSet& path, std::size_t n) {
    for (std::size_t p = 0; p < path.capacity(); ++p)
        for (std::size_t d = 0; d < path.dim(); ++d)
            for (std::size_t k = 0; k < path.num_slices(); ++k)
                path.pos(p, d, k) = 0.4 * static_cast<double>(p) - 0.3 + 0.2 * static_cast<double>(d);
    path.wrap_positions();
    path.rebuild_tables();
    one_class(path);
    path.set_active_particles(n);
}

void check_populations(const SectorBlocks& run, const std::vector<double>& exact) {
    std::vector<double> mean, error;
    sector_fractions(run, mean, error);
    INFO("sector transitions " << run.transitions << "; longest run in G " << run.longest_open
         << " attempts" << run.longest_state);
    CHECK(run.transitions > 200);
    double chi2 = 0.0;
    for (std::size_t n = 0; n < exact.size(); ++n) {
        INFO("N = " << n << ": " << mean[n] << " +/- " << error[n] << " exact " << exact[n]);
        CHECK(std::abs(mean[n] - exact[n]) <= 4.0 * error[n] + 2e-3);
        if (error[n] > 0.0) chi2 += std::pow((mean[n] - exact[n]) / error[n], 2);
    }
    // The fractions sum to one, so they carry exact.size() - 1 degrees of
    // freedom; a 5-sigma bound on chi^2 for that many.
    const double dof = static_cast<double>(exact.size() - 1);
    INFO("chi2 = " << chi2 << " for " << dof << " dof");
    CHECK(chi2 < dof + 5.0 * std::sqrt(2.0 * dof));
}

}  // namespace

TEST_CASE("Ideal bosons in a periodic box: sector populations are e^{beta mu N} Z_N",
          "[pimc][grand_canonical][physics]") {
    struct Point {
        double beta, mu;
        int levels, max_length, swap_length;
    };
    // tau mu = -0.1 at the first point: one e^{tau mu} miscounted per
    // N -> N + 1 transition would move P(N+1)/P(N) by 10 %, many sigma.
    for (const Point& pt : {Point{1.0, -1.0, 3, 6, 4}, Point{3.0, 0.2, 4, 8, 5}}) {
        CAPTURE(pt.beta, pt.mu);
        const double tau = 0.1, L = 4.0;
        const std::size_t M = static_cast<std::size_t>(std::lround(pt.beta / tau)), dim = 2;
        const std::size_t n_max = 4;
        const SimulationCell cell(2, std::vector<double>{L, 0.0, 0.0, L});
        PathSet path(M, n_max, dim, tau, cell);
        start_path(path, 1);
        const Hamiltonian free;
        std::vector<Action> actions;
        actions.emplace_back(one_group(n_max, dim), free, dim);  // [0] unused
        for (std::size_t n = 1; n <= n_max; ++n) actions.emplace_back(one_group(n, dim), free, dim);
        Action worm_action(one_group(n_max, dim), free, dim);
        GrandCanonicalSpec spec;
        spec.chemical_potential = pt.mu;
        spec.n_min = 0;
        spec.n_max = n_max;
        GrandCanonicalWormMove worm(pt.max_length, pt.swap_length, std::nan(""), spec);
        const SectorBlocks run = run_grand_canonical(path, actions, worm_action, worm, pt.levels,
                                                     20261119, 20, 200, 20000, 0);
        const auto ring = [&](std::size_t K) { return pimc_reference::periodic_ring(K, tau, L, dim); };
        check_populations(run, exact_populations(pt.beta, pt.mu, 0, n_max, M, ring));
    }
}

TEST_CASE("Ideal bosons in a harmonic trap: populations do not depend on the insertion width",
          "[pimc][grand_canonical][physics]") {
    const double beta = 1.0, tau = 0.1, mu = 0.5, omega = 1.0;
    const std::size_t M = 10, dim = 2, n_max = 3;
    const std::vector<ParticleGroup> groups = one_group(n_max, dim);
    const auto trap = [&](std::size_t n) {
        Hamiltonian ham;
        ham.add_component(std::make_unique<HarmonicTrap>(one_group(n, dim), omega * omega));
        return ham;
    };
    const auto ring = [&](std::size_t K) { return pimc_reference::harmonic_ring(K, tau, omega, dim); };
    const std::vector<double> exact = exact_populations(beta, mu, 0, n_max, M, ring);
    for (const double width : {0.8, 1.6}) {
        CAPTURE(width);
        PathSet path(M, n_max, dim, tau);
        start_path(path, 1);
        std::vector<Action> actions;
        actions.emplace_back(groups, trap(n_max), dim);  // [0] unused
        for (std::size_t n = 1; n <= n_max; ++n) actions.emplace_back(one_group(n, dim), trap(n), dim);
        Action worm_action(groups, trap(n_max), dim);
        GrandCanonicalSpec spec;
        spec.chemical_potential = mu;
        spec.n_min = 0;
        spec.n_max = n_max;
        spec.insertion_width = width;
        GrandCanonicalWormMove worm(6, 4, std::nan(""), spec);
        const SectorBlocks run = run_grand_canonical(path, actions, worm_action, worm, 3,
                                                     4242, 20, 200, 20000, 0);
        check_populations(run, exact);
    }
}

TEST_CASE("A one-sector window samples the canonical ensemble with gaps",
          "[pimc][grand_canonical][physics]") {
    // n_min = n_max = 3: no insert or remove is ever accepted, every other
    // sub-move runs; the energy must be the exact canonical one.
    const double tau = 1.0 / 16.0, omega = 1.0;
    const std::size_t M = 16, dim = 2, N = 3;
    Hamiltonian trap;
    trap.add_component(std::make_unique<HarmonicTrap>(one_group(N, dim), omega * omega));
    const auto ring = [&](std::size_t K) { return pimc_reference::harmonic_ring(K, tau, omega, dim); };
    const pimc_reference::Canonical exact = pimc_reference::recursion(N, M, +1, ring);
    PathSet path(M, N, dim, tau);
    start_path(path, N);
    std::vector<Action> actions;
    for (std::size_t n = 0; n <= N; ++n) actions.emplace_back(one_group(N, dim), trap, dim);
    Action worm_action(one_group(N, dim), trap, dim);
    GrandCanonicalSpec spec;
    spec.chemical_potential = 0.7;  // irrelevant at one sector, but it must cancel
    spec.n_min = N;
    spec.n_max = N;
    spec.insertion_width = 1.0;
    GrandCanonicalWormMove worm(6, 4, std::nan(""), spec);
    const SectorBlocks run = run_grand_canonical(path, actions, worm_action, worm, 3, 99,
                                                 20, 200, 20000, 10);
    CHECK(run.transitions == 0);
    // Jackknife over blocks of the ratio sum E / sum w.
    const std::size_t blocks = run.weight.size();
    double total_w = 0.0, total_e = 0.0;
    for (std::size_t b = 0; b < blocks; ++b) {
        total_w += run.weight[b][N];
        total_e += run.energy[b][N];
    }
    const double energy = total_e / total_w;
    double var = 0.0;
    for (std::size_t b = 0; b < blocks; ++b) {
        const double jack = (total_e - run.energy[b][N]) / (total_w - run.weight[b][N]);
        var += (jack - energy) * (jack - energy);
    }
    const double error = std::sqrt(var * static_cast<double>(blocks - 1) / static_cast<double>(blocks));
    INFO("E = " << energy << " +/- " << error << " exact " << exact.energy);
    CHECK(std::abs(energy - exact.energy) <= 4.0 * error + 1e-3);
    CHECK(error < 0.02 * exact.energy);
    CHECK(extra(worm, "swap_accepts") > 0);
    CHECK(extra(worm, "advance_accepts") > 0);
    CHECK(extra(worm, "insert_accepts") == 0);
}

// ── Interacting bosons: exact F(2) - F(1) at finite M (validation) ───────────
//
// Two bosons of mass 1 in a 2-D trap (omega = 1) with C / r^3, C = 1, at
// beta = 2 and M = 16.  In the primitive approximation the centre of mass
// separates at every slice and a discretized harmonic ring does not depend on
// the mass, so Z_2 / Z_1 = Z_rel, the relative path integral (mass 1/2) over
// even partial waves; tests/references/inverse_power_trap_primitive.py
// evaluates it by a radial transfer matrix on 300 Gauss-Legendre nodes
// (changes of 5e-13 in ln Z_rel on doubling them; 1.6e-13 against the closed
// forms at C = 0):
//
//   py -3 tests/references/inverse_power_trap_primitive.py --beta 2 --slices 16
//         --omega 1 --strength 1 --exponent 3
//
// It is exact at this M, like the ideal references above, so the chain must
// reproduce it within its statistical error: this is the first test in which
// the interaction enters P(N + 1)/P(N), the quantity that gives mu.
namespace {

constexpr double kDipoleDeltaF = 1.847461147235;          // F(2) - F(1)
constexpr double kDipoleNoExchangeDeltaF = 1.899507970692; // identity cycles only
constexpr double kDipoleEnergy2 = 3.442269884856;          // E(N = 2), thermodynamic

// Jackknife over blocks of f(sum_b a_b, sum_b c_b): {value, error}.
template <class F>
std::pair<double, double> jackknife(const std::vector<double>& a, const std::vector<double>& c, F f) {
    const std::size_t blocks = a.size();
    double total_a = 0.0, total_c = 0.0;
    for (std::size_t b = 0; b < blocks; ++b) {
        total_a += a[b];
        total_c += c[b];
    }
    const double value = f(total_a, total_c);
    double var = 0.0;
    for (std::size_t b = 0; b < blocks; ++b) {
        const double jack = f(total_a - a[b], total_c - c[b]);
        var += (jack - value) * (jack - value);
    }
    return {value, std::sqrt(var * static_cast<double>(blocks - 1) / static_cast<double>(blocks))};
}

std::vector<double> column(const std::vector<std::vector<double>>& blocks, std::size_t n) {
    std::vector<double> out;
    out.reserve(blocks.size());
    for (const auto& row : blocks) out.push_back(row[n]);
    return out;
}

// harmonic_trap (omega^2 = 1) + intra-group C / r^3 on rows [0, n), open cell.
Hamiltonian trapped_dipoles(std::size_t n) {
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(one_group(n, 2), 1.0));
    if (n > 1)
        ham.add_component(std::make_unique<InversePowerTerm>(group(0, n, 2), group(0, n, 2), 1.0, 3.0,
                                                             SimulationCell{}));
    return ham;
}

}  // namespace

TEST_CASE("Two dipolar bosons in a trap: a window of one and two particles measures the exact F(2) - F(1)",
          "[pimc][grand_canonical][physics]") {
    const double beta = 2.0, omega = 1.0;
    const std::size_t M = 16, dim = 2, n_max = 2;
    const double tau = beta / static_cast<double>(M);
    // mu near F(2) - F(1): both sectors carry about half of the weight.
    const double mu = 1.85;
    PathSet path(M, n_max, dim, tau);
    start_path(path, 1);
    std::vector<Action> actions;
    actions.emplace_back(one_group(n_max, dim), trapped_dipoles(n_max), dim);  // [0] unused
    for (std::size_t n = 1; n <= n_max; ++n)
        actions.emplace_back(one_group(n, dim), trapped_dipoles(n), dim);
    Action worm_action(one_group(n_max, dim), trapped_dipoles(n_max), dim);
    GrandCanonicalSpec spec;
    spec.chemical_potential = mu;
    spec.n_min = 1;
    spec.n_max = n_max;
    spec.insertion_width = 0.8;
    GrandCanonicalWormMove worm(6, 4, std::nan(""), spec);
    const SectorBlocks run = run_grand_canonical(path, actions, worm_action, worm, 3, 20261119,
                                                 20, 200, 20000, 10);
    INFO("sector transitions " << run.transitions << "; longest run in G " << run.longest_open);
    CHECK(run.transitions > 200);

    // F(2) - F(1) = mu - T ln[P(2)/P(1)] (closed-path counts of each sector).
    const auto [delta_f, delta_f_error] =
        jackknife(column(run.weight, 2), column(run.weight, 1),
                  [&](double w2, double w1) { return mu - std::log(w2 / w1) / beta; });
    const auto ring = [&](std::size_t K) { return pimc_reference::harmonic_ring(K, tau, omega, dim); };
    const double ideal_delta_f =
        -std::log(pimc_reference::recursion(2, M, +1, ring).partition /
                  pimc_reference::recursion(1, M, +1, ring).partition) / beta;
    INFO("F(2) - F(1) = " << delta_f << " +/- " << delta_f_error << " exact " << kDipoleDeltaF
         << " (ideal " << ideal_delta_f << ", identity cycles only " << kDipoleNoExchangeDeltaF << ")");
    CHECK(std::abs(delta_f - kDipoleDeltaF) <= 4.0 * delta_f_error);
    // Power: dropping the interaction, or the exchange cycles, moves the
    // reference by more than 5 sigma.
    CHECK(std::abs(kDipoleDeltaF - ideal_delta_f) > 5.0 * delta_f_error);
    CHECK(std::abs(kDipoleDeltaF - kDipoleNoExchangeDeltaF) > 5.0 * delta_f_error);

    // Each sector is canonical: E(1) is the ideal ring, E(2) the interacting pair.
    const auto ratio = [](double e, double w) { return e / w; };
    const auto [energy_1, energy_1_error] = jackknife(column(run.energy, 1), column(run.weight, 1), ratio);
    const auto [energy_2, energy_2_error] = jackknife(column(run.energy, 2), column(run.weight, 2), ratio);
    const double exact_1 = pimc_reference::recursion(1, M, +1, ring).energy;
    INFO("E(1) = " << energy_1 << " +/- " << energy_1_error << " exact " << exact_1);
    INFO("E(2) = " << energy_2 << " +/- " << energy_2_error << " exact " << kDipoleEnergy2
         << " (ideal " << pimc_reference::recursion(2, M, +1, ring).energy << ")");
    CHECK(std::abs(energy_1 - exact_1) <= 4.0 * energy_1_error);
    CHECK(std::abs(energy_2 - kDipoleEnergy2) <= 4.0 * energy_2_error);
    for (const char* kind : {"insert", "remove", "swap"}) {
        CAPTURE(kind);
        CHECK(extra(worm, std::string(kind) + "_accepts") > 0);
    }
}
