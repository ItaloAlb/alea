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
    periodic_spec.chemical_potential = {8.0};
    periodic_spec.n_min = 1;
    periodic_spec.n_max = 4;
    GrandCanonicalSpec open_spec;
    open_spec.chemical_potential = {2.0};
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

// Exact P(N) over [n_min, n_max] from the absolute Z_N, with one mu per pair
// N <-> N + 1 (pair_mu[N - n_min]): P(N) ∝ e^{beta sum_{n_min <= K < N} mu_K} Z_N.
std::vector<double> exact_populations(double beta, const std::vector<double>& pair_mu,
                                      std::size_t n_min, std::size_t n_max, std::size_t M,
                                      const std::function<pimc_reference::Ring(std::size_t)>& ring) {
    std::vector<double> p(n_max + 1, 0.0);
    double total = 0.0, exponent = 0.0;
    for (std::size_t n = n_min; n <= n_max; ++n) {
        if (n > n_min) exponent += beta * pair_mu.at(n - 1 - n_min);
        const double z = n == 0 ? 1.0 : pimc_reference::recursion(n, M, +1, ring).partition;
        p[n] = std::exp(exponent) * z;
        total += p[n];
    }
    for (double& x : p) x /= total;
    return p;
}

// The same with one mu_in for every pair: P(N) ∝ e^{beta mu_in N} Z_N.
std::vector<double> exact_populations(double beta, double mu, std::size_t n_min,
                                      std::size_t n_max, std::size_t M,
                                      const std::function<pimc_reference::Ring(std::size_t)>& ring) {
    return exact_populations(beta, std::vector<double>(n_max - n_min, mu), n_min, n_max, M, ring);
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
        spec.chemical_potential = {pt.mu};
        spec.n_min = 0;
        spec.n_max = n_max;
        GrandCanonicalWormMove worm(pt.max_length, pt.swap_length, std::nan(""), spec);
        const SectorBlocks run = run_grand_canonical(path, actions, worm_action, worm, pt.levels,
                                                     20261119, 20, 200, 20000, 0);
        const auto ring = [&](std::size_t K) { return pimc_reference::periodic_ring(K, tau, L, dim); };
        check_populations(run, exact_populations(pt.beta, pt.mu, 0, n_max, M, ring));
    }
}

TEST_CASE("The worm takes one mu or one per pair, and row R carries the mu of the pair R - 1 <-> R",
          "[pimc][grand_canonical]") {
    GrandCanonicalSpec spec;
    spec.n_min = 2;
    spec.n_max = 5;
    spec.chemical_potential = {1.0, 2.0, 3.0};
    const GrandCanonicalWormMove pairs(4, 3, 0.1, spec);
    CHECK(pairs.row_chemical_potential(2) == 1.0);  // G at R = n_min: the first pair's
    CHECK(pairs.row_chemical_potential(3) == 1.0);  // 2 <-> 3
    CHECK(pairs.row_chemical_potential(4) == 2.0);  // 3 <-> 4
    CHECK(pairs.row_chemical_potential(5) == 3.0);  // 4 <-> 5
    spec.chemical_potential = {7.0};
    const GrandCanonicalWormMove one(4, 3, 0.1, spec);
    for (std::size_t rows = 2; rows <= 5; ++rows) CHECK(one.row_chemical_potential(rows) == 7.0);
    for (const std::vector<double>& bad :
         {std::vector<double>{}, std::vector<double>{1.0, 2.0},
          std::vector<double>{1.0, std::nan(""), 3.0}}) {
        spec.chemical_potential = bad;
        CAPTURE(bad);
        CHECK_THROWS_AS(GrandCanonicalWormMove(4, 3, 0.1, spec), std::invalid_argument);
    }
}

// One mu per pair (the multicanonical weight of N): with mu_K = F(K + 1) -
// F(K) + delta_K, from the exact Z_N, P(N) ∝ e^{beta sum_{K < N} delta_K}.
// Offsets of a few tenths at beta = 3 spread the populations by factors of
// e^{beta delta} ~ 2 to 3 in both directions, so a sub-move that took the mu
// of a neighbouring pair (or one mu for all) would miss them by many sigma,
// while every sector is still visited.
TEST_CASE("Ideal bosons in a periodic box with one mu per pair: populations are "
          "e^{beta sum_K mu_K} Z_N",
          "[pimc][grand_canonical][physics]") {
    const double beta = 3.0, tau = 0.1, L = 4.0;
    const std::size_t M = 30, dim = 2, n_max = 4;
    const SimulationCell cell(2, std::vector<double>{L, 0.0, 0.0, L});
    const auto ring = [&](std::size_t K) { return pimc_reference::periodic_ring(K, tau, L, dim); };
    const std::vector<double> delta{0.3, -0.25, 0.2, -0.35};
    std::vector<double> pair_mu;
    double z_lower = 1.0;  // Z_0
    for (std::size_t n = 0; n < n_max; ++n) {
        const double z_upper = pimc_reference::recursion(n + 1, M, +1, ring).partition;
        pair_mu.push_back(-std::log(z_upper / z_lower) / beta + delta[n]);
        z_lower = z_upper;
    }
    CAPTURE(pair_mu);
    PathSet path(M, n_max, dim, tau, cell);
    start_path(path, 1);
    const Hamiltonian free;
    std::vector<Action> actions;
    actions.emplace_back(one_group(n_max, dim), free, dim);  // [0] unused
    for (std::size_t n = 1; n <= n_max; ++n) actions.emplace_back(one_group(n, dim), free, dim);
    Action worm_action(one_group(n_max, dim), free, dim);
    GrandCanonicalSpec spec;
    spec.chemical_potential = pair_mu;
    spec.n_min = 0;
    spec.n_max = n_max;
    GrandCanonicalWormMove worm(8, 5, std::nan(""), spec);
    const SectorBlocks run = run_grand_canonical(path, actions, worm_action, worm, 4, 20261120,
                                                 20, 200, 20000, 0);
    const std::vector<double> exact = exact_populations(beta, pair_mu, 0, n_max, M, ring);
    // Power: the populations of one mu_in (their mean) differ from these.
    double mean_mu = 0.0;
    for (const double m : pair_mu) mean_mu += m / static_cast<double>(n_max);
    const std::vector<double> one_mu = exact_populations(beta, mean_mu, 0, n_max, M, ring);
    for (std::size_t n = 0; n <= n_max; ++n) {
        CAPTURE(n, exact[n], one_mu[n]);
        CHECK(exact[n] > 0.05);
    }
    CHECK(std::abs(exact[1] - one_mu[1]) > 0.1);
    check_populations(run, exact);
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
        spec.chemical_potential = {mu};
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
    spec.chemical_potential = {0.7};  // irrelevant at one sector, but it must cancel
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
    spec.chemical_potential = {mu};
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

// ── Detailed balance, every proposal probability written out (request test 4) ─
//
// W(x) K(x -> y) = W(y) K(y -> x) with K = T min(1, r) holds when (a) the
// ratio r of every proposal is W(y) T(y -> x) / (W(x) T(x -> y)) and (b) the
// code proposes with the T that ratio assumes.  The two tests below check
// both without the acceptance formulas of grand_canonical_worm.cpp:
//  (a) along chains with interactions, ln r of every accepted sub-move
//      (last_log_acceptance) against ln W(y) T(y -> x) - ln W(x) T(x -> y),
//      W the full weight (rho0 of every present link, the oracle S_V,
//      e^{beta mu N} or C e^{tau mu n_b}) and T the explicit proposal density
//      (selection, uniform draws, insertion density, and the free walk or
//      bridge density written link by link), so the cancellations the move
//      relies on happen here numerically;
//  (b) from fixed configurations of free particles, whose acceptances do not
//      depend on the drawn coordinates, the frequency of every discrete
//      outcome against its enumerated probability (chi^2), the moments of the
//      drawn walks, bridges and tails, and the rejections at the window.
// W is the weight of unlabeled configurations: the 1/N! of Z_N cancels the
// N! labelings of the rows, and every sub-move acts alike on every labeling
// (insert appends the last row, remove refills the slot with the last row, a
// swap may relabel one side of a slice), so the tests compare configurations
// by their beads, not by their row labels.

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <numeric>
#include <tuple>

namespace {

using Bead = PathSet::Bead;
using Worm = GrandCanonicalWormMove;
constexpr double kNoWay = -std::numeric_limits<double>::infinity();

// What the weights and proposal densities of one grand-canonical chain need
// (mass 1, d = 2 throughout).
struct Rules {
    double constant = 0.0;      // C
    std::vector<double> mu;     // mu_in: one value, or one per pair N <-> N + 1 from n_min
    std::size_t n_min = 0, n_max = 0;
    std::size_t max_length = 0;   // L
    std::size_t swap_length = 0;  // m_s
    double p_select = 1.0;        // the driver's probability of the worm in Z
    double area = 0.0;            // periodic cell: uniform tail density 1 / area
    double width = 0.0;           // open cell: Gaussian tail density
    std::array<double, 2> center{};
};

double log_insertion(const Rules& rules, const double* r) {
    if (rules.area > 0.0) return -std::log(rules.area);
    double log_q = 0.0;
    for (std::size_t d = 0; d < 2; ++d) {
        const double x = (r[d] - rules.center[d]) / rules.width;
        log_q += -0.5 * std::log(2.0 * std::numbers::pi * rules.width * rules.width) - 0.5 * x * x;
    }
    return log_q;
}

bool same_bead(Bead a, Bead b) { return a.particle == b.particle && a.slice == b.slice; }

Bead walk_forward(const PathSet& path, Bead from, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) from = path.bead_after(from.particle, from.slice);
    return from;
}

// [p * M + k]: every bead of the active rows except, in an open path, the g
// beads that follow the head along its worldline.
std::vector<char> present_beads(const PathSet& path) {
    const std::size_t M = path.num_slices();
    std::vector<char> mask(path.capacity() * M, 0);
    std::fill(mask.begin(), mask.begin() + static_cast<std::ptrdiff_t>(path.active_particles() * M), 1);
    if (path.is_open()) {
        Bead b = path.head();
        for (std::size_t i = 0; i < path.gap(); ++i) {
            b = path.bead_after(b.particle, b.slice);
            mask[b.particle * M + b.slice] = 0;
        }
    }
    return mask;
}

bool present(const PathSet& path, const std::vector<char>& mask, Bead b) {
    return mask[b.particle * path.num_slices() + b.slice] != 0;
}

void coordinates(const PathSet& path, Bead b, double* r) {
    for (std::size_t d = 0; d < path.dim(); ++d) r[d] = path.pos(b.particle, d, b.slice);
}

// Minimum-image separation of bead b from bead a.
void separation(const PathSet& path, Bead a, Bead b, double* sep) {
    for (std::size_t d = 0; d < path.dim(); ++d)
        sep[d] = path.pos(b.particle, d, b.slice) - path.pos(a.particle, d, a.slice);
    if (path.cell().is_periodic()) path.cell().min_image(sep);
}

// ln rho0(b - a; n tau), the normalized free propagator of mass 1.
double log_rho0(const PathSet& path, Bead a, Bead b, std::size_t n) {
    double sep[SimulationCell::kMaxDim];
    separation(path, a, b, sep);
    const double t = static_cast<double>(n) * path.tau();
    double value = 0.0;
    for (std::size_t d = 0; d < path.dim(); ++d)
        value += -0.5 * std::log(2.0 * std::numbers::pi * t) - sep[d] * sep[d] / (2.0 * t);
    return value;
}

// Density of the n beads of a free walk of n links from `from`: the product
// of the links' rho0.
double log_walk(const PathSet& path, Bead from, std::size_t n) {
    double value = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const Bead next = path.bead_after(from.particle, from.slice);
        value += log_rho0(path, from, next, 1);
        from = next;
    }
    return value;
}

// Density of the n - 1 interior beads of a free bridge over n links from
// `from` to the bead n links after it: prod rho0(link) / rho0(end - from; n).
double log_bridge(const PathSet& path, Bead from, std::size_t n) {
    return log_walk(path, from, n) - log_rho0(path, from, walk_forward(path, from, n), n);
}

// The mu of the beads of row R = 1, 2, ...: the pair R - 1 <-> R, entry
// R - 1 - n_min of a per-pair list; rows at or below n_min (G configurations
// with n_min rows) take the first entry.
double row_mu(const Rules& rules, std::size_t row) {
    if (rules.mu.size() == 1 || row <= rules.n_min + 1) return rules.mu.front();
    return rules.mu.at(row - 1 - rules.n_min);
}

// ln W (BPS Sec. III at discrete tau, unlabeled configurations):
//   Z: e^{Phi(N M)} prod_links rho0 e^{-S_V},
//   G: C e^{Phi(n_b)} prod_{present links} rho0 e^{-S_V},
// with Phi(n) = tau sum_{j = 1..n} mu(row of bead j), bead j counted in row
// ceil(j / M): written bead by bead, not as the per-move tau mu_R dn_b the
// worm uses, and e^{beta mu_in N}, e^{tau mu_in n_b} for one mu.  A link is
// present when both its beads are and it does not leave the head (the cut
// link of a gap-free worm).
double log_weight(const PathSet& path, const Rules& rules, const SimulationCell& cell,
                  const std::vector<Hamiltonian>& hams) {
    const std::vector<char> mask = present_beads(path);
    const std::size_t M = path.num_slices(), R = path.active_particles();
    double value = -oracle_potential_action(path, cell, hams);
    std::size_t beads = 0;
    for (std::size_t p = 0; p < R; ++p)
        for (std::size_t k = 0; k < M; ++k) {
            const Bead a{p, k};
            if (!present(path, mask, a)) continue;
            ++beads;
            const Bead b = path.bead_after(p, k);
            if ((path.is_open() && same_bead(a, path.head())) || !present(path, mask, b)) continue;
            value += log_rho0(path, a, b, 1);
        }
    double phi = 0.0;
    for (std::size_t j = 1; j <= beads; ++j) phi += path.tau() * row_mu(rules, (j + M - 1) / M);
    if (!path.is_open()) return value + phi;
    return value + std::log(rules.constant) + phi;
}

// The swap's heat-bath weight of target bead `to` from `from`: rho0 over m_s
// links up to its normalization, pruned beyond e^{-kMinusLogEpsilon}.
double heat_weight(const PathSet& path, Bead from, Bead to, std::size_t ms) {
    double sep[SimulationCell::kMaxDim];
    separation(path, from, to, sep);
    double exponent = 0.0;
    for (std::size_t d = 0; d < path.dim(); ++d) exponent += sep[d] * sep[d];
    exponent /= 2.0 * static_cast<double>(ms) * path.tau();
    return exponent > CycleTable::kMinusLogEpsilon ? 0.0 : std::exp(-exponent);
}

// Sum of the heat-bath weights over the present beads of `slice`.
double heat_sum(const PathSet& path, const std::vector<char>& mask, Bead from, std::size_t slice,
                std::size_t ms) {
    double sum = 0.0;
    for (std::size_t j = 0; j < path.active_particles(); ++j)
        if (present(path, mask, {j, slice})) sum += heat_weight(path, from, {j, slice}, ms);
    return sum;
}

// A swap from `head` to target `alpha` exists when the m_s - 1 beads before
// alpha and the bead zeta before them are present, zeta is not the head, and
// both heat baths (from the head and, for the reverse, from zeta) can pick
// alpha.  Returns zeta through `zeta`.
bool swap_exists(const PathSet& path, const std::vector<char>& mask, Bead head, Bead alpha,
                 std::size_t ms, Bead& zeta) {
    Bead b = alpha;
    for (std::size_t i = 0; i < ms; ++i) {
        b = path.bead_before(b.particle, b.slice);
        if (!present(path, mask, b)) return false;
    }
    if (same_bead(b, head)) return false;
    if (!(heat_weight(path, head, alpha, ms) > 0.0 && heat_weight(path, b, alpha, ms) > 0.0))
        return false;
    zeta = b;
    return true;
}

// ln T of the swap from head h to alpha in `from`: kSwap, the heat-bath
// probability of alpha, and the free bridge that `to` holds from the bead
// `h_in_to` (h's coordinates in the labels of `to`) over m_s links.
double log_swap(const PathSet& from, Bead h, Bead alpha, const PathSet& to, Bead h_in_to,
                std::size_t ms) {
    const std::vector<char> mask = present_beads(from);
    Bead zeta{};
    if (!swap_exists(from, mask, h, alpha, ms, zeta)) return kNoWay;
    return std::log(Worm::kSwap) + std::log(heat_weight(from, h, alpha, ms)) -
           std::log(heat_sum(from, mask, h, alpha.slice, ms)) + log_bridge(to, h_in_to, ms);
}

enum Kind { kOpenKind, kCloseKind, kInsertKind, kRemoveKind, kAdvanceKind, kRecedeKind,
            kSwapKind, kKinds };
constexpr const char* kKindNames[kKinds] = {"open", "close", "insert", "remove",
                                            "advance", "recede", "swap"};

// An accepted step x -> y: its sub-move, told apart by the sector, the row
// count and the gap, and the discrete choices it made, in x's labels.
struct Step {
    int kind = kKinds;
    std::size_t m = 0;            // advance / recede length, insert / remove links
    Bead head{}, zeta{}, alpha{}; // swap, in x: the old head, the new head, the target
    Bead head_in_y{};             // swap: the old head's bead in y's labels
    std::string key;              // the discrete outcome
};

// The present bead of `slice` with exactly the coordinates r.
Bead find_bead(const PathSet& path, const std::vector<char>& mask, std::size_t slice,
               const double* r) {
    std::size_t found = path.capacity(), matches = 0;
    for (std::size_t p = 0; p < path.active_particles(); ++p) {
        if (!present(path, mask, {p, slice})) continue;
        bool equal = true;
        for (std::size_t d = 0; d < path.dim(); ++d) equal = equal && path.pos(p, d, slice) == r[d];
        if (equal) {
            found = p;
            ++matches;
        }
    }
    REQUIRE(matches == 1);
    return {found, slice};
}

std::string bead_key(Bead b) {
    return std::to_string(b.particle) + " " + std::to_string(b.slice);
}

Step classify(const PathSet& x, const PathSet& y, std::size_t ms) {
    const std::size_t M = x.num_slices(), rows_x = x.active_particles(), rows_y = y.active_particles();
    Step s;
    if (!x.is_open()) {
        REQUIRE(y.is_open());
        if (rows_y == rows_x) {
            s.kind = kOpenKind;
            s.key = "open " + bead_key(y.head()) + " g " + std::to_string(y.gap());
        } else {
            REQUIRE(rows_y == rows_x + 1);
            s.kind = kInsertKind;
            s.m = M - 1 - y.gap();
            s.key = "insert " + std::to_string(y.tail().slice) + " m " + std::to_string(s.m);
        }
    } else if (!y.is_open()) {
        if (rows_y == rows_x) {
            s.kind = kCloseKind;
            s.key = "close";
        } else {
            REQUIRE(rows_y + 1 == rows_x);
            s.kind = kRemoveKind;
            s.m = M - 1 - x.gap();
            s.key = "remove";
        }
    } else {
        REQUIRE(rows_y == rows_x);
        if (y.gap() != x.gap()) {
            s.kind = y.gap() < x.gap() ? kAdvanceKind : kRecedeKind;
            s.m = y.gap() < x.gap() ? x.gap() - y.gap() : y.gap() - x.gap();
            s.key = std::string(kKindNames[s.kind]) + " " + std::to_string(s.m);
        } else {
            s.kind = kSwapKind;
            const std::vector<char> mask_x = present_beads(x), mask_y = present_beads(y);
            double r[SimulationCell::kMaxDim], r_alpha[SimulationCell::kMaxDim];
            s.head = x.head();
            coordinates(y, y.head(), r);
            s.zeta = find_bead(x, mask_x, s.head.slice, r);
            s.alpha = walk_forward(x, s.zeta, ms);
            coordinates(x, s.head, r);
            s.head_in_y = find_bead(y, mask_y, s.head.slice, r);
            // y's new segment leaves the old head and ends at alpha.
            coordinates(x, s.alpha, r_alpha);
            coordinates(y, walk_forward(y, s.head_in_y, ms), r);
            for (std::size_t d = 0; d < x.dim(); ++d) REQUIRE(r[d] == r_alpha[d]);
            s.key = "swap " + bead_key(s.alpha);
        }
    }
    return s;
}

// ln W(y) T(y -> x) - ln W(x) T(x -> y) of the step x -> y, every T written
// out from the proposal of its sub-move (header of grand_canonical_worm.h);
// kNoWay when the step or its reverse cannot be proposed.
double explicit_log_ratio(const PathSet& x, const PathSet& y, const Step& s, const Rules& rules,
                          const SimulationCell& cell, const std::vector<Hamiltonian>& hams) {
    const std::size_t M = x.num_slices(), L = rules.max_length;
    const double log_l = std::log(static_cast<double>(L));
    // Z -> G by open (the worm selected, then open: a uniform bead of the R M
    // and a uniform gap in [0, L]); G -> Z by close (kClose, then a free
    // bridge over the g + 1 links).  {T(open), T(close)}.
    const auto open_close = [&](const PathSet& closed, const PathSet& open) {
        const std::size_t g = open.gap(), R = closed.active_particles();
        if (g > L || R != open.active_particles()) return std::pair{kNoWay, kNoWay};
        return std::pair{std::log(rules.p_select * Worm::kOpenZ) -
                             std::log(static_cast<double>(R * M)) - std::log(static_cast<double>(L + 1)),
                         std::log(Worm::kClose) + log_bridge(closed, open.head(), g + 1)};
    };
    // Z(N) -> G(N + 1) by insert (the worm selected, then insert: a uniform
    // slice, a uniform length m in [1, L], the tail from the insertion
    // density, a free walk of m links); back by remove (kRemove; the worm is
    // a row of its own).  {T(insert), T(remove)}.
    const auto insert_remove = [&](const PathSet& closed, const PathSet& open) {
        const std::size_t N = closed.active_particles(), m = M - 1 - open.gap();
        const bool own_row = open.next_particle(open.head().particle) == open.head().particle;
        if (N + 1 != open.active_particles() || N >= rules.n_max || N < rules.n_min || m < 1 ||
            m > L || !own_row)
            return std::pair{kNoWay, kNoWay};
        double r_tail[SimulationCell::kMaxDim];
        coordinates(open, open.tail(), r_tail);
        return std::pair{std::log(rules.p_select * (1.0 - Worm::kOpenZ)) -
                             std::log(static_cast<double>(M)) - log_l + log_insertion(rules, r_tail) +
                             log_walk(open, open.tail(), m),
                         std::log(Worm::kRemove)};
    };
    // G(g) -> G(g - m) by advance (kAdvance, a uniform m in [1, L] with
    // m <= g, a free walk of m links from the head); back by recede (kRecede,
    // a uniform m with g - m + m <= M - 1).  {T(advance), T(recede)}.
    const auto advance_recede = [&](const PathSet& long_gap, const PathSet& short_gap) {
        const std::size_t m = long_gap.gap() - short_gap.gap();
        if (m < 1 || m > L || long_gap.gap() > M - 1) return std::pair{kNoWay, kNoWay};
        return std::pair{std::log(Worm::kAdvance) - log_l + log_walk(short_gap, long_gap.head(), m),
                         std::log(Worm::kRecede) - log_l};
    };
    double forward = kNoWay, reverse = kNoWay;
    switch (s.kind) {
    case kOpenKind: std::tie(forward, reverse) = open_close(x, y); break;
    case kCloseKind: std::tie(reverse, forward) = open_close(y, x); break;
    case kInsertKind: std::tie(forward, reverse) = insert_remove(x, y); break;
    case kRemoveKind: std::tie(reverse, forward) = insert_remove(y, x); break;
    case kAdvanceKind: std::tie(forward, reverse) = advance_recede(x, y); break;
    case kRecedeKind: std::tie(reverse, forward) = advance_recede(y, x); break;
    case kSwapKind: {
        const std::size_t ms = rules.swap_length;
        forward = log_swap(x, s.head, s.alpha, y, s.head_in_y, ms);
        reverse = log_swap(y, y.head(), walk_forward(y, s.head_in_y, ms), x, s.zeta, ms);
        break;
    }
    default: FAIL("unclassified step");
    }
    if (!(forward > kNoWay && reverse > kNoWay)) return kNoWay;
    return log_weight(y, rules, cell, hams) + reverse - log_weight(x, rules, cell, hams) - forward;
}

// A copy of a path's state (PathSet itself is not copyable).
struct Snapshot {
    std::vector<double> positions;
    std::vector<std::size_t> permutation;
    std::size_t active = 0, gap = 0;
    bool open = false;
    Bead head{};
};

Snapshot snapshot(const PathSet& path) {
    const auto pi = path.permutation();
    return {path.all_positions(), {pi.begin(), pi.end()}, path.active_particles(),
            path.gap(), path.is_open(), path.head()};
}

void restore(PathSet& path, const Snapshot& s) {
    path.close();
    std::vector<std::size_t> identity(path.capacity());
    std::iota(identity.begin(), identity.end(), std::size_t{0});
    path.set_permutation(identity);
    path.set_active_particles(s.active);
    path.set_permutation(s.permutation);
    std::copy(s.positions.begin(), s.positions.end(), path.positions().begin());
    path.rebuild_tables();
    if (s.open) path.open_with_gap(s.head, s.gap);
}

}  // namespace

TEST_CASE("Every accepted grand-canonical sub-move has the Metropolis ratio of its weights "
          "and proposal densities written out",
          "[pimc][grand_canonical][worm][balance]") {
    struct Case {
        std::string name;
        SimulationCell cell;
        HamiltonianFactory build;
        GrandCanonicalSpec spec;
        double charge;
        double area;  // the cell's, from its lattice vectors: 3 x (3 sqrt(3) / 2)
    };
    // The periodic case has one mu per pair (1 <-> 2, 2 <-> 3, 3 <-> 4), 1.5
    // apart: a sub-move that took the mu of a neighbouring row would be off
    // by tau 1.5 per bead, ~0.08 in ln r per bead; the open case has one mu.
    GrandCanonicalSpec periodic_spec;
    periodic_spec.chemical_potential = {6.5, 8.0, 9.5};
    periodic_spec.n_min = 1;
    periodic_spec.n_max = 4;
    GrandCanonicalSpec open_spec;
    open_spec.chemical_potential = {2.0};
    open_spec.n_min = 0;
    open_spec.n_max = 3;
    open_spec.insertion_width = 0.8;
    const auto trap = [](std::size_t n, const SimulationCell&) { return trapped_coulomb(n); };
    const std::vector<Case> cases{
        {"moire + inverse_power, hexagonal L = 3, one mu per pair", hexagonal_cell(),
         dipolar_moire, periodic_spec, 0.0, 4.5 * std::numbers::sqrt3},
        {"harmonic_trap + coulomb, open, one mu", SimulationCell{}, +trap, open_spec, 1.0, 0.0}};
    for (const auto& c : cases) {
        SECTION(c.name) {
            const std::size_t M = 12, dim = 2, capacity = c.spec.n_max, L = 4, ms = 3;
            const double tau = 0.05, constant = 0.03, p_select = 0.5;
            PathSet path(M, capacity, dim, tau, c.cell), before(M, capacity, dim, tau, c.cell);
            fill_distinct(path);
            path.wrap_positions();
            path.rebuild_tables();
            one_class(path);
            one_class(before);
            path.set_active_particles(2);
            Action worm_action(one_group(capacity, dim, c.charge), c.build(capacity, c.cell), dim);
            std::vector<Action> sector;
            for (std::size_t n = 1; n <= capacity; ++n)
                sector.emplace_back(one_group(n, dim, c.charge), c.build(n, c.cell), dim);
            GrandCanonicalWormMove worm(static_cast<int>(L), static_cast<int>(ms), constant, c.spec);
            worm.set_selection_probability(p_select);
            BisectionMove bisection(2);
            numerics::Random rng(20261120);
            const std::vector<Hamiltonian> hams = by_count(c.build, capacity, c.cell);
            Rules rules;
            rules.constant = constant;
            rules.mu = c.spec.chemical_potential;
            rules.n_min = c.spec.n_min;
            rules.n_max = c.spec.n_max;
            rules.max_length = L;
            rules.swap_length = ms;
            rules.p_select = p_select;
            rules.area = c.area;
            rules.width = c.spec.insertion_width;

            std::array<std::size_t, kKinds> checked{};
            std::map<std::size_t, std::size_t> rows_checked;  // R of the step's G configuration
            double worst = 0.0;
            for (int attempt = 0; attempt < 20000; ++attempt) {
                if (!path.is_open() && attempt % 2 == 1 && path.active_particles() > 0) {
                    bisection.attempt(path, sector[path.active_particles() - 1], rng);
                    continue;
                }
                const Snapshot x = snapshot(path);
                if (!worm.attempt(path, worm_action, rng)) continue;
                restore(before, x);
                const Step step = classify(before, path, ms);
                const double expected = explicit_log_ratio(before, path, step, rules, c.cell, hams);
                CAPTURE(attempt, kKindNames[step.kind], step.key);
                REQUIRE(std::isfinite(expected));
                REQUIRE(worm.last_log_acceptance() ==
                        Catch::Approx(expected).epsilon(1e-10).margin(1e-9));
                worst = std::max(worst, std::abs(worm.last_log_acceptance() - expected));
                ++checked[step.kind];
                ++rows_checked[path.is_open() ? path.active_particles() : before.active_particles()];
            }
            for (const auto& [key, value] : worm.extra_diagnostics()) UNSCOPED_INFO(key << " = " << value);
            UNSCOPED_INFO("largest |ln r - explicit| = " << worst);
            for (int k = 0; k < kKinds; ++k) {
                CAPTURE(kKindNames[k], checked[k]);
                CHECK(checked[k] >= 20);
            }
            // Steps in G with every row count above the window's lowest, so
            // that each pair's mu enters some checked ratio.
            for (std::size_t rows = std::max<std::size_t>(c.spec.n_min, 1) + 1; rows <= c.spec.n_max;
                 ++rows) {
                CAPTURE(rows, rows_checked[rows]);
                CHECK(rows_checked[rows] >= 20);
            }
        }
    }
}

namespace {

// Rows of three particles near distinct centres on small distinct circles:
// compact rows (every link far inside the minimum image of a box of side 4)
// a heat-bath distance apart, and no two beads of a slice alike.
void fixed_rings(PathSet& path) {
    const double centre[3][2] = {{0.5, 0.5}, {1.3, 0.8}, {0.9, 1.6}};
    const double M = static_cast<double>(path.num_slices());
    for (std::size_t p = 0; p < path.capacity(); ++p)
        for (std::size_t k = 0; k < path.num_slices(); ++k) {
            const double phase = 2.0 * std::numbers::pi * static_cast<double>(k) / M + 0.7 * static_cast<double>(p);
            const double radius = 0.12 + 0.03 * static_cast<double>(p);
            path.pos(p, 0, k) = centre[p][0] + radius * std::cos(phase) + 0.01 * static_cast<double>(k);
            path.pos(p, 1, k) = centre[p][1] + radius * std::sin(phase) - 0.007 * static_cast<double>(k);
        }
    path.wrap_positions();
    path.rebuild_tables();
}

// Every discrete outcome of one attempt from x with its probability, for free
// particles and mu = 0: from Z (C large) every open and insert proposed is
// accepted; from G (C small) every close, remove, advance and recede, and a
// swap to alpha with min(1, Sigma_h / Sigma_zeta) (the heat-bath sums from the
// head and from zeta).  The rest is the probability that nothing changes.
std::map<std::string, double> enumerate_outcomes(const PathSet& x, const Rules& rules) {
    std::map<std::string, double> p;
    const std::size_t M = x.num_slices(), R = x.active_particles(), L = rules.max_length,
                      ms = rules.swap_length;
    const double l = static_cast<double>(L);
    if (!x.is_open()) {
        for (std::size_t q = 0; q < R; ++q)
            for (std::size_t k = 0; k < M; ++k)
                for (std::size_t g = 0; g <= L; ++g)
                    p["open " + bead_key({q, k}) + " g " + std::to_string(g)] =
                        Worm::kOpenZ / (static_cast<double>(R * M) * (l + 1.0));
        if (R < rules.n_max)
            for (std::size_t k = 0; k < M; ++k)
                for (std::size_t m = 1; m <= L; ++m)
                    p["insert " + std::to_string(k) + " m " + std::to_string(m)] =
                        (1.0 - Worm::kOpenZ) / (static_cast<double>(M) * l);
        return p;
    }
    const std::size_t g = x.gap();
    for (std::size_t m = 1; m <= L; ++m) {
        if (m <= g) p["advance " + std::to_string(m)] = Worm::kAdvance / l;  // not past the tail
        if (g + m <= M - 1) p["recede " + std::to_string(m)] = Worm::kRecede / l;  // not a full row
    }
    if (g <= L) p["close"] = Worm::kClose;
    const Bead h = x.head();
    const std::size_t worm_links = M - 1 - g;
    if (x.next_particle(h.particle) == h.particle && worm_links >= 1 && worm_links <= L &&
        R - 1 >= rules.n_min)
        p["remove"] = Worm::kRemove;
    const std::vector<char> mask = present_beads(x);
    const std::size_t target = (h.slice + ms) % M;
    const double sigma_head = heat_sum(x, mask, h, target, ms);
    for (std::size_t j = 0; j < R; ++j) {
        const Bead alpha{j, target};
        Bead zeta{};
        if (!present(x, mask, alpha) || !swap_exists(x, mask, h, alpha, ms, zeta)) continue;
        const double sigma_zeta = heat_sum(x, mask, zeta, target, ms);
        p["swap " + bead_key(alpha)] = Worm::kSwap * heat_weight(x, h, alpha, ms) / sigma_head *
                                       std::min(1.0, sigma_head / sigma_zeta);
    }
    return p;
}

// Running moments of standardized samples.
struct Moments {
    double n = 0.0, sum = 0.0, sum2 = 0.0;
    void add(double z) {
        n += 1.0;
        sum += z;
        sum2 += z * z;
    }
};

// A standardized Gaussian sample: mean 0 and variance 1 within 5 sigma.
void check_standard(const Moments& m, const std::string& what) {
    INFO(what << ": " << m.n << " samples, mean " << m.sum / m.n << ", variance "
              << m.sum2 / m.n - (m.sum / m.n) * (m.sum / m.n));
    REQUIRE(m.n > 1000.0);
    const double mean = m.sum / m.n, variance = m.sum2 / m.n - mean * mean;
    CHECK(std::abs(mean) < 5.0 / std::sqrt(m.n));
    CHECK(std::abs(variance - 1.0) < 5.0 * std::sqrt(2.0 / m.n));
}

// The n links of the walk from `from`, each axis over sqrt(tau): N(0, 1).
void add_walk(const PathSet& path, Bead from, std::size_t n, Moments& m) {
    for (std::size_t i = 0; i < n; ++i) {
        const Bead next = path.bead_after(from.particle, from.slice);
        double sep[SimulationCell::kMaxDim];
        separation(path, from, next, sep);
        for (std::size_t d = 0; d < path.dim(); ++d) m.add(sep[d] / std::sqrt(path.tau()));
        from = next;
    }
}

// The interior beads of the bridge of n links from `from`, off the straight
// line over their standard deviation: N(0, 1) per axis (Levy bridge: bead i
// has variance tau i (n - i) / n per axis).
void add_bridge(const PathSet& path, Bead from, std::size_t n, std::map<std::string, Moments>& m,
                const std::string& label) {
    double end[SimulationCell::kMaxDim];
    separation(path, from, walk_forward(path, from, n), end);
    Bead b = from;
    for (std::size_t i = 1; i < n; ++i) {
        b = path.bead_after(b.particle, b.slice);
        double sep[SimulationCell::kMaxDim];
        separation(path, from, b, sep);
        const double fraction = static_cast<double>(i) / static_cast<double>(n);
        const double sd = std::sqrt(path.tau() * static_cast<double>(i) * (1.0 - fraction));
        for (std::size_t d = 0; d < path.dim(); ++d)
            m[label + " bead " + std::to_string(i)].add((sep[d] - fraction * end[d]) / sd);
    }
}

}  // namespace

TEST_CASE("Grand-canonical sub-moves draw every discrete outcome with its enumerated "
          "probability",
          "[pimc][grand_canonical][worm][balance]") {
    const std::size_t M = 8, capacity = 3, dim = 2, L = 3, ms = 3;
    const double tau = 0.1, side = 4.0, width = 0.7;
    enum class Window { none, insert, remove };
    struct State {
        std::string name;
        bool periodic;
        std::size_t active;
        std::vector<std::size_t> permutation;
        bool open;
        Bead head;
        std::size_t gap, n_min;
        Window window;
        int attempts;
    };
    const std::vector<State> states{
        {"closed, N < n_max: open and insert", true, 2, {0, 1, 2}, false, {0, 0}, 0, 1,
         Window::none, 200000},
        {"closed at N = n_max: the window rejects every insert", true, 3, {0, 1, 2}, false,
         {0, 0}, 0, 1, Window::insert, 60000},
        {"worm across two rows, gap 2: swap, advance up to the tail, recede, close; no remove",
         true, 2, {1, 0, 2}, true, {0, 5}, 2, 1, Window::none, 120000},
        {"worm alone in its row, gap 5: remove; no close; recede short of a full row", true, 2,
         {0, 1, 2}, true, {1, 2}, 5, 1, Window::none, 120000},
        {"the same worm at R = n_min: the window rejects remove", true, 2, {0, 1, 2}, true,
         {1, 2}, 5, 2, Window::remove, 60000},
        {"worm alone in its row, gap 2: too long to remove", true, 2, {0, 1, 2}, true, {1, 5}, 2,
         1, Window::none, 60000},
        {"closed, open cell: tails from the Gaussian insertion density", false, 2, {0, 1, 2},
         false, {0, 0}, 0, 1, Window::none, 60000},
    };
    for (const State& st : states) {
        SECTION(st.name) {
            const SimulationCell cell =
                st.periodic ? SimulationCell(2, std::vector<double>{side, 0.0, 0.0, side}) : SimulationCell{};
            PathSet x(M, capacity, dim, tau, cell), path(M, capacity, dim, tau, cell);
            fixed_rings(x);
            one_class(x);
            one_class(path);
            x.set_active_particles(st.active);
            x.set_permutation(st.permutation);
            if (st.open) x.open_with_gap(st.head, st.gap);
            const Snapshot start = snapshot(x);
            restore(path, start);

            Rules rules;
            rules.constant = st.open ? 1e-60 : 1e60;  // every close/remove, open/insert accepted
            rules.n_min = st.n_min;
            rules.n_max = capacity;
            rules.max_length = L;
            rules.swap_length = ms;
            rules.area = st.periodic ? side * side : 0.0;
            rules.width = st.periodic ? 0.0 : width;
            rules.center = {0.3, -0.2};
            GrandCanonicalSpec spec;
            spec.chemical_potential = {0.0};
            spec.n_min = st.n_min;
            spec.n_max = capacity;
            if (!st.periodic) {
                spec.insertion_width = width;
                spec.insertion_center = {rules.center[0], rules.center[1]};
            }
            GrandCanonicalWormMove worm(static_cast<int>(L), static_cast<int>(ms), rules.constant, spec);
            worm.set_selection_probability(0.5);
            const Hamiltonian free;
            Action action(one_group(capacity, dim), free, dim);
            numerics::Random rng(7 + static_cast<std::uint64_t>(st.attempts) + st.gap);

            const std::map<std::string, double> expected = enumerate_outcomes(x, rules);
            std::map<std::string, std::size_t> observed;
            Moments walk_links, tail_offsets;
            std::map<std::string, Moments> bridges;
            std::vector<double> tail_cells(16, 0.0);
            for (int a = 0; a < st.attempts; ++a) {
                if (!worm.attempt(path, action, rng)) continue;
                const Step s = classify(x, path, ms);
                ++observed[s.key];
                if (s.kind == kInsertKind) {
                    add_walk(path, path.tail(), s.m, walk_links);
                    double r[SimulationCell::kMaxDim];
                    coordinates(path, path.tail(), r);
                    if (st.periodic) {
                        std::size_t cell_index = 0;
                        for (std::size_t d = 0; d < dim; ++d) {
                            const double f = r[d] / side - std::floor(r[d] / side);
                            cell_index = 4 * cell_index + std::min<std::size_t>(3, static_cast<std::size_t>(4.0 * f));
                        }
                        tail_cells[cell_index] += 1.0;
                    } else {
                        for (std::size_t d = 0; d < dim; ++d) tail_offsets.add((r[d] - rules.center[d]) / width);
                    }
                } else if (s.kind == kAdvanceKind) {
                    add_walk(path, x.head(), s.m, walk_links);
                } else if (s.kind == kCloseKind) {
                    add_bridge(path, x.head(), x.gap() + 1, bridges, "close");
                } else if (s.kind == kSwapKind) {
                    add_bridge(path, s.head_in_y, ms, bridges, "swap");
                }
                restore(path, start);
            }

            // chi^2 of the outcome counts over the outcomes of nonzero
            // probability and the attempts that changed nothing.
            const double n = st.attempts;
            double chi2 = 0.0, p_total = 0.0, observed_total = 0.0;
            std::size_t classes = 0;
            for (const auto& [key, p] : expected) {
                const auto it = observed.find(key);
                const double count = it == observed.end() ? 0.0 : static_cast<double>(it->second);
                UNSCOPED_INFO(key << ": " << count << " expected " << n * p);
                chi2 += (count - n * p) * (count - n * p) / (n * p);
                p_total += p;
                observed_total += count;
                ++classes;
            }
            for (const auto& [key, count] : observed) {
                CAPTURE(key, count);
                CHECK(expected.count(key) == 1);  // an outcome of probability 0
            }
            const double rest = n - observed_total, p_rest = 1.0 - p_total;
            UNSCOPED_INFO("unchanged: " << rest << " expected " << n * p_rest);
            if (p_rest > 1e-9) {
                chi2 += (rest - n * p_rest) * (rest - n * p_rest) / (n * p_rest);
                ++classes;
            } else {
                CHECK(rest == 0.0);
            }
            const double dof = static_cast<double>(classes - 1);
            INFO("chi2 = " << chi2 << " for " << dof << " dof");
            CHECK(chi2 < dof + 5.0 * std::sqrt(2.0 * dof));

            // The window rejects exactly the proposals that would leave it.
            const double window = extra(worm, "window_rejections");
            if (st.window == Window::insert) {
                CHECK(window == extra(worm, "insert_proposals"));
                CHECK(window > 0.0);
            } else if (st.window == Window::remove) {
                CHECK(window == extra(worm, "remove_proposals"));
                CHECK(window > 0.0);
            } else {
                CHECK(window == 0.0);
            }
            CHECK(extra(worm, "frame_rejections") == 0.0);

            // The drawn coordinates: free walks, Levy bridges, the tails.
            if (walk_links.n > 0.0) check_standard(walk_links, "walk links");
            for (const auto& [label, m] : bridges) check_standard(m, label);
            if (tail_offsets.n > 0.0) check_standard(tail_offsets, "tail offsets / width");
            const double tails = std::accumulate(tail_cells.begin(), tail_cells.end(), 0.0);
            if (tails > 0.0) {
                double tail_chi2 = 0.0;
                for (const double count : tail_cells)
                    tail_chi2 += (count - tails / 16.0) * (count - tails / 16.0) / (tails / 16.0);
                INFO("uniform tails: chi2 = " << tail_chi2 << " for 15 dof");
                CHECK(tail_chi2 < 15.0 + 5.0 * std::sqrt(30.0));
            }
        }
    }
}


// ── The same bookkeeping with the Takahashi-Imada action ────────────────────

#include "pimc/primitive_action.h"
#include "pimc/takahashi_imada_action.h"

namespace {
// tau V_k + tau^3 W_k of the beads `rows` of `path` at slice k, from a one-slice
// path holding exactly those beads and a Takahashi-Imada action of that size;
// row `moved` (if any) sits at `at` instead.
double oracle_ti_slice(const PathSet& path, const SimulationCell& cell, const Hamiltonian& ham,
                       const std::vector<std::size_t>& rows, std::size_t k,
                       std::size_t moved = HamiltonianComponent::npos,
                       std::span<const double> at = {}) {
    PathSet one(1, rows.size(), path.dim(), path.tau(), cell);
    for (std::size_t i = 0; i < rows.size(); ++i)
        for (std::size_t d = 0; d < path.dim(); ++d)
            one.pos(i, d, 0) = rows[i] == moved ? at[d] : path.pos(rows[i], d, k);
    one.wrap_positions();
    one.rebuild_tables();
    const auto g = group(0, rows.size(), path.dim());
    TakahashiImadaPotentialAction ti(std::span<const ParticleGroup>(&g, 1), ham, path.dim());
    return ti.potential_action_slice(one, 0);
}

// S_V of the beads present at every slice (the missing bead of an open path
// left out).
double oracle_ti_action(const PathSet& path, const SimulationCell& cell,
                        const std::vector<Hamiltonian>& hams) {
    double action = 0.0;
    for (std::size_t k = 0; k < path.num_slices(); ++k) {
        std::vector<std::size_t> rows;
        for (std::size_t j = 0; j < path.active_particles(); ++j)
            if (path.missing_row(k) != j) rows.push_back(j);
        if (!rows.empty()) action += oracle_ti_slice(path, cell, hams[rows.size()], rows, k);
    }
    return action;
}

Action ti_action(std::size_t n, const Hamiltonian& ham, std::size_t dim) {
    const auto g = group(0, n, dim);
    Action action(std::span<const ParticleGroup>(&g, 1), ham, dim, false, false);
    action.add_component(std::make_unique<PrimitiveKineticAction>(
        std::span<const ParticleGroup>(&g, 1), dim));
    action.add_component(std::make_unique<TakahashiImadaPotentialAction>(
        std::span<const ParticleGroup>(&g, 1), ham, dim));
    return action;
}

// A component of the closed-path action of sector N that adds nothing to the
// action and checks every bead delta a closed move (bisection, centroid) asks
// the Takahashi-Imada action for: the delta of bead (p, k) must equal the
// oracle slice action with the bead moved minus the oracle slice action as it
// is.  Like the Takahashi-Imada action it is evaluated jointly on slices that
// hold several staged beads, so it sees exactly the bead deltas that action
// is asked for.  The path must be closed with N particles, since a closed
// move gets the action of the current N.
class TiDeltaCheck final : public ActionComponent {
public:
    TiDeltaCheck(std::size_t n, const Hamiltonian& ham, const SimulationCell& cell, std::size_t dim)
        : n_(n), ham_(ham.clone()), cell_(&cell), group_(group(0, n, dim)),
          ti_(std::span<const ParticleGroup>(&group_, 1), ham, dim) {}
    TiDeltaCheck(const TiDeltaCheck& other)
        : n_(other.n_), ham_(other.ham_.clone()), cell_(other.cell_), group_(other.group_),
          ti_(other.ti_) {}
    bool slice_local_deltas() const noexcept override { return true; }
    bool joint_slice_deltas() const noexcept override { return true; }
    bool supports_free_bridge() const noexcept override { return true; }
    void prepare(std::size_t M) override { ti_.prepare(M); }
    double total_action(const PathSet&) override { return 0.0; }
    double delta_potential_particle(PathSet& path, std::size_t p, std::size_t k,
                                    std::span<const double> proposed) override {
        REQUIRE(!path.is_open());
        REQUIRE(path.active_particles() == n_);
        REQUIRE(p < n_);
        const double engine = ti_.delta_potential_particle(path, p, k, proposed);
        std::vector<std::size_t> rows(n_);
        for (std::size_t j = 0; j < n_; ++j) rows[j] = j;
        const double oracle = oracle_ti_slice(path, *cell_, ham_, rows, k, p, proposed) -
                              oracle_ti_slice(path, *cell_, ham_, rows, k);
        CAPTURE(p, k);
        ++checks;
        REQUIRE(engine == Catch::Approx(oracle).epsilon(1e-9).margin(1e-9));
        return 0.0;
    }
    std::unique_ptr<ActionComponent> clone() const override {
        return std::make_unique<TiDeltaCheck>(*this);
    }
    static inline std::size_t checks = 0;

private:
    std::size_t n_;
    Hamiltonian ham_;
    const SimulationCell* cell_;
    ParticleGroup group_;
    TakahashiImadaPotentialAction ti_;
};

// The smallest distance between two beads present at one slice, minimum
// image, from the coordinates alone (not the distance tables).
double closest_present_pair(const PathSet& path) {
    double best = std::numeric_limits<double>::infinity();
    const std::size_t n = path.active_particles(), dim = path.dim();
    for (std::size_t k = 0; k < path.num_slices(); ++k) {
        const std::size_t missing = path.missing_row(k);
        for (std::size_t i = 0; i < n; ++i) {
            if (i == missing) continue;
            for (std::size_t j = i + 1; j < n; ++j) {
                if (j == missing) continue;
                double del[SimulationCell::kMaxDim];
                for (std::size_t d = 0; d < dim; ++d) del[d] = path.pos(i, d, k) - path.pos(j, d, k);
                best = std::min(best, std::sqrt(path.cell().min_image_dist2(del)));
            }
        }
    }
    return best;
}
}  // namespace

// Worm moves on the capacity's action, bisection and centroid moves on the
// action of the current N, all with the Takahashi-Imada potential, against a
// one-slice oracle of S_V.  Two beads closer than r_floor = 0.2 at one slice
// weigh e^{-(tau V + tau^3 W)} < e^{-40} here (C/r^3 with C = 5.44): a sampled
// pair that close means a move accepted on a wrong action change.
TEST_CASE("Grand-canonical worm and closed-path moves change the Takahashi-Imada S_V by exactly "
          "their deltas", "[pimc][grand_canonical][worm][takahashi_imada]") {
    GrandCanonicalSpec spec;
    spec.chemical_potential = {8.0};
    spec.n_min = 1;
    spec.n_max = 4;
    const SimulationCell cell = hexagonal_cell();
    const std::size_t M = 12, dim = 2, capacity = spec.n_max;
    const double tau = 0.05, r_floor = 0.2;
    PathSet path(M, capacity, dim, tau, cell);
    fill_distinct(path);
    path.wrap_positions();
    path.rebuild_tables();
    one_class(path);
    path.set_active_particles(2);
    Action worm_action = ti_action(capacity, dipolar_moire(capacity, cell), dim);
    std::vector<Action> sector;
    for (std::size_t n = 1; n <= capacity; ++n) {
        Action a = ti_action(n, dipolar_moire(n, cell), dim);
        a.add_component(std::make_unique<TiDeltaCheck>(n, dipolar_moire(n, cell), cell, dim));
        sector.push_back(std::move(a));
    }
    GrandCanonicalWormMove worm(4, 3, 0.03, spec);
    worm.set_selection_probability(0.5);
    BisectionMove bisection(2);
    CentroidMove centroid(0.2);
    numerics::Random rng(20261121);
    const std::vector<Hamiltonian> hams = by_count(dipolar_moire, capacity, cell);
    TiDeltaCheck::checks = 0;
    double s_v = oracle_ti_action(path, cell, hams);
    std::size_t worm_accepts = 0, closed_accepts = 0;
    for (int attempt = 0; attempt < 40000; ++attempt) {
        bool accepted = false;
        const bool closed_move = !path.is_open() && attempt % 2 == 1;
        if (closed_move) {
            Action& a = sector[path.active_particles() - 1];
            accepted = attempt % 4 == 1 ? bisection.attempt(path, a, rng) : centroid.attempt(path, a, rng);
            closed_accepts += accepted;
        } else {
            accepted = worm.attempt(path, worm_action, rng);
            worm_accepts += accepted;
        }
        const double s_new = oracle_ti_action(path, cell, hams);
        CAPTURE(attempt, closed_move, path.active_particles(), path.is_open());
        if (accepted && !closed_move)
            REQUIRE(s_new - s_v == Catch::Approx(worm.last_delta_action()).epsilon(1e-9).margin(1e-9));
        if (closed_move) {
            // The sector action of the closed path is the oracle.
            Action& a = sector[path.active_particles() - 1];
            double potential_now = 0.0;
            for (std::size_t k = 0; k < M; ++k) potential_now += a.potential_action_slice(path, k);
            REQUIRE(potential_now == Catch::Approx(s_new).epsilon(1e-10).margin(1e-10));
        }
        if (!accepted) REQUIRE(s_new == Catch::Approx(s_v).epsilon(1e-12).margin(1e-12));
        s_v = s_new;
        require_present_tables(path);
        REQUIRE(closest_present_pair(path) > r_floor);
    }
    CHECK(worm_accepts > 1000);
    CHECK(closed_accepts > 100);
    CHECK(TiDeltaCheck::checks > 0);
}
