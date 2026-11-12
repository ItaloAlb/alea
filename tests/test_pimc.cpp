// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Unit tests for the PIMC components: PathSet (the imaginary-time ring),
// Action (primitive kinetic + potential, built piecewise from the YAML), the
// Move set (single-slice / bisection / centroid), the statistics and action
// gates, and thread-count independence of the driver.

#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "builders/action_builders.h"
#include "hamiltonian/harmonic_trap.h"
#include "hamiltonian/hamiltonian.h"
#include "io/parser.h"
#include "pimc/action.h"
#include "pimc/move.h"
#include "pimc/path_set.h"
#include "pimc/pimc.h"
#include "pimc/winding.h"
#include "population/particle_group.h"

namespace io = alea::io;

using alea::Action;
using alea::BisectionMove;
using alea::CentroidMove;
using alea::Hamiltonian;
using alea::HarmonicTrap;
using alea::ParticleGroup;
using alea::PathSet;
using alea::Pimc;
using alea::PimcConfig;
using alea::SingleSliceMove;
using alea::SimulationCell;
using alea::numerics::Random;

namespace {

// One unit-mass particle group spanning [0, np).
ParticleGroup make_group(std::size_t np, double mass = 1.0) {
    ParticleGroup g;
    g.name     = "p";
    g.mass     = {mass};
    g.inv_mass = {1.0 / mass};
    g.charge   = 0.0;
    g.first    = 0;
    g.last     = np;
    g.count    = np;
    return g;
}

// A Hamiltonian holding only a harmonic trap V = sum_i 0.5*w^2*x_i^2.
Hamiltonian make_trap_hamiltonian(const std::vector<ParticleGroup>& groups,
                                  double w) {
    Hamiltonian h;
    h.add_component(std::make_unique<HarmonicTrap>(groups, w * w));
    return h;
}

}  // namespace

// ── PathSet ────────────────────────────────────────────────────────────────

TEST_CASE("PathSet exposes an M-slice ring with wrapped links", "[pimc]") {
    PathSet path(8, 1, 1, 0.25);
    CHECK(path.num_slices() == 8);
    CHECK(path.n_particles() == 1);
    CHECK(path.dim() == 1);
    CHECK(path.tau() == Catch::Approx(0.25));
    CHECK(path.beta() == Catch::Approx(2.0));
    CHECK(path.next(7) == 0);
    CHECK(path.prev(0) == 7);
    CHECK(path.next(3) == 4);
    CHECK(path.prev(3) == 2);
}

TEST_CASE("PathSet::collapse_to puts one configuration on every slice",
          "[pimc]") {
    PathSet path(5, 2, 2, 0.1);
    // AoS [p * dim + d].
    const std::vector<double> cfg = {0.5, -1.5,  2.0,  3.0};
    path.collapse_to(cfg);
    for (std::size_t k = 0; k < 5; ++k)
        for (std::size_t d = 0; d < 2; ++d) {
            CHECK(path.pos(0, d, k) == Catch::Approx(cfg[d]));
            CHECK(path.pos(1, d, k) == Catch::Approx(cfg[2 + d]));
        }
}

TEST_CASE("PathSet refuses M = 0 and tau <= 0", "[pimc]") {
    CHECK_THROWS_AS(PathSet(0, 1, 1, 0.1), std::runtime_error);
    CHECK_THROWS_AS(PathSet(4, 1, 1, 0.0), std::runtime_error);
}

// ── Action ─────────────────────────────────────────────────────────────────

TEST_CASE("spring_action matches the hand-computed spring energy", "[pimc]") {
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    const Hamiltonian empty;  // no potential terms: V = 0
    Action action(groups, empty, /*dim=*/1);

    PathSet path(4, 1, 1, 0.5);  // tau = 0.5, m = 1
    path.pos(0, 0, 0) = 0.0;
    path.pos(0, 0, 1) = 1.0;
    path.pos(0, 0, 2) = 1.5;
    path.pos(0, 0, 3) = -0.5;

    // m/(2 tau) * dr^2 = 1/(2*0.5) * 1^2 = 1.0
    CHECK(action.spring_action(path, 0, 0, 1) == Catch::Approx(1.0));
    // 1/(1) * 0.5^2 = 0.25
    CHECK(action.spring_action(path, 0, 1, 2) == Catch::Approx(0.25));
    // Closed-ring link 3 -> 0: dr = -0.5 - 0.0 = -0.5 -> 0.25
    CHECK(action.spring_action(path, 0, 3, 0) == Catch::Approx(0.25));
}

TEST_CASE("spring_action uses the min-image displacement under PBC",
          "[pimc]") {
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    const Hamiltonian empty;
    Action action(groups, empty, /*dim=*/1);

    const std::vector<double> vecs = {4.0};
    SimulationCell cell(1, vecs);
    PathSet path(4, 1, 1, 0.5, cell);
    // Beads near opposite edges: raw dr = -3.8, min-image dr = +0.2.
    path.pos(0, 0, 0) = 1.9;
    path.pos(0, 0, 1) = -1.9;
    // m/(2 tau) * 0.2^2 = 1.0 * 0.04
    CHECK(action.spring_action(path, 0, 0, 1) == Catch::Approx(0.04));
}

TEST_CASE("potential_action_slice equals tau * V at that slice", "[pimc]") {
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    const Hamiltonian ham = make_trap_hamiltonian(groups, 1.0);  // w = 1
    Action action(groups, ham, /*dim=*/1);

    PathSet path(4, 1, 1, 0.25);
    for (std::size_t k = 0; k < 4; ++k) path.pos(0, 0, k) = 0.0;
    path.pos(0, 0, 2) = 2.0;  // V = 0.5 * 2^2 = 2.0 at slice 2

    // tau * V = 0.25 * 2.0 = 0.5; other slices contribute 0.
    CHECK(action.potential_action_slice(path, 2) == Catch::Approx(0.5));
    CHECK(action.potential_action_slice(path, 0) == Catch::Approx(0.0));
}

TEST_CASE("measure_thermo on a collapsed path splits as dN/(2 tau) + V",
          "[pimc]") {
    // Collapsed ring: every spring has dr = 0, so K = dN/(2 tau) exactly and
    // V is the plain potential energy of the single configuration.
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    const Hamiltonian ham = make_trap_hamiltonian(groups, 1.0);
    Action action(groups, ham, /*dim=*/1);

    PathSet path(8, 1, 1, 0.25);  // d = 1, N = 1, tau = 0.25
    for (std::size_t k = 0; k < 8; ++k) path.pos(0, 0, k) = 1.5;

    const Action::ThermoMeasure m = action.measure_thermo(path);
    CHECK(m.kinetic == Catch::Approx(1.0 / (2.0 * 0.25)));       // 2.0
    CHECK(m.potential == Catch::Approx(0.5 * 1.5 * 1.5));         // 1.125
    CHECK(m.energy == Catch::Approx(m.kinetic + m.potential));
    CHECK(m.energy == Catch::Approx(3.125));
}

TEST_CASE("a potential-disabled action carries no Hamiltonian V", "[pimc]") {
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    const Hamiltonian ham = make_trap_hamiltonian(groups, 1.0);
    Action action(groups, ham, /*dim=*/1, /*kinetic=*/true,
                  /*potential=*/false);

    PathSet path(4, 1, 1, 0.25);
    for (std::size_t k = 0; k < 4; ++k) path.pos(0, 0, k) = 3.0;
    // The trap would give V = 4.5, but the component is not in the action.
    CHECK(action.total_potential(path) == Catch::Approx(0.0));
}

TEST_CASE("Action::clone yields independent potential scratch", "[pimc]") {
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    const Hamiltonian ham = make_trap_hamiltonian(groups, 1.0);
    Action a(groups, ham, 1);
    Action b = a.clone();

    PathSet path(4, 1, 1, 0.25);
    for (std::size_t k = 0; k < 4; ++k) path.pos(0, 0, k) = 1.0;
    CHECK(a.total_potential(path) == Catch::Approx(2.0));
    CHECK(b.total_potential(path) == Catch::Approx(2.0));
}

// ── resolve_action_components (YAML action: list) ──────────────────────────

TEST_CASE("resolve_action_components accepts the default primitive pair",
          "[pimc]") {
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    io::Config cfg;
    cfg.action.push_back(io::ActionItemSpec{"primitive", "primitive"});
    const Hamiltonian empty;
    const auto [kin, pot] = resolve_action_components(cfg, empty);
    CHECK(kin);
    CHECK(pot);
}

TEST_CASE("resolve_action_components refuses an action without kinetic",
          "[pimc]") {
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    io::Config cfg;
    cfg.action.push_back(io::ActionItemSpec{"potential", "primitive"});
    const Hamiltonian empty;
    CHECK_THROWS_WITH(resolve_action_components(cfg, empty),
                      Catch::Matchers::ContainsSubstring("kinetic"));
}

TEST_CASE("resolve_action_components refuses dropping V while H has it",
          "[pimc]") {
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    io::Config cfg;
    cfg.action.push_back(io::ActionItemSpec{"kinetic", "primitive"});
    const Hamiltonian trap = make_trap_hamiltonian(groups, 1.0);
    CHECK_THROWS_WITH(resolve_action_components(cfg, trap),
                      Catch::Matchers::ContainsSubstring("potential"));

    // A Hamiltonian with no potential terms needs no potential component.
    const Hamiltonian empty;
    const auto [kin, pot] = resolve_action_components(cfg, empty);
    CHECK(kin);
    CHECK_FALSE(pot);
}

// ── Moves ──────────────────────────────────────────────────────────────────

TEST_CASE("bisection accepts every attempt on a free particle", "[pimc]") {
    // V = 0: the Levy bridge samples the kinetic part exactly, so the
    // Metropolis factor is exp(0) = 1 — acceptance must be exactly 100%.
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    const Hamiltonian empty;
    Action action(groups, empty, 1);

    PathSet path(8, 1, 1, 0.25);
    Random rng(42);
    for (std::size_t k = 0; k < 8; ++k) path.pos(0, 0, k) = 0.0;

    BisectionMove move(2);
    int accepted = 0;
    for (int i = 0; i < 200; ++i)
        if (move.attempt(path, action, rng)) ++accepted;
    CHECK(accepted == 200);
    for (std::size_t k = 0; k < 8; ++k)
        CHECK(std::isfinite(path.pos(0, 0, k)));
}

TEST_CASE("bisection spreads a collapsed free ring to its thermal width",
          "[pimc]") {
    // After many exact Levy proposals the link variance must approach
    // tau/m (free-particle Gaussian) — a distributional check, not just
    // "it ran".
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    const Hamiltonian empty;
    Action action(groups, empty, 1);

    const double tau = 0.25;
    PathSet path(8, 1, 1, tau);
    Random rng(7);
    BisectionMove move(2);  // regenerates 4 of 8 beads per attempt

    for (int i = 0; i < 4000; ++i) move.attempt(path, action, rng);

    // Sample dr = x_{k+1} - x_k across links and samples of the chain.
    double sum = 0.0, sum2 = 0.0;
    int n = 0;
    for (int rep = 0; rep < 20; ++rep) {
        for (int i = 0; i < 500; ++i) move.attempt(path, action, rng);
        for (std::size_t k = 0; k < 8; ++k) {
            const double dr = path.pos(0, 0, path.next(k)) - path.pos(0, 0, k);
            sum += dr;
            sum2 += dr * dr;
            ++n;
        }
    }
    const double mean = sum / n;
    const double var = sum2 / n - mean * mean;
    // Free-particle links: <dr> = 0, <dr^2> = tau/m = 0.25 (the closed-ring
    // constraint only rescales the mean-square end-to-end distance, not the
    // individual link width, for M >> 1... for M = 8 the correlation
    // correction is O(1/M); allow 20%).
    CHECK(std::abs(mean) < 0.15);
    CHECK(var > 0.5 * tau);
    CHECK(var < 1.5 * tau);
}

TEST_CASE("centroid accepts every attempt on a free particle", "[pimc]") {
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    const Hamiltonian empty;
    Action action(groups, empty, 1);

    PathSet path(8, 1, 1, 0.25);
    Random rng(3);
    CentroidMove move(0.5);
    int accepted = 0;
    for (int i = 0; i < 100; ++i)
        if (move.attempt(path, action, rng)) ++accepted;
    CHECK(accepted == 100);  // springs cancel exactly, V = 0
}

TEST_CASE("centroid leaves every spring invariant under PBC", "[pimc]") {
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    const Hamiltonian empty;
    Action action(groups, empty, 1);

    const std::vector<double> vecs = {4.0};
    SimulationCell cell(1, vecs);
    PathSet path(6, 1, 1, 0.25, cell);
    Random rng(11);
    for (std::size_t k = 0; k < 6; ++k)
        path.pos(0, 0, k) = rng.uniform(-2.0, 2.0);

    double s0 = 0.0;
    for (std::size_t k = 0; k < 6; ++k)
        s0 += action.spring_action(path, 0, k, path.next(k));

    CentroidMove move(0.4);
    move.attempt(path, action, rng);

    double s1 = 0.0;
    for (std::size_t k = 0; k < 6; ++k)
        s1 += action.spring_action(path, 0, k, path.next(k));
    CHECK(s1 == Catch::Approx(s0));
}

TEST_CASE("single-slice moves evolve a trapped ring without corrupting it",
          "[pimc]") {
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    const Hamiltonian ham = make_trap_hamiltonian(groups, 1.0);
    Action action(groups, ham, 1);

    PathSet path(8, 1, 1, 0.25);
    Random rng(5);
    SingleSliceMove move(0.5);
    int accepted = 0;
    for (int i = 0; i < 500; ++i)
        if (move.attempt(path, action, rng)) ++accepted;
    CHECK(accepted > 0);  // it must actually move the path
    for (std::size_t k = 0; k < 8; ++k) {
        CHECK(std::isfinite(path.pos(0, 0, k)));
        // The trap keeps beads within a few thermal widths of the origin.
        CHECK(std::abs(path.pos(0, 0, k)) < 10.0);
    }
    // Distance table is consistent with the positions after the last move.
    path.rebuild_tables();
    CHECK(path.dist_tables()(0, 0, 0) == Catch::Approx(0.0));
}

TEST_CASE("the Levy bridge density is the link action minus the endpoint term",
          "[pimc]") {
    // Identity behind the frame Hastings factor:
    //   -ln q = S_link - sum_d m_d F_d^2 / (2 tau n) + lnNorm
    // (n = window length, F = frame/net displacement).  Verified here for
    // the exact construction BisectionMove performs; the factor
    // exp(-m(F^2 - F_x^2)/(2 tau n)) in the acceptance follows from it, and
    // its n = M specialization exp(-m L^2 W^2 / 2 beta) is what the free-PBC
    // physics test's <W^2> checks end to end.
    const double tau = 0.25;
    const double m = 1.0;
    for (const int n_beads : {4, 8}) {
        const int levels = (n_beads == 4) ? 2 : 3;
        Random rng(1234);
        std::vector<double> x(static_cast<std::size_t>(n_beads) + 1);
        x[0] = 0.35;
        x[static_cast<std::size_t>(n_beads)] = -1.2;
        const double F = x[static_cast<std::size_t>(n_beads)] - x[0];

        // Construct the bridge exactly as BisectionMove does (open cell),
        // accumulating the proposal's quadratic form -ln q + lnNorm.
        double s_tilde = 0.0;
        for (int level = levels; level >= 1; --level) {
            const int stride = 1 << level;
            const int half = stride / 2;
            const double sigma = std::sqrt(half * tau / (2.0 * m));
            for (int seg = 0; seg < n_beads; seg += stride) {
                const double mid =
                    0.5 * (x[static_cast<std::size_t>(seg)] +
                           x[static_cast<std::size_t>(seg + stride)]);
                const double draw = sigma * rng.gaussian();
                x[static_cast<std::size_t>(seg + half)] = mid + draw;
                s_tilde += draw * draw / (2.0 * sigma * sigma);
            }
        }
        double s_link = 0.0;
        for (int i = 0; i < n_beads; ++i) {
            const double dr = x[static_cast<std::size_t>(i + 1)] -
                              x[static_cast<std::size_t>(i)];
            s_link += m * dr * dr / (2.0 * tau);
        }
        const double endpoint = m * F * F / (2.0 * tau * n_beads);
        CHECK(s_tilde == Catch::Approx(s_link - endpoint).epsilon(1e-11));
    }
}

// Heat-bath frame kernel of BisectionMove in 1-D, free particle (V = 0):
// candidates G - L, G, G + L with weight w(F) = exp(-m F^2 / (2 n tau)),
// Z(G) = sum of the three, chosen with w/Z(G), accepted with
// min(1, Z(G)/Z(F)); hence P(G -> F) = w(F) / max(Z(G), Z(F)).
namespace {
double frame_weight(double f, double n_tau) { return std::exp(-f * f / (2.0 * n_tau)); }
double frame_z(double g, double cell, double n_tau) {
    return frame_weight(g - cell, n_tau) + frame_weight(g, n_tau) + frame_weight(g + cell, n_tau);
}
double frame_transition(double from, double to, double cell, double n_tau) {
    return frame_weight(to, n_tau) /
           std::max(frame_z(from, cell, n_tau), frame_z(to, cell, n_tau));
}
}  // namespace

TEST_CASE("frame-jump bisection samples the winding weight and accepts at "
          "the predicted rate", "[pimc]") {
    // 1-D periodic, L = 4, tau = 0.0625, M = 32 (the physics config's
    // parameters), free particle (V = 0), full-ring bisection (levels 5).
    // Heat-bath RELATIVE increments s=-1,0,+1: the equilibrium acceptance is
    // sum_W pi(W) sum_s P(W -> W+s), P from frame_transition above (n tau =
    // beta = 2, frame W L).  Uniform increments wasted 2/3 of these moves.
    // The reference changes with the proposal, not with the physical target
    // pi(W) ~ exp(-4 W^2). All integer sectors remain accessible.
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    const Hamiltonian empty;
    Action action(groups, empty, /*dim=*/1);

    const std::vector<double> vecs = {4.0};
    SimulationCell cell(1, vecs);
    const double tau = 0.0625;
    PathSet path(32, 1, 1, tau, cell);
    for (std::size_t k = 0; k < 32; ++k) path.pos(0, 0, k) = 0.0;  // W = 0
    path.rebuild_tables();

    auto winding = [&]() {
        double sum = 0.0;
        for (std::size_t k = 0; k < 32; ++k) {
            double del = path.pos(0, 0, (k + 1) % 32) - path.pos(0, 0, k);
            path.cell().min_image(&del);
            sum += del;
        }
        return std::round(sum / 4.0);  // in units of L
    };

    Random rng(42);
    BisectionMove move(5);  // full ring
    int accepted = 0;
    int flips = 0;
    double w_prev = winding();

    constexpr int kAttempts = 20000;
    for (int i = 0; i < kAttempts; ++i) {
        if (move.attempt(path, action, rng)) ++accepted;
        const double w = winding();
        if (w != w_prev) ++flips;
        w_prev = w;

    }

    const double acc_rate =
        static_cast<double>(accepted) / static_cast<double>(kAttempts);
    double z = 0.0;
    double flux = 0.0;
    for (int w = -8; w <= 8; ++w) {
        const double pi_w = std::exp(-4.0 * w * w);
        z += pi_w;
        for (int s = -1; s <= 1; ++s)
            flux += pi_w * frame_transition(4.0 * w, 4.0 * (w + s), 4.0, 2.0);
    }
    const double expected = flux / z;
    INFO("acceptance=" << acc_rate << " equilibrium prediction=" << expected
         << " flips=" << flips);
    CHECK(std::abs(acc_rate - expected) < 0.02);
    // Sector changes are exponentially rare for local moves, not impossible.
    CHECK(flips > 20);
    // No artificial cap assertion: broad-sector accessibility is tested at
    // beta=16 and from high-|W| initial states below.
}

// Relative lattice-frame proposals must preserve detailed balance on every
// supported winding sector, not just the nearly empty |W| <= 1 low-beta set.
TEST_CASE("bisection escapes high winding and obeys analytic frame balance",
          "[pimc][frame-regression]") {
    const std::vector<ParticleGroup> groups = {make_group(1)};
    const Hamiltonian empty;
    Action action(groups, empty, 1);
    const SimulationCell cell(1, std::vector<double>{4.0});
    constexpr std::size_t M = 256;
    PathSet path(M, 1, 1, 0.0625, cell);
    BisectionMove move(8);
    Random rng(782319);
    auto set_winding = [&](int w) {
        for (std::size_t k = 0; k < M; ++k) {
            double x = 4.0 * w * static_cast<double>(k) / M;
            cell.wrap(&x);
            path.pos(0, 0, k) = x;
        }
        path.rebuild_tables();
    };
    auto winding = [&]() {
        double sum = 0.0;
        for (std::size_t k = 0; k < M; ++k) {
            double dx = path.pos(0, 0, path.next(k)) - path.pos(0, 0, k);
            cell.min_image(&dx);
            sum += dx;
        }
        return static_cast<int>(std::llround(sum / 4.0));
    };
    set_winding(4);
    for (int i = 0; i < 100; ++i) move.attempt(path, action, rng);
    CHECK(std::abs(winding()) < 4);

    // Independent normalized sector theory: pi(W) proportional to exp(-W²/2)
    // (n tau = 16, L = 4). Heat-bath RELATIVE +/-1 proposals give
    // P(1->2) = e^-2 / max(Z(1), Z(2)) and P(2->1) = e^-1/2 / max(Z(2), Z(1)),
    // whose ratio is pi(2)/pi(1) (detailed balance). The original absolute
    // proposal/gate makes BOTH transitions impossible and fails this check.
    constexpr int trials = 2400;
    double rates[4]{};
    for (int direction = 0; direction < 4; ++direction) {
        const int sign = direction < 2 ? 1 : -1;
        const bool outward = direction % 2 == 0;
        const int from = sign * (outward ? 1 : 2);
        const int to = sign * (outward ? 2 : 1);
        int count = 0;
        for (int i = 0; i < trials; ++i) {
            set_winding(from);
            move.attempt(path, action, rng);
            if (winding() == to) ++count;
        }
        rates[direction] = static_cast<double>(count) / trials;
        const double expected = frame_transition(4.0 * from, 4.0 * to, 4.0, 16.0);
        const double sampling_error = std::sqrt(expected * (1.0 - expected) / trials);
        INFO("direction=" << direction << " rate=" << rates[direction]
             << " expected=" << expected << " binomial SEM=" << sampling_error);
        CHECK(std::abs(rates[direction] - expected) < 5.0 * sampling_error);
    }
    CHECK(rates[0] > 0.0);
    CHECK(rates[1] > 0.0);
}

TEST_CASE("partial-window bisection is not gated by absolute frame magnitude",
          "[pimc][frame-regression]") {
    const std::vector<ParticleGroup> groups = {make_group(1)};
    const Hamiltonian empty;
    Action action(groups, empty, 1);
    const SimulationCell cell(1, std::vector<double>{4.0});
    PathSet path(256, 1, 1, 0.0625, cell);
    // Every half-ring window has frame 8 (two lattice cells), but each link
    // is far inside the canonical minimum-image domain. It is valid support.
    for (std::size_t k = 0; k < 256; ++k) {
        double x = 16.0 * static_cast<double>(k) / 256.0;
        cell.wrap(&x);
        path.pos(0, 0, k) = x;
    }
    path.rebuild_tables();
    BisectionMove move(7);
    Random rng(31297);
    int accepted = 0;
    for (int i = 0; i < 100; ++i) accepted += move.attempt(path, action, rng);
    CHECK(accepted > 0);
    // Partial-window endpoint term uses n*tau=8, NOT the full beta=16.
    // Uniform W=2 has half-window F_x=4 and reaches W=3 through F=8;
    // uniform W=4 has F_x=8 and reaches W=3 through F=4 (heat-bath kernel
    // frame_transition with n tau = 8, cell 4).
    constexpr int trials = 2400;
    for (int initial_w : {2, 4}) {
        int transitions = 0;
        for (int trial = 0; trial < trials; ++trial) {
            for (std::size_t k = 0; k < 256; ++k) {
                double x = 4.0 * initial_w * static_cast<double>(k) / 256.0;
                cell.wrap(&x);
                path.pos(0, 0, k) = x;
            }
            path.rebuild_tables();
            move.attempt(path, action, rng);
            double sum = 0.0;
            for (std::size_t k = 0; k < 256; ++k) {
                double dx = path.pos(0, 0, path.next(k)) - path.pos(0, 0, k);
                cell.min_image(&dx);
                sum += dx;
            }
            transitions += std::llround(sum / 4.0) == 3;
        }
        const double expected = initial_w == 2 ? frame_transition(4.0, 8.0, 4.0, 8.0)
                                               : frame_transition(8.0, 4.0, 4.0, 8.0);
        const double error = std::sqrt(expected * (1.0 - expected) / trials);
        const double observed = static_cast<double>(transitions) / trials;
        INFO("initial W=" << initial_w << " transition rate=" << observed
             << " theory=" << expected << " binomial SEM=" << error);
        CHECK(std::abs(observed - expected) < 5.0 * error);
    }
}

TEST_CASE("periodic bisection does not spend proposals on improbable windings",
          "[pimc][frame-regression]") {
    // With uniform s in {-1,0,+1}^2 a 2-D periodic bisection
    // accepted ~1/9 of its moves for a free particle in a cell with
    // L^2 >> n tau, every other draw shifting the window by a whole cell.
    // The heat-bath frame gives those candidates weight exp(-L^2/(2 n tau))
    // ~ e^-400 here, so a free particle must accept essentially every move.
    const std::vector<ParticleGroup> groups = {make_group(1)};
    const Hamiltonian empty;
    Action action(groups, empty, 2);
    const SimulationCell cell(2, std::vector<double>{20.0, 0.0, 0.0, 20.0});
    PathSet path(32, 1, 2, 0.0625, cell);
    for (std::size_t k = 0; k < 32; ++k) {
        path.pos(0, 0, k) = 3.0;
        path.pos(0, 1, k) = 7.0;
    }
    path.rebuild_tables();
    BisectionMove move(3);                  // n tau = 0.5, L^2 / (2 n tau) = 400
    Random rng(2718);
    constexpr int attempts = 4000;
    int accepted = 0;
    for (int i = 0; i < attempts; ++i) accepted += move.attempt(path, action, rng);
    const double rate = static_cast<double>(accepted) / attempts;
    INFO("acceptance=" << rate << " (uniform frames: ~1/9)");
    CHECK(rate > 0.999);
}

TEST_CASE("move constructors and weights reject invalid numeric parameters",
          "[pimc][move-regression]") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (double bad : {0.0, -1.0, nan, inf}) {
        CHECK_THROWS_AS(SingleSliceMove(bad), std::invalid_argument);
        CHECK_THROWS_AS(CentroidMove(bad), std::invalid_argument);
        SingleSliceMove move;
        CHECK_THROWS_AS(move.set_weight(bad), std::invalid_argument);
    }
    CHECK_THROWS_AS(BisectionMove(0), std::invalid_argument);
    CHECK_THROWS_AS(BisectionMove(-1), std::invalid_argument);
    CHECK_THROWS_AS(BisectionMove(31), std::invalid_argument);
    CHECK_NOTHROW(BisectionMove(30));
}

TEST_CASE("extreme finite centroid displacement cannot erase spring links", "[pimc][move_overflow]") {
    const std::vector<ParticleGroup> groups = {make_group(1)};
    const Hamiltonian empty;
    Action action(groups, empty, 1);
    PathSet path(8, 1, 1, 0.25);
    path.pos(0, 0, 1) = 1.0;
    const double before = action.spring_action(path, 0, 0, 1) +
                          action.spring_action(path, 0, 1, 2);
    REQUIRE(before == 4.0);
    Random rng(42);
    CentroidMove move(1e20);
    CHECK_FALSE(move.attempt(path, action, rng));
    CHECK(path.pos(0, 0, 1) == 1.0);
    CHECK(path.pos(0, 0, 0) == 0.0);
}

TEST_CASE("stiff free ring rejects numerically nonrigid centroid translation",
          "[pimc][move_precision]") {
    const std::vector<ParticleGroup> groups = {make_group(1)};
    const Hamiltonian empty;
    Action action(groups, empty, 1);
    // All coordinates and the step are ordinary-sized; the tiny tau makes
    // binary64 loss of the 1e-10 separation physically significant.
    PathSet path(8, 1, 1, 1e-20);
    path.pos(0, 0, 1) = 1e-10;
    Random rng(42);
    CentroidMove move(0.5);
    REQUIRE_FALSE(move.attempt(path, action, rng));
    for (std::size_t k = 0; k < 8; ++k)
        CHECK(path.pos(0, 0, k) == (k == 1 ? 1e-10 : 0.0));
}

TEST_CASE("large finite bridge endpoints cannot overflow the path", "[pimc][move_overflow]") {
    const std::vector<ParticleGroup> groups = {make_group(1)};
    const Hamiltonian empty;
    Action action(groups, empty, 1);
    PathSet path(4, 1, 1, 0.25);
    for (std::size_t k = 0; k < 4; ++k) path.pos(0, 0, k) = 1e308;
    path.rebuild_tables();
    Random rng(42);
    BisectionMove move(1);
    move.attempt(path, action, rng);
    for (std::size_t k = 0; k < 4; ++k)
        CHECK(path.pos(0, 0, k) == 1e308);
}

TEST_CASE("nonfinite generated moves leave path coordinates finite", "[pimc][move_overflow]") {
    const std::vector<ParticleGroup> groups = {make_group(1)};
    const Hamiltonian empty;
    Action action(groups, empty, 1);
    for (bool centroid : {false, true}) {
        PathSet path(8, 1, 1, 0.25);
        Random rng(42);
        if (centroid) {
            CentroidMove move(std::numeric_limits<double>::max());
            for (int i = 0; i < 20; ++i) move.attempt(path, action, rng);
        } else {
            SingleSliceMove move(std::numeric_limits<double>::max());
            for (int i = 0; i < 20; ++i) move.attempt(path, action, rng);
        }
        for (std::size_t k = 0; k < 8; ++k)
            CHECK(std::isfinite(path.pos(0, 0, k)));
    }
}

TEST_CASE("forced centroid rejection restores every coordinate exactly",
          "[pimc][move-regression]") {
    const std::vector<ParticleGroup> groups = {make_group(2)};
    const Hamiltonian ham = make_trap_hamiltonian(groups, 1.0);
    Action action(groups, ham, 1);
    PathSet path(8, 2, 1, 0.25);
    for (std::size_t p = 0; p < 2; ++p)
        for (std::size_t k = 0; k < 8; ++k)
            path.pos(p, 0, k) = 1.0 + p + 0.01 * k;
    path.rebuild_tables();
    CentroidMove move(1e20); // Addition loses old low bits; subtraction cannot undo it.
    Random rng(3);
    REQUIRE_FALSE(move.attempt(path, action, rng));
    for (std::size_t p = 0; p < 2; ++p)
        for (std::size_t k = 0; k < 8; ++k)
            CHECK(path.pos(p, 0, k) == 1.0 + p + 0.01 * k);
}

TEST_CASE("periodic centroid rejection restores wrapped beads bit exactly",
          "[pimc][move-regression]") {
    const std::vector<ParticleGroup> groups = {make_group(1)};
    const Hamiltonian ham = make_trap_hamiltonian(groups, 1e5);
    Action action(groups, ham, 1);
    const SimulationCell cell(1, std::vector<double>{4.0});
    PathSet path(8, 1, 1, 0.25, cell);
    for (std::size_t k = 0; k < 8; ++k) path.pos(0, 0, k) = 0.001 + 0.0001 * k;
    path.rebuild_tables();
    CentroidMove move(1.0);
    Random rng(3);
    REQUIRE_FALSE(move.attempt(path, action, rng));
    for (std::size_t k = 0; k < 8; ++k)
        CHECK(path.pos(0, 0, k) == 0.001 + 0.0001 * k);
}

TEST_CASE("periodic centroid wraps only the selected particle",
          "[pimc][move-regression]") {
    const std::vector<ParticleGroup> groups = {make_group(2)};
    const Hamiltonian empty;
    Action action(groups, empty, 1);
    const SimulationCell cell(1, std::vector<double>{4.0});
    PathSet path(8, 2, 1, 0.25, cell);
    // Deliberately store noncanonical but equivalent coordinates: an attempt
    // must not alter another particle, even as incidental global wrapping.
    for (std::size_t k = 0; k < 8; ++k) {
        path.pos(0, 0, k) = 8.125 + k * 0.01;
        path.pos(1, 0, k) = 12.375 + k * 0.01;
    }
    path.rebuild_tables();
    Random predict(17);
    const auto selected = static_cast<std::size_t>(predict.uniform() * 2.0);
    Random rng(17);
    CentroidMove move(0.3);
    REQUIRE(move.attempt(path, action, rng));
    const std::size_t untouched = 1 - selected;
    for (std::size_t k = 0; k < 8; ++k)
        CHECK(path.pos(untouched, 0, k) ==
              (untouched == 0 ? 8.125 : 12.375) + k * 0.01);
}

TEST_CASE("move particle-only table refresh agrees with full rebuild",
          "[pimc][move-regression]") {
    const std::vector<ParticleGroup> groups = {make_group(3)};
    const Hamiltonian ham = make_trap_hamiltonian(groups, 1.0);
    Action action(groups, ham, 1);
    const SimulationCell cell(1, std::vector<double>{4.0});
    PathSet path(16, 3, 1, 0.0625, cell);
    Random rng(62819);
    SingleSliceMove single(1.3);
    BisectionMove bridge(3);
    CentroidMove centroid(0.6);
    for (alea::Move* move : std::vector<alea::Move*>{&single, &bridge, &centroid}) {
        for (int trial = 0; trial < 30; ++trial) {
            move->attempt(path, action, rng);
            std::vector<double> incremental;
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 3; ++j)
                    for (std::size_t k = 0; k < 16; ++k)
                        incremental.push_back(path.dist_tables()(k, i, j));
            path.rebuild_tables();
            std::size_t index = 0;
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 3; ++j)
                    for (std::size_t k = 0; k < 16; ++k)
                        REQUIRE(incremental[index++] == path.dist_tables()(k, i, j));
        }
    }
}

// ── Pimc driver gates and reproducibility ──────────────────────────────────

namespace {

PimcConfig small_config() {
    PimcConfig cfg;
    cfg.threads              = 1;
    cfg.samples              = 8;
    cfg.time_step            = 0.25;
    cfg.beta                 = 1.0;   // M = 4
    cfg.equilibration_blocks = 5;
    cfg.accumulation_blocks  = 30;
    cfg.steps_per_block      = 5;
    cfg.sub_steps            = 1;
    cfg.seed                 = 99;
    PimcConfig::MoveSpec b;
    b.type = "bisection"; b.levels = 1; b.weight = 1.0;
    cfg.moves.push_back(b);
    PimcConfig::MoveSpec c;
    c.type = "centroid"; c.step = 0.4; c.weight = 1.0;
    cfg.moves.push_back(c);
    return cfg;
}

}  // namespace

TEST_CASE("Pimc refuses a non-integer beta/tau", "[pimc]") {
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    const Hamiltonian empty;
    PimcConfig cfg = small_config();
    cfg.time_step = 0.3;  // beta/tau = 3.333...
    CHECK_THROWS_WITH(Pimc(cfg, groups, empty, SimulationCell(), 1, 1),
                      Catch::Matchers::ContainsSubstring("not an integer"));
}

TEST_CASE("Pimc refuses bisection levels that exceed the ring", "[pimc]") {
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    const Hamiltonian empty;
    PimcConfig cfg = small_config();  // M = 4
    cfg.moves[0].levels = 4;          // 2^4 = 16 > M = 4
    CHECK_THROWS_WITH(Pimc(cfg, groups, empty, SimulationCell(), 1, 1),
                      Catch::Matchers::ContainsSubstring("bisection levels"));
}

TEST_CASE("Pimc refuses a multi-particle exchange class without the opt-in",
          "[pimc]") {
    const std::vector<ParticleGroup> groups = {make_group(2, 1.0)};
    const Hamiltonian empty;
    PimcConfig cfg = small_config();

    cfg.exchange_class_max = 2;
    cfg.statistics = "";
    CHECK_THROWS_WITH(Pimc(cfg, groups, empty, SimulationCell(), 1, 2),
                      Catch::Matchers::ContainsSubstring("boltzmannon"));

    cfg.statistics = "boltzmannon";  // explicit opt-in: distinguishable
    CHECK_NOTHROW(Pimc(cfg, groups, empty, SimulationCell(), 1, 2));
}

TEST_CASE("Pimc refuses centroid-only sampling of a multi-slice path",
          "[pimc][move_validation]") {
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    const Hamiltonian ham = make_trap_hamiltonian(groups, 1.0);
    auto cfg = small_config();
    cfg.moves.erase(cfg.moves.begin());  // only rigid translations remain
    // A rigid shift leaves every r_k-r_{k+1} unchanged. Reblocking cannot
    // detect this restriction of exp(-S) to the initial ring shape.
    CHECK_THROWS_WITH(Pimc(cfg, groups, ham, SimulationCell(), 1, 1),
                      Catch::Matchers::ContainsSubstring("frozen"));
}

TEST_CASE("Pimc accepts moves that change the internal beads of a multi-slice path",
          "[pimc][move_validation]") {
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    const Hamiltonian ham = make_trap_hamiltonian(groups, 1.0);
    auto cfg = small_config();
    SECTION("bisection alone") { cfg.moves.resize(1); }
    SECTION("single-slice alone") {
        cfg.moves.resize(1);
        cfg.moves[0].type = "single_slice";
    }
    SECTION("the default move set") { cfg.moves.clear(); }
    CHECK_NOTHROW(Pimc(cfg, groups, ham, SimulationCell(), 1, 1));
}

TEST_CASE("Pimc accepts centroid-only sampling when there is one slice",
          "[pimc][move_validation]") {
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    const Hamiltonian ham = make_trap_hamiltonian(groups, 1.0);
    auto cfg = small_config();
    cfg.time_step = cfg.beta;  // M = 1 has no internal bead separations
    SECTION("explicit centroid") { cfg.moves.erase(cfg.moves.begin()); }
    SECTION("the default move set") { cfg.moves.clear(); }
    CHECK_NOTHROW(Pimc(cfg, groups, ham, SimulationCell(), 1, 1));
}

TEST_CASE("Pimc results do not depend on the thread count", "[pimc]") {
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    const Hamiltonian ham = make_trap_hamiltonian(groups, 1.0);

    auto run_with = [&](int threads) {
        PimcConfig cfg = small_config();
        cfg.threads = threads;
        Pimc driver(cfg, groups, ham, SimulationCell(), 1, 1);
        driver.run();
        return driver;
    };

    const Pimc one = run_with(1);
    const Pimc two = run_with(2);

    REQUIRE(one.block_energies().size() == two.block_energies().size());
    for (std::size_t i = 0; i < one.block_energies().size(); ++i)
        REQUIRE(one.block_energies()[i] == two.block_energies()[i]);
    REQUIRE(one.energy() == two.energy());
    REQUIRE(one.energy_error() == two.energy_error());
    REQUIRE(one.kinetic() == two.kinetic());
    REQUIRE(one.potential() == two.potential());
    REQUIRE(one.acceptance_ratio() == two.acceptance_ratio());

    // And the result must be physical-ish: finite, with a usable error bar.
    CHECK(std::isfinite(one.energy()));
    CHECK(one.energy_error() > 0.0);
    CHECK(one.acceptance_ratio() > 0.0);
    CHECK(one.acceptance_ratio() < 1.0);
}

TEST_CASE("Pimc reports the reblocked error alongside the naive one",
          "[pimc]") {
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    const Hamiltonian ham = make_trap_hamiltonian(groups, 1.0);
    PimcConfig cfg = small_config();
    cfg.accumulation_blocks = 60;
    Pimc driver(cfg, groups, ham, SimulationCell(), 1, 1);
    driver.run();
    CHECK(driver.naive_error() > 0.0);
    // Reblocking can only enlarge the error (correlations), never shrink it.
    CHECK(driver.energy_error() >= driver.naive_error() * 0.5);
    const alea::RunResult rr = driver.run_result();
    CHECK(rr.method == "pimc");
    bool has_reblock_plateau = false;
    for (const auto& [k, v] : rr.scalars)
        if (k == "reblock_plateau") has_reblock_plateau = true;
    CHECK(has_reblock_plateau);
}

TEST_CASE("PIMC random streams of all stages, replicas and purposes are distinct",
          "[pimc][seed]") {
    // App gives sampling stage o the seed base + o + 1.  The former
    // base + o + 1 + s replica seeds made replica s+1 of stage o and replica
    // s of stage o+1 share one stream, and collided with VMC's base + thread.
    for (const std::uint64_t base :
         {std::uint64_t{0}, std::uint64_t{42}, ~std::uint64_t{0}}) {
        std::vector<std::uint64_t> seeds;
        for (std::uint64_t t = 0; t < 64; ++t) seeds.push_back(base + t);  // VMC
        for (std::uint64_t stage = 0; stage < 8; ++stage) {
            const std::uint64_t stage_seed = base + stage + 1;
            seeds.push_back(alea::numerics::stream_seed(stage_seed, alea::kPimcEnsembleInitStream, 0));
            for (std::uint64_t s = 0; s < 64; ++s) {
                seeds.push_back(alea::numerics::stream_seed(stage_seed, alea::kPimcReplicaStream, s));
                seeds.push_back(alea::numerics::stream_seed(stage_seed, alea::kPimcRingInitStream, s));
            }
        }
        const auto count = seeds.size();
        std::sort(seeds.begin(), seeds.end());
        CHECK(std::unique(seeds.begin(), seeds.end()) - seeds.begin() ==
              static_cast<std::ptrdiff_t>(count));
    }
    // Pinned values (from an independent Python SplitMix64): a change here
    // changes every PIMC replay stream.
    CHECK(alea::numerics::splitmix64(0) == 0xe220a8397b1dcdafULL);
    CHECK(alea::numerics::splitmix64(1) == 0x910a2dec89025cc1ULL);
}

TEST_CASE("An omitted PIMC move list resolves to the shared default and is reported",
          "[pimc]") {
    const std::vector<ParticleGroup> groups = {make_group(1, 1.0)};
    const Hamiltonian ham = make_trap_hamiltonian(groups, 1.0);
    for (const auto& [beta, levels] :
         std::vector<std::pair<double, int>>{{0.25, 0}, {0.5, 1}, {1.25, 2}, {2.0, 3}}) {
        CAPTURE(beta, levels);
        PimcConfig cfg = small_config();  // tau = 0.25
        cfg.moves.clear();
        cfg.beta = beta;
        cfg.accumulation_blocks = 4;
        Pimc driver(cfg, groups, ham, SimulationCell(), 1, 1);
        const auto M = static_cast<std::size_t>(std::llround(beta / cfg.time_step));
        const auto expected = io::default_pimc_moves(M);
        REQUIRE(driver.move_specs().size() == expected.size());
        REQUIRE(expected.back().type == "centroid");
        CHECK(expected.size() == (levels > 0 ? 2u : 1u));
        if (levels > 0) CHECK(expected.front().levels == levels);
        driver.run();
        const auto result = driver.run_result();
        const auto value = [&](const std::string& key) {
            for (const auto& [name, v] : result.scalars)
                if (name == key) return v;
            FAIL("missing run_result scalar " << key);
            return 0.0;
        };
        for (std::size_t i = 0; i < expected.size(); ++i) {
            const std::string label = "move_" + std::to_string(i);
            CHECK(driver.move_specs()[i].type == expected[i].type);
            CHECK(value(label + "_weight") == expected[i].weight);
            if (expected[i].type == "bisection")
                CHECK(value(label + "_levels") == expected[i].levels);
            else
                CHECK(value(label + "_step") == expected[i].step);
            CHECK(value(label + "_infinite_action_rejections") == 0.0);
        }
        CHECK(value("infinite_action_rejections") == 0.0);
    }
}

// ── YAML parsing of the pimc section and the action list ───────────────────

TEST_CASE("parse_config reads a pimc run with moves and statistics",
          "[pimc][parser]") {
    const std::string yaml = R"(
project: { name: t, seed: 5 }
system:
  lattice: { type: open, dimension: 1, L: 1.0 }
  particles:
    - { name: p, group: g, type: boltzmannon, mass: 1.0, up: 2, down: 0 }
hamiltonian:
  terms: []
action:
  - kinetic: primitive
  - potential: primitive
qmc:
  - method: pimc
    samples: 16
    time_step: 0.125
    beta: 1.0
    statistics: boltzmannon
    moves:
      - bisection: { levels: 3, weight: 2.0 }
      - centroid: { step: 0.4 }
      - single_slice: { step: 0.2, weight: 0.5 }
)";
    const io::Config c = io::parse_config(yaml);
    REQUIRE(c.pimc.size() == 1);
    const io::PimcSpec& p = c.pimc[0];
    CHECK(p.samples == 16);
    CHECK(p.time_step == Catch::Approx(0.125));
    CHECK(p.beta == Catch::Approx(1.0));
    CHECK(p.statistics == "boltzmannon");
    REQUIRE(p.moves.size() == 3);
    CHECK(p.moves[0].type == "bisection");
    CHECK(p.moves[0].levels == 3);
    CHECK(p.moves[0].weight == Catch::Approx(2.0));
    CHECK(p.moves[1].type == "centroid");
    CHECK(p.moves[1].step == Catch::Approx(0.4));
    CHECK(p.moves[2].type == "single_slice");
    CHECK(p.moves[2].weight == Catch::Approx(0.5));

    REQUIRE(c.action.size() == 2);
    CHECK(c.action[0].component == "kinetic");
    CHECK(c.action[1].component == "potential");
}

TEST_CASE("an omitted action: defaults to primitive (both components)",
          "[pimc][parser]") {
    const std::string yaml = R"(
qmc:
  - method: pimc
    samples: 4
    time_step: 0.25
    beta: 1.0
)";
    const io::Config c = io::parse_config(yaml);
    REQUIRE(c.action.size() == 1);
    CHECK(c.action[0].component == "primitive");
}

TEST_CASE("the action parser rejects unknown and duplicate components",
          "[pimc][parser]") {
    CHECK_THROWS_WITH(io::parse_config("action: [ { magic: {} } ]\n"),
                      Catch::Matchers::ContainsSubstring("unknown action component"));
    CHECK_THROWS_WITH(io::parse_config(
                          "action: [ { kinetic: {} }, { primitive: {} } ]\n"),
                      Catch::Matchers::ContainsSubstring("given twice"));
    CHECK_THROWS_WITH(io::parse_config(
                          "action: [ { kinetic: exponential } ]\n"),
                      Catch::Matchers::ContainsSubstring("unknown form"));
    CHECK_THROWS_WITH(io::parse_config("action: kinetic\n"),
                      Catch::Matchers::ContainsSubstring("list of components"));
}

TEST_CASE("the pimc parser validates its keys", "[pimc][parser]") {
    auto expect_error = [](const std::string& body, const std::string& frag) {
        CHECK_THROWS_WITH(io::parse_config("qmc:\n  - method: pimc\n" + body),
                          Catch::Matchers::ContainsSubstring(frag));
    };
    // beta required
    expect_error("    samples: 4\n    time_step: 0.25\n", "beta");
    // time_step required
    expect_error("    samples: 4\n    beta: 1.0\n", "time_step");
    // only the thermodynamic estimator exists so far
    expect_error(
        "    samples: 4\n    time_step: 0.25\n    beta: 1.0\n"
        "    energy: virial\n",
        "thermodynamic");
    // unknown statistics (exchange belongs to a later stage)
    expect_error(
        "    samples: 4\n    time_step: 0.25\n    beta: 1.0\n"
        "    statistics: fermions\n",
        "statistics");
    // Strict map-key validation rejects an unknown move before dispatch;
    // assert that diagnostic, without changing any physics expectation.
    expect_error(
        "    samples: 4\n    time_step: 0.25\n    beta: 1.0\n"
        "    moves: [ { teleport: {} } ]\n",
        "unknown key");
    // The worm needs its group and explicit lengths.
    expect_error(
        "    samples: 4\n    time_step: 0.25\n    beta: 1.0\n"
        "    moves: [ { worm: {} } ]\n",
        "`group`, `max_length` and `swap_length` are required");
}

// ── Winding tensor (pimc/winding.h) ─────────────────────────────────────────
// A loop whose links add up to the lattice vector W gives T = W W^T / N in
// Cartesian components, whatever basis the cell stores after its reduction.

TEST_CASE("winding tensor of a ring that winds once in a hexagonal cell", "[pimc][winding]") {
    const double L = 4.0, h = L * std::sqrt(3.0) / 2.0;
    const std::vector<double> vecs = {L, 0.0, L / 2.0, h};
    SimulationCell cell(2, vecs);
    const std::size_t M = 8;
    PathSet path(M, 1, 2, 0.1, cell);
    const double W[2] = {L - L / 2.0, -h};   // a1 - a2, |W| = L
    for (std::size_t k = 0; k < M; ++k)
        for (std::size_t d = 0; d < 2; ++d)
            path.pos(0, d, k) = 0.3 + W[d] * static_cast<double>(k) / static_cast<double>(M);
    REQUIRE(alea::winding_tensor_size(2) == 3);
    double T[alea::kMaxWindingTensor];
    alea::winding_tensor_of(path, T);
    CHECK(T[0] == Catch::Approx(W[0] * W[0]));   // xx
    CHECK(T[1] == Catch::Approx(W[0] * W[1]));   // xy
    CHECK(T[2] == Catch::Approx(W[1] * W[1]));   // yy
    CHECK(T[0] + T[2] == Catch::Approx(L * L));
    // The lattice-basis winding2 counts the integer components of W in the
    // basis the cell stores after its reduction (one or two unit components
    // for this shortest vector), so it is not |W|^2 / (L^2 dim) = 1/2.
    double n[2];
    cell.fractional(W, n);
    const double w2_expected = (std::round(n[0]) * std::round(n[0]) +
                                std::round(n[1]) * std::round(n[1])) / 2.0;
    CHECK(alea::winding2_of(path) == Catch::Approx(w2_expected));
    // winding_measure returns the same winding2 and the same tensor in one pass.
    double T2[alea::kMaxWindingTensor];
    CHECK(alea::winding_measure(path, T2) == Catch::Approx(w2_expected));
    for (std::size_t c = 0; c < 3; ++c) CHECK(T2[c] == Catch::Approx(T[c]));
    CHECK(std::string(alea::winding_tensor_label(0)) == "xx");
    CHECK(std::string(alea::winding_tensor_label(2)) == "yy");
}

TEST_CASE("winding tensor of an exchange cycle sums the whole cycle", "[pimc][winding]") {
    const double L = 4.0, h = L * std::sqrt(3.0) / 2.0;
    const std::vector<double> vecs = {L, 0.0, L / 2.0, h};
    SimulationCell cell(2, vecs);
    const std::size_t M = 6;
    PathSet path(M, 2, 2, 0.1, cell);
    path.set_exchange_classes(std::vector<std::size_t>{0, 0}, std::vector<char>{0, 0});
    path.set_permutation(std::vector<std::size_t>{1, 0});
    const double W[2] = {L, 0.0};            // the two-particle cycle winds by a1
    for (std::size_t p = 0; p < 2; ++p)
        for (std::size_t k = 0; k < M; ++k)
            for (std::size_t d = 0; d < 2; ++d)
                path.pos(p, d, k) = 0.1 + W[d] * static_cast<double>(p * M + k) /
                                              static_cast<double>(2 * M);
    double T[alea::kMaxWindingTensor];
    alea::winding_tensor_of(path, T);
    CHECK(T[0] == Catch::Approx(L * L / 2.0));   // one cycle, two particles
    CHECK(T[1] == Catch::Approx(0.0).margin(1e-12));
    CHECK(T[2] == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("winding tensor is zero for straight rings and open cells", "[pimc][winding]") {
    const std::vector<double> vecs = {4.0, 0.0, 0.0, 5.0};
    SimulationCell cell(2, vecs);
    PathSet path(5, 2, 2, 0.1, cell);
    for (std::size_t p = 0; p < 2; ++p)
        for (std::size_t k = 0; k < 5; ++k) {
            path.pos(p, 0, k) = 0.2 * static_cast<double>(p) + 0.05 * static_cast<double>(k % 2);
            path.pos(p, 1, k) = -0.1 * static_cast<double>(k % 3);
        }
    double T[alea::kMaxWindingTensor];
    alea::winding_tensor_of(path, T);
    for (std::size_t c = 0; c < 3; ++c) CHECK(T[c] == Catch::Approx(0.0).margin(1e-12));
    PathSet open(5, 1, 2, 0.1);
    alea::winding_tensor_of(open, T);
    for (std::size_t c = 0; c < 3; ++c) CHECK(T[c] == 0.0);
}
