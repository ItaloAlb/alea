// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <vector>

#include "hamiltonian/moire.h"
#include "numerics/special_functions.h"
#include "population/walker_set.h"

using alea::MoireTerm;
using alea::WalkerSet;
using alea::moire_reciprocal_vectors;
using Catch::Approx;

// ── Geometry helper: triangular first-shell reciprocal vectors ──────────────

TEST_CASE("moire_reciprocal_vectors gives 3 vectors of magnitude 4pi/(sqrt3 a_M)",
          "[moire][geometry]") {
    const double a_M = 10.0;
    const auto g = moire_reciprocal_vectors(a_M, /*theta0=*/0.0, /*dim=*/2);

    REQUIRE(g.size() == 3);
    const double expected_mag = 4.0 * alea::numerics::PI / (std::sqrt(3.0) * a_M);
    for (const auto& gi : g) {
        REQUIRE(gi.size() == 2);
        const double mag = std::sqrt(gi[0] * gi[0] + gi[1] * gi[1]);
        CHECK(mag == Approx(expected_mag));
    }
}

TEST_CASE("moire_reciprocal_vectors places the 3 vectors 120 deg apart from theta0",
          "[moire][geometry]") {
    const double a_M = 7.0;
    const double theta0 = 0.3;  // arbitrary orientation
    const auto g = moire_reciprocal_vectors(a_M, theta0, /*dim=*/2);

    const double mag = 4.0 * alea::numerics::PI / (std::sqrt(3.0) * a_M);
    const double third = 2.0 * alea::numerics::PI / 3.0;
    for (std::size_t i = 0; i < 3; ++i) {
        const double angle = theta0 + third * static_cast<double>(i);
        CHECK(g[i][0] == Approx(mag * std::cos(angle)));
        CHECK(g[i][1] == Approx(mag * std::sin(angle)));
    }
}

// ── MoireTerm::evaluate_batch — the one-body potential V(r) ──────────────────

TEST_CASE("MoireTerm adds V0 + 2*Vtilde*sum cos(g.r + phi) for one particle",
          "[moire]") {
    // One particle at r = (0.5, 0.25); explicit g vectors so the expectation is
    // self-contained.  V0=0.1, Vtilde=0.2, phi=0.
    WalkerSet ws(/*nw=*/1, /*np=*/1, /*dim=*/2);
    auto pos = ws.positions();
    pos[0] = 0.5;   // p0.x
    pos[1] = 0.25;  // p0.y

    const std::vector<std::vector<double>> g = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
    const double V0 = 0.1, Vtilde = 0.2, phi = 0.0;
    MoireTerm moire(/*dim=*/2, V0, Vtilde, phi, g, {{0, 1}});
    moire.evaluate_batch(ws);

    const double expected =
        V0 + 2.0 * Vtilde *
                 (std::cos(0.5 + phi) + std::cos(0.25 + phi) + std::cos(0.75 + phi));
    CHECK(ws.local_energies()[0] == Approx(expected));
}

TEST_CASE("MoireTerm adds V0 once per particle in the group (Vtilde=0)",
          "[moire]") {
    // 3 particles in one group, Vtilde=0 → modulation vanishes and each particle
    // contributes exactly V0, so the total is 3*V0.
    WalkerSet ws(/*nw=*/1, /*np=*/3, /*dim=*/2);
    // Positions are irrelevant when Vtilde=0; leave them at the origin.

    const std::vector<std::vector<double>> g = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
    const double V0 = 0.7;
    MoireTerm moire(/*dim=*/2, V0, /*Vtilde=*/0.0, /*phi=*/0.4, g, {{0, 3}});
    moire.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Approx(3.0 * V0));
}

TEST_CASE("MoireTerm acts only on the listed group ranges", "[moire]") {
    // 3 particles, but the term lists only groups [0,1) and [2,3): the middle
    // particle (index 1) is excluded, so with Vtilde=0 the total is 2*V0, not 3*V0.
    WalkerSet ws(/*nw=*/1, /*np=*/3, /*dim=*/2);

    const std::vector<std::vector<double>> g = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
    const double V0 = 0.5;
    MoireTerm moire(/*dim=*/2, V0, /*Vtilde=*/0.0, /*phi=*/0.0, g, {{0, 1}, {2, 3}});
    moire.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Approx(2.0 * V0));
}

TEST_CASE("MoireTerm evaluates each walker at its own position", "[moire][batch]") {
    // 2 walkers, 1 particle. SoA index = d*nw + w (p=0, nw=2).
    //   w=0: r=(0.5, 0.0)
    //   w=1: r=(1.0, 0.3)
    WalkerSet ws(/*nw=*/2, /*np=*/1, /*dim=*/2);
    auto pos = ws.positions();
    pos[0] = 0.5;  // w0.x
    pos[1] = 1.0;  // w1.x
    pos[2] = 0.0;  // w0.y
    pos[3] = 0.3;  // w1.y

    const std::vector<std::vector<double>> g = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
    const double V0 = 0.0, Vtilde = 0.5, phi = 0.2;
    MoireTerm moire(/*dim=*/2, V0, Vtilde, phi, g, {{0, 1}});
    moire.evaluate_batch(ws);

    auto V = [&](double x, double y) {
        return V0 + 2.0 * Vtilde *
                        (std::cos(x + phi) + std::cos(y + phi) + std::cos(x + y + phi));
    };
    CHECK(ws.local_energies()[0] == Approx(V(0.5, 0.0)));
    CHECK(ws.local_energies()[1] == Approx(V(1.0, 0.3)));
}

TEST_CASE("MoireTerm accumulates onto the existing local energy", "[moire]") {
    // The Composite Hamiltonian sums components: evaluate_batch must add, not
    // overwrite.  Seed le[0] and check the moiré contribution is added on top.
    WalkerSet ws(/*nw=*/1, /*np=*/1, /*dim=*/2);
    ws.local_energies()[0] = 3.0;  // pretend another term already ran

    const std::vector<std::vector<double>> g = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
    const double V0 = 0.25;
    MoireTerm moire(/*dim=*/2, V0, /*Vtilde=*/0.0, /*phi=*/0.0, g, {{0, 1}});
    moire.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Approx(3.0 + V0));
    CHECK(moire.name() == "moire");
}
