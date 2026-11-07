// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <vector>

#include "population/particle_group.h"
#include "population/walker_set.h"
#include "hamiltonian/rytova_keldysh.h"
#include "numerics/special_functions.h"
#include "numerics/keldysh_kernel.h"

using alea::KeldyshTerm;
using alea::ParticleGroup;
using alea::WalkerSet;
using alea::numerics::PI;
using alea::numerics::keldysh_kernel;
using alea::numerics::keldysh_cheb;

namespace {

ParticleGroup kgroup(std::size_t first, std::size_t last, double charge) {
    ParticleGroup g;
    g.name = "g";
    g.charge = charge;
    g.first = first;
    g.last = last;
    return g;
}

// Reference Rytova–Keldysh pair energy, straight from the definition
// V_ab(r) = (π q_a q_b)/(2 ε r₀) · [H₀(r/r₀) − Y₀(r/r₀)], using the
// high-precision scalar kernel (the same reference the Chebyshev term is fit
// to).  Validates the *term's assembly* (prefactor, charge product, r₀ scaling,
// pair summation) up to the Chebyshev approximation error (~5e-9).
double Vrk(double qa, double qb, double eps, double r0, double r) {
    const double x = r / r0;
    return (PI * qa * qb / (2.0 * eps * r0)) * keldysh_kernel(x);
}

}  // namespace

// ── Layer offset (bilayer e–h) ─────────────────────────────────────────────

TEST_CASE("KeldyshTerm with a layer offset uses the 3D distance",
          "[keldysh][layer]") {
    // p0=(0,0) q=+1, p1=(3,0) q=-1 → in-plane r=3.  With layer offset d=4 the
    // pair feels the screened potential at the 3D distance sqrt(3^2+4^2)=5.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 3.0;  // p1.x
    ws.dist_tables().build_all(ws.positions());

    KeldyshTerm kt(kgroup(0, 1, +1.0), kgroup(1, 2, -1.0), /*epsilon=*/1.0,
                   /*r0=*/2.0, /*d_layer=*/4.0);
    kt.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(Vrk(+1, -1, 1.0, 2.0, 5.0)));
}

// ── Single-pair energy ─────────────────────────────────────────────────────

TEST_CASE("KeldyshTerm inter-group single pair matches the definition",
          "[keldysh]") {
    // p0=(0,0) group A (q=+1), p1=(2,0) group B (q=-1), eps=1, r0=1, r=2.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;  // p1.x
    ws.dist_tables().build_all(ws.positions());

    KeldyshTerm kt(kgroup(0, 1, +1.0), kgroup(1, 2, -1.0), /*epsilon=*/1.0,
                   /*r0=*/1.0);
    kt.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(Vrk(+1, -1, 1.0, 1.0, 2.0)));
    // Sanity: opposite charges → attractive (negative).
    CHECK(ws.local_energies()[0] < 0.0);
}

TEST_CASE("KeldyshTerm intra-group counts each pair once", "[keldysh]") {
    // Two particles in one group [0,2), q=+1, eps=1, r0=1, r=2.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());

    KeldyshTerm kt(kgroup(0, 2, +1.0), kgroup(0, 2, +1.0), /*epsilon=*/1.0,
                   /*r0=*/1.0);
    kt.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(Vrk(+1, +1, 1.0, 1.0, 2.0)));
}

TEST_CASE("KeldyshTerm scales the energy by 1/epsilon", "[keldysh]") {
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());

    KeldyshTerm kt(kgroup(0, 1, +1.0), kgroup(1, 2, -1.0), /*epsilon=*/2.0,
                   /*r0=*/1.0);
    kt.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(Vrk(+1, -1, 2.0, 1.0, 2.0)));
}

TEST_CASE("KeldyshTerm r0 rescales both prefactor and argument", "[keldysh]") {
    // r0 = 0.5 enters as π/(2 ε r₀) and through the argument r/r₀ = 4.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());

    KeldyshTerm kt(kgroup(0, 1, +1.0), kgroup(1, 2, -1.0), /*epsilon=*/1.0,
                   /*r0=*/0.5);
    kt.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(Vrk(+1, -1, 1.0, 0.5, 2.0)));
}

TEST_CASE("KeldyshTerm same-sign charges give repulsion", "[keldysh][charge]") {
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());

    KeldyshTerm kt(kgroup(0, 1, +2.0), kgroup(1, 2, +3.0), /*epsilon=*/1.0,
                   /*r0=*/1.0);
    kt.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(Vrk(+2, +3, 1.0, 1.0, 2.0)));
    CHECK(ws.local_energies()[0] > 0.0);  // repulsive
}

// ── Multi-walker batch ─────────────────────────────────────────────────────

TEST_CASE("KeldyshTerm evaluates each walker independently in batch",
          "[keldysh][batch]") {
    // 4 walkers, inter-group A[0,1) q=+1, B[1,2) q=-1, eps=1, r0=1.
    //   w=0: r=2,  w=1: r=4,  w=2: r=1,  w=3: r=3.
    const std::size_t nw = 4;
    WalkerSet ws(nw, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[8]  = 2.0;  // w=0, p1.x
    pos[9]  = 4.0;  // w=1, p1.x
    pos[10] = 1.0;  // w=2, p1.x
    pos[15] = 3.0;  // w=3, p1.y  (p1.x stays 0)
    ws.dist_tables().build_all(ws.positions());

    KeldyshTerm kt(kgroup(0, 1, +1.0), kgroup(1, 2, -1.0), /*epsilon=*/1.0,
                   /*r0=*/1.0);
    kt.evaluate_batch(ws);

    auto le = ws.local_energies();
    CHECK(le[0] == Catch::Approx(Vrk(+1, -1, 1.0, 1.0, 2.0)));
    CHECK(le[1] == Catch::Approx(Vrk(+1, -1, 1.0, 1.0, 4.0)));
    CHECK(le[2] == Catch::Approx(Vrk(+1, -1, 1.0, 1.0, 1.0)));
    CHECK(le[3] == Catch::Approx(Vrk(+1, -1, 1.0, 1.0, 3.0)));
}

// ── Multi-particle sum over pairs ──────────────────────────────────────────

TEST_CASE("KeldyshTerm inter-group sums over all cross-pairs",
          "[keldysh][multi_particle]") {
    // p0, p1 in A[0,2), p2 in B[2,3). p0=(0,0), p1=(0,3), p2=(4,0).
    // Pairs (p0,p2): r=4, (p1,p2): r=5.
    WalkerSet ws(/*nw=*/1, /*np=*/3, /*dim=*/2);
    auto pos = ws.positions();
    pos[3] = 3.0;  // p1.y
    pos[4] = 4.0;  // p2.x
    ws.dist_tables().build_all(ws.positions());

    KeldyshTerm kt(kgroup(0, 2, +1.0), kgroup(2, 3, -1.0), /*epsilon=*/1.0,
                   /*r0=*/1.0);
    kt.evaluate_batch(ws);

    const double expected =
        Vrk(+1, -1, 1.0, 1.0, 4.0) + Vrk(+1, -1, 1.0, 1.0, 5.0);
    CHECK(ws.local_energies()[0] == Catch::Approx(expected));
}

// ── Physical limit: large r/r₀ recovers Coulomb ────────────────────────────

TEST_CASE("KeldyshTerm approaches Coulomb q_a q_b/(eps r) for r >> r0",
          "[keldysh][physics]") {
    // For r/r₀ ≫ 1 the kernel H₀ − Y₀ → 2/(π x), so V → q_a q_b/(ε r).
    // Independent of the special-function internals.
    const double r = 60.0, r0 = 1.0, eps = 1.0;
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = r;  // p1.x
    ws.dist_tables().build_all(ws.positions());

    KeldyshTerm kt(kgroup(0, 1, +1.0), kgroup(1, 2, -1.0), eps, r0);
    kt.evaluate_batch(ws);

    const double coulomb = (+1.0 * -1.0) / (eps * r);
    // Relative agreement improves like r₀/r; at r/r₀ = 60 it is well under 1%.
    CHECK(ws.local_energies()[0] == Catch::Approx(coulomb).epsilon(0.01));
}

// ── Chebyshev kernel accuracy vs the high-precision scalar reference ─────────

TEST_CASE("Keldysh Chebyshev kernel matches the reference over [x_lo, x_hi]",
          "[keldysh][chebyshev]") {
    // The vectorized term evaluates K(x) = H₀(x) − Y₀(x) via a Chebyshev
    // expansion in t = x^(1/8); it must track the scalar reference to the
    // calibrated tolerance (~5e-9) across the whole tabulated band.
    // Sweep the Chebyshev band [x_lo, x_hi); x == x_hi already belongs to the
    // asymptotic branch (covered by the next test).
    const auto& cheb = keldysh_cheb();
    double max_err = 0.0;
    const int n = 100000;
    for (int i = 0; i < n; ++i) {
        const double x = cheb.x_lo + (cheb.x_hi - cheb.x_lo) * i / n;
        max_err = std::max(max_err, std::abs(cheb.eval(x) - keldysh_kernel(x)));
    }
    INFO("max |cheb - ref| = " << max_err);
    CHECK(max_err < 1.0e-8);
}

TEST_CASE("Keldysh Chebyshev kernel uses the exact asymptotic past x_hi",
          "[keldysh][chebyshev]") {
    // For x ≥ x_hi the kernel uses the six-term DLMF tail. Retain the older
    // two-term accuracy budget here; the independent continuity test adds a
    // tighter integral reference for the improved tail.
    const auto& cheb = keldysh_cheb();
    for (double x : {50.0, 80.0, 150.0}) {
        const double rel = std::abs(cheb.eval(x) - keldysh_kernel(x)) /
                           std::abs(keldysh_kernel(x));
        INFO("x = " << x << "  rel err = " << rel);
        CHECK(rel < 1.0e-5);
    }
}
