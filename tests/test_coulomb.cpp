// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <vector>

#include "population/particle_group.h"
#include "population/walker_set.h"
#include "hamiltonian/coulomb.h"

using alea::CoulombTerm;
using alea::ParticleGroup;
using alea::WalkerSet;

namespace {

ParticleGroup cgroup(std::size_t first, std::size_t last, double charge) {
    ParticleGroup g;
    g.name = "g";
    g.charge = charge;
    g.first = first;
    g.last = last;
    return g;
}

}  // namespace

// ── Original tests (kept as-is) ────────────────────────────────────────────

TEST_CASE("CoulombTerm reports its 1/r cusp residue q_a q_b/eps for the pair",
          "[coulomb][cusp]") {
    // R = lim_{r→0} r V(r) = q_a q_b / eps.  Used to set a Jastrow cusp Γ.
    const ParticleGroup ga = cgroup(0, 1, +1.0);
    const ParticleGroup gb = cgroup(1, 2, -1.0);
    CoulombTerm coul(ga, gb, /*epsilon=*/2.0);

    // The term's own pair (either order) → (+1)(−1)/2 = −0.5.
    CHECK(coul.coulomb_cusp_residue(ga, gb) == Catch::Approx(-0.5));
    CHECK(coul.coulomb_cusp_residue(gb, ga) == Catch::Approx(-0.5));

    // An unrelated pair → 0 (this term does not couple it).
    const ParticleGroup other = cgroup(5, 6, +1.0);
    CHECK(coul.coulomb_cusp_residue(ga, other) == Catch::Approx(0.0));
}

TEST_CASE("CoulombTerm accumulates inter-group energy q_a q_b/(eps r)",
          "[coulomb]") {
    // p0=(0,0) in group A (q=+1), p1=(2,0) in group B (q=-1), eps=1, r=2.
    // V = (+1)(-1) / (1 * 2) = -0.5.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;  // p1 x
    ws.dist_tables().build_all(ws.positions());

    CoulombTerm coul(cgroup(0, 1, +1.0), cgroup(1, 2, -1.0), /*epsilon=*/1.0);
    coul.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(-0.5));
}

TEST_CASE("CoulombTerm accumulates intra-group energy q^2/(eps r) once per pair",
          "[coulomb]") {
    // Two particles in one group [0,2), q=+1, eps=1, r=2.
    // Only the pair (0,1) contributes: V = (+1)^2 / (1 * 2) = 0.5.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());

    CoulombTerm coul(cgroup(0, 2, +1.0), cgroup(0, 2, +1.0), /*epsilon=*/1.0);
    coul.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(0.5));
}

TEST_CASE("CoulombTerm scales the energy by 1/epsilon", "[coulomb]") {
    // Inter-group q_a q_b = -1, r = 2, eps = 2 → V = -1 / (2 * 2) = -0.25.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());

    CoulombTerm coul(cgroup(0, 1, +1.0), cgroup(1, 2, -1.0), /*epsilon=*/2.0);
    coul.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(-0.25));
}

// ── Layer offset (bilayer e–h) ─────────────────────────────────────────────

TEST_CASE("CoulombTerm with a layer offset uses the 3D distance",
          "[coulomb][layer]") {
    // p0=(0,0) q=+1, p1=(3,0) q=-1 → in-plane r=3.  With layer offset d=4 the
    // 3D separation is sqrt(3^2+4^2)=5, so V = (+1)(-1)/(1*5) = -0.2 (not -1/3).
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 3.0;  // p1.x
    ws.dist_tables().build_all(ws.positions());

    CoulombTerm coul(cgroup(0, 1, +1.0), cgroup(1, 2, -1.0),
                     /*epsilon=*/1.0, /*d_layer=*/4.0);
    coul.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(-0.2));
}

TEST_CASE("CoulombTerm with a layer offset has no 1/r cusp residue",
          "[coulomb][layer][cusp]") {
    // V = q_a q_b / (eps sqrt(r^2 + d^2)) is finite at r=0 → lim_{r->0} r V = 0.
    const ParticleGroup ga = cgroup(0, 1, +1.0);
    const ParticleGroup gb = cgroup(1, 2, -1.0);
    CoulombTerm coul(ga, gb, /*epsilon=*/1.0, /*d_layer=*/0.5);
    CHECK(coul.coulomb_cusp_residue(ga, gb) == Catch::Approx(0.0));
}

// ── Charge-variation tests ─────────────────────────────────────────────────

TEST_CASE("CoulombTerm inter-group with non-unit charges", "[coulomb][charge]") {
    // p0=(0,0) in group A (q=+2), p1=(3,0) in group B (q=-3), eps=1, r=3.
    // V = (+2)(-3) / (1 * 3) = -6 / 3 = -2.0.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 3.0;  // p1.x
    ws.dist_tables().build_all(ws.positions());

    CoulombTerm coul(cgroup(0, 1, +2.0), cgroup(1, 2, -3.0), /*epsilon=*/1.0);
    coul.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(-2.0));
}

TEST_CASE("CoulombTerm intra-group with non-unit charge", "[coulomb][charge]") {
    // Two particles in one group [0,2), q=+3, eps=1, r=2.
    // Only the pair (0,1) contributes: V = (+3)^2 / (1 * 2) = 9 / 2 = 4.5.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());

    CoulombTerm coul(cgroup(0, 2, +3.0), cgroup(0, 2, +3.0), /*epsilon=*/1.0);
    coul.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(4.5));
}

TEST_CASE("CoulombTerm non-unit charges with non-unit epsilon",
          "[coulomb][charge]") {
    // p0=(0,0) in group A (q=+2), p1=(4,0) in group B (q=-3), eps=4, r=4.
    // V = (+2)(-3) / (4 * 4) = -6 / 16 = -0.375.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 4.0;
    ws.dist_tables().build_all(ws.positions());

    CoulombTerm coul(cgroup(0, 1, +2.0), cgroup(1, 2, -3.0), /*epsilon=*/4.0);
    coul.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(-0.375));
}

TEST_CASE("CoulombTerm inter-group with zero charge gives zero energy",
          "[coulomb][charge]") {
    // p0 in group A (q=0), p1 in group B (q=+5), r=2, eps=1.
    // V = 0 * 5 / 2 = 0.0.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());

    CoulombTerm coul(cgroup(0, 1, 0.0), cgroup(1, 2, +5.0), /*epsilon=*/1.0);
    coul.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(0.0));
}

TEST_CASE("CoulombTerm intra-group with zero charge gives zero energy",
          "[coulomb][charge]") {
    // Two particles in one group, q=0, r=2 → V = 0^2 / 2 = 0.0.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());

    CoulombTerm coul(cgroup(0, 2, 0.0), cgroup(0, 2, 0.0), /*epsilon=*/1.0);
    coul.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(0.0));
}

TEST_CASE("CoulombTerm inter-group with same-sign charges (repulsion)",
          "[coulomb][charge]") {
    // Both charges positive: repulsion → positive potential energy.
    // p0 in group A (q=+2), p1 in group B (q=+3), r=2, eps=1.
    // V = (+2)(+3) / (1 * 2) = +3.0.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());

    CoulombTerm coul(cgroup(0, 1, +2.0), cgroup(1, 2, +3.0), /*epsilon=*/1.0);
    coul.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(3.0));
}

TEST_CASE("CoulombTerm inter-group with both negative charges (repulsion)",
          "[coulomb][charge]") {
    // Both negative: repulsion → positive potential energy.
    // p0 in group A (q=-2), p1 in group B (q=-4), r=2, eps=1.
    // V = (-2)(-4) / (1 * 2) = +4.0.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());

    CoulombTerm coul(cgroup(0, 1, -2.0), cgroup(1, 2, -4.0), /*epsilon=*/1.0);
    coul.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(4.0));
}

TEST_CASE("CoulombTerm fractional charge values", "[coulomb][charge]") {
    // p0 in group A (q=+0.5), p1 in group B (q=-1.5), r=3, eps=1.
    // V = (+0.5)(-1.5) / (1 * 3) = -0.75 / 3 = -0.25.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 3.0;
    ws.dist_tables().build_all(ws.positions());

    CoulombTerm coul(cgroup(0, 1, +0.5), cgroup(1, 2, -1.5), /*epsilon=*/1.0);
    coul.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(-0.25));
}

// ── Multi-walker batch evaluation ──────────────────────────────────────────

TEST_CASE("CoulombTerm evaluates each walker independently in batch",
          "[coulomb][batch]") {
    // 4 walkers, 2 particles, 2D. Inter-group: group A [0,1) q=+1, group B [1,2) q=-1, eps=1.
    //
    // Walker positions (all p0 at origin):
    //   w=0: p1=(2,0)  → r=2  → V = -1/2 = -0.5
    //   w=1: p1=(4,0)  → r=4  → V = -1/4 = -0.25
    //   w=2: p1=(1,0)  → r=1  → V = -1/1 = -1.0
    //   w=3: p1=(0,3)  → r=3  → V = -1/3 ≈ -0.333333...
    const std::size_t nw = 4;
    WalkerSet ws(nw, /*np=*/2, /*dim=*/2);

    // SoA layout: pos[p*dim*nw + d*nw + w]
    // p0: indices 0..3 (x), 4..7 (y) — all zero, leave at default.
    // p1.x: indices 8..11
    auto pos = ws.positions();
    pos[8]  = 2.0;  // w=0, p1.x
    pos[9]  = 4.0;  // w=1, p1.x
    pos[10] = 1.0;  // w=2, p1.x
    // pos[11] stays 0  — w=3, p1.x
    pos[15] = 3.0;  // w=3, p1.y

    ws.dist_tables().build_all(ws.positions());

    CoulombTerm coul(cgroup(0, 1, +1.0), cgroup(1, 2, -1.0), /*epsilon=*/1.0);
    coul.evaluate_batch(ws);

    auto le = ws.local_energies();
    CHECK(le[0] == Catch::Approx(-0.5));
    CHECK(le[1] == Catch::Approx(-0.25));
    CHECK(le[2] == Catch::Approx(-1.0));
    CHECK(le[3] == Catch::Approx(-1.0 / 3.0));
}

TEST_CASE("CoulombTerm multi-walker intra-group batch",
          "[coulomb][batch]") {
    // 2 walkers, 2 particles in same group [0,2), q=+2, eps=1.
    //   w=0: p1=(4,0) → r=4 → V = (+2)^2/4 = 4/4 = 1.0
    //   w=1: p1=(1,0) → r=1 → V = (+2)^2/1 = 4.0
    const std::size_t nw = 2;
    WalkerSet ws(nw, /*np=*/2, /*dim=*/2);

    auto pos = ws.positions();
    // SoA: p1.x at indices p*dim*nw + d*nw + w = 1*2*2 + 0*2 + w = 4 + w
    pos[4] = 4.0;  // w=0, p1.x
    pos[5] = 1.0;  // w=1, p1.x

    ws.dist_tables().build_all(ws.positions());

    CoulombTerm coul(cgroup(0, 2, +2.0), cgroup(0, 2, +2.0), /*epsilon=*/1.0);
    coul.evaluate_batch(ws);

    auto le = ws.local_energies();
    CHECK(le[0] == Catch::Approx(1.0));
    CHECK(le[1] == Catch::Approx(4.0));
}

// ── Multi-particle inter-group (sum over pairs) ────────────────────────────

TEST_CASE("CoulombTerm inter-group sums over all cross-pairs",
          "[coulomb][multi_particle]") {
    // 3 particles: p0, p1 in group A [0,2), p2 in group B [2,3).
    // q_a=+1, q_b=-1, eps=1.
    // Positions: p0=(0,0), p1=(0,3), p2=(4,0).
    // Pairs: (p0,p2): r=4,  (p1,p2): r = sqrt(4²+3²) = 5.
    // V = (+1)(-1)/4 + (+1)(-1)/5 = -0.25 - 0.2 = -0.45.
    WalkerSet ws(/*nw=*/1, /*np=*/3, /*dim=*/2);
    auto pos = ws.positions();
    // SoA: pos[p*dim*nw + d*nw + w], nw=1
    // p0: pos[0]=0 (x), pos[1]=0 (y) — leave at default
    // p1: pos[2]=0 (x), pos[3]=3 (y)
    pos[3] = 3.0;  // p1.y
    // p2: pos[4]=4 (x), pos[5]=0 (y)
    pos[4] = 4.0;  // p2.x

    ws.dist_tables().build_all(ws.positions());

    CoulombTerm coul(cgroup(0, 2, +1.0), cgroup(2, 3, -1.0), /*epsilon=*/1.0);
    coul.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(-0.45));
}

TEST_CASE("CoulombTerm multi-particle inter-group with non-unit charges",
          "[coulomb][multi_particle][charge]") {
    // 3 particles: 2 in group A [0,2) q=+2, 1 in group B [2,3) q=-3, eps=2.
    // Positions: p0=(0,0), p1=(3,0), p2=(0,4).
    // r_02 = sqrt(0²+4²) = 4.  r_12 = sqrt(3²+4²) = 5.
    // V = (+2)(-3)/(2*4) + (+2)(-3)/(2*5) = -6/8 - 6/10 = -0.75 - 0.6 = -1.35.
    WalkerSet ws(/*nw=*/1, /*np=*/3, /*dim=*/2);
    auto pos = ws.positions();
    // p0: (0,0) — default
    // p1: (3,0)
    pos[2] = 3.0;  // p1.x
    // p2: (0,4)
    pos[5] = 4.0;  // p2.y

    ws.dist_tables().build_all(ws.positions());

    CoulombTerm coul(cgroup(0, 2, +2.0), cgroup(2, 3, -3.0), /*epsilon=*/2.0);
    coul.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(-1.35));
}

// ── Identical intra-group particles with multiple pairs ────────────────────

TEST_CASE("CoulombTerm intra-group with 3 particles sums over all N(N-1)/2 pairs",
          "[coulomb][multi_particle]") {
    // 3 particles in one group [0,3), q=+1, eps=1.
    // Positions: p0=(0,0), p1=(3,0), p2=(0,4).
    // Pairs: (0,1): r=3  → 1²/3 = 1/3
    //        (0,2): r=4  → 1²/4 = 0.25
    //        (1,2): r=5  → 1²/5 = 0.2
    // V = 1/3 + 1/4 + 1/5 ≈ 0.333333 + 0.25 + 0.2 = 0.783333...
    WalkerSet ws(/*nw=*/1, /*np=*/3, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 3.0;  // p1.x
    pos[5] = 4.0;  // p2.y

    ws.dist_tables().build_all(ws.positions());

    CoulombTerm coul(cgroup(0, 3, +1.0), cgroup(0, 3, +1.0), /*epsilon=*/1.0);
    coul.evaluate_batch(ws);

    const double expected = 1.0 / 3.0 + 1.0 / 4.0 + 1.0 / 5.0;
    CHECK(ws.local_energies()[0] == Catch::Approx(expected));
}

TEST_CASE("CoulombTerm intra-group 3 particles with non-unit charge",
          "[coulomb][multi_particle][charge]") {
    // 3 particles in one group, q=+2, eps=1.
    // Same positions: r=3,4,5.
    // V = 4/3 + 4/4 + 4/5 = 4*(1/3 + 1/4 + 1/5) ≈ 4*0.783333 = 3.133333...
    WalkerSet ws(/*nw=*/1, /*np=*/3, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 3.0;  // p1.x
    pos[5] = 4.0;  // p2.y

    ws.dist_tables().build_all(ws.positions());

    CoulombTerm coul(cgroup(0, 3, +2.0), cgroup(0, 3, +2.0), /*epsilon=*/1.0);
    coul.evaluate_batch(ws);

    const double expected = 4.0 * (1.0 / 3.0 + 1.0 / 4.0 + 1.0 / 5.0);
    CHECK(ws.local_energies()[0] == Catch::Approx(expected));
}
