// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

#include "population/particle_group.h"
#include "population/walker_set.h"
#include "hamiltonian/hamiltonian_component.h"
#include "hamiltonian/coulomb.h"
#include "hamiltonian/rytova_keldysh.h"
#include "hamiltonian/screened_coulomb.h"

#include "fast_screened_coulomb.h"  // library C API, for exact references

using alea::CoulombTerm;
using alea::HamiltonianComponent;
using alea::KeldyshTerm;
using alea::ParticleGroup;
using alea::ScreenedCoulombTerm;
using alea::screened_coulomb_stack_issue;

TEST_CASE("Screened fit enforces separate sampled value and force targets and preserves the radial domain", "[fsc-diagnostics]") {
    const std::vector eps{1.5,1.5,1.5,1.5},interfaces{-1.,1.,3.};
    const auto fitted=alea::build_screened_fit(eps,interfaces,2,3,2.,.01);
    std::unique_ptr<FSCContext,decltype(&fsc_free)> ctx(fsc_create(4,eps.data(),interfaces.data(),2,3,2.),&fsc_free);
    REQUIRE(ctx);
    double k_max,rho_max; int n_quad;
    fsc_optimal_params(ctx.get(),1e-4,1e-3,1.5,&k_max,&n_quad,&rho_max);
    CHECK(fitted.requested_error_pct==.01);
    CHECK(fitted.estimated_error_pct==0.);
    CHECK(fitted.estimated_force_error_pct==0.);
    CHECK(fitted.rho_max==rho_max);
    for (double rho : {.01,.0101,.05,.5,1.,10.,rho_max}) {
        const double radius=std::hypot(rho,2.);
        CHECK(fsc_controlled_eval(fitted.fit.get(),rho)==Catch::Approx(2/(1.5*radius)).epsilon(2e-15));
        CHECK(fsc_controlled_d1(fitted.fit.get(),rho)==Catch::Approx(-2*rho/(1.5*radius*radius*radius)).epsilon(2e-15));
    }
    const auto loose=alea::build_screened_fit({1.5},{},1,0,0.,100.);
    CHECK(loose.approximation_notes.find("enforced_value_force_target_pct=5")!=std::string::npos);
}

TEST_CASE("Screened fit rejects invalid percent targets before expensive stack work", "[fsc-diagnostics]") {
    for (double target : {0.,-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})
        CHECK_THROWS_AS(alea::build_screened_fit({1.5},{},1,0,0.,target),std::invalid_argument);
}
using alea::WalkerSet;

namespace {

ParticleGroup sgroup(std::size_t first, std::size_t last, double charge) {
    ParticleGroup g;
    g.name   = "g";
    g.charge = charge;
    g.first  = first;
    g.last   = last;
    return g;
}

// Place particle 1 at in-plane distance r from particle 0 (p0 at the origin) in a
// single-walker, two-particle, 2D set.  Positions are SoA: [p*dim*nw + d*nw + w],
// so for nw=1,np=2,dim=2 the index of p1.x is 2.
void put_pair_at(WalkerSet& ws, double r) {
    ws.positions()[2] = r;
    ws.dist_tables().build_all(ws.positions());
}

// Energy a two-group term assigns to a single e–h pair separated by `rho`.
double pair_energy(HamiltonianComponent& term, double rho) {
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);  // local_energies start at 0
    put_pair_at(ws, rho);
    term.evaluate_batch(ws);
    return ws.local_energies()[0];
}

}  // namespace

// ── Homogeneous (N=1) stack must reduce to the bare Coulomb term ────────────
//
// The library's direct-exciton path uses asymptotic subtraction, so a single
// layer returns exactly V = 2/(ε ρ) [Ry]; the ½ Ry→Hartree factor then gives
// 1/(ε ρ), the bare Coulomb tail.  This validates the ½ factor + charge product
// independently of any fit accuracy.

TEST_CASE("ScreenedCoulombTerm homogeneous stack equals Coulomb (eps=1)",
          "[screened][coulomb]") {
    const double r = 2.0, eps = 1.0;

    WalkerSet ws_s(/*nw=*/1, /*np=*/2, /*dim=*/2);
    put_pair_at(ws_s, r);
    ScreenedCoulombTerm st(sgroup(0, 1, +1.0), sgroup(1, 2, -1.0),
                           /*epsilons=*/{eps}, /*interfaces=*/{},
                           /*source_layer=*/1, /*target_layer=*/0,
                           /*z_offset=*/0.0, /*fit_tol=*/0.1);
    st.evaluate_batch(ws_s);

    CHECK(ws_s.local_energies()[0] == Catch::Approx(-1.0 / (eps * r)).epsilon(1e-3));
    CHECK(ws_s.local_energies()[0] < 0.0);  // opposite charges attract

    // …and it must agree with CoulombTerm on the same geometry.
    WalkerSet ws_c(/*nw=*/1, /*np=*/2, /*dim=*/2);
    put_pair_at(ws_c, r);
    CoulombTerm ct(sgroup(0, 1, +1.0), sgroup(1, 2, -1.0), eps);
    ct.evaluate_batch(ws_c);
    CHECK(ws_s.local_energies()[0] ==
          Catch::Approx(ws_c.local_energies()[0]).epsilon(1e-3));
}

TEST_CASE("ScreenedCoulombTerm homogeneous scales like 1/(eps r)",
          "[screened][coulomb]") {
    const double r = 3.0, eps = 5.0;
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    put_pair_at(ws, r);
    ScreenedCoulombTerm st(sgroup(0, 1, +1.0), sgroup(1, 2, -1.0), {eps}, {}, 1, 0,
                           0.0, 0.1);
    st.evaluate_batch(ws);
    CHECK(ws.local_energies()[0] == Catch::Approx(-1.0 / (eps * r)).epsilon(1e-3));
}

TEST_CASE("ScreenedCoulombTerm same-sign charges repel (intra-group)",
          "[screened][charge]") {
    const double r = 2.0, eps = 1.0;
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    put_pair_at(ws, r);
    // One group of two like charges → a single i<j pair, counted once.
    ScreenedCoulombTerm st(sgroup(0, 2, +1.0), sgroup(0, 2, +1.0), {eps}, {}, 1, 0,
                           0.0, 0.1);
    st.evaluate_batch(ws);
    CHECK(ws.local_energies()[0] == Catch::Approx(+1.0 / (eps * r)).epsilon(1e-3));
    CHECK(ws.local_energies()[0] > 0.0);
}

// ── Multi-walker batch: each walker evaluated independently ─────────────────

TEST_CASE("ScreenedCoulombTerm evaluates each walker independently",
          "[screened][batch]") {
    const double eps = 1.0;
    const std::size_t nw = 3;
    WalkerSet ws(nw, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    // p1.x of walker w lives at index 1*dim*nw + 0*nw + w = 6 + w.
    pos[6] = 2.0;
    pos[7] = 4.0;
    pos[8] = 1.0;
    ws.dist_tables().build_all(ws.positions());

    ScreenedCoulombTerm st(sgroup(0, 1, +1.0), sgroup(1, 2, -1.0), {eps}, {}, 1, 0,
                           0.0, 0.1);
    st.evaluate_batch(ws);

    auto le = ws.local_energies();
    CHECK(le[0] == Catch::Approx(-1.0 / (eps * 2.0)).epsilon(1e-3));
    CHECK(le[1] == Catch::Approx(-1.0 / (eps * 4.0)).epsilon(1e-3));
    CHECK(le[2] == Catch::Approx(-1.0 / (eps * 1.0)).epsilon(1e-3));
}

// ── Genuine multilayer stack: the fit must track the exact ETM potential ────

TEST_CASE("ScreenedCoulombTerm multilayer matches the exact ETM potential",
          "[screened][multilayer]") {
    // Bilayer MoS2 between vacuum (the library's documented example): the source
    // layer (2) is centred at the origin and spans z∈[-3,3]; direct exciton.
    std::vector<double> eps = {1.0, 14.0, 14.0, 1.0};
    std::vector<double> itf = {-3.0, 3.0, 9.0};
    const int    src = 2, tgt = 0;  // direct: t = c = 2
    const double fit_tol = 0.5;     // %

    // Reference: the library's own exact integral, with the same auto-tuned
    // bounds the term uses internally (fsc_optimal_params defaults).
    FSCContext* ctx = fsc_create(static_cast<int>(eps.size()), eps.data(),
                                 itf.data(), src, tgt, 0.0);
    REQUIRE(ctx != nullptr);
    double k_max = 0.0, rho_max = 0.0;
    int    n_quad = 0;
    fsc_optimal_params(ctx, 1e-4, 1e-3, 1.5, &k_max, &n_quad, &rho_max);

    ScreenedCoulombTerm st(sgroup(0, 1, +1.0), sgroup(1, 2, -1.0), eps, itf, src,
                           tgt, 0.0, fit_tol);

    for (double r : {1.0, 3.0, 7.0}) {
        WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
        put_pair_at(ws, r);
        st.evaluate_batch(ws);

        const double v_exact  = fsc_potential(ctx, r, k_max, 1e-8);  // Ry
        const double expected = 0.5 * (+1.0 * -1.0) * v_exact;       // ½ q_a q_b
        INFO("r = " << r);
        CHECK(ws.local_energies()[0] == Catch::Approx(expected).epsilon(0.02));
        CHECK(ws.local_energies()[0] < 0.0);  // e–h attraction
    }
    fsc_free(ctx);
}

// ── clone() shares the immutable fit and reproduces the energy exactly ──────

TEST_CASE("ScreenedCoulombTerm clone reproduces the same energy",
          "[screened][clone]") {
    const double r = 2.5, eps = 3.0;
    ScreenedCoulombTerm st(sgroup(0, 1, +1.0), sgroup(1, 2, -1.0), {eps}, {}, 1, 0,
                           0.0, 0.1);
    auto clone = st.clone();
    CHECK(clone->name() == "screened_coulomb");

    WalkerSet ws1(/*nw=*/1, /*np=*/2, /*dim=*/2);
    put_pair_at(ws1, r);
    WalkerSet ws2(/*nw=*/1, /*np=*/2, /*dim=*/2);
    put_pair_at(ws2, r);
    st.evaluate_batch(ws1);
    clone->evaluate_batch(ws2);
    CHECK(ws1.local_energies()[0] == Catch::Approx(ws2.local_energies()[0]));
}

// ── Degenerate-stack detection ──────────────────────────────────────────────
//
// `screened_coulomb_stack_issue` is the cheap, pure geometric gate; the term's
// constructor also has a k_max backstop for the subtler cases.  The verdicts below
// were validated directly against the library (configs that pin fsc_optimal_params'
// k_max at ~1.6e4 hang the fit).

TEST_CASE("screened_coulomb_stack_issue accepts well-conditioned stacks",
          "[screened][validate]") {
    CHECK(screened_coulomb_stack_issue({1.0}, {}, 1).empty());            // homogeneous
    CHECK(screened_coulomb_stack_issue({1, 14, 14, 1}, {-3, 3, 9}, 2).empty());
    CHECK(screened_coulomb_stack_issue({1, 10, 1}, {-1.5, 1.5}, 2).empty());
}

TEST_CASE("screened_coulomb_stack_issue flags degenerate stacks",
          "[screened][validate]") {
    // Semi-infinite (boundary) source layer — degenerate for any geometry.
    CHECK_FALSE(screened_coulomb_stack_issue({2, 10}, {-3.0}, 2).empty());          // c = N
    CHECK_FALSE(screened_coulomb_stack_issue({10, 2}, {3.0}, 1).empty());           // c = 1
    CHECK_FALSE(screened_coulomb_stack_issue({1, 14, 14, 1}, {-3, 3, 9}, 1).empty());
    CHECK_FALSE(screened_coulomb_stack_issue({1, 14, 14, 1}, {-3, 3, 9}, 4).empty());
    // Interior source but the origin is on / outside the layer.
    CHECK_FALSE(screened_coulomb_stack_issue({1, 10, 1}, {0.0, 3.0}, 2).empty());   // on lower interface
    CHECK_FALSE(screened_coulomb_stack_issue({1, 10, 1}, {1.0, 3.0}, 2).empty());   // origin below the layer
    // Basic validity.
    CHECK_FALSE(screened_coulomb_stack_issue({1, -3, 1}, {-3, 3}, 2).empty());      // non-positive ε
    CHECK_FALSE(screened_coulomb_stack_issue({1, 10, 1}, {3, -3}, 2).empty());      // non-monotonic
    CHECK_FALSE(screened_coulomb_stack_issue({1, 10, 1}, {0.0}, 2).empty());        // wrong interface count
    CHECK_FALSE(screened_coulomb_stack_issue({1, 10, 1}, {-1, 1}, 5).empty());      // source_layer out of range
    CHECK_FALSE(screened_coulomb_stack_issue({1e308,1e308,1e307},{-1,1},2).empty());
    CHECK_FALSE(screened_coulomb_stack_issue({1.,std::numeric_limits<double>::infinity(),1.},{-1,1},2).empty());
    CHECK_FALSE(screened_coulomb_stack_issue({1.,4.,1.},{-1,std::numeric_limits<double>::infinity()},2).empty());
}

TEST_CASE("ScreenedCoulombTerm throws (not hangs) on a degenerate stack",
          "[screened][validate]") {
    // The original hang case: a 2-layer stack whose source is the top (semi-infinite)
    // layer.  Must fail fast via the geometric gate, before any library call.
    CHECK_THROWS_AS(ScreenedCoulombTerm(sgroup(0, 1, +1.0), sgroup(1, 2, -1.0),
                                        {2.0, 10.0}, {-3.0}, /*source_layer=*/2,
                                        /*target_layer=*/0, /*z_offset=*/0.0,
                                        /*fit_tol=*/0.5),
                    std::invalid_argument);
}

TEST_CASE("ScreenedCoulombTerm k_max backstop catches a near-degenerate interior stack",
          "[screened][validate]") {
    // Interior source, origin strictly inside (passes the geometric gate) but only
    // 1e-4 Bohr from an interface: ε(k) reaches ε_c only near k ~ 1/δ ~ 1e4, so
    // fsc_optimal_params runs k_max to its cap.  The backstop must throw rather than
    // let fsc_fit_auto stall (fsc_optimal_params itself returns quickly).
    //
    // (This used to be 0.01 Bohr.  That stack only LOOKED degenerate: the old
    // transfer-matrix ε(k) lost all precision near k ~ 100 and never converged.
    // With the stable reflection-coefficient ε(k) it converges at k ~ 500 and the
    // fit completes -- in about a minute, so it is no test of the backstop.)
    CHECK(screened_coulomb_stack_issue({1.0, 10.0, 1.0}, {-1e-4, 3.0}, 2).empty());
    CHECK_THROWS(ScreenedCoulombTerm(sgroup(0, 1, +1.0), sgroup(1, 2, -1.0),
                                     {1.0, 10.0, 1.0}, {-1e-4, 3.0}, /*source_layer=*/2,
                                     /*target_layer=*/0, /*z_offset=*/0.0,
                                     /*fit_tol=*/0.5));
}

// ── Physics: relation to Rytova–Keldysh ─────────────────────────────────────
//
// Rytova–Keldysh is the ZERO-THICKNESS, infinite-contrast limit of the screened
// potential: a 2D polarizable sheet with a single screening length r0 = ε·t/2.  A
// finite slab [1, ε, 1] of thickness t evaluated by the ETM `screened_coulomb`
// should therefore (i) reproduce KeldyshTerm(r0 = ε·t/2) when the slab is thin and
// high-contrast, and (ii) deviate at short range (ρ ≲ t) for a physically thick
// slab, where it recovers the bulk-Coulomb core 1/(ε·ρ) that the sheet model lacks.
// Both regimes were mapped against the library before being asserted here.

TEST_CASE("ScreenedCoulombTerm reproduces Rytova-Keldysh in the thin, high-contrast limit",
          "[screened][rk][physics]") {
    // eps=160, t=0.5 → r0 = ε·t/2 = 40 Bohr.  The slab is ~zero-thickness, so the
    // finite-slab potential and the 2D RK sheet coincide to the fit tolerance.
    const double eps_slab = 160.0, t = 0.5, r0 = eps_slab * t / 2.0;

    ScreenedCoulombTerm st(sgroup(0, 1, +1.0), sgroup(1, 2, -1.0),
                           {1.0, eps_slab, 1.0}, {-t / 2.0, t / 2.0},
                           /*source_layer=*/2, /*target_layer=*/0,
                           /*z_offset=*/0.0, /*fit_tol=*/0.1);
    KeldyshTerm kt(sgroup(0, 1, +1.0), sgroup(1, 2, -1.0), /*epsilon=*/1.0, r0);

    for (double rho : {1.0, 2.0, 5.0, 10.0, 20.0}) {
        const double vs = pair_energy(st, rho);
        const double vk = pair_energy(kt, rho);
        INFO("rho = " << rho << "  V_screened = " << vs << "  V_RK = " << vk);
        CHECK(vs < 0.0);  // e–h attraction
        CHECK(vs == Catch::Approx(vk).epsilon(0.02));  // agree to ~2%
    }
}

TEST_CASE("ScreenedCoulombTerm diverges from Rytova-Keldysh at short range for a thick slab",
          "[screened][rk][physics]") {
    // eps=4, t=20 → same r0 = 40 Bohr, but the slab is thick: at ρ ≲ t the finite
    // slab feels its bulk dielectric (V → 1/(ε·ρ)), far more attractive than the RK
    // sheet, while the two reconverge at large ρ.
    const double eps_slab = 4.0, t = 20.0, r0 = eps_slab * t / 2.0;

    ScreenedCoulombTerm st(sgroup(0, 1, +1.0), sgroup(1, 2, -1.0),
                           {1.0, eps_slab, 1.0}, {-t / 2.0, t / 2.0},
                           /*source_layer=*/2, /*target_layer=*/0,
                           /*z_offset=*/0.0, /*fit_tol=*/0.1);
    KeldyshTerm kt(sgroup(0, 1, +1.0), sgroup(1, 2, -1.0), /*epsilon=*/1.0, r0);

    SECTION("short range: screened is much more attractive than RK") {
        const double rho = 1.0;
        const double vs = pair_energy(st, rho);
        const double vk = pair_energy(kt, rho);
        INFO("rho = " << rho << "  V_screened = " << vs << "  V_RK = " << vk);
        CHECK(vs < vk);                                       // strictly more bound
        CHECK(std::abs(vs) > 1.5 * std::abs(vk));             // ~2.85× in the probe
        // …and it approaches the bulk-Coulomb core 1/(ε·ρ) that RK has no notion of.
        CHECK(std::abs(vs) == Catch::Approx(1.0 / (eps_slab * rho)).epsilon(0.3));
    }
    SECTION("long range: the two reconverge") {
        const double rho = 50.0;
        const double vs = pair_energy(st, rho);
        const double vk = pair_energy(kt, rho);
        INFO("rho = " << rho << "  V_screened = " << vs << "  V_RK = " << vk);
        CHECK(vs == Catch::Approx(vk).epsilon(0.05));
    }
}

// ── Kato cusp residue: intralayer diverges (cuspful), interlayer is finite ───
//
// coulomb_cusp_residue = q_a q_b · lim_{ρ→0} ρ·V(ρ).  An INTRALAYER (direct) screened
// pair keeps the bulk-Coulomb core V→1/(ε_source·ρ) ⇒ residue = q_a q_b/ε_source (a
// repulsive cusp the e–e/h–h Jastrow must cancel).  An INTERLAYER pair is finite at the
// origin (the baked layer offset) ⇒ residue 0.  Regression for the bug where every
// screened pair reported 0, leaving like-charge trials cuspless.

TEST_CASE("ScreenedCoulombTerm intralayer reports the bulk-Coulomb cusp residue",
          "[screened][cusp]") {
    // Bilayer MoS2 stack; source layer 2 is the interior TMD (ε_source = 14).  Direct
    // (intralayer) pair of like charges → residue = +1/14.
    const std::vector<double> eps = {1.0, 14.0, 14.0, 1.0};
    const std::vector<double> itf = {-3.0, 3.0, 9.0};
    ScreenedCoulombTerm ee(sgroup(0, 2, +1.0), sgroup(0, 2, +1.0), eps, itf,
                           /*source_layer=*/2, /*target_layer=*/0,
                           /*z_offset=*/0.0, /*fit_tol=*/0.1);
    CHECK(ee.coulomb_cusp_residue(sgroup(0, 2, +1.0), sgroup(0, 2, +1.0)) ==
          Catch::Approx(1.0 / 14.0).epsilon(0.05));
}

TEST_CASE("ScreenedCoulombTerm interlayer is cuspless (residue 0)",
          "[screened][cusp]") {
    // Interlayer e–h across the bilayer (source 2 → target 3, z_offset = T): V is finite
    // at ρ→0, so there is no 1/ρ pole and the residue must be 0.
    const std::vector<double> eps = {4.5, 14.0, 14.0, 4.5};
    const std::vector<double> itf = {-5.669, 5.669, 17.007};
    ScreenedCoulombTerm eh(sgroup(0, 1, +1.0), sgroup(1, 2, -1.0), eps, itf,
                           /*source_layer=*/2, /*target_layer=*/3,
                           /*z_offset=*/11.338, /*fit_tol=*/0.1);
    CHECK(eh.coulomb_cusp_residue(sgroup(0, 1, +1.0), sgroup(1, 2, -1.0)) ==
          Catch::Approx(0.0).margin(1e-4));
}

TEST_CASE("A short screened interlayer pair has exactly zero cusp despite fit residuals",
          "[screened][cusp][pimc-admission]") {
    // A finite source-observer separation makes the Fourier kernel decay as
    // exp(-k*|z|); no 1/r pole exists. Two-point extrapolation of a fitted
    // smooth potential can instead report a small positive numerical residue.
    const auto electron = sgroup(0, 1, -1.0), hole = sgroup(1, 2, 1.0);
    ScreenedCoulombTerm interlayer(electron, hole, {1.0, 2.0, 2.0, 1.0},
                                  {-1.0, 1.0, 3.0}, 2, 3, 2.0, 0.001);
    CHECK(interlayer.coulomb_cusp_residue(electron, hole) == 0.0);
    CHECK(interlayer.coulomb_cusp_residue(hole, electron) == 0.0);
    const auto cloned = interlayer.clone();
    CHECK(cloned->coulomb_cusp_residue(electron, hole) == 0.0);
    // The reported capability must not change the fitted potential values.
    CHECK(pair_energy(interlayer, 0.0) == pair_energy(interlayer, 0.01));
}

TEST_CASE("ScreenedCoulombTerm homogeneous cusp residue matches Coulomb",
          "[screened][cusp]") {
    // A single layer reduces to bare Coulomb, so the cusp residue must equal
    // CoulombTerm's q_a q_b/ε on the same geometry.
    const double eps = 5.0;
    ScreenedCoulombTerm st(sgroup(0, 1, +1.0), sgroup(1, 2, -1.0), {eps}, {},
                           /*source_layer=*/1, /*target_layer=*/0,
                           /*z_offset=*/0.0, /*fit_tol=*/0.1);
    CoulombTerm ct(sgroup(0, 1, +1.0), sgroup(1, 2, -1.0), eps);
    const double rs = st.coulomb_cusp_residue(sgroup(0, 1, +1.0), sgroup(1, 2, -1.0));
    const double rc = ct.coulomb_cusp_residue(sgroup(0, 1, +1.0), sgroup(1, 2, -1.0));
    CHECK(rc == Catch::Approx(-1.0 / eps));          // q_a q_b/ε = −1/5
    CHECK(rs == Catch::Approx(rc).epsilon(0.05));    // screened matches Coulomb
}

TEST_CASE("A weak screened intralayer pole is not erased by the fit resolution floor",
          "[screened][cusp][pimc-admission]") {
    // Its magnitude does not regularize 1/r: primitive attraction remains
    // non-integrable for every positive epsilon, including epsilon=1e8.
    const auto electron = sgroup(0, 1, -1.0), hole = sgroup(1, 2, 1.0);
    ScreenedCoulombTerm weak(electron, hole, {1e8}, {}, 1, 0, 0.0, 0.1);
    CHECK(weak.coulomb_cusp_residue(electron, hole) == Catch::Approx(-1e-8).epsilon(1e-12));
    const auto fit=alea::build_screened_fit({1e8},{},1,0,0.,.1);
    CHECK(fit.rho_max==2*fit.rho_min);
    CHECK(fsc_controlled_eval(fit.fit.get(),.015)==Catch::Approx(2e-8/.015).epsilon(1e-14));
}

