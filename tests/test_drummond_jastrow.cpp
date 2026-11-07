// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "population/particle_group.h"
#include "population/walker_set.h"
#include "wavefunction/drummond_jastrow.h"
#include "wavefunction/trial_wavefunction.h"

using alea::DrummondFunction;
using alea::DrummondJastrow;
using alea::Parameter;
using alea::ParticleGroup;
using alea::TrialWavefunction;
using alea::WalkerSet;

// ---------------------------------------------------------------------------
//  DrummondFunction unit tests
//
//  u(r) = (r−L)^C [ α0 + α1 r + Σ_{l≥2} αl r^l ],  α1 = Γ/(−L)^C + α0 C/L.
// ---------------------------------------------------------------------------

TEST_CASE("DrummondFunction enforces the cusp du/dr at 0 = gamma for any alpha0",
          "[drummond]") {
    // C=3, L=5, Γ=−1.  α1 = −1/(−125) = 0.008.  By construction du(0)=Γ for any α0.
    for (double a0 : {0.0, 2.0, -1.3}) {
        DrummondFunction f(/*C=*/3, /*gamma=*/-1.0, /*L=*/5.0, {a0});
        double u = 0.0, du = 0.0, d2u = 0.0;
        f.evaluate(0.0, u, du, d2u);
        CHECK(du == Catch::Approx(-1.0));
    }
}

TEST_CASE("DrummondFunction matches hand-computed u, u', u'' inside the cutoff",
          "[drummond]") {
    // C=3, L=5, Γ=−1, α0=0 → α1=0.008.  At r=2:
    //   g=(2−5)^3=−27, P=α1 r=0.016, g'=27, g''=−18.
    //   u   = g P            = −0.432
    //   u'  = g'P + g α1     =  0.216
    //   u'' = g''P + 2g'α1   =  0.144
    DrummondFunction f(3, -1.0, 5.0, {0.0});
    double u = 0.0, du = 0.0, d2u = 0.0;
    f.evaluate(2.0, u, du, d2u);
    CHECK(u   == Catch::Approx(-0.432));
    CHECK(du  == Catch::Approx(0.216));
    CHECK(d2u == Catch::Approx(0.144));
}

TEST_CASE("DrummondFunction vanishes at and beyond the cutoff", "[drummond]") {
    DrummondFunction f(3, -1.0, 5.0, {1.5, 0.3});  // arbitrary coeffs
    double u = 0.0, du = 0.0, d2u = 0.0;

    f.evaluate(5.0, u, du, d2u);          // exactly at L
    CHECK(u == Catch::Approx(0.0));
    CHECK(du == Catch::Approx(0.0));
    CHECK(d2u == Catch::Approx(0.0));

    f.evaluate(6.0, u, du, d2u);          // beyond L
    CHECK(u == Catch::Approx(0.0));
    CHECK(du == Catch::Approx(0.0));
    CHECK(d2u == Catch::Approx(0.0));
}

TEST_CASE("DrummondFunction derivatives match finite differences", "[drummond]") {
    DrummondFunction f(/*C=*/3, /*gamma=*/-0.7, /*L=*/5.0, {1.5, 0.3});
    const double r = 2.5, h = 1e-5;

    double u0 = 0, du0 = 0, d2u0 = 0;  f.evaluate(r, u0, du0, d2u0);
    double up = 0, dup = 0, d2up = 0;  f.evaluate(r + h, up, dup, d2up);
    double um = 0, dum = 0, d2um = 0;  f.evaluate(r - h, um, dum, d2um);

    CHECK(du0  == Catch::Approx((up - um) / (2.0 * h)).margin(1e-5));
    CHECK(d2u0 == Catch::Approx((up - 2.0 * u0 + um) / (h * h)).margin(1e-3));
}

TEST_CASE("DrummondFunction batched evaluate matches the scalar version",
          "[drummond]") {
    DrummondFunction f(3, -1.0, 5.0, {1.5, 0.3});
    const std::vector<double> r = {0.5, 1.0, 2.0, 4.0, 6.0};  // 6.0 is past L
    std::vector<double> u(r.size()), du(r.size()), d2u(r.size());
    f.evaluate(r, u, du, d2u);

    for (std::size_t w = 0; w < r.size(); ++w) {
        double su = 0, sdu = 0, sd2u = 0;
        f.evaluate(r[w], su, sdu, sd2u);
        CHECK(u[w]   == Catch::Approx(su));
        CHECK(du[w]  == Catch::Approx(sdu));
        CHECK(d2u[w] == Catch::Approx(sd2u));
    }
}

TEST_CASE("DrummondFunction param_derivatives match finite differences",
          "[drummond]") {
    // C=3, gamma=-0.7, L=5.0, coeffs=[α0=1.5, α2=0.3] → 3 params: α0, α2, L.
    DrummondFunction f(3, -0.7, 5.0, {1.5, 0.3});

    const double r = 2.0, eps = 1e-6;
    std::vector<double> out(3);
    f.param_derivatives(r, out);

    // Finite-difference check for α0.
    double up, dp, d2p;
    DrummondFunction f_a0p(3, -0.7, 5.0, {1.5 + eps, 0.3});
    f_a0p.evaluate(r, up, dp, d2p);
    double um;
    f.evaluate(r, um, dp, d2p);
    double fd_a0 = (up - um) / eps;
    CHECK(out[0] == Catch::Approx(fd_a0).epsilon(1e-4));

    // Finite-difference check for α2.
    DrummondFunction f_a2p(3, -0.7, 5.0, {1.5, 0.3 + eps});
    f_a2p.evaluate(r, up, dp, d2p);
    double fd_a2 = (up - um) / eps;
    CHECK(out[1] == Catch::Approx(fd_a2).epsilon(1e-3));

    // Finite-difference check for L.
    DrummondFunction f_Lp(3, -0.7, 5.0 + eps, {1.5, 0.3});
    f_Lp.evaluate(r, up, dp, d2p);
    double fd_L = (up - um) / eps;
    CHECK(out[2] == Catch::Approx(fd_L).epsilon(1e-3));

    // Beyond cutoff: all zero.
    f.param_derivatives(6.0, out);
    CHECK(out[0] == 0.0);
    CHECK(out[1] == 0.0);
    CHECK(out[2] == 0.0);
}

// ---------------------------------------------------------------------------
//  DrummondJastrow unit tests
// ---------------------------------------------------------------------------

namespace {

ParticleGroup group(std::size_t first, std::size_t last) {
    ParticleGroup g;
    g.name = "g";
    g.charge = 0.0;
    g.first = first;
    g.last = last;
    return g;
}

// params = [α0, L].  All fixed by default (no optimisation).
std::vector<Parameter> dru_params(double a0, double L, bool fixed = true) {
    return {{"p0", a0, fixed, false}, {"L", L, fixed, false}};
}

}  // namespace

TEST_CASE("DrummondJastrow evaluate_batch accumulates log psi = sum u(r_ij)",
          "[drummond_jastrow]") {
    // p0=(0,0), p1=(2,0) → r=2.  C=3, L=5, Γ=−1, α0=0 → u(2) = −0.432.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());

    DrummondJastrow jas(3, -1.0, group(0, 1), group(1, 2), dru_params(0.0, 5.0));
    jas.evaluate_batch(ws);

    CHECK(jas.log_psi_c()[0] == Catch::Approx(-0.432));
    CHECK(ws.log_psi()[0] == Catch::Approx(-0.432));
}

TEST_CASE("DrummondJastrow evaluate_batch accumulates gradient and laplacian",
          "[drummond_jastrow]") {
    // r=2, u'(2)=0.216, u''(2)=0.144 (see DrummondFunction test above).
    //   grad_0 = u'(r)(r0−r1)/r = 0.216·(−1,0) = (−0.216, 0)
    //   lap per-dim: ∂²_x = u'' = 0.144 ;  ∂²_y = u'/r = 0.108
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());

    DrummondJastrow jas(3, -1.0, group(0, 1), group(1, 2), dru_params(0.0, 5.0));
    jas.evaluate_batch(ws);

    auto g = ws.gradients();
    CHECK(g[0] == Catch::Approx(-0.216));  // p0, x
    CHECK(g[1] == Catch::Approx(0.0));     // p0, y
    CHECK(g[2] == Catch::Approx(0.216));   // p1, x
    CHECK(g[3] == Catch::Approx(0.0));     // p1, y

    auto lap = ws.laplacians();
    CHECK(lap[0] == Catch::Approx(0.144));  // p0, x
    CHECK(lap[1] == Catch::Approx(0.108));  // p0, y
    CHECK(lap[2] == Catch::Approx(0.144));  // p1, x
    CHECK(lap[3] == Catch::Approx(0.108));  // p1, y
}

TEST_CASE("DrummondJastrow gives zero contribution beyond the cutoff",
          "[drummond_jastrow]") {
    // r=8 with L=5 → outside cutoff → no contribution to log psi / grad / lap.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 8.0;
    ws.dist_tables().build_all(ws.positions());

    DrummondJastrow jas(3, -1.0, group(0, 1), group(1, 2), dru_params(1.5, 5.0));
    jas.evaluate_batch(ws);

    CHECK(ws.log_psi()[0] == Catch::Approx(0.0));
    for (double v : ws.gradients()) CHECK(v == Catch::Approx(0.0));
    for (double v : ws.laplacians()) CHECK(v == Catch::Approx(0.0));
}

TEST_CASE("DrummondJastrow infers same-group and counts each pair once",
          "[drummond_jastrow]") {
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());

    DrummondJastrow jas(3, -1.0, group(0, 2), group(0, 2), dru_params(0.0, 5.0));
    jas.evaluate_batch(ws);

    CHECK(ws.log_psi()[0] == Catch::Approx(-0.432));  // single pair, not doubled
}

TEST_CASE("DrummondJastrow ratio_batch agrees with evaluate_batch at the proposal",
          "[drummond_jastrow]") {
    // L=10 keeps both old (r=5) and new (r=4) distances inside the cutoff.
    DrummondJastrow jas(3, -1.0, group(0, 1), group(1, 2), dru_params(0.5, 10.0));

    WalkerSet cur(1, 2, 2);
    auto cp = cur.positions();
    cp[2] = 3.0;  // p1=(3,4)
    cp[3] = 4.0;
    cur.dist_tables().build_all(cur.positions());
    jas.evaluate_batch(cur);
    const double log_cur = cur.log_psi()[0];

    std::vector<double> new_pos = {3.0, 0.0};  // move p0 → (3,0), r'=4
    cur.propose_move_batch(0, new_pos);
    jas.ratio_batch(cur, 0);

    WalkerSet prop(1, 2, 2);
    auto pp = prop.positions();
    pp[0] = 3.0;
    pp[2] = 3.0;
    pp[3] = 4.0;
    prop.dist_tables().build_all(prop.positions());
    jas.evaluate_batch(prop);
    const double log_prop = prop.log_psi()[0];

    CHECK(cur.log_ratios()[0] == Catch::Approx(log_prop - log_cur));
    CHECK(cur.new_grad()[0] == Catch::Approx(prop.gradients()[0]));
    CHECK(cur.new_grad()[1] == Catch::Approx(prop.gradients()[1]));
    CHECK(cur.new_lap()[0]  == Catch::Approx(prop.laplacians()[0]));
    CHECK(cur.new_lap()[1]  == Catch::Approx(prop.laplacians()[1]));
}

namespace {

void setup_current(WalkerSet& ws) {  // p0=(0,0), p1=(3,4) → r=5
    auto pos = ws.positions();
    pos[2] = 3.0;
    pos[3] = 4.0;
    ws.dist_tables().build_all(ws.positions());
}

void setup_proposed(WalkerSet& ws) {  // p0=(3,0), p1=(3,4) → r=4
    auto pos = ws.positions();
    pos[0] = 3.0;
    pos[2] = 3.0;
    pos[3] = 4.0;
    ws.dist_tables().build_all(ws.positions());
}

}  // namespace

TEST_CASE("DrummondJastrow apply_move_batch matches a full eval on accept",
          "[drummond_jastrow]") {
    DrummondJastrow jas(3, -1.0, group(0, 1), group(1, 2), dru_params(0.5, 10.0));
    WalkerSet cur(1, 2, 2);
    setup_current(cur);
    jas.evaluate_batch(cur);

    std::vector<double> new_pos = {3.0, 0.0};
    cur.propose_move_batch(0, new_pos);
    jas.ratio_batch(cur, 0);
    std::array<bool, 1> accept = {true};
    jas.apply_move_batch(cur, 0, accept);

    DrummondJastrow ref(3, -1.0, group(0, 1), group(1, 2), dru_params(0.5, 10.0));
    WalkerSet prop(1, 2, 2);
    setup_proposed(prop);
    ref.evaluate_batch(prop);

    CHECK(jas.log_psi_c()[0] == Catch::Approx(ref.log_psi_c()[0]));
    for (std::size_t k = 0; k < jas.grad_c().size(); ++k)
        CHECK(jas.grad_c()[k] == Catch::Approx(ref.grad_c()[k]).margin(1e-12));
    for (std::size_t k = 0; k < jas.lap_c().size(); ++k)
        CHECK(jas.lap_c()[k] == Catch::Approx(ref.lap_c()[k]).margin(1e-12));
    for (std::size_t k = 0; k < cur.gradients().size(); ++k)
        CHECK(cur.gradients()[k] == Catch::Approx(prop.gradients()[k]).margin(1e-12));
}

TEST_CASE("DrummondJastrow apply_move_batch is a no-op on reject",
          "[drummond_jastrow]") {
    DrummondJastrow jas(3, -1.0, group(0, 1), group(1, 2), dru_params(0.5, 10.0));
    WalkerSet cur(1, 2, 2);
    setup_current(cur);
    jas.evaluate_batch(cur);

    const std::vector<double> grad0(jas.grad_c().begin(), jas.grad_c().end());
    const double log0 = jas.log_psi_c()[0];

    std::vector<double> new_pos = {3.0, 0.0};
    cur.propose_move_batch(0, new_pos);
    jas.ratio_batch(cur, 0);
    std::array<bool, 1> reject = {false};
    jas.apply_move_batch(cur, 0, reject);

    CHECK(jas.log_psi_c()[0] == Catch::Approx(log0));
    for (std::size_t k = 0; k < grad0.size(); ++k)
        CHECK(jas.grad_c()[k] == Catch::Approx(grad0[k]).margin(1e-12));
}

TEST_CASE("DrummondJastrow ratio/apply over disjoint crowds equal the full batch",
          "[drummond_jastrow][crowd]") {
    const std::size_t nw = 4, np = 2, dim = 2;
    auto setup = [&](WalkerSet& ws, TrialWavefunction& psi) {
        auto pos = ws.positions();  // p0=(0,0); p1 x = 1..4 (all < L=10)
        for (std::size_t w = 0; w < nw; ++w)
            pos[1 * dim * nw + 0 * nw + w] = static_cast<double>(w + 1);
        ws.dist_tables().build_all(ws.positions());
        psi.add_component(std::make_unique<DrummondJastrow>(
            3, -1.0, group(0, 1), group(1, 2), dru_params(0.5, 10.0)));
        psi.evaluate_batch(ws);
        std::vector<double> np0(dim * nw, 0.0);  // move p0 x → 0.5
        for (std::size_t w = 0; w < nw; ++w) np0[0 * nw + w] = 0.5;
        ws.propose_move_batch(0, np0);
    };

    WalkerSet wf(nw, np, dim); TrialWavefunction pf; setup(wf, pf);
    pf.ratio_batch(wf, 0);
    const std::vector<double> lr_full(wf.log_ratios().begin(), wf.log_ratios().end());

    WalkerSet ws(nw, np, dim); TrialWavefunction ps; setup(ws, ps);
    ps.ratio_batch(ws, 0, 0, 2);
    ps.ratio_batch(ws, 0, 2, 4);
    for (std::size_t w = 0; w < nw; ++w)
        CHECK(ws.log_ratios()[w] == Catch::Approx(lr_full[w]));

    std::array<bool, 4> acc = {true, true, true, true};
    pf.apply_move_batch(wf, 0, acc);
    ps.apply_move_batch(ws, 0, acc, 0, 2);
    ps.apply_move_batch(ws, 0, acc, 2, 4);
    for (std::size_t w = 0; w < nw; ++w)
        CHECK(ws.log_psi()[w] == Catch::Approx(wf.log_psi()[w]).margin(1e-12));
    for (std::size_t k = 0; k < np * dim * nw; ++k)
        CHECK(ws.gradients()[k] == Catch::Approx(wf.gradients()[k]).margin(1e-12));
}

// ---------------------------------------------------------------------------
//  Rebuild and clone
// ---------------------------------------------------------------------------

TEST_CASE("DrummondJastrow rebuild reflects an updated coefficient",
          "[drummond_jastrow][rebuild]") {
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());

    // α0 = 0 → u(2) = −0.432 (the linear-in-α1 part only).
    DrummondJastrow jas(3, -1.0, group(0, 1), group(1, 2), dru_params(0.0, 5.0));
    jas.evaluate_batch(ws);
    REQUIRE(jas.log_psi_c()[0] == Catch::Approx(-0.432));

    // Bump α0 → 1.  Adds (r−L)^C·α0 = (−27)·1 = −27 to u(2).  Note α1 also
    // shifts (α1 = Γ/(−L)^C + α0 C/L), so check against a fresh component.
    jas.parameters()[0].value = 1.0;
    jas.rebuild();
    jas.evaluate_batch(ws);

    DrummondFunction ref(3, -1.0, 5.0, {1.0});
    double u = 0, du = 0, d2u = 0;
    ref.evaluate(2.0, u, du, d2u);
    CHECK(jas.log_psi_c()[0] == Catch::Approx(u));
}

TEST_CASE("DrummondJastrow clone carries the rebuilt kernel",
          "[drummond_jastrow][rebuild]") {
    DrummondJastrow jas(3, -1.0, group(0, 1), group(1, 2), dru_params(0.0, 5.0));
    jas.parameters()[0].value = 2.0;
    jas.rebuild();

    auto cloned = jas.clone();
    auto* dru = dynamic_cast<DrummondJastrow*>(cloned.get());
    REQUIRE(dru != nullptr);

    WalkerSet ws(1, 2, 2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());
    dru->evaluate_batch(ws);

    DrummondFunction ref(3, -1.0, 5.0, {2.0});
    double u = 0, du = 0, d2u = 0;
    ref.evaluate(2.0, u, du, d2u);
    CHECK(dru->log_psi_c()[0] == Catch::Approx(u));
}

// ---------------------------------------------------------------------------
//  Parameter derivatives
// ---------------------------------------------------------------------------

TEST_CASE("DrummondJastrow d-logpsi-d-alpha0 and d-logpsi-d-L match hand values",
          "[drummond_jastrow][derivatives]") {
    // Single walker at r=2, C=3, L=5, Γ=−1, α0=0 (only α0 and L free).
    //   ∂u/∂α0 = (r−L)^C [1 + (C/L) r]            = −27·2.2     = −59.4
    //   ∂u/∂L  = −C(r−L)^{C−1} P + (r−L)^C α1'_L r
    //            = −3·9·0.016 + (−27)(−0.0048)(2) = −0.1728
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());

    DrummondJastrow jas(3, -1.0, group(0, 1), group(1, 2),
                        dru_params(0.0, 5.0, /*fixed=*/false));
    jas.evaluate_batch(ws);
    REQUIRE(jas.num_optimizable_params() == 2);

    std::vector<double> derivs(2 * ws.num_walkers());
    jas.compute_parameter_derivatives(ws, derivs);

    CHECK(derivs[0] == Catch::Approx(-59.4));    // ∂/∂α0
    CHECK(derivs[1] == Catch::Approx(-0.1728));  // ∂/∂L
}

TEST_CASE("DrummondJastrow derivatives match finite differences (alpha0, alpha2, L)",
          "[drummond_jastrow][derivatives]") {
    // params = [α0, α2, L], all free. Single walker at r=2.5 (inside L=5).
    std::vector<Parameter> params = {
        {"p0", 1.5, false, false},
        {"p1", 0.3, false, false},  // α2
        {"L",  5.0, false, false},
    };

    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.5;
    ws.dist_tables().build_all(ws.positions());

    DrummondJastrow jas(3, -0.7, group(0, 1), group(1, 2), params);
    jas.evaluate_batch(ws);
    const std::size_t n_opt = jas.num_optimizable_params();
    REQUIRE(n_opt == 3);

    std::vector<double> derivs(n_opt * ws.num_walkers());
    jas.compute_parameter_derivatives(ws, derivs);

    const double eps = 1e-6;
    for (std::size_t k = 0; k < n_opt; ++k) {
        jas.parameters()[k].value += eps;
        jas.rebuild();
        jas.evaluate_batch(ws);
        const double lp = jas.log_psi_c()[0];

        jas.parameters()[k].value -= 2.0 * eps;
        jas.rebuild();
        jas.evaluate_batch(ws);
        const double lm = jas.log_psi_c()[0];

        jas.parameters()[k].value += eps;
        jas.rebuild();

        const double fd = (lp - lm) / (2.0 * eps);
        CHECK(derivs[k] == Catch::Approx(fd).margin(1e-5));
    }
}

TEST_CASE("DrummondJastrow derivatives span multiple walkers",
          "[drummond_jastrow][derivatives]") {
    // 2 walkers at r=2 and r=3; only α0 free → ∂u/∂α0 = (r−L)^C[1+(C/L)r].
    WalkerSet ws(/*nw=*/2, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[4] = 2.0;  // p1 x, w0
    pos[5] = 3.0;  // p1 x, w1
    ws.dist_tables().build_all(ws.positions());

    std::vector<Parameter> params = {{"p0", 0.0, false, false},
                                     {"L", 5.0, true, false}};  // L fixed
    DrummondJastrow jas(3, -1.0, group(0, 1), group(1, 2), params);
    jas.evaluate_batch(ws);
    REQUIRE(jas.num_optimizable_params() == 1);

    std::vector<double> derivs(1 * ws.num_walkers());
    jas.compute_parameter_derivatives(ws, derivs);

    // w0: (2−5)^3 (1 + 0.6·2) = −27·2.2 = −59.4
    // w1: (3−5)^3 (1 + 0.6·3) = −8·2.8  = −22.4
    CHECK(derivs[0] == Catch::Approx(-59.4));
    CHECK(derivs[1] == Catch::Approx(-22.4));
}

TEST_CASE("DrummondJastrow parameter gradient derivatives match finite differences",
          "[drummond_jastrow][derivatives]") {
    // Three particles in one group, two walkers; α0, α2 and L all free, so
    // every branch (α0, αl, L) of the radial f_k', f_k'' is exercised.
    // Walker 1 puts one pair beyond the cutoff (r ≥ L → zero contribution).
    constexpr std::size_t nw = 2, np = 3, dim = 2;
    const double xy[nw][np][dim] = {{{0.1, 0.2}, {1.7, -0.4}, {-0.9, 1.3}},
                                    {{0.0, 0.0}, {2.2, 0.9}, {-3.6, 1.5}}};
    auto place = [&](WalkerSet& ws) {
        auto pos = ws.positions();
        for (std::size_t w = 0; w < nw; ++w)
            for (std::size_t p = 0; p < np; ++p)
                for (std::size_t d = 0; d < dim; ++d)
                    pos[(p * dim + d) * nw + w] = xy[w][p][d];
        ws.dist_tables().build_all(ws.positions());
    };
    const std::vector<double> base = {0.35, -0.02, 4.5};  // α0, α2, L
    auto make = [&](const std::vector<double>& v) {
        std::vector<Parameter> par = {{"a0", v[0], false, false},
                                      {"a2", v[1], false, false},
                                      {"L", v[2], false, false}};
        return DrummondJastrow(3, -0.5, group(0, 3), group(0, 3), par);
    };

    WalkerSet ws(nw, np, dim);
    place(ws);
    DrummondJastrow jas = make(base);
    jas.evaluate_batch(ws);
    REQUIRE(jas.num_optimizable_params() == 3);

    const std::size_t block = np * dim * nw;
    std::vector<double> dg(3 * block), dl(3 * block);
    REQUIRE(jas.supports_parameter_gradient_derivatives());
    REQUIRE(jas.compute_parameter_gradient_derivatives(ws, dg, dl));

    const double h = 1e-6;
    for (std::size_t k = 0; k < 3; ++k) {
        auto up = base, dn = base;
        up[k] += h;
        dn[k] -= h;
        WalkerSet wu(nw, np, dim), wd(nw, np, dim);
        place(wu);
        place(wd);
        DrummondJastrow ju = make(up), jd = make(dn);
        ju.evaluate_batch(wu);
        jd.evaluate_batch(wd);
        for (std::size_t i = 0; i < block; ++i) {
            const double fd_g = (ju.grad_c()[i] - jd.grad_c()[i]) / (2 * h);
            const double fd_l = (ju.lap_c()[i] - jd.lap_c()[i]) / (2 * h);
            CHECK(dg[k * block + i] == Catch::Approx(fd_g).epsilon(1e-5).margin(1e-7));
            CHECK(dl[k * block + i] == Catch::Approx(fd_l).epsilon(1e-5).margin(1e-7));
        }
    }
}

// ---------------------------------------------------------------------------
//  One u(r) over several group pairs (YAML `pairs:`)
// ---------------------------------------------------------------------------

namespace {

// Four groups of two particles: [0,2), [2,4), [4,6), [6,8).  The list mixes
// same-group and two-group pairs and gives one two-group pair in reverse order.
std::vector<alea::JastrowGroupPair> four_group_pairs() {
    return {{group(0, 2), group(0, 2)}, {group(2, 4), group(2, 4)},
            {group(0, 2), group(4, 6)}, {group(6, 8), group(2, 4)}};
}

std::vector<Parameter> free_params() {  // α0, α2, L, all optimisable
    return {{"a0", 0.35, false, false}, {"a2", -0.02, false, false}, {"L", 4.5, false, false}};
}

TrialWavefunction combined_factor() {
    TrialWavefunction psi;
    psi.add_component(std::make_unique<DrummondJastrow>(3, -0.5, four_group_pairs(),
                                                        free_params()));
    return psi;
}

TrialWavefunction separate_factors() {
    TrialWavefunction psi;
    for (const alea::JastrowGroupPair& gp : four_group_pairs())
        psi.add_component(std::make_unique<DrummondJastrow>(3, -0.5, gp.a, gp.b,
                                                            free_params()));
    return psi;
}

void place_eight(WalkerSet& ws) {
    const std::size_t nw = ws.num_walkers();
    auto pos = ws.positions();
    for (std::size_t w = 0; w < nw; ++w)
        for (std::size_t p = 0; p < 8; ++p)
            for (std::size_t d = 0; d < 2; ++d)
                pos[(p * 2 + d) * nw + w] = 1.6 * std::sin(1.3 * p + 0.9 * d + 0.4 * w);
    ws.dist_tables().build_all(ws.positions());
}

}  // namespace

TEST_CASE("A Drummond factor over several group pairs equals the sum of single-pair "
          "factors through a sequence of moves", "[drummond_jastrow][pairs]") {
    constexpr std::size_t nw = 3, np = 8, dim = 2;
    WalkerSet a(nw, np, dim), b(nw, np, dim);
    place_eight(a);
    place_eight(b);
    TrialWavefunction combined = combined_factor(), separate = separate_factors();
    combined.evaluate_batch(a);
    separate.evaluate_batch(b);

    auto same_state = [&] {
        for (std::size_t w = 0; w < nw; ++w)
            REQUIRE(a.log_psi()[w] == Catch::Approx(b.log_psi()[w]).margin(1e-12));
        for (std::size_t k = 0; k < a.gradients().size(); ++k) {
            REQUIRE(a.gradients()[k] == Catch::Approx(b.gradients()[k]).margin(1e-12));
            REQUIRE(a.laplacians()[k] == Catch::Approx(b.laplacians()[k]).margin(1e-12));
        }
    };
    same_state();

    for (std::size_t move = 0; move < 64; ++move) {
        const std::size_t p = move % np;
        std::vector<double> proposal(dim * nw);
        for (std::size_t d = 0; d < dim; ++d)
            for (std::size_t w = 0; w < nw; ++w)
                proposal[d * nw + w] = a.positions()[(p * dim + d) * nw + w]
                                     + 0.3 * std::sin(0.7 * move + d + 2 * w);
        a.propose_move_batch(p, proposal);
        b.propose_move_batch(p, proposal);
        combined.ratio_batch(a, p);
        separate.ratio_batch(b, p);
        for (std::size_t w = 0; w < nw; ++w)
            REQUIRE(a.log_ratios()[w] == Catch::Approx(b.log_ratios()[w]).margin(1e-12));
        for (std::size_t k = 0; k < dim * nw; ++k) {
            REQUIRE(a.new_grad()[k] == Catch::Approx(b.new_grad()[k]).margin(1e-12));
            REQUIRE(a.new_lap()[k] == Catch::Approx(b.new_lap()[k]).margin(1e-12));
        }
        auto accept_a = a.accept_mask();
        auto accept_b = b.accept_mask();
        for (std::size_t w = 0; w < nw; ++w) accept_a[w] = accept_b[w] = (move + w) % 3 != 0;
        combined.apply_move_batch(a, p, accept_a);
        separate.apply_move_batch(b, p, accept_b);
        a.apply_move(p, proposal, a.new_grad(), a.new_lap(), accept_a);
        b.apply_move(p, proposal, b.new_grad(), b.new_lap(), accept_b);
        same_state();
    }
}

TEST_CASE("Parameter derivatives of a shared u(r) are the sums over its group pairs",
          "[drummond_jastrow][pairs][derivatives]") {
    // The parameters are shared, so ∂ log ψ/∂p (and its gradient and
    // laplacian) of the combined factor is the sum of the single-pair ones.
    constexpr std::size_t nw = 3, np = 8, dim = 2, n_opt = 3;
    WalkerSet ws(nw, np, dim);
    place_eight(ws);
    DrummondJastrow combined(3, -0.5, four_group_pairs(), free_params());
    combined.evaluate_batch(ws);
    std::vector<double> d_comb(n_opt * nw);
    combined.compute_parameter_derivatives(ws, d_comb);
    const std::size_t block = np * dim * nw;
    std::vector<double> g_comb(n_opt * block), l_comb(n_opt * block);
    REQUIRE(combined.compute_parameter_gradient_derivatives(ws, g_comb, l_comb));

    std::vector<double> d_sum(n_opt * nw, 0.0), g_sum(n_opt * block, 0.0),
        l_sum(n_opt * block, 0.0);
    for (const alea::JastrowGroupPair& gp : four_group_pairs()) {
        WalkerSet one(nw, np, dim);
        place_eight(one);
        DrummondJastrow single(3, -0.5, gp.a, gp.b, free_params());
        single.evaluate_batch(one);
        std::vector<double> d(n_opt * nw), g(n_opt * block), l(n_opt * block);
        single.compute_parameter_derivatives(one, d);
        REQUIRE(single.compute_parameter_gradient_derivatives(one, g, l));
        for (std::size_t k = 0; k < d.size(); ++k) d_sum[k] += d[k];
        for (std::size_t k = 0; k < g.size(); ++k) { g_sum[k] += g[k]; l_sum[k] += l[k]; }
    }
    for (std::size_t k = 0; k < d_sum.size(); ++k)
        CHECK(d_comb[k] == Catch::Approx(d_sum[k]).margin(1e-12));
    for (std::size_t k = 0; k < g_sum.size(); ++k) {
        CHECK(g_comb[k] == Catch::Approx(g_sum[k]).margin(1e-12));
        CHECK(l_comb[k] == Catch::Approx(l_sum[k]).margin(1e-12));
    }
}

TEST_CASE("A group pair given twice in one factor is refused",
          "[drummond_jastrow][pairs]") {
    // Twice, or once reversed, would double-count every pair of it.
    CHECK_THROWS_AS(DrummondJastrow(3, -0.5,
                                    std::vector<alea::JastrowGroupPair>{
                                        {group(0, 2), group(2, 4)}, {group(2, 4), group(0, 2)}},
                                    free_params()),
                    std::invalid_argument);
    CHECK_THROWS_AS(DrummondJastrow(3, -0.5, std::vector<alea::JastrowGroupPair>{},
                                    free_params()),
                    std::invalid_argument);
}
