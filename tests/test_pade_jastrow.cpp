// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <array>
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "population/particle_group.h"
#include "population/walker_set.h"
#include "population/walker.h"
#include "wavefunction/pade_jastrow.h"
#include "wavefunction/trial_wavefunction.h"

using alea::PadeFunction;
using alea::ParticleGroup;
using alea::Parameter;
using alea::PadeJastrow;
using alea::WalkerSet;
using alea::TrialWavefunction;

TEST_CASE("Pade proposal reuse survives mixed crowds and changing population strides",
          "[pade_jastrow][crowd]") {
    const std::array<double, 4> vectors{5, 0, 1.8, 4.5};
    const alea::SimulationCell cell(2, vectors);
    auto make_psi = [] {
        ParticleGroup a{}, b{};
        a.first = 0; a.last = 2;
        b.first = 2; b.last = 4;
        TrialWavefunction psi;
        psi.add_component(std::make_unique<PadeJastrow>(1, 1, a, a,
            std::vector<Parameter>{{"a", 0.3, true}, {"b", 0.5, true}}));
        psi.add_component(std::make_unique<PadeJastrow>(1, 1, a, b,
            std::vector<Parameter>{{"a", -1, true}, {"b", 0.5, true}}));
        return psi;
    };
    std::vector<alea::Walker> walkers;
    walkers.reserve(3);
    for (int w = 0; w < 3; ++w) {
        walkers.emplace_back(4, 2);
        const std::array<double, 8> pos{-1.0, 0.1*w, 0.3, 1.2, 1.3, -0.8, 2.1, 0.6};
        walkers.back().set_positions(pos);
    }
    WalkerSet ws(3, 4, 2, cell);
    for (auto& walker : walkers) ws.add_walker(walker);
    ws.load_all();
    auto psi = make_psi();
    psi.evaluate_batch(ws);
    for (std::size_t move = 0; move < 400; ++move) {
        if (move == 100 || move == 250) {
            const std::vector<std::size_t> src = move == 100
                ? std::vector<std::size_t>{2, 0, 0, 1, 2} : std::vector<std::size_t>{4, 1};
            ws.apply_permutation(src);
            psi.permute_caches(src);
            psi = psi.clone();
        }
        const auto nw = ws.num_walkers(), p = move % 4;
        std::vector<double> proposal(2 * nw); // [dim * nw]
        for (std::size_t d = 0; d < 2; ++d)
            for (std::size_t w = 0; w < nw; ++w)
                proposal[d*nw+w] = ws.positions()[(p*2+d)*nw+w]
                    + 0.2 * std::sin(0.7*move+d+2*w);
        ws.propose_move_batch(p, proposal);
#pragma omp parallel for num_threads(3)
        for (int t = 0; t < 3; ++t)
            psi.ratio_batch(ws, p, nw*t/3, nw*(t+1)/3);
        auto accept = ws.accept_mask();
        for (std::size_t w = 0; w < nw; ++w) accept[w] = (move+w)%3 != 0;
#pragma omp parallel for num_threads(3)
        for (int t = 0; t < 3; ++t)
            psi.apply_move_batch(ws, p, accept, nw*t/3, nw*(t+1)/3);
        ws.apply_move(p, proposal, ws.new_grad(), ws.new_lap(), accept);
        if (move % 13 == 0 || move == 399) {
            WalkerSet fresh(nw, 4, 2, cell);
            std::copy(ws.positions().begin(), ws.positions().end(), fresh.positions().begin());
            fresh.dist_tables().build_all(fresh.positions());
            auto ref = make_psi();
            ref.evaluate_batch(fresh);
            for (std::size_t w = 0; w < nw; ++w)
                REQUIRE(ws.log_psi()[w] == Catch::Approx(fresh.log_psi()[w]).margin(2e-12));
            for (std::size_t k = 0; k < ws.gradients().size(); ++k) {
                REQUIRE(ws.gradients()[k] == Catch::Approx(fresh.gradients()[k]).margin(2e-12));
                REQUIRE(ws.laplacians()[k] == Catch::Approx(fresh.laplacians()[k]).margin(2e-11));
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  PadeFunction unit tests
// ---------------------------------------------------------------------------

TEST_CASE("PadeFunction evaluates u(r) for the standard a*r/(1+b*r) form",
          "[pade]") {
    // a = 1, b = 0.5 → u(r) = r / (1 + 0.5 r). At r = 2: u = 2 / 2 = 1.
    PadeFunction pade(/*num_order=*/1, /*den_order=*/1, {1.0, 0.5});

    double u = 0.0, du = 0.0, d2u = 0.0;
    pade.evaluate(/*r=*/2.0, u, du, d2u);

    CHECK(u == Catch::Approx(1.0));
}

TEST_CASE("PadeFunction evaluates u'(r) = a/(1+b r)^2 for the standard form",
          "[pade]") {
    // a = 1, b = 0.5, r = 2 → u'(r) = 1 / (1 + 1)^2 = 0.25.
    PadeFunction pade(/*num_order=*/1, /*den_order=*/1, {1.0, 0.5});

    double u = 0.0, du = 0.0, d2u = 0.0;
    pade.evaluate(/*r=*/2.0, u, du, d2u);

    CHECK(du == Catch::Approx(0.25));
}

TEST_CASE("PadeFunction evaluates u''(r) = -2 a b/(1+b r)^3 for the standard form",
          "[pade]") {
    // a = 1, b = 0.5, r = 2 → u''(r) = -2*1*0.5 / (1 + 1)^3 = -1/8 = -0.125.
    PadeFunction pade(/*num_order=*/1, /*den_order=*/1, {1.0, 0.5});

    double u = 0.0, du = 0.0, d2u = 0.0;
    pade.evaluate(/*r=*/2.0, u, du, d2u);

    CHECK(d2u == Catch::Approx(-0.125));
}

TEST_CASE("PadeFunction is finite at r=0 (cusp region)", "[pade]") {
    // a = 1, b = 0.5 at r = 0: u = 0, u' = a_1 = 1, u'' = -2 a b = -1.
    PadeFunction pade(/*num_order=*/1, /*den_order=*/1, {1.0, 0.5});

    double u = 0.0, du = 0.0, d2u = 0.0;
    pade.evaluate(/*r=*/0.0, u, du, d2u);

    CHECK(u == Catch::Approx(0.0));
    CHECK(du == Catch::Approx(1.0));
    CHECK(d2u == Catch::Approx(-1.0));
}

TEST_CASE("PadeFunction handles higher-order numerator and empty denominator",
          "[pade]") {
    // num_order=2, den_order=0, coeffs=[a_1=0, a_2=1] → u(r) = r^2 (Q = 1).
    // At r = 3: u = 9, u' = 2r = 6, u'' = 2.
    PadeFunction pade(/*num_order=*/2, /*den_order=*/0, {0.0, 1.0});

    double u = 0.0, du = 0.0, d2u = 0.0;
    pade.evaluate(/*r=*/3.0, u, du, d2u);

    CHECK(u == Catch::Approx(9.0));
    CHECK(du == Catch::Approx(6.0));
    CHECK(d2u == Catch::Approx(2.0));
}

TEST_CASE("PadeFunction batched evaluate matches the scalar version", "[pade]") {
    PadeFunction pade(/*num_order=*/1, /*den_order=*/1, {1.0, 0.5});

    const std::vector<double> r = {0.5, 1.0, 2.0, 4.0};
    std::vector<double> u(r.size()), du(r.size()), d2u(r.size());
    pade.evaluate(r, u, du, d2u);

    for (std::size_t w = 0; w < r.size(); ++w) {
        double su = 0.0, sdu = 0.0, sd2u = 0.0;
        pade.evaluate(r[w], su, sdu, sd2u);
        CHECK(u[w] == Catch::Approx(su));
        CHECK(du[w] == Catch::Approx(sdu));
        CHECK(d2u[w] == Catch::Approx(sd2u));
    }
}

// ---------------------------------------------------------------------------
//  PadeJastrow unit tests
// ---------------------------------------------------------------------------

namespace {

// A particle group spanning indices [first, last); only the range matters here.
ParticleGroup group(std::size_t first, std::size_t last) {
    ParticleGroup g;
    g.name = "g";
    g.charge = 0.0;
    g.first = first;
    g.last = last;
    return g;
}

// u(r) = r : a_1 = 1, b_1 = 0 (so u = r/(1+0)).
std::vector<Parameter> linear_u() {
    return {{"a1", 1.0, true}, {"b1", 0.0, true}};
}

}  // namespace

TEST_CASE("PadeJastrow evaluate_batch accumulates log psi = sum u(r_ij)",
          "[pade_jastrow]") {
    // 1 walker, 2 particles in 2D: p0 = (0,0), p1 = (2,0) → r_01 = 2.
    // Inter-group (group_a = {0}, group_b = {1}), u(r) = r → log psi = u(2) = 2.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();  // [p*dim*nw + d*nw + w], nw = 1
    pos[2] = 2.0;               // p1 x-coordinate
    ws.dist_tables().build_all(ws.positions());

    PadeJastrow jas(/*num_order=*/1, /*den_order=*/1,
                    group(0, 1), group(1, 2), linear_u());
    jas.evaluate_batch(ws);

    CHECK(jas.log_psi_c()[0] == Catch::Approx(2.0));
    CHECK(ws.log_psi()[0] == Catch::Approx(2.0));
}

TEST_CASE("permute_caches reorders the committed caches column-by-column",
          "[pade_jastrow][branch]") {
    // 3 walkers, 2 particles in 2D. p0=(0,0); p1=(1,0),(2,0),(3,0) → r=1,2,3.
    // Inter-group, u(r)=r → log_psi_c[w] = r_w = {1,2,3}.
    const std::size_t nw = 3;
    WalkerSet ws(nw, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();  // [p*dim*nw + d*nw + w]
    pos[1 * 2 * nw + 0 * nw + 0] = 1.0;  // p1 x, w0
    pos[1 * 2 * nw + 0 * nw + 1] = 2.0;  // p1 x, w1
    pos[1 * 2 * nw + 0 * nw + 2] = 3.0;  // p1 x, w2
    ws.dist_tables().build_all(ws.positions());

    PadeJastrow jas(1, 1, group(0, 1), group(1, 2), linear_u());
    jas.evaluate_batch(ws);

    REQUIRE(jas.log_psi_c().size() == nw);
    CHECK(jas.log_psi_c()[0] == Catch::Approx(1.0));
    CHECK(jas.log_psi_c()[1] == Catch::Approx(2.0));
    CHECK(jas.log_psi_c()[2] == Catch::Approx(3.0));

    // Snapshot grad/lap column for walker 2 (k=p1,x → index (1*2+0)).
    const std::size_t k_p1x = (1 * 2 + 0);
    const double g_w2 = jas.grad_c()[k_p1x * nw + 2];
    const double l_w2 = jas.lap_c()[k_p1x * nw + 2];

    // new col0 ← old 2, col1 ← old 0, col2 ← old 0 (duplicate).
    std::vector<std::size_t> src = {2, 0, 0};
    jas.permute_caches(src);

    REQUIRE(jas.log_psi_c().size() == 3);
    CHECK(jas.log_psi_c()[0] == Catch::Approx(3.0));
    CHECK(jas.log_psi_c()[1] == Catch::Approx(1.0));
    CHECK(jas.log_psi_c()[2] == Catch::Approx(1.0));

    // grad/lap keep the [k*nw + w] layout and follow the columns.
    REQUIRE(jas.grad_c().size() == 2 * 2 * 3);
    REQUIRE(jas.lap_c().size() == 2 * 2 * 3);
    CHECK(jas.grad_c()[k_p1x * 3 + 0] == Catch::Approx(g_w2));  // col0 ← old w2
    CHECK(jas.lap_c()[k_p1x * 3 + 0] == Catch::Approx(l_w2));
}

TEST_CASE("ratio/apply over disjoint crowds equal the full batch",
          "[pade_jastrow][crowd]") {
    // Processing walkers [0,2) then [2,4) must equal processing [0,4) at once.
    const std::size_t nw = 4, np = 2, dim = 2;
    auto setup = [&](WalkerSet& ws, TrialWavefunction& psi) {
        auto pos = ws.positions();                       // p0=(0,0); p1 x = 1..4
        for (std::size_t w = 0; w < nw; ++w)
            pos[1 * dim * nw + 0 * nw + w] = static_cast<double>(w + 1);
        ws.dist_tables().build_all(ws.positions());
        psi.add_component(std::make_unique<PadeJastrow>(
            1, 1, group(0, 1), group(1, 2), linear_u()));
        psi.evaluate_batch(ws);
        std::vector<double> np0(dim * nw, 0.0);          // move p0 x → 0.5
        for (std::size_t w = 0; w < nw; ++w) np0[0 * nw + w] = 0.5;
        ws.propose_move_batch(0, np0);
    };

    WalkerSet wf(nw, np, dim); TrialWavefunction pf; setup(wf, pf);
    pf.ratio_batch(wf, 0);
    const std::vector<double> lr_full(wf.log_ratios().begin(), wf.log_ratios().end());
    const std::vector<double> ng_full(wf.new_grad().begin(), wf.new_grad().end());

    WalkerSet ws(nw, np, dim); TrialWavefunction ps; setup(ws, ps);
    ps.ratio_batch(ws, 0, 0, 2);
    ps.ratio_batch(ws, 0, 2, 4);
    for (std::size_t w = 0; w < nw; ++w)
        CHECK(ws.log_ratios()[w] == Catch::Approx(lr_full[w]));
    for (std::size_t k = 0; k < dim * nw; ++k)
        CHECK(ws.new_grad()[k] == Catch::Approx(ng_full[k]).margin(1e-12));

    // Accept all; committed caches + batch totals must match too.
    std::array<bool, 4> acc = {true, true, true, true};
    pf.apply_move_batch(wf, 0, acc);
    ps.apply_move_batch(ws, 0, acc, 0, 2);
    ps.apply_move_batch(ws, 0, acc, 2, 4);
    for (std::size_t w = 0; w < nw; ++w)
        CHECK(ws.log_psi()[w] == Catch::Approx(wf.log_psi()[w]).margin(1e-12));
    for (std::size_t k = 0; k < np * dim * nw; ++k)
        CHECK(ws.gradients()[k] == Catch::Approx(wf.gradients()[k]).margin(1e-12));
}

TEST_CASE("PadeJastrow evaluate_batch accumulates the log-psi gradient",
          "[pade_jastrow]") {
    // Same geometry: p0 = (0,0), p1 = (2,0), r = 2, u(r) = r → u'(r) = 1.
    // grad_i log psi = u'(r) (r_i - r_j)/r.
    //   grad_0 = (0-2, 0)/2 = (-1, 0)
    //   grad_1 = (2-0, 0)/2 = ( 1, 0)
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());

    PadeJastrow jas(1, 1, group(0, 1), group(1, 2), linear_u());
    jas.evaluate_batch(ws);

    auto g = ws.gradients();  // [p*dim*nw + d*nw + w], nw = 1
    CHECK(g[0] == Catch::Approx(-1.0));  // p0, x
    CHECK(g[1] == Catch::Approx(0.0));   // p0, y
    CHECK(g[2] == Catch::Approx(1.0));   // p1, x
    CHECK(g[3] == Catch::Approx(0.0));   // p1, y
}

TEST_CASE("PadeJastrow evaluate_batch accumulates the log-psi laplacian",
          "[pade_jastrow]") {
    // p0 = (0,0), p1 = (2,0), r = 2, dim = 2, u(r) = r → u' = 1, u'' = 0.
    // lap_i log psi = u''(r) + (dim-1)/r * u'(r) = 0 + (1/2)*1 = 0.5 per particle.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());

    PadeJastrow jas(1, 1, group(0, 1), group(1, 2), linear_u());
    jas.evaluate_batch(ws);

    auto lap = ws.laplacians();  // [p*dim*nw + d*nw + w], nw = 1
    // Per-dimension: ∂²_x = 0, ∂²_y = 0.5 for both particles.
    CHECK(lap[0] == Catch::Approx(0.0));  // p0, x
    CHECK(lap[1] == Catch::Approx(0.5));  // p0, y
    CHECK(lap[2] == Catch::Approx(0.0));  // p1, x
    CHECK(lap[3] == Catch::Approx(0.5));  // p1, y
}

TEST_CASE("PadeJastrow infers same-group and counts each pair once",
          "[pade_jastrow]") {
    // Both particles in one group [0,2): same_group is inferred (group_a==group_b).
    // Only the single pair (0,1) contributes → log psi = u(2) = 2 (not 4).
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());

    PadeJastrow jas(1, 1, group(0, 2), group(0, 2), linear_u());
    jas.evaluate_batch(ws);

    CHECK(ws.log_psi()[0] == Catch::Approx(2.0));
}

TEST_CASE("PadeJastrow evaluate_batch adds to existing batch totals",
          "[pade_jastrow]") {
    // A component accumulates (+=) into the batch; TrialWavefunction zeroes
    // first and sums components. Pre-seed log psi and confirm it is added to.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());
    ws.log_psi()[0] = 10.0;  // pre-existing contribution from another component

    PadeJastrow jas(1, 1, group(0, 1), group(1, 2), linear_u());
    jas.evaluate_batch(ws);

    CHECK(ws.log_psi()[0] == Catch::Approx(12.0));  // 10 + u(2)
}

TEST_CASE("PadeJastrow ratio_batch computes log(psi'/psi) for the moved particle",
          "[pade_jastrow]") {
    // p0=(0,0), p1=(2,0)?  Use p0=(0,0), p1=(3,4) → r=5. Move p0 to (3,0) → r'=4.
    // u(r)=r → log(psi'/psi) = u(4) - u(5) = -1.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 3.0;  // p1, x
    pos[3] = 4.0;  // p1, y
    ws.dist_tables().build_all(ws.positions());          // old r_01 = 5

    std::vector<double> new_pos = {3.0, 0.0};            // move p0 → (3,0)
    ws.propose_move_batch(/*p=*/0, new_pos);             // pending r'_01 = 4

    PadeJastrow jas(1, 1, group(0, 1), group(1, 2), linear_u());
    jas.ratio_batch(ws, /*p=*/0);

    CHECK(ws.log_ratios()[0] == Catch::Approx(-1.0));
}

TEST_CASE("PadeJastrow ratio_batch computes the moved particle's new grad/lap",
          "[pade_jastrow]") {
    // p0=(0,0), p1=(3,4). Move p0 to (3,0) → r'=4, u'(4)=1, u''(4)=0.
    // new grad_0 = u'(r') (r'_0 - r_1)/r' = (0,-4)/4 = (0, -1).
    // new lap_0  = u''(r') + (dim-1)/r' u'(r') = 0 + (1/4)*1 = 0.25.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 3.0;
    pos[3] = 4.0;
    ws.dist_tables().build_all(ws.positions());

    std::vector<double> new_pos = {3.0, 0.0};
    ws.propose_move_batch(/*p=*/0, new_pos);  // also records new_pos()

    PadeJastrow jas(1, 1, group(0, 1), group(1, 2), linear_u());
    jas.ratio_batch(ws, /*p=*/0);

    auto g = ws.new_grad();  // [dim*nw], nw = 1
    CHECK(g[0] == Catch::Approx(0.0));    // x
    CHECK(g[1] == Catch::Approx(-1.0));   // y
    CHECK(ws.new_lap()[0] == Catch::Approx(0.25));
}

TEST_CASE("PadeJastrow ratio_batch agrees with evaluate_batch at the proposal",
          "[pade_jastrow]") {
    PadeJastrow jas(1, 1, group(0, 1), group(1, 2), linear_u());

    // Current config: p0=(0,0), p1=(3,4).
    WalkerSet cur(1, 2, 2);
    auto cp = cur.positions();
    cp[2] = 3.0;
    cp[3] = 4.0;
    cur.dist_tables().build_all(cur.positions());
    jas.evaluate_batch(cur);
    const double log_cur = cur.log_psi()[0];

    // Propose p0 → (3,0) and get the incremental ratio + new grad/lap.
    std::vector<double> new_pos = {3.0, 0.0};
    cur.propose_move_batch(0, new_pos);
    jas.ratio_batch(cur, 0);

    // Proposed config built from scratch: p0=(3,0), p1=(3,4).
    WalkerSet prop(1, 2, 2);
    auto pp = prop.positions();
    pp[0] = 3.0;  // p0 x
    pp[2] = 3.0;  // p1 x
    pp[3] = 4.0;  // p1 y
    prop.dist_tables().build_all(prop.positions());
    jas.evaluate_batch(prop);
    const double log_prop = prop.log_psi()[0];

    // log(ψ'/ψ) must equal the difference of the full log ψ values.
    CHECK(cur.log_ratios()[0] == Catch::Approx(log_prop - log_cur));
    // The incremental new gradient/laplacian of p0 must match the full eval.
    CHECK(cur.new_grad()[0] == Catch::Approx(prop.gradients()[0]));
    CHECK(cur.new_grad()[1] == Catch::Approx(prop.gradients()[1]));
    CHECK(cur.new_lap()[0] == Catch::Approx(prop.laplacians()[0]));  // ∂²_x
    CHECK(cur.new_lap()[1] == Catch::Approx(prop.laplacians()[1]));  // ∂²_y
}

// Helper: drive a single-particle move on `ws` with `jas`, current→proposed,
// returning nothing; leaves caches/totals updated for the accept decision.
namespace {

void setup_current(WalkerSet& ws) {  // p0=(0,0), p1=(3,4)
    auto pos = ws.positions();
    pos[2] = 3.0;
    pos[3] = 4.0;
    ws.dist_tables().build_all(ws.positions());
}

void setup_proposed(WalkerSet& ws) {  // p0=(3,0), p1=(3,4)
    auto pos = ws.positions();
    pos[0] = 3.0;
    pos[2] = 3.0;
    pos[3] = 4.0;
    ws.dist_tables().build_all(ws.positions());
}

}  // namespace

TEST_CASE("apply_move_batch updates log_psi_c to the proposed value on accept",
          "[pade_jastrow]") {
    PadeJastrow jas(1, 1, group(0, 1), group(1, 2), linear_u());
    WalkerSet cur(1, 2, 2);
    setup_current(cur);
    jas.evaluate_batch(cur);  // log_psi_c = u(5) = 5

    std::vector<double> new_pos = {3.0, 0.0};
    cur.propose_move_batch(0, new_pos);
    jas.ratio_batch(cur, 0);
    std::array<bool, 1> accept = {true};
    jas.apply_move_batch(cur, 0, accept);  // log_psi_c += u(4)-u(5) → 4

    PadeJastrow ref(1, 1, group(0, 1), group(1, 2), linear_u());
    WalkerSet prop(1, 2, 2);
    setup_proposed(prop);
    ref.evaluate_batch(prop);  // log_psi_c = u(4) = 4

    CHECK(jas.log_psi_c()[0] == Catch::Approx(ref.log_psi_c()[0]));
}

TEST_CASE("apply_move_batch matches a full eval of the proposed config on accept",
          "[pade_jastrow]") {
    PadeJastrow jas(1, 1, group(0, 1), group(1, 2), linear_u());
    WalkerSet cur(1, 2, 2);
    setup_current(cur);
    jas.evaluate_batch(cur);

    std::vector<double> new_pos = {3.0, 0.0};
    cur.propose_move_batch(0, new_pos);
    jas.ratio_batch(cur, 0);
    std::array<bool, 1> accept = {true};
    jas.apply_move_batch(cur, 0, accept);

    PadeJastrow ref(1, 1, group(0, 1), group(1, 2), linear_u());
    WalkerSet prop(1, 2, 2);
    setup_proposed(prop);
    ref.evaluate_batch(prop);

    // Component caches (p AND its partner) match the full proposed evaluation.
    // margin() is needed because some entries are exactly 0 (Approx's relative
    // tolerance collapses to zero there).
    for (std::size_t k = 0; k < jas.grad_c().size(); ++k)
        CHECK(jas.grad_c()[k] == Catch::Approx(ref.grad_c()[k]).margin(1e-12));
    for (std::size_t k = 0; k < jas.lap_c().size(); ++k)
        CHECK(jas.lap_c()[k] == Catch::Approx(ref.lap_c()[k]).margin(1e-12));

    // Batch totals received the delta and now equal the proposed totals.
    for (std::size_t k = 0; k < cur.gradients().size(); ++k)
        CHECK(cur.gradients()[k] == Catch::Approx(prop.gradients()[k]).margin(1e-12));
    for (std::size_t k = 0; k < cur.laplacians().size(); ++k)
        CHECK(cur.laplacians()[k] == Catch::Approx(prop.laplacians()[k]).margin(1e-12));
}

TEST_CASE("apply_move_batch leaves caches and totals untouched on reject",
          "[pade_jastrow]") {
    PadeJastrow jas(1, 1, group(0, 1), group(1, 2), linear_u());
    WalkerSet cur(1, 2, 2);
    setup_current(cur);
    jas.evaluate_batch(cur);

    const std::vector<double> grad0(jas.grad_c().begin(), jas.grad_c().end());
    const std::vector<double> lap0(jas.lap_c().begin(), jas.lap_c().end());
    const double log0 = jas.log_psi_c()[0];
    const std::vector<double> bgrad0(cur.gradients().begin(), cur.gradients().end());

    std::vector<double> new_pos = {3.0, 0.0};
    cur.propose_move_batch(0, new_pos);
    jas.ratio_batch(cur, 0);
    std::array<bool, 1> reject = {false};
    jas.apply_move_batch(cur, 0, reject);

    CHECK(jas.log_psi_c()[0] == Catch::Approx(log0));
    for (std::size_t k = 0; k < grad0.size(); ++k)
        CHECK(jas.grad_c()[k] == Catch::Approx(grad0[k]).margin(1e-12));
    for (std::size_t k = 0; k < lap0.size(); ++k)
        CHECK(jas.lap_c()[k] == Catch::Approx(lap0[k]).margin(1e-12));
    for (std::size_t k = 0; k < bgrad0.size(); ++k)
        CHECK(cur.gradients()[k] == Catch::Approx(bgrad0[k]).margin(1e-12));
}

// ---------------------------------------------------------------------------
//  Rebuild tests (wavefunction optimisation support)
// ---------------------------------------------------------------------------

TEST_CASE("PadeJastrow rebuild changes the wavefunction after parameter update",
          "[pade_jastrow][rebuild]") {
    // Start with u(r) = r (a=1). Evaluate, then change a → 2,
    // rebuild, and confirm evaluate_batch now gives 2× the log-psi.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;  // p1 at (2,0), r_01 = 2
    ws.dist_tables().build_all(ws.positions());

    PadeJastrow jas(/*num_order=*/1, /*den_order=*/1,
                    group(0, 1), group(1, 2), linear_u());
    jas.evaluate_batch(ws);
    REQUIRE(jas.log_psi_c()[0] == Catch::Approx(2.0));  // u(r)=r → u(2)=2

    // Mutate the parameter and rebuild the Padé kernel.
    jas.parameters()[0].value = 2.0;
    jas.rebuild();

    // Re-evaluate: u(r) = 2r → u(2) = 4.
    jas.evaluate_batch(ws);
    CHECK(jas.log_psi_c()[0] == Catch::Approx(4.0));
}

TEST_CASE("PadeJastrow rebuild preserves results when parameters are unchanged",
          "[pade_jastrow][rebuild]") {
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());

    PadeJastrow jas(1, 1, group(0, 1), group(1, 2), linear_u());
    jas.evaluate_batch(ws);
    const double before = jas.log_psi_c()[0];

    jas.rebuild();
    jas.evaluate_batch(ws);
    CHECK(jas.log_psi_c()[0] == Catch::Approx(before));
}

TEST_CASE("PadeJastrow clone preserves rebuild state",
          "[pade_jastrow][rebuild]") {
    // After rebuilding, clone() should carry the updated kernel.
    PadeJastrow jas(1, 1, group(0, 1), group(1, 2), linear_u());
    jas.parameters()[0].value = 3.0;
    jas.rebuild();

    auto cloned = jas.clone();
    auto* pade_clone = dynamic_cast<PadeJastrow*>(cloned.get());
    REQUIRE(pade_clone != nullptr);

    WalkerSet ws(1, 2, 2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());

    pade_clone->evaluate_batch(ws);
    // u(r) = 3r → u(2) = 6.
    CHECK(pade_clone->log_psi_c()[0] == Catch::Approx(6.0));
}

// ---------------------------------------------------------------------------
//  Parameter derivative tests
// ---------------------------------------------------------------------------

namespace {

// Linear Jastrow with a1 non-fixed: u(r) = a1 * r,  ∂ log ψ /∂ a1 = Σ r_ij.
std::vector<Parameter> linear_u_opt() {
    return {{"a1", 1.0, false}, {"b1", 0.0, true}};
}

}  // namespace

TEST_CASE("PadeJastrow derivative for linear u(r)=a*r equals sum of distances",
          "[pade_jastrow][derivatives]") {
    // Single walker, 2 particles at r=2: ∂ log ψ / ∂a1 = r = 2.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());

    PadeJastrow jas(1, 1, group(0, 1), group(1, 2), linear_u_opt());
    jas.evaluate_batch(ws);

    const std::size_t n_opt = jas.num_optimizable_params();
    REQUIRE(n_opt == 1);

    std::vector<double> derivs(n_opt * ws.num_walkers());
    jas.compute_parameter_derivatives(ws, derivs);

    CHECK(derivs[0] == Catch::Approx(2.0));  // d log ψ / da1 = r = 2
}

TEST_CASE("PadeJastrow derivatives match finite differences",
          "[pade_jastrow][derivatives]") {
    // Use u(r) = (a1*r + a2*r²) / (1 + b1*r).
    // Check ∂ log ψ / ∂a1, ∂ log ψ / ∂a2, ∂ log ψ / ∂b1 against
    // (log ψ(p+ε) - log ψ(p-ε)) / (2ε).
    std::vector<Parameter> params = {
        {"a1", 1.5, false},
        {"a2", 0.3, false},
        {"b1", 0.2, false}
    };

    // Single walker, positions set up so r_01 = 2.5.
    // (PadeJastrow layout: pos[p*dim*nw + d*nw + w])
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    // p0 at (0,0), p1 at (2.5, 0): r = 2.5.
    pos[2] = 2.5;  // p1, x, w0
    ws.dist_tables().build_all(ws.positions());

    PadeJastrow jas(/*num_order=*/2, /*den_order=*/1,
                    group(0, 1), group(1, 2), params);
    jas.evaluate_batch(ws);

    const std::size_t n_opt = jas.num_optimizable_params();
    REQUIRE(n_opt == 3);

    std::vector<double> derivs(n_opt * ws.num_walkers());
    jas.compute_parameter_derivatives(ws, derivs);

    const double eps = 1e-6;
    for (std::size_t k = 0; k < n_opt; ++k) {
        // Perturb p_k forward
        jas.parameters()[k].value += eps;
        jas.rebuild();
        jas.evaluate_batch(ws);
        std::vector<double> log_plus(ws.num_walkers());
        for (std::size_t w = 0; w < ws.num_walkers(); ++w)
            log_plus[w] = jas.log_psi_c()[w];

        // Perturb p_k backward
        jas.parameters()[k].value -= 2.0 * eps;
        jas.rebuild();
        jas.evaluate_batch(ws);
        std::vector<double> log_minus(ws.num_walkers());
        for (std::size_t w = 0; w < ws.num_walkers(); ++w)
            log_minus[w] = jas.log_psi_c()[w];

        // Restore
        jas.parameters()[k].value += eps;

        for (std::size_t w = 0; w < ws.num_walkers(); ++w) {
            const double fd = (log_plus[w] - log_minus[w]) / (2.0 * eps);
            const double analytical = derivs[k * ws.num_walkers() + w];
            CHECK(analytical == Catch::Approx(fd).margin(1e-5));
        }
    }
}

TEST_CASE("PadeJastrow fixed parameters contribute zero optimisable params",
          "[pade_jastrow][derivatives]") {
    // All parameters fixed → zero optimisable parameters.
    PadeJastrow jas(1, 1, group(0, 1), group(1, 2), linear_u());  // both fixed
    CHECK(jas.num_optimizable_params() == 0);
}

TEST_CASE("PadeJastrow derivatives work with multiple walkers",
          "[pade_jastrow][derivatives]") {
    // 2 walkers, different distances. Verify per-walker derivatives.
    // Layout: pos[p*dim*nw + d*nw + w], nw=2, dim=2.
    WalkerSet ws(/*nw=*/2, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    // w0: p0=(0,0), p1=(2,0) → r=2
    pos[4] = 2.0;  // p1, x, w0
    // w1: p0=(0,0), p1=(3,0) → r=3
    pos[5] = 3.0;  // p1, x, w1
    ws.dist_tables().build_all(ws.positions());

    PadeJastrow jas(1, 1, group(0, 1), group(1, 2), linear_u_opt());
    jas.evaluate_batch(ws);

    std::vector<double> derivs(2);
    jas.compute_parameter_derivatives(ws, derivs);

    // d log ψ / da1 = r → 2 for w0, 3 for w1.
    CHECK(derivs[0] == Catch::Approx(2.0));  // w0
    CHECK(derivs[1] == Catch::Approx(3.0));  // w1
}

TEST_CASE("PadeJastrow parameter gradient derivatives match finite differences",
          "[pade_jastrow][derivatives]") {
    // Three particles in one group, two walkers, a (2,2) Padé with a_2 fixed so
    // the non-fixed → output-slot mapping is exercised for both a_k and b_k.
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
    const std::vector<double> base = {0.5, -0.07, 0.8, 0.12};  // a1 a2 b1 b2
    const bool fixed[4] = {false, true, false, false};
    auto make = [&](const std::vector<double>& v) {
        std::vector<Parameter> par;
        const char* names[4] = {"a1", "a2", "b1", "b2"};
        for (int i = 0; i < 4; ++i) par.push_back({names[i], v[i], fixed[i]});
        return PadeJastrow(2, 2, group(0, 3), group(0, 3), par);
    };

    WalkerSet ws(nw, np, dim);
    place(ws);
    PadeJastrow jas = make(base);
    jas.evaluate_batch(ws);
    REQUIRE(jas.num_optimizable_params() == 3);
    REQUIRE(jas.supports_parameter_gradient_derivatives());

    const std::size_t block = np * dim * nw;
    std::vector<double> dg(3 * block), dl(3 * block);
    REQUIRE(jas.compute_parameter_gradient_derivatives(ws, dg, dl));

    const std::size_t live[3] = {0, 2, 3};
    const double h = 1e-6;
    for (std::size_t s = 0; s < 3; ++s) {
        auto up = base, dn = base;
        up[live[s]] += h;
        dn[live[s]] -= h;
        WalkerSet wu(nw, np, dim), wd(nw, np, dim);
        place(wu);
        place(wd);
        PadeJastrow ju = make(up), jd = make(dn);
        ju.evaluate_batch(wu);
        jd.evaluate_batch(wd);
        for (std::size_t i = 0; i < block; ++i) {
            const double fd_g = (ju.grad_c()[i] - jd.grad_c()[i]) / (2 * h);
            const double fd_l = (ju.lap_c()[i] - jd.lap_c()[i]) / (2 * h);
            CHECK(dg[s * block + i] == Catch::Approx(fd_g).epsilon(1e-5).margin(1e-7));
            CHECK(dl[s * block + i] == Catch::Approx(fd_l).epsilon(1e-5).margin(1e-7));
        }
    }
}
