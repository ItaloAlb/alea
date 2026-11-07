// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Nosanow one-body factor log ψ = −α Σ_i |r_i − R_i|²: the component contract
// (hand sums, finite differences of log ψ, ratio/commit against a fresh
// evaluation, crowds, permutation, clone, parameter derivatives) and two
// physical checks — periodicity of the minimum-image fold and the exact
// harmonic-oscillator ground state at α = mω/2.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "population/simulation_cell.h"
#include "population/walker_set.h"
#include "wavefunction/nosanow_jastrow.h"
#include "wavefunction/trial_wavefunction.h"

using alea::NosanowJastrow;
using alea::Parameter;
using alea::SimulationCell;
using alea::TrialWavefunction;
using alea::WalkerSet;

namespace {

// Four particles; the factor acts on particles 1..3 (first = 1), particle 0
// belongs to another group and must be left alone.  Three walkers.
constexpr std::size_t kNw = 3, kNp = 4, kDim = 2, kFirst = 1, kSites = 3;
const std::vector<double> kSiteXY = {0.0, 0.0, 1.0, 0.2, -0.4, 0.9};
constexpr double kXY[kNw][kNp][kDim] = {
    {{5.0, 5.0}, {0.1, -0.2}, {1.3, 0.1}, {-0.5, 1.4}},
    {{-3.0, 2.0}, {-0.6, 0.4}, {0.7, 0.5}, {0.2, 0.6}},
    {{0.0, 0.0}, {0.05, 0.05}, {1.0, 0.2}, {-1.2, 0.3}}};

NosanowJastrow make(double alpha, bool fixed = true,
                    const SimulationCell& cell = SimulationCell{},
                    const std::vector<double>& sites = kSiteXY) {
    return NosanowJastrow(kDim, kFirst, sites, cell,
                          {Parameter{"alpha", alpha, fixed, true}});
}

void place(WalkerSet& ws, const double (&xy)[kNw][kNp][kDim]) {
    auto pos = ws.positions();
    for (std::size_t w = 0; w < kNw; ++w)
        for (std::size_t p = 0; p < kNp; ++p)
            for (std::size_t d = 0; d < kDim; ++d)
                pos[(p * kDim + d) * kNw + w] = xy[w][p][d];
    ws.dist_tables().build_all(ws.positions());
}

double log_psi_at(NosanowJastrow jas, const double (&xy)[kNw][kNp][kDim], std::size_t w,
                  const SimulationCell& cell = SimulationCell{}) {
    WalkerSet ws(kNw, kNp, kDim, cell);
    place(ws, xy);
    jas.evaluate_batch(ws);
    return jas.log_psi_c()[w];
}

void copy_xy(const double (&from)[kNw][kNp][kDim], double (&to)[kNw][kNp][kDim]) {
    std::copy(&from[0][0][0], &from[0][0][0] + kNw * kNp * kDim, &to[0][0][0]);
}

}  // namespace

TEST_CASE("Nosanow log psi is -alpha times the summed squared distance to each own site",
          "[nosanow]") {
    const double alpha = 0.8;
    NosanowJastrow jas = make(alpha);
    WalkerSet ws(kNw, kNp, kDim);
    place(ws, kXY);
    std::fill(ws.log_psi().begin(), ws.log_psi().end(), -1.5);  // must be added to
    jas.evaluate_batch(ws);
    for (std::size_t w = 0; w < kNw; ++w) {
        double sum = 0.0;
        for (std::size_t k = 0; k < kSites; ++k) {
            const double dx = kXY[w][kFirst + k][0] - kSiteXY[2 * k];
            const double dy = kXY[w][kFirst + k][1] - kSiteXY[2 * k + 1];
            sum -= alpha * (dx * dx + dy * dy);
        }
        CHECK(jas.log_psi_c()[w] == Catch::Approx(sum).epsilon(1e-14));
        CHECK(ws.log_psi()[w] == Catch::Approx(sum - 1.5).epsilon(1e-14));
        // Particle 0 is not in the group: no gradient, no laplacian.
        for (std::size_t d = 0; d < kDim; ++d) {
            CHECK(jas.grad_c()[(0 * kDim + d) * kNw + w] == 0.0);
            CHECK(jas.lap_c()[(0 * kDim + d) * kNw + w] == 0.0);
        }
    }
}

TEST_CASE("Nosanow gradient and laplacian match finite differences of log psi",
          "[nosanow]") {
    // Open cell, and a periodic one in which particle 3 of walker 0 sits near
    // the edge, so its own site is closest through the boundary.
    const double L = 3.0;
    const SimulationCell periodic(2, std::array<double, 4>{L, 0.0, 0.5 * L,
                                                           0.5 * std::sqrt(3.0) * L});
    for (const SimulationCell& cell : {SimulationCell{}, periodic}) {
        double xy[kNw][kNp][kDim];
        copy_xy(kXY, xy);
        if (cell.is_periodic()) { xy[0][3][0] = -0.4 + L; xy[0][3][1] = 0.85; }
        NosanowJastrow jas = make(cell.is_periodic() ? 7.0 : 0.8, true, cell);
        WalkerSet ws(kNw, kNp, kDim, cell);
        place(ws, xy);
        jas.evaluate_batch(ws);
        const double h = 1e-4;
        for (std::size_t w = 0; w < kNw; ++w)
            for (std::size_t p = 0; p < kNp; ++p)
                for (std::size_t d = 0; d < kDim; ++d) {
                    double xp[kNw][kNp][kDim], xm[kNw][kNp][kDim];
                    copy_xy(xy, xp);
                    copy_xy(xy, xm);
                    xp[w][p][d] += h;
                    xm[w][p][d] -= h;
                    const double alpha = jas.alpha();
                    const double lp = log_psi_at(make(alpha, true, cell), xp, w, cell);
                    const double lm = log_psi_at(make(alpha, true, cell), xm, w, cell);
                    const std::size_t k = (p * kDim + d) * kNw + w;
                    INFO("periodic = " << cell.is_periodic() << ", w = " << w
                                       << ", p = " << p << ", d = " << d);
                    CHECK(jas.grad_c()[k] ==
                          Catch::Approx((lp - lm) / (2 * h)).epsilon(1e-8).margin(1e-9));
                    CHECK(jas.lap_c()[k] ==
                          Catch::Approx((lp - 2 * jas.log_psi_c()[w] + lm) / (h * h))
                              .epsilon(1e-5).margin(1e-5));
                }
    }
}

TEST_CASE("Nosanow is periodic: a lattice translation of a particle changes nothing",
          "[nosanow][periodic]") {
    const double L = 3.0;
    const std::array<double, 4> a{L, 0.0, 0.5 * L, 0.5 * std::sqrt(3.0) * L};
    const SimulationCell cell(2, a);
    double shifted[kNw][kNp][kDim];
    copy_xy(kXY, shifted);
    shifted[1][2][0] += a[0] - a[2];  // particle 2 of walker 1 by a₁ − a₂
    shifted[1][2][1] += a[1] - a[3];
    const double before = log_psi_at(make(7.0, true, cell), kXY, 1, cell);
    CHECK(log_psi_at(make(7.0, true, cell), shifted, 1, cell) ==
          Catch::Approx(before).epsilon(1e-13));
}

TEST_CASE("Nosanow at alpha = m omega / 2 is the exact oscillator ground state",
          "[nosanow]") {
    // Three free particles of mass 1 in a trap ω = 1.6 centered on their own
    // sites (open cell): E_L = −½ Σ (∇² log ψ + |∇ log ψ|²) + Σ ½ω²|r − R|²
    // is the constant 3 · ω (2-D: ω per particle) at every configuration.
    const double omega = 1.6;
    NosanowJastrow jas = make(0.5 * omega);
    WalkerSet ws(kNw, kNp, kDim);
    place(ws, kXY);
    jas.evaluate_batch(ws);
    for (std::size_t w = 0; w < kNw; ++w) {
        double e_l = 0.0;
        for (std::size_t k = 0; k < kSites; ++k) {
            const std::size_t p = kFirst + k;
            for (std::size_t d = 0; d < kDim; ++d) {
                const std::size_t i = (p * kDim + d) * kNw + w;
                const double del = kXY[w][p][d] - kSiteXY[k * kDim + d];
                e_l += -0.5 * (ws.laplacians()[i] + ws.gradients()[i] * ws.gradients()[i]) +
                       0.5 * omega * omega * del * del;
            }
        }
        CHECK(e_l == Catch::Approx(3.0 * omega).epsilon(1e-13));
    }
}

TEST_CASE("Nosanow ratio and commit equal a fresh evaluation, accept and reject",
          "[nosanow]") {
    const double L = 3.0;
    const SimulationCell periodic(2, std::array<double, 4>{L, 0.0, 0.5 * L,
                                                           0.5 * std::sqrt(3.0) * L});
    for (const SimulationCell& cell : {SimulationCell{}, periodic})
        for (bool accept_all : {true, false})
            for (std::size_t p : {std::size_t{0}, std::size_t{2}}) {
                const double alpha = cell.is_periodic() ? 7.0 : 0.8;
                NosanowJastrow jas = make(alpha, true, cell);
                WalkerSet ws(kNw, kNp, kDim, cell);
                place(ws, kXY);
                jas.evaluate_batch(ws);
                const std::vector<double> log0(jas.log_psi_c().begin(), jas.log_psi_c().end());
                const std::vector<double> grad0(ws.gradients().begin(), ws.gradients().end());

                double moved[kNw][kNp][kDim];
                copy_xy(kXY, moved);
                for (std::size_t w = 0; w < kNw; ++w) {
                    moved[w][p][0] += 0.3 - 0.2 * w;
                    moved[w][p][1] -= 0.1 + 0.15 * w;
                }
                std::vector<double> proposal(kDim * kNw);
                for (std::size_t d = 0; d < kDim; ++d)
                    for (std::size_t w = 0; w < kNw; ++w)
                        proposal[d * kNw + w] = moved[w][p][d];
                ws.propose_move_batch(p, proposal);
                std::fill(ws.log_ratios().begin(), ws.log_ratios().end(), 0.0);
                std::fill(ws.new_grad().begin(), ws.new_grad().end(), 0.0);
                std::fill(ws.new_lap().begin(), ws.new_lap().end(), 0.0);
                jas.ratio_batch(ws, p);

                NosanowJastrow ref = make(alpha, true, cell);
                WalkerSet wr(kNw, kNp, kDim, cell);
                place(wr, moved);
                ref.evaluate_batch(wr);
                INFO("periodic = " << cell.is_periodic() << ", p = " << p);
                for (std::size_t w = 0; w < kNw; ++w) {
                    CHECK(ws.log_ratios()[w] ==
                          Catch::Approx(ref.log_psi_c()[w] - log0[w]).epsilon(1e-13).margin(1e-14));
                    for (std::size_t d = 0; d < kDim; ++d) {
                        const std::size_t k = (p * kDim + d) * kNw + w;
                        CHECK(ws.new_grad()[d * kNw + w] == Catch::Approx(ref.grad_c()[k]).margin(1e-13));
                        CHECK(ws.new_lap()[d * kNw + w] == Catch::Approx(ref.lap_c()[k]).margin(1e-13));
                    }
                }
                std::array<bool, kNw> accept{};
                accept.fill(accept_all);
                jas.apply_move_batch(ws, p, accept);
                for (std::size_t w = 0; w < kNw; ++w)
                    CHECK(jas.log_psi_c()[w] ==
                          Catch::Approx(accept_all ? ref.log_psi_c()[w] : log0[w]).epsilon(1e-13));
                for (std::size_t k = 0; k < grad0.size(); ++k)
                    CHECK(jas.grad_c()[k] ==
                          Catch::Approx(accept_all ? ref.grad_c()[k] : grad0[k]).margin(1e-13));
            }
}

TEST_CASE("Nosanow crowd split, permutation and clone reproduce the full batch",
          "[nosanow][crowd]") {
    auto setup = [](WalkerSet& ws, TrialWavefunction& psi) {
        place(ws, kXY);
        psi.add_component(std::make_unique<NosanowJastrow>(make(0.8)));
        psi.evaluate_batch(ws);
        std::vector<double> proposal(kDim * kNw);
        for (std::size_t w = 0; w < kNw; ++w) {
            proposal[0 * kNw + w] = 0.9 + 0.1 * w;
            proposal[1 * kNw + w] = 0.3;
        }
        ws.propose_move_batch(2, proposal);
    };
    WalkerSet wf(kNw, kNp, kDim); TrialWavefunction pf; setup(wf, pf);
    WalkerSet ws(kNw, kNp, kDim); TrialWavefunction ps; setup(ws, ps);
    pf.ratio_batch(wf, 2);
    ps.ratio_batch(ws, 2, 0, 2);
    ps.ratio_batch(ws, 2, 2, 3);
    for (std::size_t w = 0; w < kNw; ++w)
        CHECK(ws.log_ratios()[w] == Catch::Approx(wf.log_ratios()[w]).epsilon(1e-15));
    std::array<bool, kNw> accept = {false, true, true};
    pf.apply_move_batch(wf, 2, accept);
    ps.apply_move_batch(ws, 2, accept, 0, 1);
    ps.apply_move_batch(ws, 2, accept, 1, 3);
    for (std::size_t w = 0; w < kNw; ++w)
        CHECK(ws.log_psi()[w] == Catch::Approx(wf.log_psi()[w]).epsilon(1e-15));

    // Permutation carries the per-particle shares: after it, a ratio must
    // still subtract the right old share.
    NosanowJastrow jas = make(0.8);
    WalkerSet wp(kNw, kNp, kDim);
    place(wp, kXY);
    jas.evaluate_batch(wp);
    const std::vector<double> log0(jas.log_psi_c().begin(), jas.log_psi_c().end());
    const std::vector<std::size_t> src = {2, 0, 0, 1};
    jas.permute_caches(src);
    REQUIRE(jas.log_psi_c().size() == src.size());
    for (std::size_t c = 0; c < src.size(); ++c) CHECK(jas.log_psi_c()[c] == log0[src[c]]);
    // The walkers as they are after the same permutation: column c holds the
    // configuration of old walker src[c].
    const std::size_t nw = src.size();
    WalkerSet wq(nw, kNp, kDim);
    for (std::size_t c = 0; c < nw; ++c)
        for (std::size_t p = 0; p < kNp; ++p)
            for (std::size_t d = 0; d < kDim; ++d)
                wq.positions()[(p * kDim + d) * nw + c] = kXY[src[c]][p][d];
    wq.dist_tables().build_all(wq.positions());
    std::vector<double> proposal(kDim * nw, 0.25);
    wq.propose_move_batch(3, proposal);
    jas.ratio_batch(wq, 3);
    for (std::size_t c = 0; c < nw; ++c) {
        const double dx_old = kXY[src[c]][3][0] - kSiteXY[4], dy_old = kXY[src[c]][3][1] - kSiteXY[5];
        const double dx_new = 0.25 - kSiteXY[4], dy_new = 0.25 - kSiteXY[5];
        CHECK(wq.log_ratios()[c] ==
              Catch::Approx(-0.8 * (dx_new * dx_new + dy_new * dy_new - dx_old * dx_old -
                                    dy_old * dy_old)).epsilon(1e-13));
    }

    // Clone: bit for bit, including an α moved by the optimizer.
    NosanowJastrow original = make(0.8, /*fixed=*/false);
    original.parameters()[0].value = 1.3;
    original.rebuild();
    auto copy = original.clone();
    WalkerSet wa(kNw, kNp, kDim), wb(kNw, kNp, kDim);
    place(wa, kXY);
    place(wb, kXY);
    original.evaluate_batch(wa);
    copy->evaluate_batch(wb);
    for (std::size_t w = 0; w < kNw; ++w) CHECK(wb.log_psi()[w] == wa.log_psi()[w]);
    for (std::size_t k = 0; k < kNp * kDim * kNw; ++k) CHECK(wb.gradients()[k] == wa.gradients()[k]);
}

TEST_CASE("Nosanow alpha derivatives match finite differences", "[nosanow][derivatives]") {
    NosanowJastrow jas = make(0.8, /*fixed=*/false);
    WalkerSet ws(kNw, kNp, kDim);
    place(ws, kXY);
    jas.evaluate_batch(ws);
    REQUIRE(jas.num_optimizable_params() == 1);
    std::vector<double> derivs(kNw);
    jas.compute_parameter_derivatives(ws, derivs);
    const std::size_t block = kNp * kDim * kNw;
    std::vector<double> dg(block), dl(block);
    REQUIRE(jas.compute_parameter_gradient_derivatives(ws, dg, dl));

    const double h = 1e-6;
    NosanowJastrow up = make(0.8 + h), dn = make(0.8 - h);
    WalkerSet wu(kNw, kNp, kDim), wd(kNw, kNp, kDim);
    place(wu, kXY);
    place(wd, kXY);
    up.evaluate_batch(wu);
    dn.evaluate_batch(wd);
    for (std::size_t w = 0; w < kNw; ++w)
        CHECK(derivs[w] ==
              Catch::Approx((up.log_psi_c()[w] - dn.log_psi_c()[w]) / (2 * h)).epsilon(1e-8));
    for (std::size_t k = 0; k < block; ++k) {
        CHECK(dg[k] == Catch::Approx((up.grad_c()[k] - dn.grad_c()[k]) / (2 * h)).margin(1e-8));
        CHECK(dl[k] == Catch::Approx((up.lap_c()[k] - dn.lap_c()[k]) / (2 * h)).margin(1e-8));
    }
    CHECK(make(0.8, /*fixed=*/true).num_optimizable_params() == 0);
}

TEST_CASE("Nosanow refuses a non-positive alpha and a fold that is not negligible",
          "[nosanow]") {
    CHECK_THROWS_AS(make(0.0), std::invalid_argument);
    CHECK_THROWS_AS(make(-1.0), std::invalid_argument);
    // Hexagonal cell of side 3: R_WS = 1.5.  exp(−α R_WS²) ≤ 1e-6 needs
    // α ≥ ln(1e6)/2.25 ≈ 6.14.
    const SimulationCell cell(2, std::array<double, 4>{3.0, 0.0, 1.5, 1.5 * std::sqrt(3.0)});
    CHECK_THROWS_AS(make(6.0, true, cell), std::invalid_argument);
    CHECK_NOTHROW(make(6.3, true, cell));
    NosanowJastrow jas = make(7.0, /*fixed=*/false, cell);
    jas.parameters()[0].value = 2.0;  // an optimizer step out of range
    CHECK_THROWS_AS(jas.rebuild(), std::runtime_error);
    // An open cell has no fold: any α > 0 is fine.
    CHECK_NOTHROW(make(1e-3));
}
