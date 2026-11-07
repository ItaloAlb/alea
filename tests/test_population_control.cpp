// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// DMC population control (qmc/population_control.h): the E_ref-centred branch
// weight of QMCPACK SFNBranch::branchWeight and the E_T feedback of Umrigar,
// Nightingale & Runge, JCP 99, 2865 (1993), and the branch cutoff e_cut,
// checked on plain numbers.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "qmc/population_control.h"

using alea::BranchCutoff;
using alea::PopulationControl;
using Catch::Approx;

TEST_CASE("branch_weight: the QMCPACK window is centred on E_ref", "[population_control][dmc]") {
    //   x = max(E_ref − E_new, E_ref − E_old)
    //   τ' = τ/2, tapered linearly to 0 over e_cut < x ≤ e_max = 1.5·e_cut
    //   w = exp(τ' · (2·E_T − E_new − E_old)),   e_cut = 2/√τ (DRV)
    const double tau   = 0.01;
    const double e_cut = 2.0 / std::sqrt(tau);  // = 20, e_max = 30

    PopulationControl pc(tau, /*n_substeps=*/1, /*generations=*/10.0, /*target=*/2,
                         BranchCutoff::drv);
    pc.start(/*energy=*/-1.0);  // E_T = E_ref = -1
    CHECK(pc.energy_cutoff() == e_cut);
    CHECK(pc.etrial() == Approx(-1.0));

    auto bare = [&](double et, double en, double eo) {
        return std::exp(tau * (et - 0.5 * (en + eo)));
    };

    // Inside the window → bare formula (spawn below E_T, kill above).
    CHECK(pc.branch_weight(-1.0, -1.0) == Approx(1.0));
    CHECK(pc.branch_weight(-1.5, -1.0) == Approx(bare(-1.0, -1.5, -1.0)));
    CHECK(pc.branch_weight(-0.5, -1.0) == Approx(bare(-1.0, -0.5, -1.0)));
    CHECK(pc.branch_weight(-5.0, -1.0) == Approx(std::exp(tau * 2.0)));

    // Low-energy outlier in the taper: x = 25 → τ' = τ/2 · (1 − 5/10).
    CHECK(pc.branch_weight(-1.0 - 25.0, -1.0) == Approx(std::exp(0.25 * tau * (-2.0 + 27.0))));
    CHECK(e_cut < 25.0);

    // Beyond e_max: τ' = 0 → the walker neither spawns nor dies.
    CHECK(pc.branch_weight(-50.0, -1.0) == Approx(1.0));

    // The HIGH side is not filtered (as in QMCPACK): the walker simply dies.
    CHECK(pc.branch_weight(50.0, -1.0) == Approx(bare(-1.0, 50.0, -1.0)));
}

TEST_CASE("branch_weight: an E_trial swing keeps the relative weights", "[population_control][dmc]") {
    // Regression for the HEG rs=30/40 failure: with the old window centred on
    // E_trial, a population-control swing of E_T by more than e_cut clamped
    // EVERY walker to the same weight, erasing the projection.  Centred on
    // E_ref, the swing only rescales all weights by one common factor.
    const double tau = 0.01;
    PopulationControl pc(tau, /*n_substeps=*/1, /*generations=*/10.0, /*target=*/4,
                         BranchCutoff::drv);
    pc.start(-1.0);

    const double ratio_before = pc.branch_weight(-1.5, -1.5) / pc.branch_weight(-1.0, -1.0);

    // W = N·e^5 → E_T = E_ref − ln(e^5)/(g·τ) = -1 − 50: far beyond e_cut = 20.
    pc.update(/*ensemble_energy=*/-1.0, /*total_weight=*/4.0 * std::exp(5.0), /*accumulate=*/false);
    REQUIRE(pc.etrial() == Approx(-51.0));

    const double ratio_after = pc.branch_weight(-1.5, -1.5) / pc.branch_weight(-1.0, -1.0);
    CHECK(ratio_after == Approx(ratio_before));
    CHECK(ratio_after == Approx(std::exp(tau * 0.5)));
    CHECK(pc.branch_weight(-1.0, -1.0) == Approx(std::exp(-tau * 50.0)));
}

TEST_CASE("update: a population on target leaves E_trial at E_ref", "[population_control][dmc]") {
    PopulationControl pc(/*tau=*/0.01, /*n_substeps=*/1, /*generations=*/10.0, /*target=*/4);
    pc.start(-1.0);
    // W = 2 + 1 + 0.3 + 0.7 = 4 = N_target → ln(W/N) = 0.
    pc.update(/*ensemble_energy=*/-1.0, /*total_weight=*/4.0, /*accumulate=*/false);
    CHECK(pc.etrial() == Approx(-1.0).margin(1e-12));
    // No weight at all leaves E_T where it was rather than taking log(0).
    pc.update(-2.0, /*total_weight=*/0.0, /*accumulate=*/false);
    CHECK(pc.etrial() == Approx(-1.0).margin(1e-12));
}

TEST_CASE("update: E_ref tracks a running average during accumulation", "[population_control][dmc]") {
    // g = 10, τ·n_substeps = 0.01·2 → feedback rate 1/(g·τ·n_sub) = 5.
    PopulationControl pc(/*tau=*/0.01, /*n_substeps=*/2, /*generations=*/10.0, /*target=*/4);
    pc.start(-0.5);  // E_ref starts at -0.5

    // Accumulation: E_ref is the cumulative weighted-average energy.  Feeding
    // a constant E = -1 drives it to -1; with W == target the log feedback
    // vanishes, so E_T == E_ref.
    pc.begin_accumulation();
    for (int i = 0; i < 5; ++i) pc.update(-1.0, /*total_weight=*/4.0, /*accumulate=*/true);
    CHECK(pc.etrial() == Approx(-1.0));

    // Population off target → E_T = E_ref − ln(W/target)/(g·τ·n_substeps).
    pc.update(-1.0, /*total_weight=*/8.0, /*accumulate=*/true);
    CHECK(pc.etrial() == Approx(-1.0 - 5.0 * std::log(8.0 / 4.0)));

    // Equilibration: E_ref is the instantaneous ensemble energy.
    pc.update(-2.0, /*total_weight=*/4.0, /*accumulate=*/false);
    CHECK(pc.etrial() == Approx(-2.0));

    // begin_accumulation() restarts the running average.
    pc.begin_accumulation();
    pc.update(-3.0, 4.0, /*accumulate=*/true);
    CHECK(pc.etrial() == Approx(-3.0));

    // start() begins a new run: E_T = E_ref = energy, no memory of the old average.
    pc.start(-7.0);
    pc.update(-7.0, 4.0, /*accumulate=*/true);
    CHECK(pc.etrial() == Approx(-7.0));
}

TEST_CASE("generations of zero disable the E_trial feedback", "[population_control][dmc]") {
    PopulationControl pc(/*tau=*/0.01, /*n_substeps=*/1, /*generations=*/0.0, /*target=*/4);
    pc.start(-1.0);
    pc.update(-1.0, /*total_weight=*/40.0, /*accumulate=*/false);
    CHECK(pc.etrial() == Approx(-1.0));
}

// ── The branch cutoff e_cut (population_control.h) ──────────────────────────

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();

// Weights of a fixed set of steps around E_ref = E_T = e_ref, spanning the
// window, the taper and beyond, in units of `scale`.
std::vector<double> window_weights(PopulationControl& pc, double e_ref, double scale) {
    pc.start(e_ref);
    std::vector<double> w;
    for (double x : {-3.0, -0.5, 0.0, 0.7, 4.0, 9.0, 13.0, 40.0})
        w.push_back(pc.branch_weight(e_ref - x * scale, e_ref - 0.3 * scale));
    return w;
}
}  // namespace

TEST_CASE("universal cutoff: e_cut = alpha sqrt(sigma sqrt(N) / tau)",
          "[population_control][dmc]") {
    PopulationControl pc(/*tau=*/1e-3, 1, 10.0, 8, BranchCutoff::universal,
                         /*alpha=*/2.0, /*n_particles=*/16);
    // No spread yet: nothing is cut.
    CHECK(pc.energy_cutoff() == kInf);
    pc.set_energy_spread(5.0);
    CHECK(pc.energy_spread() == 5.0);
    CHECK(pc.energy_cutoff() == Approx(2.0 * std::sqrt(5.0 * 4.0 / 1e-3)));
    // In units of σ: α/√(σ₁τ), σ₁ = σ/√N.
    CHECK(pc.energy_cutoff() / 5.0 == Approx(2.0 / std::sqrt(5.0 / 4.0 * 1e-3)));
}

TEST_CASE("universal cutoff weights are independent of the energy unit; drv's are not",
          "[population_control][dmc]") {
    // Rescale the energy unit by λ: E → λE, σ → λσ, τ → τ/λ.  Every exponent
    // τ'(2E_T − E_new − E_old) is unit-free, so a cutoff that is an energy
    // leaves every weight unchanged.
    const double tau = 0.01, sigma = 3.0, e_ref = -10.0, lambda = 37.0;
    PopulationControl base(tau, 1, 10.0, 8, BranchCutoff::universal, 2.0, 4);
    PopulationControl scaled(tau / lambda, 1, 10.0, 8, BranchCutoff::universal, 2.0, 4);
    base.set_energy_spread(sigma);
    scaled.set_energy_spread(lambda * sigma);
    CHECK(scaled.energy_cutoff() == Approx(lambda * base.energy_cutoff()));
    // Steps measured in units of e_cut/10 so they cross the taper.
    const double unit = base.energy_cutoff() / 10.0;
    const auto w_base   = window_weights(base, e_ref, unit);
    const auto w_scaled = window_weights(scaled, lambda * e_ref, lambda * unit);
    for (std::size_t i = 0; i < w_base.size(); ++i)
        CHECK(w_scaled[i] == Approx(w_base[i]).epsilon(1e-12));
    // The taper and the zero region were both reached.
    CHECK(base.is_cut(e_ref - 13.0 * unit, e_ref));
    CHECK(w_base.back() == Approx(1.0));

    // drv's 2/√τ is no energy: the same walker 25 below E_ref is tapered at
    // τ = 0.01 (e_cut = 20, e_max = 30) but beyond e_max once the unit changes.
    PopulationControl drv_base(tau, 1, 10.0, 8, BranchCutoff::drv);
    PopulationControl drv_scaled(tau / lambda, 1, 10.0, 8, BranchCutoff::drv);
    drv_base.start(e_ref);
    drv_scaled.start(lambda * e_ref);
    const double w1 = drv_base.branch_weight(e_ref - 25.0, e_ref);
    const double w2 = drv_scaled.branch_weight(lambda * (e_ref - 25.0), lambda * e_ref);
    CHECK(w2 == Approx(1.0));
    CHECK(w1 != Approx(w2));
}

TEST_CASE("universal cutoff is size-consistent", "[population_control][dmc]") {
    // M independent copies: σ → √M σ, N → M N.  e_cut/σ must not move, so the
    // same fraction of walkers is cut.  drv's fixed e_cut cuts more as σ grows.
    const double tau = 0.02, sigma = 0.7;
    PopulationControl one(tau, 1, 10.0, 8, BranchCutoff::universal, 2.0, 5);
    PopulationControl many(tau, 1, 10.0, 8, BranchCutoff::universal, 2.0, 5 * 16);
    one.set_energy_spread(sigma);
    many.set_energy_spread(4.0 * sigma);
    CHECK(many.energy_cutoff() / (4.0 * sigma) == Approx(one.energy_cutoff() / sigma));
}

TEST_CASE("universal cutoff recedes as 1/sqrt(tau) in units of sigma",
          "[population_control][dmc]") {
    PopulationControl coarse(1e-2, 1, 10.0, 8, BranchCutoff::universal, 2.0, 3);
    PopulationControl fine(1e-4, 1, 10.0, 8, BranchCutoff::universal, 2.0, 3);
    coarse.set_energy_spread(1.5);
    fine.set_energy_spread(1.5);
    CHECK(fine.energy_cutoff() == Approx(10.0 * coarse.energy_cutoff()));
}

TEST_CASE("no spread, no time step, or scheme none leaves nothing to cut",
          "[population_control][dmc]") {
    // σ = 0: every walker at the same E_L (an exact trial).
    PopulationControl exact(0.01, 1, 10.0, 8, BranchCutoff::universal, 2.0, 2);
    exact.set_energy_spread(0.0);
    CHECK(exact.energy_cutoff() == kInf);
    exact.start(-1.0);
    CHECK(exact.branch_weight(-1e6, -1.0) == Approx(std::exp(0.005 * (-2.0 + 1e6 + 1.0))));
    CHECK_FALSE(exact.is_cut(-1e6, -1.0));

    PopulationControl frozen(0.0, 1, 10.0, 8, BranchCutoff::universal, 2.0, 2);
    frozen.set_energy_spread(3.0);
    CHECK(frozen.energy_cutoff() == kInf);

    PopulationControl none(0.01, 1, 10.0, 8, BranchCutoff::none);
    none.set_energy_spread(3.0);
    CHECK(none.energy_cutoff() == kInf);
    CHECK(none.energy_spread() == 3.0);   // recorded for the diagnostics

    // drv records σ but its e_cut stays 2/√τ.
    PopulationControl drv(0.04, 1, 10.0, 8, BranchCutoff::drv);
    drv.set_energy_spread(3.0);
    CHECK(drv.energy_cutoff() == 10.0);
}

TEST_CASE("is_cut flags exactly the steps the taper reaches", "[population_control][dmc]") {
    PopulationControl pc(0.01, 1, 10.0, 8, BranchCutoff::drv);   // e_cut = 20
    pc.start(0.0);
    CHECK_FALSE(pc.is_cut(-19.9, 0.0));
    CHECK(pc.is_cut(-20.1, 0.0));
    CHECK(pc.is_cut(0.0, -20.1));        // either end of the step
    CHECK_FALSE(pc.is_cut(500.0, 0.0));  // the high side is never filtered
}

TEST_CASE("PopulationControl refuses an unusable universal cutoff", "[population_control][dmc]") {
    using alea::kDefaultBranchCutoffAlpha;
    CHECK_THROWS_AS(PopulationControl(0.01, 1, 10.0, 8, BranchCutoff::universal, 0.0, 4),
                    std::invalid_argument);
    CHECK_THROWS_AS(PopulationControl(0.01, 1, 10.0, 8, BranchCutoff::universal,
                                      std::numeric_limits<double>::quiet_NaN(), 4),
                    std::invalid_argument);
    CHECK_THROWS_AS(PopulationControl(0.01, 1, 10.0, 8, BranchCutoff::universal,
                                      kDefaultBranchCutoffAlpha, 0),
                    std::invalid_argument);
    // alpha is not read by drv or none.
    CHECK_NOTHROW(PopulationControl(0.01, 1, 10.0, 8, BranchCutoff::drv, 0.0, 4));
}

// ── Suppressed reweighting (Anderson, Per & Umrigar, JCP 160, 104110 (2024)) ──

TEST_CASE("suppression_factor is (sqrt(pi)/2) erf(x)/x with its limits", "[population_control][dmc]") {
    using alea::suppression_factor;
    const double half_sqrt_pi = 0.5 * std::sqrt(3.14159265358979323846);
    CHECK(suppression_factor(0.0) == 1.0);
    // The small-x series and the erf form agree across the switch at 1e-4.
    CHECK(suppression_factor(0.99e-4) == Approx(half_sqrt_pi * std::erf(0.99e-4) / 0.99e-4)
                                             .epsilon(1e-14));
    CHECK(suppression_factor(1e-3) == Approx(1.0 - 1e-6 / 3.0).epsilon(1e-12));
    CHECK(suppression_factor(1.0) == Approx(half_sqrt_pi * std::erf(1.0)));
    // f → √π/(2x): the suppressed excursion (E_ref − E_L)·f stays finite when
    // E_L ∝ 1/d and x ∝ 1/d near a node.
    CHECK(suppression_factor(1e6) * 1e6 == Approx(half_sqrt_pi));
    CHECK(suppression_factor(kInf) == 0.0);
    // Monotone decreasing from 1 to 0.
    double prev = 1.0;
    for (double x = 0.05; x < 20.0; x *= 1.5) {
        const double f = suppression_factor(x);
        CHECK(f < prev);
        CHECK(f > 0.0);
        prev = f;
    }
}

TEST_CASE("suppressed branch_weight with f = 1 and tau_eff = tau is the bare weight",
          "[population_control][dmc]") {
    const double tau = 0.01;
    PopulationControl pc(tau, 1, 10.0, 8, BranchCutoff::drv);   // e_cut = 20
    pc.start(-1.0);
    pc.update(/*ensemble_energy=*/-1.2, /*total_weight=*/9.0, /*accumulate=*/false);
    // Inside the window, in the taper and beyond e_max, on both sides.
    for (double e_new : {-1.0, -1.5, 3.0, -25.0, -60.0, 40.0})
        for (double e_old : {-1.2, -0.7, -26.0}) {
            CHECK(pc.branch_weight(e_new, e_old, 1.0, 1.0, tau) ==
                  Approx(pc.branch_weight(e_new, e_old)).epsilon(1e-12));
            CHECK(pc.is_cut(e_new, e_old, 1.0, 1.0) == pc.is_cut(e_new, e_old));
        }
}

TEST_CASE("effective branch weight uses the diffusion time in the energy exponent",
          "[population_control][dmc][effective]") {
    // Anderson, Per & Umrigar, JCP 160, 104110 (2024), Eqs. 6-8 with f=1:
    // w = exp[tau_eff * (E_T - (E_new + E_old)/2)]. Disable the cutoff to
    // isolate time-step bookkeeping, including E_T != E_ref after feedback.
    PopulationControl pc(/*tau=*/0.36, 1, 10.0, 512, BranchCutoff::none);
    pc.start(-31.0);
    pc.update(/*ensemble_energy=*/-31.1, /*total_weight=*/600.0, false);
    const double e_trial = -31.1 - std::log(600.0 / 512.0) / 3.6;
    REQUIRE(pc.etrial() == Approx(e_trial));
    for (const double tau_eff : {0.0, 0.09, 0.3371, 0.36}) {
        const double expected = std::exp(tau_eff * (e_trial + 31.25));
        CHECK(pc.branch_weight(-31.3, -31.2, 1.0, 1.0, tau_eff) ==
              Approx(expected).epsilon(1e-12));
    }
    CHECK(pc.branch_weight(-31.3, -31.2, 1.0, 1.0, 0.3371) !=
          Approx(pc.branch_weight(-31.3, -31.2)).epsilon(1e-8));
}

TEST_CASE("suppressed branch_weight stays finite as E_L diverges at a node",
          "[population_control][dmc]") {
    // A walker at distance d from a node: E_L = E_ref ∓ A/d and |∇ ln Ψ| = 1/d,
    // so u = τ/d² (unit mass) and x = scale·τ·√u ∝ 1/d.  The bare weight
    // exp(τ·A/d) diverges (spawn side) or vanishes (kill side); the suppressed
    // one tends to exp(τ·[E_T − E_ref] ± τ·A·√π/(2·scale·τ^{3/2})), a constant.
    const double tau = 0.01, A = 0.5, e_ref = -1.0;
    PopulationControl pc(tau, 1, 10.0, 8, BranchCutoff::none);
    pc.start(e_ref);
    const double scale = alea::suppression_scale(alea::kDefaultSuppressionKappa,
                                                 /*N=*/4.0, tau, /*theta=*/0.05);
    auto weight = [&](double d, double sign) {
        const double e_l = e_ref - sign * A / d;
        const double f = alea::suppression_factor(scale * tau * std::sqrt(tau) / d);
        return pc.branch_weight(e_l, e_ref, f, 1.0, tau);
    };
    const double half_sqrt_pi = 0.5 * std::sqrt(3.14159265358979323846);
    const double limit_term = 0.5 * tau * A * half_sqrt_pi / (scale * tau * std::sqrt(tau));
    for (double sign : {+1.0, -1.0}) {
        const double w_limit = std::exp(sign * limit_term);
        CHECK(weight(1e-6, sign) == Approx(w_limit).epsilon(1e-5));
        CHECK(weight(1e-8, sign) == Approx(w_limit).epsilon(1e-7));
    }
    // The spawn side gains weight, the kill side loses it, both finitely.
    CHECK(weight(1e-8, +1.0) > 1.0);
    CHECK(weight(1e-8, -1.0) < 1.0);
    CHECK(weight(1e-8, -1.0) > 0.0);
}

TEST_CASE("the cutoff tapers the suppressed excursion, not the raw one",
          "[population_control][dmc]") {
    PopulationControl pc(0.01, 1, 10.0, 8, BranchCutoff::drv);   // e_cut = 20
    pc.start(0.0);
    CHECK(pc.is_cut(-30.0, 0.0, 1.0, 1.0));
    CHECK_FALSE(pc.is_cut(-30.0, 0.0, 0.5, 1.0));   // (0 + 30)·0.5 = 15 < 20
    CHECK(pc.is_cut(0.0, -30.0, 1.0, 0.9));          // 27 > 20, either end
}

TEST_CASE("suppression_scale: x is unit-free, size-consistent and reduces to c V tau_eff / sqrt(N)",
          "[population_control][dmc]") {
    using alea::suppression_scale;
    const double kappa = 1.1, n = 6.0, tau = 0.02, tau_eff = 0.019, theta = 0.3, u = 2.5;
    const double x = suppression_scale(kappa, n, tau, theta) * tau_eff * std::sqrt(u);
    // Energy unit scaled by λ: every time (τ, τ_eff, θ) → /λ; u = Σ (τ/m)(∂ ln Ψ)²
    // is a pure number.
    const double lambda = 37.0;
    CHECK(suppression_scale(kappa, n, tau / lambda, theta / lambda) * (tau_eff / lambda) *
              std::sqrt(u) == Approx(x));
    // M independent copies: u → M·u, N → M·N, θ unchanged.
    const double M = 16.0;
    CHECK(suppression_scale(kappa, M * n, tau, theta) * tau_eff * std::sqrt(M * u) ==
          Approx(x));
    // Unit masses: u = τ·V², and x = c·V·τ_eff/√N with c = κ/√θ (their Eq. 14).
    const double V = std::sqrt(u / tau);
    CHECK(x == Approx(kappa / std::sqrt(theta) * V * tau_eff / std::sqrt(n)));
    // No time step: no suppression; no correlation: full suppression.
    CHECK(suppression_scale(kappa, n, 0.0, theta) == 0.0);
    CHECK(suppression_scale(kappa, n, tau, 0.0) == kInf);
}
