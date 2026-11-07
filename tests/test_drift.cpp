// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>

#include "numerics/green_function.h"

using alea::numerics::langevin_log_ratio_d;
using alea::numerics::langevin_log_ratio_drift_d;
using alea::numerics::umrigar_drift_scale;

TEST_CASE("umrigar_drift_scale returns tau/m for a vanishing velocity",
          "[drift]") {
    CHECK(umrigar_drift_scale(0.01, 0.0) == Catch::Approx(0.01));
    CHECK(umrigar_drift_scale(0.01, 1e-12) == Catch::Approx(0.01));
    // Small but finite velocity stays within O(v) of tau/m.
    CHECK(umrigar_drift_scale(0.01, 1e-4) == Catch::Approx(0.01).epsilon(1e-3));
}

TEST_CASE("umrigar_drift_scale matches the closed form", "[drift]") {
    // tau/m = 0.5, v_sq = 2, a = 1  ->  s = (-1 + sqrt(3)) / 2.
    CHECK(umrigar_drift_scale(0.5, 2.0, 1.0) ==
          Catch::Approx((-1.0 + std::sqrt(3.0)) / 2.0));
}

TEST_CASE("umrigar_drift_scale bounds the drift step at a node", "[drift]") {
    const double tm    = 0.01;
    const double bound = std::sqrt(2.0 * tm);  // |s*g| -> sqrt(2 tau/m) as |g| -> inf
    for (double vsq : {1e4, 1e6, 1e8, 1e10}) {
        const double step = umrigar_drift_scale(tm, vsq) * std::sqrt(vsq);
        CHECK(step < bound);          // never exceeds the asymptotic cap
        CHECK(step > 0.9 * bound);    // and approaches it for large velocity
    }
    // The scale itself decreases monotonically with velocity.
    CHECK(umrigar_drift_scale(tm, 1.0) > umrigar_drift_scale(tm, 10.0));
    CHECK(umrigar_drift_scale(tm, 10.0) > umrigar_drift_scale(tm, 100.0));
}

TEST_CASE("langevin_log_ratio_drift_d reduces to the bare-drift form",
          "[drift]") {
    // With df = (tau/m) g and db = (tau/m) g', the generalized form must equal
    // the original langevin_log_ratio_d (no rescaling).
    const double tm = 0.02, delta = 0.3, g = 1.5, gp = -0.7;
    CHECK(langevin_log_ratio_drift_d(delta, tm * g, tm * gp, tm) ==
          Catch::Approx(langevin_log_ratio_d(delta, g, gp, tm)));
}

TEST_CASE("langevin_log_ratio_drift_d is 0 at tau/m = 0 (no move)", "[drift]") {
    // τ=0: drift and Δ are 0; the helper must return 0, not 0/0 = NaN.
    CHECK(langevin_log_ratio_drift_d(0.0, 0.0, 0.0, 0.0) == 0.0);
}
