// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

#include "numerics/lpqhi_basis.h"

using alea::numerics::LpqhiBasis;
using Catch::Matchers::WithinAbs;

TEST_CASE("LPQHI coefficients are value, slope and curvature at the knot",
          "[lpqhi]") {
    // The breakup leans on this twice: it pins the boundary condition at r_c by
    // *assigning* the last three coefficients, and reads the long-range
    // self-energy straight off t_0.  Both would fail silently, not loudly.
    const LpqhiBasis b(12, 1.0);
    for (int i : {1, 4, 7, 10}) {
        const double r = b.knot(i);
        for (int a = 0; a < 3; ++a) {
            const int n = 3 * i + a;
            INFO("knot " << i << ", alpha " << a);
            CHECK_THAT(b.h(n, r),       WithinAbs(a == 0 ? 1.0 : 0.0, 1e-13));
            CHECK_THAT(b.dh_dr(n, r),   WithinAbs(a == 1 ? 1.0 : 0.0, 1e-13));
            CHECK_THAT(b.d2h_dr2(n, r), WithinAbs(a == 2 ? 1.0 : 0.0, 1e-13));
        }
    }
}

TEST_CASE("LPQHI functions vanish outside their two half-panels", "[lpqhi]") {
    const LpqhiBasis b(10, 1.0);
    for (int n = 0; n < b.size(); ++n) {
        const double lo = b.support_lo(n), hi = b.support_hi(n);
        for (double r = 0.0; r <= 1.0; r += 0.002) {
            if (r < lo - 1e-12 || r > hi + 1e-12) {
                CHECK(b.h(n, r) == 0.0);
                CHECK(b.dh_dr(n, r) == 0.0);
            }
        }
    }
}

TEST_CASE("LPQHI: only the first function survives at the origin", "[lpqhi]") {
    // f_long(0) = Σ t_n h_n(0) = t_0, so the breakup's self-energy term
    // −½ Σ q_i² f_long(0) collapses to a single coefficient.
    const LpqhiBasis b(15, 1.0);
    CHECK_THAT(b.h(0, 0.0), WithinAbs(1.0, 1e-15));
    for (int n = 1; n < b.size(); ++n) {
        INFO("n = " << n);
        CHECK_THAT(b.h(n, 0.0), WithinAbs(0.0, 1e-14));
    }
}

TEST_CASE("LPQHI dh_dr agrees with a finite difference of h", "[lpqhi]") {
    // Catches a sign slip on the mirrored half, where the chain rule flips.
    // Sampled off the knots: dh_dr is smooth inside each half-panel.
    const LpqhiBasis b(9, 1.0);
    const double h = 1e-6;
    double worst = 0.0;
    for (int n = 0; n < b.size(); ++n) {
        for (double r = 0.05; r < 0.95; r += 0.0037) {
            const double fd = (b.h(n, r + h) - b.h(n, r - h)) / (2 * h);
            worst = std::max(worst, std::abs(fd - b.dh_dr(n, r)));
        }
    }
    INFO("max |fd - dh_dr| = " << worst);
    CHECK(worst < 1e-7);
}

TEST_CASE("LPQHI reconstructs a smooth function to sixth order", "[lpqhi]") {
    // The property that actually matters: load the coefficients with
    // (f, f', f'') at each knot and the expansion must interpolate f.  A
    // quintic Hermite mesh converges as δ⁶, so halving the spacing must buy
    // about 64x.  This is a far stronger statement than continuity, and it
    // exercises value, slope and curvature functions together.
    auto f   = [](double r) { return std::exp(-1.3 * r) * std::cos(2.1 * r); };
    auto df  = [](double r) { return -std::exp(-1.3 * r) *
                              (1.3 * std::cos(2.1 * r) + 2.1 * std::sin(2.1 * r)); };
    auto d2f = [](double r) { return std::exp(-1.3 * r) *
                              ((1.69 - 4.41) * std::cos(2.1 * r) +
                               2 * 1.3 * 2.1 * std::sin(2.1 * r)); };

    std::vector<double> err;
    for (int nk : {9, 17, 33}) {
        const LpqhiBasis b(nk, 1.0);
        std::vector<double> t(b.size());
        for (int i = 0; i < b.n_knots(); ++i) {
            t[3 * i + 0] = f(b.knot(i));
            t[3 * i + 1] = df(b.knot(i));
            t[3 * i + 2] = d2f(b.knot(i));
        }
        auto eval = [&](double r) {
            double s = 0.0;
            for (int n = 0; n < b.size(); ++n) s += t[n] * b.h(n, r);
            return s;
        };
        // interpolates exactly at the knots
        for (int i = 0; i < b.n_knots(); ++i) {
            CHECK_THAT(eval(b.knot(i)), WithinAbs(f(b.knot(i)), 1e-12));
        }
        double worst = 0.0;
        for (double r = 0.0; r <= 1.0; r += 1.0 / 4001) {
            worst = std::max(worst, std::abs(eval(r) - f(r)));
        }
        err.push_back(worst);
    }
    INFO("errors: " << err[0] << ", " << err[1] << ", " << err[2]);
    // δ⁶ would be 64x per halving; demand a clear 30x so the test is not
    // brittle against the boundary panels, which see a one-sided stencil.
    CHECK(err[0] / err[1] > 30.0);
    CHECK(err[1] / err[2] > 30.0);
}
