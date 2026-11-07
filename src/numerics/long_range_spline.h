// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <vector>

namespace alea::numerics {

// Hot-loop evaluator for the long-range half f_long(r) produced by the
// optimized breakup, valid INSIDE r_cut.
//
// Beyond r_cut it returns 0, which is NOT f_long there — f_long equals f(r)
// outside the cutoff, so that f_short = f − f_long vanishes.  The caller must
// therefore mask: f_short = (r < r_cut) ? (f(r) − eval(r)) : 0.  Subtracting
// an unmasked eval() adds the FULL potential for every distant pair instead of
// nothing.  Returning f(r) here instead would drag the potential into a hot
// loop that is deliberately free of it.
//
// ── Why not the Chebyshev table used by the erfc splitting ─────────────────
// That table works because an erfc-damped V_long is analytic, so Chebyshev
// converges exponentially.  The breakup's f_long is a quintic Hermite spline:
// C² with a jump in the third derivative at every knot, so Chebyshev converges
// only algebraically.  Measured on a 20-knot fit:
//
//     NC = 32   9.8e-10      NC = 128   3.1e-10
//     NC = 64   1.0e-09      NC = 256   5.4e-11
//
// Reaching 1e-11 — the level at which it stops dominating the breakup's own
// ~7e-10 — would take NC ≈ 500, which is not a hot-loop option.  Worse, any
// residual here is not a rounding error but an INCONSISTENCY: the reciprocal
// sum uses f̃_long from the exact spline, so a real-space f_long that differs
// from it breaks the identity f = f_short + f_long and leaves an error that no
// amount of k-space resolution removes.
//
// So the spline is evaluated directly, which is exact.  Inside a panel only six
// basis functions are non-zero — three from each end knot — so this is six
// quintics, comparable in flops to a 32-term Clenshaw.
struct LongRangeSpline {
    // Coefficients pre-scaled by δ^α, laid out 3 per knot, so evaluation needs
    // no pow() and no per-α branch.
    std::vector<double> scaled;
    double r_cut    = 0.0;
    double inv_delta = 0.0;
    int    n_knots  = 0;

    [[nodiscard]] double eval(double r) const noexcept {
        if (r >= r_cut) {
            return 0.0;
        }
        const double t = r * inv_delta;
        int i = static_cast<int>(t);
        if (i > n_knots - 2) {
            i = n_knots - 2;
        }
        const double x = t - i;
        const double y = 1.0 - x;
        const double* __restrict a = scaled.data() + 3 * i;   // near knot
        const double* __restrict b = a + 3;                   // far knot

        // P₀ = 1 − 10x³ + 15x⁴ − 6x⁵,  P₁ = x − 6x³ + 8x⁴ − 3x⁵,
        // P₂ = x²/2 − 3x³/2 + 3x⁴/2 − x⁵/2.
        const double x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x;
        const double y2 = y * y, y3 = y2 * y, y4 = y3 * y, y5 = y4 * y;
        const double p0 = 1.0 - 10.0 * x3 + 15.0 * x4 - 6.0 * x5;
        const double p1 = x - 6.0 * x3 + 8.0 * x4 - 3.0 * x5;
        const double p2 = 0.5 * x2 - 1.5 * x3 + 1.5 * x4 - 0.5 * x5;
        const double q0 = 1.0 - 10.0 * y3 + 15.0 * y4 - 6.0 * y5;
        const double q1 = y - 6.0 * y3 + 8.0 * y4 - 3.0 * y5;
        const double q2 = 0.5 * y2 - 1.5 * y3 + 1.5 * y4 - 0.5 * y5;

        // The far knot is approached from its left half, where the basis is
        // mirrored: parity (+1, −1, +1) for (value, slope, curvature).
        return a[0] * p0 + a[1] * p1 + a[2] * p2 +
               b[0] * q0 - b[1] * q1 + b[2] * q2;
    }

    // f_long, f_long′ and f_long″ in one pass (all 0 at and beyond r_cut, the
    // same masking contract as eval(r)).  What a split pair Jastrow needs for
    // the gradient and Laplacian of its short-range half.
    //
    // With x the position in the panel and y = 1 − x, d/dr = (1/δ) d/dx and
    // dy/dx = −1, so the far-knot terms enter the slope as −q′(y) and the
    // curvature as +q″(y).  The coefficients are pre-scaled by δ^α, so the
    // chain rule is the only δ that appears.
    void eval(double r, double& f, double& df, double& d2f) const noexcept {
        if (r >= r_cut) {
            f = df = d2f = 0.0;
            return;
        }
        const double t = r * inv_delta;
        int i = static_cast<int>(t);
        if (i > n_knots - 2) {
            i = n_knots - 2;
        }
        const double x = t - i;
        const double y = 1.0 - x;
        const double* __restrict a = scaled.data() + 3 * i;
        const double* __restrict b = a + 3;

        const double x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x;
        const double y2 = y * y, y3 = y2 * y, y4 = y3 * y, y5 = y4 * y;
        const double p0 = 1.0 - 10.0 * x3 + 15.0 * x4 - 6.0 * x5;
        const double p1 = x - 6.0 * x3 + 8.0 * x4 - 3.0 * x5;
        const double p2 = 0.5 * x2 - 1.5 * x3 + 1.5 * x4 - 0.5 * x5;
        const double q0 = 1.0 - 10.0 * y3 + 15.0 * y4 - 6.0 * y5;
        const double q1 = y - 6.0 * y3 + 8.0 * y4 - 3.0 * y5;
        const double q2 = 0.5 * y2 - 1.5 * y3 + 1.5 * y4 - 0.5 * y5;
        // P′ and P″ (the same polynomials, differentiated in their argument).
        const double dp0 = -30.0 * x2 + 60.0 * x3 - 30.0 * x4;
        const double dp1 = 1.0 - 18.0 * x2 + 32.0 * x3 - 15.0 * x4;
        const double dp2 = x - 4.5 * x2 + 6.0 * x3 - 2.5 * x4;
        const double dq0 = -30.0 * y2 + 60.0 * y3 - 30.0 * y4;
        const double dq1 = 1.0 - 18.0 * y2 + 32.0 * y3 - 15.0 * y4;
        const double dq2 = y - 4.5 * y2 + 6.0 * y3 - 2.5 * y4;
        const double ddp0 = -60.0 * x + 180.0 * x2 - 120.0 * x3;
        const double ddp1 = -36.0 * x + 96.0 * x2 - 60.0 * x3;
        const double ddp2 = 1.0 - 9.0 * x + 18.0 * x2 - 10.0 * x3;
        const double ddq0 = -60.0 * y + 180.0 * y2 - 120.0 * y3;
        const double ddq1 = -36.0 * y + 96.0 * y2 - 60.0 * y3;
        const double ddq2 = 1.0 - 9.0 * y + 18.0 * y2 - 10.0 * y3;

        f = a[0] * p0 + a[1] * p1 + a[2] * p2 +
            b[0] * q0 - b[1] * q1 + b[2] * q2;
        df = inv_delta * (a[0] * dp0 + a[1] * dp1 + a[2] * dp2 -
                          (b[0] * dq0 - b[1] * dq1 + b[2] * dq2));
        d2f = inv_delta * inv_delta *
              (a[0] * ddp0 + a[1] * ddp1 + a[2] * ddp2 +
               b[0] * ddq0 - b[1] * ddq1 + b[2] * ddq2);
    }
};

}  // namespace alea::numerics
