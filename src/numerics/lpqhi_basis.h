// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace alea::numerics {

// Locally Piecewise Quintic Hermite Interpolant basis, on a uniform knot mesh
// over [0, r_c].  This is the basis of the Natoli–Ceperley optimized breakup
// (J. Comput. Phys. 117, 171 (1995)); the layout follows QMCPACK's LPQHIBasis.
//
// ── Layout ──────────────────────────────────────────────────────────────────
// Three functions per knot, indexed n = 3i + α with α ∈ {0,1,2}.  Each has
// compact support [r_{i-1}, r_{i+1}] — two half-panels, with *different*
// polynomials on each half.  Quadratures against these functions must therefore
// be split at the knot: a panel straddling r_i integrates across a kink in the
// third derivative and silently loses accuracy.
//
// ── Why the coefficients are meaningful on their own ────────────────────────
// At its own knot, h(3i+0, r_i) = 1 with zero first and second derivative,
// h(3i+1, ·) has unit first derivative and nothing else, h(3i+2, ·) unit
// second.  So a expansion Σ t_n h_n(r) has
//     t_{3i} = f(r_i),   t_{3i+1} = f'(r_i),   t_{3i+2} = f''(r_i).
// Two consequences used by the breakup: the boundary condition at r_c is
// imposed by *assigning* the last three coefficients (not by a constraint),
// and f(0) — the self-energy of the long-range part — is simply t_0, because
// every other basis function vanishes at the origin.
class LpqhiBasis {
public:
    LpqhiBasis() = default;
    LpqhiBasis(int n_knots, double r_cut)
        : n_knots_(n_knots), r_cut_(r_cut),
          delta_(r_cut / (n_knots - 1)) {}

    [[nodiscard]] int n_knots() const noexcept { return n_knots_; }
    [[nodiscard]] int size() const noexcept { return 3 * n_knots_; }
    [[nodiscard]] double r_cut() const noexcept { return r_cut_; }
    [[nodiscard]] double delta() const noexcept { return delta_; }
    [[nodiscard]] double knot(int i) const noexcept { return delta_ * i; }

    // Support of basis function n, clipped to [0, r_c].
    [[nodiscard]] double support_lo(int n) const noexcept {
        return std::max(0.0, delta_ * (n / 3 - 1));
    }
    [[nodiscard]] double support_hi(int n) const noexcept {
        return std::min(r_cut_, delta_ * (n / 3 + 1));
    }

    [[nodiscard]] double h(int n, double r) const noexcept {
        const int i = n / 3, a = n - 3 * i;
        const double rb = delta_ * i;
        if (r < delta_ * (i - 1) || r > support_hi(n)) {
            return 0.0;
        }
        const bool left = (r <= rb);
        const double x = left ? (rb - r) / delta_ : (r - rb) / delta_;
        double p = 0.0;
        for (int m = 5; m >= 0; --m) {
            p = p * x + kS[a][m];
        }
        const double s = std::pow(delta_, a) * p;
        return left ? kMirror[a] * s : s;
    }

    // dh/dr.  The chain rule flips sign on the left half, where x = (r_i−r)/δ.
    [[nodiscard]] double dh_dr(int n, double r) const noexcept {
        const int i = n / 3, a = n - 3 * i;
        const double rb = delta_ * i;
        if (r < delta_ * (i - 1) || r > support_hi(n)) {
            return 0.0;
        }
        const bool left = (r <= rb);
        const double x = left ? (rb - r) / delta_ : (r - rb) / delta_;
        double dp = 0.0;
        for (int m = 5; m >= 1; --m) {
            dp = dp * x + m * kS[a][m];
        }
        const double s = std::pow(delta_, a) * dp / delta_;
        return left ? -kMirror[a] * s : s;
    }

    // d²h/dr².  The chain rule squares, so the mirrored half keeps its sign
    // and only the parity factor survives.  h is C² but not C³: this is
    // continuous across a knot while its own derivative kinks there, which is
    // why a central difference of dh_dr across a knot converges only as O(h/δ³)
    // and is useless as a check.
    [[nodiscard]] double d2h_dr2(int n, double r) const noexcept {
        const int i = n / 3, a = n - 3 * i;
        const double rb = delta_ * i;
        if (r < delta_ * (i - 1) || r > support_hi(n)) {
            return 0.0;
        }
        const bool left = (r <= rb);
        const double x = left ? (rb - r) / delta_ : (r - rb) / delta_;
        double d2p = 0.0;
        for (int m = 5; m >= 2; --m) {
            d2p = d2p * x + m * (m - 1) * kS[a][m];
        }
        const double s = std::pow(delta_, a) * d2p / (delta_ * delta_);
        return left ? kMirror[a] * s : s;
    }

private:
    // Quintic Hermite coefficients, in powers of the local variable x.
    static constexpr double kS[3][6] = {{1, 0, 0, -10, 15, -6},
                                        {0, 1, 0, -6, 8, -3},
                                        {0, 0, 0.5, -1.5, 1.5, -0.5}};
    // Parity of each function under x → −x, i.e. reflection about the knot.
    static constexpr double kMirror[3] = {1, -1, 1};

    int    n_knots_ = 0;
    double r_cut_   = 0.0;
    double delta_   = 0.0;
};

}  // namespace alea::numerics
