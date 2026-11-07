// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cmath>

#include "numerics/special_functions.h"

namespace alea::numerics {

// Chebyshev table for the LONG-RANGE half of an Ewald splitting, V_long(r).
//
// ── Why V_long and not V_short ─────────────────────────────────────────────
// V_long is the inverse Hankel transform of an erfc-damped kernel, so it is
// analytic and even in r for ANY eps(k) — a Chebyshev fit converges on it
// exponentially.  V_short = V - V_long is not: it inherits whatever
// singularity the potential has at the origin.  Tabulating r*V_short instead
// does not rescue it either; that removes Coulomb's 1/r but leaves Keldysh's
// r*log(r), whose infinite derivative at the origin drags a 32-coefficient fit
// down to ~1e-4.  Measured, and a stretched coordinate only reached ~1e-5
// while wrecking the Coulomb case.
//
// Fitting V_long instead gives, at NC = 32 over the same range:
//     Coulomb 1.5e-11,  Keldysh r0=0.8  6.6e-13,  Keldysh r0=0.1  4.0e-12
//
// The cost is that the hot loop must evaluate V(r) itself and subtract, so the
// real-space kernel is templated on the potential.  That is a fair trade: V(r)
// already has a vectorized implementation for every potential in the project.
//
// The subtraction V(r) - V_long(r) cancels catastrophically at large r, where
// both tend to the same tail.  That is harmless here: what is needed is
// ABSOLUTE accuracy on a quantity that is itself of order the tolerance, and
// the lost digits are relative ones on a number heading to zero.
//
// ── Why per-setup rather than a process-wide singleton ─────────────────────
// S depends on the splitting parameter, the cell AND the potential, so unlike
// keldysh_kernel.h's universal K(x) it cannot be shared.  It is held inside the
// immutable EwaldSetup, which clone() shares by pointer across OpenMP threads —
// the ScreenedCoulombTerm pattern (screened_coulomb.h:82-89).
//
// Calibration: NC = 32 was measured to give ~1e-11 absolute for erfc over
// [0, 6.5].  build_ewald gates on the error it ACTUALLY achieves rather than
// trusting that, because a different eps(k) gives a different S.
inline constexpr int kLongRangeNC = 32;

struct LongRangeTable {
    std::array<double, kLongRangeNC> coeffs{};
    double r_max  = 0.0;  // table upper edge; inputs are clamped to it
    double map_a  = 0.0;  // affine r -> xi in [-1, 1]:  xi = map_a*r + map_b
    double map_b  = 0.0;

    // Scalar reference, for tests and for the fit's own error check.
    [[nodiscard]] double eval(double r) const noexcept {
        const double rc = r < r_max ? r : r_max;
        const double xi = map_a * rc + map_b;
        double d = 0.0, dd = 0.0, sv;
        for (int k = kLongRangeNC - 1; k >= 1; --k) {
            sv = d;
            d  = 2.0 * xi * d - dd + coeffs[k];
            dd = sv;
        }
        return xi * d - dd + coeffs[0];
    }
};

namespace detail {
// Fully unrolled Clenshaw, so the omp-simd loop over walkers contains no nested
// runtime loop (which would block the vectorizer).  Named distinctly from the
// Keldysh kernel's own recursion: the two headers are included together and a
// shared name in `detail` collides.
template <int K>
[[gnu::always_inline]] inline double sr_clenshaw_rec(double two_xi, double d,
                                                     double dd,
                                                     const double* __restrict c) {
    if constexpr (K >= 1)
        return sr_clenshaw_rec<K - 1>(two_xi, two_xi * d - dd + c[K], d, c);
    else
        return 0.5 * two_xi * d - dd + c[0];
}
}  // namespace detail

[[gnu::always_inline]] inline double long_range_clenshaw(double xi,
                                                          const double* __restrict c) {
    return detail::sr_clenshaw_rec<kLongRangeNC - 1>(2.0 * xi, 0.0, 0.0, c);
}

}  // namespace alea::numerics
