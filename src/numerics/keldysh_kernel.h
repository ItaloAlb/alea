// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cmath>

#include "special_functions.h"  // stvh0, PI

namespace alea::numerics {

[[gnu::always_inline]] inline double keldysh_clenshaw(double xi,
    const double* __restrict c);

// Six terms of DLMF 11.6.1 at nu=0 (https://dlmf.nist.gov/11.6.E1).
// K=(2/pi)/x * sum_n (-1)^n [(2n-1)!!]^2/x^(2n), truncated at n=5.
// Used only for x>=32: the first omitted term bounds the value remainder.
[[gnu::always_inline]] inline double keldysh_tail(double x) noexcept {
    const double inv=1.0/x, u=inv*inv;
    return (2.0/PI)*inv*(1.0+u*(-1.0+u*(9.0+u*(-225.0+u*(11025.0-893025.0*u)))));
}

inline double keldysh_tail_d1(double x) noexcept {
    const double inv=1.0/x, u=inv*inv;
    return (2.0/PI)*inv*inv*(-1.0+u*(3.0+u*(-45.0+u*(1575.0+u*(-99225.0+9823275.0*u)))));
}

inline double keldysh_tail_d2(double x) noexcept {
    const double inv=1.0/x, u=inv*inv;
    return (2.0/PI)*inv*inv*inv*(2.0+u*(-12.0+u*(270.0+u*(-12600.0+u*(992250.0-117879300.0*u)))));
}

// Vectorizable approximation of the Rytova–Keldysh radial kernel
//
//   K(x) = H0(x) - Y0(x),   x = r / r0   (dimensionless, universal in r0).
//
// The scalar special functions (stvh0/jy0b) do not vectorize (branchy domains,
// convergence-break series).  Here K is replaced by a Chebyshev expansion in a
// stretched coordinate, evaluated by the Clenshaw recurrence — pure FMA, no
// gather, no transcendental beyond the cheap (vectorizing) sqrt of the map.
//
// Regions joined by a quintic partition of unity:
//   x < x_lo            : clamp to x_lo (retained low-radius regularization).
//   x_lo <= x <= 32     : Chebyshev in t = x^(1/p) (Clenshaw).
//   32 < x < 40         : C² blend, s(u)=u³(10-15u+6u²), u=(x-32)/8.
//   x >= 40             : six-term inverse-power tail (Coulomb limit).
// The former two-term hard switch had a value jump of about -5.74e-8.
//
// Coordinate map (knee tamer): K has a logarithmic knee as x→0 that makes a
// plain Chebyshev fit converge slowly (singularity glued to the interval edge).
// Mapping to t = x^(1/p) with p a power of two (so t = repeated sqrt, which
// vectorizes under -fno-math-errno) stretches the small-x region and pushes the
// singularity away, collapsing the degree.  Calibrated choice (max error vs the
// high-precision reference stvh0 − std::cyl_neumann over [x_lo, x_hi]):
//   p = 8, x_lo = 1e-3, x_hi = 40, NC = 24  →  ~5e-9 sampled value error.
//
// Reference for building the coefficients uses std::cyl_neumann(0,x) (C++17,
// ~1e-14) for Y0 instead of the fast jy0b (~1e-8); this runs once at startup,
// zero runtime cost.
struct KeldyshChebyshev {
    static constexpr int    NC   = 24;      // number of Chebyshev coefficients
    static constexpr double kP   = 8.0;     // coordinate map  t = x^(1/p)
    static constexpr double kXlo = 1.0e-3;  // table lower edge (clamp below)
    static constexpr double kXhi = 40.0;    // table upper edge (asymptotic above)
    static constexpr double kXjoin = 32.0;  // C² transition starts here

    std::array<double, NC> coeffs{}; // [Chebyshev order]
    double x_lo = kXlo;
    double x_hi = kXhi;
    double map_a = 0.0;  // affine t -> ξ ∈ [-1,1]:  ξ = map_a·t + map_b
    double map_b = 0.0;

    KeldyshChebyshev() {
        const double t_lo = std::pow(kXlo, 1.0 / kP);
        const double t_hi = std::pow(kXhi, 1.0 / kP);
        map_a = 2.0 / (t_hi - t_lo);
        map_b = -(t_lo + t_hi) / (t_hi - t_lo);

        // Chebyshev projection of f(t) = K_ref(t^p) on t ∈ [t_lo, t_hi].
        // Oversample (M ≫ NC) so the discrete coefficients match the continuous
        // ones to well below the target accuracy.
        constexpr int M = 8 * NC;
        std::array<double, M> f{};
        for (int j = 0; j < M; ++j) {
            const double xi = std::cos(PI * (j + 0.5) / M);
            const double t  = 0.5 * ((t_hi - t_lo) * xi + (t_hi + t_lo));
            const double x  = std::pow(t, kP);
            f[j] = keldysh_kernel(x);  // high-precision K_ref
        }
        for (int k = 0; k < NC; ++k) {
            double s = 0.0;
            for (int j = 0; j < M; ++j)
                s += f[j] * std::cos(PI * k * (j + 0.5) / M);
            coeffs[k] = 2.0 * s / M;
        }
        coeffs[0] *= 0.5;  // store c0/2 so Clenshaw closes with "+ coeffs[0]"
    }

    // One value evaluator shared by scalar, open/periodic batch and move paths.
    // Both safely clamped branches are evaluated; there is no special-function
    // call or allocation in the measurement loop.
    [[nodiscard, gnu::always_inline]] inline double eval(double x) const noexcept {
        const double xc = x < x_lo ? x_lo : (x > x_hi ? x_hi : x);
        const double t  = std::sqrt(std::sqrt(std::sqrt(xc)));  // x^(1/8)
        const double xi = map_a * t + map_b;
        const double fitted = keldysh_clenshaw(xi,coeffs.data());
        const double tail = keldysh_tail(x > kXjoin ? x : kXjoin);
        const double raw=(x-kXjoin)/(x_hi-kXjoin);
        const double u=raw < 0.0 ? 0.0 : (raw > 1.0 ? 1.0 : raw);
        const double s=u*u*u*(10.0+u*(-15.0+6.0*u));
        // Tail-first avoids cancellation of a tiny far field against the
        // clamped endpoint value when the blend has reached exactly one.
        return tail+(1.0-s)*(fitted-tail);
    }

    // Ordinary branch derivative of the SAMPLED approximation, not the ideal
    // Struve/Bessel kernel. Differentiated Clenshaw and dxi/dx=a*x^(1/8)/(8x).
    // The low clamp is constant. Include the derivative of the C² blend.
    [[nodiscard]] double d1(double x) const noexcept {
        if (x < x_lo) return 0.0;
        if (x >= x_hi) return keldysh_tail_d1(x);
        double value, first, second;
        fitted_derivatives(x,value,first,second);
        if (x <= kXjoin) return first;
        const double inv_width=1.0/(x_hi-kXjoin), u=(x-kXjoin)*inv_width;
        const double s=u*u*u*(10.0+u*(-15.0+6.0*u));
        const double ds=30.0*u*u*(1.0-u)*(1.0-u)*inv_width;
        return first+s*(keldysh_tail_d1(x)-first)+ds*(keldysh_tail(x)-value);
    }

    // Setup-time curvature for matching the sampled Ewald cutoff. At the
    // low-clamp kink use the interior convention; outside the clamp it is zero.
    [[nodiscard]] double d2(double x) const noexcept {
        if (x < x_lo) return 0.0;
        if (x >= x_hi) return keldysh_tail_d2(x);
        double value, first, second;
        fitted_derivatives(x,value,first,second);
        if (x <= kXjoin) return second;
        const double inv_width=1.0/(x_hi-kXjoin), u=(x-kXjoin)*inv_width;
        const double s=u*u*u*(10.0+u*(-15.0+6.0*u));
        const double ds=30.0*u*u*(1.0-u)*(1.0-u)*inv_width;
        const double d2s=60.0*u*(1.0-u)*(1.0-2.0*u)*inv_width*inv_width;
        return second+s*(keldysh_tail_d2(x)-second)+
            2.0*ds*(keldysh_tail_d1(x)-first)+d2s*(keldysh_tail(x)-value);
    }

private:
    void fitted_derivatives(double x, double& value, double& first,
                            double& second) const noexcept {
        const double t = std::sqrt(std::sqrt(std::sqrt(x)));
        const double xi = map_a*t+map_b;
        double b1=0.0, b2=0.0, db1=0.0, db2=0.0, d2b1=0.0, d2b2=0.0;
        for (int k=NC-1; k>=1; --k) {
            const double b0=2.0*xi*b1-b2+coeffs[k];
            const double db0=2.0*b1+2.0*xi*db1-db2;
            const double d2b0=4.0*db1+2.0*xi*d2b1-d2b2;
            b2=b1; b1=b0; db2=db1; db1=db0;
            d2b2=d2b1; d2b1=d2b0;
        }
        const double dxi=map_a*(t/(kP*x));
        const double d2xi=dxi*((1.0/kP-1.0)/x);
        value=xi*b1-b2+coeffs[0];
        first=(b1+xi*db1-db2)*dxi;
        second=(2.0*db1+xi*d2b1-d2b2)*dxi*dxi+(b1+xi*db1-db2)*d2xi;
    }
};

// Fully unrolled Clenshaw recurrence for Σ' c_k T_k(ξ) (c[0] pre-halved).
// Template recursion + always_inline produces straight-line FMAs with no loop —
// avoids a nested runtime loop in the omp-simd walker kernel. Actual SIMD
// generation depends on compiler flags and surrounding control flow.
namespace detail {
template <int K>
[[gnu::always_inline]] inline double clenshaw_rec(double two_xi, double d, double dd,
                                                  const double* __restrict c) {
    if constexpr (K >= 1)
        return clenshaw_rec<K - 1>(two_xi, two_xi * d - dd + c[K], d, c);
    else
        return 0.5 * two_xi * d - dd + c[0];  // xi·d − dd + c[0]
}
}  // namespace detail

[[gnu::always_inline]] inline double keldysh_clenshaw(double xi,
                                                      const double* __restrict c) {
    return detail::clenshaw_rec<KeldyshChebyshev::NC - 1>(2.0 * xi, 0.0, 0.0, c);
}

// Single shared instance — K(x) is universal (independent of r0/ε/charge/pair),
// so one coefficient set serves the whole program.  Read-only after the
// thread-safe static initialisation.
inline const KeldyshChebyshev& keldysh_cheb() {
    static const KeldyshChebyshev table;
    return table;
}

}  // namespace alea::numerics
