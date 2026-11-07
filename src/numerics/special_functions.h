// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>

namespace alea::numerics {

inline constexpr double PI = std::numbers::pi_v<double>;

inline constexpr long double PI_L = std::numbers::pi_v<long double>;

// Modified Bessel function of the second kind K_ν(x), x > 0, any real ν, from
// the integral representation (DLMF 10.32.9)
//
//     K_ν(x) = ∫₀^∞ e^{−x cosh t} cosh(νt) dt.
//
// The integrand is even and analytic in t, so the trapezoidal rule on [0, ∞)
// converges geometrically in 1/h (an error of order e^{−π²/h}): h = 0.05 is at
// machine precision, and the sum stops once the terms underflow.  Setup-time
// use only (Hankel transforms of the Jastrow tail model), not a hot loop.
inline double bessel_k(double nu, double x) {
    constexpr double h = 0.05;
    double s = 0.5 * std::exp(-x);
    for (int i = 1; i < 100000; ++i) {
        const double t = i * h;
        const double expo = -x * std::cosh(t) + std::abs(nu) * t;
        if (expo < -745.0) {
            break;
        }
        s += std::exp(-x * std::cosh(t)) * std::cosh(nu * t);
    }
    return s * h;
}

// Exponentially scaled K_ν: e^x K_ν(x), x > 0, from the same representation
// written without cancellation (cosh t − 1 = 2 sinh²(t/2)),
//
//     e^x K_ν(x) = ∫₀^∞ exp(−2x sinh²(t/2)) cosh(νt) dt.
//
// It neither underflows at large x (where bessel_k returns 0) nor loses digits.
// For large x the integrand is a Gaussian of width 1/√x, whose trapezoidal
// error is ~2 exp(−2π²/(x h²)); h = min(0.05, 0.25/√x) keeps that at
// exp(−32π²) for every x, and the sum stops once a term falls below e^{−45}
// of the first.  (A double-precision transcription of this rule agreed with
// scipy's Cephes k0e/k1e to 2e-15 relative over x ∈ [1e-3, 1e5].)  Setup-time
// use only: it builds ScaledBesselK01's table, which
// tests/test_zero_energy_jastrow.cpp checks against std::cyl_bessel_k.
inline double bessel_k_scaled(double nu, double x) {
    const double h = std::min(0.05, 0.25 / std::sqrt(x));
    double s = 0.5;
    for (int i = 1; i < 1000000; ++i) {
        const double t = i * h;
        const double sinh_half = std::sinh(0.5 * t);
        const double expo = 2.0 * x * sinh_half * sinh_half;
        if (expo - std::abs(nu) * t > 45.0) {
            break;
        }
        s += std::exp(-expo) * std::cosh(nu * t);
    }
    return s * h;
}

// Struve function of order zero, H0(x).
//
// Reference implementation adapted from Zhang & Jin, "Computation of Special
// Functions" (Wiley, 1996): power series for x ≤ 20, asymptotic expansion
// (with the leading Y0 term) for x > 20.
inline double stvh0(double x) {
    double s = 1.0;
    double r = 1.0;
    double sh0;

    if (x <= 20.0) {
        double a0 = 2.0 * x / PI;
        for (int k = 1; k <= 60; ++k) {
            r = -r * (x / (2.0 * k + 1.0)) * (x / (2.0 * k + 1.0));
            s = s + r;
            if (std::abs(r) < std::abs(s) * 1.0e-12) {
                break;
            }
        }
        sh0 = a0 * s;
    } else {
        int km = static_cast<int>(0.5 * (x + 1.0));
        if (x >= 50.0) {
            km = 25;
        }
        for (int k = 1; k <= km; ++k) {
            double term = (2.0 * k - 1.0) / x;
            r = -r * term * term;
            s = s + r;
            if (std::abs(r) < std::abs(s) * 1.0e-12) {
                break;
            }
        }

        double t = 4.0 / x;
        double t2 = t * t;
        double p0 = ((((-0.37043e-5 * t2 + 0.173565e-4) * t2 - 0.487613e-4) * t2 + 0.17343e-3) * t2 - 0.1753062e-2) * t2 + 0.3989422793;
        double q0 = t * (((((0.32312e-5 * t2 - 0.142078e-4) * t2 + 0.342468e-4) * t2 - 0.869791e-4) * t2 + 0.4564324e-3) * t2 - 0.0124669441);
        double ta0 = x - 0.25 * PI;
        double by0 = 2.0 / std::sqrt(x) * (p0 * std::sin(ta0) + q0 * std::cos(ta0));
        sh0 = 2.0 / (PI * x) * s + by0;
    }

    return sh0;
}

// Bessel function of the second kind, order zero, Y0(x).
//
// Reference implementation adapted from Zhang & Jin, "Computation of Special
// Functions" (Wiley, 1996): rational approximation for x ≤ 4, asymptotic form
// for x > 4.  At x = 0 the function diverges to −∞; a large finite sentinel is
// returned to keep callers numerically well-defined.
inline double jy0b(double x) {
    if (x == 0.0) {
        return -1.0e300;
    } else if (x <= 4.0) {
        double t = x / 4.0;
        double t2 = t * t;

        double bj0 = (((((( -0.5014415e-3*t2 + 0.76771853e-2 )*t2 - 0.0709253492 )*t2
                + 0.4443584263 )*t2 - 1.7777560599 )*t2 + 3.9999973021 )*t2
                - 3.9999998721 )*t2 + 1.0;

        double by0 = (((((( -0.567433e-4*t2 + 0.859977e-3 )*t2 - 0.94855882e-2 )*t2
                + 0.0772975809 )*t2 - 0.4261737419 )*t2 + 1.4216421221 )*t2
                - 2.3498519931 )*t2 + 1.0766115157;
        by0 = by0 * t2 + 0.3674669052;
        by0 = 2.0 / PI * std::log(x / 2.0) * bj0 + by0;

        return by0;
    } else {
        double t = 4.0 / x;
        double t2 = t * t;
        double a0 = std::sqrt(2.0 / (PI * x));

        double p0 = ((((-0.9285e-5*t2 + 0.43506e-4 )*t2 - 0.122226e-3 )*t2
                + 0.434725e-3 )*t2 - 0.4394275e-2 )*t2 + 0.999999997;
        double q0 = t*((((( 0.8099e-5*t2 - 0.35614e-4 )*t2 + 0.85844e-4 )*t2
                - 0.218024e-3 )*t2 + 0.1144106e-2 )*t2 - 0.031249995);
        double ta0 = x - 0.25 * PI;
        double by0 = a0 * (p0 * std::sin(ta0) + q0 * std::cos(ta0));

        return by0;
    }
}

// Rytova–Keldysh radial kernel:  K(x) = H0(x) − Y0(x).
//
// Positive for all x > 0.  Asymptotics: K(x) → 2/(π x) for x ≫ 1 (so the
// potential recovers Coulomb 1/r), and K(x) → −Y0(x) → +∞ logarithmically as
// x → 0 (a soft divergence, far milder than 1/r).
//
// This is the high-precision reference: Y0 from std::cyl_neumann (C++17, ~1e-14)
// rather than the fast jy0b (~1e-8). Used to build/validate the vectorized
// Chebyshev kernel (see keldysh_kernel.h); not on any hot path itself.
// Seam between the two branches below.  See the comment in keldysh_kernel.
inline constexpr double kKeldyshSeam = 21.0;

inline double keldysh_kernel(double x) {
    // Two traps make the naive `stvh0(x) - cyl_neumann(0,x)` lose ~8 digits
    // near the seam.  Both are avoided here; measured worst-case error across
    // the seam drops from 1.5e-9 to 2.9e-11.
    //
    // (1) x ≤ seam — H0's power series alternates with terms that GROW until
    //     k ≈ x/2: at x = 20 the largest term is 6.1e5 against a sum of
    //     7.4e-3, i.e. ~8 digits of cancellation.  Accumulated in double only
    //     8 digits survive (measured error 1.8e-9); in long double, 11 do
    //     (~1e-12).  This is not a hot path — the hot path is the Chebyshev
    //     table in keldysh_kernel.h, which this function *builds*.
    if (x <= kKeldyshSeam) {
        long double s = 1.0L, r = 1.0L;
        const long double a0 = 2.0L * static_cast<long double>(x) / PI_L;
        for (int k = 1; k <= 200; ++k) {
            const long double q = static_cast<long double>(x) / (2.0L * k + 1.0L);
            r = -r * q * q;
            s += r;
            if (std::abs(r) < std::abs(s) * 1.0e-22L) {
                break;
            }
        }
        return static_cast<double>(a0 * s) - std::cyl_neumann(0.0, x);
    }
    // (2) x > seam — do NOT build H0 and then subtract Y0.  stvh0's asymptotic
    //     branch returns (2/πx)·s + by0, where by0 is a 6-term rational fit to
    //     Y0 accurate to only ~1e-10.  Subtracting the *true* Y0 (cyl_neumann,
    //     ~1e-16) leaves that fit's error intact.  Analytically the two Y0
    //     cancel and K(x) → (2/πx)·s, with no by0 and no cancellation at all.
    double s = 1.0, r = 1.0;
    const int km = (x >= 50.0) ? 25 : static_cast<int>(0.5 * (x + 1.0));
    for (int k = 1; k <= km; ++k) {
        const double t = (2.0 * k - 1.0) / x;
        r = -r * t * t;
        s += r;
        if (std::abs(r) < std::abs(s) * 1.0e-17) {
            break;
        }
    }
    return 2.0 / (PI * x) * s;
}


// First and second derivatives of K(x) = H0(x) - Y0(x).
//
// From the integral representation K(x) = (2/pi) int_0^inf exp(-x sinh t) dt,
// differentiating under the integral sign:
//     K'(x)  = -(2/pi) int_0^inf sinh(t)   exp(-x sinh t) dt
//     K''(x) =  (2/pi) int_0^inf sinh^2(t) exp(-x sinh t) dt
// The alternative, K' = 2/pi - (H1 - Y1), would need a Struve H1 the project
// does not have.  Validated: the same quadrature at order 0 reproduces
// keldysh_kernel to ~1e-15, and the derivatives agree with a central
// difference of it to ~1e-9 in the range where that difference is itself
// trustworthy (it is not near the origin, where K has a logarithm, nor at
// large x, where eps/h^2 dominates).
//
// Setup-time only: the breakup needs these at r_c alone, one evaluation each.
namespace detail {

inline double keldysh_moment(double x, int order) {
    // Gauss-Legendre, 16 points per panel.
    static constexpr double gx[8] = {
        0.0950125098376374, 0.2816035507792589, 0.4580167776572274,
        0.6178762444026438, 0.7554044083550030, 0.8656312023878318,
        0.9445750230732326, 0.9894009349916499};
    static constexpr double gw[8] = {
        0.1894506104550685, 0.1826034150449236, 0.1691565193950025,
        0.1495959888165767, 0.1246289712555339, 0.0951585116824928,
        0.0622535239386479, 0.0271524594117541};
    // exp(-x sinh t) is below 1e-17 once x sinh t exceeds 40.
    const double t_max = std::asinh(40.0 / std::max(x, 1.0e-12));
    constexpr int panels = 400;
    const double h = t_max / panels;
    double sum = 0.0;
    for (int i = 0; i < panels; ++i) {
        const double c = (i + 0.5) * h, r = 0.5 * h;
        for (int j = 0; j < 8; ++j) {
            for (const double t : {c + r * gx[j], c - r * gx[j]}) {
                const double sh = std::sinh(t);
                double term = std::exp(-x * sh);
                if (order == 1) {
                    term *= -sh;
                } else if (order == 2) {
                    term *= sh * sh;
                }
                sum += gw[j] * term;
            }
        }
    }
    return (2.0 / PI) * sum * 0.5 * h;
}

}  // namespace detail

inline double keldysh_kernel_d1(double x) {
    return detail::keldysh_moment(x, 1);
}

inline double keldysh_kernel_d2(double x) {
    return detail::keldysh_moment(x, 2);
}

}  // namespace alea::numerics
