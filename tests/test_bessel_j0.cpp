// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>

#include "numerics/bessel_j0.h"

using alea::numerics::bessel_j0;
using alea::numerics::kJ0AsymptoticFrom;

namespace {

// Reference 1 (independent METHOD): J0(x) = (1/π) ∫₀^π cos(x sin t) dt.
// Chosen deliberately over std::cyl_bessel_j, which is NOT a valid yardstick
// here: above x ≈ 60 libstdc++ is the less accurate of the two (measured
// 8.2e-14 at x = 600 against 7.5e-16 for the expansion).  This integral shares
// no machinery with the asymptotic expansion, so agreement is real evidence.
long double j0_integral(long double x, int panels) {
    static constexpr long double gx[8] = {
        0.0950125098376374L, 0.2816035507792589L, 0.4580167776572274L,
        0.6178762444026438L, 0.7554044083550030L, 0.8656312023878318L,
        0.9445750230732326L, 0.9894009349916499L};
    static constexpr long double gw[8] = {
        0.1894506104550685L, 0.1826034150449236L, 0.1691565193950025L,
        0.1495959888165767L, 0.1246289712555339L, 0.0951585116824928L,
        0.0622535239386479L, 0.0271524594117541L};
    constexpr long double pi = 3.14159265358979323846264338327950288L;
    const long double h = pi / panels;
    long double s = 0.0L;
    for (int i = 0; i < panels; ++i) {
        const long double c = i * h + 0.5L * h, r = 0.5L * h;
        for (int j = 0; j < 8; ++j) {
            s += gw[j] * (std::cos(x * std::sin(c + r * gx[j])) +
                          std::cos(x * std::sin(c - r * gx[j])));
        }
    }
    return s * 0.5L * h / pi;
}

// Reference 2 (same method, more precision): the Hankel expansion in long
// double, with one extra term in each of P and Q.  Cheap, so it carries the
// dense sweep; the integral above carries the independence.
long double j0_hankel_ld(long double x) {
    const long double z = 1.0L / x, z2 = z * z;
    const long double p = 1.0L + z2 * (-7.03125e-2L +
                          z2 * (1.121520996093750e-1L +
                          z2 * (-5.725014209747314e-1L +
                          z2 * (6.074042001273483e0L +
                          z2 * (-1.100171402692467e2L)))));
    const long double q = z * (-1.25e-1L +
                          z2 * (7.32421875e-2L +
                          z2 * (-2.271080017089844e-1L +
                          z2 * (1.727727502584457e0L +
                          z2 * (-2.438052969955606e1L)))));
    const long double w = x - 0.78539816339744830961566084581988L;
    return std::sqrt(0.63661977236758134307553505349006L * z) *
           (p * std::cos(w) - q * std::sin(w));
}

}  // namespace

TEST_CASE("bessel_j0 delegates below the asymptotic split", "[bessel]") {
    // Below the split libstdc++ is the more accurate branch (2.0e-15 against
    // 1.2e-14 at x = 40), so bessel_j0 must simply BE it there.
    for (double x : {0.0, 1e-8, 0.5, 1.0, 5.0, 12.5, 39.9999}) {
        CHECK(bessel_j0(x) == std::cyl_bessel_j(0.0, x));
    }
}

TEST_CASE("bessel_j0 matches an independent integral representation",
          "[bessel]") {
    // The breakup amplifies a J0 error into the energy by ~600x (measured) and
    // its own accuracy is ~3e-10, so anything below ~1e-13 here is invisible
    // downstream.  Worst point is the split itself, at ~1.2e-14.
    double worst = 0.0;
    double worst_x = 0.0;
    for (double x : {40.0, 40.5, 42.0, 47.3, 57.3, 100.0, 250.0, 777.7,
                     1500.0, 3000.0, 6000.0}) {
        const auto ref = static_cast<double>(j0_integral(x, 3000));
        const double err = std::abs(bessel_j0(x) - ref);
        if (err > worst) {
            worst = err;
            worst_x = x;
        }
    }
    INFO("max |bessel_j0 - integral| = " << worst << " at x = " << worst_x);
    CHECK(worst < 1.0e-13);
}

TEST_CASE("bessel_j0 holds accuracy across the whole asymptotic range",
          "[bessel]") {
    // Dense sweep against the higher-order expansion — catches a coefficient
    // typo or a lost term, which the sparse integral check could step over.
    double worst = 0.0;
    for (double x = kJ0AsymptoticFrom; x < 8000.0; x += 0.0137) {
        worst = std::max(worst, std::abs(bessel_j0(x) -
                                static_cast<double>(j0_hankel_ld(x))));
    }
    INFO("max |bessel_j0 - hankel_ld| = " << worst);
    CHECK(worst < 1.0e-13);
}

TEST_CASE("bessel_j0 is continuous across the split", "[bessel]") {
    // A jump here would inject a step into the breakup's c_n(k) quadrature —
    // precisely how the Keldysh kernel's x = 20 seam used to poison the fit.
    // The jump cannot be zero: the branches meet carrying their own errors,
    // ~2.0e-15 (libstdc++) and ~1.2e-14 (the 4-term expansion), so ~1.5e-14 is
    // the expected size, not a defect.  What matters is downstream: 1.5e-14
    // times the ~600x amplification is ~9e-12, against the breakup's 2.9e-10.
    const double lo = bessel_j0(std::nextafter(kJ0AsymptoticFrom, 0.0));
    const double hi = bessel_j0(kJ0AsymptoticFrom);
    INFO("jump at x = " << kJ0AsymptoticFrom << " is " << std::abs(hi - lo));
    CHECK(std::abs(hi - lo) < 1.0e-13);
}
