// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Radial pair functions fed to the optimized breakup when a pair Jastrow is
// Ewald-summed: the modified Bessel K_ν the Padé tail model needs, and the tail
// transform T(k) = 2π ∫_{r_c}^∞ r J₀(kr) f(r) dr that replaces f̃(k).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

#include "numerics/pair_radial.h"
#include "numerics/special_functions.h"

using alea::numerics::bessel_k;
using alea::numerics::pade_radial;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {
constexpr double kPi = 3.14159265358979323846;
}

TEST_CASE("bessel_k matches tabulated and closed-form values", "[pair_radial]") {
    // Abramowitz & Stegun, Table 9.8.
    CHECK_THAT(bessel_k(0.0, 1.0), WithinRel(0.42102443824070834, 1e-13));
    CHECK_THAT(bessel_k(1.0, 1.0), WithinRel(0.60190723019723457, 1e-13));
    CHECK_THAT(bessel_k(0.0, 0.1), WithinRel(2.4270690247020166, 1e-13));
    CHECK_THAT(bessel_k(1.0, 2.5), WithinRel(0.07389081634774707, 1e-9));
    // Half-integer orders are elementary: K_{1/2}(x) = √(π/2x) e^{−x}.
    for (double x : {1e-3, 0.3, 4.0, 40.0}) {
        INFO("x = " << x);
        CHECK_THAT(bessel_k(0.5, x), WithinRel(std::sqrt(kPi / (2 * x)) * std::exp(-x), 1e-13));
    }
}

TEST_CASE("pade_radial subtracts the constant tail and keeps the shape",
          "[pair_radial]") {
    // u = (a₁r + a₂r²)/(1 + b₁r + b₂r²) tends to u∞ = a₂/b₂.  Ψ is defined up to a
    // constant, so only u − u∞ is summed over images; its derivatives are u's.
    const std::vector<double> a{-1.0, 0.3}, b{0.5, 0.2};
    const auto f = pade_radial(a, b, /*r_cut=*/6.0);
    auto u = [&](double r) {
        return (a[0] * r + a[1] * r * r) / (1 + b[0] * r + b[1] * r * r);
    };
    for (double r : {0.1, 1.0, 5.0, 40.0}) {
        INFO("r = " << r);
        CHECK_THAT(f.value(r), WithinAbs(u(r) - a[1] / b[1], 1e-14));
        const double h = 1e-5;
        CHECK_THAT(f.d1(r), WithinAbs((u(r + h) - u(r - h)) / (2 * h), 1e-8));
        CHECK_THAT(f.d2(r), WithinAbs((u(r + h) - 2 * u(r) + u(r - h)) / (h * h), 1e-5));
    }
}

TEST_CASE("pade_radial tail transform does not depend on the model width",
          "[pair_radial]") {
    // The analytic tail model is subtracted and added back in closed form, so
    // its width s must cancel exactly; what is left is the numerical error of
    // the remainder integral.
    const std::vector<double> a{-1.0, 0.3}, b{0.5, 0.2};
    const double r_cut = 6.0;
    const auto narrow = pade_radial(a, b, r_cut, /*model_width=*/1.5);
    const auto wide   = pade_radial(a, b, r_cut, /*model_width=*/9.0);
    for (double k : {0.05, 0.4, 1.3, 7.0, 60.0}) {
        INFO("k = " << k);
        const double t = narrow.tail(k);
        CHECK_THAT(wide.tail(k), WithinAbs(t, 1e-9 * (1 + std::abs(t))));
    }
}

TEST_CASE("pade_radial refuses a numerator of higher order than the denominator",
          "[pair_radial]") {
    // u ~ r^{N−M} grows without bound: the image sum does not exist.
    CHECK_THROWS(pade_radial(std::vector<double>{-1.0, 0.1}, std::vector<double>{0.5}, 6.0));
    CHECK_THROWS(pade_radial(std::vector<double>{-1.0}, std::vector<double>{}, 6.0));
}
