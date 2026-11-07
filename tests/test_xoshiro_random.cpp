// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

#include "numerics/xoshiro_random.h"

// The VMC/DMC generator (src/numerics/xoshiro_random.h): the engine against
// the reference implementation, and the uniforms and ziggurat Gaussians
// against their exact distributions.  The statistical checks use a fixed
// seed and fail at p < 1e-6 (Pearson χ²) or beyond 5σ (moments).

using alea::numerics::Xoshiro256pp;
using alea::numerics::XoshiroRandom;

namespace {

double normal_cdf(double x) { return 0.5 * std::erfc(-x / std::numbers::sqrt2); }

// Upper-tail probability of a χ² with k degrees of freedom by the
// Wilson–Hilferty approximation, (χ²/k)^(1/3) ≈ N(1 − 2/(9k), 2/(9k)),
// accurate to a few percent of p for k ≥ 100.
double chi_square_upper_tail(double chi2, std::size_t k) {
    const double s = 2.0 / (9.0 * static_cast<double>(k));
    const double z = (std::cbrt(chi2 / static_cast<double>(k)) - (1.0 - s)) / std::sqrt(s);
    return 0.5 * std::erfc(z / std::numbers::sqrt2);
}

}  // namespace

TEST_CASE("xoshiro256++ reproduces the reference implementation's outputs", "[random]") {
    // xoshiro256plusplus.c from the state {1, 2, 3, 4}.  The first value is
    // rotl(1 + 4, 23) + 1 = 41943041 by hand.
    Xoshiro256pp engine(std::array<std::uint64_t, 4>{1, 2, 3, 4});
    const std::array<std::uint64_t, 10> reference{
        41943041ULL,             58720359ULL,
        3588806011781223ULL,     3591011842654386ULL,
        9228616714210784205ULL,  9973669472204895162ULL,
        14011001112246962877ULL, 12406186145184390807ULL,
        15849039046786891736ULL, 10450023813501588000ULL};
    for (const std::uint64_t value : reference) CHECK(engine() == value);
}

TEST_CASE("XoshiroRandom repeats its stream for a seed and differs between seeds", "[random]") {
    XoshiroRandom a(42), b(42), c(43);
    bool c_differs = false;
    for (int i = 0; i < 100; ++i) {
        const double ua = a.uniform(), ga = a.gaussian();
        CHECK(ua == b.uniform());
        CHECK(ga == b.gaussian());
        c_differs = c_differs || ua != c.uniform() || ga != c.gaussian();
    }
    CHECK(c_differs);
}

// No square bracket in a test name: CMake would not split the discovered
// test list inside an unmatched one.
TEST_CASE("XoshiroRandom uniforms are uniform between 0 and 1, never reaching 1", "[random]") {
    constexpr std::size_t n_samples = 10'000'000, n_bins = 1000;
    XoshiroRandom rng(20261020);
    std::vector<double> counts(n_bins, 0.0);
    double sum = 0.0;
    bool in_range = true;
    for (std::size_t i = 0; i < n_samples; ++i) {
        const double u = rng.uniform();
        in_range = in_range && u >= 0.0 && u < 1.0;
        sum += u;
        counts[static_cast<std::size_t>(u * n_bins)] += 1.0;
    }
    REQUIRE(in_range);
    const double n = static_cast<double>(n_samples);
    CHECK(std::abs(sum / n - 0.5) < 5.0 * std::sqrt(1.0 / (12.0 * n)));
    const double expected = n / n_bins;
    double chi2 = 0.0;
    for (const double c : counts) chi2 += (c - expected) * (c - expected) / expected;
    CHECK(chi_square_upper_tail(chi2, n_bins - 1) > 1e-6);
}

TEST_CASE("Ziggurat Gaussians follow the standard normal distribution, tails included",
          "[random]") {
    // 10⁸ draws in 0.02-wide bins over [−4, 4] and the tail bins [4, 4.5),
    // [4.5, 5), [5, ∞) on each side (≈29 expected beyond 5).  The ziggurat's
    // own tail starts at r ≈ 3.654, inside the binned range, so the base strip,
    // its exact tail sampler and the layers are all tested.
    constexpr std::size_t n_samples = 100'000'000;
    constexpr double width = 0.02, x_max = 4.0;
    constexpr std::size_t n_central = 400;
    const std::array<double, 4> tail_edges{4.0, 4.5, 5.0, INFINITY};
    const std::size_t n_bins = n_central + 2 * (tail_edges.size() - 1);
    auto bin_of = [&](double x) -> std::size_t {
        const double a = std::abs(x);
        if (a < x_max) return static_cast<std::size_t>((x + x_max) / width);
        std::size_t t = 0;
        while (a >= tail_edges[t + 1]) ++t;
        return n_central + (x > 0 ? 0 : tail_edges.size() - 1) + t;
    };

    XoshiroRandom rng(7);
    std::vector<double> counts(n_bins, 0.0);
    double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, lag1 = 0.0, previous = 0.0;
    bool all_finite = true;
    for (std::size_t i = 0; i < n_samples; ++i) {
        const double x = rng.gaussian();
        if (!std::isfinite(x)) {
            all_finite = false;
            continue;
        }
        counts[bin_of(x)] += 1.0;
        const double x2 = x * x;
        s1 += x; s2 += x2; s3 += x2 * x; s4 += x2 * x2;
        lag1 += x * previous;
        previous = x;
    }

    REQUIRE(all_finite);
    const double n = static_cast<double>(n_samples);
    double chi2 = 0.0;
    for (std::size_t b = 0; b < n_bins; ++b) {
        double p;
        if (b < n_central) {
            const double lo = -x_max + width * static_cast<double>(b);
            p = normal_cdf(lo + width) - normal_cdf(lo);
        } else {
            const std::size_t t = (b - n_central) % (tail_edges.size() - 1);
            p = normal_cdf(-tail_edges[t]) - normal_cdf(-tail_edges[t + 1]);   // same both sides
        }
        const double expected = n * p;
        chi2 += (counts[b] - expected) * (counts[b] - expected) / expected;
    }
    INFO("chi2 = " << chi2 << " for " << n_bins - 1 << " degrees of freedom");
    CHECK(chi_square_upper_tail(chi2, n_bins - 1) > 1e-6);

    // Moments of N(0, 1): mean 0, variance 1, skewness 0, kurtosis 3, with
    // standard errors 1, √2, √15, √96 over √n; and no lag-1 correlation.
    const double root_n = std::sqrt(n);
    CHECK(std::abs(s1 / n) < 5.0 / root_n);
    CHECK(std::abs(s2 / n - 1.0) < 5.0 * std::sqrt(2.0) / root_n);
    CHECK(std::abs(s3 / n) < 5.0 * std::sqrt(15.0) / root_n);
    CHECK(std::abs(s4 / n - 3.0) < 5.0 * std::sqrt(96.0) / root_n);
    CHECK(std::abs(lag1 / n) < 5.0 / root_n);
}
