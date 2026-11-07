// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "numerics/statistics.h"

using alea::numerics::block_mean_error;
using alea::numerics::reblock;

namespace {
template<class R>
void check_selected_row(const R& result, std::vector<double> chain, bool& later) {
    if constexpr (requires { result.selected_level; result.selected_n_blocks; result.error_on_error; }) {
        later |= result.selected_level > result.level;
        for (int i = 0; i < result.selected_level; ++i) {
            if (chain.size() % 2 != 0) chain.pop_back();
            for (std::size_t j = 0; j + 1 < chain.size(); j += 2)
                chain[j / 2] = 0.5 * (chain[j] + chain[j + 1]);
            chain.resize(chain.size() / 2);
        }
        CHECK(result.selected_n_blocks == chain.size());
        const auto direct = block_mean_error(chain);
        CHECK(result.error == Catch::Approx(direct.error));
        CHECK(result.error_on_error == Catch::Approx(result.error /
            std::sqrt(2.0 * static_cast<double>(chain.size() - 1))));
    } else {
        FAIL("ReblockResult lacks diagnostics identifying the row that supplied its reported error");
    }
}
template<class R> void check_short_diagnostics(const R& result, std::size_t n) {
    if constexpr (requires { result.selected_level; result.selected_n_blocks; result.error_on_error; }) {
        CHECK(result.selected_level == 0);
        CHECK(result.selected_n_blocks == n);
        CHECK(result.error_on_error == 0.0);
    } else {
        FAIL("ReblockResult lacks selected-row diagnostics");
    }
}
} // namespace

TEST_CASE("block_mean_error returns the mean and standard error of block means",
          "[statistics]") {
    // {-1,-2,-3}: mean = -2, s² = (1+0+1)/2 = 1, error = sqrt(1/3).
    std::vector<double> blocks = {-1.0, -2.0, -3.0};
    auto stats = block_mean_error(blocks);

    CHECK(stats.mean == Catch::Approx(-2.0));
    CHECK(stats.error == Catch::Approx(std::sqrt(1.0 / 3.0)));
}

TEST_CASE("block_mean_error gives zero error for identical blocks",
          "[statistics]") {
    std::vector<double> blocks = {5.0, 5.0, 5.0};
    auto stats = block_mean_error(blocks);

    CHECK(stats.mean == Catch::Approx(5.0));
    CHECK(stats.error == Catch::Approx(0.0));
}

TEST_CASE("block_mean_error cannot estimate error from a single block",
          "[statistics]") {
    std::vector<double> blocks = {3.0};
    auto stats = block_mean_error(blocks);

    CHECK(stats.mean == Catch::Approx(3.0));
    CHECK(stats.error == Catch::Approx(0.0));
}

// ── Flyvbjerg–Petersen reblocking ───────────────────────────────────────────

namespace {

// Fixed-seed white noise (independent samples).
std::vector<double> white_noise(int n) {
    std::mt19937_64 rng(42);
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<double> x(static_cast<std::size_t>(n));
    for (auto& v : x) v = g(rng);
    return x;
}

// AR(1) series with lag-1 correlation phi (fixed seed).
std::vector<double> ar1(int n, double phi) {
    std::mt19937_64 rng(7);
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<double> x(static_cast<std::size_t>(n));
    x[0] = g(rng);
    for (int i = 1; i < n; ++i)
        x[static_cast<std::size_t>(i)] =
            phi * x[static_cast<std::size_t>(i - 1)] +
            std::sqrt(1.0 - phi * phi) * g(rng);
    return x;
}

}  // namespace

TEST_CASE("reblock diagnostics identify the conservative selected row, not onset",
          "[statistics][reblock][selected-row]") {
    bool selected_later = false;
    for (const auto& chain : {white_noise(8192), ar1(8192, 0.9), ar1(1024, 0.5)})
        check_selected_row(reblock(chain), chain, selected_later);
    CHECK(selected_later);
    check_short_diagnostics(reblock(std::vector<double>{}), 0);
    check_short_diagnostics(reblock(std::vector<double>{2.0}), 1);
}

TEST_CASE("reblock of white noise recovers the naive error",
          "[statistics][reblock]") {
    // Independent samples: no autocorrelation, so the plateau error must
    // agree with the naive SEM (they estimate the same quantity).
    const std::vector<double> x = white_noise(8192);
    const auto r = reblock(x);

    CHECK(r.plateau_found);
    CHECK(r.error > 0.0);
    const double ratio = r.error / r.naive_error;
    CHECK(ratio > 0.7);
    CHECK(ratio < 1.5);
    CHECK(r.mean == Catch::Approx(0.0).margin(0.1));
    // tau_int of white noise is ~0 (inefficiency ~1).
    CHECK(r.tau_int < 1.0);
}

TEST_CASE("reblock corrects a strongly correlated series", "[statistics][reblock]") {
    // AR(1) with phi = 0.9: tau_int ~ phi/(1-phi) = 9 samples, so the naive
    // SEM underestimates by ~sqrt(1+2*tau_int) = sqrt(19) ~ 4.4.
    const std::vector<double> x = ar1(8192, 0.9);
    const auto r = reblock(x);

    CHECK(r.plateau_found);
    const double ratio = r.error / r.naive_error;
    CHECK(ratio > 3.0);
    CHECK(ratio < 6.0);
    // tau_int within a factor of ~2 of the analytic value.
    CHECK(r.tau_int > 4.0);
    CHECK(r.tau_int < 20.0);
}

TEST_CASE("reblock plateau sits at the repetition period", "[statistics][reblock]") {
    // Each independent sample repeated 16 times: the SEM must keep growing
    // until the block length reaches 16, then plateau at ~sqrt(16)x the
    // naive SEM of the full (repeated) series.
    const std::vector<double> base = white_noise(1024);
    std::vector<double> x;
    x.reserve(base.size() * 16);
    for (double v : base)
        for (int r = 0; r < 16; ++r) x.push_back(v);

    const auto res = reblock(x);
    CHECK(res.plateau_found);
    const double ratio = res.error / res.naive_error;
    CHECK(ratio > 3.0);
    CHECK(ratio < 5.5);
    CHECK(res.tau_int > 5.0);  // ~ (16-1)/2 = 7.5 for a top-hat correlation
}

TEST_CASE("reblock survives a series too short for a plateau",
          "[statistics][reblock]") {
    const std::vector<double> x = {1.0, 2.0, 3.0};
    const auto r = reblock(x);
    CHECK(r.mean == Catch::Approx(2.0));
    CHECK(r.error >= 0.0);
    CHECK(r.naive_error > 0.0);
}

// ── reblock_chains: independent chains (VMC threads) ───────────────────────

namespace {

// K independent unit-variance AR(1) chains, concatenated [chain * n + i].
std::vector<double> ar1_chains(std::size_t k, int n, double phi) {
    std::vector<double> x;
    x.reserve(k * static_cast<std::size_t>(n));
    for (std::size_t c = 0; c < k; ++c) {
        std::mt19937_64 rng(1000 + c);
        std::normal_distribution<double> g(0.0, 1.0);
        double v = g(rng);
        x.push_back(v);
        for (int i = 1; i < n; ++i) {
            v = phi * v + std::sqrt(1.0 - phi * phi) * g(rng);
            x.push_back(v);
        }
    }
    return x;
}

}  // namespace

TEST_CASE("reblock_chains of one chain is reblock", "[statistics][reblock_chains]") {
    const std::vector<double> x = ar1(1024, 0.5);
    const auto single = reblock(x);
    const auto pooled = alea::numerics::reblock_chains(x, 1);
    CHECK(pooled.mean == single.mean);
    CHECK(pooled.error == single.error);
    CHECK(pooled.naive_error == single.naive_error);
    CHECK(pooled.tau_int == single.tau_int);
    CHECK(pooled.selected_level == single.selected_level);
    CHECK(pooled.plateau_found == single.plateau_found);
}

TEST_CASE("reblock_chains recovers the analytic error of the grand mean of AR(1) chains",
          "[statistics][reblock_chains]") {
    // K chains of n samples, lag-1 correlation phi, unit variance: for n >> tau
    //   Var(grand mean) = (1 + 2 Σ_{j>=1} phi^j) / (K n) = (1 + phi) / ((1 - phi) K n),
    // since the autocorrelation at lag j is phi^j.  phi = 0.8 makes the naive
    // SEM 3x too small.  The plateau error is the
    // largest SEM on the plateau, whose rows have >= 16 pooled blocks (an
    // error-on-error up to ~25 %), so it may sit up to that much above exact.
    constexpr std::size_t k = 8;
    constexpr int n = 2048;
    constexpr double phi = 0.8;
    const auto r = alea::numerics::reblock_chains(ar1_chains(k, n, phi), k);
    const double exact = std::sqrt((1.0 + phi) / ((1.0 - phi) * k * n));

    CHECK(r.plateau_found);
    CHECK(r.error / r.naive_error > 2.2);
    CHECK(r.error / exact > 0.8);
    CHECK(r.error / exact < 1.5);
    CHECK(r.tau_int > 2.0);  // exact phi / (1 - phi) = 4 samples
    CHECK(r.tau_int < 10.0);
}

TEST_CASE("reblock_chains never pairs blocks of two different chains",
          "[statistics][reblock_chains]") {
    // White-noise chains with fixed, very different offsets.  The offsets are
    // not random, so the grand mean's error is the within-chain one,
    // 1/sqrt(K n).  Coarse-graining across a boundary would mix the offsets
    // into the block variance and inflate the error by orders of magnitude.
    constexpr std::size_t k = 4;
    constexpr int n = 1024;
    std::vector<double> x = ar1_chains(k, n, 0.0);
    for (std::size_t c = 0; c < k; ++c)
        for (int i = 0; i < n; ++i) x[c * n + static_cast<std::size_t>(i)] += 100.0 * c;

    const auto pooled = alea::numerics::reblock_chains(x, k);
    const double exact = 1.0 / std::sqrt(static_cast<double>(k * n));
    CHECK(pooled.mean == Catch::Approx(150.0).margin(0.1));
    CHECK(pooled.error / exact > 0.8);
    CHECK(pooled.error / exact < 1.5);
    CHECK(reblock(x).error > 50.0 * pooled.error);  // what concatenation would give
}

TEST_CASE("reblock_chains refuses a series that does not split into equal chains",
          "[statistics][reblock_chains]") {
    const std::vector<double> x(10, 1.0);
    CHECK_THROWS_AS(alea::numerics::reblock_chains(x, 3), std::invalid_argument);
    CHECK_THROWS_AS(alea::numerics::reblock_chains(x, 0), std::invalid_argument);
    const auto one_block_each = alea::numerics::reblock_chains(x, 10);
    CHECK(one_block_each.mean == 1.0);
    CHECK(one_block_each.error == 0.0);
}

// ── ratio_of_means: the signed estimator <O s>/<s> ─────────────────────────

TEST_CASE("ratio_of_means with a constant denominator is the reblocked mean",
          "[statistics][ratio]") {
    // An AR(1) chain: correlated blocks, so the reblocking level is > 0.
    std::mt19937_64 rng(2024);
    std::normal_distribution<double> noise(0.0, 1.0);
    std::vector<double> x(4096), one(4096, 1.0);
    double prev = 0.0;
    for (double& v : x) { prev = 0.9 * prev + noise(rng); v = 3.0 + prev; }
    const auto ratio = alea::numerics::ratio_of_means(x, one);
    const auto plain = reblock(x);
    CHECK(ratio.mean == Catch::Approx(plain.mean).epsilon(1e-14));
    // The jackknife over the coarse blocks of the reblocking's selected
    // level reproduces that level's SEM exactly (den = 1).
    CHECK(ratio.level == plain.selected_level);
    CHECK(ratio.error == Catch::Approx(plain.error).epsilon(1e-10));
    CHECK(ratio.jackknife_bias == Catch::Approx(0.0).margin(1e-12));
    CHECK(ratio.reliable);
}

TEST_CASE("ratio_of_means is exact when the numerator is proportional to the sign",
          "[statistics][ratio]") {
    std::mt19937_64 rng(7);
    std::bernoulli_distribution positive(0.75);
    std::vector<double> sign(2048), num(2048);
    for (std::size_t b = 0; b < sign.size(); ++b) {
        sign[b] = positive(rng) ? 1.0 : -1.0;
        num[b] = 2.5 * sign[b];
    }
    const auto ratio = alea::numerics::ratio_of_means(num, sign);
    CHECK(ratio.mean == Catch::Approx(2.5).epsilon(1e-14));
    CHECK(ratio.error == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("ratio_of_means error matches the delta method for independent blocks",
          "[statistics][ratio]") {
    // Independent numerator and denominator: var(R) ~ (var_x / mu_y^2 +
    // mu_x^2 var_y / mu_y^4) / n to leading order.
    std::mt19937_64 rng(99);
    std::normal_distribution<double> x_dist(1.5, 0.4), y_dist(0.6, 0.1);
    const std::size_t n = 8192;
    std::vector<double> x(n), y(n);
    for (std::size_t b = 0; b < n; ++b) { x[b] = x_dist(rng); y[b] = y_dist(rng); }
    const auto ratio = alea::numerics::ratio_of_means(x, y);
    const double delta_method = std::sqrt((0.4 * 0.4 / (0.6 * 0.6) +
        1.5 * 1.5 * 0.1 * 0.1 / std::pow(0.6, 4)) / static_cast<double>(n));
    CHECK(ratio.mean == Catch::Approx(1.5 / 0.6).epsilon(0.01));
    // The jackknife error is itself an estimate from n_c coarse blocks, with
    // relative error-on-error 1/sqrt(2(n_c-1)); compare within 3 of those.
    const double relative_eoe =
        1.0 / std::sqrt(2.0 * (static_cast<double>(ratio.n_coarse_blocks) - 1.0));
    CHECK(std::abs(ratio.error - delta_method) <= 3.0 * relative_eoe * delta_method);
}

TEST_CASE("ratio_of_means refuses mismatched series and reports an undefined ratio",
          "[statistics][ratio]") {
    const std::vector<double> a{1.0, 2.0, 3.0}, b{1.0, 1.0};
    CHECK_THROWS_AS(alea::numerics::ratio_of_means(a, b), std::invalid_argument);
    const std::vector<double> zero{1.0, -1.0, 0.0};
    const auto ratio = alea::numerics::ratio_of_means(a, zero);
    CHECK(std::isnan(ratio.mean));
    CHECK_FALSE(ratio.reliable);
}

// ── robust_spread: 1.4826 · MAD ──────────────────────────────────────────────

TEST_CASE("robust_spread is 1.4826 times the median absolute deviation", "[statistics]") {
    std::vector<double> scratch;
    // Odd count: median 3, deviations {2, 1, 0, 1, 2} → MAD 1.
    const std::vector<double> odd{5.0, 1.0, 3.0, 2.0, 4.0};
    CHECK(alea::numerics::robust_spread(odd, scratch) == Catch::Approx(1.4826));
    // Even count: median (2 + 3)/2 = 2.5, deviations {1.5, 0.5, 0.5, 1.5} → 1.
    const std::vector<double> even{4.0, 1.0, 3.0, 2.0};
    CHECK(alea::numerics::robust_spread(even, scratch) == Catch::Approx(1.4826));
    // Shifting and scaling the data shifts nothing and scales the spread.
    std::vector<double> moved(odd);
    for (double& v : moved) v = -7.0 * v + 1e4;
    CHECK(alea::numerics::robust_spread(moved, scratch) == Catch::Approx(7.0 * 1.4826));
    // Constant data and empty data have no spread.
    const std::vector<double> flat(9, 2.5), none;
    CHECK(alea::numerics::robust_spread(flat, scratch) == 0.0);
    CHECK(alea::numerics::robust_spread(none, scratch) == 0.0);
}

TEST_CASE("robust_spread recovers a Gaussian sigma and ignores a minority of outliers",
          "[statistics]") {
    std::mt19937_64 rng(20261025);
    std::normal_distribution<double> gauss(-3.0, 2.0);
    std::vector<double> x(20001);
    for (double& v : x) v = gauss(rng);
    std::vector<double> scratch;
    // The MAD's sampling error at n = 2e4 is about 1.17σ/√n ≈ 0.8%.
    CHECK(alea::numerics::robust_spread(x, scratch) == Catch::Approx(2.0).epsilon(0.03));

    // 1% of the walkers at a divergence 1e6 below: the sample standard deviation
    // explodes by orders of magnitude, the robust spread moves by about 1%.
    for (std::size_t i = 0; i < x.size(); i += 100) x[i] = -1e6;
    double mean = 0.0, var = 0.0;
    for (double v : x) mean += v;
    mean /= static_cast<double>(x.size());
    for (double v : x) var += (v - mean) * (v - mean);
    const double sd = std::sqrt(var / static_cast<double>(x.size() - 1));
    CHECK(sd > 1e4);
    CHECK(alea::numerics::robust_spread(x, scratch) == Catch::Approx(2.0).epsilon(0.05));
}

namespace {
// K AR(1) chains x_t = a·x_{t−1} + √(1 − a²)·ε_t started in equilibrium,
// stored [step * K + chain]: ρ(k) = a^k, Σ_{k≥1} ρ(k) = a/(1 − a).
std::vector<double> ar1_chains(double a, std::size_t K, std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::vector<double> x(K * n);
    for (std::size_t c = 0; c < K; ++c) x[c] = gauss(rng);
    const double s = std::sqrt(1.0 - a * a);
    for (std::size_t t = 1; t < n; ++t)
        for (std::size_t c = 0; c < K; ++c)
            x[t * K + c] = a * x[(t - 1) * K + c] + s * gauss(rng);
    return x;
}
}  // namespace

TEST_CASE("integrated_autocorrelation recovers a/(1-a) of AR(1) chains", "[statistics]") {
    using alea::numerics::integrated_autocorrelation;
    // a = 0.8: Σρ = 4.  Sokal's window stops at W ≈ 5·(½ + 4) = 23, where the
    // truncated sum is 4·(1 − 0.8^23) = 3.98; 256 chains of 2000 steps make
    // the noise a few hundredths.
    const auto x = ar1_chains(0.8, 256, 2000, 11);
    const auto r = integrated_autocorrelation(x, 256);
    CHECK(r.window_found);
    CHECK(r.n_steps == 2000);
    CHECK(r.window >= 20);
    CHECK(r.window <= 30);
    CHECK(r.tau_int == Catch::Approx(4.0 * (1.0 - std::pow(0.8, r.window))).margin(0.15));

    // Uncorrelated chains: Σρ ≈ 0 and the window closes at once.
    const auto iid = ar1_chains(0.0, 256, 500, 12);
    const auto r0 = integrated_autocorrelation(iid, 256);
    CHECK(r0.window_found);
    CHECK(r0.window <= 4);
    CHECK(std::abs(r0.tau_int) < 0.02);
}

TEST_CASE("integrated_autocorrelation pools chains: more chains, same answer, less noise",
          "[statistics]") {
    using alea::numerics::integrated_autocorrelation;
    // The chain index is the fast one; one long chain and many short ones of
    // the same process estimate the same Σρ.
    const auto many = ar1_chains(0.5, 512, 400, 21);
    const auto one  = ar1_chains(0.5, 1, 200000, 22);
    const auto rm = integrated_autocorrelation(many, 512);
    const auto r1 = integrated_autocorrelation(one, 1);
    CHECK(rm.tau_int == Catch::Approx(1.0).margin(0.05));
    CHECK(r1.tau_int == Catch::Approx(1.0).margin(0.05));
}

TEST_CASE("integrated_autocorrelation reports a window it could not close", "[statistics]") {
    using alea::numerics::integrated_autocorrelation;
    // a = 0.99 (Σρ = 99) seen over 200 steps: the window needs ~500 lags.
    const auto x = ar1_chains(0.99, 64, 200, 31);
    const auto r = integrated_autocorrelation(x, 64);
    CHECK_FALSE(r.window_found);
    CHECK(r.window == 50);   // n/4
    CHECK(r.tau_int < 99.0);

    // Degenerate input: constant chains, a single step.
    const std::vector<double> flat(40, 3.0);
    CHECK(integrated_autocorrelation(flat, 4).tau_int == 0.0);
    CHECK_FALSE(integrated_autocorrelation(flat, 4).window_found);
    CHECK(integrated_autocorrelation(std::vector<double>(4, 1.0), 4).n_steps == 1);

    CHECK_THROWS_AS(integrated_autocorrelation(flat, 0), std::invalid_argument);
    CHECK_THROWS_AS(integrated_autocorrelation(flat, 3), std::invalid_argument);
    CHECK_THROWS_AS(integrated_autocorrelation(flat, 4, 0.0), std::invalid_argument);
}

TEST_CASE("integrated_time of a sampled Ornstein-Uhlenbeck process does not depend on the step",
          "[statistics]") {
    using alea::numerics::integrated_autocorrelation;
    using alea::numerics::integrated_time;
    // ρ(t) = e^{−t/θ} sampled every dt is AR(1) with a = e^{−dt/θ}.  The
    // trapezoid gives (dt/2)·coth(dt/2θ) = θ·(1 + (dt/θ)²/12 + …); the bare
    // sum dt·Σ_{k≥1}ρ(k) = dt/(e^{dt/θ} − 1) ≈ θ − dt/2 shrinks with dt.
    const double theta = 0.5;
    for (const double dt : {0.05, 0.18, 0.36}) {
        const double a = std::exp(-dt / theta);
        const auto x = ar1_chains(a, 512, 4000, 41);
        const auto r = integrated_autocorrelation(x, 512);
        REQUIRE(r.window_found);
        const double expected = 0.5 * dt / std::tanh(0.5 * dt / theta);
        CHECK(integrated_time(r, dt) == Catch::Approx(expected).epsilon(0.03));
        CHECK(std::abs(integrated_time(r, dt) / theta - 1.0) < 0.06);   // O(dt²)
        // The sum alone: 0.68·θ at dt = 0.36, as the bilayer's θ at 4τ1 was.
        CHECK(dt * r.tau_int == Catch::Approx(dt / std::expm1(dt / theta)).epsilon(0.03));
    }
}
