// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace alea::numerics {

struct MeanError {
    double mean  = 0.0;
    double error = 0.0;
};

// Mean of the block means and the standard error of that mean, assuming the
// blocks are decorrelated:
//     mean  = (1/M) Σ_b  x_b
//     s²    = (1/(M-1)) Σ_b (x_b - mean)²     (sample variance)
//     error = sqrt(s² / M)
// For M < 2 the error cannot be estimated and is returned as 0.
[[nodiscard]] MeanError block_mean_error(std::span<const double> block_means);

// Flyvbjerg–Petersen reblocking ("Error estimates on averages of correlated
// data", J. Chem. Phys. 91, 461 (1989)), the C++ counterpart of
// tools/reblock.py: coarse-grain the series by averaging adjacent pairs,
// watch the standard error of the mean (SEM) grow with block length until it
// plateaus, and report the plateau value — the correlation-corrected error —
// together with the statistical inefficiency s = (error/naive)² and the
// integrated autocorrelation time tau_int = (s-1)/2 (in samples).
//
// Plateau onset: the smallest level whose increment to the next level is
// within the combined error-on-error (Jonsson, PRE 98, 043304 (2018)), only
// among levels with enough blocks for the SEM to be meaningful (>= 16 blocks,
// falling back to >= 4 for short series).  plateau_found is false when no
// onset could be verified (the noise-dominated tail was reached), meaning the
// error bar should be reported with a warning rather than trusted.
struct ReblockResult {
    double mean         = 0.0;
    double naive_error  = 0.0;  // SEM assuming independence (level 0)
    double error        = 0.0;  // plateau SEM (correlation-corrected)
    double tau_int      = 0.0;  // integrated autocorrelation time (samples)
    int    block_length = 1;    // samples per decorrelated block = ceil(s)
    int    level        = 0;    // plateau-onset level (2^level samples/block)
    bool   plateau_found = false;
    // Actual row supplying the conservative max-SEM error; level above stays
    // the historical plateau onset for backwards-compatible callers.
    int    selected_level = 0;
    std::size_t selected_n_blocks = 0;
    double error_on_error = 0.0; // selected SEM / sqrt(2*(n_blocks-1))
};

// Reblock a single series.  Needs at least 2 samples; for fewer the result
// degenerates (mean only, zero errors).  Never throws on short input.
[[nodiscard]] ReblockResult reblock(std::span<const double> series);

// Reblock K independent chains of equal length, stored one after another in
// `chains` ([chain * length + sample]), e.g. the per-thread block energies of
// VMC.  Each chain is coarse-grained on its own, so no block straddles two
// chains.  At level t the chains give independent estimates of the grand mean
// (1/K) Σ_i m_i, so
//     SEM_t = sqrt(Σ_i SEM_{i,t}²) / K,
//     error-on-error = SEM_t / sqrt(2 Σ_i (n_{i,t} - 1)),
// with n_{i,t} the blocks of chain i at level t; the plateau rule is the one
// of reblock(), applied to the pooled rows (n_blocks = Σ_i n_{i,t}).  K = 1 is
// exactly reblock().  tools/reblock.py's reblock_chains applies the same rule.
// Throws std::invalid_argument when K = 0 or the length is not a multiple of K.
[[nodiscard]] ReblockResult reblock_chains(std::span<const double> chains,
                                           std::size_t n_chains);

// Ratio of means R = <num>/<den> from per-block series of equal length: the
// signed Monte Carlo estimator <O s>/<s> (Ceperley, Rev. Mod. Phys. 67, 279
// (1995), Sec. VI; Dornheim, Groth & Bonitz, Phys. Rep. 744, 1 (2018)).
//
//   mean  = sum_b num_b / sum_b den_b over all blocks (the plain ratio);
//   error = jackknife over n_c coarse blocks of B = 2^L input blocks,
//           L = the larger selected reblocking level of the two series, so
//           the coarse blocks are decorrelated and the num/den covariance is
//           kept:  R_(j) = (X - X_j)/(Y - Y_j),
//           sigma^2 = (n_c-1)/n_c sum_j (R_(j) - mean_j R_(j))^2.
//   jackknife_bias = (n_c-1)(mean_j R_(j) - X/Y), the leading O(1/n_c) bias
//           of the ratio — reported, NOT subtracted from `mean`.
// With den constant the error is exactly the reblocked SEM of num/den at L.
// A zero denominator sum gives NaN mean and error (the ratio is undefined).
// Throws std::invalid_argument when the series lengths differ.
struct RatioResult {
    double mean = 0.0;
    double error = 0.0;
    double jackknife_bias = 0.0;
    int level = 0;                  // coarse blocks hold 2^level input blocks
    std::size_t n_coarse_blocks = 0;
    // Both series reached a reblocking plateau and n_coarse_blocks >= 16.
    bool reliable = false;
    ReblockResult numerator, denominator;
};
[[nodiscard]] RatioResult ratio_of_means(std::span<const double> num,
                                         std::span<const double> den);

// Robust standard deviation from the median absolute deviation,
//     s = 1.4826 · median_i |x_i − median(x)|,
// 1.4826 = 1/Φ⁻¹(3/4) making s = σ for a Gaussian (Hampel, J. Am. Stat. Assoc.
// 69, 383 (1974)).  Unlike the sample standard deviation it ignores a minority
// of outliers (breakdown point 50%), which is what the DMC branch cutoff needs:
// its scale must not be set by the very tail it is meant to cut.  The median of
// an even count is the mean of the two middle values.  `scratch` holds a copy
// of x (no allocation once its capacity suffices).  0 for an empty x.
[[nodiscard]] double robust_spread(std::span<const double> x,
                                   std::vector<double>& scratch);

// Integrated autocorrelation time of K equivalent stationary chains sampled
// together, stored step-major with the chain fastest ([step * K + chain],
// the walker-fastest layout of the drivers):
//     ρ(k)    = C(k)/C(0),  C(k) = ⟨(x_t − x̄)(x_{t+k} − x̄)⟩ over chains and t,
//               x̄ the grand mean of every chain and step;
//     tau_int = Σ_{k=1..W} ρ(k)   (samples; T = 1 + 2·tau_int is the
//               statistical inefficiency, the convention of ReblockResult),
// summed up to Sokal's automatic window, the smallest W with
// W ≥ window_factor·(½ + Σ_{k≤W} ρ(k)) (A. D. Sokal, "Monte Carlo methods in
// statistical mechanics", Cargèse lectures (1996), Sec. 3).  Averaging ρ(k)
// over the K chains makes it K times less noisy than one chain of the same
// length, so a short run of many walkers suffices.  window_found is false when
// no lag up to n_steps/4 met the condition (tau_int is then the sum to that
// lag, an underestimate).  Fewer than 2 steps, or chains with zero variance,
// give tau_int = 0 with window_found = false.  Throws std::invalid_argument
// when K = 0, the length is not a multiple of K, or window_factor ≤ 0.
struct AutocorrelationResult {
    double      tau_int      = 0.0;  // Σ_{k=1..W} ρ(k), in samples
    std::size_t window       = 0;    // W
    std::size_t n_steps      = 0;    // samples per chain
    bool        window_found = false;
};
[[nodiscard]] AutocorrelationResult integrated_autocorrelation(
    std::span<const double> series, std::size_t n_chains, double window_factor = 5.0);

// The correlation time of a continuous process sampled every dt:
//     θ = ∫_0^∞ ρ(t) dt ≈ dt·(½ + Σ_{k≥1} ρ(k)),
// the trapezoid rule, whose ½ is the half interval at t = 0 where ρ = 1.
// dt·Σ_{k≥1} ρ(k) alone is short by dt/2, which makes θ depend on the
// sampling step.  For an Ornstein–Uhlenbeck process, ρ(t) = e^{−t/θ}, the
// rule gives (dt/2)·coth(dt/2θ) = θ·(1 + (dt/θ)²/12 + …).
[[nodiscard]] inline double integrated_time(const AutocorrelationResult& r,
                                            double dt) noexcept {
    return dt * (0.5 + r.tau_int);
}

}  // namespace alea::numerics
