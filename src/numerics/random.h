// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>
#include <random>

namespace alea::numerics {

// Thin wrapper around std::mt19937_64. One instance per simulation
// (and, eventually, one per walker for parallelism).
class Random {
public:
    explicit Random(std::uint64_t seed) : engine_(seed) {}

    // Uniform sample in [0, 1).
    double uniform() { return uniform_dist_(engine_); }

    // Uniform sample in [lo, hi).
    double uniform(double lo, double hi) { return lo + (hi - lo) * uniform(); }

    // Standard normal sample (mean 0, stddev 1).
    double gaussian() { return gaussian_dist_(engine_); }

    std::mt19937_64& engine() { return engine_; }

private:
    std::mt19937_64 engine_;
    std::uniform_real_distribution<double> uniform_dist_{0.0, 1.0};
    std::normal_distribution<double>       gaussian_dist_{0.0, 1.0};
};

// SplitMix64 output function: a bijective 64-bit mixer whose outputs for
// consecutive inputs are statistically unrelated (Steele, Lea & Flood,
// OOPSLA 2014, "Fast splittable pseudorandom number generators"; constants
// from Vigna's reference splitmix64.c).
[[nodiscard]] constexpr std::uint64_t splitmix64(std::uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Seed of the independent stream `index` of kind `stream` under the run seed
// `base`.  Hashing, rather than base + index, keeps streams of different
// stages, replicas and purposes apart: base + s and base' + s' coincide
// whenever base - base' = s' - s, e.g. replica s+1 of one sampling stage and
// replica s of the next.  Distinct (base, stream, index) collide only with
// 64-bit hash probability.
[[nodiscard]] constexpr std::uint64_t stream_seed(std::uint64_t base,
                                                  std::uint64_t stream,
                                                  std::uint64_t index) noexcept {
    return splitmix64(splitmix64(splitmix64(base) ^ stream) ^ index);
}

}  // namespace alea::numerics
