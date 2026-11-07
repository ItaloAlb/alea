// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <span>

#include "numerics/xoshiro_random.h"

namespace alea {

// Cache-blocked sweeps for VMC and DMC.
//
// A sweep moves every particle once in every walker of a crowd.  With the
// particle loop outside and the walker loop inside, each walker's committed
// state is evicted between its own moves once the crowd's state exceeds the
// caches.  A 61-electron determinant keeps ~180 KB per walker (A = D⁻¹, D and
// its derivative rows), and each accepted rank-one update streams A twice, so
// a 1024-walker sweep becomes memory-bandwidth bound.  Sweeping a block of
// walkers through all particles before moving to the next block keeps the
// block's state cache-resident.
//
// Walkers do not interact within a sweep, so the order matters only through
// the random numbers.  They are drawn first, for the whole crowd, in the order
// the unblocked sweep consumed them: per particle p, dim·n Gaussian
// displacements (dimension-major), then n acceptance uniforms.  Each walker
// reads its own entries, so the trajectory is the same for every block size.
// Every (particle, walker) pair gets a uniform even when the acceptance test
// does not consume it (log A ≥ 0, or a node crossing rejected before the
// test); unused uniforms are discarded independently of the move, so each
// Metropolis test still uses a fresh U(0,1) and detailed balance is unchanged.

// Draw one sweep's random numbers for walkers [w_begin, w_end).
//   randoms: [np * (dim + 1) * stride]; entry ((p * (dim + 1) + d) * stride + w)
//            holds the Gaussian for dimension d < dim, the uniform for d = dim.
inline void draw_sweep_randoms(numerics::XoshiroRandom& rng, std::size_t np, std::size_t dim,
                               std::size_t stride, std::size_t w_begin, std::size_t w_end,
                               std::span<double> randoms) {
    for (std::size_t p = 0; p < np; ++p) {
        double* row = randoms.data() + p * (dim + 1) * stride;
        for (std::size_t d = 0; d < dim; ++d)
            for (std::size_t w = w_begin; w < w_end; ++w) row[d * stride + w] = rng.gaussian();
        for (std::size_t w = w_begin; w < w_end; ++w) row[dim * stride + w] = rng.uniform();
    }
}

// Walkers per cache block.  `requested` > 0 forces a size (measurements and
// tests); 0 sizes the block so its committed state fits `cache_budget`, about
// the per-core L3 share of the target CPUs (2.5 MB on a Xeon E5-2680 v4, 4 MB
// on an EPYC 7713).  The part that must stay resident is the determinant
// inverse, which every move rewrites; the other rows are streamed about once
// per sweep, so whole-state blocks somewhat above the share still work.
// Measured on the Xeon, 14 threads, HEG61 (216 KB per walker): DMC is flat
// at 2.86-2.89 s for blocks of 8-32 walkers versus 5.1 s unblocked, VMC
// within noise of unblocked; 4 MiB gives 19 walkers there and 8 for N = 127.
// The block never drops below `min_block` walkers, so walker-contiguous SoA
// loops keep at least two AVX2 vectors of work.  Small systems get the whole
// crowd.  The block size never changes results (see above); it only moves
// the cache/bandwidth trade-off, and the constant is a tunable, not a law.
inline std::size_t sweep_block_walkers(std::size_t requested, std::size_t bytes_per_walker,
                                       std::size_t n_walkers) {
    constexpr std::size_t cache_budget = std::size_t{4} << 20;
    constexpr std::size_t min_block = 8;
    if (n_walkers == 0) return 1;
    if (requested > 0) return std::min(requested, n_walkers);
    const std::size_t fit = bytes_per_walker > 0 ? cache_budget / bytes_per_walker : n_walkers;
    return std::min(n_walkers, std::max(fit, min_block));
}

}  // namespace alea
