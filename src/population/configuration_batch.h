// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

#include "particle_batch.h"

namespace alea {

// A ParticleBatch filled from stored configurations: positions and distance
// tables, nothing a trial wavefunction adds.  Forward walking measures the
// ancestors it keeps through it, with the same Estimator::accumulate and
// Hamiltonian::evaluate_potential the live walkers use.
//
// The batch is built once for `capacity` configurations; load() then sets any
// count up to it without reallocating (std::vector keeps its capacity when it
// shrinks or regrows within it).
class ConfigurationBatch final : public ParticleBatch {
public:
    ConfigurationBatch(std::size_t capacity, std::size_t np, std::size_t dim,
                       const SimulationCell& cell)
        : ParticleBatch(capacity, np, dim, cell), capacity_(capacity) {}

    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

    // Configuration c becomes column src[c] of `source`, an SoA position
    // buffer [np * dim * source_nw] (walker index fastest, already wrapped
    // into the cell); then the distance tables are rebuilt.  O(np² · src.size()).
    void load(std::span<const double> source, std::size_t source_nw,
              std::span<const std::size_t> src) {
        const std::size_t n = src.size();
        if (n > capacity_)
            throw std::logic_error("ConfigurationBatch: " + std::to_string(n) +
                                   " configurations exceed the capacity " +
                                   std::to_string(capacity_));
        if (source.size() < np_ * dim_ * source_nw)
            throw std::logic_error("ConfigurationBatch: source buffer too small");
        nw_ = n;
        positions_.resize(np_ * dim_ * n);
        new_pos_.resize(dim_ * n);
        pending_dist_rows_.resize(np_ * n);
        for (std::size_t pd = 0; pd < np_ * dim_; ++pd) {
            const double* __restrict from = source.data() + pd * source_nw;
            double* __restrict to = positions_.data() + pd * n;
            for (std::size_t c = 0; c < n; ++c) to[c] = from[src[c]];
        }
        dist_tables_.resize(n);
        dist_tables_.build_all(positions_);
    }

private:
    std::size_t capacity_;
};

}  // namespace alea
