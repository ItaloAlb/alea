// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "population/particle_batch.h"

#include <stdexcept>
#include <string>

namespace alea {

ParticleBatch::ParticleBatch(std::size_t nw, std::size_t np, std::size_t dim,
                             const SimulationCell& cell)
    // Normalise the cell's dimension to the batch's.  A default-constructed
    // SimulationCell is open with dim 0, and its dimension-length loops (the
    // open branch of min_image_batch, min_image_dist2) would then be no-ops for
    // callers that fold unconditionally rather than under `if constexpr`.
    : nw_(nw), np_(np), dim_(dim),
      cell_(cell.is_periodic() ? cell : SimulationCell(dim)),
      dist_tables_(nw, np, dim, cell_) {
    if (cell_.is_periodic() && cell_.dim() != dim)
        throw std::runtime_error(
            "ParticleBatch: simulation cell dimension " + std::to_string(cell_.dim()) +
            " does not match the configuration dimension " + std::to_string(dim));
    positions_.assign(np * dim * nw, 0.0);
    new_pos_.assign(dim * nw, 0.0);
    pending_dist_rows_.assign(np * nw, 0.0);
}

void ParticleBatch::refresh_particle_tables(std::size_t p,
        std::size_t begin, std::size_t end) {
    if (end > nw_) end = nw_;
    if (p >= np_ || begin > end)
        throw std::runtime_error("ParticleBatch::refresh_particle_tables: invalid range");
    if (begin == end) return;
    for (std::size_t d = 0; d < dim_; ++d)
        for (std::size_t k = begin; k < end; ++k)
            new_pos_[d * nw_ + k] = positions_[p * dim_ * nw_ + d * nw_ + k];
    dist_tables_.update_batch(p, positions_, new_pos_, pending_dist_rows_, begin, end - begin);
    dist_tables_.commit_unconditionally(p, pending_dist_rows_, begin, end);
}

void ParticleBatch::propose_move_batch(std::size_t p,
                                       std::span<const double> new_pos,
                                       std::size_t w_begin, std::size_t w_end) {
    if (w_end > nw_) w_end = nw_;
    if (w_begin >= w_end) return;
    // Record the proposed position so the evaluation and the commit can read
    // it (over the active range; new_pos may alias new_pos_).
    if (new_pos.data() != new_pos_.data()) {
        for (std::size_t d = 0; d < dim_; ++d)
            for (std::size_t w = w_begin; w < w_end; ++w)
                new_pos_[d * nw_ + w] = new_pos[d * nw_ + w];
    }
    dist_tables_.update_batch(p, positions_, new_pos, pending_dist_rows_,
                              w_begin, w_end - w_begin);
}

void ParticleBatch::commit_move(std::size_t p, std::span<const double> new_pos,
                                std::span<const bool> accept_mask,
                                std::size_t w_begin, std::size_t w_end) {
    if (w_end > nw_) w_end = nw_;
    if (w_begin >= w_end) return;
    dist_tables_.commit_batch(p, pending_dist_rows_, accept_mask,
                              w_begin, w_end - w_begin);
    for (std::size_t w = w_begin; w < w_end; ++w) {
        if (!accept_mask[w]) continue;
        for (std::size_t d = 0; d < dim_; ++d)
            positions_[p * dim_ * nw_ + d * nw_ + w] = new_pos[d * nw_ + w];
    }
    // Callers that need the raw unwrapped displacement ΔR = new_pos − pos
    // (the Langevin Metropolis–Hastings correction, DMC's rr_proposed /
    // rr_accepted) form it BEFORE committing, so wrapping here is safe.
    if (cell_.is_periodic()) {
        double r[SimulationCell::kMaxDim];
        for (std::size_t w = w_begin; w < w_end; ++w) {
            if (!accept_mask[w]) continue;
            for (std::size_t d = 0; d < dim_; ++d)
                r[d] = positions_[p * dim_ * nw_ + d * nw_ + w];
            cell_.wrap(r);
            for (std::size_t d = 0; d < dim_; ++d)
                positions_[p * dim_ * nw_ + d * nw_ + w] = r[d];
        }
    }
}

void ParticleBatch::wrap_all_positions() {
    if (!cell_.is_periodic()) return;
    double r[SimulationCell::kMaxDim];
    for (std::size_t w = 0; w < nw_; ++w)
        for (std::size_t p = 0; p < np_; ++p) {
            for (std::size_t d = 0; d < dim_; ++d)
                r[d] = positions_[p * dim_ * nw_ + d * nw_ + w];
            cell_.wrap(r);
            for (std::size_t d = 0; d < dim_; ++d)
                positions_[p * dim_ * nw_ + d * nw_ + w] = r[d];
        }
}

void ParticleBatch::permute_configurations(std::span<const std::size_t> src) {
    const std::size_t new_nw = src.size();
    gathered_positions_.resize(np_ * dim_ * new_nw);
    for (std::size_t pd = 0; pd < np_ * dim_; ++pd) {
        const double* __restrict o = positions_.data() + pd * nw_;
        double* __restrict n = gathered_positions_.data() + pd * new_nw;
        for (std::size_t j = 0; j < new_nw; ++j) n[j] = o[src[j]];
    }
    positions_.swap(gathered_positions_);
    dist_tables_.apply_permutation(src);

    nw_ = new_nw;
    new_pos_.assign(dim_ * new_nw, 0.0);
    pending_dist_rows_.assign(np_ * new_nw, 0.0);
}

}  // namespace alea
