// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "population/batched_distance_table.h"

#include <algorithm>
#include <cmath>

namespace alea {

BatchedDistanceTable::BatchedDistanceTable(std::size_t nw, std::size_t np,
                                           std::size_t dim,
                                           const SimulationCell& cell)
    : nw_(nw), np_(np), dim_(dim), cell_(cell), data_(np * np * nw, 0.0),
      mask_(nw, 0.0),
      delta_(cell.is_periodic() ? dim * nw : 0, 0.0) {}

template <class BC>
void BatchedDistanceTable::build_all_impl(const double* __restrict pos) {
    // Each (i,j) distance row is contiguous over walkers. Accumulate Σ_d diff²
    // in place, then take the square root — every loop over w is vectorizable.
    for (std::size_t i = 0; i < np_; ++i) {
        for (std::size_t j = 0; j < np_; ++j) {
            double* __restrict row = data_.data() + i * np_ * nw_ + j * nw_;
#pragma omp simd
            for (std::size_t w = 0; w < nw_; ++w) row[w] = 0.0;

            if constexpr (BC::periodic) {
                // Fold Δ into the Wigner–Seitz cell first: on a general Bravais
                // lattice the fold mixes Cartesian components, so it cannot be
                // done inside the per-dimension accumulation below.
                cell_.min_image_batch(pos + i * dim_ * nw_, pos + j * dim_ * nw_,
                                      nw_, delta_.data(), 0, nw_);
                for (std::size_t d = 0; d < dim_; ++d) {
                    const double* __restrict dl = delta_.data() + d * nw_;
#pragma omp simd
                    for (std::size_t w = 0; w < nw_; ++w) row[w] += dl[w] * dl[w];
                }
            } else {
                for (std::size_t d = 0; d < dim_; ++d) {
                    const double* __restrict xi = pos + (i * dim_ + d) * nw_;
                    const double* __restrict xj = pos + (j * dim_ + d) * nw_;
#pragma omp simd
                    for (std::size_t w = 0; w < nw_; ++w) {
                        const double diff = xi[w] - xj[w];
                        row[w] += diff * diff;
                    }
                }
            }
#pragma omp simd
            for (std::size_t w = 0; w < nw_; ++w) row[w] = std::sqrt(row[w]);
        }
    }
}

void BatchedDistanceTable::build_all(std::span<const double> positions) {
    if (cell_.is_periodic()) build_all_impl<PeriodicBC>(positions.data());
    else                     build_all_impl<OpenBC>(positions.data());
}

template <class BC>
void BatchedDistanceTable::update_batch_impl(std::size_t p,
                                             const double* __restrict pos,
                                             const double* __restrict new_pos,
                                             double* __restrict out, std::size_t w0,
                                             std::size_t w_end) const {
    // Use the output row as the squared-distance accumulator (no scratch).
    for (std::size_t k = 0; k < np_; ++k) {
        double* __restrict row = out + k * nw_;
        if (k == p) {
#pragma omp simd
            for (std::size_t w = w0; w < w_end; ++w) row[w] = 0.0;  // self distance
            continue;
        }
#pragma omp simd
        for (std::size_t w = w0; w < w_end; ++w) row[w] = 0.0;

        if constexpr (BC::periodic) {
            cell_.min_image_batch(new_pos, pos + k * dim_ * nw_, nw_, delta_.data(),
                                  w0, w_end);
            for (std::size_t d = 0; d < dim_; ++d) {
                const double* __restrict dl = delta_.data() + d * nw_;
#pragma omp simd
                for (std::size_t w = w0; w < w_end; ++w) row[w] += dl[w] * dl[w];
            }
        } else {
            for (std::size_t d = 0; d < dim_; ++d) {
                const double* __restrict xk = pos + (k * dim_ + d) * nw_;
                const double* __restrict xp = new_pos + d * nw_;
#pragma omp simd
                for (std::size_t w = w0; w < w_end; ++w) {
                    const double diff = xp[w] - xk[w];
                    row[w] += diff * diff;
                }
            }
        }
#pragma omp simd
        for (std::size_t w = w0; w < w_end; ++w) row[w] = std::sqrt(row[w]);
    }
}

void BatchedDistanceTable::update_batch(std::size_t p,
                                        std::span<const double> positions,
                                        std::span<const double> new_pos,
                                        std::span<double> out,
                                        std::size_t w0, std::size_t wn) const {
    const std::size_t w_end = (wn == npos) ? nw_ : std::min(w0 + wn, nw_);
    if (cell_.is_periodic())
        update_batch_impl<PeriodicBC>(p, positions.data(), new_pos.data(), out.data(),
                                      w0, w_end);
    else
        update_batch_impl<OpenBC>(p, positions.data(), new_pos.data(), out.data(), w0,
                                  w_end);
}

void BatchedDistanceTable::commit_batch(std::size_t p,
                                        std::span<const double> new_row,
                                        std::span<const bool> accept,
                                        std::size_t w0, std::size_t wn) {
    const std::size_t w_end = (wn == npos) ? nw_ : std::min(w0 + wn, nw_);
    // Materialise the accept mask as 0.0/1.0 so the per-row selects below operate
    // purely on doubles and vectorize.  mask_ is pre-sized to nw_ (ctor /
    // apply_permutation): it must NOT be resized here — commit_batch runs
    // concurrently over disjoint crowds on the shared table, and a resize would
    // reallocate the shared buffer from several threads at once (double-free).
    // The per-w writes below are disjoint across crowds, hence race-free.
    for (std::size_t w = w0; w < w_end; ++w) mask_[w] = accept[w] ? 1.0 : 0.0;
    const double* __restrict m = mask_.data();

    for (std::size_t k = 0; k < np_; ++k) {
        if (k == p) continue;  // self distance stays 0; avoids aliasing the two rows
        double* __restrict row_pk = data_.data() + p * np_ * nw_ + k * nw_;
        double* __restrict row_kp = data_.data() + k * np_ * nw_ + p * nw_;
        const double* __restrict nr = new_row.data() + k * nw_;
        // Branchless select: accepted → new row, rejected → unchanged.
#pragma omp simd
        for (std::size_t w = w0; w < w_end; ++w) {
            row_pk[w] += m[w] * (nr[w] - row_pk[w]);
            row_kp[w] += m[w] * (nr[w] - row_kp[w]);
        }
    }
}

void BatchedDistanceTable::commit_unconditionally(std::size_t p,
        std::span<const double> new_row, std::size_t begin, std::size_t end) {
    for (std::size_t j = 0; j < np_; ++j) {
        const double* source = new_row.data() + j * nw_;
        double* row_pj = data_.data() + (p * np_ + j) * nw_;
        double* row_jp = data_.data() + (j * np_ + p) * nw_;
#pragma omp simd
        for (std::size_t k = begin; k < end; ++k) {
            row_pj[k] = source[k];
            row_jp[k] = source[k];
        }
    }
}

void BatchedDistanceTable::apply_permutation(std::span<const std::size_t> src) {
    const std::size_t new_nw = src.size();
    gathered_data_.resize(np_ * np_ * new_nw);
    // For each pair (i,j), gather the per-walker row: new[c] = old[src[c]].
    // Rows are independent, so DMC's serial branch step spreads them over the
    // worker team (np²·nw doubles, ~30 MB for 61 particles and 1024 walkers).
    const auto rows = static_cast<std::ptrdiff_t>(np_ * np_);
#pragma omp parallel for schedule(static) if (np_ * np_ * new_nw > (std::size_t{1} << 16))
    for (std::ptrdiff_t row = 0; row < rows; ++row) {
        const double* __restrict old_row = data_.data() + static_cast<std::size_t>(row) * nw_;
        double* __restrict new_row = gathered_data_.data() + static_cast<std::size_t>(row) * new_nw;
        for (std::size_t c = 0; c < new_nw; ++c)
            new_row[c] = old_row[src[c]];
    }
    data_.swap(gathered_data_);
    nw_   = new_nw;
    mask_.assign(new_nw, 0.0);  // keep commit_batch's scratch sized to nw_
    if (cell_.is_periodic()) delta_.assign(dim_ * new_nw, 0.0);
}

}  // namespace alea
