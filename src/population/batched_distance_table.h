// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "population/simulation_cell.h"

namespace alea {

// Batched pairwise distance table for nw walkers and np particles.
//
// Data layout (SoA, walker index varies fastest):
//   data_[i * np * nw + j * nw + w]
//
// For a given pair (i, j), all walker distances r_ij(w) are contiguous in
// memory, enabling SIMD over the walker dimension.
//
// Distances are minimum-image when the cell is periodic: the stored r_ij is the
// separation to the NEAREST periodic image of j, which is the only choice
// invariant under a shift of the cell origin.  The table stores scalars only —
// components that need the folded displacement vector (for gradient directions)
// obtain it from SimulationCell::min_image_batch.
class BatchedDistanceTable {
public:
    // The cell is held by value: it is a small POD, and copying it removes any
    // lifetime coupling when a WalkerSet is cloned per OpenMP thread.  A
    // default-constructed (open) cell reproduces the open-boundary behaviour
    // exactly, with no fold in the generated code.
    BatchedDistanceTable(std::size_t nw, std::size_t np, std::size_t dim,
                         const SimulationCell& cell = SimulationCell{});

    std::size_t nw()  const noexcept { return nw_; }
    std::size_t np()  const noexcept { return np_; }
    std::size_t dim() const noexcept { return dim_; }

    const SimulationCell& cell() const noexcept { return cell_; }

    double operator()(std::size_t w, std::size_t i, std::size_t j) const noexcept {
        return data_[i * np_ * nw_ + j * nw_ + w];
    }

    // Contiguous distances r(i, j, ·) over all walkers — unit-stride in w,
    // suitable for feeding a vectorized kernel.
    std::span<const double> row(std::size_t i, std::size_t j) const noexcept {
        return std::span<const double>(data_.data() + (i * np_ * nw_ + j * nw_),
                                       nw_);
    }

    // Build all nw distance tables from SoA positions.
    // positions: [p * dim * nw + d * nw + w]
    void build_all(std::span<const double> positions);

    // Compute new distance row for particle p at new_pos for all walkers.
    // new_pos:   [dim * nw] SoA — proposed position of particle p
    // positions: [np * dim * nw] SoA — current positions of all particles
    // out:       [np * nw] SoA — r(p, k, w) for all k and walkers w
    // [w0, w0+wn) restricts the work to a contiguous walker range (a crowd);
    // wn == npos means "to the end". Stride is always the global nw().
    void update_batch(std::size_t p,
                      std::span<const double> positions,
                      std::span<const double> new_pos,
                      std::span<double>       out,
                      std::size_t w0 = 0,
                      std::size_t wn = npos) const;

    // Commit accepted rows for particle p over the walker range [w0, w0+wn).
    // new_row: [np * nw] output of update_batch
    // accept:  [nw] boolean mask
    void commit_batch(std::size_t p,
                      std::span<const double> new_row,
                      std::span<const bool>   accept,
                      std::size_t w0 = 0,
                      std::size_t wn = npos);

    // Replace a particle row for every configuration in [begin,end), with no
    // accept mask or arithmetic interpolation. Used after direct path updates;
    // exact assignment also avoids inf-inf in an unchanged singular row.
    void commit_unconditionally(std::size_t p, std::span<const double> new_row,
                                std::size_t begin, std::size_t end);

    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    // Rebuild the table for a new set of walkers: new column c is a copy of
    // old column src[c].  Supports reorder, duplication and removal — the
    // distances travel with the walker, so no recomputation from positions.
    // After the call nw() == src.size().
    void apply_permutation(std::span<const std::size_t> src);

    std::span<const double> data() const noexcept { return data_; }

private:
    // Boundary-policy-templated kernels.  Dispatch happens once per call, above
    // the walker loop, so the inner loop carries no runtime branch and the open
    // path compiles to exactly the code it did before periodic support existed.
    template <class BC>
    void build_all_impl(const double* __restrict pos);
    template <class BC>
    void update_batch_impl(std::size_t p, const double* __restrict pos,
                           const double* __restrict new_pos, double* __restrict out,
                           std::size_t w0, std::size_t w_end) const;

    std::size_t nw_, np_, dim_;
    SimulationCell cell_;
    std::vector<double> data_;  // [np * np * nw]
    std::vector<double> gathered_data_; // permutation destination, [np * np * new_nw]
    std::vector<double> mask_;  // commit_batch accept mask as 0.0/1.0 scratch
    // Folded-displacement scratch for the periodic path, [dim * nw].  Sized once
    // (ctor / apply_permutation) and never resized in a batch call: like mask_,
    // concurrent crowds write disjoint [w0, w_end) slices of it, so sharing it
    // across a DMC crowd split is race-free.  Unused on the open path.
    mutable std::vector<double> delta_;
};

}  // namespace alea
