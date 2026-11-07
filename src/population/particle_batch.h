// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "batched_distance_table.h"

namespace alea {

// A batch of nw particle configurations R_w = (r_1, …, r_np) in one simulation
// cell, with their pair-distance tables — everything a potential V(R) or a
// structural estimator (density, g(r), S(k)) needs, and nothing a trial
// wavefunction adds.
//
// What a "configuration" means is up to the owner:
//   WalkerSet : ParticleBatch   one configuration per walker (VMC/DMC),
//   PathSet   : ParticleBatch   one configuration per imaginary-time slice (PIMC).
// Consumers that only need positions take `const ParticleBatch&`, so the same
// potential and estimator code serves every driver.
//
// SoA layout with the configuration index varying fastest:
//   positions: [p * dim * nw + d * nw + w]
//   new_pos:   [d * nw + w]            (one proposed particle at a time)
//   pending:   [k * nw + w]            (distances of the proposed particle)
//
// Positions are kept wrapped into the cell under periodic boundaries.
//
// A non-virtual base: no vtable and no indirection on the hot path.  The
// destructor is protected so a derived batch can never be deleted through a
// ParticleBatch pointer.
class ParticleBatch {
public:
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    ParticleBatch(const ParticleBatch&)            = delete;
    ParticleBatch& operator=(const ParticleBatch&) = delete;

    const SimulationCell& cell() const noexcept { return cell_; }

    // Number of configurations (walkers in VMC/DMC, slices in PIMC).  It is the
    // stride of every [.. * nw + w] array.
    std::size_t num_walkers() const noexcept { return nw_; }
    std::size_t n_particles() const noexcept { return np_; }
    std::size_t dim()         const noexcept { return dim_; }

    // positions: [p * dim * nw + d * nw + w]
    std::span<double>       positions()       { return positions_; }
    std::span<const double> positions() const { return positions_; }

    // Proposed position of the particle being moved: [d * nw + w].
    std::span<double>       new_pos()       { return new_pos_; }
    std::span<const double> new_pos() const { return new_pos_; }

    // Pending distance rows from propose_move_batch: [k * nw + w]
    std::span<const double> pending_dist_rows() const { return pending_dist_rows_; }

    BatchedDistanceTable&       dist_tables()       { return dist_tables_; }
    const BatchedDistanceTable& dist_tables() const { return dist_tables_; }

    // Compute pending distance rows for particle p at new_pos, over the
    // configuration range [w_begin, w_end) (default = whole batch).  Must be
    // called before the move is evaluated and committed.
    void propose_move_batch(std::size_t p, std::span<const double> new_pos,
                            std::size_t w_begin = 0, std::size_t w_end = npos);

    // Refresh only particle p's distance rows from current positions over
    // [begin,end). Reuses proposal scratch; invalid ranges throw runtime_error.
    // Does not change coordinates and does not retain a proposed move.
    void refresh_particle_tables(std::size_t p, std::size_t begin = 0,
                                 std::size_t end = npos);

protected:
    // The cell is held by value (small POD, no lifetime coupling when a batch
    // is cloned per OpenMP thread).
    ParticleBatch(std::size_t nw, std::size_t np, std::size_t dim,
                  const SimulationCell& cell);
    ~ParticleBatch() = default;
    ParticleBatch(ParticleBatch&&) noexcept            = default;
    ParticleBatch& operator=(ParticleBatch&&) noexcept = default;

    // Commit particle p's proposed move for the accepted configurations of
    // [w_begin, w_end): distance rows, then positions, then wrap into the cell.
    // The wrap runs in its own loop after the commit so the open path executes
    // exactly the code it did before periodic support existed.
    void commit_move(std::size_t p, std::span<const double> new_pos,
                     std::span<const bool> accept_mask,
                     std::size_t w_begin, std::size_t w_end);

    // Fold every position into the cell (no-op when open).  Checkpointed or
    // externally supplied coordinates may sit in any image; every observable is
    // invariant under a lattice translation, so this only keeps coordinates
    // bounded and lets one-body terms read positions modulo the cell.
    void wrap_all_positions();

    // New configuration column c is a copy of old column src[c] (reorder,
    // duplicate, remove).  Positions and distance tables travel; the
    // per-proposal scratch is resized.  Afterwards num_walkers() == src.size().
    void permute_configurations(std::span<const std::size_t> src);

    std::size_t    nw_, np_, dim_;
    SimulationCell cell_;

    std::vector<double> positions_;          // [np * dim * nw]
    std::vector<double> gathered_positions_; // permutation destination, same layout
    std::vector<double> new_pos_;            // [dim * nw]
    std::vector<double> pending_dist_rows_;  // [np * nw]

    BatchedDistanceTable dist_tables_;
};

}  // namespace alea
