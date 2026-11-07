// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Versioned, full-precision HDF5 PIMC path checkpoints. This I/O boundary uses
// plain data: drivers import/export beads without reading or writing files.
#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

namespace alea {
class SimulationCell;
}

namespace alea::io {

// Complete path configuration (not RNG state). Each replica stores M*N*d beads
// in PathSet SoA order, without wrapping or converting coordinates, and its
// permutation pi (closing links (p, M-1) -> (pi(p), 0)). The format requires
// a bijection; the driver decides which exchanges a run admits (identity for
// distinguishable particles, within exchange classes otherwise).
struct PathCheckpoint {
    std::size_t replicas = 0;
    std::size_t m = 0;
    std::size_t np = 0;
    std::size_t dim = 0;
    double beta = 0.0;
    double tau = 0.0;
    std::vector<double> positions; // [replica][particle][dimension][slice]
    std::vector<std::size_t> permutation; // [replica][particle], pi per replica
    std::vector<double> lattice; // SimulationCell vectors [i*dim+d]; empty=open
};

// Validate nonzero/overflow-safe shapes, finite beads, positive consistent
// beta=M*tau (roundoff only), valid cell, and bijective permutations. Throws
// std::runtime_error; coordinates are never wrapped or silently reshaped.
void validate_path_checkpoint(const PathCheckpoint& checkpoint);

// Load the version-1 "alea_pimc_paths" format. Requires explicit SoA layout,
// boundary, shape, binary64 bead/time/cell data and unsigned-64 connectivity.
// Throws std::runtime_error for missing, corrupt, unsupported, or invalid data.
[[nodiscard]] PathCheckpoint read_paths(const std::filesystem::path& path);

// Validate and stage a complete checkpoint beside its destination, then replace
// the destination only after successful HDF5 close. Throws std::runtime_error on
// invalid data or I/O failure; preserves an existing destination on failure before
// replacement. Not a power-loss durability guarantee; no RNG/measurement state.
void write_paths(const std::filesystem::path& path,
                 const PathCheckpoint& checkpoint);

// Require exact replica/bead/particle/dimension shape, saved beta/tau, boundary,
// and cell vectors to match the current run. No time-grid interpolation, replica
// cycling, or alternative-cell reinterpretation. Throws std::runtime_error.
// The caller must import all beads over collapsed initialization and perform
// fresh equilibration; configuration restart is not bitwise RNG continuation.
void require_path_compatibility(const PathCheckpoint& checkpoint,
                                std::size_t replicas, std::size_t m,
                                std::size_t np, std::size_t dim, double beta,
                                double tau, const SimulationCell& cell);

} // namespace alea::io
