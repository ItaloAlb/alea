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

// Complete closed-path configuration (not RNG state).  Each replica's path
// stores np rows (its capacity), of which rows [0, N_r) are particles: N_r =
// np in a canonical run, any N_r <= np in a grand-canonical one.  Only the
// particles are stored: replica r contributes N_r*d*M beads in PathSet SoA
// order [particle][dimension][slice], without wrapping or converting
// coordinates, and its permutation pi over [0, N_r) (closing links
// (p, M-1) -> (pi(p), 0)); the replicas follow one another.  The format
// requires a bijection; the driver decides which exchanges a run admits
// (identity for distinguishable particles, within exchange classes otherwise).
//
// File versions: 1 stores n_particles rows, all of them particles; 2 (written
// since grand-canonical runs) stores the capacity and N_r per replica.  Both
// are read.
struct PathCheckpoint {
    std::size_t replicas = 0;
    std::size_t m = 0;
    std::size_t np = 0;  // capacity: rows of each replica's path
    std::size_t dim = 0;
    double beta = 0.0;
    double tau = 0.0;
    std::vector<std::size_t> active;  // [replica], N_r <= np
    std::vector<double> positions; // replica r: [particle < N_r][dimension][slice]
    std::vector<std::size_t> permutation; // replica r: N_r entries, pi over [0, N_r)
    std::vector<double> lattice; // SimulationCell vectors [i*dim+d]; empty=open
};

// Offset of replica r's first particle in the packed per-particle arrays:
// sum of N_s over s < r (multiply by d*M for positions).  Needs a valid
// `active`.
[[nodiscard]] std::size_t first_particle(const PathCheckpoint& checkpoint,
                                         std::size_t replica);

// Validate nonzero/overflow-safe shapes (a nonzero capacity; a replica may
// hold no particle), N_r <= capacity, finite beads, positive consistent
// beta=M*tau (roundoff only), valid cell, and bijective permutations. Throws
// std::runtime_error; coordinates are never wrapped or silently reshaped.
void validate_path_checkpoint(const PathCheckpoint& checkpoint);

// Load an "alea_pimc_paths" file of version 1 (every row a particle: N_r =
// n_particles) or 2. Requires explicit SoA layout, boundary, shape, binary64
// bead/time/cell data and unsigned-64 counts and connectivity. Throws
// std::runtime_error for missing, corrupt, unsupported, or invalid data.
[[nodiscard]] PathCheckpoint read_paths(const std::filesystem::path& path);

// Write version 2. Validate and stage a complete checkpoint beside its destination, then replace
// the destination only after successful HDF5 close. Throws std::runtime_error on
// invalid data or I/O failure; preserves an existing destination on failure before
// replacement. Not a power-loss durability guarantee; no RNG/measurement state.
void write_paths(const std::filesystem::path& path,
                 const PathCheckpoint& checkpoint);

// Require the exact replica/slice/dimension shape, saved beta/tau, boundary
// and cell vectors of the current run, and every replica's particle count in
// [n_min, n_max] (n_min = n_max = N for a canonical run; the saved capacity
// is not compared). No time-grid interpolation, replica cycling, or
// alternative-cell reinterpretation. Throws std::runtime_error.
// The caller must import all beads over collapsed initialization and perform
// fresh equilibration; configuration restart is not bitwise RNG continuation.
void require_path_compatibility(const PathCheckpoint& checkpoint,
                                std::size_t replicas, std::size_t m,
                                std::size_t n_min, std::size_t n_max,
                                std::size_t dim, double beta, double tau,
                                const SimulationCell& cell);

} // namespace alea::io
