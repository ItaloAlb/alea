// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

namespace alea::io {

// Walker-population checkpoint (HDF5).
//
// Purpose: carry a DMC walker population from one run into the next — either to
// resume/extend a long run, or to adiabatically continue a population across a
// slowly-varying external parameter, where a fresh start cannot reach a basin
// that is separated from every generic initial condition by an impassable
// barrier.
//
// File layout: one flat 1-D double dataset "positions" of length nw·np·dim in
// AoS row order (walker-major: [w][p][d]), plus root attributes n_walkers,
// n_particles, dim.  AoS rows are the natural exchange format: the reader
// scatters row w straight into Walker::set_positions.
//
// A periodic run also stores its cell as the attribute "lattice_vectors"
// (row-major [i*dim+d]); the loader compares it against the run's own cell so a
// continuation cannot silently resume a configuration sampled at a different
// density.  Absent (or empty) means the checkpoint came from an open-boundary
// run.

struct WalkerFile {
    std::size_t         nw  = 0;   // number of walkers stored
    std::size_t         np  = 0;   // particles per walker
    std::size_t         dim = 0;   // spatial dimension
    std::vector<double> positions; // [nw][np][dim] AoS, length nw·np·dim
    std::vector<double> lattice;   // cell vectors [i*dim+d]; empty → open
};

// Serialise a population to `path`.  `positions_soa` is the WalkerSet SoA
// ([p*dim*nw + d*nw + w]); it is de-interleaved to AoS rows on write.
// Failures are logged and swallowed: a failed checkpoint must not take down a
// finished simulation.  (write_run_result no longer swallows: it writes
// all-or-nothing and throws, and main reports the lost file.)
//
// `tag`/`blocks_done` make milestone snapshots self-describing (attrs "tag"
// and "blocks_done", e.g. "eq"/0, "a50"/512); defaults write neither.
void write_walkers(const std::filesystem::path& path,
                   std::size_t np, std::size_t dim, std::size_t nw,
                   std::span<const double> positions_soa,
                   std::string_view tag = {}, long long blocks_done = -1,
                   std::span<const double> lattice_vectors = {});

// Load a population checkpoint.  Throws std::runtime_error on a missing or
// malformed file — a continuation run silently falling back to random initial
// positions would corrupt the physics, so the failure must be loud.
[[nodiscard]] WalkerFile read_walkers(const std::filesystem::path& path);

}  // namespace alea::io
