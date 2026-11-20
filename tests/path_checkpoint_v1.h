// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// A version-1 "alea_pimc_paths" file, as written before the format stored
// per-replica particle counts: every one of n_particles rows is a particle.
// The current writer produces version 2 only, so the tests that read the
// older format build it here.
#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>

#include "io/path_io.h"

namespace alea::test {

inline void write_version1_paths(const std::filesystem::path& path,
                                 const alea::io::PathCheckpoint& cp) {
    for (const std::size_t n : cp.active)
        if (n != cp.np)
            throw std::invalid_argument("a version-1 path file has every row a particle");
    HighFive::File file(path.string(), HighFive::File::Excl);
    file.createAttribute("format", std::string("alea_pimc_paths"));
    file.createAttribute("version", std::uint64_t{1});
    file.createAttribute("layout", std::string("replica_particle_dimension_slice_soa"));
    file.createAttribute("n_replicas", static_cast<std::uint64_t>(cp.replicas));
    file.createAttribute("n_slices", static_cast<std::uint64_t>(cp.m));
    file.createAttribute("n_particles", static_cast<std::uint64_t>(cp.np));
    file.createAttribute("dim", static_cast<std::uint64_t>(cp.dim));
    file.createAttribute("beta", cp.beta);
    file.createAttribute("tau", cp.tau);
    file.createAttribute("boundary", std::string(cp.lattice.empty() ? "open" : "periodic"));
    if (!cp.lattice.empty()) file.createAttribute("lattice_vectors", cp.lattice);
    file.createDataSet("positions", cp.positions);
    file.createDataSet("permutation",
                       std::vector<std::uint64_t>(cp.permutation.begin(), cp.permutation.end()));
}

}  // namespace alea::test
