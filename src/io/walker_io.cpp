// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "io/walker_io.h"

#include <exception>
#include <stdexcept>
#include <string>

#include <highfive/H5File.hpp>

#include "io/logger.h"

namespace alea::io {

void write_walkers(const std::filesystem::path& path,
                   std::size_t np, std::size_t dim, std::size_t nw,
                   std::span<const double> positions_soa,
                   std::string_view tag, long long blocks_done,
                   std::span<const double> lattice_vectors) {
    if (path.empty()) return;

    // De-interleave the WalkerSet SoA ([p*dim*nw + d*nw + w]) into AoS rows
    // ([w][p][d]) — the natural per-walker exchange format.
    std::vector<double> aos(nw * np * dim);
    for (std::size_t w = 0; w < nw; ++w)
        for (std::size_t p = 0; p < np; ++p)
            for (std::size_t d = 0; d < dim; ++d)
                aos[(w * np + p) * dim + d] =
                    positions_soa[p * dim * nw + d * nw + w];

    try {
        HighFive::File file(path.string(), HighFive::File::Overwrite);
        file.createDataSet("positions", aos);
        file.createAttribute("n_walkers",   nw);
        file.createAttribute("n_particles", np);
        file.createAttribute("dim",         dim);
        if (!tag.empty())
            file.createAttribute("tag", std::string(tag));
        if (blocks_done >= 0)
            file.createAttribute("blocks_done", blocks_done);
        if (!lattice_vectors.empty())
            file.createAttribute(
                "lattice_vectors",
                std::vector<double>(lattice_vectors.begin(), lattice_vectors.end()));
        Logger::milestone("walkers: wrote " + std::to_string(nw)
                          + " walkers to " + path.string());
    } catch (const std::exception& e) {
        Logger::warn("walkers: failed to write " + path.string() + ": "
                     + e.what());
    }
}

WalkerFile read_walkers(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path))
        throw std::runtime_error("walkers_in: file not found: " + path.string());

    WalkerFile wf;
    try {
        HighFive::File file(path.string(), HighFive::File::ReadOnly);
        file.getAttribute("n_walkers").read(wf.nw);
        file.getAttribute("n_particles").read(wf.np);
        file.getAttribute("dim").read(wf.dim);
        file.getDataSet("positions").read(wf.positions);
        // Optional: absent for checkpoints written by an open-boundary run.
        if (file.hasAttribute("lattice_vectors"))
            file.getAttribute("lattice_vectors").read(wf.lattice);
    } catch (const std::exception& e) {
        throw std::runtime_error("walkers_in: failed to read " + path.string()
                                 + ": " + e.what());
    }
    if (wf.nw == 0 || wf.positions.size() != wf.nw * wf.np * wf.dim)
        throw std::runtime_error("walkers_in: corrupt checkpoint " + path.string()
                                 + " (nw=" + std::to_string(wf.nw)
                                 + ", size=" + std::to_string(wf.positions.size())
                                 + ")");
    return wf;
}

}  // namespace alea::io
