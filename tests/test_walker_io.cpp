// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/walker_io.h"

namespace fs = std::filesystem;

namespace {

// Unique scratch path for this test binary; removed at the end of each test.
fs::path scratch_file(const char* name) {
    return fs::temp_directory_path()
           / (std::string("alea_walker_io_") + name + ".h5");
}

}  // namespace

TEST_CASE("walker_io round-trips a SoA population", "[walker_io]") {
    const std::size_t nw = 5, np = 3, dim = 2;

    // Fill the WalkerSet SoA [p*dim*nw + d*nw + w] with a distinct value per
    // (w, p, d) so any indexing slip in the de-interleave shows up.
    std::vector<double> soa(np * dim * nw);
    for (std::size_t p = 0; p < np; ++p)
        for (std::size_t d = 0; d < dim; ++d)
            for (std::size_t w = 0; w < nw; ++w)
                soa[p * dim * nw + d * nw + w] =
                    100.0 * static_cast<double>(w)
                    + 10.0 * static_cast<double>(p)
                    + static_cast<double>(d);

    const fs::path path = scratch_file("roundtrip");
    alea::io::write_walkers(path, np, dim, nw, soa);
    REQUIRE(fs::exists(path));

    const alea::io::WalkerFile wf = alea::io::read_walkers(path);
    CHECK(wf.nw == nw);
    CHECK(wf.np == np);
    CHECK(wf.dim == dim);
    REQUIRE(wf.positions.size() == nw * np * dim);

    // AoS row order [w][p][d] must match the de-interleaved SoA exactly
    // (pure copy, so bitwise equality is the right check).
    for (std::size_t w = 0; w < nw; ++w)
        for (std::size_t p = 0; p < np; ++p)
            for (std::size_t d = 0; d < dim; ++d)
                CHECK(wf.positions[(w * np + p) * dim + d]
                      == soa[p * dim * nw + d * nw + w]);

    fs::remove(path);
}

TEST_CASE("read_walkers throws on a missing file", "[walker_io]") {
    const fs::path path = scratch_file("does_not_exist");
    fs::remove(path);
    REQUIRE_THROWS_AS(alea::io::read_walkers(path), std::runtime_error);
}

TEST_CASE("write_walkers swallows write failures", "[walker_io]") {
    // A failed checkpoint must not take down a finished simulation: writing to
    // an unwritable location logs a warning instead of throwing.
    const std::vector<double> soa(4, 1.0);  // np=2, dim=1, nw=2
    REQUIRE_NOTHROW(alea::io::write_walkers(
        "/nonexistent_dir_alea_walker_io/wk.h5", 2, 1, 2, soa));
}
