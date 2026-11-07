// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// PIMC path checkpoints preserve every bead and reject incompatible or corrupt
// continuations. Scratch files belong exclusively to these tests.
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>
#include "population/simulation_cell.h"

#include "io/path_io.h"

namespace fs = std::filesystem;
using alea::SimulationCell;
using alea::io::PathCheckpoint;
using Catch::Matchers::ContainsSubstring;

namespace {

// A fresh directory ensures cleanup never targets an existing user checkpoint.
struct Scratch {
    fs::path directory;
    Scratch() {
        static std::uint64_t serial = 0;
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        directory = fs::temp_directory_path() /
                    ("alea_path_io_" + std::to_string(tick) + "_" +
                     std::to_string(++serial));
        if (!fs::create_directory(directory))
            throw std::runtime_error("cannot create unique path_io scratch directory");
    }
    ~Scratch() { std::error_code ec; fs::remove_all(directory, ec); }
    fs::path file() const { return directory / "paths.h5"; }
};

// Distinct replica, particle, dimension, and slice digits expose transpositions.
PathCheckpoint example(bool periodic = false) {
    PathCheckpoint cp;
    cp.replicas = 2;
    cp.m = 8;
    cp.np = 3;
    cp.dim = 2;
    cp.beta = 2.0;
    cp.tau = 0.25;
    cp.positions.resize(cp.replicas * cp.m * cp.np * cp.dim);
    cp.permutation.resize(cp.replicas * cp.np);
    for (std::size_t r = 0; r < cp.replicas; ++r) {
        for (std::size_t p = 0; p < cp.np; ++p) {
            cp.permutation[r * cp.np + p] = p;
            for (std::size_t d = 0; d < cp.dim; ++d)
                for (std::size_t k = 0; k < cp.m; ++k)
                    cp.positions[((r * cp.np + p) * cp.dim + d) * cp.m + k] =
                        1000.0 * r + 100.0 * p + 10.0 * d + k +
                        std::nextafter(0.25, 1.0);
        }
    }
    cp.positions.front() = std::nextafter(1.0, 2.0);
    cp.positions[1] = -0.0;
    cp.positions.back() = -std::numeric_limits<double>::denorm_min();
    if (periodic) cp.lattice = {4.0, 0.0, 1.0, 5.0};
    return cp;
}

void check_equal(const PathCheckpoint& actual, const PathCheckpoint& expected) {
    CHECK(actual.replicas == expected.replicas);
    CHECK(actual.m == expected.m);
    CHECK(actual.np == expected.np);
    CHECK(actual.dim == expected.dim);
    CHECK(actual.beta == expected.beta);
    CHECK(actual.tau == expected.tau);
    REQUIRE(actual.positions.size() == expected.positions.size());
    for (std::size_t i = 0; i < actual.positions.size(); ++i)
        CHECK(std::bit_cast<std::uint64_t>(actual.positions[i]) ==
              std::bit_cast<std::uint64_t>(expected.positions[i]));
    CHECK(actual.permutation == expected.permutation);
    CHECK(actual.lattice == expected.lattice);
}

} // namespace

TEST_CASE("path_io preserves all open and periodic replica beads exactly", "[path_io]") {
    const bool periodic = GENERATE(false, true);
    const auto cp = example(periodic);
    Scratch scratch;
    REQUIRE_NOTHROW(alea::io::validate_path_checkpoint(cp));
    alea::io::write_paths(scratch.file(), cp);
    check_equal(alea::io::read_paths(scratch.file()), cp);
    HighFive::File file(scratch.file().string(), HighFive::File::ReadOnly);
    CHECK(file.getDataSet("positions").getDimensions() ==
          std::vector<std::size_t>{cp.positions.size()});
    CHECK(file.getDataSet("positions").getDataType().getSize() == sizeof(double));
    CHECK(file.getDataSet("permutation").getDimensions() ==
          std::vector<std::size_t>{cp.permutation.size()});
}

TEST_CASE("path_io rejects invalid sizes and overflow before any write", "[path_io]") {
    auto cp = example();
    SECTION("zero replicas") { cp.replicas = 0; }
    SECTION("zero slices") { cp.m = 0; }
    SECTION("zero particles") { cp.np = 0; }
    SECTION("zero dimensions") { cp.dim = 0; }
    SECTION("unsupported dimensions") { cp.dim = 4; }
    SECTION("positions truncated") { cp.positions.pop_back(); }
    SECTION("positions oversized") { cp.positions.push_back(0.0); }
    SECTION("permutation truncated") { cp.permutation.pop_back(); }
    SECTION("size product overflow") { cp.replicas = std::numeric_limits<std::size_t>::max(); }
    Scratch scratch;
    REQUIRE_THROWS_AS(alea::io::write_paths(scratch.file(), cp), std::runtime_error);
    CHECK_FALSE(fs::exists(scratch.file()));
    CHECK(fs::is_empty(scratch.directory));
}

TEST_CASE("path_io rejects nonfinite beads beta and tau", "[path_io]") {
    const double invalid = GENERATE(std::numeric_limits<double>::quiet_NaN(),
                                   std::numeric_limits<double>::infinity(),
                                   -std::numeric_limits<double>::infinity());
    auto cp = example();
    SECTION("bead") { cp.positions[20] = invalid; }
    SECTION("beta") { cp.beta = invalid; }
    SECTION("tau") { cp.tau = invalid; }
    REQUIRE_THROWS_AS(alea::io::validate_path_checkpoint(cp), std::runtime_error);
}

TEST_CASE("path_io requires positive consistent beta and tau", "[path_io]") {
    auto cp = example();
    SECTION("beta zero") { cp.beta = 0.0; }
    SECTION("beta negative") { cp.beta = -2.0; }
    SECTION("tau zero") { cp.tau = 0.0; }
    SECTION("tau negative") { cp.tau = -0.25; }
    SECTION("inconsistent slice count") { cp.beta = 2.25; }
    SECTION("subtle inconsistent beta") {
        cp.beta *= 1.0 + 64.0 * std::numeric_limits<double>::epsilon();
    }
    REQUIRE_THROWS_AS(alea::io::validate_path_checkpoint(cp), std::runtime_error);
}

TEST_CASE("path_io permits rounding in the beta equals M tau identity", "[path_io]") {
    auto cp = example();
    cp.m = 3;
    cp.beta = 0.3;
    cp.tau = 0.1;
    cp.positions.resize(cp.replicas * cp.m * cp.np * cp.dim);
    REQUIRE_NOTHROW(alea::io::validate_path_checkpoint(cp));
}

TEST_CASE("path_io requires a bijective permutation in every replica", "[path_io]") {
    auto cp = example();
    SECTION("duplicate") { cp.permutation[4] = 0; }
    SECTION("out of range") { cp.permutation[5] = cp.np; }
    REQUIRE_THROWS_AS(alea::io::validate_path_checkpoint(cp), std::runtime_error);
}

TEST_CASE("path_io stores exchanges; admissibility is the driver's decision", "[path_io]") {
    auto cp = example();
    std::swap(cp.permutation[3], cp.permutation[4]);
    REQUIRE_NOTHROW(alea::io::validate_path_checkpoint(cp));
}

TEST_CASE("path_io rejects malformed or nonfinite lattice vectors", "[path_io]") {
    auto cp = example(true);
    SECTION("short lattice") { cp.lattice.pop_back(); }
    SECTION("long lattice") { cp.lattice.push_back(0.0); }
    SECTION("singular lattice") { cp.lattice = {4.0, 0.0, 8.0, 0.0}; }
    SECTION("nonfinite lattice") { cp.lattice[2] = std::numeric_limits<double>::quiet_NaN(); }
    SECTION("infinite lattice") { cp.lattice[2] = std::numeric_limits<double>::infinity(); }
    SECTION("unsupported periodic dimension") {
        cp.dim = 3;
        cp.positions.resize(cp.replicas * cp.np * cp.dim * cp.m);
        cp.lattice = {4,0,0, 0,4,0, 0,0,4};
    }
    REQUIRE_THROWS_AS(alea::io::validate_path_checkpoint(cp), std::runtime_error);
}

TEST_CASE("path_io accepts supported open dimensions", "[path_io]") {
    const std::size_t dim = GENERATE(std::size_t{1}, std::size_t{2}, std::size_t{3});
    auto cp = example();
    cp.dim = dim;
    cp.positions.resize(cp.replicas * cp.np * dim * cp.m);
    REQUIRE_NOTHROW(alea::io::validate_path_checkpoint(cp));
}

TEST_CASE("path_io compatibility requires the exact shape time grid and cell", "[path_io]") {
    auto cp = example(true);
    std::size_t replicas = cp.replicas, m = cp.m, np = cp.np, dim = cp.dim;
    double beta = cp.beta, tau = cp.tau;
    auto lattice = cp.lattice;
    SECTION("replicas cannot be cycled") { ++replicas; }
    SECTION("slices") { ++m; }
    SECTION("particles") { ++np; }
    SECTION("dimension") { --dim; }
    SECTION("beta") { beta = std::nextafter(beta, 3.0); }
    SECTION("tau") { tau = std::nextafter(tau, 1.0); }
    SECTION("same volume different shape") { lattice = {5.0,0.0,1.0,4.0}; }
    SECTION("one ULP different lattice") { lattice[0] = std::nextafter(lattice[0], 5.0); }
    SECTION("periodic to open") { lattice.clear(); }
    const SimulationCell cell = lattice.empty() ? SimulationCell(dim) : SimulationCell(2, lattice);
    REQUIRE_THROWS_AS(alea::io::require_path_compatibility(cp, replicas, m, np,
                                                         dim, beta, tau, cell),
                      std::runtime_error);
}

TEST_CASE("path_io compatibility accepts matching geometry without wrapping data", "[path_io]") {
    const bool periodic = GENERATE(false, true);
    const auto cp = example(periodic);
    const SimulationCell cell = periodic ? SimulationCell(cp.dim, cp.lattice) : SimulationCell(cp.dim);
    REQUIRE_NOTHROW(alea::io::require_path_compatibility(cp, cp.replicas, cp.m, cp.np,
                                                       cp.dim, cp.beta, cp.tau, cell));
}

TEST_CASE("path_io accepts dimensionless default open cell for an open path", "[path_io]") {
    const auto cp = example();
    REQUIRE_NOTHROW(alea::io::require_path_compatibility(cp, cp.replicas, cp.m,
                                                       cp.np, cp.dim, cp.beta,
                                                       cp.tau, SimulationCell{}));
    REQUIRE_THROWS_AS(alea::io::require_path_compatibility(cp, cp.replicas, cp.m,
                                                         cp.np, cp.dim, cp.beta,
                                                         cp.tau, SimulationCell(1)),
                      std::runtime_error);
}

TEST_CASE("path_io replaces a valid checkpoint and preserves it on invalid overwrite", "[path_io]") {
    Scratch scratch;
    auto cp = example();
    alea::io::write_paths(scratch.file(), cp);
    cp.positions[10] = -3.141592653589793;
    alea::io::write_paths(scratch.file(), cp);
    check_equal(alea::io::read_paths(scratch.file()), cp);
    auto invalid = cp;
    invalid.positions[10] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(alea::io::write_paths(scratch.file(), invalid), std::runtime_error);
    check_equal(alea::io::read_paths(scratch.file()), cp);
    CHECK(std::distance(fs::directory_iterator(scratch.directory), fs::directory_iterator{}) == 1);
}

TEST_CASE("path_io cleans staging after failed replacement without deleting target", "[path_io]") {
    Scratch scratch;
    REQUIRE(fs::create_directory(scratch.file()));
    const auto marker = scratch.file() / "owned_directory";
    REQUIRE(fs::create_directory(marker));
    REQUIRE_THROWS_AS(alea::io::write_paths(scratch.file(), example()), std::runtime_error);
    CHECK(fs::is_directory(marker));
    CHECK(std::distance(fs::directory_iterator(scratch.directory), fs::directory_iterator{}) == 1);
}

TEST_CASE("path_io rejects missing files and failed writes loudly", "[path_io]") {
    Scratch scratch;
    REQUIRE_THROWS_AS(alea::io::read_paths(scratch.file()), std::runtime_error);
    REQUIRE_THROWS_AS(alea::io::write_paths(scratch.directory / "missing" / "paths.h5", example()),
                      std::runtime_error);
    REQUIRE_THROWS_AS(alea::io::write_paths({}, example()), std::runtime_error);
}

TEST_CASE("path_io rejects corrupt serialized metadata and values", "[path_io]") {
    Scratch scratch;
    const auto cp = example(true);
    alea::io::write_paths(scratch.file(), cp);
    {
        HighFive::File file(scratch.file().string(), HighFive::File::ReadWrite);
        SECTION("unknown version") { file.getAttribute("version").write(std::uint64_t{99}); }
        SECTION("wrong format") { file.getAttribute("format").write(std::string("alea_walkers")); }
        SECTION("wrong layout") { file.getAttribute("layout").write(std::string("aos")); }
        SECTION("wrong boundary") { file.getAttribute("boundary").write(std::string("open")); }
        SECTION("missing beta") { file.deleteAttribute("beta"); }
        SECTION("nonfinite tau") { file.getAttribute("tau").write(std::numeric_limits<double>::quiet_NaN()); }
        SECTION("wrong replica count") { file.getAttribute("n_replicas").write(std::uint64_t{3}); }
        SECTION("negative unsigned count") { file.getAttribute("n_replicas").write(std::uint64_t(-1)); }
        SECTION("bad permutation") {
            // Not a bijection (a swap is a valid exchange).
            auto permutation = cp.permutation;
            permutation[0] = permutation[1];
            file.getDataSet("permutation").write(permutation);
        }
        SECTION("nonfinite position") {
            auto positions = cp.positions;
            positions[1] = std::numeric_limits<double>::infinity();
            file.getDataSet("positions").write(positions);
        }
        SECTION("missing lattice") { file.deleteAttribute("lattice_vectors"); }
        SECTION("singular lattice") { file.getAttribute("lattice_vectors").write(std::vector<double>{4,0,8,0}); }
    }
    REQUIRE_THROWS_AS(alea::io::read_paths(scratch.file()), std::runtime_error);
}

TEST_CASE("path_io rejects numeric attribute rank or precision conversion", "[path_io]") {
    Scratch scratch;
    alea::io::write_paths(scratch.file(), example());
    {
        HighFive::File file(scratch.file().string(), HighFive::File::ReadWrite);
        SECTION("singleton vector instead of scalar beta") {
            file.deleteAttribute("beta");
            file.createAttribute("beta", std::vector<double>{2.0});
        }
        SECTION("binary32 beta") {
            file.deleteAttribute("beta");
            file.createAttribute("beta", 2.0f);
        }
        SECTION("signed replica metadata") {
            file.deleteAttribute("n_replicas");
            file.createAttribute("n_replicas", std::int64_t{2});
        }
    }
    REQUIRE_THROWS_AS(alea::io::read_paths(scratch.file()), std::runtime_error);
}

TEST_CASE("path_io rejects permutation dataset rank size and signed type", "[path_io]") {
    Scratch scratch;
    const auto cp = example();
    alea::io::write_paths(scratch.file(), cp);
    {
        HighFive::File file(scratch.file().string(), HighFive::File::ReadWrite);
        file.unlink("permutation");
        SECTION("short permutation") {
            file.createDataSet("permutation", std::vector<std::uint64_t>{0,1,2});
        }
        SECTION("permutation matrix") {
            auto ds = file.createDataSet<std::uint64_t>("permutation", HighFive::DataSpace({2,3}));
            const std::uint64_t identity[] = {0,1,2,0,1,2};
            ds.write_raw(identity);
        }
        SECTION("signed permutation") {
            file.createDataSet("permutation", std::vector<std::int64_t>{0,1,2,0,1,2});
        }
    }
    REQUIRE_THROWS_AS(alea::io::read_paths(scratch.file()), std::runtime_error);
}

TEST_CASE("path_io refuses walker checkpoint files as incompatible formats", "[path_io]") {
    Scratch scratch;
    {
        HighFive::File file(scratch.file().string(), HighFive::File::Excl);
        file.createAttribute("n_walkers", std::uint64_t{2});
        file.createAttribute("n_particles", std::uint64_t{3});
        file.createAttribute("dim", std::uint64_t{2});
        file.createDataSet("positions", std::vector<double>(12));
    }
    REQUIRE_THROWS_WITH(alea::io::read_paths(scratch.file()), ContainsSubstring("format"));
}

TEST_CASE("path_io rejects dataset rank and precision reinterpretation", "[path_io]") {
    Scratch scratch;
    const auto cp = example();
    alea::io::write_paths(scratch.file(), cp);
    {
        HighFive::File file(scratch.file().string(), HighFive::File::ReadWrite);
        file.unlink("positions");
        SECTION("matrix instead of flat SoA") {
            auto ds = file.createDataSet<double>("positions", HighFive::DataSpace({cp.replicas, cp.positions.size()/cp.replicas}));
            ds.write_raw(cp.positions.data());
        }
        SECTION("float instead of lossless double") {
            file.createDataSet("positions", std::vector<float>(cp.positions.size(), 0.0f));
        }
        SECTION("integer positions") {
            file.createDataSet("positions", std::vector<std::uint64_t>(cp.positions.size(), 0));
        }
    }
    REQUIRE_THROWS_AS(alea::io::read_paths(scratch.file()), std::runtime_error);
}
