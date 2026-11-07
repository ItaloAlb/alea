// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// io::write_run_result: a results file either holds everything a run produced
// or does not exist, and a write that fails says so.  A half-written HDF5 file
// opens without complaint in any reader, so "it exists" must mean "it is
// complete".
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>

#include "io/results_writer.h"
#include "qmc/run_result.h"

namespace fs = std::filesystem;
using Catch::Matchers::ContainsSubstring;

namespace {

// A fresh, empty run directory for one test case.
fs::path fresh_run_dir(const char* name) {
    const fs::path dir = fs::temp_directory_path() / (std::string("alea_results_writer_") + name);
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

alea::RunResult vmc_result() {
    alea::RunResult r;
    r.method = "vmc";
    r.add_scalar("energy", -1.25);
    r.add_scalar("energy_error", 0.01);
    r.add_metadata("seed", "42");
    r.add_series("acc_block_energy", {-1.2, -1.3});
    r.add_series("gofr_r", {0.1, 0.3, 0.5});
    return r;
}

// Every entry of the run directory, so a stray partial file shows up.
std::vector<std::string> entries(const fs::path& dir) {
    std::vector<std::string> out;
    for (const auto& e : fs::directory_iterator(dir)) out.push_back(e.path().filename().string());
    return out;
}

}  // namespace

TEST_CASE("a written results file holds every series and attribute and nothing else is left",
          "[results_writer]") {
    const fs::path dir = fresh_run_dir("complete");
    alea::io::write_run_result(vmc_result(), dir, 0);
    CHECK(entries(dir) == std::vector<std::string>{"0_vmc.h5"});

    {  // closed before the cleanup: Windows will not delete an open file
        HighFive::File file((dir / "0_vmc.h5").string(), HighFive::File::ReadOnly);
        CHECK(file.getDataSet("acc_block_energy").read<std::vector<double>>() ==
              std::vector<double>{-1.2, -1.3});
        CHECK(file.getDataSet("gofr_r").read<std::vector<double>>() ==
              std::vector<double>{0.1, 0.3, 0.5});
        CHECK(file.getAttribute("energy").read<double>() == -1.25);
        CHECK(file.getAttribute("energy_error").read<double>() == 0.01);
        CHECK(file.getAttribute("seed").read<std::string>() == "42");
    }
    fs::remove_all(dir);
}

TEST_CASE("a failed write leaves no results file and raises the reason",
          "[results_writer]") {
    // Two datasets under one name cannot be stored: the second createDataSet
    // fails after the first series is already in the file.  That partial file
    // used to remain, without any attribute (the energy included), and the
    // error was only a log warning.
    const fs::path dir = fresh_run_dir("failed_series");
    alea::RunResult r = vmc_result();
    r.add_series("gofr_r", {9.0});
    CHECK_THROWS_WITH(alea::io::write_run_result(r, dir, 0),
                      ContainsSubstring("failed to write") && ContainsSubstring("0_vmc.h5"));
    CHECK(entries(dir).empty());

    // Scalars and metadata share the root attributes.
    alea::RunResult attrs = vmc_result();
    attrs.add_metadata("energy", "twice");
    CHECK_THROWS_WITH(alea::io::write_run_result(attrs, dir, 1),
                      ContainsSubstring("failed to write") && ContainsSubstring("1_vmc.h5"));
    CHECK(entries(dir).empty());
    fs::remove_all(dir);
}

TEST_CASE("a results file that already exists is replaced whole", "[results_writer]") {
    const fs::path dir = fresh_run_dir("replace");
    alea::RunResult first;
    first.method = "dmc";
    first.add_series("old_series", {1.0});
    alea::io::write_run_result(first, dir, 3);

    alea::RunResult second;
    second.method = "dmc";
    second.add_series("new_series", {2.0});
    alea::io::write_run_result(second, dir, 3);

    CHECK(entries(dir) == std::vector<std::string>{"3_dmc.h5"});
    {
        HighFive::File file((dir / "3_dmc.h5").string(), HighFive::File::ReadOnly);
        CHECK(file.listObjectNames() == std::vector<std::string>{"new_series"});
    }
    fs::remove_all(dir);
}

TEST_CASE("no run directory means no results file", "[results_writer]") {
    CHECK_NOTHROW(alea::io::write_run_result(vmc_result(), fs::path{}, 0));
}
