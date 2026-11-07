// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Contract for the diagonal electron-without-nearby-hole indicator in a PIMC
// path. For a uniform free pair in a 2D periodic cell, U_e = 1 - pi*r_cut^2/A
// when r_cut is below the Wigner-Seitz inradius. Reblocking follows Flyvbjerg
// and Petersen, J. Chem. Phys. 91, 461 (1989), on synchronized replica blocks.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "numerics/statistics.h"
#include "pimc/path_set.h"
#include "population/particle_group.h"
#include "qmc/run_result.h"

#include "estimators/electron_hole_proximity.h"

using namespace alea;
using Catch::Approx;

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr const char* kPrefix = "electron_hole_proximity_e_h";

// Name and charge each explicit, nonempty particle-group index range.
ParticleGroup group(std::string name, std::size_t first, std::size_t last,
                    double charge) {
    ParticleGroup g{};
    g.name = std::move(name);
    g.first = first;
    g.last = last;
    g.charge = charge;
    g.count = last - first;
    return g;
}

// Read one named observable, failing clearly if the requested name is absent.
std::span<const double> series(const RunResult& result, const std::string& name) {
    const auto it = std::find_if(result.series.begin(), result.series.end(),
                                 [&](const auto& entry) { return entry.first == name; });
    INFO("missing series: " << name);
    REQUIRE(it != result.series.end());
    return it->second;
}

// Read a numeric metadata field, retaining the exact threshold value.
double scalar(const RunResult& result, const std::string& name) {
    const auto it = std::find_if(result.scalars.begin(), result.scalars.end(),
                                 [&](const auto& entry) { return entry.first == name; });
    INFO("missing scalar: " << name);
    REQUIRE(it != result.scalars.end());
    return it->second;
}

// Read string provenance, including the measure's non-binding warning.
std::string metadata(const RunResult& result, const std::string& name) {
    const auto it = std::find_if(result.metadata.begin(), result.metadata.end(),
                                 [&](const auto& entry) { return entry.first == name; });
    INFO("missing metadata: " << name);
    REQUIRE(it != result.metadata.end());
    return it->second;
}

// Measure a completed block from one closed path with explicit slice weights.
void block(Estimator& estimator, const PathSet& path,
           std::span<const double> weights) {
    estimator.start_block();
    estimator.accumulate(path, weights);
    estimator.stop_block();
}
}  // namespace

TEST_CASE("Proximity counts electrons without ANY hole closer than the cutoff on each slice",
          "[pimc][electron_hole_proximity]") {
    const SimulationCell cell(2, std::vector<double>{4.0, 0.0, 0.0, 4.0});
    const std::vector<ParticleGroup> groups{
        group("e", 0, 2, -1.0), group("h", 2, 4, +1.0)};
    ElectronHoleProximity estimator(groups, "e", "h", 0.5, cell);
    PathSet path(4, 4, 2, 0.25, cell);
    // k=0: e0/h0 are across the periodic seam (r=.2), while e1/h1
    // are exactly on the threshold (r=.5, so e1 IS unproximized).
    path.pos(0, 0, 0) = -1.9; path.pos(1, 0, 0) = 0.5;
    path.pos(2, 0, 0) = +1.9; path.pos(3, 0, 0) = 1.0;
    // k=1: both electrons have at least one hole inside the cutoff.
    path.pos(0, 0, 1) = -1.9; path.pos(1, 0, 1) = 0.0;
    path.pos(2, 0, 1) = 0.0; path.pos(3, 0, 1) = +1.9;
    // k=2: neither electron has a nearby hole, regardless of pairing labels.
    path.pos(0, 0, 2) = 0.0; path.pos(1, 0, 2) = -1.0;
    path.pos(2, 0, 2) = 1.0; path.pos(3, 0, 2) = 1.5;
    // k=3: the same hole is nearest to both electrons, and both count as close.
    path.pos(0, 0, 3) = 0.0; path.pos(1, 0, 3) = 0.0;
    path.pos(2, 0, 3) = 0.0; path.pos(3, 0, 3) = 1.0;
    path.rebuild_tables();
    const std::vector<double> unit(4, 1.0);
    block(estimator, path, unit);
    RunResult result;
    estimator.finalize(result);
    CHECK(estimator.type() == "electron_hole_proximity");
    CHECK(series(result, std::string(kPrefix) + "_U_e")[0] == Approx(3.0 / 8.0));
    CHECK(series(result, std::string(kPrefix) + "_C_e")[0] == Approx(5.0 / 8.0));
    CHECK(scalar(result, std::string(kPrefix) + "_r_cut") == Approx(0.5));
    CHECK(metadata(result, std::string(kPrefix) + "_electron_group") == "e");
    CHECK(metadata(result, std::string(kPrefix) + "_hole_group") == "h");
    CHECK(metadata(result, std::string(kPrefix) + "_measure").find("without any hole")
          != std::string::npos);
    CHECK(metadata(result, std::string(kPrefix) + "_measure").find("not bound")
          != std::string::npos);
    CHECK(metadata(result, std::string(kPrefix) + "_finite_size_caveat").find("finite")
          != std::string::npos);

    ElectronHoleProximity weighted(groups, "e", "h", 0.5, cell);
    const std::vector<double> weights{1.0, 1.0, 2.0, 0.0};
    block(weighted, path, weights);
    RunResult wr;
    weighted.finalize(wr);
    CHECK(series(wr, std::string(kPrefix) + "_U_e")[0] == Approx(5.0 / 8.0));
    CHECK(series(wr, std::string(kPrefix) + "_C_e")[0] == Approx(3.0 / 8.0));

    ElectronHoleProximity open(groups, "e", "h", 0.5);
    PathSet open_path(1, 4, 2, 0.25);
    open_path.pos(0, 0, 0) = -1.9; open_path.pos(1, 0, 0) = 0.5;
    open_path.pos(2, 0, 0) = +1.9; open_path.pos(3, 0, 0) = 1.0;
    open_path.rebuild_tables();
    const std::vector<double> one{1.0};
    block(open, open_path, one);
    RunResult open_result;
    open.finalize(open_result);
    CHECK(series(open_result, std::string(kPrefix) + "_U_e")[0] == Approx(1.0));
    CHECK(metadata(open_result, std::string(kPrefix) + "_boundary") == "open");
}

TEST_CASE("Proximity validates the cutoff, named opposite charges, ranges, cell and weights",
          "[pimc][electron_hole_proximity]") {
    const SimulationCell cell(2, std::vector<double>{4.0, 0.0, 0.0, 4.0});
    const std::vector<ParticleGroup> groups{
        group("e", 0, 1, -1.0), group("h", 1, 2, +1.0)};
    for (const double invalid : {0.0, -0.5, std::numeric_limits<double>::infinity(),
                                  -std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::quiet_NaN()}) {
        CHECK_THROWS_AS(ElectronHoleProximity(groups, "e", "h", invalid, cell),
                        std::runtime_error);
    }
    CHECK_NOTHROW(ElectronHoleProximity(groups, "e", "h", 2.0, cell));
    CHECK_THROWS_AS(ElectronHoleProximity(groups, "e", "h", 2.01, cell),
                    std::runtime_error);
    CHECK_THROWS_AS(ElectronHoleProximity(groups, "missing", "h", 0.5),
                    std::runtime_error);
    CHECK_THROWS_AS(ElectronHoleProximity(groups, "e", "missing", 0.5),
                    std::runtime_error);
    CHECK_THROWS_AS(ElectronHoleProximity(groups, "e", "e", 0.5),
                    std::runtime_error);
    auto bad = groups;
    bad[1].charge = -1.0;
    CHECK_THROWS_AS(ElectronHoleProximity(bad, "e", "h", 0.5),
                    std::runtime_error);
    bad = groups;
    bad[0].charge = 0.0;
    CHECK_THROWS_AS(ElectronHoleProximity(bad, "e", "h", 0.5),
                    std::runtime_error);
    bad = groups;
    bad[1].charge = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(ElectronHoleProximity(bad, "e", "h", 0.5),
                    std::runtime_error);
    bad = groups;
    bad[1].first = bad[1].last;
    CHECK_THROWS_AS(ElectronHoleProximity(bad, "e", "h", 0.5),
                    std::runtime_error);
    bad = groups;
    bad[1].first = 0;
    CHECK_THROWS_AS(ElectronHoleProximity(bad, "e", "h", 0.5),
                    std::runtime_error);
    bad = groups;
    bad[1].name = "e";
    CHECK_THROWS_AS(ElectronHoleProximity(bad, "e", "h", 0.5),
                    std::runtime_error);
    bad = groups;
    bad[1].name.clear();
    CHECK_THROWS_AS(ElectronHoleProximity(bad, "e", "h", 0.5),
                    std::runtime_error);

    ElectronHoleProximity estimator(groups, "e", "h", 0.5, cell);
    PathSet path(2, 2, 2, 0.25, cell);
    path.rebuild_tables();
    estimator.start_block();
    const std::vector<double> short_weights{1.0};
    CHECK_THROWS_AS(estimator.accumulate(path, short_weights), std::runtime_error);
    const std::vector<double> negative_weights{1.0, -1.0};
    CHECK_THROWS_AS(estimator.accumulate(path, negative_weights), std::runtime_error);
    const std::vector<double> nan_weights{1.0, std::numeric_limits<double>::quiet_NaN()};
    CHECK_THROWS_AS(estimator.accumulate(path, nan_weights), std::runtime_error);
    PathSet wrong_cell(2, 2, 2, 0.25);
    wrong_cell.rebuild_tables();
    const std::vector<double> unit{1.0, 1.0};
    CHECK_THROWS_AS(estimator.accumulate(wrong_cell, unit), std::runtime_error);
}

TEST_CASE("Proximity averages synchronized replica blocks without joining chains",
          "[pimc][electron_hole_proximity][reblock]") {
    const auto groups = std::vector<ParticleGroup>{
        group("e", 0, 1, -1.0), group("h", 1, 2, +1.0)};
    constexpr std::size_t kBlocks = 256;
    ElectronHoleProximity blueprint(groups, "e", "h", 0.5);
    CHECK(blueprint.reblocking_storage_bytes(kBlocks) ==
          kBlocks * 2 * sizeof(double));
    blueprint.configure_reblocking(kBlocks);
    auto left = blueprint.clone();
    auto right = blueprint.clone();
    auto collector = blueprint.clone();
    PathSet path(1, 2, 2, 0.2);
    const std::vector<double> weights{1.0};
    std::vector<double> grand_chain;
    grand_chain.reserve(kBlocks);
    for (std::size_t b = 0; b < kBlocks; ++b) {
        const double left_u = ((b / 16) % 3 == 0) ? 1.0 : 0.0;
        path.pos(1, 0, 0) = left_u != 0.0 ? 1.0 : 0.0;
        path.rebuild_tables();
        block(*left, path, weights);
        path.pos(1, 0, 0) = 0.0;
        path.rebuild_tables();
        block(*right, path, weights);
        grand_chain.push_back(left_u * 0.5);
    }
    collector->collect(*left);
    collector->collect(*right);
    RunResult result;
    collector->finalize(result);
    const auto expected = numerics::reblock(grand_chain);
    const std::string u = std::string(kPrefix) + "_U_e";
    const std::string c = std::string(kPrefix) + "_C_e";
    CHECK(series(result, u)[0] == Approx(expected.mean));
    CHECK(series(result, u + "_error")[0] == Approx(expected.error));
    CHECK(series(result, u + "_error_naive")[0] == Approx(expected.naive_error));
    CHECK(series(result, u + "_reblock_length")[0] ==
          Approx(static_cast<double>(std::size_t{1} << expected.selected_level)));
    CHECK(series(result, c)[0] == Approx(1.0 - expected.mean));
    CHECK(series(result, c + "_error")[0] == Approx(expected.error));
    CHECK(scalar(result, u + "_reblock_replicas") == Approx(2.0));
    CHECK(scalar(result, c + "_reblock_replicas") == Approx(2.0));
    CHECK(expected.error > expected.naive_error);

    ElectronHoleProximity legacy(groups, "e", "h", 0.5);
    CHECK_THROWS_AS(legacy.collect(*left), std::runtime_error);
    ElectronHoleProximity mismatched(groups, "e", "h", 0.75);
    mismatched.configure_reblocking(kBlocks);
    for (std::size_t b = 0; b < kBlocks; ++b) block(mismatched, path, weights);
    CHECK_THROWS_AS(collector->collect(mismatched), std::runtime_error);
    CHECK_THROWS_AS(blueprint.configure_reblocking(kBlocks), std::runtime_error);
    CHECK_THROWS_AS(blueprint.reblocking_storage_bytes(
                        std::numeric_limits<std::size_t>::max()), std::runtime_error);
    ElectronHoleProximity oversized(groups, "e", "h", 0.5);
    CHECK_THROWS_AS(oversized.configure_reblocking(5 * 1024 * 1024),
                    std::runtime_error);
    CHECK_THROWS_AS(blueprint.finalize(result), std::runtime_error);
}

TEST_CASE("Uniform free electron and hole obey the 2D periodic disk exclusion limit",
          "[pimc][electron_hole_proximity][analytic]") {
    const SimulationCell cell(2, std::vector<double>{4.0, 0.0, 0.0, 4.0});
    const auto groups = std::vector<ParticleGroup>{
        group("e", 0, 1, -1.0), group("h", 1, 2, +1.0)};
    constexpr std::size_t kSide = 64;
    constexpr std::size_t kSlices = 32;
    constexpr std::size_t kBlocks = kSide * kSide / kSlices;
    constexpr double kCut = 0.5;
    ElectronHoleProximity estimator(groups, "e", "h", kCut, cell);
    estimator.configure_reblocking(kBlocks);
    PathSet path(kSlices, 2, 2, 0.125, cell);
    const std::vector<double> weights(kSlices, 1.0);
    std::vector<double> independent_block_means;
    for (std::size_t b = 0; b < kBlocks; ++b) {
        std::size_t free_count = 0;
        for (std::size_t k = 0; k < kSlices; ++k) {
            const auto index = b * kSlices + k;
            const auto x_index = index % kSide;
            const auto y_index = index / kSide;
            path.pos(1, 0, k) = 4.0 * (static_cast<double>(x_index) + 0.5) /
                                    static_cast<double>(kSide) - 2.0;
            path.pos(1, 1, k) = 4.0 * (static_cast<double>(y_index) + 0.5) /
                                    static_cast<double>(kSide) - 2.0;
            const double x = path.pos(1, 0, k), y = path.pos(1, 1, k);
            free_count += x * x + y * y >= kCut * kCut;
        }
        path.rebuild_tables();
        block(estimator, path, weights);
        independent_block_means.push_back(static_cast<double>(free_count) /
                                          static_cast<double>(kSlices));
    }
    RunResult result;
    estimator.finalize(result);
    const std::string u = std::string(kPrefix) + "_U_e";
    const double expected_integral = 1.0 - kPi * kCut * kCut / cell.volume();
    const auto stats = numerics::reblock(independent_block_means);
    CHECK(series(result, u)[0] == Approx(expected_integral).margin(0.002));
    CHECK(series(result, u)[0] == Approx(stats.mean));
    CHECK(series(result, u + "_error")[0] == Approx(stats.error));
    CHECK(series(result, u + "_error_reliable")[0] ==
          (stats.plateau_found && stats.selected_n_blocks >= 16 ? 1.0 : 0.0));
    CHECK(scalar(result, std::string(kPrefix) + "_r_cut") == Approx(kCut));
}

TEST_CASE("VMC/DMC block moments merge across thread clones like one chain of blocks",
          "[electron_hole_proximity][vmc]") {
    // The VMC driver gives every thread a clone and collects them; DMC weights
    // its walkers.  Without PIMC reblocking the estimator keeps block moments,
    // merged by Chan's parallel rule, so the thread split must not matter:
    // blocks U = {1/2, 1} on one clone and {0, 1/2} on the other give
    // mean 1/2 and SEM sqrt(sum (U - 1/2)^2 / ((n - 1) n)) = sqrt(1/24).
    const std::vector<ParticleGroup> groups{group("e", 0, 2, -1.0), group("h", 2, 4, +1.0)};
    PathSet config(1, 4, 2, 0.25);  // one configuration, open cell
    const std::vector<double> unit{1.0};
    // e0 at x = -1.9, e1 at x = 0.5; a hole on an electron makes it "near".
    const auto measure_u = [&](Estimator& e, bool e0_near, bool e1_near) {
        config.pos(0, 0, 0) = -1.9;
        config.pos(1, 0, 0) = 0.5;
        config.pos(2, 0, 0) = e0_near ? -1.9 : 1.9;
        config.pos(3, 0, 0) = e1_near ? 0.5 : 1.0;  // r = 0.5 = r_cut is not inside
        config.rebuild_tables();
        block(e, config, unit);
    };
    ElectronHoleProximity left(groups, "e", "h", 0.5);
    ElectronHoleProximity right(groups, "e", "h", 0.5);
    measure_u(left, true, false);   // U = 1/2
    measure_u(left, false, false);  // U = 1
    measure_u(right, true, true);   // U = 0
    measure_u(right, false, true);  // U = 1/2
    auto merged = left.clone();
    merged->collect(right);
    RunResult result;
    merged->finalize(result);
    const std::string u = std::string(kPrefix) + "_U_e";
    CHECK(series(result, u)[0] == Approx(0.5));
    CHECK(series(result, u + "_error")[0] == Approx(std::sqrt(1.0 / 24.0)));
    CHECK(series(result, std::string(kPrefix) + "_C_e")[0] == Approx(0.5));

    // DMC: walker weights enter as configuration weights within a block.
    ElectronHoleProximity weighted(groups, "e", "h", 0.5);
    PathSet walkers(2, 4, 2, 0.25);  // two walkers: U = 1 and U = 0
    for (std::size_t w = 0; w < 2; ++w) {
        walkers.pos(0, 0, w) = -1.9;
        walkers.pos(1, 0, w) = 0.5;
        walkers.pos(2, 0, w) = w == 0 ? 1.9 : -1.9;
        walkers.pos(3, 0, w) = w == 0 ? 1.0 : 0.5;
    }
    walkers.rebuild_tables();
    const std::vector<double> dmc_weights{3.0, 1.0};
    block(weighted, walkers, dmc_weights);
    RunResult wr;
    weighted.finalize(wr);
    CHECK(series(wr, u)[0] == Approx(3.0 / 4.0));
}
