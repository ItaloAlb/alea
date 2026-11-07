// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <span>
#include <string>
#include <vector>

#include "estimators/pair_correlation.h"
#include "population/particle_group.h"
#include "population/walker.h"
#include "population/walker_set.h"
#include "qmc/run_result.h"

using alea::ParticleGroup;
using alea::PairCorrelation;
using alea::RunResult;
using alea::Walker;
using alea::WalkerSet;

namespace {

// Two single-particle groups "A" (particle 0) and "B" (particle 1).
std::vector<ParticleGroup> two_groups_AB() {
    return {ParticleGroup{"A", {}, {}, 0.0, /*first=*/0, /*last=*/1, /*count=*/1},
            ParticleGroup{"B", {}, {}, 0.0, /*first=*/1, /*last=*/2, /*count=*/1}};
}

const std::vector<double>& series(const RunResult& r, const std::string& name) {
    for (const auto& [n, v] : r.series)
        if (n == name) return v;
    FAIL("series not found: " << name);
    static const std::vector<double> empty;
    return empty;
}

bool has_series(const RunResult& r, const std::string& name) {
    for (const auto& [n, v] : r.series)
        if (n == name) return true;
    return false;
}

}  // namespace

TEST_CASE("PairCorrelation bins one pair into the bin of its distance",
          "[pair_correlation]") {
    // One walker, two particles: p0=(0,0), p1=(3,4) → r = 5.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2);
    std::vector<double> p0 = {0.0, 0.0, 3.0, 4.0};  // AoS [p*dim+d]
    w0.set_positions(p0);
    ws.add_walker(w0);
    ws.load_all();

    const auto groups = two_groups_AB();
    // rmax=10, n_bins=10 → Δ=1 → distance 5 lands in bin 5.
    PairCorrelation pc(std::span<const ParticleGroup>(groups), /*dim=*/2,
                       /*rmax=*/10.0, /*n_bins=*/10);

    pc.start_block();
    pc.accumulate(ws);
    pc.stop_block();

    RunResult r;
    pc.finalize(r);

    const std::vector<double>& g_eh = series(r, "gofr_A_B");
    REQUIRE(g_eh.size() == 10);
    CHECK(g_eh[5] > 0.0);
    for (std::size_t b = 0; b < g_eh.size(); ++b)
        if (b != 5) CHECK(g_eh[b] == Catch::Approx(0.0));

    // The radial axis is emitted once, with bin centers.
    const std::vector<double>& r_axis = series(r, "gofr_r");
    REQUIRE(r_axis.size() == 10);
    CHECK(r_axis[5] == Catch::Approx(5.5));
}

TEST_CASE("PairCorrelation enumerates all G(G+1)/2 group-pair functions",
          "[pair_correlation]") {
    std::vector<ParticleGroup> groups = {
        ParticleGroup{"A", {}, {}, 0.0, 0, 1, 1},
        ParticleGroup{"B", {}, {}, 0.0, 1, 2, 1},
        ParticleGroup{"C", {}, {}, 0.0, 2, 3, 1}};
    PairCorrelation pc(std::span<const ParticleGroup>(groups), 2, 10.0, 10);

    CHECK(pc.n_functions() == 6);  // 3 groups → 3*4/2
    // Triangular ids are distinct and dense over [0, 6).
    std::vector<std::size_t> ids = {
        pc.pair_id(0, 0), pc.pair_id(0, 1), pc.pair_id(0, 2),
        pc.pair_id(1, 1), pc.pair_id(1, 2), pc.pair_id(2, 2)};
    std::sort(ids.begin(), ids.end());
    CHECK(ids == std::vector<std::size_t>{0, 1, 2, 3, 4, 5});
    // Order-independent.
    CHECK(pc.pair_id(2, 0) == pc.pair_id(0, 2));
}

TEST_CASE("PairCorrelation skips empty (single-particle) AA functions but keeps AB",
          "[pair_correlation]") {
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2);
    std::vector<double> p0 = {0.0, 0.0, 3.0, 4.0};  // r = 5
    w0.set_positions(p0);
    ws.add_walker(w0);
    ws.load_all();

    const auto groups = two_groups_AB();
    PairCorrelation pc(std::span<const ParticleGroup>(groups), 2, 10.0, 10);

    // Single-particle groups → no intra-group pairs; one inter-group pair.
    CHECK(pc.pair_count(pc.pair_id(0, 0)) == 0);
    CHECK(pc.pair_count(pc.pair_id(1, 1)) == 0);
    CHECK(pc.pair_count(pc.pair_id(0, 1)) == 1);

    pc.start_block();
    pc.accumulate(ws);
    pc.stop_block();
    RunResult r;
    pc.finalize(r);

    CHECK(has_series(r, "gofr_A_B"));
    CHECK_FALSE(has_series(r, "gofr_A_A"));
    CHECK_FALSE(has_series(r, "gofr_B_B"));
}

TEST_CASE("PairCorrelation bins a real intra-group (AA) pair",
          "[pair_correlation]") {
    // One group of two electrons.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2);
    std::vector<double> p0 = {0.0, 0.0, 3.0, 4.0};  // r = 5
    w0.set_positions(p0);
    ws.add_walker(w0);
    ws.load_all();

    std::vector<ParticleGroup> groups = {
        ParticleGroup{"e", {}, {}, 0.0, /*first=*/0, /*last=*/2, /*count=*/2}};
    PairCorrelation pc(std::span<const ParticleGroup>(groups), 2, 10.0, 10);

    CHECK(pc.n_functions() == 1);
    CHECK(pc.pair_count(pc.pair_id(0, 0)) == 1);  // one e-e pair

    pc.start_block();
    pc.accumulate(ws);
    pc.stop_block();
    RunResult r;
    pc.finalize(r);

    const std::vector<double>& g_ee = series(r, "gofr_e_e");
    REQUIRE(g_ee.size() == 10);
    CHECK(g_ee[5] > 0.0);
}

TEST_CASE("PairCorrelation applies the 2D annulus normalization",
          "[pair_correlation]") {
    // One pair at r=5 → one count in bin 5; one block of unit weight.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2);
    std::vector<double> p0 = {0.0, 0.0, 3.0, 4.0};  // r = 5
    w0.set_positions(p0);
    ws.add_walker(w0);
    ws.load_all();

    const auto groups = two_groups_AB();
    PairCorrelation pc(std::span<const ParticleGroup>(groups), /*dim=*/2, 10.0, 10);
    pc.start_block();
    pc.accumulate(ws);
    pc.stop_block();
    RunResult r;
    pc.finalize(r);

    // g[bin] = (count/weight) / (n_pairs · annulus_area),  Δ = 1.
    // annulus(bin 5) = π(6² − 5²) = 11π, n_pairs = 1.
    constexpr double pi = 3.14159265358979323846;
    const double expected = 1.0 / (1.0 * pi * (36.0 - 25.0));
    CHECK(series(r, "gofr_A_B")[5] == Catch::Approx(expected));
}

TEST_CASE("PairCorrelation applies the 3D spherical-shell normalization",
          "[pair_correlation]") {
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/3);
    Walker w0(2, 3);
    std::vector<double> p0 = {0.0, 0.0, 0.0, 3.0, 4.0, 0.0};  // r = 5
    w0.set_positions(p0);
    ws.add_walker(w0);
    ws.load_all();

    const auto groups = two_groups_AB();
    PairCorrelation pc(std::span<const ParticleGroup>(groups), /*dim=*/3, 10.0, 10);
    pc.start_block();
    pc.accumulate(ws);
    pc.stop_block();
    RunResult r;
    pc.finalize(r);

    // shell(bin 5) = (4π/3)(6³ − 5³) = (4π/3)(216 − 125).
    constexpr double pi = 3.14159265358979323846;
    const double expected = 1.0 / ((4.0 / 3.0) * pi * (216.0 - 125.0));
    CHECK(series(r, "gofr_A_B")[5] == Catch::Approx(expected));
}

TEST_CASE("PairCorrelation weights each walker by its DMC weight (mixed)",
          "[pair_correlation]") {
    // Walker 0: pair at r=5 (bin 5), weight 3.  Walker 1: r=2 (bin 2), weight 1.
    WalkerSet ws(/*nw=*/2, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2), w1(2, 2);
    w0.set_positions(std::vector<double>{0.0, 0.0, 3.0, 4.0});  // r = 5
    w1.set_positions(std::vector<double>{0.0, 0.0, 2.0, 0.0});  // r = 2
    w0.set_weight(3.0);
    w1.set_weight(1.0);
    ws.add_walker(w0);
    ws.add_walker(w1);
    ws.load_all();  // pulls per-walker weights into the SoA

    const auto groups = two_groups_AB();
    PairCorrelation pc(std::span<const ParticleGroup>(groups), 2, 10.0, 10);
    pc.start_block();
    pc.accumulate(ws);
    pc.stop_block();
    RunResult r;
    pc.finalize(r);

    // Undo the shell factor → recover ⟨pairs-per-config in bin⟩, which is the
    // weight fraction: bin5 → 3/(3+1), bin2 → 1/(3+1).
    constexpr double pi = 3.14159265358979323846;
    const std::vector<double>& g = series(r, "gofr_A_B");
    CHECK(g[5] * (pi * (36.0 - 25.0)) == Catch::Approx(0.75));
    CHECK(g[2] * (pi * (9.0 - 4.0)) == Catch::Approx(0.25));
}

TEST_CASE("PairCorrelation reports a per-bin error from block scatter",
          "[pair_correlation]") {
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2);
    ws.add_walker(w0);

    const auto groups = two_groups_AB();
    PairCorrelation pc(std::span<const ParticleGroup>(groups), 2, 10.0, 10);

    // Block 1: pair in bin 5.  Block 2: pair in bin 2 (bin 5 empty).
    // → bin 5 sees block values {1, 0}: mean 0.5, std error 0.5.
    w0.set_positions(std::vector<double>{0.0, 0.0, 3.0, 4.0});  // r = 5
    ws.load_all();
    pc.start_block();
    pc.accumulate(ws);
    pc.stop_block();

    w0.set_positions(std::vector<double>{0.0, 0.0, 2.0, 0.0});  // r = 2
    ws.load_all();
    pc.start_block();
    pc.accumulate(ws);
    pc.stop_block();

    RunResult r;
    pc.finalize(r);

    constexpr double pi   = 3.14159265358979323846;
    const double      a5  = pi * (36.0 - 25.0);
    CHECK(series(r, "gofr_A_B")[5] == Catch::Approx(0.5 / a5));
    CHECK(series(r, "gofr_A_B_error")[5] == Catch::Approx(0.5 / a5));
}

TEST_CASE("PairCorrelation::collect merges block history across clones",
          "[pair_correlation]") {
    const auto groups = two_groups_AB();

    auto run_one_block = [&](const std::vector<double>& pos) {
        WalkerSet ws(1, 2, 2);
        Walker w(2, 2);
        w.set_positions(pos);
        ws.add_walker(w);
        ws.load_all();
        PairCorrelation pc(std::span<const ParticleGroup>(groups), 2, 10.0, 10);
        pc.start_block();
        pc.accumulate(ws);
        pc.stop_block();
        return pc;
    };

    PairCorrelation a = run_one_block({0.0, 0.0, 3.0, 4.0});  // bin 5
    PairCorrelation b = run_one_block({0.0, 0.0, 2.0, 0.0});  // bin 2
    a.collect(b);  // a now holds both blocks

    RunResult r;
    a.finalize(r);

    // Each bin seen in one of two blocks → mean 0.5 there.
    constexpr double pi = 3.14159265358979323846;
    CHECK(series(r, "gofr_A_B")[5] == Catch::Approx(0.5 / (pi * (36.0 - 25.0))));
    CHECK(series(r, "gofr_A_B")[2] == Catch::Approx(0.5 / (pi * (9.0 - 4.0))));
}
