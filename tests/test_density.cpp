// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "estimators/density.h"
#include "population/particle_group.h"
#include "population/walker.h"
#include "population/walker_set.h"
#include "qmc/run_result.h"

using alea::Density;
using alea::ParticleGroup;
using alea::RunResult;
using alea::Walker;
using alea::WalkerSet;

namespace {

// One single-particle group "A" (particle 0).
std::vector<ParticleGroup> one_group_A() {
    return {ParticleGroup{"A", {}, {}, 0.0, /*first=*/0, /*last=*/1, /*count=*/1}};
}

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

double scalar(const RunResult& r, const std::string& name) {
    for (const auto& [n, v] : r.scalars)
        if (n == name) return v;
    FAIL("scalar not found: " << name);
    return 0.0;
}

}  // namespace

TEST_CASE("Density constant nonbinary normalization retains a finite zero error",
          "[density][roundoff]") {
    // Three particles in the sole periodic bin, V=10: every block is ρ=3/10,
    // so its analytic variance is zero. With FMA, the old first Welford
    // increment can be negative because 3*(1/10) is not exactly representable.
    const alea::SimulationCell cell(2, std::vector<double>{5, 0, 0, 2});
    WalkerSet ws(1, 3, 2, cell);
    Walker walker(3, 2);
    walker.set_positions(std::vector<double>{0.1, 0.1, 0.2, 0.2, 0.3, 0.3});
    ws.add_walker(walker);
    ws.load_all();
    const std::vector<ParticleGroup> groups{
        ParticleGroup{"A", {}, {}, 0.0, 0, 3, 3}};
    Density density(groups, 2, 1.0, 1, false, cell);
    for (int b = 0; b < 8; ++b) {
        density.start_block();
        density.accumulate(ws);
        density.stop_block();
        RunResult result;
        density.finalize(result);
        const double rho = series(result, "density_A")[0];
        const double error = series(result, "density_A_error")[0];
        CHECK(rho == Catch::Approx(0.3).epsilon(2e-15));
        REQUIRE(std::isfinite(error));
        CHECK(error >= 0.0);
        CHECK(error <= 2 * std::numeric_limits<double>::epsilon() * rho);
    }
}

TEST_CASE("Density streaming and merged errors agree with independent two-pass statistics",
          "[density][roundoff]") {
    // A sole open bin of area 10 measures occupancy/10. Histories are the
    // small integers below, so mean and SEM have a direct two-pass oracle.
    const std::vector<ParticleGroup> groups{
        ParticleGroup{"A", {}, {}, 0.0, 0, 3, 3}};
    const double rmax = std::sqrt(10.0) / 2;
    Density all(groups, 2, rmax, 1), left(groups, 2, rmax, 1), right(groups, 2, rmax, 1);
    WalkerSet ws(1, 3, 2);
    const std::array<int, 6> counts{0, 3, 1, 2, 3, 1}; // [block]
    long double mean = 0.0L;
    for (int count : counts) mean += static_cast<long double>(count) / 10.0L;
    mean /= counts.size();
    long double m2 = 0.0L;
    for (int count : counts) {
        const long double delta = static_cast<long double>(count) / 10.0L - mean;
        m2 += delta * delta;
    }
    const double expected_error = static_cast<double>(
        std::sqrt(m2 / (counts.size() * (counts.size() - 1))));
    for (std::size_t b = 0; b < counts.size(); ++b) {
        for (int p = 0; p < 3; ++p) {
            ws.positions()[2 * p] = p < counts[b] ? 0.0 : 5.0;
            ws.positions()[2 * p + 1] = 0.0;
        }
        Density& part = b < 3 ? left : right;
        all.start_block();
        part.start_block();
        all.accumulate(ws);
        part.accumulate(ws);
        all.stop_block();
        part.stop_block();
    }
    left.collect(right);
    for (const Density* density : {&all, &left}) {
        RunResult result;
        density->finalize(result);
        CHECK(series(result, "density_A")[0] == Catch::Approx(static_cast<double>(mean)).epsilon(2e-14));
        CHECK(series(result, "density_A_error")[0] == Catch::Approx(expected_error).epsilon(2e-14));
    }
}

TEST_CASE("Density variance stays finite when its unweighted squared delta would overflow",
          "[density][roundoff]") {
    // Two block densities 0 and h have mean h/2 and SEM h/2. Choose h so
    // h² overflows but the correct M2=h²/2 remains representable.
    constexpr double high_density = 1.6e154;
    const double rmax = std::sqrt(3.0 / high_density) / 2;
    const std::vector<ParticleGroup> groups{
        ParticleGroup{"A", {}, {}, 0.0, 0, 3, 3}};
    Density density(groups, 2, rmax, 1), left(groups, 2, rmax, 1), right(groups, 2, rmax, 1);
    WalkerSet ws(1, 3, 2);
    for (int b = 0; b < 2; ++b) {
        for (int p = 0; p < 3; ++p) ws.positions()[2 * p] = b == 0 ? 5.0 : 0.0;
        Density& part = b == 0 ? left : right;
        density.start_block();
        part.start_block();
        density.accumulate(ws);
        part.accumulate(ws);
        density.stop_block();
        part.stop_block();
    }
    left.collect(right);
    for (const Density* estimator : {&density, &left}) {
        RunResult result;
        estimator->finalize(result);
        const double error = series(result, "density_A_error")[0];
        REQUIRE(std::isfinite(error));
        CHECK(series(result, "density_A")[0] == Catch::Approx(high_density / 2).epsilon(2e-14));
        CHECK(error == Catch::Approx(high_density / 2).epsilon(2e-14));
    }
}

TEST_CASE("Density bins one particle into its row-major 2D cell", "[density]") {
    // rmax=2, n_bins=4 → Δ=1, V_cell=1, grid x,y ∈ [−2, 2).
    // Particle at (0.5, 1.5): ix = ⌊0.5+2⌋ = 2, iy = ⌊1.5+2⌋ = 3
    // → cell = ix·n_bins + iy = 11.
    WalkerSet ws(/*nw=*/1, /*np=*/1, /*dim=*/2);
    Walker w0(1, 2);
    w0.set_positions(std::vector<double>{0.5, 1.5});
    ws.add_walker(w0);
    ws.load_all();

    const auto groups = one_group_A();
    Density den(std::span<const ParticleGroup>(groups), /*dim=*/2,
                /*rmax=*/2.0, /*n_bins=*/4);
    CHECK(den.n_cells() == 16);
    CHECK(den.cell_volume() == Catch::Approx(1.0));

    den.start_block();
    den.accumulate(ws);
    den.stop_block();

    RunResult r;
    den.finalize(r);

    const std::vector<double>& rho = series(r, "density_A");
    REQUIRE(rho.size() == 16);
    CHECK(rho[11] == Catch::Approx(1.0));  // 1 count / (W=1 · V=1)
    for (std::size_t c = 0; c < rho.size(); ++c)
        if (c != 11) CHECK(rho[c] == Catch::Approx(0.0));
}

TEST_CASE("Density integrates to the group particle count", "[density]") {
    // One group of two particles, both inside the grid.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2);
    w0.set_positions(std::vector<double>{0.5, 0.5, -1.5, 1.5});
    ws.add_walker(w0);
    ws.load_all();

    std::vector<ParticleGroup> groups = {
        ParticleGroup{"e", {}, {}, 0.0, /*first=*/0, /*last=*/2, /*count=*/2}};
    Density den(std::span<const ParticleGroup>(groups), 2, 2.0, 4);
    den.start_block();
    den.accumulate(ws);
    den.stop_block();

    RunResult r;
    den.finalize(r);

    const std::vector<double>& rho = series(r, "density_e");
    double integral = 0.0;
    for (double v : rho) integral += v * den.cell_volume();
    CHECK(integral == Catch::Approx(2.0));
}

TEST_CASE("Density drops out-of-grid particles but keeps the config weight",
          "[density]") {
    // A's particle sits outside the grid; B's inside.  The config still has
    // weight 1 in W_block, so density_B is normalized by the full weight.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2);
    w0.set_positions(std::vector<double>{3.0, 0.0, 0.5, 0.5});
    ws.add_walker(w0);
    ws.load_all();

    const auto groups = two_groups_AB();
    Density den(std::span<const ParticleGroup>(groups), 2, 2.0, 4);
    den.start_block();
    den.accumulate(ws);
    den.stop_block();

    RunResult r;
    den.finalize(r);

    for (double v : series(r, "density_A")) CHECK(v == Catch::Approx(0.0));
    // (0.5, 0.5) → ix = 2, iy = 2 → cell 10; value 1/(1·1).
    CHECK(series(r, "density_B")[10] == Catch::Approx(1.0));
}

TEST_CASE("Density weights each walker by its DMC weight (mixed)", "[density]") {
    // Walker 0 at cell (2,2)=10 with weight 3; walker 1 at cell (0,0)=0 with
    // weight 1 → weight fractions 3/4 and 1/4 (V_cell = 1).
    WalkerSet ws(/*nw=*/2, /*np=*/1, /*dim=*/2);
    Walker w0(1, 2), w1(1, 2);
    w0.set_positions(std::vector<double>{0.5, 0.5});
    w1.set_positions(std::vector<double>{-1.5, -1.5});
    w0.set_weight(3.0);
    w1.set_weight(1.0);
    ws.add_walker(w0);
    ws.add_walker(w1);
    ws.load_all();  // pulls per-walker weights into the SoA

    const auto groups = one_group_A();
    Density den(std::span<const ParticleGroup>(groups), 2, 2.0, 4);
    den.start_block();
    den.accumulate(ws);
    den.stop_block();

    RunResult r;
    den.finalize(r);

    const std::vector<double>& rho = series(r, "density_A");
    CHECK(rho[10] == Catch::Approx(0.75));
    CHECK(rho[0] == Catch::Approx(0.25));
}

TEST_CASE("Density clamps the +rmax edge and keeps -rmax in bin 0", "[density]") {
    // x = −rmax is inside (first bin); x = +rmax is outside (half-open grid).
    WalkerSet ws(/*nw=*/2, /*np=*/1, /*dim=*/2);
    Walker w0(1, 2), w1(1, 2);
    w0.set_positions(std::vector<double>{-2.0, -2.0});  // → cell (0,0) = 0
    w1.set_positions(std::vector<double>{2.0, 0.0});    // dropped
    ws.add_walker(w0);
    ws.add_walker(w1);
    ws.load_all();

    const auto groups = one_group_A();
    Density den(std::span<const ParticleGroup>(groups), 2, 2.0, 4);
    den.start_block();
    den.accumulate(ws);
    den.stop_block();

    RunResult r;
    den.finalize(r);

    const std::vector<double>& rho = series(r, "density_A");
    CHECK(rho[0] == Catch::Approx(0.5));  // 1/(W=2 · V=1)
    for (std::size_t c = 1; c < rho.size(); ++c)
        CHECK(rho[c] == Catch::Approx(0.0));
}

TEST_CASE("Density reports a per-cell error from block scatter", "[density]") {
    WalkerSet ws(/*nw=*/1, /*np=*/1, /*dim=*/2);
    Walker w0(1, 2);
    ws.add_walker(w0);

    const auto groups = one_group_A();
    Density den(std::span<const ParticleGroup>(groups), 2, 2.0, 4);

    // Block 1: particle in cell 10.  Block 2: particle in cell 0.
    // → cell 10 sees block values {1, 0}: mean 0.5, std error 0.5.
    w0.set_positions(std::vector<double>{0.5, 0.5});
    ws.load_all();
    den.start_block();
    den.accumulate(ws);
    den.stop_block();

    w0.set_positions(std::vector<double>{-1.5, -1.5});
    ws.load_all();
    den.start_block();
    den.accumulate(ws);
    den.stop_block();

    RunResult r;
    den.finalize(r);

    CHECK(series(r, "density_A")[10] == Catch::Approx(0.5));
    CHECK(series(r, "density_A_error")[10] == Catch::Approx(0.5));
    CHECK(series(r, "density_A")[0] == Catch::Approx(0.5));
    CHECK(series(r, "density_A_error")[0] == Catch::Approx(0.5));
}

TEST_CASE("Density::collect merges Welford stats across clones", "[density]") {
    const auto groups = one_group_A();

    auto run_one_block = [&](const std::vector<double>& pos) {
        WalkerSet ws(1, 1, 2);
        Walker w(1, 2);
        w.set_positions(pos);
        ws.add_walker(w);
        ws.load_all();
        Density den(std::span<const ParticleGroup>(groups), 2, 2.0, 4);
        den.start_block();
        den.accumulate(ws);
        den.stop_block();
        return den;
    };

    Density a = run_one_block({0.5, 0.5});    // cell 10
    Density b = run_one_block({-1.5, -1.5});  // cell 0
    a.collect(b);  // a now holds both blocks
    CHECK(a.n_blocks() == 2);

    RunResult r;
    a.finalize(r);

    // Identical to accumulating both blocks in one instance: mean 0.5,
    // error 0.5 at each cell (pins the Chan merge ≡ block concatenation).
    CHECK(series(r, "density_A")[10] == Catch::Approx(0.5));
    CHECK(series(r, "density_A_error")[10] == Catch::Approx(0.5));
    CHECK(series(r, "density_A")[0] == Catch::Approx(0.5));
    CHECK(series(r, "density_A_error")[0] == Catch::Approx(0.5));
}

TEST_CASE("Density::collect handles an empty side in either direction",
          "[density]") {
    const auto groups = one_group_A();

    auto run_one_block = [&]() {
        WalkerSet ws(1, 1, 2);
        Walker w(1, 2);
        w.set_positions(std::vector<double>{0.5, 0.5});  // cell 10
        ws.add_walker(w);
        ws.load_all();
        Density den(std::span<const ParticleGroup>(groups), 2, 2.0, 4);
        den.start_block();
        den.accumulate(ws);
        den.stop_block();
        return den;
    };

    // empty.collect(full) adopts full's stats.
    Density empty1(std::span<const ParticleGroup>(groups), 2, 2.0, 4);
    Density full1 = run_one_block();
    empty1.collect(full1);
    CHECK(empty1.n_blocks() == 1);
    RunResult r1;
    empty1.finalize(r1);
    CHECK(series(r1, "density_A")[10] == Catch::Approx(1.0));

    // full.collect(empty) is a no-op.
    Density full2 = run_one_block();
    Density empty2(std::span<const ParticleGroup>(groups), 2, 2.0, 4);
    full2.collect(empty2);
    CHECK(full2.n_blocks() == 1);
    RunResult r2;
    full2.finalize(r2);
    CHECK(series(r2, "density_A")[10] == Catch::Approx(1.0));
}

TEST_CASE("Density bins a 3D grid with the correct flat index and volume",
          "[density]") {
    // rmax=2, n_bins=4 → Δ=1, V_cell=1.  Particle at (0.5, 1.5, −0.5):
    // ix=2, iy=3, iz=1 → cell = (2·4 + 3)·4 + 1 = 45.
    WalkerSet ws(/*nw=*/1, /*np=*/1, /*dim=*/3);
    Walker w0(1, 3);
    w0.set_positions(std::vector<double>{0.5, 1.5, -0.5});
    ws.add_walker(w0);
    ws.load_all();

    const auto groups = one_group_A();
    Density den(std::span<const ParticleGroup>(groups), /*dim=*/3, 2.0, 4);
    CHECK(den.n_cells() == 64);

    den.start_block();
    den.accumulate(ws);
    den.stop_block();

    RunResult r;
    den.finalize(r);

    const std::vector<double>& rho = series(r, "density_A");
    REQUIRE(rho.size() == 64);
    CHECK(rho[45] == Catch::Approx(1.0));
    for (std::size_t c = 0; c < rho.size(); ++c)
        if (c != 45) CHECK(rho[c] == Catch::Approx(0.0));
}

TEST_CASE("Density emits grid edges and self-describing scalars", "[density]") {
    WalkerSet ws(1, 1, 2);
    Walker w0(1, 2);
    w0.set_positions(std::vector<double>{0.5, 0.5});
    ws.add_walker(w0);
    ws.load_all();

    const auto groups = one_group_A();
    Density den(std::span<const ParticleGroup>(groups), 2, 2.0, 4);
    den.start_block();
    den.accumulate(ws);
    den.stop_block();

    RunResult r;
    den.finalize(r);

    const std::vector<double>& edges = series(r, "density_edges");
    REQUIRE(edges.size() == 5);  // n_bins + 1
    CHECK(edges.front() == Catch::Approx(-2.0));
    CHECK(edges.back() == Catch::Approx(2.0));
    CHECK(edges[1] == Catch::Approx(-1.0));

    CHECK(scalar(r, "density_dim") == Catch::Approx(2.0));
    CHECK(scalar(r, "density_n_bins") == Catch::Approx(4.0));
    CHECK(scalar(r, "density_rmax") == Catch::Approx(2.0));
    CHECK(scalar(r, "density_n_blocks") == Catch::Approx(1.0));
}

TEST_CASE("Density error is zero for a single block", "[density]") {
    WalkerSet ws(1, 1, 2);
    Walker w0(1, 2);
    w0.set_positions(std::vector<double>{0.5, 0.5});
    ws.add_walker(w0);
    ws.load_all();

    const auto groups = one_group_A();
    Density den(std::span<const ParticleGroup>(groups), 2, 2.0, 4);
    den.start_block();
    den.accumulate(ws);
    den.stop_block();

    RunResult r;
    den.finalize(r);

    for (double v : series(r, "density_A_error")) CHECK(v == Catch::Approx(0.0));
}

TEST_CASE("Density per_particle resolves one histogram per particle",
          "[density]") {
    // One group "e" of two particles at different cells → per-particle series
    // e_p0 / e_p1, each integrating to 1, binned at its own cell.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2);
    w0.set_positions(std::vector<double>{0.5, 0.5, -1.5, -1.5});  // cells 10, 0
    ws.add_walker(w0);
    ws.load_all();

    std::vector<ParticleGroup> groups = {
        ParticleGroup{"e", {}, {}, 0.0, /*first=*/0, /*last=*/2, /*count=*/2}};
    Density den(std::span<const ParticleGroup>(groups), 2, 2.0, 4,
                /*per_particle=*/true);
    den.start_block();
    den.accumulate(ws);
    den.stop_block();

    RunResult r;
    den.finalize(r);

    const std::vector<double>& p0 = series(r, "density_e_p0");
    const std::vector<double>& p1 = series(r, "density_e_p1");
    CHECK(p0[10] == Catch::Approx(1.0));
    CHECK(p0[0] == Catch::Approx(0.0));
    CHECK(p1[0] == Catch::Approx(1.0));
    CHECK(p1[10] == Catch::Approx(0.0));
    double int0 = 0.0, int1 = 0.0;
    for (std::size_t c = 0; c < p0.size(); ++c) {
        int0 += p0[c] * den.cell_volume();
        int1 += p1[c] * den.cell_volume();
    }
    CHECK(int0 == Catch::Approx(1.0));
    CHECK(int1 == Catch::Approx(1.0));
}

TEST_CASE("Density rejects grids above the 2^24 cell cap", "[density]") {
    const auto groups = one_group_A();
    // 512³ = 2^27 > 2^24.
    CHECK_THROWS(Density(std::span<const ParticleGroup>(groups), /*dim=*/3,
                         /*rmax=*/1.0, /*n_bins=*/512));
}
