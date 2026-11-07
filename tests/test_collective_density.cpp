// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "estimators/collective_density.h"
#include "estimators/density.h"
#include "population/particle_group.h"
#include "population/walker.h"
#include "population/walker_set.h"
#include "qmc/run_result.h"

using alea::CollectiveDensity;
using alea::CollectiveTerm;
using alea::Density;
using alea::ParticleGroup;
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

double scalar(const RunResult& r, const std::string& name) {
    for (const auto& [n, v] : r.scalars)
        if (n == name) return v;
    FAIL("scalar not found: " << name);
    return 0.0;
}

bool has_series(const RunResult& r, const std::string& name) {
    for (const auto& [n, v] : r.series)
        if (n == name) return true;
    return false;
}

// v = 1·r_A + 2·r_B over the two_groups_AB layout.
std::vector<CollectiveTerm> terms_A_plus_2B() {
    return {{"A", 1.0}, {"B", 2.0}};
}

// s = r_A + r_B (the dipole/sum coordinate of the physics motivation).
std::vector<CollectiveTerm> terms_sum_AB() {
    return {{"A", 1.0}, {"B", 1.0}};
}

}  // namespace

TEST_CASE("CollectiveDensity bins the linear combination into its row-major "
          "cell", "[collective_density]") {
    // rmax=2, n_bins=4 → Δ=1, V_cell=1, grid ∈ [−2, 2).
    // r_A=(0.5, 1.0), r_B=(0.25, 0.25) with coeffs (+1, +2)
    // → v = (1.0, 1.5): ix = ⌊1+2⌋ = 3, iy = ⌊1.5+2⌋ = 3 → cell = 3·4+3 = 15.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2);
    w0.set_positions(std::vector<double>{0.5, 1.0, 0.25, 0.25});
    ws.add_walker(w0);
    ws.load_all();

    const auto groups = two_groups_AB();
    const auto terms  = terms_A_plus_2B();
    CollectiveDensity cd(std::span<const ParticleGroup>(groups), /*dim=*/2,
                         /*rmax=*/2.0, /*n_bins=*/4, "v",
                         std::span<const CollectiveTerm>(terms));
    CHECK(cd.n_cells() == 16);
    CHECK(cd.cell_volume() == Catch::Approx(1.0));

    cd.start_block();
    cd.accumulate(ws);
    cd.stop_block();

    RunResult r;
    cd.finalize(r);

    const std::vector<double>& rho = series(r, "cdensity_v");
    REQUIRE(rho.size() == 16);
    CHECK(rho[15] == Catch::Approx(1.0));  // 1 count / (W=1 · V=1)
    for (std::size_t c = 0; c < rho.size(); ++c)
        if (c != 15) CHECK(rho[c] == Catch::Approx(0.0));

    // Namespaced grid metadata.
    CHECK(scalar(r, "cdensity_v_dim") == Catch::Approx(2.0));
    CHECK(scalar(r, "cdensity_v_n_bins") == Catch::Approx(4.0));
    CHECK(scalar(r, "cdensity_v_rmax") == Catch::Approx(2.0));
    CHECK(series(r, "cdensity_v_edges").size() == 5);
}

TEST_CASE("CollectiveDensity integrates to one (one coordinate per walker)",
          "[collective_density]") {
    WalkerSet ws(/*nw=*/3, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2), w1(2, 2), w2(2, 2);
    w0.set_positions(std::vector<double>{0.5, 0.5, 0.25, -0.5});
    w1.set_positions(std::vector<double>{-1.0, 0.5, 0.5, 0.75});
    w2.set_positions(std::vector<double>{0.0, 0.0, -0.75, 0.25});
    ws.add_walker(w0);
    ws.add_walker(w1);
    ws.add_walker(w2);
    ws.load_all();

    const auto groups = two_groups_AB();
    const auto terms  = terms_sum_AB();
    CollectiveDensity cd(std::span<const ParticleGroup>(groups), 2, 2.0, 4,
                         "s", std::span<const CollectiveTerm>(terms));
    cd.start_block();
    cd.accumulate(ws);
    cd.stop_block();

    RunResult r;
    cd.finalize(r);

    double integral = 0.0;
    for (double v : series(r, "cdensity_s")) integral += v * cd.cell_volume();
    CHECK(integral == Catch::Approx(1.0));
}

TEST_CASE("Anticorrelated pair: cdensity_s is a delta while density_rho1 "
          "spreads", "[collective_density]") {
    // The physics motivator (cross-paired channel): ρ2 = −ρ1 exactly, so the
    // sum s = ρ1+ρ2 lands at the origin for EVERY walker while ρ1 itself
    // occupies different cells — the marginal cannot see the correlation.
    WalkerSet ws(/*nw=*/2, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2), w1(2, 2);
    w0.set_positions(std::vector<double>{1.0, 1.0, -1.0, -1.0});
    w1.set_positions(std::vector<double>{-0.7, 0.3, 0.7, -0.3});
    ws.add_walker(w0);
    ws.add_walker(w1);
    ws.load_all();

    std::vector<ParticleGroup> groups = {
        ParticleGroup{"rho1", {}, {}, 0.0, 0, 1, 1},
        ParticleGroup{"rho2", {}, {}, 0.0, 1, 2, 1}};
    std::vector<CollectiveTerm> terms = {{"rho1", 1.0}, {"rho2", 1.0}};

    CollectiveDensity cd(std::span<const ParticleGroup>(groups), 2, 2.0, 4,
                         "s", std::span<const CollectiveTerm>(terms));
    Density den(std::span<const ParticleGroup>(groups), 2, 2.0, 4);

    cd.start_block();
    den.start_block();
    cd.accumulate(ws);
    den.accumulate(ws);
    cd.stop_block();
    den.stop_block();

    // Both estimators finalize into ONE RunResult — the key families must
    // coexist (this is what one driver's results .h5 will hold).
    RunResult r;
    cd.finalize(r);
    den.finalize(r);
    CHECK(has_series(r, "cdensity_s"));
    CHECK(has_series(r, "density_rho1"));

    // s = (0,0) for both walkers → single cell ⌊0+2⌋·4 + ⌊0+2⌋ = 10.
    const std::vector<double>& s = series(r, "cdensity_s");
    CHECK(s[10] == Catch::Approx(1.0));
    for (std::size_t c = 0; c < s.size(); ++c)
        if (c != 10) CHECK(s[c] == Catch::Approx(0.0));

    // ρ1 spreads: (1,1) → cell 15, (−0.7,0.3) → cell 1·4+2 = 6.
    const std::vector<double>& rho1 = series(r, "density_rho1");
    CHECK(rho1[15] == Catch::Approx(0.5));
    CHECK(rho1[6] == Catch::Approx(0.5));
}

TEST_CASE("CollectiveDensity drops out-of-grid v but keeps the config weight",
          "[collective_density]") {
    // Walker 0's v is outside the grid; walker 1's inside.  Both configs enter
    // W_block, so the in-box cell is normalized by the full weight sum.
    WalkerSet ws(/*nw=*/2, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2), w1(2, 2);
    w0.set_positions(std::vector<double>{2.0, 0.0, 1.5, 0.0});  // v = (3.5, 0)
    w1.set_positions(std::vector<double>{0.25, 0.25, 0.25, 0.25});  // v = (0.5, 0.5)
    ws.add_walker(w0);
    ws.add_walker(w1);
    ws.load_all();

    const auto groups = two_groups_AB();
    const auto terms  = terms_sum_AB();
    CollectiveDensity cd(std::span<const ParticleGroup>(groups), 2, 2.0, 4,
                         "s", std::span<const CollectiveTerm>(terms));
    cd.start_block();
    cd.accumulate(ws);
    cd.stop_block();

    RunResult r;
    cd.finalize(r);

    // (0.5, 0.5) → cell 10; 1 count / (W=2 · V=1) = 0.5.
    const std::vector<double>& s = series(r, "cdensity_s");
    CHECK(s[10] == Catch::Approx(0.5));
    double integral = 0.0;
    for (double v : s) integral += v * cd.cell_volume();
    CHECK(integral == Catch::Approx(0.5));  // half the weight left the grid
}

TEST_CASE("CollectiveDensity weights each walker by its DMC weight",
          "[collective_density]") {
    // v = r_A (single identity term).  Walker 0 at cell 10 with weight 3;
    // walker 1 at cell 0 with weight 1 → fractions 0.75 / 0.25.
    WalkerSet ws(/*nw=*/2, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2), w1(2, 2);
    w0.set_positions(std::vector<double>{0.5, 0.5, 0.0, 0.0});
    w1.set_positions(std::vector<double>{-1.5, -1.5, 0.0, 0.0});
    w0.set_weight(3.0);
    w1.set_weight(1.0);
    ws.add_walker(w0);
    ws.add_walker(w1);
    ws.load_all();  // pulls per-walker weights into the SoA

    const auto groups = two_groups_AB();
    std::vector<CollectiveTerm> terms = {{"A", 1.0}};
    CollectiveDensity cd(std::span<const ParticleGroup>(groups), 2, 2.0, 4,
                         "vA", std::span<const CollectiveTerm>(terms));
    cd.start_block();
    cd.accumulate(ws);
    cd.stop_block();

    RunResult r;
    cd.finalize(r);

    const std::vector<double>& rho = series(r, "cdensity_vA");
    CHECK(rho[10] == Catch::Approx(0.75));
    CHECK(rho[0] == Catch::Approx(0.25));
}

TEST_CASE("CollectiveDensity grid is half-open: +rmax drops, -rmax lands in "
          "bin 0", "[collective_density]") {
    WalkerSet ws(/*nw=*/2, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2), w1(2, 2);
    w0.set_positions(std::vector<double>{2.0, 0.0, 0.0, 0.0});   // v_x = +rmax
    w1.set_positions(std::vector<double>{-2.0, 0.0, 0.0, 0.0});  // v_x = −rmax
    ws.add_walker(w0);
    ws.add_walker(w1);
    ws.load_all();

    const auto groups = two_groups_AB();
    std::vector<CollectiveTerm> terms = {{"A", 1.0}};
    CollectiveDensity cd(std::span<const ParticleGroup>(groups), 2, 2.0, 4,
                         "vA", std::span<const CollectiveTerm>(terms));
    cd.start_block();
    cd.accumulate(ws);
    cd.stop_block();

    RunResult r;
    cd.finalize(r);

    // Only walker 1 lands: (−2, 0) → ix = 0, iy = 2 → cell 2; W_block = 2.
    const std::vector<double>& rho = series(r, "cdensity_vA");
    CHECK(rho[2] == Catch::Approx(0.5));
    double integral = 0.0;
    for (double v : rho) integral += v;
    CHECK(integral == Catch::Approx(0.5));
}

TEST_CASE("CollectiveDensity per-cell error is the SEM over blocks",
          "[collective_density]") {
    // Cell 10 sees block values {1, 0} → mean 0.5, SEM = sqrt(0.5/(1·2)) = 0.5.
    WalkerSet in_box(/*nw=*/1, /*np=*/2, /*dim=*/2);
    Walker wa(2, 2);
    wa.set_positions(std::vector<double>{0.25, 0.25, 0.25, 0.25});  // v → cell 10
    in_box.add_walker(wa);
    in_box.load_all();

    WalkerSet elsewhere(/*nw=*/1, /*np=*/2, /*dim=*/2);
    Walker wb(2, 2);
    wb.set_positions(std::vector<double>{-1.0, -1.0, -0.5, -0.5});  // v = (−1.5,−1.5) → cell 0
    elsewhere.add_walker(wb);
    elsewhere.load_all();

    const auto groups = two_groups_AB();
    const auto terms  = terms_sum_AB();
    CollectiveDensity cd(std::span<const ParticleGroup>(groups), 2, 2.0, 4,
                         "s", std::span<const CollectiveTerm>(terms));

    cd.start_block();
    cd.accumulate(in_box);
    cd.stop_block();

    RunResult r1;
    cd.finalize(r1);
    // Single block → no error estimate.
    CHECK(series(r1, "cdensity_s_error")[10] == Catch::Approx(0.0));

    cd.start_block();
    cd.accumulate(elsewhere);
    cd.stop_block();

    RunResult r2;
    cd.finalize(r2);
    CHECK(scalar(r2, "cdensity_s_n_blocks") == Catch::Approx(2.0));
    CHECK(series(r2, "cdensity_s")[10] == Catch::Approx(0.5));
    CHECK(series(r2, "cdensity_s_error")[10] == Catch::Approx(0.5));
}

TEST_CASE("CollectiveDensity collect merges clones like concatenated blocks",
          "[collective_density]") {
    WalkerSet in_box(/*nw=*/1, /*np=*/2, /*dim=*/2);
    Walker wa(2, 2);
    wa.set_positions(std::vector<double>{0.25, 0.25, 0.25, 0.25});
    in_box.add_walker(wa);
    in_box.load_all();

    WalkerSet elsewhere(/*nw=*/1, /*np=*/2, /*dim=*/2);
    Walker wb(2, 2);
    wb.set_positions(std::vector<double>{-1.0, -1.0, -0.5, -0.5});
    elsewhere.add_walker(wb);
    elsewhere.load_all();

    const auto groups = two_groups_AB();
    const auto terms  = terms_sum_AB();
    auto make = [&] {
        return CollectiveDensity(std::span<const ParticleGroup>(groups), 2,
                                 2.0, 4, "s",
                                 std::span<const CollectiveTerm>(terms));
    };

    // Thread A: blocks {in_box, elsewhere}; thread B: block {in_box}.
    CollectiveDensity a = make();
    a.start_block(); a.accumulate(in_box);    a.stop_block();
    a.start_block(); a.accumulate(elsewhere); a.stop_block();
    CollectiveDensity b = make();
    b.start_block(); b.accumulate(in_box);    b.stop_block();

    // Reference: one estimator seeing all three blocks sequentially.
    CollectiveDensity ref = make();
    ref.start_block(); ref.accumulate(in_box);    ref.stop_block();
    ref.start_block(); ref.accumulate(elsewhere); ref.stop_block();
    ref.start_block(); ref.accumulate(in_box);    ref.stop_block();

    a.collect(b);
    RunResult rm, rr;
    a.finalize(rm);
    ref.finalize(rr);

    const std::vector<double>& merged = series(rm, "cdensity_s");
    const std::vector<double>& expect = series(rr, "cdensity_s");
    const std::vector<double>& em     = series(rm, "cdensity_s_error");
    const std::vector<double>& ee     = series(rr, "cdensity_s_error");
    REQUIRE(merged.size() == expect.size());
    for (std::size_t c = 0; c < merged.size(); ++c) {
        CHECK(merged[c] == Catch::Approx(expect[c]).margin(1e-12));
        CHECK(em[c] == Catch::Approx(ee[c]).margin(1e-12));
    }

    // Empty-side handling in both directions.
    CollectiveDensity empty1 = make();
    empty1.collect(ref);  // adopt
    RunResult ra;
    empty1.finalize(ra);
    CHECK(series(ra, "cdensity_s")[10] ==
          Catch::Approx(series(rr, "cdensity_s")[10]));

    CollectiveDensity full = make();
    full.start_block(); full.accumulate(in_box); full.stop_block();
    CollectiveDensity empty2 = make();
    full.collect(empty2);  // no-op
    CHECK(full.n_blocks() == 1);
}

TEST_CASE("Two CollectiveDensity instances namespace their output",
          "[collective_density]") {
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2);
    w0.set_positions(std::vector<double>{0.5, 0.5, 0.25, -0.25});
    ws.add_walker(w0);
    ws.load_all();

    const auto groups = two_groups_AB();
    std::vector<CollectiveTerm> sum = {{"A", 1.0}, {"B", 1.0}};
    std::vector<CollectiveTerm> dif = {{"A", 1.0}, {"B", -1.0}};
    CollectiveDensity cs(std::span<const ParticleGroup>(groups), 2, 2.0, 4,
                         "s", std::span<const CollectiveTerm>(sum));
    CollectiveDensity cdif(std::span<const ParticleGroup>(groups), 2, 2.0, 4,
                           "dif", std::span<const CollectiveTerm>(dif));
    for (CollectiveDensity* e : {&cs, &cdif}) {
        e->start_block();
        e->accumulate(ws);
        e->stop_block();
    }

    RunResult r;
    cs.finalize(r);
    cdif.finalize(r);

    CHECK(has_series(r, "cdensity_s"));
    CHECK(has_series(r, "cdensity_s_error"));
    CHECK(has_series(r, "cdensity_s_edges"));
    CHECK(has_series(r, "cdensity_dif"));
    CHECK(has_series(r, "cdensity_dif_error"));
    CHECK(has_series(r, "cdensity_dif_edges"));
    // s = (0.75, 0.25) → cell 10; dif = (0.25, 0.75) → cell 10 as well, but
    // the series are independent objects.
    CHECK(series(r, "cdensity_s")[10] == Catch::Approx(1.0));
    CHECK(series(r, "cdensity_dif")[10] == Catch::Approx(1.0));
}

TEST_CASE("CollectiveDensity constructor validates its spec",
          "[collective_density]") {
    const auto groups = two_groups_AB();
    const auto terms  = terms_sum_AB();
    const auto gspan  = std::span<const ParticleGroup>(groups);
    const auto tspan  = std::span<const CollectiveTerm>(terms);

    // Empty name (it namespaces the output keys).
    CHECK_THROWS_AS(CollectiveDensity(gspan, 2, 2.0, 4, "", tspan),
                    std::runtime_error);
    // Empty terms.
    CHECK_THROWS_AS(CollectiveDensity(gspan, 2, 2.0, 4, "s", {}),
                    std::runtime_error);
    // Unknown group name.
    std::vector<CollectiveTerm> bad = {{"nope", 1.0}};
    CHECK_THROWS_AS(CollectiveDensity(gspan, 2, 2.0, 4, "s",
                                      std::span<const CollectiveTerm>(bad)),
                    std::runtime_error);
    // Non-positive grid parameters.
    CHECK_THROWS_AS(CollectiveDensity(gspan, 2, -1.0, 4, "s", tspan),
                    std::runtime_error);
    CHECK_THROWS_AS(CollectiveDensity(gspan, 2, 2.0, 0, "s", tspan),
                    std::runtime_error);
    // 2^24 cell cap: 4097² > 2^24.
    CHECK_THROWS_AS(CollectiveDensity(gspan, 2, 2.0, 4097, "s", tspan),
                    std::runtime_error);
}
