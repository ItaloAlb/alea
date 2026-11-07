// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <array>
#include <vector>

#include "population/batched_distance_table.h"

using alea::BatchedDistanceTable;

TEST_CASE("BatchedDistanceTable reports the sizes it was constructed with",
          "[distance_table]") {
    BatchedDistanceTable table(/*nw=*/4, /*np=*/3, /*dim=*/2);

    CHECK(table.nw() == 4);
    CHECK(table.np() == 3);
    CHECK(table.dim() == 2);
}

TEST_CASE("row(i,j) exposes the contiguous walker distances for a pair",
          "[distance_table]") {
    // Two walkers: r_01 = 5 (walker 0) and 10 (walker 1).
    BatchedDistanceTable table(/*nw=*/2, /*np=*/2, /*dim=*/2);
    std::vector<double> positions = {
        0.0, 0.0,   // p0 x : w0, w1
        0.0, 0.0,   // p0 y
        3.0, 6.0,   // p1 x
        4.0, 8.0,   // p1 y
    };
    table.build_all(positions);

    auto r01 = table.row(0, 1);  // length nw, contiguous over walkers
    REQUIRE(r01.size() == 2);
    CHECK(r01[0] == Catch::Approx(5.0));
    CHECK(r01[1] == Catch::Approx(10.0));
    // Matches the element accessor.
    CHECK(r01[0] == Catch::Approx(table(0, 0, 1)));
    CHECK(r01[1] == Catch::Approx(table(1, 0, 1)));
}

TEST_CASE("build_all computes pairwise distances with a zero diagonal",
          "[distance_table]") {
    // One walker, two particles in 2D: (0,0) and (3,4) → distance 5.
    BatchedDistanceTable table(/*nw=*/1, /*np=*/2, /*dim=*/2);

    // SoA layout: positions[p * dim * nw + d * nw + w].
    std::vector<double> positions = {0.0, 0.0,    // particle 0: x, y
                                     3.0, 4.0};   // particle 1: x, y
    table.build_all(positions);

    CHECK(table(0, 0, 1) == Catch::Approx(5.0));
    CHECK(table(0, 1, 0) == Catch::Approx(5.0));
    CHECK(table(0, 0, 0) == Catch::Approx(0.0));
    CHECK(table(0, 1, 1) == Catch::Approx(0.0));
}

TEST_CASE("build_all keeps walkers independent (walker index varies fastest)",
          "[distance_table]") {
    // Two walkers, two particles, 2D. Particle 0 sits at the origin in both
    // walkers; particle 1 differs per walker so each walker has its own r_01.
    //   walker 0: p1 = (3, 4) → r = 5
    //   walker 1: p1 = (6, 8) → r = 10
    BatchedDistanceTable table(/*nw=*/2, /*np=*/2, /*dim=*/2);

    // SoA: positions[p * dim * nw + d * nw + w], nw = 2.
    std::vector<double> positions = {
        0.0, 0.0,   // p0, x : w0, w1
        0.0, 0.0,   // p0, y : w0, w1
        3.0, 6.0,   // p1, x : w0, w1
        4.0, 8.0,   // p1, y : w0, w1
    };
    table.build_all(positions);

    CHECK(table(0, 0, 1) == Catch::Approx(5.0));
    CHECK(table(1, 0, 1) == Catch::Approx(10.0));
    // Symmetry holds independently in each walker.
    CHECK(table(1, 1, 0) == Catch::Approx(10.0));
}

TEST_CASE("update_batch computes the proposed distance row for one particle",
          "[distance_table]") {
    // One walker, two particles. Current: p0 = (0,0), p1 = (3,4).
    // Propose moving p0 to (3,0): new distance to p1 = (3,4) is 4.
    BatchedDistanceTable table(/*nw=*/1, /*np=*/2, /*dim=*/2);

    std::vector<double> positions = {0.0, 0.0,    // p0
                                     3.0, 4.0};   // p1
    std::vector<double> new_pos = {3.0, 0.0};     // dim * nw = 2 * 1
    std::vector<double> out(2 * 1);               // np * nw

    table.update_batch(/*p=*/0, positions, new_pos, out);

    // out[k * nw + w]
    CHECK(out[0] == Catch::Approx(0.0));   // self-distance r(0,0)
    CHECK(out[1] == Catch::Approx(4.0));   // r(0,1) at proposed position
}

TEST_CASE("commit_batch writes accepted rows in both directions",
          "[distance_table]") {
    // Current: p0 = (0,0), p1 = (3,4) → r_01 = 5.
    // Accepted move of p0 to (3,0) → r_01 = 4 must appear as both (0,1) and (1,0).
    BatchedDistanceTable table(/*nw=*/1, /*np=*/2, /*dim=*/2);
    std::vector<double> positions = {0.0, 0.0, 3.0, 4.0};
    table.build_all(positions);

    std::vector<double> new_row = {0.0, 4.0};   // np * nw, from update_batch(p=0)
    std::array<bool, 1> accept = {true};
    table.commit_batch(/*p=*/0, new_row, accept);

    CHECK(table(0, 0, 1) == Catch::Approx(4.0));
    CHECK(table(0, 1, 0) == Catch::Approx(4.0));
    CHECK(table(0, 1, 1) == Catch::Approx(0.0));   // untouched diagonal
}

TEST_CASE("commit_batch updates accepted walkers and leaves rejected ones",
          "[distance_table]") {
    // Two walkers, identical start: p0 = (0,0), p1 = (3,4) → r_01 = 5.
    // Propose moving p0 to (3,0) (r_01 = 4) in both, but accept only walker 0.
    BatchedDistanceTable table(/*nw=*/2, /*np=*/2, /*dim=*/2);
    std::vector<double> positions = {0.0, 0.0,   // p0 x : w0, w1
                                     0.0, 0.0,   // p0 y
                                     3.0, 3.0,   // p1 x
                                     4.0, 4.0};  // p1 y
    table.build_all(positions);

    std::vector<double> new_row = {0.0, 0.0,    // k=0 self
                                   4.0, 4.0};   // k=1, both walkers
    std::array<bool, 2> accept = {true, false};
    table.commit_batch(/*p=*/0, new_row, accept);

    CHECK(table(0, 0, 1) == Catch::Approx(4.0));   // walker 0 accepted
    CHECK(table(0, 1, 0) == Catch::Approx(4.0));
    CHECK(table(1, 0, 1) == Catch::Approx(5.0));   // walker 1 rejected → old r
    CHECK(table(1, 1, 0) == Catch::Approx(5.0));
}
