// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <array>
#include <algorithm>
#include <vector>

#include "numerics/xoshiro_random.h"
#include "population/walker.h"
#include "population/walker_set.h"

using alea::Walker;
using alea::WalkerSet;

TEST_CASE("reused population gathers carry every real and phase plane through changing strides",
          "[walker_set][branch]") {
    const std::array<double, 4> vectors{5, 0, 1.2, 4};
    const alea::SimulationCell cell(2, vectors);
    WalkerSet ws(3, 2, 2, cell);
    std::array<Walker, 3> walkers{Walker(2, 2), Walker(2, 2), Walker(2, 2)};
    for (std::size_t w = 0; w < 3; ++w) {
        const std::array<double, 4> pos{0.1*w, 0.2, 1.0+0.2*w, -0.3};
        walkers[w].set_positions(pos);
        ws.add_walker(walkers[w]);
    }
    ws.load_all();
    ws.enable_phase();
    auto planes = [&] {
        return std::array{ws.positions(), ws.gradients(), ws.laplacians(), ws.log_psi(),
            ws.local_energies(), ws.phase_gradients(), ws.phase_laplacians(),
            ws.local_energies_imag(), ws.rr_proposed(), ws.rr_accepted(),
            ws.old_energies(), ws.weights(), ws.multiplicities()};
    };
    auto scalars = [&] { return std::array{ws.ids(), ws.parent_ids(), ws.ages()}; };
    std::size_t tag = 1;
    auto initial = planes();
    // Preserve positions and their independently built distance table.
    for (std::size_t p = 1; p < initial.size(); ++p)
        for (double& value : initial[p]) value = static_cast<double>(tag++);
    for (auto plane : scalars()) for (auto& value : plane) value = tag++;
    for (const auto& src : {std::vector<std::size_t>{2, 0, 0, 1, 2},
                           std::vector<std::size_t>{4, 2},
                           std::vector<std::size_t>{1, 0, 1, 1, 0, 0},
                           std::vector<std::size_t>{0}}) {
        const auto old_nw = ws.num_walkers();
        std::vector<std::vector<double>> old;
        for (auto plane : planes()) old.emplace_back(plane.begin(), plane.end());
        std::vector<std::vector<std::size_t>> old_scalar;
        for (auto plane : scalars()) old_scalar.emplace_back(plane.begin(), plane.end());
        const auto table = ws.dist_tables().data();
        const std::vector<double> old_distances(table.begin(), table.end());
        ws.apply_permutation(src);
        const auto current = planes();
        for (std::size_t p = 0; p < current.size(); ++p)
            for (std::size_t row = 0; row < old[p].size()/old_nw; ++row)
                for (std::size_t w = 0; w < src.size(); ++w)
                    REQUIRE(current[p][row*src.size()+w] == old[p][row*old_nw+src[w]]);
        const auto current_scalar = scalars();
        for (std::size_t p = 0; p < current_scalar.size(); ++p)
            for (std::size_t w = 0; w < src.size(); ++w)
                REQUIRE(current_scalar[p][w] == old_scalar[p][src[w]]);
        for (std::size_t pair = 0; pair < 4; ++pair)
            for (std::size_t w = 0; w < src.size(); ++w)
                REQUIRE(ws.dist_tables().data()[pair*src.size()+w] ==
                        old_distances[pair*old_nw+src[w]]);
        for (bool accept : ws.accept_mask()) REQUIRE_FALSE(accept);
    }
}

TEST_CASE("WalkerSet reports the dimensions it was constructed with",
          "[walker_set]") {
    WalkerSet ws(/*nw=*/4, /*np=*/3, /*dim=*/2);

    CHECK(ws.num_walkers() == 4);
    CHECK(ws.n_particles() == 3);
    CHECK(ws.dim() == 2);
}

TEST_CASE("WalkerSet sizes its SoA working arrays", "[walker_set]") {
    const std::size_t nw = 4, np = 3, dim = 2;
    WalkerSet ws(nw, np, dim);

    CHECK(ws.positions().size() == np * dim * nw);
    CHECK(ws.gradients().size() == np * dim * nw);
    CHECK(ws.laplacians().size() == np * dim * nw);
    CHECK(ws.log_psi().size() == nw);
    CHECK(ws.local_energies().size() == nw);

    CHECK(ws.dist_tables().nw() == nw);
    CHECK(ws.dist_tables().np() == np);
    CHECK(ws.dist_tables().dim() == dim);
}

TEST_CASE("load_all gathers walker positions into SoA and builds distances",
          "[walker_set]") {
    WalkerSet ws(/*nw=*/2, /*np=*/2, /*dim=*/2);

    // Walker positions are AoS: [p*dim + d].
    Walker w0(2, 2), w1(2, 2);
    std::vector<double> p0 = {0.0, 0.0, 3.0, 4.0};  // p0=(0,0), p1=(3,4) → r=5
    std::vector<double> p1 = {0.0, 0.0, 6.0, 8.0};  // p0=(0,0), p1=(6,8) → r=10
    w0.set_positions(p0);
    w1.set_positions(p1);
    ws.add_walker(w0);
    ws.add_walker(w1);

    ws.load_all();

    // SoA: positions[p*dim*nw + d*nw + w]. Particle 1, both walkers:
    auto pos = ws.positions();
    CHECK(pos[4] == Catch::Approx(3.0));  // p1, x, w0
    CHECK(pos[5] == Catch::Approx(6.0));  // p1, x, w1
    CHECK(pos[6] == Catch::Approx(4.0));  // p1, y, w0
    CHECK(pos[7] == Catch::Approx(8.0));  // p1, y, w1

    CHECK(ws.dist_tables()(0, 0, 1) == Catch::Approx(5.0));
    CHECK(ws.dist_tables()(1, 0, 1) == Catch::Approx(10.0));
}

TEST_CASE("save_all scatters SoA positions and energies back to walkers",
          "[walker_set]") {
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2);
    ws.add_walker(w0);

    // Write directly into the SoA working state, then scatter back.
    auto pos = ws.positions();  // [p*dim*nw + d*nw + w], nw = 1
    pos[2] = 3.0;               // p1, x
    pos[3] = 4.0;               // p1, y
    ws.local_energies()[0] = -1.5;

    ws.save_all();

    auto wp = w0.positions();  // AoS [p*dim + d]
    CHECK(wp[2] == Catch::Approx(3.0));  // p1, x
    CHECK(wp[3] == Catch::Approx(4.0));  // p1, y
    CHECK(w0.local_energy() == Catch::Approx(-1.5));
}

TEST_CASE("propose_move_batch fills the pending distance row for one particle",
          "[walker_set]") {
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2);
    std::vector<double> p = {0.0, 0.0, 3.0, 4.0};  // p0=(0,0), p1=(3,4)
    w0.set_positions(p);
    ws.add_walker(w0);
    ws.load_all();

    // Propose moving particle 0 to (3,0): new distance to p1=(3,4) is 4.
    std::vector<double> new_pos = {3.0, 0.0};  // [dim*nw], nw = 1
    ws.propose_move_batch(/*p=*/0, new_pos);

    auto pend = ws.pending_dist_rows();  // [np*nw]
    CHECK(pend[0] == Catch::Approx(0.0));  // self r(0,0)
    CHECK(pend[1] == Catch::Approx(4.0));  // r(0,1) proposed

    // The proposed position is recorded in new_pos() for ratio_batch/apply_move.
    auto np = ws.new_pos();  // [dim*nw]
    CHECK(np[0] == Catch::Approx(3.0));
    CHECK(np[1] == Catch::Approx(0.0));
}

TEST_CASE("apply_move commits position, distance, grad and lap for accepted",
          "[walker_set]") {
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2);
    std::vector<double> p = {0.0, 0.0, 3.0, 4.0};  // p0=(0,0), p1=(3,4), r=5
    w0.set_positions(p);
    ws.add_walker(w0);
    ws.load_all();

    std::vector<double> new_pos = {3.0, 0.0};   // move p0 to (3,0) → r=4
    ws.propose_move_batch(/*p=*/0, new_pos);

    std::vector<double> new_grad = {0.5, 0.0};  // [dim*nw]
    std::vector<double> new_lap = {-0.25, 0.0};   // [dim*nw]: x=-0.25, y=0
    std::array<bool, 1> accept = {true};
    ws.apply_move(/*p=*/0, new_pos, new_grad, new_lap, accept);

    auto pos = ws.positions();  // [p*dim*nw + d*nw + w], nw = 1
    CHECK(pos[0] == Catch::Approx(3.0));  // p0, x
    CHECK(pos[1] == Catch::Approx(0.0));  // p0, y
    CHECK(ws.dist_tables()(0, 0, 1) == Catch::Approx(4.0));  // committed
    CHECK(ws.gradients()[0] == Catch::Approx(0.5));   // p0, x grad
    CHECK(ws.gradients()[1] == Catch::Approx(0.0));   // p0, y grad
    CHECK(ws.laplacians()[0] == Catch::Approx(-0.25));  // p0, x lap
    CHECK(ws.laplacians()[1] == Catch::Approx(0.0));    // p0, y lap
}

TEST_CASE("apply_permutation reorders and duplicates walker columns",
          "[walker_set][branch]") {
    // 3 walkers, np=2, dim=2. Distinct positions so r(0,1) = 1, 2, 3.
    WalkerSet ws(/*nw=*/3, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2), w1(2, 2), w2(2, 2);
    w0.set_positions(std::vector<double>{0.0, 0.0, 1.0, 0.0});  // r=1
    w1.set_positions(std::vector<double>{0.0, 0.0, 2.0, 0.0});  // r=2
    w2.set_positions(std::vector<double>{0.0, 0.0, 3.0, 0.0});  // r=3
    ws.add_walker(w0);
    ws.add_walker(w1);
    ws.add_walker(w2);
    ws.load_all();

    // Tag per-walker scalar/vector state so we can follow each column.
    for (std::size_t w = 0; w < 3; ++w) {
        ws.log_psi()[w]        = 10.0 + static_cast<double>(w);
        ws.local_energies()[w] = -(static_cast<double>(w) + 1.0);
        ws.old_energies()[w]   = 100.0 + static_cast<double>(w);
        // gradient of particle 0, x-component for walker w = w.
        ws.gradients()[0 * 2 * 3 + 0 * 3 + w] = static_cast<double>(w);
    }

    // new col0 ← old 2, col1 ← old 0, col2 ← old 0 (duplicate).
    std::vector<std::size_t> src = {2, 0, 0};
    ws.apply_permutation(src);

    REQUIRE(ws.num_walkers() == 3);

    // Distance r(0,1): {3, 1, 1}.
    CHECK(ws.dist_tables()(0, 0, 1) == Catch::Approx(3.0));
    CHECK(ws.dist_tables()(1, 0, 1) == Catch::Approx(1.0));
    CHECK(ws.dist_tables()(2, 0, 1) == Catch::Approx(1.0));

    // positions of particle 1, x: {3, 1, 1}. Layout [p*dim*nw + d*nw + w].
    auto pos = ws.positions();
    const std::size_t nw = 3;
    CHECK(pos[1 * 2 * nw + 0 * nw + 0] == Catch::Approx(3.0));
    CHECK(pos[1 * 2 * nw + 0 * nw + 1] == Catch::Approx(1.0));
    CHECK(pos[1 * 2 * nw + 0 * nw + 2] == Catch::Approx(1.0));

    // Per-walker scalars follow the columns.
    CHECK(ws.log_psi()[0]        == Catch::Approx(12.0));
    CHECK(ws.log_psi()[1]        == Catch::Approx(10.0));
    CHECK(ws.log_psi()[2]        == Catch::Approx(10.0));
    CHECK(ws.local_energies()[0] == Catch::Approx(-3.0));
    CHECK(ws.local_energies()[1] == Catch::Approx(-1.0));
    CHECK(ws.old_energies()[0]   == Catch::Approx(102.0));

    // gradient (p0,x) follows: {2, 0, 0}.
    CHECK(ws.gradients()[0] == Catch::Approx(2.0));
    CHECK(ws.gradients()[1] == Catch::Approx(0.0));
    CHECK(ws.gradients()[2] == Catch::Approx(0.0));
}

TEST_CASE("apply_permutation carries per-walker scalar columns",
          "[walker_set][branch]") {
    WalkerSet ws(/*nw=*/3, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2), w1(2, 2), w2(2, 2);
    ws.add_walker(w0);
    ws.add_walker(w1);
    ws.add_walker(w2);
    ws.load_all();

    ws.weights()[0] = 10.0; ws.weights()[1] = 20.0; ws.weights()[2] = 30.0;
    ws.multiplicities()[0] = 1.0; ws.multiplicities()[1] = 2.0; ws.multiplicities()[2] = 3.0;
    ws.ids()[0] = 100; ws.ids()[1] = 200; ws.ids()[2] = 300;
    ws.parent_ids()[0] = 7; ws.parent_ids()[1] = 8; ws.parent_ids()[2] = 9;
    ws.ages()[0] = 1; ws.ages()[1] = 2; ws.ages()[2] = 3;

    ws.apply_permutation(std::vector<std::size_t>{2, 0, 0});

    CHECK(ws.weights()[0] == Catch::Approx(30.0));
    CHECK(ws.weights()[1] == Catch::Approx(10.0));
    CHECK(ws.weights()[2] == Catch::Approx(10.0));
    CHECK(ws.multiplicities()[0] == Catch::Approx(3.0));
    CHECK(ws.ids()[0] == 300);
    CHECK(ws.ids()[1] == 100);
    CHECK(ws.parent_ids()[0] == 9);
    CHECK(ws.ages()[0] == 3);
    CHECK(ws.ages()[1] == 1);
}

TEST_CASE("apply_permutation can shrink the population (kill walkers)",
          "[walker_set][branch]") {
    WalkerSet ws(/*nw=*/3, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2), w1(2, 2), w2(2, 2);
    w0.set_positions(std::vector<double>{0.0, 0.0, 1.0, 0.0});
    w1.set_positions(std::vector<double>{0.0, 0.0, 2.0, 0.0});
    w2.set_positions(std::vector<double>{0.0, 0.0, 3.0, 0.0});
    ws.add_walker(w0);
    ws.add_walker(w1);
    ws.add_walker(w2);
    ws.load_all();

    // Keep only old walker 1.
    std::vector<std::size_t> src = {1};
    ws.apply_permutation(src);

    REQUIRE(ws.num_walkers() == 1);
    CHECK(ws.positions().size() == 2 * 2 * 1);
    CHECK(ws.local_energies().size() == 1);
    CHECK(ws.dist_tables().nw() == 1);
    CHECK(ws.dist_tables()(0, 0, 1) == Catch::Approx(2.0));
}

TEST_CASE("branch fissions/kills by int(weight+u) and resets weights",
          "[walker_set][branch]") {
    WalkerSet ws(/*nw=*/4, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2), w1(2, 2), w2(2, 2), w3(2, 2);
    for (Walker* w : {&w0, &w1, &w2, &w3}) ws.add_walker(*w);
    ws.load_all();
    auto pos = ws.positions();
    for (std::size_t w = 0; w < 4; ++w)
        pos[1 * 2 * 4 + 0 * 4 + w] = 10.0 + static_cast<double>(w);  // p1 x tag

    // Integer weights → m = int(k+u) = k deterministically.
    ws.weights()[0] = 0.0;  // killed
    ws.weights()[1] = 1.0;  // 1 copy
    ws.weights()[2] = 2.0;  // 2 copies
    ws.weights()[3] = 3.0;  // 3 copies

    alea::numerics::XoshiroRandom rng(123);
    (void)ws.branch(rng, /*target=*/4);  // n_max=9, n_min=1 → no bound clipping

    REQUIRE(ws.num_walkers() == 6);            // 0+1+2+3
    // Sources in order: [1, 2,2, 3,3,3] → p1 x = {11,12,12,13,13,13}.
    auto p = ws.positions();
    const std::size_t nw = 6;
    CHECK(p[1 * 2 * nw + 0 * nw + 0] == Catch::Approx(11.0));
    CHECK(p[1 * 2 * nw + 0 * nw + 1] == Catch::Approx(12.0));
    CHECK(p[1 * 2 * nw + 0 * nw + 3] == Catch::Approx(13.0));
    for (std::size_t w = 0; w < nw; ++w) {
        CHECK(ws.weights()[w] == Catch::Approx(1.0));
        CHECK(ws.multiplicities()[w] == Catch::Approx(1.0));
    }
}

TEST_CASE("branch clips population to n_max and lifts to n_min",
          "[walker_set][branch]") {
    {  // explosion → n_max = 2*target+1
        WalkerSet ws(4, 2, 2);
        Walker a(2, 2), b(2, 2), c(2, 2), d(2, 2);
        for (Walker* w : {&a, &b, &c, &d}) ws.add_walker(*w);
        ws.load_all();
        for (std::size_t w = 0; w < 4; ++w) ws.weights()[w] = 5.0;  // total 20
        alea::numerics::XoshiroRandom rng(1);
        (void)ws.branch(rng, /*target=*/2);  // n_max = 5
        CHECK(ws.num_walkers() == 5);
    }
    {  // collapse → n_min = target/5 + 1
        WalkerSet ws(4, 2, 2);
        Walker a(2, 2), b(2, 2), c(2, 2), d(2, 2);
        for (Walker* w : {&a, &b, &c, &d}) ws.add_walker(*w);
        ws.load_all();
        ws.weights()[0] = 1.0;  // others 0 → total 1
        ws.weights()[1] = 0.0; ws.weights()[2] = 0.0; ws.weights()[3] = 0.0;
        alea::numerics::XoshiroRandom rng(2);
        (void)ws.branch(rng, /*target=*/20);  // n_min = 5
        CHECK(ws.num_walkers() == 5);
    }
    {  // total extinction → keep the single best walker
        WalkerSet ws(3, 2, 2);
        Walker a(2, 2), b(2, 2), c(2, 2);
        for (Walker* w : {&a, &b, &c}) ws.add_walker(*w);
        ws.load_all();
        ws.weights()[0] = 0.0; ws.weights()[1] = 0.0; ws.weights()[2] = 0.0;
        alea::numerics::XoshiroRandom rng(3);
        (void)ws.branch(rng, /*target=*/4);
        CHECK(ws.num_walkers() >= 1);
    }
}

TEST_CASE("apply_move leaves rejected walkers untouched", "[walker_set]") {
    // Two walkers, identical start: p0=(0,0), p1=(3,4), r=5. Propose p0→(3,0)
    // (r=4) for both, accept only walker 0.
    WalkerSet ws(/*nw=*/2, /*np=*/2, /*dim=*/2);
    Walker w0(2, 2), w1(2, 2);
    std::vector<double> p = {0.0, 0.0, 3.0, 4.0};
    w0.set_positions(p);
    w1.set_positions(p);
    ws.add_walker(w0);
    ws.add_walker(w1);
    ws.load_all();

    std::vector<double> new_pos = {3.0, 3.0, 0.0, 0.0};  // [dim*nw]
    ws.propose_move_batch(/*p=*/0, new_pos);

    std::vector<double> new_grad = {0.5, 0.5, 0.0, 0.0};
    std::vector<double> new_lap = {-0.25, -0.25, 0.0, 0.0};  // [dim*nw]
    std::array<bool, 2> accept = {true, false};
    ws.apply_move(/*p=*/0, new_pos, new_grad, new_lap, accept);

    auto pos = ws.positions();
    CHECK(pos[0] == Catch::Approx(3.0));  // p0 x, w0 accepted
    CHECK(pos[1] == Catch::Approx(0.0));  // p0 x, w1 rejected → unchanged
    CHECK(ws.dist_tables()(0, 0, 1) == Catch::Approx(4.0));  // w0 committed
    CHECK(ws.dist_tables()(1, 0, 1) == Catch::Approx(5.0));  // w1 unchanged
    CHECK(ws.gradients()[0] == Catch::Approx(0.5));  // w0
    CHECK(ws.gradients()[1] == Catch::Approx(0.0));  // w1 unchanged
    CHECK(ws.laplacians()[0] == Catch::Approx(-0.25));  // w0
    CHECK(ws.laplacians()[1] == Catch::Approx(0.0));    // w1 unchanged
}
