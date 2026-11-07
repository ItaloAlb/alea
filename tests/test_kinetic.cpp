// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <vector>

#include "population/particle_group.h"
#include "population/walker_set.h"
#include "hamiltonian/kinetic.h"

using alea::Kinetic;
using alea::ParticleGroup;
using alea::WalkerSet;

namespace {

ParticleGroup kgroup(std::size_t first, std::size_t last, double inv_m) {
    ParticleGroup g;
    g.name = "g";
    g.first = first;
    g.last = last;
    g.inv_mass = {inv_m, inv_m};
    return g;
}

}  // namespace

TEST_CASE("Kinetic accumulates -1/2 (1/m)(lap + |grad|^2) per particle",
          "[kinetic]") {
    // One particle, dim=2, inv_mass=(1,1). grad=(1,0) → |grad|²=1.
    // Isotropic lap: ∂²_x = ∂²_y = -1.0 (total -2.0).
    // K = -1/2 * [1*(-1 + 1) + 1*(-1 + 0)] = 0.5.
    WalkerSet ws(/*nw=*/1, /*np=*/1, /*dim=*/2);
    auto g = ws.gradients();   // [p*dim*nw + d*nw + w], nw=1
    g[0] = 1.0;                // p0, x
    g[1] = 0.0;                // p0, y
    auto l = ws.laplacians();  // [p*dim*nw + d*nw + w]
    l[0] = -1.0;               // p0, x
    l[1] = -1.0;               // p0, y

    Kinetic kin({kgroup(0, 1, /*inv_m=*/1.0)});
    kin.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(0.5));
}

TEST_CASE("Kinetic scales each particle by its inverse mass", "[kinetic]") {
    // Same as above but inv_mass=(0.5, 0.5) → K = 0.25.
    WalkerSet ws(/*nw=*/1, /*np=*/1, /*dim=*/2);
    auto g = ws.gradients();
    g[0] = 1.0;
    g[1] = 0.0;
    auto l = ws.laplacians();
    l[0] = -1.0;
    l[1] = -1.0;

    Kinetic kin({kgroup(0, 1, /*inv_m=*/0.5)});
    kin.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(0.25));
}

TEST_CASE("Kinetic sums over particles using per-group inverse masses",
          "[kinetic]") {
    // p0: grad=(1,0), |grad|²=1,  lap=(-1,-1),  inv_m=(1,1)    →  0.5
    // p1: grad=(0,2), |grad|²=4,  lap=(-1.5,-1.5), inv_m=(0.5,0.5) → -0.25
    // total = 0.25
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto g = ws.gradients();   // [p*dim*nw + d*nw + w], nw=1
    g[0] = 1.0;   // p0 x
    g[1] = 0.0;   // p0 y
    g[2] = 0.0;   // p1 x
    g[3] = 2.0;   // p1 y
    auto l = ws.laplacians();
    l[0] = -1.0;   // p0 x
    l[1] = -1.0;   // p0 y
    l[2] = -1.5;   // p1 x
    l[3] = -1.5;   // p1 y

    Kinetic kin({kgroup(0, 1, 1.0), kgroup(1, 2, 0.5)});
    kin.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(0.25));
}

TEST_CASE("Kinetic handles anisotropic inverse mass", "[kinetic][anisotropy]") {
    // One particle, dim=2. inv_mass = (1.0, 0.5). grad=(1,2), |grad|²=(1,4).
    // lap = (-0.5, -0.3).
    // K = -1/2 * [1.0*(-0.5 + 1) + 0.5*(-0.3 + 4)] = -1/2*[0.5 + 1.85] = -1.175
    WalkerSet ws(/*nw=*/1, /*np=*/1, /*dim=*/2);
    auto g = ws.gradients();
    g[0] = 1.0;
    g[1] = 2.0;
    auto l = ws.laplacians();
    l[0] = -0.5;
    l[1] = -0.3;

    ParticleGroup pg;
    pg.name = "g";
    pg.first = 0;
    pg.last = 1;
    pg.inv_mass = {1.0, 0.5};
    Kinetic kin({pg});
    kin.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(-1.175));
}
