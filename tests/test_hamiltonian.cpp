// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <memory>
#include <stdexcept>
#include <vector>

#include "population/particle_group.h"
#include "population/walker_set.h"
#include "hamiltonian/hamiltonian.h"
#include "hamiltonian/kinetic.h"
#include "hamiltonian/coulomb.h"

using alea::CoulombTerm;
using alea::Hamiltonian;
using alea::Kinetic;
using alea::ParticleGroup;
using alea::WalkerSet;

namespace {

ParticleGroup hgroup(std::size_t first, std::size_t last, double charge,
                     double inv_m) {
    ParticleGroup g;
    g.name = "g";
    g.charge = charge;
    g.first = first;
    g.last = last;
    g.inv_mass = {inv_m, inv_m};
    return g;
}

// p0=(0,0), p1=(2,0); grad_0=(1,0), |grad_0|²=1; grad_1=(0,0).
// Isotropic lap_0 (total -4) → ∂²_x = ∂²_y = -2.0.
// Kinetic (inv_m=1):  -1/2 [1·(-2+1) + 1·(-2+0)] = 1.5
// Coulomb (q=±1,ε=1): (+1)(-1)/2 = -0.5
// total local energy = 1.0
void setup(WalkerSet& ws) {
    auto pos = ws.positions();
    pos[2] = 2.0;  // p1 x
    ws.dist_tables().build_all(ws.positions());
    auto g = ws.gradients();
    g[0] = 1.0;  // p0 x
    // laplacian: [p*dim*nw + d*nw + w], nw=1
    ws.laplacians()[0] = -2.0;  // p0, x
    ws.laplacians()[1] = -2.0;  // p0, y
}

Hamiltonian make_hamiltonian() {
    Hamiltonian h;
    h.add_component(std::make_unique<Kinetic>(
        std::vector<ParticleGroup>{hgroup(0, 1, +1.0, 1.0), hgroup(1, 2, -1.0, 1.0)}));
    h.add_component(std::make_unique<CoulombTerm>(
        hgroup(0, 1, +1.0, 1.0), hgroup(1, 2, -1.0, 1.0), /*epsilon=*/1.0));
    return h;
}

}  // namespace

TEST_CASE("Hamiltonian evaluate_batch zeroes then sums its components",
          "[hamiltonian]") {
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    setup(ws);
    ws.local_energies()[0] = 999.0;  // garbage must be overwritten, not added to

    Hamiltonian h = make_hamiltonian();
    h.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(1.0));  // K (1.5) + V (-0.5)
}

TEST_CASE("Hamiltonian clone is an independent deep copy", "[hamiltonian]") {
    Hamiltonian h = make_hamiltonian();
    Hamiltonian c = h.clone();

    // Same components, but distinct objects (not shared pointers).
    REQUIRE(c.components().size() == h.components().size());
    CHECK(c.components()[0].get() != h.components()[0].get());
    CHECK(c.components()[1].get() != h.components()[1].get());

    // And the clone evaluates to the same energy.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    setup(ws);
    c.evaluate_batch(ws);
    CHECK(ws.local_energies()[0] == Catch::Approx(1.0));
}

TEST_CASE("Hamiltonian evaluate_potential sums only the potential terms over "
          "the requested range", "[hamiltonian]") {
    // Three configurations with e-h separations r = 2, 1, 4 along x.  The
    // kinetic term would add 1.5 to configuration 0 (setup's gradients), so a
    // result of exactly -1/r shows it was skipped.
    WalkerSet ws(/*nw=*/3, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();          // [p * dim * nw + d * nw + w]
    pos[2 * 3 + 0] = 2.0;               // p1 x, w=0
    pos[2 * 3 + 1] = 1.0;               // p1 x, w=1
    pos[2 * 3 + 2] = 4.0;               // p1 x, w=2
    ws.dist_tables().build_all(ws.positions());
    ws.gradients()[0] = 1.0;
    ws.laplacians()[0] = ws.laplacians()[3] = -2.0;

    const Hamiltonian h = make_hamiltonian();
    std::vector<double> v_out{999.0, 999.0, 999.0};
    h.evaluate_potential(ws, v_out, /*w_begin=*/1, /*w_end=*/3);

    CHECK(v_out[0] == 999.0);                       // outside the range: untouched
    CHECK(v_out[1] == Catch::Approx(-1.0));         // zeroed, then -1/1
    CHECK(v_out[2] == Catch::Approx(-0.25));        // -1/4

    // The same V a walker's local energy carries, minus the kinetic part.
    h.evaluate_potential(ws, v_out);
    CHECK(v_out[0] == Catch::Approx(-0.5));
}

TEST_CASE("the kinetic term is not a potential and refuses evaluate_potential",
          "[hamiltonian]") {
    const Hamiltonian h = make_hamiltonian();
    const auto& kinetic = *h.components()[0];
    const auto& coulomb = *h.components()[1];
    CHECK_FALSE(kinetic.is_potential());
    CHECK(coulomb.is_potential());

    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    setup(ws);
    std::vector<double> v_out(1, 0.0);
    CHECK_THROWS_AS(kinetic.evaluate_potential(ws, v_out), std::logic_error);
}
