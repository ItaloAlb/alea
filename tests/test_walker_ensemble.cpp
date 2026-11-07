// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "population/particle_group.h"
#include "population/walker_ensemble.h"

using alea::ParticleGroup;
using alea::WalkerEnsemble;

namespace {

ParticleGroup grp(std::size_t first, std::size_t last) {
    ParticleGroup g;
    g.name = "g";
    g.first = first;
    g.last = last;
    return g;
}

}  // namespace

TEST_CASE("WalkerEnsemble allocates walkers sized to the particle groups",
          "[walker_ensemble]") {
    // Two groups: [0,1) and [1,3) → 3 particles total.
    WalkerEnsemble ens(/*n_walkers=*/4, {grp(0, 1), grp(1, 3)}, /*dim=*/2);

    CHECK(ens.num_walkers() == 4);
    CHECK(ens.target_walkers() == 4);
    CHECK(ens.n_particles() == 3);
    CHECK(ens.dim() == 2);
    REQUIRE(ens.groups().size() == 2);

    REQUIRE(ens.walkers().size() == 4);
    CHECK(ens.walkers()[0].n_particles() == 3);
    CHECK(ens.walkers()[0].dim() == 2);
}
