// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Full-path continuation is separate from ordinary walker carry-over.  Check
// complete beads, permutation, grid and lattice before replacing any path.
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "io/path_io.h"
#include "pimc/pimc.h"

namespace {
alea::ParticleGroup one_particle() {
    alea::ParticleGroup group;
    group.first = 0;
    group.last = 1;
    group.mass = {1.0};
    group.inv_mass = {1.0};
    return group;
}
alea::PimcConfig small_config() {
    alea::PimcConfig cfg;
    cfg.beta = 0.25;
    cfg.time_step = 0.0625;
    cfg.samples = 2;
    cfg.threads = 1;
    cfg.equilibration_blocks = 0;
    cfg.accumulation_blocks = 1;
    cfg.steps_per_block = 1;
    cfg.sub_steps = 1;
    return cfg;
}
}

TEST_CASE("PIMC path import preserves every noncollapsed bead and rejects incompatible checkpoints atomically", "[pimc][checkpoint]") {
    const std::array groups{one_particle()};
    const alea::Hamiltonian free;
    alea::Pimc driver(small_config(), groups, free, alea::SimulationCell{}, 1, 1);
    const auto original = driver.export_paths();
    REQUIRE(original.replicas == 2);
    REQUIRE(original.m == 4);
    REQUIRE(original.positions.size() == 8);
    REQUIRE(original.permutation == std::vector<std::size_t>{0, 0});
    auto cp = original;
    cp.positions = {0.0, .1, .2, .3, -.1, -.2, -.3, -.4};
    driver.import_paths(cp);
    CHECK(driver.export_paths().positions == cp.positions);
    CHECK(driver.final_configurations().empty());
    auto incompatible = cp;
    incompatible.tau = .125;
    REQUIRE_THROWS(driver.import_paths(incompatible));
    CHECK(driver.export_paths().positions == cp.positions);
    incompatible = cp;
    incompatible.permutation[1] = 1;
    REQUIRE_THROWS(driver.import_paths(incompatible));
    CHECK(driver.export_paths().positions == cp.positions);
    incompatible = cp;
    incompatible.positions[4] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS(driver.import_paths(incompatible));
    CHECK(driver.export_paths().positions == cp.positions);
    // With no incoming walkers, run must not collapse the imported ring.
    // A zero-step run is forbidden, so an exact pre-run path query above
    // verifies import rather than relying on random MC acceptance.
}

TEST_CASE("PIMC checkpoint compatibility refuses a different periodic lattice", "[pimc][checkpoint]") {
    const std::array groups{one_particle()};
    const alea::Hamiltonian free;
    const std::array lattice{4.0};
    alea::Pimc driver(small_config(), groups, free, alea::SimulationCell(1, lattice), 1, 1);
    auto cp = driver.export_paths();
    REQUIRE(cp.lattice.size() == 1);
    cp.lattice[0] = 5.0;
    REQUIRE_THROWS(driver.import_paths(cp));
}
