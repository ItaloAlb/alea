// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Numerical and approximation contracts for the primitive boltzmannon driver.
// Statistical bars do not cover finite-tau or periodic-image truncation bias.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <array>
#include <cmath>
#include <limits>
#include "pimc/pimc.h"
#include "hamiltonian/harmonic_trap.h"

namespace {
alea::ParticleGroup particle() {
    alea::ParticleGroup g;
    g.first = 0; g.last = 1; g.mass = {1.0}; g.inv_mass = {1.0};
    return g;
}
alea::PimcConfig config() {
    alea::PimcConfig c;
    c.samples = 2; c.threads = 1; c.beta = 2; c.time_step = 0.0625;
    c.equilibration_blocks = 0; c.accumulation_blocks = 2;
    c.steps_per_block = 2; c.sub_steps = 1;
    return c;
}
}

TEST_CASE("PIMC validates numeric inputs before allocations and integer conversions", "[pimc][contract]") {
    const std::array groups{particle()}; const alea::Hamiltonian ham;
    auto c = config();
    SECTION("negative inverse temperature") { c.beta = -1; }
    SECTION("nonfinite inverse temperature") { c.beta = std::numeric_limits<double>::quiet_NaN(); }
    SECTION("nonfinite time step") { c.time_step = std::numeric_limits<double>::infinity(); }
    SECTION("zero replicas") { c.samples = 0; }
    SECTION("negative replicas") { c.samples = -1; }
    SECTION("negative accumulation count") { c.accumulation_blocks = -1; }
    SECTION("no measurement steps") { c.steps_per_block = 0; }
    SECTION("negative sweeps") { c.sub_steps = -1; }
    SECTION("no accumulated samples") { c.accumulation_blocks = 0; }
    REQUIRE_THROWS_AS(alea::Pimc(c, groups, ham, alea::SimulationCell{}, 1, 1), std::runtime_error);
}

TEST_CASE("PIMC refuses uncontrolled periodic minimum-image kernels", "[pimc][contract]") {
    auto c = config(); c.beta = 32; c.time_step = 32;
    const std::array groups{particle()}; const alea::Hamiltonian ham;
    const std::array lattice{4.0}; const alea::SimulationCell cell(1, lattice);
    REQUIRE_THROWS_WITH(alea::Pimc(c, groups, ham, cell, 1, 1),
        Catch::Matchers::ContainsSubstring("minimum-image"));
}

TEST_CASE("PIMC refuses an action with a different spatial dimension",
          "[pimc][contract][geometry]") {
    const std::array groups{particle()};
    const alea::Hamiltonian free;
    const alea::Action wrong_dimension(groups, free, 2);
    REQUIRE_THROWS_WITH(
        alea::Pimc(config(), groups, free, alea::SimulationCell{}, 1, 1,
                   nullptr, &wrong_dimension),
        Catch::Matchers::ContainsSubstring("action dimension"));
}

TEST_CASE("PIMC and PathSet refuse a periodic cell with a different dimension",
          "[pimc][contract][geometry]") {
    const std::array groups{particle()};
    const alea::Hamiltonian free;
    const std::array lattice{8.0, 0.0, 0.0, 8.0};
    const alea::SimulationCell cell(2, lattice);
    SECTION("driver construction") {
        REQUIRE_THROWS_WITH(alea::Pimc(config(), groups, free, cell, 1, 1),
            Catch::Matchers::ContainsSubstring("cell dimension"));
    }
    SECTION("standalone path construction") {
        REQUIRE_THROWS_WITH(alea::PathSet(4, 1, 1, 0.0625, cell),
            Catch::Matchers::ContainsSubstring("cell dimension"));
    }
}

TEST_CASE("Path initialization validates and canonicalizes external coordinates", "[pimc][contract]") {
    const std::array lattice{4.0}; const alea::SimulationCell cell(1, lattice);
    SECTION("collapse wraps coordinates") {
        alea::PathSet path(4, 1, 1, .0625, cell);
        const std::array coords{13.0}; path.collapse_to(coords);
        CHECK(path.pos(0, 0, 0) == 1.0);
    }
    SECTION("nonfinite coordinates") {
        alea::PathSet path(4, 1, 1, .0625, cell);
        const std::array coords{std::numeric_limits<double>::infinity()};
        REQUIRE_THROWS_AS(path.collapse_to(coords), std::runtime_error);
    }
    SECTION("nonfinite time step") {
        REQUIRE_THROWS_AS(alea::PathSet(4, 1, 1, std::numeric_limits<double>::quiet_NaN(), cell), std::runtime_error);
    }
}

TEST_CASE("PIMC results state approximation and insufficient-statistics limitations", "[pimc][contract]") {
    const std::array groups{particle()}; const alea::Hamiltonian ham;
    auto c = config();
    alea::Pimc driver(c, groups, ham, alea::SimulationCell{}, 1, 1);
    driver.run();
    CHECK_THAT(driver.summary(), Catch::Matchers::ContainsSubstring("finite-tau"));
    CHECK_THAT(driver.summary(), Catch::Matchers::ContainsSubstring("equilibration"));
    CHECK_THAT(driver.summary(), Catch::Matchers::ContainsSubstring("WARNING"));
    const auto r = driver.run_result();
    bool found = false;
    for (const auto& [key, val] : r.scalars)
        if (key == "energy_effective_blocks") { found = true; CHECK(val <= 2.0); }
    CHECK(found);
}

TEST_CASE("PIMC exposes free-CM correction and per-move diagnostics without outgoing walker conversion", "[pimc][contract]") {
    const std::array groups{particle()};
    const alea::Hamiltonian free;
    auto c = config();
    alea::Pimc driver(c, groups, free, alea::SimulationCell{}, 1, 1);
    driver.run();
    const auto r = driver.run_result();
    const auto scalar = [&](const std::string& key) {
        for (const auto& [name, value] : r.scalars)
            if (name == key) return value;
        FAIL_CHECK("missing scalar " << key);
        return 0.0;
    };
    CHECK(scalar("internal_energy") == driver.energy() - 1.0 / (2.0 * c.beta));
    CHECK(scalar("cm_energy_correction") == 1.0 / (2.0 * c.beta));
    CHECK(scalar("move_0_proposals") + scalar("move_1_proposals") ==
          c.samples * c.accumulation_blocks * c.steps_per_block * c.sub_steps);
    CHECK(driver.final_configurations().empty());
}

