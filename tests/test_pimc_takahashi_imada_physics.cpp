// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Takahashi-Imada action through YAML -> App -> Pimc, against exact values of
// the same finite-M path integral (not the continuum), each with a power
// check: the primitive action at the same M must lie more than 5 sigma away,
// so the tau^3 |grad V|^2 term is what the run is measured on.
//
// (1) Harmonic oscillator: the TI slice is a primitive slice with
//     w^2 (1 + tau^2 w^2 / 12), a Gaussian ring with closed-form E and the
//     Hellmann-Feynman <V> (tests/configs/pimc_ho_1d_takahashi_imada.yaml).
// (2) Two dipoles in a trap, grand-canonical window {1, 2}: the radial
//     transfer matrix of tests/references/inverse_power_trap_primitive.py
//     with --action takahashi_imada (the centre of mass still separates).
//     Tests the canonical moves, the worm's bead actions, the sector energies
//     and F(2) - F(1) with the cross and pair terms of |grad V|^2.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>

#include "io/parser.h"
#include "pimc/pimc.h"
#include "qmc/app.h"

namespace {
alea::io::Config load_config(const std::string& file) {
    return alea::io::parse_config_file(std::string(ALEA_CONFIG_DIR) + "/" + file);
}

template<class Inspect>
void run_single_stage(const alea::io::Config& cfg, Inspect inspect) {
    alea::App app(cfg);
    alea::DriverPack pack = app.build_drivers();
    REQUIRE(pack.drivers.size() == 1);
    auto* pimc = dynamic_cast<alea::Pimc*>(pack.drivers.front().get());
    REQUIRE(pimc != nullptr);
    pimc->run();
    inspect(*pimc);
}
}  // namespace

TEST_CASE("The Takahashi-Imada action reproduces the discretized harmonic oscillator",
          "[pimc][physics][takahashi_imada]") {
    constexpr double kEnergy = 0.6562772779106;          // TI, M = 4
    constexpr double kHalf = 0.3281386390;               // <V> = <K>
    constexpr double kPrimitiveEnergy = 0.6405228757944; // primitive, M = 4
    run_single_stage(load_config("pimc_ho_1d_takahashi_imada.yaml"), [&](const alea::Pimc& pimc) {
        INFO(pimc.summary());
        const double e = pimc.energy(), se = pimc.energy_error();
        CHECK(std::abs(e - kEnergy) <= 4.0 * se + 1e-4);
        CHECK(std::abs(kPrimitiveEnergy - kEnergy) > 5.0 * se);
        CHECK(std::abs(pimc.potential() - kHalf) <= 4.0 * pimc.potential_error() + 1e-4);
        CHECK(std::abs(pimc.kinetic() - kHalf) <= 4.0 * pimc.kinetic_error() + 1e-4);
        CHECK(pimc.summary().find("takahashi_imada") != std::string::npos);
    });
}

TEST_CASE("The Takahashi-Imada action reproduces the discretized 3-D harmonic oscillator",
          "[pimc][physics][takahashi_imada][3d]") {
    // Three independent 1-D rings (pimc_ho_3d_takahashi_imada.yaml): the
    // moves, the forces and the |grad V|^2 sum run over three axes.
    constexpr double kEnergy = 1.9688318337318;           // TI, M = 4
    constexpr double kHalf = 0.9844159168659;             // <V> = <K>
    constexpr double kPrimitiveEnergy = 1.9215686273832;  // primitive, M = 4
    run_single_stage(load_config("pimc_ho_3d_takahashi_imada.yaml"), [&](const alea::Pimc& pimc) {
        INFO(pimc.summary());
        const double e = pimc.energy(), se = pimc.energy_error();
        CHECK(std::abs(e - kEnergy) <= 4.0 * se + 1e-4);
        CHECK(std::abs(kPrimitiveEnergy - kEnergy) > 5.0 * se);
        CHECK(std::abs(pimc.potential() - kHalf) <= 4.0 * pimc.potential_error() + 1e-4);
        CHECK(std::abs(pimc.kinetic() - kHalf) <= 4.0 * pimc.kinetic_error() + 1e-4);
    });
}

TEST_CASE("A Takahashi-Imada run refuses the virial energy estimators", "[pimc][takahashi_imada]") {
    auto cfg = load_config("pimc_ho_1d_takahashi_imada.yaml");
    cfg.pimc.front().energy = "centroid_virial";
    alea::App app(cfg);
    try {
        (void)app.build_drivers();
        FAIL("a centroid-virial estimator with the Takahashi-Imada action was accepted");
    } catch (const std::exception& e) {
        CHECK(std::string(e.what()).find("Takahashi-Imada") != std::string::npos);
    }
}

TEST_CASE("Two trapped dipoles with the Takahashi-Imada action give the exact F(2) - F(1) and sector energies",
          "[pimc][grand_canonical][physics][takahashi_imada]") {
    constexpr double kDeltaF = 1.843217428155;   // TI, M = 8
    constexpr double kEnergy1 = 1.313004021710;
    constexpr double kEnergy2 = 3.432125016076;
    constexpr double kPrimitiveDeltaF = 1.807635592069;  // primitive, M = 8
    constexpr double kPrimitiveEnergy2 = 3.346367849628;
    run_single_stage(load_config("pimc_gc_trapped_dipoles_takahashi_imada.yaml"),
                     [&](const alea::Pimc& pimc) {
        REQUIRE(pimc.grand_canonical());
        const alea::PimcGrandCanonicalResult& gc = pimc.grand_canonical_result();
        INFO(pimc.summary());
        REQUIRE(gc.sectors.size() == 2);
        CHECK(gc.pair_transitions[0] >= 100.0);
        const double mu = gc.mu[0], mu_error = gc.mu_error[0];
        CHECK(std::abs(mu - kDeltaF) <= 4.0 * mu_error);
        CHECK(std::abs(kPrimitiveDeltaF - kDeltaF) > 5.0 * mu_error);
        const alea::PimcSectorResult& one = gc.sectors[0];
        const alea::PimcSectorResult& two = gc.sectors[1];
        CHECK(std::abs(one.energy - kEnergy1) <= 4.0 * one.energy_error);
        CHECK(std::abs(two.energy - kEnergy2) <= 4.0 * two.energy_error);
        CHECK(std::abs(kPrimitiveEnergy2 - kEnergy2) > 5.0 * two.energy_error);
    });
}
