// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Verification of the reusable driver with mixed-species many-body
// systems. The coupled oscillator is a test-only analytic reference; the
// production configuration exercises the existing layered Coulomb builders.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include "builders/action_builders.h"
#include "hamiltonian/harmonic_trap.h"
#include "io/parser.h"
#include "pimc/pimc.h"
#include "qmc/app.h"

namespace {
constexpr std::size_t n_electrons = 3, n_holes = 3, np = 6, dim = 2;
constexpr std::array electron_mass{1.0, 2.0}, hole_mass{3.0, 5.0};

// V_pair = kappa/2 sum_{e,h} |r_e-r_h|^2, added to mass-scaled traps.
// For each axis the mass-weighted normal-mode squared frequencies are:
// omega^2, omega^2+kappa*(Nh/me+Ne/mh),
// omega^2+kappa*Nh/me (Ne-1 copies), omega^2+kappa*Ne/mh (Nh-1 copies).
// The primitive ring then factorizes into independent Gaussian oscillators
// (Ceperley, Rev. Mod. Phys. 67, 279 (1995), section II). Its finite-M energy
// is exact: sum_modes tau*lambda/M sum_k [4 sin^2(pi*k/M)+tau^2*lambda]^-1.
// Comparing at finite M avoids assigning a continuum Trotter tolerance.
class CoupledSpeciesPotential final : public alea::HamiltonianComponent {
public:
    explicit CoupledSpeciesPotential(double kappa) : kappa_(kappa) {}
    std::string name() const override { return "test_coupled_species"; }
    void evaluate_potential(const alea::ParticleBatch& batch, std::span<double> out,
                            std::size_t begin, std::size_t end) const override {
        end = std::min(end, batch.num_walkers());
        for (std::size_t k = begin; k < end; ++k)
            for (std::size_t e = 0; e < n_electrons; ++e)
                for (std::size_t h = n_electrons; h < np; ++h) {
                    const double distance = batch.dist_tables()(k, e, h);
                    out[k] += 0.5 * kappa_ * distance * distance;
                }
    }
    [[nodiscard]] std::unique_ptr<alea::HamiltonianComponent> clone() const override {
        return std::make_unique<CoupledSpeciesPotential>(*this);
    }
private:
    double kappa_;
};

std::vector<alea::ParticleGroup> mixed_groups() {
    alea::ParticleGroup electrons{}, holes{};
    electrons.name = "electrons"; electrons.first = 0; electrons.last = n_electrons;
    electrons.count = n_electrons; electrons.charge = -1.0;
    holes.name = "holes"; holes.first = n_electrons; holes.last = np;
    holes.count = n_holes; holes.charge = 1.0;
    for (std::size_t d = 0; d < dim; ++d) {
        electrons.mass.push_back(electron_mass[d]);
        electrons.inv_mass.push_back(1.0 / electron_mass[d]);
        holes.mass.push_back(hole_mass[d]);
        holes.inv_mass.push_back(1.0 / hole_mass[d]);
    }
    return {electrons, holes};
}

double discrete_mode_energy(double lambda, std::size_t slices, double tau) {
    double sum = 0.0;
    for (std::size_t k = 0; k < slices; ++k) {
        const double sine = std::sin(std::numbers::pi * static_cast<double>(k) /
                                     static_cast<double>(slices));
        sum += 1.0 / (4.0 * sine * sine + tau * tau * lambda);
    }
    return tau * lambda * sum / static_cast<double>(slices);
}

alea::io::Config layered_config(bool periodic) {
    auto cfg = alea::io::parse_config_file(
        std::string(ALEA_CONFIG_DIR) + "/pimc_many_body_eh.yaml");
    if (periodic) {
        cfg.lattice.type = "periodic";
        cfg.lattice.L = 8.0;
        // Explicit contract test of truncated interactions, not an Ewald or
        // finite-size physics reference. The kinetic image approximation also
        // remains the driver's declared minimum-image Gaussian.
        cfg.lattice.truncation = "minimum_image";
    }
    return cfg;
}

// Independent coordinate-based pair sum: do not read production distance tables
// or potential components. The fixture contains 3 ee + 3 hh + 9 eh pairs.
double layered_potential(const alea::PathSet& path, std::size_t k) {
    double potential = 0.0;
    for (std::size_t p = 0; p < np; ++p)
        for (std::size_t d = 0; d < dim; ++d) {
            const double mass = p < n_electrons ? electron_mass[d] : hole_mass[d];
            potential += 0.5 * mass * 0.64 * path.pos(p, d, k) * path.pos(p, d, k);
        }
    for (std::size_t p = 0; p < np; ++p)
        for (std::size_t q = p + 1; q < np; ++q) {
            const bool p_is_electron = p < n_electrons;
            const bool q_is_electron = q < n_electrons;
            const bool cross_species = p_is_electron != q_is_electron;
            double radius2 = cross_species ? 1.0 : 0.0;
            for (std::size_t d = 0; d < dim; ++d) {
                double delta = path.pos(p, d, k) - path.pos(q, d, k);
                if (path.cell().is_periodic()) delta = std::remainder(delta, 8.0);
                radius2 += delta * delta;
            }
            const double coefficient = cross_species ? -0.25 :
                (p_is_electron ? 0.5 : 1.0 / 3.0);
            potential += coefficient / std::sqrt(radius2);
        }
    return potential;
}
} // namespace

TEST_CASE("PIMC rejects a nonfinite many-body initial action before sampling",
          "[pimc][many-body][initial-action]") {
    auto cfg = layered_config(false);
    cfg.pimc.at(0).samples = 1;
    cfg.pimc.at(0).equilibration_blocks = 0;
    cfg.pimc.at(0).accumulation_blocks = 1;
    cfg.pimc.at(0).steps_per_block = 1;
    cfg.pimc.at(0).sub_steps = 1;
    bool imported = false;
    SECTION("coincident particles in the initial ensemble") {
        for (auto& particle : cfg.particles) particle.initialization = "fixed";
    }
    SECTION("coincident particles in an imported full path") { imported = true; }
    alea::App app(cfg);
    auto pack = app.build_drivers();
    auto* driver = dynamic_cast<alea::Pimc*>(pack.drivers.at(0).get());
    REQUIRE(driver != nullptr);
    if (imported) {
        auto checkpoint = driver->export_paths();
        const auto accepted=checkpoint.positions;
        std::fill(checkpoint.positions.begin(), checkpoint.positions.end(), 0.0);
        REQUIRE_THROWS_WITH(driver->import_paths(checkpoint),
                            Catch::Matchers::ContainsSubstring("checkpoint action is nonfinite"));
        CHECK(driver->export_paths().positions==accepted);
    } else {
        REQUIRE_THROWS_WITH(driver->run(),
                            Catch::Matchers::ContainsSubstring("initial action is nonfinite"));
    }
}

TEST_CASE("PIMC samples the finite-M normal modes of a coupled six-particle mixture",
          "[pimc][physics][many-body]") {
    constexpr double omega2 = 1.0, kappa = 0.4;
    const auto groups = mixed_groups();
    alea::Hamiltonian ham;
    ham.add_component(std::make_unique<alea::HarmonicTrap>(groups, omega2));
    ham.add_component(std::make_unique<CoupledSpeciesPotential>(kappa));
    alea::PimcConfig cfg;
    cfg.seed = 341981;
    cfg.threads = 2; cfg.samples = 48;
    cfg.beta = 2.0; cfg.time_step = 0.25;
    cfg.equilibration_blocks = 150; cfg.accumulation_blocks = 512;
    cfg.steps_per_block = 20; cfg.sub_steps = 6;
    cfg.moves = {{"bisection", 1.0, 2, 0.5}, {"centroid", 1.0, 1, 0.45},
                 {"single_slice", 1.0, 1, 0.35}};
    double exact = 0.0;
    for (std::size_t d = 0; d < dim; ++d) {
        const double electron_shift = kappa * n_holes / electron_mass[d];
        const double hole_shift = kappa * n_electrons / hole_mass[d];
        exact += discrete_mode_energy(omega2, 8, cfg.time_step);
        exact += discrete_mode_energy(omega2 + electron_shift + hole_shift, 8, cfg.time_step);
        exact += (n_electrons - 1) * discrete_mode_energy(omega2 + electron_shift, 8, cfg.time_step);
        exact += (n_holes - 1) * discrete_mode_energy(omega2 + hole_shift, 8, cfg.time_step);
    }
    alea::Pimc driver(cfg, groups, ham, alea::SimulationCell{}, dim, np);
    driver.run();
    INFO(driver.summary());
    INFO("exact finite-M energy = " << exact);
    // Fixed seed, four reblocked SEM plus the same 1e-3 floor used by the
    // existing physics tests. Precision caps prevent vacuous comparisons.
    CHECK(std::abs(driver.energy() - exact) <= 4.0 * driver.energy_error() + 1e-3);
    CHECK(std::abs(driver.kinetic() - exact / 2.0) <= 4.0 * driver.kinetic_error() + 1e-3);
    CHECK(std::abs(driver.potential() - exact / 2.0) <= 4.0 * driver.potential_error() + 1e-3);
    CHECK(driver.energy_error() < 0.05);
    CHECK(driver.kinetic_error() < 0.05);
    CHECK(driver.potential_error() < 0.05);
    CHECK(driver.plateau_found());
    CHECK(driver.energy() == Catch::Approx(driver.kinetic() + driver.potential()).margin(1e-10));
}

TEST_CASE("Layered many-body action and particle deltas match an independent coordinate sum",
          "[pimc][many-body][action]") {
    bool periodic = false;
    SECTION("open boundaries") {}
    SECTION("periodic minimum-image boundaries") { periodic = true; }
    const auto cfg = layered_config(periodic);
    const auto groups = alea::build_groups(cfg);
    const auto cell = alea::build_cell(cfg, np);
    const auto ham = alea::build_hamiltonian(cfg, groups, cell);
    auto action = alea::build_action(cfg, groups, ham, dim);
    alea::PathSet path(8, np, dim, 0.0625, cell);
    const std::array x{-1.4, -0.2, 1.0, -0.9, 0.5, 1.7};
    const std::array y{0.7, -0.8, 0.3, -0.2, 1.1, -0.6};
    for (std::size_t p = 0; p < np; ++p)
        for (std::size_t k = 0; k < path.num_slices(); ++k) {
            const double shift = 0.04 * std::sin(0.3 * static_cast<double>(k + p));
            path.pos(p, 0, k) = x[p] + shift;
            path.pos(p, 1, k) = y[p] - shift;
            if (periodic && p < 2) path.pos(p, 0, k) = (p == 0 ? 3.85 : -3.8) + shift;
        }
    path.rebuild_tables();
    double potential_sum = 0.0, springs = 0.0;
    for (std::size_t k = 0; k < path.num_slices(); ++k) {
        potential_sum += layered_potential(path, k);
        for (std::size_t p = 0; p < np; ++p)
            for (std::size_t d = 0; d < dim; ++d) {
                const double mass = p < n_electrons ? electron_mass[d] : hole_mass[d];
                double delta = path.pos(p, d, path.next(k)) - path.pos(p, d, k);
                if (periodic) delta = std::remainder(delta, 8.0);
                springs += mass * delta * delta / (2.0 * path.tau());
            }
    }
    CHECK(action.total_action(path) == Catch::Approx(springs + path.tau() * potential_sum).margin(1e-10));
    const auto thermo = action.measure_thermo(path);
    CHECK(thermo.potential == Catch::Approx(potential_sum / 8.0).margin(1e-10));
    CHECK(thermo.kinetic == Catch::Approx(np * dim / (2.0 * path.tau()) - springs / path.beta()).margin(1e-10));
    for (std::size_t p = 0; p < np; ++p) {
        constexpr std::size_t k = 7;
        const std::vector saved(path.positions().begin(), path.positions().end());
        const double old_potential = layered_potential(path, k);
        const std::array old{path.pos(p, 0, k), path.pos(p, 1, k)};
        const std::array proposed{old[0] + 0.5, old[1] - 0.3};
        const double delta = action.delta_potential_particle(path, p, k, proposed);
        CHECK(std::vector(path.positions().begin(), path.positions().end()) == saved);
        for (std::size_t d = 0; d < dim; ++d)
            path.pos(p, d, k) = periodic ? std::remainder(proposed[d], 8.0) : proposed[d];
        const double expected = path.tau() * (layered_potential(path, k) - old_potential);
        CHECK(delta == Catch::Approx(expected).margin(1e-10));
        for (std::size_t d = 0; d < dim; ++d) path.pos(p, d, k) = old[d];
        path.rebuild_tables();
    }
}

TEST_CASE("Layered many-body sampling and path continuation are thread-count independent",
          "[pimc][many-body][checkpoint]") {
    bool periodic = false;
    SECTION("open boundaries") {}
    SECTION("periodic minimum-image boundaries") { periodic = true; }
    auto cfg = layered_config(periodic);
    alea::App app_one(cfg);
    auto pack_one = app_one.build_drivers();
    auto* one = dynamic_cast<alea::Pimc*>(pack_one.drivers.at(0).get());
    REQUIRE(one != nullptr);
    one->run();
    const auto checkpoint = one->export_paths();
    cfg.pimc.at(0).threads = 3;
    alea::App app_three(cfg);
    auto pack_three = app_three.build_drivers();
    auto* three = dynamic_cast<alea::Pimc*>(pack_three.drivers.at(0).get());
    REQUIRE(three != nullptr);
    three->run();
    CHECK(three->export_paths().positions == checkpoint.positions);
    CHECK(three->run_result().scalars == one->run_result().scalars);
    CHECK(three->run_result().series == one->run_result().series);
    for (std::size_t i = 0; i < cfg.pimc.at(0).moves.size(); ++i) {
        const auto& results = one->run_result().scalars;
        const auto label = "move_" + std::to_string(i) + "_proposals";
        const auto proposal = std::find_if(results.begin(), results.end(),
            [&](const auto& item) { return item.first == label; });
        REQUIRE(proposal != results.end());
        CHECK(proposal->second > 0.0);
    }
    // Configuration continuation, not RNG/statistics continuation: both fresh
    // drivers import the same complete path and then use identical fresh seeds.
    auto pack_restart_one = app_one.build_drivers();
    auto pack_restart_three = app_three.build_drivers();
    auto* restart_one = dynamic_cast<alea::Pimc*>(pack_restart_one.drivers.at(0).get());
    auto* restart_three = dynamic_cast<alea::Pimc*>(pack_restart_three.drivers.at(0).get());
    REQUIRE(restart_one != nullptr);
    REQUIRE(restart_three != nullptr);
    restart_one->import_paths(checkpoint);
    restart_three->import_paths(checkpoint);
    CHECK(restart_one->export_paths().positions == checkpoint.positions);
    restart_one->run(); restart_three->run();
    CHECK(restart_one->export_paths().positions == restart_three->export_paths().positions);
    CHECK(restart_one->run_result().series == restart_three->run_result().series);
}
