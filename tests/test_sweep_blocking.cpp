// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include "hamiltonian/hamiltonian.h"
#include "hamiltonian/kinetic.h"
#include "numerics/random.h"
#include "population/particle_group.h"
#include "population/walker_ensemble.h"
#include "qmc/dmc.h"
#include "qmc/vmc.h"
#include "wavefunction/orbital_set.h"
#include "wavefunction/pade_jastrow.h"
#include "wavefunction/slater_determinant.h"
#include "wavefunction/trial_wavefunction.h"

// Cache-blocked sweeps (src/qmc/sweep_blocking.h) draw a sweep's random
// numbers before any walker moves and give each walker its own entries, so
// the walker block size must not change a single bit of a VMC or DMC run:
// energies, block series, DMC populations and final coordinates.

using namespace alea;

namespace {

constexpr std::size_t kElectrons = 19;   // closed plane-wave shell, N = 3² + 3·2 + 2²
const std::array<double, 4> kCell{10, 0, 5, 5 * std::sqrt(3.0)};   // hexagonal

ParticleGroup electrons() {
    ParticleGroup g;
    g.name = "e";
    g.charge = -1.0;
    g.first = 0;
    g.last = kElectrons;
    g.count = kElectrons;
    g.mass = {1.0, 1.0};
    g.inv_mass = {1.0, 1.0};
    return g;
}

// A plane-wave determinant with a same-spin Padé factor.  The cell is
// periodic, so N = 19 uses the walker-tile matrix layout for batches below
// 64 walkers and the contiguous per-walker layout above.
TrialWavefunction make_trial(const SimulationCell& cell) {
    std::vector<SlaterDeterminant::Block> blocks;
    blocks.push_back({0, kElectrons, make_plane_wave_orbitals(cell, kElectrons)});
    TrialWavefunction psi;
    psi.add_component(std::make_unique<SlaterDeterminant>(std::move(blocks)));
    std::vector<Parameter> params = {{"a", 0.25, true}, {"b", 0.5, true}};
    psi.add_component(std::make_unique<PadeJastrow>(1, 1, electrons(), electrons(),
                                                    std::move(params)));
    return psi;
}

// Triangular-crystal sites (well-conditioned Fourier matrix) with a random
// displacement per walker, so walkers start from distinct configurations.
WalkerEnsemble make_ensemble(std::size_t n_walkers, const SimulationCell& cell) {
    WalkerEnsemble ensemble(n_walkers, {electrons()}, 2, cell);
    constexpr int i = 3, j = 2, n = static_cast<int>(kElectrons);
    std::vector<std::array<int, 2>> fractions;
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b) {
            const std::array<int, 2> f{((i + j) * a + j * b) % n, ((-j * a + i * b) % n + n) % n};
            if (std::find(fractions.begin(), fractions.end(), f) == fractions.end())
                fractions.push_back(f);
        }
    REQUIRE(fractions.size() == kElectrons);
    numerics::Random rng(907);
    std::vector<double> positions(2 * kElectrons);   // [particle * dim + d]
    for (auto& walker : ensemble.walkers()) {
        for (std::size_t p = 0; p < kElectrons; ++p) {
            const auto f = fractions[p];
            positions[2 * p]     = (f[0] * kCell[0] + f[1] * kCell[2]) / n + rng.uniform(-0.3, 0.3);
            positions[2 * p + 1] = (f[0] * kCell[1] + f[1] * kCell[3]) / n + rng.uniform(-0.3, 0.3);
        }
        walker.set_positions(positions);
    }
    return ensemble;
}

Hamiltonian make_hamiltonian() {
    Hamiltonian ham;
    ham.add_component(std::make_unique<Kinetic>(std::vector<ParticleGroup>{electrons()}));
    return ham;
}

struct Outcome {
    double energy = 0.0;
    std::vector<double> block_energies, populations, coordinates;
};

Outcome run_vmc(std::size_t block, bool drift, int threads, std::size_t n_walkers) {
    const SimulationCell cell(2, kCell);
    WalkerEnsemble ensemble = make_ensemble(n_walkers, cell);
    const TrialWavefunction psi = make_trial(cell);
    const Hamiltonian ham = make_hamiltonian();
    VMCConfig cfg;
    cfg.n_equilibration_blocks = 2;
    cfg.n_accumulation_blocks = 4;
    cfg.n_steps = 3;
    cfg.n_substeps = 2;
    cfg.tau = 0.3;
    cfg.use_drift = drift;
    cfg.n_threads = threads;
    cfg.seed = 11;
    cfg.sweep_block_walkers = block;
    VMC vmc(cfg, ensemble, psi, ham);
    vmc.run();
    Outcome out;
    out.energy = vmc.energy();
    out.block_energies.assign(vmc.block_energies().begin(), vmc.block_energies().end());
    out.coordinates = vmc.final_configurations();
    return out;
}

Outcome run_dmc(std::size_t block, int threads, std::size_t n_walkers) {
    const SimulationCell cell(2, kCell);
    WalkerEnsemble ensemble = make_ensemble(n_walkers, cell);
    const TrialWavefunction psi = make_trial(cell);
    const Hamiltonian ham = make_hamiltonian();
    DMCConfig cfg;
    cfg.n_equilibration_blocks = 2;
    cfg.n_accumulation_blocks = 4;
    cfg.n_steps = 3;
    cfg.n_substeps = 2;
    cfg.tau = 0.05;
    cfg.n_threads = threads;
    cfg.seed = 13;
    cfg.sweep_block_walkers = block;
    DMC dmc(cfg, ensemble, psi, ham);
    dmc.run();
    Outcome out;
    out.energy = dmc.energy();
    out.block_energies.assign(dmc.block_energies().begin(), dmc.block_energies().end());
    out.populations.assign(dmc.block_population().begin(), dmc.block_population().end());
    out.coordinates = dmc.final_configurations();
    return out;
}

void require_identical(const Outcome& a, const Outcome& b) {
    REQUIRE(std::isfinite(a.energy));
    REQUIRE(a.energy == b.energy);
    REQUIRE(a.block_energies == b.block_energies);
    REQUIRE(a.populations == b.populations);
    REQUIRE(a.coordinates.size() == b.coordinates.size());
    REQUIRE(a.coordinates == b.coordinates);
}

}  // namespace

TEST_CASE("VMC results do not depend on the sweep's walker block size",
          "[vmc][sweep_blocking]") {
    const bool drift = GENERATE(false, true);
    CAPTURE(drift);
    // One thread with 24 walkers: the determinant uses the walker-tile layout.
    const Outcome whole = run_vmc(24, drift, 1, 24);
    for (std::size_t block : {std::size_t{1}, std::size_t{3}, std::size_t{8}, std::size_t{0}}) {
        CAPTURE(block);
        require_identical(run_vmc(block, drift, 1, 24), whole);
    }
}

TEST_CASE("DMC results do not depend on the sweep's walker block size",
          "[dmc][sweep_blocking]") {
    // 72 walkers on two threads: contiguous per-walker matrices, crowds of 36
    // that branching resizes, so blocks straddle a changing crowd boundary.
    const Outcome whole = run_dmc(1000, 2, 72);
    REQUIRE(whole.populations.size() == 4);
    for (std::size_t block : {std::size_t{1}, std::size_t{5}, std::size_t{0}}) {
        CAPTURE(block);
        require_identical(run_dmc(block, 2, 72), whole);
    }
}
