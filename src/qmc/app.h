// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "../hamiltonian/hamiltonian.h"
#include "../io/parser.h"
#include "../population/particle_group.h"
#include "../population/walker_ensemble.h"
#include "../wavefunction/trial_wavefunction.h"
#include "dmc.h"
#include "driver.h"
#include "linear_optimize.h"
#include "vmc.h"
#include "../pimc/pimc.h"

namespace alea {

// ── Builders: translate a parsed Config into domain objects. ────────────────

[[nodiscard]] std::vector<ParticleGroup> build_groups(const io::Config& cfg);

[[nodiscard]] VMCConfig build_vmc_config(const io::QmcSpec& spec,
                                          std::uint64_t seed);

[[nodiscard]] DMCConfig build_dmc_config(const io::QmcSpec& spec,
                                          std::uint64_t seed);

[[nodiscard]] PimcConfig build_pimc_config(const io::PimcSpec& spec,
                                            std::uint64_t seed);

// `ham`, when provided, supplies the short-range 1/r residues that set the cusp
// of any Drummond jastrow (Γ = 2μR/(d−1)).  Pass nullptr for Padé-only configs
// or when no cusp derivation is wanted (Drummond then defaults to cuspless).
//
// `cell` is needed by any orbital set that must be periodic: plane waves read
// the reciprocal lattice off it, and Gaussian crystal orbitals fold r - c to the
// minimum image.  It defaults to an open cell, which leaves every previously
// supported wavefunction byte-identical.
[[nodiscard]] TrialWavefunction build_wavefunction(
    const io::Config& cfg,
    const std::vector<ParticleGroup>& groups,
    const Hamiltonian* ham = nullptr,
    const SimulationCell& cell = SimulationCell{});

// Crystal-site anchors for every crystal-mode determinant (`orbitals: {sites:
// ...}`, or explicit `centers` with one shared `alpha`) and every `nosanow`
// factor: [np*dim], NaN in the slots of particles that are not on a site, plus
// the largest Gaussian spread of those factors.  ONE source of truth for two
// consumers -- the sites and the initial walker positions -- because a
// localized crystal started from uniform positions hands the determinant a
// near-singular matrix (every Gaussian is exponentially small away from its
// own site), and a Nosanow factor a huge −α|r − R|².
struct CrystalAnchors {
    std::vector<double> pos;
    double              jitter = 0.0;
};
[[nodiscard]] CrystalAnchors crystal_site_anchors(
    const io::Config& cfg,
    const std::vector<ParticleGroup>& groups,
    const SimulationCell& cell);

// The start every `pairing` determinant needs: column particle j of a block
// begins next to row particle j, displaced by a Gaussian of width `spread` per
// axis.  `partner` is [np], −1 for a particle with no partner (rows, unpaired
// rows, other groups).  A compact φ (cutoff L far below the cell, as for
// tightly bound pairs) started from uniform positions vanishes for nearly every
// row–column pair, so the determinant would start at zero.
struct PairPartners {
    std::vector<std::ptrdiff_t> partner;
    double                      spread = 0.0;
};
[[nodiscard]] PairPartners pairing_partners(
    const io::Config& cfg,
    const std::vector<ParticleGroup>& groups,
    const SimulationCell& cell);

// The simulation cell the config describes for `n_particles` particles: open,
// or periodic from `vectors`, `rs`, `L: [Lx, Ly]`, or `L` with `shape`.
[[nodiscard]] SimulationCell build_cell(const io::Config& cfg, std::size_t n_particles);

// `cell` defaults to an open cell, which means no Ewald term and therefore the
// per-pair CoulombTerms — exactly the pre-periodic behaviour.
[[nodiscard]] Hamiltonian build_hamiltonian(
    const io::Config& cfg,
    const std::vector<ParticleGroup>& groups,
    const SimulationCell& cell = SimulationCell{});

[[nodiscard]] OptimizeConfig build_optimize_config(const io::OptimizerSpec& spec,
                                                    std::uint64_t seed);

// ── App: build drivers from Config and run them serially. ──────────────────

struct DriverPack {
    std::vector<std::unique_ptr<Driver>> drivers;
    // Shareable objects that the drivers reference.  These must outlive the
    // drivers.  psi is shared so that an optimiser can update parameters and
    // a subsequent VMC/DMC run sees the improved values.
    std::shared_ptr<TrialWavefunction> psi;
    std::shared_ptr<Hamiltonian>       hamiltonian;
    std::vector<std::shared_ptr<WalkerEnsemble>> ensembles;
    // Per-driver walker-checkpoint output path ("" → none), parallel to
    // `drivers`.  main serialises driver->final_population() to this path
    // after run() — keeping file I/O out of the qmc layer.
    std::vector<std::string> walkers_out;
    // Full PIMC path checkpoints are separate from walker carry-over.
    // main owns I/O; these paths are parallel to drivers (empty on VMC/DMC).
    std::vector<std::string> paths_in;
    std::vector<std::string> paths_out;
    // Per-driver: 1 if the driver was seeded from an explicit `walkers_in`
    // checkpoint, which then takes precedence over the carry-over below.
    std::vector<char> walkers_in_set;
};

// Overwrite the ensemble's walker positions with `rows` (AoS [w][p][d],
// n_rows = rows.size()/(np·dim) configurations), cycling round-robin when the
// counts differ: walker w takes row w % n_rows.  Surplus rows are dropped;
// missing ones are duplicates that decorrelate during equilibration.  Same
// rule as QMCPACK's MCPopulation::createWalkers (iw % num_existing_walkers).
void seed_walker_rows(WalkerEnsemble& ensemble, std::span<const double> rows);

// Driver-to-driver carry-over (default for every VMC/DMC driver, as in
// QMCPACK: QMCDriverNew::finalize "carries over" the population to the next
// section).  After driver `from` has run, seed driver `to` with its final
// configurations and reload them — unless `to` was given an explicit
// walkers_in, or `from` has nothing to hand on (e.g. the optimiser).
// Returns the number of configurations carried (0 = nothing done).
std::size_t carry_over_walkers(DriverPack& pack, std::size_t from, std::size_t to);

class App {
public:
    explicit App(io::Config config);

    // Build Driver objects and their associated ensembles from the config.
    // Domain objects (psi, hamiltonian) are created once and shared by
    // reference — so an optimiser updates parameters and subsequent VMC/DMC
    // runs see the improved wavefunction.
    //
    // `checkpoint_dir`, when non-empty, is where every driver's `walkers_out`
    // filename (a bare name — no path — in the YAML) actually lands: each
    // becomes `checkpoint_dir / walkers_out`, milestones included. Left empty
    // (the default), a `walkers_out` path is used exactly as written in the
    // YAML, relative to the process's working directory — kept only so
    // existing callers/tests that pass no directory are unaffected.
    [[nodiscard]] DriverPack build_drivers(
        const std::filesystem::path& checkpoint_dir = {});

private:
    io::Config cfg_;
};

}  // namespace alea
