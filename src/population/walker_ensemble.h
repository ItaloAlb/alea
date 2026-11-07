// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "particle_group.h"
#include "simulation_cell.h"
#include "walker.h"

namespace alea {

// The starting configurations of one stage, shared by every driver kind
// (VMC, the optimiser, DMC, PIMC): one Walker per sample, the particle groups
// and the cell.  It holds no sampler state — DMC's population control lives in
// the driver (qmc/population_control.h).
class WalkerEnsemble {
public:
    // The cell travels with the ensemble so every driver-side WalkerSet is
    // built with the same geometry.  Defaulting it to open leaves existing
    // construction sites (and the open-boundary physics) unchanged.
    WalkerEnsemble(std::size_t n_walkers,
                   std::vector<ParticleGroup> groups,
                   std::size_t dim,
                   const SimulationCell& cell = SimulationCell{});

    std::span<Walker>       walkers()       { return walkers_; }
    std::span<const Walker> walkers() const { return walkers_; }

    std::size_t num_walkers()   const noexcept { return walkers_.size(); }
    std::size_t target_walkers() const noexcept { return target_walkers_; }
    std::size_t n_particles()   const noexcept { return n_particles_; }
    std::size_t dim()           const noexcept { return dim_; }
    const SimulationCell& cell() const noexcept { return cell_; }

    std::span<const ParticleGroup> groups() const noexcept { return groups_; }

private:
    std::size_t n_particles_;
    std::size_t dim_;
    SimulationCell cell_;
    std::size_t target_walkers_;

    std::vector<ParticleGroup> groups_;
    std::vector<Walker>        walkers_;
};

}  // namespace alea
