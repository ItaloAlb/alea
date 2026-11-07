// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "population/walker_ensemble.h"

#include <utility>

#include "population/walker.h"

namespace alea {

WalkerEnsemble::WalkerEnsemble(std::size_t n_walkers,
                               std::vector<ParticleGroup> groups,
                               std::size_t dim,
                               const SimulationCell& cell)
    : n_particles_(0),
      dim_(dim),
      cell_(cell),
      target_walkers_(n_walkers),
      groups_(std::move(groups)) {
    for (const ParticleGroup& g : groups_) n_particles_ += g.size();
    walkers_.assign(n_walkers, Walker(n_particles_, dim_));
}

}  // namespace alea
