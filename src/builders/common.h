// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Helpers shared by every builder in this directory (wavefunction and
// Hamiltonian alike).

#include <span>
#include <string>

#include "../io/parser.h"
#include "../population/particle_group.h"

namespace alea {

[[nodiscard]] const ParticleGroup& find_group(std::span<const ParticleGroup> groups,
                                              const std::string& name);

// Out-of-plane distance |z_a − z_b| between two groups' planes, from their
// particles' `z` (a group with no particle spec sits at z = 0).  A pair in
// different planes feels √(r² + d²): this is what lets a bilayer e–h exciton be
// solved in real coordinates.
[[nodiscard]] double layer_separation(const io::Config& cfg,
                                      const std::string& group_a,
                                      const std::string& group_b);

}  // namespace alea
