// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// `type: coulomb` → V = q_a q_b / (ε √(r² + d²)), d the groups' layer
// separation: a CoulombTerm per spec, or its Ewald kernel under PBC.
#include "builders/hamiltonian_builders.h"

#include <stdexcept>
#include <string>

#include "hamiltonian/coulomb.h"

namespace alea {

std::unique_ptr<HamiltonianComponent> build_component(
    const io::CoulombSpec& c, const HamiltonianContext& ctx) {
    return std::make_unique<CoulombTerm>(
        find_group(ctx.groups, c.pair[0]), find_group(ctx.groups, c.pair[1]),
        c.epsilon, layer_separation(ctx.cfg, c.pair[0], c.pair[1]));
}

// Every pair gets the bare Coulomb kernel at its layer separation, listed in a
// spec or not.  The reciprocal sum couples every charge through one structure
// factor, so the cell carries ONE epsilon, and specs that disagree are refused.
numerics::PotentialSpec ewald_kernel(std::span<const io::CoulombSpec> specs,
                                     const ParticleGroup& a, const ParticleGroup& b,
                                     const HamiltonianContext& ctx) {
    const double eps0 = specs.front().epsilon;
    for (const io::CoulombSpec& cs : specs)
        if (cs.epsilon != eps0)
            throw std::runtime_error(
                "periodic lattice with `coulomb` terms of differing epsilon (" +
                std::to_string(eps0) + " vs " + std::to_string(cs.epsilon) +
                ").  The Ewald reciprocal sum couples every charge through a "
                "single structure factor, so one cell carries one epsilon.");
    return numerics::PotentialSpec::coulomb(
        eps0, layer_separation(ctx.cfg, a.name, b.name));
}

}  // namespace alea
