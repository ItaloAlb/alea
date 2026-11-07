// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// `type: screened_coulomb` → the multilayer-dielectric interaction from the
// fast-screened-coulomb library (ETM, Cavalcante PRB 97, 125427 (2018)): a
// ScreenedCoulombTerm per spec, or its exact Ewald kernel under PBC.
#include "builders/hamiltonian_builders.h"

#include <stdexcept>

#include "hamiltonian/screened_coulomb.h"

namespace alea {

// The fit bakes in both the stack ε(k) and the out-of-plane layer separation,
// so z_offset defaults to the groups' plane separation when not given
// explicitly — consistent with how Coulomb/Keldysh derive d.
std::unique_ptr<HamiltonianComponent> build_component(
    const io::ScreenedCoulombSpec& s, const HamiltonianContext& ctx) {
    const double z = s.has_z_offset
                         ? s.z_offset
                         : layer_separation(ctx.cfg, s.pair[0], s.pair[1]);
    return std::make_unique<ScreenedCoulombTerm>(
        find_group(ctx.groups, s.pair[0]), find_group(ctx.groups, s.pair[1]),
        s.epsilons, s.interfaces, s.source_layer, s.target_layer, z, s.fit_tol);
}

// A screened pair needs its OWN stack (it cannot be guessed), for a group's
// self-pair too: under PBC every charge meets its own images.  The layer offset
// defaults to the groups' plane separation.  Only an uncharged group, whose
// couplings are inert, may go without and borrows the first stack.
numerics::PotentialSpec ewald_kernel(std::span<const io::ScreenedCoulombSpec> specs,
                                     const ParticleGroup& a, const ParticleGroup& b,
                                     const HamiltonianContext& ctx) {
    const io::ScreenedCoulombSpec* ss = nullptr;
    for (const io::ScreenedCoulombSpec& s : specs)
        if ((s.pair[0] == a.name && s.pair[1] == b.name) ||
            (s.pair[0] == b.name && s.pair[1] == a.name))
            ss = &s;
    if (!ss && a.charge != 0.0 && b.charge != 0.0)
        throw std::runtime_error(
            "periodic `screened_coulomb` run with no stack for group pair (" +
            a.name + ", " + b.name + ").  Under periodic boundaries every "
            "charge interacts with every other one's periodic images, its "
            "own group's included, so each charged pair -- self-pairs too "
            "-- needs a `screened_coulomb` term giving its stack.");
    const io::ScreenedCoulombSpec& s = ss ? *ss : specs.front();
    const double z = !ss ? s.z_offset
                         : (s.has_z_offset ? s.z_offset
                                           : layer_separation(ctx.cfg, a.name, b.name));
    return build_screened_ewald_kernel(s.epsilons, s.interfaces, s.source_layer,
                                       s.target_layer, z, ctx.cell.ws_radius());
}

}  // namespace alea
