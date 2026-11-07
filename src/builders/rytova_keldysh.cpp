// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// `type: rytova_keldysh` → the thin-film screened V_RK(√(r² + d²)) (Rytova 1967,
// Keldysh 1979): a KeldyshTerm per spec, or its Ewald kernel under PBC.
#include "builders/hamiltonian_builders.h"

#include "hamiltonian/rytova_keldysh.h"

namespace alea {

std::unique_ptr<HamiltonianComponent> build_component(
    const io::KeldyshSpec& k, const HamiltonianContext& ctx) {
    return std::make_unique<KeldyshTerm>(
        find_group(ctx.groups, k.pair[0]), find_group(ctx.groups, k.pair[1]),
        k.epsilon, k.r0, layer_separation(ctx.cfg, k.pair[0], k.pair[1]));
}

// A pair takes (ε, r0) from the spec that names it, else from the first spec
// (the single-potential behaviour).  The ε must agree across specs — the Ewald
// term refuses otherwise, since the kernels share one long-range 1/(εk).
numerics::PotentialSpec ewald_kernel(std::span<const io::KeldyshSpec> specs,
                                     const ParticleGroup& a, const ParticleGroup& b,
                                     const HamiltonianContext& ctx) {
    const io::KeldyshSpec* ks = &specs.front();
    for (const io::KeldyshSpec& k : specs)
        if ((k.pair[0] == a.name && k.pair[1] == b.name) ||
            (k.pair[0] == b.name && k.pair[1] == a.name))
            ks = &k;
    return numerics::PotentialSpec::rytova_keldysh(
        ks->epsilon, ks->r0, layer_separation(ctx.cfg, a.name, b.name));
}

}  // namespace alea
