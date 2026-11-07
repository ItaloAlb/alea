// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// `type: moire` → the one-body moiré potential V(r) = V0 + 2Ṽ Σ_{i=1}^3
// cos(g_i·r + φ) on every particle of the listed groups.  The three g_i are the
// first shell of a triangular superlattice of period a_M, oriented by theta0.
#include "builders/hamiltonian_builders.h"

#include <utility>

#include "hamiltonian/moire.h"

namespace alea {

std::unique_ptr<HamiltonianComponent> build_component(
    const io::MoireSpec& m, const HamiltonianContext& ctx) {
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    ranges.reserve(m.groups.size());
    for (const std::string& gname : m.groups) {
        const ParticleGroup& g = find_group(ctx.groups, gname);
        ranges.emplace_back(g.first, g.last);
    }
    return std::make_unique<MoireTerm>(
        ctx.dim, m.V0, m.Vtilde, m.phi,
        moire_reciprocal_vectors(m.a_M, m.theta0, ctx.dim), std::move(ranges));
}

}  // namespace alea
