// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// `type: inverse_power` → V = C / r^p between two groups (p = 3: dipoles
// polarized perpendicular to the plane).  One InversePowerTerm per spec in open
// AND periodic cells: for p > d the image sum converges absolutely, so the term
// carries its own periodic form (minimum image over the Wigner–Seitz cell plus
// a continuum tail outside it) and is never part of the Ewald sum.
#include "builders/hamiltonian_builders.h"

#include <stdexcept>
#include <string>

#include "hamiltonian/inverse_power.h"
#include "io/logger.h"

namespace alea {

std::unique_ptr<HamiltonianComponent> build_component(
    const io::InversePowerSpec& s, const HamiltonianContext& ctx) {
    const ParticleGroup& a = find_group(ctx.groups, s.pair[0]);
    const ParticleGroup& b = find_group(ctx.groups, s.pair[1]);
    // A layer offset would make the pair interact through √(r² + d²), which
    // the term does not implement; refuse rather than drop the offset.
    if (layer_separation(ctx.cfg, s.pair[0], s.pair[1]) != 0.0)
        throw std::runtime_error(
            "inverse_power between groups in different planes (" + s.pair[0] +
            ", " + s.pair[1] + "): an out-of-plane offset is not implemented "
            "for this term; put both groups in the same plane (equal `z`).");
    auto term = std::make_unique<InversePowerTerm>(a, b, s.strength, s.exponent,
                                                   ctx.cell);
    if (ctx.cell.is_periodic())
        io::Logger::info(
            "inverse_power [" + s.pair[0] + ", " + s.pair[1] + "]: pairs at their "
            "minimum image; continuum tail outside the Wigner-Seitz cell E_tail = " +
            std::to_string(term->tail_energy()) +
            " per cell, included in the local energy.");
    return term;
}

}  // namespace alea
