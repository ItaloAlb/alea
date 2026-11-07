// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// `- center_of_mass_momentum:` → the Galilean boost ψ → e^{iK·R_cm}ψ.  A pure
// phase: it makes the trial complex, so DMC runs fixed phase.
#include "builders/wavefunction_builders.h"

#include <stdexcept>

#include "wavefunction/center_of_mass_momentum.h"

namespace alea {

std::unique_ptr<WavefunctionComponent> build_component(
    const io::CenterOfMassMomentumSpec& boost, const WavefunctionContext& ctx) {
    const std::vector<double>& K = boost.K;
    if (K.size() != ctx.dim)
        throw std::runtime_error(
            "center_of_mass_momentum: K has " + std::to_string(K.size()) +
            " components, the system is " + std::to_string(ctx.dim) + "-dimensional");
    if (ctx.cfg.lattice.type == "periodic")
        throw std::runtime_error(
            "center_of_mass_momentum is not supported on a periodic cell: "
            "e^{ik_i·r_i} is single-valued there only for k_i on the "
            "reciprocal lattice (that is the twist machinery's job).");
    return std::make_unique<CenterOfMassMomentum>(ctx.groups, K);
}

}  // namespace alea
