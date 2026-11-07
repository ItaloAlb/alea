// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// `- nosanow:` items → NosanowJastrow, the one-body localization factor of a
// crystal: log ψ = −α Σ_i |r_i − R_i|², particle i of the group on site i.
#include "builders/wavefunction_builders.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "wavefunction/nosanow_jastrow.h"

namespace alea {

std::unique_ptr<WavefunctionComponent> build_component(const io::NosanowSpec& spec,
                                                       const WavefunctionContext& ctx) {
    const std::string label = "nosanow on group " + spec.group;
    const ParticleGroup& g = find_group(ctx.groups, spec.group);
    if (g.size() == 0) throw std::runtime_error(label + ": the group has no particles");
    std::vector<double> sites =
        lattice_sites(spec.sites, spec.centers, g, ctx.cell, ctx.dim, label);
    // α > 0 is kept by the sign lock (α = e^θ) while optimizing; the component
    // refuses an α whose minimum-image fold is not negligible.
    std::vector<Parameter> params{
        Parameter{"alpha", spec.alpha, spec.alpha_fixed, /*sign_locked=*/true}};
    try {
        return std::make_unique<NosanowJastrow>(ctx.dim, g.first, std::move(sites),
                                                ctx.cell, std::move(params));
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error(label + ": " + e.what());
    }
}

}  // namespace alea
