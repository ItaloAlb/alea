// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// YAML action resolution and component builders; the action domain has no
// parser.
#include <utility>
#include <stdexcept>
#include "io/parser.h"
#include "pimc/action.h"
#include "pimc/fixed_node_restriction.h"
#include "pimc/primitive_action.h"
#include "pimc/pair_product_action.h"
#include "wavefunction/trial_wavefunction.h"

namespace alea {
inline std::pair<bool, bool> resolve_action_components(const io::Config& cfg,
                                                       const Hamiltonian& ham) {
    bool kinetic = cfg.action.empty(), potential = cfg.action.empty();
    for (const auto& spec : cfg.action) {
        if (spec.component == "primitive" || spec.component == "kinetic")
            kinetic = true;
        if (spec.component == "primitive" || spec.component == "potential" ||
            spec.component == "pair_product")
            potential = true;
    }
    if (!kinetic)
        throw std::runtime_error(
            "action: PIMC needs the `kinetic` component (the free-particle "
            "springs) — without it the path carries no time structure.  Use '- "
            "primitive: {}' or list both components.");
    if (!potential)
        for (const auto& component : ham.components())
            if (component->is_potential())
                throw std::runtime_error(
                    "action: the Hamiltonian has potential terms but the "
                    "action list has no `potential` component — the run would "
                    "sample a free-particle action and report it as your "
                    "system.  Add '- potential: primitive' (or '- primitive: "
                    "{}').");
    return {kinetic, potential};
}

// Each YAML item adds genuine independently cloned action components.
inline void build_action_component(const io::ActionItemSpec& spec,
                                   Action& action,
                                   std::span<const ParticleGroup> groups,
                                   const Hamiltonian& ham, std::size_t dim) {
    if (spec.component == "pair_product") {
        if (spec.form != "finite_grid")
            throw std::invalid_argument(
                "action pair_product: unsupported form " + spec.form);
        action.add_component(std::make_unique<PairProductAction>(
            groups, ham, dim, spec.extent, spec.grid_points));
        Hamiltonian one_body;
        for (const auto& component : ham.components())
            if (component->is_potential() &&
                component->pair_action_role() == PairActionRole::one_body)
                one_body.add_component(component->clone());
        action.add_component(
            std::make_unique<PrimitivePotentialAction>(groups, one_body));
        return;
    }
    if (spec.form != "primitive")
        throw std::invalid_argument("action: unsupported form " + spec.form);
    if (spec.component == "primitive" || spec.component == "kinetic")
        action.add_component(
            std::make_unique<PrimitiveKineticAction>(groups, dim));
    if (spec.component == "primitive" || spec.component == "potential")
        action.add_component(
            std::make_unique<PrimitivePotentialAction>(groups, ham));
    if (spec.component != "primitive" && spec.component != "kinetic" &&
        spec.component != "potential")
        throw std::invalid_argument("action: unknown component " +
                                    spec.component);
}
// `- nodes: wavefunction`: the fixed-node restriction takes the determinant
// blocks of the `wavefunction:` items, the same objects VMC/DMC evaluate, so a
// fixed-node DMC run of the same document sees the identical nodal surface.
// Positive factors (Jastrows) carry no nodes and are ignored.  Links are
// corrected by the image action (the bead-only check has an O(sqrt(tau)) bias).
inline std::unique_ptr<FixedNodeRestriction> build_fixed_node_restriction(
    const io::Config& cfg, const TrialWavefunction* psi,
    std::span<const ParticleGroup> groups, std::size_t dim) {
    std::size_t np = 0;
    for (const ParticleGroup& g : groups) np += g.size();
    std::vector<double> inv_mass(np * dim);  // [np*dim]
    for (const ParticleGroup& g : groups)
        for (std::size_t p = g.first; p < g.last; ++p)
            for (std::size_t d = 0; d < dim; ++d) inv_mass[p * dim + d] = g.inv_mass[d];
    if (!cfg.optimizers.empty())
        throw std::runtime_error(
            "action nodes: the restriction copies the nodes when the drivers are "
            "built, and an optimizer stage would change them afterwards.  "
            "Optimize in a separate run and freeze the nodal parameters.");
    if (!psi)
        throw std::runtime_error("action nodes: no trial wavefunction was built");
    std::vector<FixedNodeRestriction::Block> blocks;
    for (const auto& component : psi->components()) {
        if (!component->is_nodal()) continue;
        if (component->is_complex())
            throw std::runtime_error(
                "action nodes: a complex (twisted or fixed-phase) determinant has "
                "no real nodal surface; PIMC fixed-node needs real slater items");
        const auto views = component->nodal_blocks();
        if (views.empty())
            throw std::runtime_error("action nodes: wavefunction item '" +
                                     component->name() +
                                     "' is nodal but does not expose its "
                                     "determinant blocks");
        for (const NodalBlockView& view : views)
            blocks.push_back({view.first, view.last, view.orbitals->clone()});
    }
    if (blocks.empty())
        throw std::runtime_error(
            "action nodes: the wavefunction has no slater item, so there are no "
            "nodes to restrict the paths to");
    return std::make_unique<FixedNodeRestriction>(std::move(blocks), np, dim,
                                                  std::move(inv_mass),
                                                  FixedNodeRestriction::NodeCrossing::image);
}

// psi: the run's trial wavefunction, read only by a `nodes` item.
inline Action build_action(const io::Config& cfg,
                           std::span<const ParticleGroup> groups,
                           const Hamiltonian& ham, std::size_t dim,
                           const TrialWavefunction* psi = nullptr) {
    (void)resolve_action_components(cfg, ham);
    Action action(groups, ham, dim, false, false);
    if (cfg.action.empty()) {
        build_action_component(io::ActionItemSpec{"primitive", "primitive"},
                               action, groups, ham, dim);
        return action;
    }
    for (const auto& spec : cfg.action) {
        if (spec.component == "nodes")
            action.add_component(build_fixed_node_restriction(cfg, psi, groups, dim));
        else
            build_action_component(spec, action, groups, ham, dim);
    }
    return action;
}
} // namespace alea
