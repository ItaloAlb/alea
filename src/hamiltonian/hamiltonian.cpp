// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "hamiltonian/hamiltonian.h"

#include <algorithm>
#include <cmath>

#include "population/walker_set.h"

namespace alea {

void Hamiltonian::add_component(std::unique_ptr<HamiltonianComponent> component) {
    components_.push_back(std::move(component));
}

void Hamiltonian::evaluate_batch(WalkerSet& batch, std::size_t w_begin,
                                 std::size_t w_end) {
    const std::size_t nw = batch.num_walkers();
    if (w_end > nw) w_end = nw;
    std::span<double> le = batch.local_energies();
    for (std::size_t w = w_begin; w < w_end; ++w) le[w] = 0.0;
    if (batch.has_phase()) {
        std::span<double> im = batch.local_energies_imag();
        for (std::size_t w = w_begin; w < w_end; ++w) im[w] = 0.0;
    }
    for (const auto& c : components_) c->evaluate_batch(batch, w_begin, w_end);
}

void Hamiltonian::evaluate_potential(const ParticleBatch& configs,
                                     std::span<double> v_out,
                                     std::size_t w_begin, std::size_t w_end) const {
    const std::size_t nw = configs.num_walkers();
    if (w_end > nw) w_end = nw;
    for (std::size_t w = w_begin; w < w_end; ++w) v_out[w] = 0.0;
    for (const auto& c : components_)
        if (c->is_potential()) c->evaluate_potential(configs, v_out, w_begin, w_end);
}

bool Hamiltonian::supports_potential_gradient() const noexcept {
    for (const auto& c : components_)
        if (c->is_potential() && !c->supports_potential_gradient()) return false;
    return true;
}

HamiltonianGradientWorkspace Hamiltonian::make_potential_gradient_workspace(
    const ParticleBatch& configs) const {
    HamiltonianGradientWorkspace workspace;
    workspace.owner_ = this;
    workspace.np_ = configs.n_particles();
    workspace.dim_ = configs.dim();
    workspace.nw_ = configs.num_walkers();
    workspace.components_.reserve(components_.size());
    for (const auto& c : components_) {
        if (c->is_potential()) {
            if (!c->supports_potential_gradient())
                throw std::logic_error(c->name() + ": potential gradient is not supported");
            auto scratch = c->make_potential_gradient_workspace(configs);
            if (!scratch || !scratch->matches(configs, *c))
                throw std::logic_error(c->name() + ": invalid potential gradient workspace");
            workspace.components_.push_back(std::move(scratch));
        } else {
            workspace.components_.push_back(nullptr);
        }
    }
    return workspace;
}

void Hamiltonian::evaluate_potential_gradient(const ParticleBatch& configs,
    std::span<double> out, HamiltonianGradientWorkspace& workspace,
    std::size_t w_begin, std::size_t w_end) const {
    const auto nw = configs.num_walkers(), np = configs.n_particles(), dim = configs.dim();
    if (w_end > nw) w_end = nw;
    if (workspace.owner_ != this || workspace.np_ != np || workspace.dim_ != dim ||
        workspace.nw_ != nw || workspace.components_.size() != components_.size() ||
        !potential_gradient_output_valid(configs, out) || w_begin > w_end)
        throw std::invalid_argument("potential gradient: incompatible Hamiltonian workspace, output or range");
    // Verify the entire composition before zeroing any output; moving/replacing
    // a Hamiltonian must not rebind old scratch to a different component.
    for (std::size_t c = 0; c < components_.size(); ++c) {
        const auto& scratch = workspace.components_[c];
        if (components_[c]->is_potential()) {
            if (!scratch || !scratch->matches(configs, *components_[c]))
                throw std::invalid_argument("potential gradient: component workspace changed");
        } else if (scratch) {
            throw std::invalid_argument("potential gradient: nonpotential component workspace");
        }
    }
    for (std::size_t p = 0; p < np; ++p)
        for (std::size_t d = 0; d < dim; ++d)
            for (std::size_t w = w_begin; w < w_end; ++w)
                out[p * dim * nw + d * nw + w] = 0.0;
    for (std::size_t c = 0; c < components_.size(); ++c)
        if (components_[c]->is_potential())
            components_[c]->evaluate_potential_gradient(
                configs, out, *workspace.components_[c], w_begin, w_end);
    for (std::size_t p = 0; p < np; ++p)
        for (std::size_t d = 0; d < dim; ++d)
            for (std::size_t w = w_begin; w < w_end; ++w)
                if (!std::isfinite(out[(p*dim+d)*nw+w]))
                    throw std::domain_error("potential gradient: nonfinite accumulated derivative");
}

double Hamiltonian::delta_potential_particle(ParticleBatch& configs,
    std::size_t p, std::size_t k, std::span<const double> proposed,
    std::span<double> scratch) const {
    if (p >= configs.n_particles() || k >= configs.num_walkers() ||
        configs.dim() > SimulationCell::kMaxDim || proposed.size() != configs.dim() ||
        scratch.size() < configs.num_walkers())
        throw std::invalid_argument("potential delta: invalid particle, slice or workspace");
    double delta = 0.0;
    for (const auto& component : components_)
        if (component->is_potential())
            delta += component->delta_potential_particle(configs, p, k, proposed, scratch);
    return delta;
}

bool Hamiltonian::translation_invariant() const noexcept {
    for (const auto& component : components_)
        if (component->is_potential() && !component->is_translation_invariant()) return false;
    return true;
}

void Hamiltonian::psi_dependent_energy(const WalkerSet& batch,
                                       std::span<double> out) const {
    std::fill(out.begin(), out.end(), 0.0);
    for (const auto& c : components_) c->accumulate_psi_dependent_energy(batch, out);
}

Hamiltonian Hamiltonian::clone() const {
    Hamiltonian copy;
    copy.components_.reserve(components_.size());
    for (const auto& c : components_) copy.components_.push_back(c->clone());
    return copy;
}

void Hamiltonian::local_energy_parameter_derivatives(
    const WalkerSet& batch, std::span<const double> dgrad,
    std::span<const double> dlap, std::size_t n_opt,
    std::span<double> out) const {
    std::fill(out.begin(), out.end(), 0.0);
    for (const auto& c : components_)
        c->local_energy_parameter_derivatives(batch, dgrad, dlap, n_opt, out);
}

}  // namespace alea
