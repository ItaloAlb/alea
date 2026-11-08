// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <span>
#include <vector>

#include "hamiltonian_component.h"

namespace alea {

class WalkerSet;
class ParticleBatch;
class Hamiltonian;

// Stable component-order scratch, bound to the Hamiltonian that prepared it.
// Adding/replacing/moving its components requires preparing a new workspace.
class HamiltonianGradientWorkspace {
    friend class Hamiltonian;
    const Hamiltonian* owner_ = nullptr;
    std::size_t np_ = 0, dim_ = 0, nw_ = 0;
    std::vector<std::unique_ptr<PotentialGradientWorkspace>> components_; // [component]
};

class Hamiltonian {
public:
    void add_component(std::unique_ptr<HamiltonianComponent> component);

    // Evaluate local energy over the walker range [w_begin, w_end) (default =
    // whole batch): zeros that range of batch.local_energies(), then
    // accumulates each component's contribution.  Stride is the global nw.
    void evaluate_batch(WalkerSet& batch,
                        std::size_t w_begin = 0,
                        std::size_t w_end = static_cast<std::size_t>(-1));

    // Total potential V(R_w) of the configurations [w_begin, w_end) into
    // v_out [nw]: zeros that range, then accumulates every component with
    // is_potential() — the kinetic term is skipped.  For drivers that weight
    // configurations by V alone (PIMC's potential action).  Read-only on the
    // batch: safe to call concurrently on distinct batches.
    void evaluate_potential(const ParticleBatch& configs, std::span<double> v_out,
                            std::size_t w_begin = 0,
                            std::size_t w_end = static_cast<std::size_t>(-1)) const;

    [[nodiscard]] bool supports_potential_gradient() const noexcept;
    [[nodiscard]] HamiltonianGradientWorkspace
    make_potential_gradient_workspace(const ParticleBatch& configs) const;
    // Zero only the requested SoA columns, then accumulate G=grad V in the
    // original component order. Skip nonpotential terms; no RNG/state changes.
    void evaluate_potential_gradient(const ParticleBatch& configs,
        std::span<double> gradient_out, HamiltonianGradientWorkspace& workspace,
        std::size_t w_begin = 0,
        std::size_t w_end = static_cast<std::size_t>(-1)) const;

    // Sum potential deltas; scratch must be prepared to num_walkers().
    double delta_potential_particle(ParticleBatch& configs, std::size_t p,
        std::size_t k, std::span<const double> proposed,
        std::span<double> scratch) const;
    [[nodiscard]] bool translation_invariant() const noexcept;

    // Accumulate ∂E_L/∂p_k into `out` [n_opt * nw]; zeros it first.
    void local_energy_parameter_derivatives(const WalkerSet& batch,
                                            std::span<const double> dgrad,
                                            std::span<const double> dlap,
                                            std::size_t n_opt,
                                            std::span<double> out) const;
    // Complex trial: Re and Im of ∂E_L/∂p_k into out_re / out_im [n_opt * nw]
    // (zeroed first), from both parts of the wavefunction's derivatives (see
    // HamiltonianComponent::complex_local_energy_parameter_derivatives).
    void complex_local_energy_parameter_derivatives(
        const WalkerSet& batch, std::span<const double> dgrad, std::span<const double> dlap,
        std::span<const double> dpgrad, std::span<const double> dplap, std::size_t n_opt,
        std::span<double> out_re, std::span<double> out_im) const;

    // Wavefunction-dependent part of E_L (the kinetic energy) into `out` [nw];
    // zeros it first.  Read-only: safe to call concurrently on distinct batches.
    void psi_dependent_energy(const WalkerSet& batch, std::span<double> out) const;

    // Produce an independent copy for use in a separate thread.
    [[nodiscard]] Hamiltonian clone() const;

    std::span<const std::unique_ptr<HamiltonianComponent>> components() const {
        return components_;
    }

private:
    std::vector<std::unique_ptr<HamiltonianComponent>> components_;
};

}  // namespace alea
