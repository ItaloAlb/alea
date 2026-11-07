// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <cmath>
#include <functional>
#include <span>
#include <stdexcept>

#include "population/particle_batch.h"

namespace alea {
class HamiltonianComponent;

inline bool potential_gradient_output_valid(const ParticleBatch& batch,
                                            std::span<double> out) noexcept {
    const auto positions = batch.positions();
    if (out.size() != positions.size()) return false;
    if (out.empty()) return true;
    // std::less supplies a total pointer order even for distinct allocations.
    // Refuse aliasing coordinates before the composite zeros its output.
    const std::less<const double*> less;
    return !(less(out.data(), positions.data() + positions.size()) &&
             less(positions.data(), out.data() + out.size()));
}

// Caller-owned scratch for G = grad V, in energy/length units. Gradient output
// has the coordinate SoA layout [p*dim*nw + d*nw + w]. Never use log(psi)'s
// gradient storage: V and the trial wavefunction have different derivatives.
// Analytic components need only this binding; later components may derive a
// workspace with buffers allocated at setup. One workspace per concurrent call.
class PotentialGradientWorkspace {
    friend class HamiltonianComponent;
protected:
    PotentialGradientWorkspace(const ParticleBatch& batch,
                               const HamiltonianComponent& component)
        : component_(&component), np_(batch.n_particles()), dim_(batch.dim()),
          nw_(batch.num_walkers()) {
        if (!np_ || !nw_ || !dim_ || dim_ > SimulationCell::kMaxDim)
            throw std::invalid_argument("potential gradient: invalid batch dimensions");
        for (double x : batch.positions())
            if (!std::isfinite(x))
                throw std::invalid_argument("potential gradient: nonfinite initial coordinates");
    }
public:
    virtual ~PotentialGradientWorkspace() = default;

    // Component-specific scratch may also bind geometry. The composite checks
    // this virtual predicate before zeroing output, including changed cells.
    virtual bool matches(const ParticleBatch& batch,
                 const HamiltonianComponent& component) const noexcept {
        return component_ == &component && np_ == batch.n_particles() &&
               dim_ == batch.dim() && nw_ == batch.num_walkers();
    }

    // Internal identity for typed scratch, checked before a component casts it.
    // A qualified base factory cannot masquerade as specialized workspace.
    virtual const void* implementation_tag() const noexcept { return nullptr; }

    // Validate before touching output, including the global SoA stride. Clamp
    // npos/oversized end exactly as evaluate_potential does; a reversed range
    // or a changed batch shape requires a new prepared workspace.
    std::size_t checked_end(const ParticleBatch& batch,
        const HamiltonianComponent& component, std::span<double> out,
        std::size_t begin, std::size_t end) const {
        if (end > nw_) end = nw_;
        if (!matches(batch, component) || !potential_gradient_output_valid(batch, out) ||
            begin > end)
            throw std::invalid_argument("potential gradient: incompatible workspace, output or range");
        return end;
    }

private:
    const HamiltonianComponent* component_;
    std::size_t np_, dim_, nw_;
};
} // namespace alea
