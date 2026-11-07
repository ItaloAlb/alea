// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "hamiltonian/harmonic_trap.h"

#include <algorithm>
#include <cmath>

#include "population/walker_set.h"

namespace alea {

bool HarmonicTrap::supports_open_centroid_virial() const noexcept {
    return std::all_of(half_k_.begin(), half_k_.end(),
        [](double k) { return std::isfinite(k) && k >= 0.0; }) &&
        std::all_of(center_.begin(), center_.end(),
        [](double c) { return std::isfinite(c); });
}
bool HarmonicTrap::confines_open_coordinate(std::size_t p, std::size_t d,
                                           std::size_t dim) const noexcept {
    return dim == center_.size() && d < dim && p < half_k_.size()/dim &&
           half_k_[p*dim+d] > 0.0 && supports_open_centroid_virial();
}

HarmonicTrap::HarmonicTrap(const std::vector<ParticleGroup>& groups, double omega2,
                           std::vector<double> center)
    : center_(std::move(center)) {
    std::size_t np = 0;
    for (const ParticleGroup& g : groups) np = std::max(np, g.last);
    const std::size_t dim = groups.empty() ? 0 : groups[0].mass.size();
    if (center_.empty()) center_.assign(dim, 0.0);
    half_k_.assign(np * dim, 0.0);
    for (const ParticleGroup& g : groups)
        for (std::size_t i = g.first; i < g.last; ++i)
            for (std::size_t d = 0; d < dim; ++d)
                half_k_[i * dim + d] = 0.5 * g.mass[d] * omega2;
}

void HarmonicTrap::evaluate_potential(const ParticleBatch& batch,
                                  std::span<double> v_out,
                                  std::size_t w_begin, std::size_t w_end) const {
    const std::size_t nw  = batch.num_walkers();
    const std::size_t np  = batch.n_particles();
    const std::size_t dim = batch.dim();
    if (w_end > nw) w_end = nw;
    const std::span<const double> pos = batch.positions();
    double* __restrict le = v_out.data();
    for (std::size_t i = 0; i < np; ++i)
        for (std::size_t d = 0; d < dim; ++d) {
            const double k = half_k_[i * dim + d];
            if (k == 0.0) continue;
            const double c = center_[d];
            const double* __restrict r = pos.data() + i * dim * nw + d * nw;
#pragma omp simd
            for (std::size_t w = w_begin; w < w_end; ++w)
                le[w] += k * (r[w] - c) * (r[w] - c);
        }
}

std::unique_ptr<PotentialGradientWorkspace>
HarmonicTrap::make_potential_gradient_workspace(const ParticleBatch& batch) const {
    if (center_.size() != batch.dim() || half_k_.size() != batch.n_particles()*batch.dim())
        throw std::invalid_argument("harmonic_trap: potential gradient batch shape does not match trap");
    for (double c : center_)
        if (!std::isfinite(c)) throw std::invalid_argument("harmonic_trap: nonfinite gradient center");
    for (double k : half_k_)
        if (!std::isfinite(k)) throw std::invalid_argument("harmonic_trap: nonfinite gradient curvature");
    return HamiltonianComponent::make_potential_gradient_workspace(batch);
}

void HarmonicTrap::evaluate_potential_gradient(const ParticleBatch& batch,
    std::span<double> out, PotentialGradientWorkspace& workspace,
    std::size_t w_begin, std::size_t w_end) const {
    w_end = workspace.checked_end(batch, *this, out, w_begin, w_end);
    const auto nw = batch.num_walkers(), dim = batch.dim();
    const auto positions = batch.positions(); // [np*dim*nw]
    for (std::size_t p = 0; p < batch.n_particles(); ++p)
        for (std::size_t d = 0; d < dim; ++d) {
            const double curvature = 2.0 * half_k_[p*dim+d];
            if (curvature == 0.0) continue;
            const double center = center_[d];
            const std::size_t offset = p*dim*nw+d*nw;
#pragma omp simd
            for (std::size_t w = w_begin; w < w_end; ++w)
                out[offset+w] += curvature * (positions[offset+w]-center);
        }
}

// One changed particle: O(dim), preserving wrapped one-body semantics.
double HarmonicTrap::delta_potential_particle(ParticleBatch& configs,
    std::size_t p, std::size_t k, std::span<const double> proposed,
    std::span<double>) const {
    double candidate[SimulationCell::kMaxDim];
    for (std::size_t d = 0; d < configs.dim(); ++d) candidate[d] = proposed[d];
    configs.cell().wrap(candidate);
    double result = 0.0;
    for (std::size_t d = 0; d < configs.dim(); ++d) {
        const double old = configs.positions()[p * configs.dim() * configs.num_walkers() + d * configs.num_walkers() + k] - center_[d];
        const double next = candidate[d] - center_[d];
        result += half_k_[p * configs.dim() + d] * (next*next - old*old);
    }
    return result;
}

}  // namespace alea
