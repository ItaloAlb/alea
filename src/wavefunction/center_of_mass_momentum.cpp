// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "wavefunction/center_of_mass_momentum.h"

#include <algorithm>
#include <stdexcept>

#include "population/walker_set.h"

namespace alea {

CenterOfMassMomentum::CenterOfMassMomentum(const std::vector<ParticleGroup>& groups,
                                           std::vector<double> K)
    : dim_(K.size()), K_(std::move(K)) {
    std::size_t np = 0;
    for (const ParticleGroup& g : groups) np = std::max(np, g.last);

    std::vector<double> total_mass(dim_, 0.0);
    for (const ParticleGroup& g : groups) {
        if (g.mass.size() != dim_)
            throw std::invalid_argument(
                "center_of_mass_momentum: K has " + std::to_string(dim_) +
                " components but group '" + g.name + "' has a " +
                std::to_string(g.mass.size()) + "-dimensional mass");
        for (std::size_t d = 0; d < dim_; ++d)
            total_mass[d] += static_cast<double>(g.size()) * g.mass[d];
    }

    k_.assign(np * dim_, 0.0);
    for (const ParticleGroup& g : groups)
        for (std::size_t i = g.first; i < g.last; ++i)
            for (std::size_t d = 0; d < dim_; ++d)
                k_[i * dim_ + d] = g.mass[d] * K_[d] / total_mass[d];
}

void CenterOfMassMomentum::evaluate_batch(WalkerSet& batch) {
    const std::size_t nw  = batch.num_walkers();
    const std::size_t np  = batch.n_particles();
    const std::size_t dim = batch.dim();
    prepare(nw, np, dim);

    // ∇_i φ = k_i everywhere; ∂²φ = 0 adds nothing.
    std::span<double> pg = batch.phase_gradients();
    for (std::size_t p = 0; p < np; ++p)
        for (std::size_t d = 0; d < dim; ++d) {
            const double k = k_[p * dim + d];
            double* __restrict g = pg.data() + p * dim * nw + d * nw;
            for (std::size_t w = 0; w < nw; ++w) g[w] += k;
        }
}

void CenterOfMassMomentum::ratio_batch(WalkerSet& batch, std::size_t p,
                                       std::size_t w_begin, std::size_t w_end) {
    const std::size_t nw  = batch.num_walkers();
    const std::size_t dim = batch.dim();
    if (w_end > nw) w_end = nw;

    // φ is linear, so arg(ψ'/ψ) = k_p·Δr exactly (the raw, unwrapped Δr) and
    // the proposed ∇φ is still k_p.  |ψ'/ψ| = 1: log_ratios is untouched.
    const std::span<const double> pos     = batch.positions();
    const std::span<const double> new_pos = batch.new_pos();
    std::span<double> phase = batch.phase_diffs();
    std::span<double> npg   = batch.new_phase_grad();
    for (std::size_t d = 0; d < dim; ++d) {
        const double k = k_[p * dim + d];
        const double* __restrict r0 = pos.data() + p * dim * nw + d * nw;
        const double* __restrict r1 = new_pos.data() + d * nw;
        for (std::size_t w = w_begin; w < w_end; ++w) {
            phase[w] += k * (r1[w] - r0[w]);
            npg[d * nw + w] += k;
        }
    }
}

void CenterOfMassMomentum::apply_move_batch(WalkerSet& batch, std::size_t p,
                                            std::span<const bool> accept_mask,
                                            std::size_t w_begin,
                                            std::size_t w_end) {
    // Nothing to commit: ∇φ = k_i is configuration-independent, so no partner
    // particle changes, and WalkerSet::apply_move writes the moved particle's
    // plane from new_phase_grad (= k_p).
    (void)batch; (void)p; (void)accept_mask; (void)w_begin; (void)w_end;
}

}  // namespace alea
