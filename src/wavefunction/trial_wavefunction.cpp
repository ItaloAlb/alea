// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "wavefunction/trial_wavefunction.h"

#include <algorithm>

#include "population/walker_set.h"

namespace alea {

void TrialWavefunction::add_component(
    std::unique_ptr<WavefunctionComponent> component) {
    components_.push_back(std::move(component));
}

void TrialWavefunction::evaluate_batch(WalkerSet& batch) {
    std::span<double> log_psi = batch.log_psi();
    std::span<double> grad    = batch.gradients();
    std::span<double> lap     = batch.laplacians();
    std::fill(log_psi.begin(), log_psi.end(), 0.0);
    std::fill(grad.begin(), grad.end(), 0.0);
    std::fill(lap.begin(), lap.end(), 0.0);
    if (is_complex()) {
        batch.enable_phase();
        std::span<double> pg = batch.phase_gradients();
        std::span<double> pl = batch.phase_laplacians();
        std::fill(pg.begin(), pg.end(), 0.0);
        std::fill(pl.begin(), pl.end(), 0.0);
    }

    for (const auto& c : components_) c->evaluate_batch(batch);
}

bool TrialWavefunction::is_complex() const {
    return std::any_of(components_.begin(), components_.end(),
                       [](const auto& c) { return c->is_complex(); });
}

void TrialWavefunction::prepare(std::size_t nw, std::size_t np, std::size_t dim) {
    for (const auto& c : components_) c->prepare(nw, np, dim);
}

void TrialWavefunction::ratio_batch(WalkerSet& batch, std::size_t p,
                                    std::size_t w_begin, std::size_t w_end) {
    const std::size_t nw  = batch.num_walkers();
    const std::size_t dim = batch.dim();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end) return;

    // Zero the active range of the proposed-move accumulators.
    std::span<double> new_grad   = batch.new_grad();    // [dim*nw]
    std::span<double> new_lap    = batch.new_lap();     // [dim*nw]
    std::span<double> log_ratios = batch.log_ratios();  // [nw]
    std::span<double> phase_diff = batch.phase_diffs(); // [nw]
    for (std::size_t d = 0; d < dim; ++d)
        for (std::size_t w = w_begin; w < w_end; ++w) {
            new_grad[d * nw + w] = 0.0;
            new_lap[d * nw + w]  = 0.0;
        }
    for (std::size_t w = w_begin; w < w_end; ++w) {
        log_ratios[w] = 0.0;
        phase_diff[w] = 0.0;  // fixed-node phase; sign-changing components add π
    }
    if (batch.has_phase()) {
        std::span<double> npg = batch.new_phase_grad();
        std::span<double> npl = batch.new_phase_lap();
        for (std::size_t d = 0; d < dim; ++d)
            for (std::size_t w = w_begin; w < w_end; ++w) {
                npg[d * nw + w] = 0.0;
                npl[d * nw + w] = 0.0;
            }
    }

    for (const auto& c : components_) c->ratio_batch(batch, p, w_begin, w_end);
}

void TrialWavefunction::apply_move_batch(WalkerSet& batch, std::size_t p,
                                         std::span<const bool> accept_mask,
                                         std::size_t w_begin, std::size_t w_end) {
    for (const auto& c : components_)
        c->apply_move_batch(batch, p, accept_mask, w_begin, w_end);
}

void TrialWavefunction::apply_move_batch_deferred(WalkerSet& batch, std::size_t p,
    std::span<const bool> accept, std::size_t w_begin, std::size_t w_end) {
    for (const auto& c : components_)
        c->apply_move_batch_deferred(batch, p, accept, w_begin, w_end);
}

void TrialWavefunction::refresh_particle_derivatives(WalkerSet& batch, std::size_t p,
    std::size_t w_begin, std::size_t w_end) {
    for (const auto& c : components_) c->refresh_particle_derivatives(batch, p, w_begin, w_end);
}

void TrialWavefunction::refresh_derivatives(WalkerSet& batch, std::size_t w_begin,
    std::size_t w_end) {
    for (const auto& c : components_) c->refresh_derivatives(batch, w_begin, w_end);
}

std::size_t TrialWavefunction::walker_state_bytes() const noexcept {
    std::size_t bytes = 0;
    for (const auto& c : components_) bytes += c->walker_state_bytes();
    return bytes;
}

void TrialWavefunction::shifted_pair_ratios_batch(const WalkerSet& batch, std::size_t a_first,
                                                  std::size_t a_last, std::size_t b_first,
                                                  std::size_t b_last,
                                                  std::span<const double> shift,
                                                  std::span<double> log_ratios,
                                                  std::span<double> phases) {
    std::fill(log_ratios.begin(), log_ratios.end(), 0.0);
    std::fill(phases.begin(), phases.end(), 0.0);
    for (const auto& c : components_)
        c->shifted_pair_ratios_batch(batch, a_first, a_last, b_first, b_last, shift, log_ratios,
                                     phases);
}

std::vector<std::string> TrialWavefunction::components_without_shifted_pairs() const {
    std::vector<std::string> names;
    for (const auto& c : components_)
        if (!c->supports_shifted_pairs()) names.push_back(c->name());
    return names;
}

TrialWavefunction TrialWavefunction::clone() const {
    TrialWavefunction copy;
    copy.components_.reserve(components_.size());
    for (const auto& c : components_) copy.components_.push_back(c->clone());
    return copy;
}

void TrialWavefunction::permute_caches(std::span<const std::size_t> src) {
    for (const auto& c : components_) c->permute_caches(src);
}

void TrialWavefunction::rebuild() {
    for (const auto& c : components_) c->rebuild();
}

std::size_t TrialWavefunction::total_optimizable_params() const {
    std::size_t n = 0;
    for (const auto& c : components_) n += c->num_optimizable_params();
    return n;
}

void TrialWavefunction::compute_parameter_derivatives(const WalkerSet& batch,
                                                       std::span<double> derivs) const {
    std::size_t offset = 0;
    for (const auto& c : components_) {
        const std::size_t n = c->num_optimizable_params();
        if (n == 0) continue;
        const std::size_t nw = batch.num_walkers();
        c->compute_parameter_derivatives(batch, derivs.subspan(offset, n * nw));
        offset += n * nw;
    }
}

bool TrialWavefunction::compute_parameter_gradient_derivatives(
    const WalkerSet& batch, std::span<double> dgrad, std::span<double> dlap,
    std::span<char> supported) const {
    const std::size_t block = batch.n_particles() * batch.dim() *
                              batch.num_walkers();
    std::fill(dgrad.begin(), dgrad.end(), 0.0);
    std::fill(dlap.begin(), dlap.end(), 0.0);
    std::fill(supported.begin(), supported.end(), 0);

    std::size_t offset = 0;
    bool any = false;
    for (const auto& c : components_) {
        const std::size_t n = c->num_optimizable_params();
        if (n == 0) continue;
        const bool ok = c->compute_parameter_gradient_derivatives(
            batch, dgrad.subspan(offset * block, n * block),
            dlap.subspan(offset * block, n * block));
        for (std::size_t k = 0; k < n; ++k)
            supported[offset + k] = ok ? 1 : 0;
        any = any || ok;
        offset += n;
    }
    return any;
}

}  // namespace alea
