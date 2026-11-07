// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "wavefunction/nosanow_jastrow.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "population/walker_set.h"

namespace alea {

NosanowJastrow::NosanowJastrow(std::size_t dim, std::size_t first,
                               std::vector<double> sites, const SimulationCell& cell,
                               std::vector<Parameter> params)
    : dim_(dim), first_(first), sites_(std::move(sites)), cell_(cell),
      params_(std::move(params)) {
    if (dim_ == 0 || sites_.empty() || sites_.size() % dim_ != 0)
        throw std::invalid_argument("nosanow: sites must be a non-empty [n * dim] list");
    if (params_.size() != 1)
        throw std::invalid_argument("nosanow: expects one parameter, the width alpha; "
                                    "got " + std::to_string(params_.size()));
    check_alpha(params_[0].value);
    // One-body: s_p = −α|Δ_p|², so a move needs the proposed position only.
    enable_particle_values();
}

void NosanowJastrow::check_alpha(double alpha) const {
    if (!(std::isfinite(alpha) && alpha > 0.0))
        throw std::invalid_argument("nosanow: the width alpha must be finite and > 0; "
                                    "got " + std::to_string(alpha));
    if (!cell_.is_periodic()) return;
    const double rws = cell_.ws_radius();
    const double tail = std::exp(-alpha * rws * rws);
    if (tail > kMaxFoldedTail)
        throw std::invalid_argument(
            "nosanow: alpha = " + std::to_string(alpha) + " leaves exp(-alpha R_WS^2) = " +
            std::to_string(tail) + " of the Gaussian at the Wigner-Seitz boundary "
            "(R_WS = " + std::to_string(rws) + "), where its minimum-image gradient "
            "flips; at most " + std::to_string(kMaxFoldedTail) + " is accepted.  "
            "Localize more (larger alpha) or use a larger cell.");
}

void NosanowJastrow::rebuild() {
    // An optimizer step, not a config error: reported like the other
    // components' rebuild() refusals.
    try {
        check_alpha(params_[0].value);
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error(std::string(e.what()) + " (after an optimizer step)");
    }
}

void NosanowJastrow::displacement(const double* x, std::size_t site, std::size_t nw,
                                  std::size_t w0, std::size_t w1,
                                  double* del) const noexcept {
    const double* c = sites_.data() + site * dim_;
    if (cell_.is_periodic()) {
        cell_.min_image_batch_from(x, c, nw, del, w0, w1);
        return;
    }
    for (std::size_t d = 0; d < dim_; ++d) {
        const double cd = c[d];
#pragma omp simd
        for (std::size_t w = w0; w < w1; ++w) del[d * nw + w] = x[d * nw + w] - cd;
    }
}

void NosanowJastrow::particle_terms(const double* x, std::size_t site, std::size_t nw,
                                    std::size_t w0, std::size_t w1, double* u,
                                    double* grad, double* lap) const noexcept {
    const double alpha = params_[0].value;
    displacement(x, site, nw, w0, w1, grad);  // grad holds Δ for now
    for (std::size_t w = w0; w < w1; ++w) u[w] = 0.0;
    for (std::size_t d = 0; d < dim_; ++d) {
        const double* __restrict del = grad + d * nw;
#pragma omp simd
        for (std::size_t w = w0; w < w1; ++w) u[w] -= alpha * del[w] * del[w];
    }
    for (std::size_t d = 0; d < dim_; ++d) {
        double* __restrict g = grad + d * nw;
        double* __restrict l = lap + d * nw;
#pragma omp simd
        for (std::size_t w = w0; w < w1; ++w) {
            g[w] = -2.0 * alpha * g[w];
            l[w] = -2.0 * alpha;
        }
    }
}

void NosanowJastrow::evaluate_batch(WalkerSet& batch) {
    const std::size_t nw  = batch.num_walkers();
    const std::size_t np  = batch.n_particles();
    const std::size_t dim = batch.dim();
    prepare(nw, np, dim);

    const double* pos = batch.positions().data();
    for (std::size_t k = 0; k < n_sites(); ++k) {
        const std::size_t p = first_ + k;
        double* s_p = value_c_.data() + p * nw;  // −α|Δ_p|², the share of p
        particle_terms(pos + p * dim * nw, k, nw, 0, nw, s_p,
                       grad_c_.data() + p * dim * nw, lap_c_.data() + p * dim * nw);
        for (std::size_t w = 0; w < nw; ++w) log_psi_c_[w] += s_p[w];
    }

    std::span<double> log_psi = batch.log_psi();
    for (std::size_t w = 0; w < nw; ++w) log_psi[w] += log_psi_c_[w];
    std::span<double> grad = batch.gradients();
    for (std::size_t k = 0; k < grad.size(); ++k) grad[k] += grad_c_[k];
    std::span<double> lap = batch.laplacians();
    for (std::size_t k = 0; k < lap.size(); ++k) lap[k] += lap_c_[k];
}

void NosanowJastrow::ratio_batch(WalkerSet& batch, std::size_t p,
                                 std::size_t w_begin, std::size_t w_end) {
    if (!acts_on(p)) return;
    const std::size_t nw  = batch.num_walkers();
    const std::size_t dim = batch.dim();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end) return;
    if (value_c_.size() < (p + 1) * nw)
        throw std::logic_error("nosanow: ratio_batch before evaluate_batch");

    // The old share is committed in value_c_; only r_p' is evaluated.
    particle_terms(batch.new_pos().data(), p - first_, nw, w_begin, w_end,
                   pending_value_c_.data(), pending_grad_c_.data(),
                   pending_lap_c_.data());

    const double* s_p = value_c_.data() + p * nw;
    std::span<double> lr = batch.log_ratios();
    for (std::size_t w = w_begin; w < w_end; ++w) lr[w] += pending_value_c_[w] - s_p[w];
    std::span<double> ng = batch.new_grad();
    std::span<double> nl = batch.new_lap();
    for (std::size_t d = 0; d < dim; ++d)
        for (std::size_t w = w_begin; w < w_end; ++w) {
            ng[d * nw + w] += pending_grad_c_[d * nw + w];
            nl[d * nw + w] += pending_lap_c_[d * nw + w];
        }
}

void NosanowJastrow::apply_move_batch(WalkerSet& batch, std::size_t p,
                                      std::span<const bool> accept,
                                      std::size_t w_begin, std::size_t w_end) {
    if (!acts_on(p)) return;
    const std::size_t nw  = batch.num_walkers();
    const std::size_t dim = batch.dim();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end) return;
    if (value_c_.size() < (p + 1) * nw)
        throw std::logic_error("nosanow: apply_move_batch before evaluate_batch");

    // Nothing is evaluated here: the new share, gradient and laplacian of p
    // are the pending ones left by the ratio_batch call for this same move.
    commit_particle_value(batch.log_psi(), p, accept, w_begin, w_end);

    std::span<double> b_grad = batch.gradients();
    std::span<double> b_lap  = batch.laplacians();
    for (std::size_t w = w_begin; w < w_end; ++w) {
        const double m = accept[w] ? 1.0 : 0.0;
        for (std::size_t d = 0; d < dim; ++d) {
            const std::size_t i = (p * dim + d) * nw + w;
            const double dg = m * (pending_grad_c_[d * nw + w] - grad_c_[i]);
            const double dl = m * (pending_lap_c_[d * nw + w] - lap_c_[i]);
            grad_c_[i] += dg;
            b_grad[i]  += dg;
            lap_c_[i]  += dl;
            b_lap[i]   += dl;
        }
    }
}

void NosanowJastrow::compute_parameter_derivatives(const WalkerSet& batch,
                                                   std::span<double> derivs) const {
    if (num_optimizable_params() == 0) return;
    const std::size_t nw  = batch.num_walkers();
    const std::size_t dim = batch.dim();
    std::fill(derivs.begin(), derivs.end(), 0.0);
    const double* pos = batch.positions().data();
    std::vector<double> del(dim * nw);  // [dim * nw]
    for (std::size_t k = 0; k < n_sites(); ++k) {
        displacement(pos + (first_ + k) * dim * nw, k, nw, 0, nw, del.data());
        for (std::size_t d = 0; d < dim; ++d)
            for (std::size_t w = 0; w < nw; ++w)
                derivs[w] -= del[d * nw + w] * del[d * nw + w];  // −|Δ|²
    }
}

bool NosanowJastrow::compute_parameter_gradient_derivatives(
    const WalkerSet& batch, std::span<double> dgrad, std::span<double> dlap) const {
    if (num_optimizable_params() == 0) return true;
    const std::size_t nw  = batch.num_walkers();
    const std::size_t dim = batch.dim();
    std::fill(dgrad.begin(), dgrad.end(), 0.0);
    std::fill(dlap.begin(), dlap.end(), 0.0);
    const double* pos = batch.positions().data();
    for (std::size_t k = 0; k < n_sites(); ++k) {
        const std::size_t p = first_ + k;
        double* g = dgrad.data() + p * dim * nw;  // ∂(∂_d)/∂α = −2Δ_d
        displacement(pos + p * dim * nw, k, nw, 0, nw, g);
        for (std::size_t i = 0; i < dim * nw; ++i) {
            g[i] *= -2.0;
            dlap[p * dim * nw + i] = -2.0;  // ∂(∂²_d)/∂α
        }
    }
    return true;
}

}  // namespace alea
