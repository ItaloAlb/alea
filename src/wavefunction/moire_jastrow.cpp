// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "wavefunction/moire_jastrow.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "population/walker_set.h"

namespace alea {

std::vector<double>
moire_jastrow_shell_vectors(std::span<const std::vector<double>> g,
                            std::size_t n_shells, std::size_t dim) {
    if (g.size() != 3)
        throw std::runtime_error("moire jastrow: need the 3 first-shell g vectors");
    if (n_shells < 1 || n_shells > 3)
        throw std::runtime_error("moire jastrow: shells must be 1, 2 or 3 (got " +
                                 std::to_string(n_shells) + ")");
    std::vector<double> k;
    k.reserve(n_shells * 3 * dim);
    for (std::size_t s = 0; s < n_shells; ++s)
        for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t d = 0; d < dim; ++d) {
                const double gj = g[j][d];
                const double gn = g[(j + 1) % 3][d];
                k.push_back(s == 0 ? gj : s == 1 ? gj - gn : 2.0 * gj);
            }
    return k;
}

namespace {

std::vector<Parameter> checked(std::vector<Parameter> params, std::size_t n_k) {
    if (params.empty() || params.size() % 2 != 0)
        throw std::runtime_error(
            "moire jastrow: params must hold (a, b) pairs, one pair per shell; got " +
            std::to_string(params.size()));
    if (params.size() / 2 * 3 != n_k)
        throw std::runtime_error(
            "moire jastrow: " + std::to_string(params.size() / 2) +
            " coefficient pairs but " + std::to_string(n_k / 3) + " shells of k-vectors");
    return params;
}

}  // namespace

MoireJastrow::MoireJastrow(std::size_t dim, std::vector<double> k_vectors,
                           std::vector<std::pair<std::size_t, std::size_t>> ranges,
                           std::vector<Parameter> params)
    : dim_(dim),
      k_(std::move(k_vectors)),
      ranges_(std::move(ranges)),
      params_(checked(std::move(params), dim == 0 ? 0 : k_.size() / dim)) {
    // One-body: s_p = u(r_p), so a move needs the proposed position only.
    enable_particle_values();
}

bool MoireJastrow::acts_on(std::size_t p) const noexcept {
    for (const auto& [first, last] : ranges_)
        if (p >= first && p < last) return true;
    return false;
}

void MoireJastrow::particle_terms(const double* x, std::size_t nw, std::size_t w0,
                                  std::size_t w1, double* u, double* grad,
                                  double* lap) const {
    const std::size_t dim = dim_;
    for (std::size_t w = w0; w < w1; ++w) u[w] = 0.0;
    for (std::size_t d = 0; d < dim; ++d)
        for (std::size_t w = w0; w < w1; ++w) {
            grad[d * nw + w] = 0.0;
            lap[d * nw + w]  = 0.0;
        }

    const std::size_t n_k = k_.size() / dim;
    for (std::size_t kk = 0; kk < n_k; ++kk) {
        const double  a = params_[2 * (kk / 3)].value;
        const double  b = params_[2 * (kk / 3) + 1].value;
        const double* k = k_.data() + kk * dim;
        for (std::size_t w = w0; w < w1; ++w) {
            double phase = 0.0;
            for (std::size_t d = 0; d < dim; ++d) phase += k[d] * x[d * nw + w];
            const double c = std::cos(phase);
            const double s = std::sin(phase);
            const double val = a * c + b * s;   // u term;  ∂²_d = −k_d² · val
            const double slope = b * c - a * s; // ∂_d = k_d · slope
            u[w] += val;
            for (std::size_t d = 0; d < dim; ++d) {
                grad[d * nw + w] += k[d] * slope;
                lap[d * nw + w]  -= k[d] * k[d] * val;
            }
        }
    }
}

void MoireJastrow::evaluate_batch(WalkerSet& batch) {
    const std::size_t nw  = batch.num_walkers();
    const std::size_t np  = batch.n_particles();
    const std::size_t dim = batch.dim();
    prepare(nw, np, dim);

    const double* pos = batch.positions().data();
    for (const auto& [first, last] : ranges_)
        for (std::size_t p = first; p < last; ++p) {
            double* s_p = value_c_.data() + p * nw;   // u(r_p), the share of p
            particle_terms(pos + p * dim * nw, nw, 0, nw, s_p,
                           grad_c_.data() + p * dim * nw,
                           lap_c_.data() + p * dim * nw);
            for (std::size_t w = 0; w < nw; ++w) log_psi_c_[w] += s_p[w];
        }

    std::span<double> log_psi = batch.log_psi();
    for (std::size_t w = 0; w < nw; ++w) log_psi[w] += log_psi_c_[w];
    std::span<double> grad = batch.gradients();
    for (std::size_t k = 0; k < grad.size(); ++k) grad[k] += grad_c_[k];
    std::span<double> lap = batch.laplacians();
    for (std::size_t k = 0; k < lap.size(); ++k) lap[k] += lap_c_[k];
}

void MoireJastrow::ratio_batch(WalkerSet& batch, std::size_t p,
                               std::size_t w_begin, std::size_t w_end) {
    if (!acts_on(p)) return;
    const std::size_t nw  = batch.num_walkers();
    const std::size_t dim = batch.dim();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end) return;
    if (value_c_.size() < (p + 1) * nw)
        throw std::logic_error("moire jastrow: ratio_batch before evaluate_batch");

    // The old share u(r_p) is committed in value_c_; only r_p' is evaluated.
    particle_terms(batch.new_pos().data(), nw, w_begin, w_end, pending_value_c_.data(),
                   pending_grad_c_.data(), pending_lap_c_.data());

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

void MoireJastrow::apply_move_batch(WalkerSet& batch, std::size_t p,
                                    std::span<const bool> accept,
                                    std::size_t w_begin, std::size_t w_end) {
    if (!acts_on(p)) return;
    const std::size_t nw  = batch.num_walkers();
    const std::size_t dim = batch.dim();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end) return;
    if (value_c_.size() < (p + 1) * nw)
        throw std::logic_error("moire jastrow: apply_move_batch before evaluate_batch");

    // Nothing is evaluated here: the new share, gradient and laplacian of p
    // are the pending ones left by the ratio_batch call for this same move.
    commit_particle_value(batch.log_psi(), p, accept, w_begin, w_end);

    std::span<double> b_grad   = batch.gradients();
    std::span<double> b_lap    = batch.laplacians();
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

void MoireJastrow::compute_parameter_derivatives(const WalkerSet& batch,
                                                 std::span<double> derivs) const {
    const std::size_t n_opt = num_optimizable_params();
    if (n_opt == 0) return;
    const std::size_t nw  = batch.num_walkers();
    const std::size_t dim = batch.dim();
    std::fill(derivs.begin(), derivs.end(), 0.0);

    // opt slot of each params_ index, or npos when that parameter is fixed.
    std::vector<std::size_t> slot(params_.size(), npos);
    for (std::size_t i = 0, k = 0; i < params_.size(); ++i)
        if (!params_[i].is_fixed) slot[i] = k++;

    const double*     pos = batch.positions().data();
    const std::size_t n_k = k_.size() / dim;
    for (const auto& [first, last] : ranges_)
        for (std::size_t p = first; p < last; ++p) {
            const double* x = pos + p * dim * nw;
            for (std::size_t kk = 0; kk < n_k; ++kk) {
                const std::size_t sa = slot[2 * (kk / 3)];
                const std::size_t sb = slot[2 * (kk / 3) + 1];
                const double*     k  = k_.data() + kk * dim;
                for (std::size_t w = 0; w < nw; ++w) {
                    double phase = 0.0;
                    for (std::size_t d = 0; d < dim; ++d) phase += k[d] * x[d * nw + w];
                    if (sa != npos) derivs[sa * nw + w] += std::cos(phase);
                    if (sb != npos) derivs[sb * nw + w] += std::sin(phase);
                }
            }
        }
}

bool MoireJastrow::compute_parameter_gradient_derivatives(
    const WalkerSet& batch, std::span<double> dgrad, std::span<double> dlap) const {
    const std::size_t n_opt = num_optimizable_params();
    if (n_opt == 0) return true;
    const std::size_t nw     = batch.num_walkers();
    const std::size_t np     = batch.n_particles();
    const std::size_t dim    = batch.dim();
    const std::size_t stride = np * dim * nw;
    std::fill(dgrad.begin(), dgrad.end(), 0.0);
    std::fill(dlap.begin(), dlap.end(), 0.0);

    std::vector<std::size_t> slot(params_.size(), npos);
    for (std::size_t i = 0, k = 0; i < params_.size(); ++i)
        if (!params_[i].is_fixed) slot[i] = k++;

    // ∂(∂_d u)/∂a = −k_d sin,  ∂(∂²_d u)/∂a = −k_d² cos
    // ∂(∂_d u)/∂b = +k_d cos,  ∂(∂²_d u)/∂b = −k_d² sin
    const double*     pos = batch.positions().data();
    const std::size_t n_k = k_.size() / dim;
    for (const auto& [first, last] : ranges_)
        for (std::size_t p = first; p < last; ++p) {
            const double* x = pos + p * dim * nw;
            for (std::size_t kk = 0; kk < n_k; ++kk) {
                const std::size_t sa = slot[2 * (kk / 3)];
                const std::size_t sb = slot[2 * (kk / 3) + 1];
                const double*     k  = k_.data() + kk * dim;
                for (std::size_t w = 0; w < nw; ++w) {
                    double phase = 0.0;
                    for (std::size_t d = 0; d < dim; ++d) phase += k[d] * x[d * nw + w];
                    const double c = std::cos(phase);
                    const double s = std::sin(phase);
                    for (std::size_t d = 0; d < dim; ++d) {
                        const std::size_t i = (p * dim + d) * nw + w;
                        if (sa != npos) {
                            dgrad[sa * stride + i] -= k[d] * s;
                            dlap[sa * stride + i]  -= k[d] * k[d] * c;
                        }
                        if (sb != npos) {
                            dgrad[sb * stride + i] += k[d] * c;
                            dlap[sb * stride + i]  -= k[d] * k[d] * s;
                        }
                    }
                }
            }
        }
    return true;
}

}  // namespace alea
