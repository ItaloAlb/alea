// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "wavefunction/kspace_jastrow.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "population/walker_set.h"

namespace alea {

KSpaceJastrow::KSpaceJastrow(const SimulationCell& cell, ParticleGroup group_a,
                             ParticleGroup group_b, std::size_t n_shells,
                             std::vector<Parameter> params)
    : group_a_(std::move(group_a)), group_b_(std::move(group_b)), params_(std::move(params)) {
    if (!cell.is_periodic() || cell.dim() != 2)
        throw std::invalid_argument("kspace_jastrow: needs a periodic 2-D cell");
    if (n_shells == 0 || params_.size() != n_shells)
        throw std::invalid_argument("kspace_jastrow: one coefficient per shell (" +
                                    std::to_string(n_shells) + " shells, " +
                                    std::to_string(params_.size()) + " coefficients)");

    // Half-space reciprocal vectors (n1 > 0, or n1 = 0 and n2 > 0), sorted by
    // |k|, up to the end of shell n_shells.
    const auto b = cell.reciprocal();
    const double bmin = std::min(std::hypot(b[0], b[1]), std::hypot(b[2], b[3]));
    struct K { double k, kx, ky; int n1, n2; };
    std::vector<K> ks;
    for (int nmax = 2;; nmax *= 2) {
        ks.clear();
        for (int i = 0; i <= nmax; ++i)
            for (int j = -nmax; j <= nmax; ++j) {
                if (i == 0 && j <= 0) continue;
                const double kx = i * b[0] + j * b[2], ky = i * b[1] + j * b[3];
                ks.push_back({std::hypot(kx, ky), kx, ky, i, j});
            }
        std::sort(ks.begin(), ks.end(), [](const K& x, const K& y) { return x.k < y.k; });
        // Shells are complete if the n-th shell ends well inside the scanned box.
        std::size_t shells = 0;
        double last = -1.0, end_k = 0.0;
        for (const K& q : ks) {
            if (q.k - last > 1e-9 * bmin) {
                if (shells == n_shells) { end_k = q.k; break; }
                ++shells;
                last = q.k;
            }
        }
        if (shells == n_shells && end_k > 0.0 && end_k < 0.9 * nmax * bmin) break;
    }

    std::vector<int> n1, n2;
    std::vector<double> kx, ky;
    double last = -1.0;
    for (const K& q : ks) {
        if (q.k - last > 1e-9 * bmin) {
            if (shell_k_.size() == n_shells) break;
            shell_k_.push_back(q.k);
            shell_count_.push_back(0);
            last = q.k;
        }
        ++shell_count_.back();
        shell_of_.push_back(shell_k_.size() - 1);
        n1.push_back(q.n1);
        n2.push_back(q.n2);
        kx.push_back(q.kx);
        ky.push_back(q.ky);
    }
    sum_ = KSpacePairSum(cell, group_a_, group_b_, std::move(n1), std::move(n2),
                         std::move(kx), std::move(ky), std::vector<double>(shell_of_.size(), 0.0));
    push_weights();
}

void KSpaceJastrow::push_weights() {
    std::vector<double> w(shell_of_.size());
    for (std::size_t k = 0; k < w.size(); ++k) w[k] = params_[shell_of_[k]].value;
    std::vector<std::vector<double>> dw;
    for (std::size_t s = 0; s < params_.size(); ++s) {
        if (params_[s].is_fixed) continue;
        std::vector<double> ind(shell_of_.size(), 0.0);
        for (std::size_t k = 0; k < ind.size(); ++k) ind[k] = shell_of_[k] == s ? 1.0 : 0.0;
        dw.push_back(std::move(ind));
    }
    sum_.set_weights(std::move(w), std::move(dw));
}

void KSpaceJastrow::rebuild() { push_weights(); }

void KSpaceJastrow::prepare(std::size_t nw, std::size_t np, std::size_t dim) {
    resize_caches(nw, np, dim);
    mask_.assign(nw, 0.0);
    sum_.prepare(nw);
}

void KSpaceJastrow::permute_caches(std::span<const std::size_t> src) {
    WavefunctionComponent::permute_caches(src);
    mask_.assign(src.size(), 0.0);
    sum_.permute(src);
}

void KSpaceJastrow::evaluate_batch(WalkerSet& batch) {
    prepare(batch.num_walkers(), batch.n_particles(), batch.dim());
    sum_.evaluate(batch, log_psi_c_, grad_c_, lap_c_);
    std::span<double> log_psi = batch.log_psi();
    for (std::size_t w = 0; w < log_psi.size(); ++w) log_psi[w] += log_psi_c_[w];
    std::span<double> grad = batch.gradients();
    for (std::size_t i = 0; i < grad.size(); ++i) grad[i] += grad_c_[i];
    std::span<double> lap = batch.laplacians();
    for (std::size_t i = 0; i < lap.size(); ++i) lap[i] += lap_c_[i];
}

void KSpaceJastrow::ratio_batch(WalkerSet& batch, std::size_t p, std::size_t w_begin,
                                std::size_t w_end) {
    const std::size_t nw = batch.num_walkers(), dim = batch.dim();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end) return;
    if (mask_.size() < nw) prepare(nw, batch.n_particles(), dim);
    for (std::size_t d = 0; d < dim; ++d)
        for (std::size_t w = w_begin; w < w_end; ++w) {
            pending_grad_c_[d * nw + w] = 0.0;
            pending_lap_c_[d * nw + w]  = 0.0;
        }
    sum_.ratio(batch, p, w_begin, w_end, batch.log_ratios(), pending_grad_c_, pending_lap_c_);
    std::span<double> ng = batch.new_grad();
    std::span<double> nl = batch.new_lap();
    for (std::size_t d = 0; d < dim; ++d)
        for (std::size_t w = w_begin; w < w_end; ++w) {
            ng[d * nw + w] += pending_grad_c_[d * nw + w];
            nl[d * nw + w] += pending_lap_c_[d * nw + w];
        }
}

void KSpaceJastrow::apply_move_batch(WalkerSet& batch, std::size_t p,
                                     std::span<const bool> accept_mask, std::size_t w_begin,
                                     std::size_t w_end) {
    const std::size_t nw = batch.num_walkers(), dim = batch.dim();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end) return;
    if (!group_a_.contains(p) && !group_b_.contains(p)) return;
    for (std::size_t w = w_begin; w < w_end; ++w) mask_[w] = accept_mask[w] ? 1.0 : 0.0;

    std::span<double> b_logpsi = batch.log_psi();
    std::span<double> b_grad   = batch.gradients();
    std::span<double> b_lap    = batch.laplacians();
    const std::span<double> lp[] = {log_psi_c_, b_logpsi};
    const std::span<double> gr[] = {grad_c_, b_grad};
    const std::span<double> lc[] = {lap_c_, b_lap};
    sum_.accept(batch, p, mask_, w_begin, w_end, lp, gr, lc);

    // Particle p: its new gradient/laplacian are the pending values.
    for (std::size_t d = 0; d < dim; ++d)
        for (std::size_t w = w_begin; w < w_end; ++w) {
            const std::size_t i = (p * dim + d) * nw + w;
            const double dg = mask_[w] * (pending_grad_c_[d * nw + w] - grad_c_[i]);
            const double dl = mask_[w] * (pending_lap_c_[d * nw + w] - lap_c_[i]);
            grad_c_[i] += dg;
            b_grad[i]  += dg;
            lap_c_[i]  += dl;
            b_lap[i]   += dl;
        }
}

void KSpaceJastrow::compute_parameter_derivatives(const WalkerSet& batch,
                                                  std::span<double> derivs) const {
    const std::size_t nw = batch.num_walkers(), n_opt = num_optimizable_params();
    std::fill(derivs.begin(), derivs.end(), 0.0);
    for (std::size_t s = 0; s < n_opt; ++s)
        sum_.param_log_psi(s, nw, derivs.subspan(s * nw, nw));
}

bool KSpaceJastrow::compute_parameter_gradient_derivatives(const WalkerSet& batch,
                                                           std::span<double> dgrad,
                                                           std::span<double> dlap) const {
    const std::size_t n_opt = num_optimizable_params();
    const std::size_t stride = batch.n_particles() * batch.dim() * batch.num_walkers();
    std::fill(dgrad.begin(), dgrad.end(), 0.0);
    std::fill(dlap.begin(), dlap.end(), 0.0);
    for (std::size_t s = 0; s < n_opt; ++s)
        sum_.param_grad_lap(batch, s, dgrad.subspan(s * stride, stride),
                            dlap.subspan(s * stride, stride));
    return true;
}

}  // namespace alea
