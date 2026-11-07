// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "wavefunction/wavefunction_component.h"

#include <stdexcept>

namespace alea {

void WavefunctionComponent::shifted_pair_ratios_batch(const WalkerSet&, std::size_t,
                                                      std::size_t, std::size_t, std::size_t,
                                                      std::span<const double>,
                                                      std::span<double>, std::span<double>) {
    throw std::logic_error(name() + ": ratios of a pair moved together are not implemented");
}

void WavefunctionComponent::resize_caches(std::size_t nw, std::size_t np,
                                          std::size_t dim) {
    log_psi_c_.assign(nw, 0.0);
    value_c_.assign(particle_values_ ? np * nw : 0, 0.0);
    grad_c_.assign(np * dim * nw, 0.0);
    lap_c_.assign(np * dim * nw, 0.0);

    pending_value_c_.assign(nw, 0.0);
    pending_grad_c_.assign(dim * nw, 0.0);
    pending_lap_c_.assign(dim * nw, 0.0);

    fold_delta_.assign(dim * nw, 0.0);
    fold_delta_old_.assign(dim * nw, 0.0);
}

void WavefunctionComponent::permute_caches(std::span<const std::size_t> src) {
    const std::size_t old_nw = log_psi_c_.size();
    if (old_nw == 0) return;  // never evaluated — nothing to carry
    const std::size_t new_nw = src.size();

    // log_psi_c_: one value per walker.
    std::vector<double> lp(new_nw, 0.0);
    for (std::size_t j = 0; j < new_nw; ++j) lp[j] = log_psi_c_[src[j]];
    log_psi_c_ = std::move(lp);

    // grad_c_ / lap_c_: [k * nw + w], width k = size / nw (= np*dim).
    auto gather = [&](std::vector<double>& a) {
        const std::size_t width = a.size() / old_nw;
        std::vector<double> dst(width * new_nw, 0.0);
        for (std::size_t k = 0; k < width; ++k) {
            const double* __restrict o = a.data() + k * old_nw;
            double* __restrict n = dst.data() + k * new_nw;
            for (std::size_t j = 0; j < new_nw; ++j) n[j] = o[src[j]];
        }
        a = std::move(dst);
    };
    if (!value_c_.empty()) gather(value_c_);
    gather(grad_c_);
    gather(lap_c_);

    // pending_* are transient (rebuilt by the next ratio_batch); just resize.
    const std::size_t pend_width =
        pending_grad_c_.empty() ? 0 : pending_grad_c_.size() / old_nw;
    pending_value_c_.assign(new_nw, 0.0);
    pending_grad_c_.assign(pend_width * new_nw, 0.0);
    pending_lap_c_.assign(pend_width * new_nw, 0.0);
    fold_delta_.assign(pend_width * new_nw, 0.0);
    fold_delta_old_.assign(pend_width * new_nw, 0.0);
}

void WavefunctionComponent::commit_particle_value(std::span<double> batch_log_psi,
                                                  std::size_t p,
                                                  std::span<const bool> accept,
                                                  std::size_t w_begin,
                                                  std::size_t w_end) noexcept {
    const std::size_t nw = batch_log_psi.size();
    double* __restrict s           = value_c_.data() + p * nw;
    const double* __restrict s_new = pending_value_c_.data();
    double* __restrict lpc         = log_psi_c_.data();
    double* __restrict blp         = batch_log_psi.data();
    for (std::size_t w = w_begin; w < w_end; ++w) {
        const double m    = accept[w] ? 1.0 : 0.0;
        const double dlog = m * (s_new[w] - s[w]);
        lpc[w] += dlog;
        blp[w] += dlog;
        s[w] = accept[w] ? s_new[w] : s[w];
    }
}

std::size_t WavefunctionComponent::num_optimizable_params() const {
    std::size_t n = 0;
    for (const Parameter& p : parameters())
        if (!p.is_fixed) ++n;
    return n;
}

}  // namespace alea
