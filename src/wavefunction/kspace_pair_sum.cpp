// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "wavefunction/kspace_pair_sum.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "population/walker_set.h"

namespace alea {

KSpacePairSum::KSpacePairSum(const SimulationCell& cell, ParticleGroup a, ParticleGroup b,
                             std::vector<int> n1, std::vector<int> n2, std::vector<double> kx,
                             std::vector<double> ky, std::vector<double> weights,
                             std::vector<std::vector<double>> param_weights)
    : a_(std::move(a)),
      b_(std::move(b)),
      same_group_(a_ == b_),
      kx_(std::move(kx)),
      ky_(std::move(ky)),
      phases_(cell, std::move(n1), std::move(n2)) {
    if (cell.dim() != 2 || !cell.is_periodic())
        throw std::invalid_argument("k-space pair sum: needs a periodic 2-D cell");
    for (std::size_t k = 0; k < kx_.size(); ++k) {
        kx2_.push_back(kx_[k] * kx_[k]);
        ky2_.push_back(ky_[k] * ky_[k]);
    }
    n_cached_ = a_.size() + (same_group_ ? 0 : b_.size());
    set_weights(std::move(weights), std::move(param_weights));
}

void KSpacePairSum::set_weights(std::vector<double> weights,
                                std::vector<std::vector<double>> param_weights) {
    if (weights.size() != kx_.size())
        throw std::invalid_argument("k-space pair sum: one weight per k vector");
    for (const auto& p : param_weights)
        if (p.size() != kx_.size())
            throw std::invalid_argument("k-space pair sum: one parameter weight per k vector");
    w_ = std::move(weights);
    param_w_ = std::move(param_weights);
}

void KSpacePairSum::prepare(std::size_t nw) {
    const std::size_t nk = kx_.size(), n = nw * nk;
    if (nw_ == nw && ra_re_.size() == n) return;
    nw_ = nw;
    ra_re_.assign(n, 0.0);
    ra_im_.assign(n, 0.0);
    if (!same_group_) {
        rb_re_.assign(n, 0.0);
        rb_im_.assign(n, 0.0);
    }
    pc_.assign(n * n_cached_, 0.0);
    ps_.assign(n * n_cached_, 0.0);
    pend_c_.assign(n, 0.0);
    pend_s_.assign(n, 0.0);
}

void KSpacePairSum::permute(std::span<const std::size_t> src) {
    const std::size_t nk = kx_.size();
    auto carry = [&](std::vector<double>& v, std::size_t row) {
        if (v.empty()) return;
        std::vector<double> out(src.size() * row);
        for (std::size_t j = 0; j < src.size(); ++j)
            std::copy_n(v.data() + src[j] * row, row, out.data() + j * row);
        v = std::move(out);
    };
    carry(ra_re_, nk); carry(ra_im_, nk); carry(rb_re_, nk); carry(rb_im_, nk);
    carry(pc_, n_cached_ * nk); carry(ps_, n_cached_ * nk);
    pend_c_.assign(src.size() * nk, 0.0);   // transient: rebuilt by the next ratio()
    pend_s_.assign(src.size() * nk, 0.0);
    nw_ = src.size();
}

void KSpacePairSum::phases(double x, double y, double* c, double* s) const {
    phases_(x, y, c, s);
}

namespace {

// Gradient/laplacian of Σ_k w T for one particle with phases (c, s) that sees
// the density (pr, pi):
//   g_k = 2 w (c ρ_im − s ρ_re),   l_k = 2 w (self − (c ρ_re + s ρ_im))
// with self = 1 when the particle is part of the density it sees (one group),
// 0 otherwise.  Returns Σ k_x g, Σ k_y g, Σ k_x² l, Σ k_y² l.
struct KTerms { double gx, gy, lx, ly; };
KTerms kterms(std::size_t nk, const double* __restrict w, const double* __restrict kx,
              const double* __restrict ky, const double* __restrict kx2,
              const double* __restrict ky2, const double* __restrict c,
              const double* __restrict s, const double* __restrict pr,
              const double* __restrict pi, double self) {
    double gx = 0, gy = 0, lx = 0, ly = 0;
#pragma omp simd reduction(+ : gx, gy, lx, ly)
    for (std::size_t k = 0; k < nk; ++k) {
        const double g = 2.0 * w[k] * (c[k] * pi[k] - s[k] * pr[k]);
        const double l = 2.0 * w[k] * (self - (c[k] * pr[k] + s[k] * pi[k]));
        gx += kx[k] * g;
        gy += ky[k] * g;
        lx += kx2[k] * l;
        ly += ky2[k] * l;
    }
    return {gx, gy, lx, ly};
}

// Σ_k w T for one walker: |ρ_A|² (one group) or 2 Re ρ_A ρ_B* (two).
double kspace_log(std::size_t nk, const double* __restrict w, const double* __restrict ar,
                  const double* __restrict ai, const double* __restrict br,
                  const double* __restrict bi, bool same) {
    double t = 0.0;
    if (same) {
#pragma omp simd reduction(+ : t)
        for (std::size_t k = 0; k < nk; ++k) t += w[k] * (ar[k] * ar[k] + ai[k] * ai[k]);
    } else {
#pragma omp simd reduction(+ : t)
        for (std::size_t k = 0; k < nk; ++k) t += 2.0 * w[k] * (ar[k] * br[k] + ai[k] * bi[k]);
    }
    return t;
}

}  // namespace

void KSpacePairSum::accumulate(const WalkerSet& batch, std::size_t w,
                                 std::span<const double> wts, double* log_psi,
                                 double* grad, double* lap) const {
    const std::size_t nk = kx_.size(), nw = batch.num_walkers(), dim = batch.dim();
    const double* ar = ra_re_.data() + w * nk;
    const double* ai = ra_im_.data() + w * nk;
    const double* br = same_group_ ? ar : rb_re_.data() + w * nk;
    const double* bi = same_group_ ? ai : rb_im_.data() + w * nk;
    const double* wt = wts.data();

    if (log_psi) log_psi[w] += kspace_log(nk, wt, ar, ai, br, bi, same_group_);
    if (!grad && !lap) return;

    auto add = [&](std::size_t i, const double* pr, const double* pi, double self) {
        const KTerms t = kterms(nk, wt, kx_.data(), ky_.data(), kx2_.data(), ky2_.data(),
                                cos_row(w, i), sin_row(w, i), pr, pi, self);
        if (grad) {
            grad[(i * dim + 0) * nw + w] += t.gx;
            grad[(i * dim + 1) * nw + w] += t.gy;
        }
        if (lap) {
            lap[(i * dim + 0) * nw + w] += t.lx;
            lap[(i * dim + 1) * nw + w] += t.ly;
        }
    };
    // Each particle sees ρ of the OTHER group (or its own, for one group).
    for (std::size_t i = a_.first; i < a_.last; ++i) add(i, br, bi, same_group_ ? 1.0 : 0.0);
    if (!same_group_)
        for (std::size_t j = b_.first; j < b_.last; ++j) add(j, ar, ai, 0.0);
}

void KSpacePairSum::evaluate(const WalkerSet& batch, std::span<double> log_psi,
                               std::span<double> grad, std::span<double> lap) {
    const std::size_t nw = batch.num_walkers(), dim = batch.dim(), nk = kx_.size();
    prepare(nw);
    const auto pos = batch.positions();
    for (std::size_t w = 0; w < nw; ++w) {
        auto build = [&](const ParticleGroup& g, double* re, double* im) {
            std::fill_n(re, nk, 0.0);
            std::fill_n(im, nk, 0.0);
            for (std::size_t i = g.first; i < g.last; ++i) {
                double* c = cos_row(w, i);
                double* s = sin_row(w, i);
                phases(pos[(i * dim + 0) * nw + w], pos[(i * dim + 1) * nw + w], c, s);
                for (std::size_t k = 0; k < nk; ++k) { re[k] += c[k]; im[k] += s[k]; }
            }
        };
        build(a_, ra_re_.data() + w * nk, ra_im_.data() + w * nk);
        if (!same_group_) build(b_, rb_re_.data() + w * nk, rb_im_.data() + w * nk);
        accumulate(batch, w, w_, log_psi.data(), grad.data(), lap.data());
    }
}

void KSpacePairSum::ratio(const WalkerSet& batch, std::size_t p, std::size_t w_begin,
                            std::size_t w_end, std::span<double> log_ratio,
                            std::span<double> new_grad, std::span<double> new_lap) {
    const bool in_a = a_.contains(p), in_b = !same_group_ && b_.contains(p);
    if (!in_a && !in_b) return;
    const std::size_t nw = batch.num_walkers(), nk = kx_.size();
    const auto np = batch.new_pos();
    const double* __restrict wt = w_.data();
    thread_local std::vector<double> nr, ni;
    nr.resize(nk); ni.resize(nk);

    for (std::size_t w = w_begin; w < w_end; ++w) {
        double* __restrict cn = pend_c_.data() + w * nk;
        double* __restrict sn = pend_s_.data() + w * nk;
        phases(np[0 * nw + w], np[1 * nw + w], cn, sn);
        const double* __restrict co = cos_row(w, p);
        const double* __restrict so = sin_row(w, p);
        const double* __restrict mr = (in_a ? ra_re_ : rb_re_).data() + w * nk;
        const double* __restrict mi = (in_a ? ra_im_ : rb_im_).data() + w * nk;
        double dlog = 0.0;
        KTerms t{};
        if (same_group_) {
            // p sees its own group after the move: ρ' = ρ + e'_p − e_p.
            double* __restrict qr = nr.data();
            double* __restrict qi = ni.data();
#pragma omp simd reduction(+ : dlog)
            for (std::size_t k = 0; k < nk; ++k) {
                qr[k] = mr[k] + cn[k] - co[k];
                qi[k] = mi[k] + sn[k] - so[k];
                dlog += wt[k] * (qr[k] * qr[k] + qi[k] * qi[k] - mr[k] * mr[k] - mi[k] * mi[k]);
            }
            t = kterms(nk, wt, kx_.data(), ky_.data(), kx2_.data(), ky2_.data(), cn, sn, qr,
                       qi, 1.0);
        } else {
            const double* __restrict orr = (in_a ? rb_re_ : ra_re_).data() + w * nk;
            const double* __restrict oi  = (in_a ? rb_im_ : ra_im_).data() + w * nk;
#pragma omp simd reduction(+ : dlog)
            for (std::size_t k = 0; k < nk; ++k)
                dlog += 2.0 * wt[k] * ((cn[k] - co[k]) * orr[k] + (sn[k] - so[k]) * oi[k]);
            t = kterms(nk, wt, kx_.data(), ky_.data(), kx2_.data(), ky2_.data(), cn, sn, orr,
                       oi, 0.0);
        }
        log_ratio[w] += dlog;
        new_grad[0 * nw + w] += t.gx;
        new_grad[1 * nw + w] += t.gy;
        new_lap[0 * nw + w] += t.lx;
        new_lap[1 * nw + w] += t.ly;
    }
}

void KSpacePairSum::accept(const WalkerSet& batch, std::size_t p,
                             std::span<const double> mask, std::size_t w_begin,
                             std::size_t w_end,
                             std::span<const std::span<double>> log_psi_targets,
                             std::span<const std::span<double>> grad_targets,
                             std::span<const std::span<double>> lap_targets) {
    const bool in_a = a_.contains(p), in_b = !same_group_ && b_.contains(p);
    if (!in_a && !in_b) return;
    const std::size_t nw = batch.num_walkers(), dim = batch.dim(), nk = kx_.size();
    const double* __restrict wt = w_.data();
    thread_local std::vector<double> dr_v, di_v;
    dr_v.resize(nk); di_v.resize(nk);
    double* __restrict dr = dr_v.data();
    double* __restrict di = di_v.data();

    // Particles whose gradient changes: the rest of p's group (one group), or
    // the other group (two groups).  Their phases are cached and unchanged.
    const ParticleGroup& seers = same_group_ ? a_ : (in_a ? b_ : a_);

    for (std::size_t w = w_begin; w < w_end; ++w) {
        if (mask[w] == 0.0) continue;
        const double* __restrict cn = pend_c_.data() + w * nk;
        const double* __restrict sn = pend_s_.data() + w * nk;
        double* __restrict co = cos_row(w, p);
        double* __restrict so = sin_row(w, p);
        double* __restrict mr = (in_a ? ra_re_ : rb_re_).data() + w * nk;
        double* __restrict mi = (in_a ? ra_im_ : rb_im_).data() + w * nk;

        double dlog = 0.0;
        if (same_group_) {
#pragma omp simd reduction(+ : dlog)
            for (std::size_t k = 0; k < nk; ++k) {
                dr[k] = cn[k] - co[k];
                di[k] = sn[k] - so[k];
                dlog += wt[k] * (2.0 * (mr[k] * dr[k] + mi[k] * di[k]) +
                                 dr[k] * dr[k] + di[k] * di[k]);
            }
        } else {
            const double* __restrict orr = (in_a ? rb_re_ : ra_re_).data() + w * nk;
            const double* __restrict oi  = (in_a ? rb_im_ : ra_im_).data() + w * nk;
#pragma omp simd reduction(+ : dlog)
            for (std::size_t k = 0; k < nk; ++k) {
                dr[k] = cn[k] - co[k];
                di[k] = sn[k] - so[k];
                dlog += 2.0 * wt[k] * (dr[k] * orr[k] + di[k] * oi[k]);
            }
        }
        for (const auto& t : log_psi_targets) t[w] += dlog;

        // Every other particle that sees the changed ρ moves by the δρ form of
        // the same expression (its "self" term does not change).
        for (std::size_t i = seers.first; i < seers.last; ++i) {
            if (i == p) continue;
            const KTerms t = kterms(nk, wt, kx_.data(), ky_.data(), kx2_.data(), ky2_.data(),
                                    cos_row(w, i), sin_row(w, i), dr, di, 0.0);
            for (const auto& g : grad_targets) {
                g[(i * dim + 0) * nw + w] += t.gx;
                g[(i * dim + 1) * nw + w] += t.gy;
            }
            for (const auto& l : lap_targets) {
                l[(i * dim + 0) * nw + w] += t.lx;
                l[(i * dim + 1) * nw + w] += t.ly;
            }
        }
#pragma omp simd
        for (std::size_t k = 0; k < nk; ++k) {
            mr[k] += dr[k];
            mi[k] += di[k];
            co[k] = cn[k];   // p's cached phases follow the move
            so[k] = sn[k];
        }
    }
}

void KSpacePairSum::param_log_psi(std::size_t k, std::size_t nw,
                                    std::span<double> out) const {
    const std::size_t nk = kx_.size();
    for (std::size_t w = 0; w < nw && w < nw_; ++w) {
        const double* ar = ra_re_.data() + w * nk;
        const double* ai = ra_im_.data() + w * nk;
        const double* br = same_group_ ? ar : rb_re_.data() + w * nk;
        const double* bi = same_group_ ? ai : rb_im_.data() + w * nk;
        out[w] += kspace_log(nk, param_w_[k].data(), ar, ai, br, bi, same_group_);
    }
}

void KSpacePairSum::param_grad_lap(const WalkerSet& batch, std::size_t k,
                                     std::span<double> dgrad,
                                     std::span<double> dlap) const {
    for (std::size_t w = 0; w < batch.num_walkers(); ++w)
        accumulate(batch, w, param_w_[k], nullptr, dgrad.data(), dlap.data());
}

}  // namespace alea
