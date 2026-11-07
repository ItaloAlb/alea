// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "estimators/structure_factor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "numerics/statistics.h"
#include "population/walker_set.h"
#include "qmc/run_result.h"

namespace alea {

StructureFactor::StructureFactor(std::span<const ParticleGroup> groups,
                                 const SimulationCell& cell, double kmax) {
    if (!cell.is_periodic() || cell.dim() != 2)
        throw std::runtime_error("sofk: the structure factor needs a periodic 2-D cell");
    for (const ParticleGroup& g : groups) groups_.push_back({g.name, g.first, g.last});

    // Half space of the reciprocal lattice (one of each ±k): n1 > 0, or
    // n1 = 0 and n2 > 0 — the convention of the Ewald and Jastrow k sets.
    const auto b = cell.reciprocal();
    const double b1 = std::hypot(b[0], b[1]), b2 = std::hypot(b[2], b[3]);
    const int nmax = static_cast<int>(std::ceil(kmax / std::min(b1, b2))) + 1;
    struct K { int n1, n2; double kx, ky, k; };
    std::vector<K> ks;
    for (int i = 0; i <= nmax; ++i)
        for (int j = -nmax; j <= nmax; ++j) {
            if (i == 0 && j <= 0) continue;
            const double kx = i * b[0] + j * b[2], ky = i * b[1] + j * b[3];
            const double k = std::hypot(kx, ky);
            if (k <= kmax) ks.push_back({i, j, kx, ky, k});
        }
    if (ks.empty())
        throw std::runtime_error(
            "sofk: kmax = " + std::to_string(kmax) + " is below the shortest "
            "reciprocal vector " + std::to_string(std::min(b1, b2)) + " of the cell");
    std::sort(ks.begin(), ks.end(), [](const K& x, const K& y) { return x.k < y.k; });

    std::vector<int> n1, n2;
    const double tol = 1e-9 * std::min(b1, b2);
    for (const K& q : ks) {
        if (shell_k_.empty() || q.k - shell_k_.back() > tol) {
            shell_k_.push_back(q.k);
            shell_count_.push_back(0);
        }
        ++shell_count_.back();
        shell_of_.push_back(shell_k_.size() - 1);
        kx_.push_back(q.kx);
        ky_.push_back(q.ky);
        n1.push_back(q.n1);
        n2.push_back(q.n2);
    }
    phases_ = numerics::ReciprocalPhases(cell, std::move(n1), std::move(n2));

    const std::size_t nk = kx_.size();
    acc_s_.assign(n_pairs() * nk, 0.0);
    acc_re_.assign(groups_.size() * nk, 0.0);
    acc_im_.assign(groups_.size() * nk, 0.0);
    rho_re_.assign(groups_.size() * nk, 0.0);
    rho_im_.assign(groups_.size() * nk, 0.0);
    c_.assign(nk, 0.0);
    s_.assign(nk, 0.0);
}

void StructureFactor::start_block() {
    std::fill(acc_s_.begin(), acc_s_.end(), 0.0);
    std::fill(acc_re_.begin(), acc_re_.end(), 0.0);
    std::fill(acc_im_.begin(), acc_im_.end(), 0.0);
    acc_w_ = 0.0;
    acc_abs_w_ = 0.0;
}

void StructureFactor::accumulate(const ParticleBatch& configs,
                                 std::span<const double> weights) {
    const std::size_t nw = configs.num_walkers(), nk = kx_.size(), ng = groups_.size();
    const auto pos = configs.positions();

    for (std::size_t w = 0; w < nw; ++w) {
        const double wt = weights[w];
        for (std::size_t g = 0; g < ng; ++g) {
            double* __restrict re = rho_re_.data() + g * nk;
            double* __restrict im = rho_im_.data() + g * nk;
            std::fill_n(re, nk, 0.0);
            std::fill_n(im, nk, 0.0);
            for (std::size_t i = groups_[g].first; i < groups_[g].last; ++i) {
                phases_(pos[(i * 2 + 0) * nw + w], pos[(i * 2 + 1) * nw + w], c_.data(),
                        s_.data());
#pragma omp simd
                for (std::size_t k = 0; k < nk; ++k) {
                    re[k] += c_[k];
                    im[k] += s_[k];
                }
            }
            double* __restrict ar = acc_re_.data() + g * nk;
            double* __restrict ai = acc_im_.data() + g * nk;
#pragma omp simd
            for (std::size_t k = 0; k < nk; ++k) {
                ar[k] += wt * re[k];
                ai[k] += wt * im[k];
            }
        }
        std::size_t pid = 0;
        for (std::size_t a = 0; a < ng; ++a)
            for (std::size_t b = a; b < ng; ++b, ++pid) {
                const double* __restrict ra = rho_re_.data() + a * nk;
                const double* __restrict ia = rho_im_.data() + a * nk;
                const double* __restrict rb = rho_re_.data() + b * nk;
                const double* __restrict ib = rho_im_.data() + b * nk;
                double* __restrict acc = acc_s_.data() + pid * nk;
#pragma omp simd
                for (std::size_t k = 0; k < nk; ++k)
                    acc[k] += wt * (ra[k] * rb[k] + ia[k] * ib[k]);
            }
        acc_w_ += wt;
        acc_abs_w_ += 1.0;  // = |w| for s = +/-1
    }
}

void StructureFactor::configure_reblocking(std::size_t num_blocks) {
    if (!hist_s_.empty()) throw std::runtime_error("sofk: configure reblocking before accumulation");
    reblocking_.configure(num_blocks, n_pairs() * shell_k_.size() + 2 * groups_.size() * kx_.size());
}

std::size_t StructureFactor::reblocking_storage_bytes(std::size_t num_blocks) const {
    return EstimatorReblocking::storage_bytes(num_blocks,
        n_pairs() * shell_k_.size() + 2 * groups_.size() * kx_.size());
}

void StructureFactor::stop_block() {
    const std::size_t nk = kx_.size(), ns = shell_k_.size(), ng = groups_.size();
    std::vector<double> s(n_pairs() * ns, 0.0), re(ng * nk, 0.0), im(ng * nk, 0.0);
    // A signed block is normalized by sum|w| and carries its mean sign.
    const bool signed_block = reblocking_.signed_mode();
    const double norm_weight = signed_block ? acc_abs_w_ : acc_w_;
    if (norm_weight > 0.0) {
        const double inv_w = 1.0 / norm_weight;
        std::size_t pid = 0;
        for (std::size_t a = 0; a < ng; ++a)
            for (std::size_t b = a; b < ng; ++b, ++pid) {
                const double na = double(groups_[a].last - groups_[a].first);
                const double nb = double(groups_[b].last - groups_[b].first);
                const double norm = (na > 0 && nb > 0) ? 1.0 / std::sqrt(na * nb) : 0.0;
                for (std::size_t k = 0; k < nk; ++k)
                    s[pid * ns + shell_of_[k]] +=
                        acc_s_[pid * nk + k] * inv_w * norm / double(shell_count_[shell_of_[k]]);
            }
        for (std::size_t q = 0; q < ng * nk; ++q) {
            re[q] = acc_re_[q] * inv_w;
            im[q] = acc_im_[q] * inv_w;
        }
    }
    if (reblocking_.enabled()) {
        s.insert(s.end(), re.begin(), re.end());
        s.insert(s.end(), im.begin(), im.end());
        reblocking_.append(s, signed_block && acc_abs_w_ > 0.0 ? acc_w_ / acc_abs_w_ : 1.0);
        return;
    }
    hist_s_.push_back(std::move(s));
    hist_re_.push_back(std::move(re));
    hist_im_.push_back(std::move(im));
}

void StructureFactor::collect(const Estimator& other) {
    const auto& o = dynamic_cast<const StructureFactor&>(other);
    if (reblocking_.enabled() || o.reblocking_.enabled()) {
        reblocking_.collect(o.reblocking_);
        return;
    }
    hist_s_.insert(hist_s_.end(), o.hist_s_.begin(), o.hist_s_.end());
    hist_re_.insert(hist_re_.end(), o.hist_re_.begin(), o.hist_re_.end());
    hist_im_.insert(hist_im_.end(), o.hist_im_.begin(), o.hist_im_.end());
}

void StructureFactor::finalize(RunResult& out) const {
    if (hist_s_.empty() && !reblocking_.enabled()) return;
    const std::size_t nk = kx_.size(), ns = shell_k_.size(), ng = groups_.size();
    const std::size_t nb = hist_s_.size();

    // Mean and block error of entry q over the finished blocks of `hist`.
    std::vector<double> per_block(nb);
    auto stat = [&](const std::vector<std::vector<double>>& hist, std::size_t q) {
        for (std::size_t i = 0; i < nb; ++i) per_block[i] = hist[i][q];
        return numerics::block_mean_error(per_block);
    };
    auto emit = [&](const std::string& name, const std::vector<std::vector<double>>& hist,
                    std::size_t offset, std::size_t count) {
        if (reblocking_.enabled()) {
            const std::size_t base = &hist == &hist_s_ ? 0 :
                (&hist == &hist_re_ ? n_pairs() * ns : n_pairs() * ns + ng * nk);
            reblocking_.emit(out, name, base + offset, count);
            return;
        }
        std::vector<double> mean(count), err(count);
        for (std::size_t q = 0; q < count; ++q) {
            const numerics::MeanError m = stat(hist, offset + q);
            mean[q] = m.mean;
            err[q]  = m.error;
        }
        out.add_series(name, std::move(mean));
        out.add_series(name + "_error", std::move(err));
    };

    out.add_series("sofk_k", shell_k_);
    out.add_series("sofk_shell_count",
                   std::vector<double>(shell_count_.begin(), shell_count_.end()));
    std::size_t pid = 0;
    for (std::size_t a = 0; a < ng; ++a)
        for (std::size_t b = a; b < ng; ++b, ++pid)
            emit("sofk_" + groups_[a].name + "_" + groups_[b].name, hist_s_, pid * ns, ns);

    out.add_series("sofk_kx", kx_);
    out.add_series("sofk_ky", ky_);
    out.add_series("sofk_shell", std::vector<double>(shell_of_.begin(), shell_of_.end()));
    for (std::size_t g = 0; g < ng; ++g) {
        emit("rhok_" + groups_[g].name + "_re", hist_re_, g * nk, nk);
        emit("rhok_" + groups_[g].name + "_im", hist_im_, g * nk, nk);
    }
}

}  // namespace alea
