// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "wavefunction/complex_slater_determinant.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

#include "population/walker_set.h"

namespace alea {

namespace {
// |det| floor so ln|det| stays finite on an (improbable) exact zero of ψ.
constexpr double kDetFloor = 1e-300;
}  // namespace

ComplexSlaterDeterminant::ComplexSlaterDeterminant(std::vector<Block> blocks)
    : blocks_(std::move(blocks)) {
    for (const Block& b : blocks_)
        if (b.orbitals->size() != b.last - b.first)
            throw std::invalid_argument(
                "complex_slater_determinant: block [" + std::to_string(b.first) +
                ", " + std::to_string(b.last) + ") needs " +
                std::to_string(b.last - b.first) + " orbitals, got " +
                std::to_string(b.orbitals->size()));
}

std::unique_ptr<WavefunctionComponent> ComplexSlaterDeterminant::clone() const {
    std::vector<Block> bl;
    bl.reserve(blocks_.size());
    for (const Block& b : blocks_)
        bl.push_back(Block{b.first, b.last, b.orbitals->clone()});
    return std::make_unique<ComplexSlaterDeterminant>(std::move(bl));
}

std::size_t ComplexSlaterDeterminant::block_of(std::size_t p) const noexcept {
    for (std::size_t b = 0; b < blocks_.size(); ++b)
        if (p >= blocks_[b].first && p < blocks_[b].last) return b;
    return npos;
}

void ComplexSlaterDeterminant::resize_state(std::size_t nw, std::size_t dim) {
    dim_ = dim;
    nw_  = nw;
    d_inv_.resize(blocks_.size());
    std::size_t maxN = 0;
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const std::size_t N = blocks_[b].last - blocks_[b].first;
        d_inv_[b].assign(N * N * nw, cplx(0.0, 0.0));
        maxN = std::max(maxN, N);
    }
    auto size_rows = [&](Rows& r) {
        r.vr.assign(maxN * nw, 0.0);
        r.vi.assign(maxN * nw, 0.0);
        r.gr.assign(maxN * dim * nw, 0.0);
        r.gi.assign(maxN * dim * nw, 0.0);
        r.lr.assign(maxN * dim * nw, 0.0);
        r.li.assign(maxN * dim * nw, 0.0);
    };
    rows_.resize(maxN);
    for (Rows& r : rows_) size_rows(r);
    size_rows(nrow_);
}

void ComplexSlaterDeterminant::prepare(std::size_t nw, std::size_t np,
                                       std::size_t dim) {
    resize_caches(nw, np, dim);
    phase_grad_c_.assign(np * dim * nw, 0.0);
    phase_lap_c_.assign(np * dim * nw, 0.0);
    resize_state(nw, dim);
}

void ComplexSlaterDeterminant::fill_rows(const WalkerSet& batch, std::size_t b,
                                         std::size_t moved, std::size_t w_begin,
                                         std::size_t w_end) {
    const Block& blk = blocks_[b];
    const std::size_t N   = blk.last - blk.first;
    const std::size_t nw  = batch.num_walkers();
    const std::size_t dim = batch.dim();
    const std::span<const double> pos = batch.positions();
    for (std::size_t kr = 0; kr < N; ++kr) {
        const std::size_t particle = blk.first + kr;
        const std::span<const double> src =
            particle == moved ? std::span<const double>(
                                    const_cast<WalkerSet&>(batch).new_pos())
                              : pos.subspan(particle * dim * nw, dim * nw);
        Rows& r = rows_[kr];
        blk.orbitals->evaluate(src, dim, {r.vr, r.vi}, {r.gr, r.gi}, {r.lr, r.li},
                               w_begin, w_end);
    }
}

void ComplexSlaterDeterminant::evaluate_batch(WalkerSet& batch) {
    const std::size_t nw  = batch.num_walkers();
    const std::size_t np  = batch.n_particles();
    const std::size_t dim = batch.dim();
    prepare(nw, np, dim);

    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const Block& blk = blocks_[b];
        const std::size_t N = blk.last - blk.first;
        fill_rows(batch, b, npos, 0, nw);
        cplx* __restrict dinv = d_inv_[b].data();

        for (std::size_t w = 0; w < nw; ++w) {
            Eigen::MatrixXcd D(N, N);
            for (std::size_t kr = 0; kr < N; ++kr)
                for (std::size_t i = 0; i < N; ++i)
                    D(kr, i) = cplx(rows_[kr].vr[i * nw + w], rows_[kr].vi[i * nw + w]);
            const Eigen::MatrixXcd A = D.inverse();
            log_psi_c_[w] += std::log(std::max(std::abs(D.determinant()), kDetFloor));
            for (std::size_t a = 0; a < N; ++a)
                for (std::size_t c = 0; c < N; ++c)
                    dinv[(a * N + c) * nw + w] = A(a, c);

            for (std::size_t kr = 0; kr < N; ++kr) {
                const Rows& r = rows_[kr];
                const std::size_t particle = blk.first + kr;
                for (std::size_t d = 0; d < dim; ++d) {
                    cplx G(0.0, 0.0), Lraw(0.0, 0.0);
                    for (std::size_t m = 0; m < N; ++m) {
                        const std::size_t oi = (m * dim + d) * nw + w;
                        G    += A(m, kr) * cplx(r.gr[oi], r.gi[oi]);
                        Lraw += A(m, kr) * cplx(r.lr[oi], r.li[oi]);
                    }
                    const cplx L = Lraw - G * G;
                    const std::size_t idx = (particle * dim + d) * nw + w;
                    grad_c_[idx]      += G.real();
                    lap_c_[idx]       += L.real();
                    phase_grad_c_[idx] += G.imag();
                    phase_lap_c_[idx]  += L.imag();
                }
            }
        }
    }

    std::span<double> b_logpsi = batch.log_psi();
    for (std::size_t w = 0; w < nw; ++w) b_logpsi[w] += log_psi_c_[w];
    auto add = [](std::span<double> dst, const std::vector<double>& src) {
        for (std::size_t k = 0; k < dst.size(); ++k) dst[k] += src[k];
    };
    add(batch.gradients(), grad_c_);
    add(batch.laplacians(), lap_c_);
    add(batch.phase_gradients(), phase_grad_c_);
    add(batch.phase_laplacians(), phase_lap_c_);
}

void ComplexSlaterDeterminant::ratio_batch(WalkerSet& batch, std::size_t p,
                                           std::size_t w_begin, std::size_t w_end) {
    const std::size_t nw  = batch.num_walkers();
    const std::size_t dim = batch.dim();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end) return;
    const std::size_t b = block_of(p);
    if (b == npos) return;   // particle in no block → ratio 1

    const Block& blk = blocks_[b];
    const std::size_t N    = blk.last - blk.first;
    const std::size_t jrow = p - blk.first;
    blk.orbitals->evaluate(batch.new_pos(), dim, {nrow_.vr, nrow_.vi},
                           {nrow_.gr, nrow_.gi}, {nrow_.lr, nrow_.li}, w_begin, w_end);

    std::span<double> log_ratios = batch.log_ratios();
    std::span<double> phase      = batch.phase_diffs();
    std::span<double> new_grad   = batch.new_grad();
    std::span<double> new_lap    = batch.new_lap();
    std::span<double> new_pgrad  = batch.new_phase_grad();
    std::span<double> new_plap   = batch.new_phase_lap();
    const cplx* __restrict dinv = d_inv_[b].data();

    for (std::size_t w = w_begin; w < w_end; ++w) {
        cplx R(0.0, 0.0);
        for (std::size_t m = 0; m < N; ++m)
            R += dinv[(m * N + jrow) * nw + w] *
                 cplx(nrow_.vr[m * nw + w], nrow_.vi[m * nw + w]);
        const double absR = std::abs(R);
        log_ratios[w] += std::log(std::max(absR, kDetFloor));
        phase[w]      += std::arg(R);   // continuous phase; no node gate (fixed phase)

        // A'_{m,jrow} = A_{m,jrow}/R, so ∇ln ψ' = (Σ_m A_{m,jrow} ∇φ_m(r'))/R.
        const cplx invR = absR > 0.0 ? 1.0 / R : cplx(0.0, 0.0);
        for (std::size_t d = 0; d < dim; ++d) {
            cplx G(0.0, 0.0), Lraw(0.0, 0.0);
            for (std::size_t m = 0; m < N; ++m) {
                const cplx a = dinv[(m * N + jrow) * nw + w];
                const std::size_t oi = (m * dim + d) * nw + w;
                G    += a * cplx(nrow_.gr[oi], nrow_.gi[oi]);
                Lraw += a * cplx(nrow_.lr[oi], nrow_.li[oi]);
            }
            const cplx Gp = invR * G;
            const cplx Lp = invR * Lraw - Gp * Gp;
            new_grad[d * nw + w]  += Gp.real();
            new_lap[d * nw + w]   += Lp.real();
            new_pgrad[d * nw + w] += Gp.imag();
            new_plap[d * nw + w]  += Lp.imag();
        }
    }
}

void ComplexSlaterDeterminant::apply_move_batch(WalkerSet& batch, std::size_t p,
                                                std::span<const bool> accept,
                                                std::size_t w_begin,
                                                std::size_t w_end) {
    const std::size_t nw  = batch.num_walkers();
    const std::size_t dim = batch.dim();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end) return;
    const std::size_t b = block_of(p);
    if (b == npos) return;

    const Block& blk = blocks_[b];
    const std::size_t N    = blk.last - blk.first;
    const std::size_t jrow = p - blk.first;

    // Rows at the post-move configuration: p at new_pos (unwrapped — the
    // WalkerSet wraps afterwards, which only re-gauges this row), the rest
    // where they are.
    fill_rows(batch, b, p, w_begin, w_end);

    std::span<double> b_logpsi = batch.log_psi();
    std::span<double> b_grad   = batch.gradients();
    std::span<double> b_lap    = batch.laplacians();
    std::span<double> b_pgrad  = batch.phase_gradients();
    std::span<double> b_plap   = batch.phase_laplacians();
    cplx* __restrict dinv = d_inv_[b].data();

    for (std::size_t w = w_begin; w < w_end; ++w) {
        if (!accept[w]) continue;

        // ln|R| from the OLD inverse → modulus increment.
        cplx R(0.0, 0.0);
        for (std::size_t m = 0; m < N; ++m)
            R += dinv[(m * N + jrow) * nw + w] *
                 cplx(rows_[jrow].vr[m * nw + w], rows_[jrow].vi[m * nw + w]);
        const double dlog = std::log(std::max(std::abs(R), kDetFloor));
        log_psi_c_[w] += dlog;
        b_logpsi[w]   += dlog;

        // Full re-inversion (N tiny), as in SlaterDeterminant.
        Eigen::MatrixXcd D(N, N);
        for (std::size_t kr = 0; kr < N; ++kr)
            for (std::size_t i = 0; i < N; ++i)
                D(kr, i) = cplx(rows_[kr].vr[i * nw + w], rows_[kr].vi[i * nw + w]);
        const Eigen::MatrixXcd A = D.inverse();
        for (std::size_t a = 0; a < N; ++a)
            for (std::size_t c = 0; c < N; ++c)
                dinv[(a * N + c) * nw + w] = A(a, c);

        // Partners (kr != jrow) push their delta into the batch totals; the
        // moved particle only refreshes the cache — WalkerSet::apply_move
        // overwrites its totals from new_grad/new_phase_grad (equal values).
        for (std::size_t kr = 0; kr < N; ++kr) {
            const Rows& r = rows_[kr];
            const std::size_t particle = blk.first + kr;
            for (std::size_t d = 0; d < dim; ++d) {
                cplx G(0.0, 0.0), Lraw(0.0, 0.0);
                for (std::size_t m = 0; m < N; ++m) {
                    const std::size_t oi = (m * dim + d) * nw + w;
                    G    += A(m, kr) * cplx(r.gr[oi], r.gi[oi]);
                    Lraw += A(m, kr) * cplx(r.lr[oi], r.li[oi]);
                }
                const cplx Lc = Lraw - G * G;
                const std::size_t idx = (particle * dim + d) * nw + w;
                if (kr != jrow) {
                    b_grad[idx]  += G.real() - grad_c_[idx];
                    b_lap[idx]   += Lc.real() - lap_c_[idx];
                    b_pgrad[idx] += G.imag() - phase_grad_c_[idx];
                    b_plap[idx]  += Lc.imag() - phase_lap_c_[idx];
                }
                grad_c_[idx]       = G.real();
                lap_c_[idx]        = Lc.real();
                phase_grad_c_[idx] = G.imag();
                phase_lap_c_[idx]  = Lc.imag();
            }
        }
    }
}

void ComplexSlaterDeterminant::permute_caches(std::span<const std::size_t> src) {
    const std::size_t old_nw = nw_;
    WavefunctionComponent::permute_caches(src);   // log_psi_c_, grad_c_, lap_c_
    if (old_nw == 0) return;   // never evaluated — nothing committed to carry
    const std::size_t new_nw = src.size();

    // Walker-fastest gather: column j of the new layout ← column src[j].
    auto gather = [&](auto& a) {
        using T = typename std::decay_t<decltype(a)>::value_type;
        const std::size_t width = a.size() / old_nw;
        std::decay_t<decltype(a)> dst(width * new_nw, T{});
        for (std::size_t k = 0; k < width; ++k)
            for (std::size_t j = 0; j < new_nw; ++j)
                dst[k * new_nw + j] = a[k * old_nw + src[j]];
        a = std::move(dst);
    };
    for (auto& inv : d_inv_) gather(inv);
    gather(phase_grad_c_);
    gather(phase_lap_c_);

    // Row scratch is transient: resize to the new population.
    nw_ = new_nw;
    const std::size_t dim = dim_;
    std::vector<std::vector<cplx>> keep = std::move(d_inv_);
    resize_state(new_nw, dim);
    d_inv_ = std::move(keep);
}

}  // namespace alea
