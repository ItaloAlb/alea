// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "wavefunction/backflow_slater_determinant.h"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <stdexcept>
#include <string>

#include "population/walker_set.h"

namespace alea {

namespace {

constexpr double kPi = 3.14159265358979323846;
// log|ψ'/ψ| of a proposal whose matrix is singular: exp(−700) ~ 1e−304, so the
// move is never accepted, without an infinity reaching the acceptance test.
constexpr double kSingularLogRatio = -700.0;

// Pivoted Gauss–Jordan: a = m⁻¹ for a row-major N×N matrix m (destroyed),
// with log|det m| and the sign of det m from the pivots and row swaps.
// Golub & Van Loan, Matrix Computations, 4th ed., Sec. 3.1 (2013).  Returns
// false for a singular or nonfinite matrix.
bool invert(double* __restrict m, double* __restrict a, std::size_t n, double& log_det,
            double& sign) {
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = 0; c < n; ++c) a[r * n + c] = r == c ? 1.0 : 0.0;
    log_det = 0.0;
    sign = 1.0;
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot = k;
        for (std::size_t r = k + 1; r < n; ++r)
            if (std::abs(m[r * n + k]) > std::abs(m[pivot * n + k])) pivot = r;
        const double diag = m[pivot * n + k];
        if (diag == 0.0 || !std::isfinite(diag)) return false;
        log_det += std::log(std::abs(diag));
        if (diag < 0.0) sign = -sign;
        if (pivot != k) {
            std::swap_ranges(m + k * n + k, m + k * n + n, m + pivot * n + k);
            std::swap_ranges(a + k * n, a + k * n + n, a + pivot * n);
            sign = -sign;
        }
        double* __restrict m_k = m + k * n;
        double* __restrict a_k = a + k * n;
        const double inv_diag = 1.0 / diag;
        for (std::size_t c = k + 1; c < n; ++c) m_k[c] *= inv_diag;
        for (std::size_t c = 0; c < n; ++c) a_k[c] *= inv_diag;
        for (std::size_t r = 0; r < n; ++r) {
            if (r == k) continue;
            double* __restrict m_r = m + r * n;
            double* __restrict a_r = a + r * n;
            const double factor = m_r[k];
            if (factor == 0.0) continue;
#pragma omp simd
            for (std::size_t c = k + 1; c < n; ++c) m_r[c] -= factor * m_k[c];
#pragma omp simd
            for (std::size_t c = 0; c < n; ++c) a_r[c] -= factor * a_k[c];
        }
    }
    return true;
}

}  // namespace

// ── Construction ────────────────────────────────────────────────────────────

BackflowSlaterDeterminant::BackflowSlaterDeterminant(std::vector<Block> blocks,
                                                     std::vector<Channel> channels,
                                                     const SimulationCell& cell)
    : blocks_(std::move(blocks)), cell_(cell) {
    if (blocks_.empty())
        throw std::invalid_argument("backflow determinant: no determinant block");
    first_row_ = blocks_.front().first;
    std::size_t next = first_row_;
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const Block& blk = blocks_[b];
        if (blk.first != next || blk.last <= blk.first)
            throw std::invalid_argument(
                "backflow determinant: the blocks must cover one group contiguously");
        const std::size_t n = blk.last - blk.first;
        if (!blk.orbitals || blk.orbitals->size() != n)
            throw std::invalid_argument(
                "backflow determinant: each block needs exactly as many orbitals as rows");
        if (!blk.orbitals->supports_hessian())
            throw std::invalid_argument(
                "backflow determinant: the orbitals provide no Hessian, which the "
                "backflow Laplacian needs (plane_wave and gaussian do)");
        inv_offset_.push_back(inv_size_);
        inv_size_ += n * n;
        max_block_ = std::max(max_block_, n);
        for (std::size_t i = 0; i < n; ++i) block_of_row_.push_back(b);
        next = blk.last;
    }
    n_rows_ = next - first_row_;

    for (std::size_t b = 0; b < blocks_.size(); ++b)
        for (std::size_t k = 0; k < blocks_[b].orbitals->n_parameters(); ++k) {
            Parameter p = blocks_[b].orbitals->parameter(k);
            p.name = "b" + std::to_string(b) + "_" + p.name;
            params_.push_back(p);
            orbital_owner_.emplace_back(b, k);
        }
    n_orbital_params_ = params_.size();

    // Every channel displaces the rows; a same-group (like-spin) one needs a
    // group of one spin (the transformation derives its eta'(0) = 0).
    std::vector<BackflowTransformation::Channel> tc;
    for (std::size_t c = 0; c < channels.size(); ++c) {
        const Channel& ch = channels[c];
        const bool own = ch.source_first < first_row_ + n_rows_ && first_row_ < ch.source_last;
        if (own && blocks_.size() != 1)
            throw std::invalid_argument(
                "backflow channel " + std::to_string(c) + ": a same-group channel needs a "
                "group of one spin; with both spins in one group its pairs are partly "
                "like-spin (eta'(0) = 0) and partly not.  Put each spin in its own group");
        tc.push_back({first_row_, first_row_ + n_rows_, ch.source_first, ch.source_last,
                      ch.truncation, ch.params});
    }
    std::vector<std::size_t> rows(n_rows_);
    for (std::size_t i = 0; i < n_rows_; ++i) rows[i] = first_row_ + i;
    transform_ = BackflowTransformation(std::move(tc), std::move(rows), cell_);
    const auto& bf = transform_.initial_parameters();
    params_.insert(params_.end(), bf.begin(), bf.end());
}

BackflowSlaterDeterminant::BackflowSlaterDeterminant(const BackflowSlaterDeterminant& o)
    : WavefunctionComponent(o), transform_(o.transform_), cell_(o.cell_),
      first_row_(o.first_row_), n_rows_(o.n_rows_), max_block_(o.max_block_),
      block_of_row_(o.block_of_row_), inv_offset_(o.inv_offset_), inv_size_(o.inv_size_),
      dim_(o.dim_), np_(o.np_), nw_(o.nw_), params_(o.params_),
      orbital_owner_(o.orbital_owner_), n_orbital_params_(o.n_orbital_params_),
      x_(o.x_), inv_(o.inv_), f_(o.f_), log_det_(o.log_det_), sign_(o.sign_),
      px_(o.px_), pinv_(o.pinv_), pf_(o.pf_), plog_det_(o.plog_det_), psign_(o.psign_),
      pgrad_(o.pgrad_), plap_(o.plap_), pvalid_(o.pvalid_),
      proposal_laplacian_(o.proposal_laplacian_),
      // A copy runs on its own threads: fresh, unsized scratch slots.
      scratch_(o.scratch_.size()) {
    blocks_.reserve(o.blocks_.size());
    for (const Block& b : o.blocks_) blocks_.push_back({b.first, b.last, b.orbitals->clone()});
}

std::unique_ptr<WavefunctionComponent> BackflowSlaterDeterminant::clone() const {
    return std::make_unique<BackflowSlaterDeterminant>(*this);
}

void BackflowSlaterDeterminant::rebuild() {
    for (std::size_t q = 0; q < n_orbital_params_; ++q) {
        const auto [b, k] = orbital_owner_[q];
        blocks_[b].orbitals->set_parameter(k, params_[q].value);
    }
    transform_.set_parameters(std::span<const Parameter>(params_).subspan(n_orbital_params_));
}

// ── Sizes ───────────────────────────────────────────────────────────────────

void BackflowSlaterDeterminant::resize_state(std::size_t nw) {
    nw_ = nw;
    const std::size_t nb = blocks_.size();
    x_.assign(nw * n_rows_ * dim_, 0.0);
    inv_.assign(nw * inv_size_, 0.0);
    f_.assign(nw * n_rows_ * dim_, 0.0);
    log_det_.assign(nw * nb, 0.0);
    sign_.assign(nw * nb, 1.0);
    px_.assign(nw * n_rows_ * dim_, 0.0);
    pinv_.assign(nw * inv_size_, 0.0);
    pf_.assign(nw * n_rows_ * dim_, 0.0);
    plog_det_.assign(nw * nb, 0.0);
    psign_.assign(nw * nb, 1.0);
    pgrad_.assign(nw * dim_, 0.0);
    plap_.assign(nw * dim_, 0.0);
    pvalid_.assign(nw, 0);
}

void BackflowSlaterDeterminant::prepare(std::size_t nw, std::size_t np, std::size_t dim) {
    if (first_row_ + n_rows_ > np)
        throw std::invalid_argument("backflow determinant: rows beyond the particle count");
    transform_.prepare(np, dim);
    resize_caches(nw, np, dim);
    if (np != np_ || dim != dim_) ++scratch_generation_;   // slots resize on next use
    dim_ = dim;
    np_ = np;
    resize_state(nw);
    // One slot per thread that may call in; a slot's buffers are sized the
    // first time its thread uses it (scratch()), so a per-thread clone (VMC)
    // holds one sized slot, not one per core.
    const auto threads = static_cast<std::size_t>(
        std::max({1, omp_get_max_threads(), omp_get_num_procs()}));
    if (scratch_.size() < threads) scratch_.resize(threads);
}

void BackflowSlaterDeterminant::size_scratch(Scratch& s) const {
    const std::size_t np = np_, dim = dim_;
    const std::size_t max_channel_params = transform_.max_channel_parameters();
    std::size_t max_orbital_params = 1;
    for (const Block& b : blocks_)
        max_orbital_params = std::max(max_orbital_params, b.orbitals->n_parameters());
    const std::size_t n = max_block_, rows = n_rows_;
    {
        s.pos.assign(np * dim, 0.0);
        s.x.assign(rows * dim, 0.0);
        s.val.assign(rows * n, 0.0);
        s.grad.assign(rows * n * dim, 0.0);
        s.hess.assign(rows * n * dim * dim, 0.0);
        transform_.size_tables(s.tables);
        s.pderiv.assign(transform_.n_parameters(), 0.0);
        s.work.assign(n * n, 0.0);
        s.at.assign(n * n, 0.0);
        s.gmat.assign(dim * inv_size_, 0.0);
        s.grad_a.assign(n * n, 0.0);
        s.t.assign(rows * dim * dim, 0.0);
        s.v.assign(rows * dim, 0.0);
        s.u.assign(n * n, 0.0);
        s.w.assign(n * n, 0.0);
        s.param.assign(max_channel_params, 0.0);
        s.dval.assign(n * max_orbital_params, 0.0);
        s.out_grad.assign(np * dim, 0.0);
        s.out_lap.assign(np * dim, 0.0);
    }
    s.generation = scratch_generation_;
}

std::size_t BackflowSlaterDeterminant::walker_state_bytes() const noexcept {
    if (nw_ == 0) return WavefunctionComponent::walker_state_bytes();
    const std::size_t per_walker = 2 * n_rows_ * dim_ + inv_size_ + 2 * blocks_.size();
    return WavefunctionComponent::walker_state_bytes() + per_walker * sizeof(double);
}

BackflowSlaterDeterminant::Scratch& BackflowSlaterDeterminant::scratch() const {
    const auto t = static_cast<std::size_t>(omp_get_thread_num());
    if (t >= scratch_.size())
        throw std::logic_error("backflow determinant: OpenMP thread " + std::to_string(t) +
                               " beyond the " + std::to_string(scratch_.size()) +
                               " scratch slots (one per processor); use fewer threads");
    Scratch& s = scratch_[t];
    // First use by this thread (or the sizes changed): size once.  Never
    // again in the sweeps that follow.
    if (s.generation != scratch_generation_) size_scratch(s);
    return s;
}

bool BackflowSlaterDeterminant::affects(std::size_t p) const noexcept {
    return transform_.affects(p);
}

void BackflowSlaterDeterminant::permute_caches(std::span<const std::size_t> src) {
    WavefunctionComponent::permute_caches(src);
    if (nw_ == 0) return;
    const std::size_t new_nw = src.size();
    const auto gather = [&](std::vector<double>& v, std::size_t width) {
        std::vector<double> out(new_nw * width);
        for (std::size_t c = 0; c < new_nw; ++c)
            std::copy_n(v.begin() + static_cast<std::ptrdiff_t>(src[c] * width), width,
                        out.begin() + static_cast<std::ptrdiff_t>(c * width));
        v.swap(out);
    };
    gather(x_, n_rows_ * dim_);
    gather(inv_, inv_size_);
    gather(f_, n_rows_ * dim_);
    gather(log_det_, blocks_.size());
    gather(sign_, blocks_.size());
    nw_ = new_nw;
    // Pending state is transient: the next ratio_batch rebuilds it.
    const std::size_t nb = blocks_.size();
    px_.assign(new_nw * n_rows_ * dim_, 0.0);
    pinv_.assign(new_nw * inv_size_, 0.0);
    pf_.assign(new_nw * n_rows_ * dim_, 0.0);
    plog_det_.assign(new_nw * nb, 0.0);
    psign_.assign(new_nw * nb, 1.0);
    pgrad_.assign(new_nw * dim_, 0.0);
    plap_.assign(new_nw * dim_, 0.0);
    pvalid_.assign(new_nw, 0);
}

// ── One configuration ───────────────────────────────────────────────────────

void BackflowSlaterDeterminant::load_positions(const WalkerSet& batch, std::size_t w,
                                               Scratch& s) const {
    const std::size_t nw = batch.num_walkers();
    const std::span<const double> pos = batch.positions();
    for (std::size_t k = 0; k < np_; ++k)
        for (std::size_t d = 0; d < dim_; ++d)
            s.pos[k * dim_ + d] = pos[(k * dim_ + d) * nw + w];
}

void BackflowSlaterDeterminant::evaluate_rows(Scratch& s) const {
    const std::size_t dim = dim_, n = max_block_;
    for (std::size_t i = 0; i < n_rows_; ++i) {
        const OrbitalSet& orb = *blocks_[block_of_row_[i]].orbitals;
        const std::size_t nb = orb.size();
        orb.evaluate_hessian(std::span<const double>(s.x.data() + i * dim, dim), dim,
                             std::span<double>(s.val.data() + i * n, nb),
                             std::span<double>(s.grad.data() + i * n * dim, nb * dim),
                             std::span<double>(s.hess.data() + i * n * dim * dim, nb * dim * dim),
                             0, 1);
    }
}

bool BackflowSlaterDeterminant::invert_blocks(Scratch& s, double* inv, double* log_det,
                                              double* sign) const {
    const std::size_t n_max = max_block_;
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const std::size_t n = blocks_[b].last - blocks_[b].first;
        const std::size_t row0 = blocks_[b].first - first_row_;
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t m = 0; m < n; ++m)
                s.work[r * n + m] = s.val[(row0 + r) * n_max + m];   // M[r][m] = φ_m(x_r)
        if (!invert(s.work.data(), inv + inv_offset_[b], n, log_det[b], sign[b])) return false;
    }
    return true;
}

void BackflowSlaterDeterminant::compute_f(const Scratch& s, const double* inv,
                                          double* f) const {
    const std::size_t dim = dim_, n_max = max_block_;
    for (std::size_t i = 0; i < n_rows_; ++i) {
        const std::size_t b = block_of_row_[i];
        const std::size_t n = blocks_[b].last - blocks_[b].first;
        const std::size_t r = first_row_ + i - blocks_[b].first;
        const double* A = inv + inv_offset_[b];
        for (std::size_t a = 0; a < dim; ++a) {
            double acc = 0.0;
            for (std::size_t m = 0; m < n; ++m)
                acc += A[m * n + r] * s.grad[(i * n_max + m) * dim + a];
            f[i * dim + a] = acc;
        }
    }
}

// T_i^{αγ} = Σ_m A_mi ∂_α∂_γ φ_m(x_i) of every row into s.t.
void BackflowSlaterDeterminant::hessian_traces(Scratch& s, const double* inv) const {
    const std::size_t dd = dim_ * dim_, n_max = max_block_;
    for (std::size_t i = 0; i < n_rows_; ++i) {
        const std::size_t b = block_of_row_[i];
        const std::size_t n = blocks_[b].last - blocks_[b].first;
        const std::size_t r = first_row_ + i - blocks_[b].first;
        const double* A = inv + inv_offset_[b];
        for (std::size_t ag = 0; ag < dd; ++ag) {
            double acc = 0.0;
            for (std::size_t m = 0; m < n; ++m) acc += A[m * n + r] * s.hess[(i * n_max + m) * dd + ag];
            s.t[i * dd + ag] = acc;
        }
    }
}

void BackflowSlaterDeterminant::particle_laplacian(Scratch& s, const double* inv,
                                                   const double* f, std::size_t k,
                                                   double* out) const {
    const std::size_t dim = dim_, dd = dim * dim, n_max = max_block_;
    hessian_traces(s, inv);
    for (std::size_t beta = 0; beta < dim; ++beta) {
        double lap = transform_.jacobian_column(s.tables, f, k, beta, s.v.data());
        // Diagonal (orbital Hessian) term.
        for (std::size_t i = 0; i < n_rows_; ++i) {
            const double* v = s.v.data() + i * dim;
            const double* T = s.t.data() + i * dd;
            for (std::size_t a = 0; a < dim; ++a)
                for (std::size_t g = 0; g < dim; ++g) lap += v[a] * v[g] * T[a * dim + g];
        }
        // −Σ_{i,j} U_ji U_ij per block, U_ji = Σ_m A_mj W_im, W_im = Σ_α v_iα ∂_α φ_m(x_i).
        for (std::size_t b = 0; b < blocks_.size(); ++b) {
            const std::size_t n = blocks_[b].last - blocks_[b].first;
            const std::size_t row0 = blocks_[b].first - first_row_;
            const double* A = inv + inv_offset_[b];
            bool any = false;
            for (std::size_t r = 0; r < n; ++r) {
                const double* v = s.v.data() + (row0 + r) * dim;
                for (std::size_t m = 0; m < n; ++m) {
                    double acc = 0.0;
                    for (std::size_t a = 0; a < dim; ++a)
                        acc += v[a] * s.grad[((row0 + r) * n_max + m) * dim + a];
                    s.w[r * n + m] = acc;
                    any = any || acc != 0.0;
                }
            }
            if (!any) continue;
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t m = 0; m < n; ++m) s.at[j * n + m] = A[m * n + j];
            for (std::size_t j = 0; j < n; ++j) {
                const double* at_j = s.at.data() + j * n;
                for (std::size_t r = 0; r < n; ++r) {
                    const double* w_r = s.w.data() + r * n;
                    double acc = 0.0;
#pragma omp simd reduction(+ : acc)
                    for (std::size_t m = 0; m < n; ++m) acc += at_j[m] * w_r[m];
                    s.u[j * n + r] = acc;
                }
            }
            double uu = 0.0;
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t r = 0; r < n; ++r) uu += s.u[j * n + r] * s.u[r * n + j];
            lap -= uu;
        }
        out[beta] = lap;
    }
}

void BackflowSlaterDeterminant::all_derivatives(Scratch& s, const double* inv, const double* f,
                                                double* grad, double* lap) const {
    const std::size_t dim = dim_, dd = dim * dim, n_max = max_block_;
    hessian_traces(s, inv);
    // G^α[j][r] = Σ_m A_mj ∂_α φ_m(x_r) of every block, at gmat[α*inv_size + off + j*n + r].
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const std::size_t n = blocks_[b].last - blocks_[b].first;
        const std::size_t row0 = blocks_[b].first - first_row_;
        const double* A = inv + inv_offset_[b];
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t m = 0; m < n; ++m) s.at[j * n + m] = A[m * n + j];
        for (std::size_t a = 0; a < dim; ++a) {
            for (std::size_t r = 0; r < n; ++r)
                for (std::size_t m = 0; m < n; ++m)
                    s.grad_a[r * n + m] = s.grad[((row0 + r) * n_max + m) * dim + a];
            double* G = s.gmat.data() + a * inv_size_ + inv_offset_[b];
            for (std::size_t j = 0; j < n; ++j) {
                const double* at_j = s.at.data() + j * n;
                for (std::size_t r = 0; r < n; ++r) {
                    const double* g_r = s.grad_a.data() + r * n;
                    double acc = 0.0;
#pragma omp simd reduction(+ : acc)
                    for (std::size_t m = 0; m < n; ++m) acc += at_j[m] * g_r[m];
                    G[j * n + r] = acc;
                }
            }
        }
    }
    std::fill(grad, grad + np_ * dim, 0.0);
    std::fill(lap, lap + np_ * dim, 0.0);
    for (std::size_t k = 0; k < np_; ++k) {
        if (!affects(k)) continue;
        transform_.gradient(s.tables, f, k, grad + k * dim);
        for (std::size_t beta = 0; beta < dim; ++beta) {
            double l = transform_.jacobian_column(s.tables, f, k, beta, s.v.data());
            for (std::size_t i = 0; i < n_rows_; ++i) {
                const double* v = s.v.data() + i * dim;
                const double* T = s.t.data() + i * dd;
                for (std::size_t a = 0; a < dim; ++a)
                    for (std::size_t g = 0; g < dim; ++g) l += v[a] * v[g] * T[a * dim + g];
            }
            // −Σ_{i,j} U_ji U_ij with U_ji = Σ_α v_iα G^α_ji (block by block).
            for (std::size_t b = 0; b < blocks_.size(); ++b) {
                const std::size_t n = blocks_[b].last - blocks_[b].first;
                const std::size_t row0 = blocks_[b].first - first_row_;
                for (std::size_t j = 0; j < n; ++j)
                    for (std::size_t r = 0; r < n; ++r) {
                        double acc = 0.0;
                        for (std::size_t a = 0; a < dim; ++a)
                            acc += s.v[(row0 + r) * dim + a] *
                                   s.gmat[a * inv_size_ + inv_offset_[b] + j * n + r];
                        s.u[j * n + r] = acc;
                    }
                double uu = 0.0;
                for (std::size_t j = 0; j < n; ++j)
                    for (std::size_t r = 0; r < n; ++r) uu += s.u[j * n + r] * s.u[r * n + j];
                l -= uu;
            }
            lap[k * dim + beta] = l;
        }
    }
}

// ── Full evaluation ─────────────────────────────────────────────────────────

void BackflowSlaterDeterminant::evaluate_batch(WalkerSet& batch) {
    const std::size_t nw = batch.num_walkers();
    const std::size_t np = batch.n_particles();
    const std::size_t dim = batch.dim();
    prepare(nw, np, dim);
    const std::size_t nb = blocks_.size();
    long singular = -1;

#pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t ws = 0; ws < static_cast<std::ptrdiff_t>(nw); ++ws) {
        const auto w = static_cast<std::size_t>(ws);
        Scratch& s = scratch();
        load_positions(batch, w, s);
        transform_.full(s.pos.data(), s.x.data(), s.tables);
        evaluate_rows(s);
        double* inv = inv_.data() + w * inv_size_;
        if (!invert_blocks(s, inv, log_det_.data() + w * nb, sign_.data() + w * nb)) {
#pragma omp critical(backflow_singular)
            singular = static_cast<long>(w);
            continue;
        }
        std::copy(s.x.begin(), s.x.end(), x_.begin() + static_cast<std::ptrdiff_t>(w * n_rows_ * dim));
        double* f = f_.data() + w * n_rows_ * dim;
        compute_f(s, inv, f);
        all_derivatives(s, inv, f, s.out_grad.data(), s.out_lap.data());
        double log_psi = 0.0;
        for (std::size_t b = 0; b < nb; ++b) log_psi += log_det_[w * nb + b];
        log_psi_c_[w] = log_psi;
        for (std::size_t k = 0; k < np; ++k)
            for (std::size_t d = 0; d < dim; ++d) {
                grad_c_[(k * dim + d) * nw + w] = s.out_grad[k * dim + d];
                lap_c_[(k * dim + d) * nw + w]  = s.out_lap[k * dim + d];
            }
    }
    if (singular >= 0)
        throw std::runtime_error("backflow determinant: singular at the configuration of "
                                 "walker " + std::to_string(singular) +
                                 " (a node of the trial wavefunction)");

    std::span<double> b_logpsi = batch.log_psi();
    for (std::size_t w = 0; w < nw; ++w) b_logpsi[w] += log_psi_c_[w];
    std::span<double> b_grad = batch.gradients();
    for (std::size_t k = 0; k < b_grad.size(); ++k) b_grad[k] += grad_c_[k];
    std::span<double> b_lap = batch.laplacians();
    for (std::size_t k = 0; k < b_lap.size(); ++k) b_lap[k] += lap_c_[k];
}

// ── Moves ───────────────────────────────────────────────────────────────────

void BackflowSlaterDeterminant::ratio_batch(WalkerSet& batch, std::size_t p,
                                            std::size_t w_begin, std::size_t w_end) {
    const std::size_t nw = batch.num_walkers();
    const std::size_t dim = dim_, nb = blocks_.size();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end || !affects(p)) return;

    std::span<double> log_ratios = batch.log_ratios();
    std::span<double> phase = batch.phase_diffs();
    std::span<double> new_grad = batch.new_grad();
    std::span<double> new_lap = batch.new_lap();
    const std::span<const double> new_pos = batch.new_pos();
    Scratch& s = scratch();
    for (std::size_t w = w_begin; w < w_end; ++w) {
        load_positions(batch, w, s);
        double old_r[SimulationCell::kMaxDim] = {};
        for (std::size_t d = 0; d < dim; ++d) {
            old_r[d] = s.pos[p * dim + d];
            s.pos[p * dim + d] = new_pos[d * nw + w];
        }
        // x' from the committed x: O(N) pairs, not the O(N^2) of a rebuild.
        std::copy_n(x_.begin() + static_cast<std::ptrdiff_t>(w * n_rows_ * dim), n_rows_ * dim,
                    s.x.begin());
        transform_.shift(s.pos.data(), p, old_r, s.x.data());
        transform_.particle_tables(s.pos.data(), p, s.x.data(), s.tables);
        evaluate_rows(s);
        double* pinv = pinv_.data() + w * inv_size_;
        double* plog = plog_det_.data() + w * nb;
        double* psign = psign_.data() + w * nb;
        double old_log = 0.0, new_log = 0.0, sign = 1.0;
        for (std::size_t b = 0; b < nb; ++b) old_log += log_det_[w * nb + b];
        if (!invert_blocks(s, pinv, plog, psign)) {
            pvalid_[w] = 0;
            log_ratios[w] += kSingularLogRatio;
            for (std::size_t d = 0; d < dim; ++d) pgrad_[w * dim + d] = plap_[w * dim + d] = 0.0;
            continue;
        }
        pvalid_[w] = 1;
        for (std::size_t b = 0; b < nb; ++b) {
            new_log += plog[b];
            sign *= psign[b] * sign_[w * nb + b];
        }
        log_ratios[w] += new_log - old_log;
        if (sign < 0.0) phase[w] += kPi;   // node crossing → fixed-node gate rejects

        std::copy(s.x.begin(), s.x.end(), px_.begin() + static_cast<std::ptrdiff_t>(w * n_rows_ * dim));
        double* pf = pf_.data() + w * n_rows_ * dim;
        compute_f(s, pinv, pf);
        double g[SimulationCell::kMaxDim] = {}, l[SimulationCell::kMaxDim] = {};
        transform_.gradient(s.tables, pf, p, g);
        // Skipped, l = 0, when the driver refreshes before reading it.
        if (proposal_laplacian_) particle_laplacian(s, pinv, pf, p, l);
        for (std::size_t d = 0; d < dim; ++d) {
            pgrad_[w * dim + d] = g[d];
            plap_[w * dim + d] = l[d];
            new_grad[d * nw + w] += g[d];
            new_lap[d * nw + w] += l[d];
        }
    }
}

void BackflowSlaterDeterminant::commit(WalkerSet& batch, std::size_t p, std::size_t w) {
    if (!pvalid_[w])
        throw std::logic_error("backflow determinant: a move to a singular matrix was accepted");
    const std::size_t nw = batch.num_walkers(), dim = dim_, nb = blocks_.size();
    const auto copy = [w](const std::vector<double>& from, std::vector<double>& to,
                          std::size_t width) {
        std::copy_n(from.begin() + static_cast<std::ptrdiff_t>(w * width), width,
                    to.begin() + static_cast<std::ptrdiff_t>(w * width));
    };
    copy(px_, x_, n_rows_ * dim);
    copy(pinv_, inv_, inv_size_);
    copy(pf_, f_, n_rows_ * dim);
    copy(plog_det_, log_det_, nb);
    copy(psign_, sign_, nb);
    double log_psi = 0.0;
    for (std::size_t b = 0; b < nb; ++b) log_psi += log_det_[w * nb + b];
    batch.log_psi()[w] += log_psi - log_psi_c_[w];
    log_psi_c_[w] = log_psi;
    // The moved particle's derivatives are exact and already in new_grad /
    // new_lap, which WalkerSet::apply_move writes into its totals.
    for (std::size_t d = 0; d < dim; ++d) {
        grad_c_[(p * dim + d) * nw + w] = pgrad_[w * dim + d];
        lap_c_[(p * dim + d) * nw + w]  = plap_[w * dim + d];
    }
}

void BackflowSlaterDeterminant::apply_move_batch_deferred(WalkerSet& batch, std::size_t p,
                                                          std::span<const bool> accept,
                                                          std::size_t w_begin,
                                                          std::size_t w_end) {
    const std::size_t nw = batch.num_walkers();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end || !affects(p)) return;
    for (std::size_t w = w_begin; w < w_end; ++w)
        if (accept[w]) commit(batch, p, w);
}

void BackflowSlaterDeterminant::apply_move_batch(WalkerSet& batch, std::size_t p,
                                                 std::span<const bool> accept,
                                                 std::size_t w_begin, std::size_t w_end) {
    const std::size_t nw = batch.num_walkers();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end || !affects(p)) return;
    for (std::size_t w = w_begin; w < w_end; ++w) {
        if (!accept[w]) continue;
        commit(batch, p, w);
        // Positions still hold the old r_p (WalkerSet::apply_move follows), so
        // the refresh reads the proposal for p.
        refresh_walker(batch, w, p);
    }
}

// Every particle's derivatives of walker w from the committed state, into the
// component caches and, by difference, the batch totals.  `moved` (or npos):
// its position is read from new_pos and its derivatives are left as committed.
void BackflowSlaterDeterminant::refresh_walker(WalkerSet& batch, std::size_t w,
                                               std::size_t moved) {
    const std::size_t nw = batch.num_walkers(), dim = dim_;
    Scratch& s = scratch();
    load_positions(batch, w, s);
    if (moved != npos) {
        const std::span<const double> new_pos = batch.new_pos();
        for (std::size_t d = 0; d < dim; ++d) s.pos[moved * dim + d] = new_pos[d * nw + w];
    }
    transform_.full(s.pos.data(), s.x.data(), s.tables);
    // The rebuilt x replaces the committed one, whose incremental updates
    // (shift_quasi_coordinates) have accumulated rounding since the last rebuild.
    std::copy(s.x.begin(), s.x.end(), x_.begin() + static_cast<std::ptrdiff_t>(w * n_rows_ * dim));
    evaluate_rows(s);
    all_derivatives(s, inv_.data() + w * inv_size_, f_.data() + w * n_rows_ * dim,
                    s.out_grad.data(), s.out_lap.data());
    std::span<double> bg = batch.gradients();
    std::span<double> bl = batch.laplacians();
    for (std::size_t k = 0; k < np_; ++k) {
        if (k == moved) continue;
        for (std::size_t d = 0; d < dim; ++d) {
            const std::size_t j = (k * dim + d) * nw + w;
            const double g = s.out_grad[k * dim + d], l = s.out_lap[k * dim + d];
            // Already-current fields stay untouched (subtracting and re-adding
            // could cancel a smaller component's bits).
            if (g != grad_c_[j]) bg[j] = (bg[j] - grad_c_[j]) + g;
            if (l != lap_c_[j]) bl[j] = (bl[j] - lap_c_[j]) + l;
            grad_c_[j] = g;
            lap_c_[j] = l;
        }
    }
}

void BackflowSlaterDeterminant::refresh_particle_derivatives(WalkerSet& batch, std::size_t p,
                                                             std::size_t w_begin,
                                                             std::size_t w_end) {
    const std::size_t nw = batch.num_walkers(), dim = dim_;
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end || !affects(p)) return;
    std::span<double> bg = batch.gradients();
    Scratch& s = scratch();
    // The gradient only: it is what a drift reads.  The Laplacian of p stays
    // as committed until refresh_derivatives (it needs every G^α, O(d N³)).
    for (std::size_t w = w_begin; w < w_end; ++w) {
        load_positions(batch, w, s);
        transform_.particle_tables(s.pos.data(), p, s.x.data(), s.tables);
        double g[SimulationCell::kMaxDim] = {};
        transform_.gradient(s.tables, f_.data() + w * n_rows_ * dim, p, g);
        for (std::size_t d = 0; d < dim; ++d) {
            const std::size_t j = (p * dim + d) * nw + w;
            if (g[d] != grad_c_[j]) bg[j] = (bg[j] - grad_c_[j]) + g[d];
            grad_c_[j] = g[d];
        }
    }
}

void BackflowSlaterDeterminant::refresh_derivatives(WalkerSet& batch, std::size_t w_begin,
                                                    std::size_t w_end) {
    const std::size_t nw = batch.num_walkers();
    if (w_end > nw) w_end = nw;
    for (std::size_t w = w_begin; w < w_end; ++w) refresh_walker(batch, w, npos);
}

// ── Parameter derivatives ───────────────────────────────────────────────────

void BackflowSlaterDeterminant::compute_parameter_derivatives(const WalkerSet& batch,
                                                              std::span<double> derivs) const {
    const std::size_t nw = batch.num_walkers(), dim = dim_;
    // Column of each non-fixed parameter in derivs ([k_opt * nw + w]).
    std::vector<std::size_t> column(params_.size(), npos);
    std::size_t n_opt = 0;
    for (std::size_t q = 0; q < params_.size(); ++q)
        if (!params_[q].is_fixed) column[q] = n_opt++;
    if (n_opt == 0) return;
    Scratch& s = scratch();
    for (std::size_t w = 0; w < nw; ++w) {
        load_positions(batch, w, s);
        const double* f = f_.data() + w * n_rows_ * dim;
        // Backflow parameters: Σ_i (F_i · r_ij) ∂η(r_ij)/∂p.
        transform_.parameter_derivatives(s.pos.data(), f, s.pderiv, s.param);
        for (std::size_t q = 0; q < transform_.n_parameters(); ++q)
            if (column[n_orbital_params_ + q] != npos)
                derivs[column[n_orbital_params_ + q] * nw + w] = s.pderiv[q];
        // Orbital parameters: Σ_i Σ_m A_mi ∂φ_m(x_i)/∂p at the committed x.
        for (std::size_t q = 0; q < n_orbital_params_; ++q) {
            if (column[q] == npos) continue;
            const auto [b, k] = orbital_owner_[q];
            const OrbitalSet& orb = *blocks_[b].orbitals;
            const std::size_t n = orb.size(), n_par = orb.n_parameters();
            const double* A = inv_.data() + w * inv_size_ + inv_offset_[b];
            double acc = 0.0;
            for (std::size_t r = 0; r < n; ++r) {
                const std::size_t i = blocks_[b].first - first_row_ + r;
                orb.evaluate_parameter_derivatives(
                    std::span<const double>(x_.data() + (w * n_rows_ + i) * dim, dim), dim,
                    std::span<double>(s.dval.data(), n_par * n), {}, {}, 0, 1);
                for (std::size_t m = 0; m < n; ++m) acc += A[m * n + r] * s.dval[k * n + m];
            }
            derivs[column[q] * nw + w] = acc;
        }
    }
}

}  // namespace alea
