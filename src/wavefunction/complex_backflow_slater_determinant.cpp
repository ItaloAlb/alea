// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "wavefunction/complex_backflow_slater_determinant.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <omp.h>
#include <stdexcept>
#include <string>

#include "population/walker_set.h"

namespace alea {

namespace {

using cplx = std::complex<double>;

constexpr double kPi = 3.14159265358979323846;
// log|ψ'/ψ| of a proposal whose matrix is singular (as BackflowSlaterDeterminant).
constexpr double kSingularLogRatio = -700.0;

// Pivoted Gauss–Jordan: a = m⁻¹ for a row-major complex N×N matrix m
// (destroyed), with ln|det m| and arg det m from the pivots and row swaps
// (each swap adds π).  Golub & Van Loan, Matrix Computations, 4th ed.,
// Sec. 3.1 (2013).  Returns false for a singular or nonfinite matrix.
bool invert(cplx* __restrict m, cplx* __restrict a, std::size_t n, double& log_det,
            double& arg_det) {
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = 0; c < n; ++c) a[r * n + c] = r == c ? cplx(1.0, 0.0) : cplx(0.0, 0.0);
    log_det = 0.0;
    arg_det = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot = k;
        for (std::size_t r = k + 1; r < n; ++r)
            if (std::norm(m[r * n + k]) > std::norm(m[pivot * n + k])) pivot = r;
        const cplx diag = m[pivot * n + k];
        const double mag = std::abs(diag);
        if (mag == 0.0 || !std::isfinite(mag)) return false;
        log_det += std::log(mag);
        arg_det += std::arg(diag);
        if (pivot != k) {
            std::swap_ranges(m + k * n + k, m + k * n + n, m + pivot * n + k);
            std::swap_ranges(a + k * n, a + k * n + n, a + pivot * n);
            arg_det += kPi;
        }
        cplx* __restrict m_k = m + k * n;
        cplx* __restrict a_k = a + k * n;
        const cplx inv_diag = 1.0 / diag;
        for (std::size_t c = k + 1; c < n; ++c) m_k[c] *= inv_diag;
        for (std::size_t c = 0; c < n; ++c) a_k[c] *= inv_diag;
        for (std::size_t r = 0; r < n; ++r) {
            if (r == k) continue;
            cplx* __restrict m_r = m + r * n;
            cplx* __restrict a_r = a + r * n;
            const cplx factor = m_r[k];
            if (factor == cplx(0.0, 0.0)) continue;
            for (std::size_t c = k + 1; c < n; ++c) m_r[c] -= factor * m_k[c];
            for (std::size_t c = 0; c < n; ++c) a_r[c] -= factor * a_k[c];
        }
    }
    arg_det = std::remainder(arg_det, 2.0 * kPi);
    return true;
}

}  // namespace

// ── Construction ────────────────────────────────────────────────────────────

ComplexBackflowSlaterDeterminant::ComplexBackflowSlaterDeterminant(
    std::vector<Block> blocks, std::vector<Channel> channels, const SimulationCell& cell)
    : blocks_(std::move(blocks)), cell_(cell) {
    if (blocks_.empty())
        throw std::invalid_argument("complex backflow determinant: no determinant block");
    first_row_ = blocks_.front().first;
    std::size_t next = first_row_;
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const Block& blk = blocks_[b];
        if (blk.first != next || blk.last <= blk.first)
            throw std::invalid_argument(
                "complex backflow determinant: the blocks must cover one group contiguously");
        const std::size_t n = blk.last - blk.first;
        if (!blk.orbitals || blk.orbitals->size() != n)
            throw std::invalid_argument(
                "complex backflow determinant: each block needs exactly as many orbitals "
                "as rows");
        if (!blk.orbitals->supports_hessian())
            throw std::invalid_argument(
                "complex backflow determinant: the orbitals provide no Hessian, which the "
                "backflow Laplacian needs (twisted plane waves do)");
        inv_offset_.push_back(inv_size_);
        inv_size_ += n * n;
        max_block_ = std::max(max_block_, n);
        for (std::size_t i = 0; i < n; ++i) block_of_row_.push_back(b);
        next = blk.last;
    }
    n_rows_ = next - first_row_;

    // The channel rules of BackflowSlaterDeterminant: a same-group channel
    // needs a group of one spin.
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
    params_ = transform_.initial_parameters();
}

ComplexBackflowSlaterDeterminant::ComplexBackflowSlaterDeterminant(
    const ComplexBackflowSlaterDeterminant& o)
    : WavefunctionComponent(o), transform_(o.transform_), cell_(o.cell_),
      first_row_(o.first_row_), n_rows_(o.n_rows_), max_block_(o.max_block_),
      block_of_row_(o.block_of_row_), inv_offset_(o.inv_offset_), inv_size_(o.inv_size_),
      dim_(o.dim_), np_(o.np_), nw_(o.nw_), params_(o.params_), x_(o.x_), inv_(o.inv_),
      f_(o.f_), log_det_(o.log_det_), arg_det_(o.arg_det_), phase_grad_c_(o.phase_grad_c_),
      phase_lap_c_(o.phase_lap_c_), px_(o.px_), plog_det_(o.plog_det_),
      parg_det_(o.parg_det_), pinv_(o.pinv_), pf_(o.pf_), pgrad_(o.pgrad_), plap_(o.plap_),
      pvalid_(o.pvalid_), proposal_laplacian_(o.proposal_laplacian_),
      // A copy runs on its own threads: fresh, unsized scratch slots.
      scratch_(o.scratch_.size()) {
    blocks_.reserve(o.blocks_.size());
    for (const Block& b : o.blocks_) blocks_.push_back({b.first, b.last, b.orbitals->clone()});
}

std::unique_ptr<WavefunctionComponent> ComplexBackflowSlaterDeterminant::clone() const {
    return std::make_unique<ComplexBackflowSlaterDeterminant>(*this);
}

void ComplexBackflowSlaterDeterminant::rebuild() { transform_.set_parameters(params_); }

// ── Sizes ───────────────────────────────────────────────────────────────────

void ComplexBackflowSlaterDeterminant::resize_state(std::size_t nw) {
    nw_ = nw;
    const std::size_t nb = blocks_.size(), xs = n_rows_ * dim_;
    x_.assign(nw * xs, 0.0);
    inv_.assign(nw * inv_size_, cplx(0.0, 0.0));
    f_.assign(nw * xs, cplx(0.0, 0.0));
    log_det_.assign(nw * nb, 0.0);
    arg_det_.assign(nw * nb, 0.0);
    px_.assign(nw * xs, 0.0);
    pinv_.assign(nw * inv_size_, cplx(0.0, 0.0));
    pf_.assign(nw * xs, cplx(0.0, 0.0));
    plog_det_.assign(nw * nb, 0.0);
    parg_det_.assign(nw * nb, 0.0);
    pgrad_.assign(nw * dim_, cplx(0.0, 0.0));
    plap_.assign(nw * dim_, cplx(0.0, 0.0));
    pvalid_.assign(nw, 0);
}

void ComplexBackflowSlaterDeterminant::prepare(std::size_t nw, std::size_t np,
                                               std::size_t dim) {
    if (first_row_ + n_rows_ > np)
        throw std::invalid_argument(
            "complex backflow determinant: rows beyond the particle count");
    transform_.prepare(np, dim);
    resize_caches(nw, np, dim);
    phase_grad_c_.assign(np * dim * nw, 0.0);
    phase_lap_c_.assign(np * dim * nw, 0.0);
    if (np != np_ || dim != dim_) ++scratch_generation_;   // slots resize on next use
    dim_ = dim;
    np_ = np;
    resize_state(nw);
    const auto threads = static_cast<std::size_t>(
        std::max({1, omp_get_max_threads(), omp_get_num_procs()}));
    if (scratch_.size() < threads) scratch_.resize(threads);
}

void ComplexBackflowSlaterDeterminant::size_scratch(Scratch& s) const {
    const std::size_t np = np_, dim = dim_, n = max_block_, rows = n_rows_;
    s.pos.assign(np * dim, 0.0);
    s.x.assign(rows * dim, 0.0);
    s.val.assign(rows * n, cplx(0.0, 0.0));
    s.grad.assign(rows * n * dim, cplx(0.0, 0.0));
    s.hess.assign(rows * n * dim * dim, cplx(0.0, 0.0));
    transform_.size_tables(s.tables);
    s.work.assign(n * n, cplx(0.0, 0.0));
    s.at.assign(n * n, cplx(0.0, 0.0));
    s.gmat.assign(dim * inv_size_, cplx(0.0, 0.0));
    s.grad_a.assign(n * n, cplx(0.0, 0.0));
    s.t.assign(rows * dim * dim, cplx(0.0, 0.0));
    s.v.assign(rows * dim, 0.0);
    s.u.assign(n * n, cplx(0.0, 0.0));
    s.w.assign(n * n, cplx(0.0, 0.0));
    s.f_re.assign(rows * dim, 0.0);
    s.f_im.assign(rows * dim, 0.0);
    s.out_grad.assign(np * dim, cplx(0.0, 0.0));
    s.out_lap.assign(np * dim, cplx(0.0, 0.0));
    s.param.assign(std::max<std::size_t>(transform_.max_channel_parameters(), 1), 0.0);
    s.pderiv_re.assign(transform_.n_parameters(), 0.0);
    s.pderiv_im.assign(transform_.n_parameters(), 0.0);
    s.generation = scratch_generation_;
}

std::size_t ComplexBackflowSlaterDeterminant::walker_state_bytes() const noexcept {
    if (nw_ == 0) return WavefunctionComponent::walker_state_bytes();
    // x (real), F and the inverse (complex), ln|det| and arg det per block,
    // and the two phase planes.
    const std::size_t doubles = n_rows_ * dim_ + 2 * (n_rows_ * dim_ + inv_size_) +
                                2 * blocks_.size() + 2 * np_ * dim_;
    return WavefunctionComponent::walker_state_bytes() + doubles * sizeof(double);
}

ComplexBackflowSlaterDeterminant::Scratch& ComplexBackflowSlaterDeterminant::scratch() const {
    const auto t = static_cast<std::size_t>(omp_get_thread_num());
    if (t >= scratch_.size())
        throw std::logic_error("complex backflow determinant: OpenMP thread " +
                               std::to_string(t) + " beyond the " +
                               std::to_string(scratch_.size()) +
                               " scratch slots (one per processor); use fewer threads");
    Scratch& s = scratch_[t];
    if (s.generation != scratch_generation_) size_scratch(s);
    return s;
}

void ComplexBackflowSlaterDeterminant::permute_caches(std::span<const std::size_t> src) {
    const std::size_t old_nw = nw_;
    WavefunctionComponent::permute_caches(src);
    if (old_nw == 0) return;
    const std::size_t new_nw = src.size();
    // Walker-major state: one walker's block moves as a whole.
    const auto gather_major = [&](auto& v, std::size_t width) {
        std::decay_t<decltype(v)> out(new_nw * width);
        for (std::size_t c = 0; c < new_nw; ++c)
            std::copy_n(v.begin() + static_cast<std::ptrdiff_t>(src[c] * width), width,
                        out.begin() + static_cast<std::ptrdiff_t>(c * width));
        v.swap(out);
    };
    // The phase planes are SoA, walker fastest.
    const auto gather_soa = [&](std::vector<double>& v) {
        const std::size_t width = v.size() / old_nw;
        std::vector<double> out(width * new_nw);
        for (std::size_t k = 0; k < width; ++k)
            for (std::size_t c = 0; c < new_nw; ++c) out[k * new_nw + c] = v[k * old_nw + src[c]];
        v.swap(out);
    };
    const std::size_t nb = blocks_.size(), xs = n_rows_ * dim_;
    gather_major(x_, xs);
    gather_major(inv_, inv_size_);
    gather_major(f_, xs);
    gather_major(log_det_, nb);
    gather_major(arg_det_, nb);
    gather_soa(phase_grad_c_);
    gather_soa(phase_lap_c_);
    nw_ = new_nw;
    // Pending state is transient: the next ratio_batch rebuilds it.
    px_.assign(new_nw * xs, 0.0);
    pinv_.assign(new_nw * inv_size_, cplx(0.0, 0.0));
    pf_.assign(new_nw * xs, cplx(0.0, 0.0));
    plog_det_.assign(new_nw * nb, 0.0);
    parg_det_.assign(new_nw * nb, 0.0);
    pgrad_.assign(new_nw * dim_, cplx(0.0, 0.0));
    plap_.assign(new_nw * dim_, cplx(0.0, 0.0));
    pvalid_.assign(new_nw, 0);
}

// ── One configuration ───────────────────────────────────────────────────────

void ComplexBackflowSlaterDeterminant::load_positions(const WalkerSet& batch, std::size_t w,
                                                      Scratch& s) const {
    const std::size_t nw = batch.num_walkers();
    const std::span<const double> pos = batch.positions();
    for (std::size_t k = 0; k < np_; ++k)
        for (std::size_t d = 0; d < dim_; ++d) s.pos[k * dim_ + d] = pos[(k * dim_ + d) * nw + w];
}

void ComplexBackflowSlaterDeterminant::evaluate_rows(Scratch& s) const {
    const std::size_t dim = dim_, n = max_block_;
    for (std::size_t i = 0; i < n_rows_; ++i) {
        const ComplexOrbitalSet& orb = *blocks_[block_of_row_[i]].orbitals;
        const std::size_t nb = orb.size();
        orb.evaluate_hessian(std::span<const double>(s.x.data() + i * dim, dim), dim,
                             std::span<cplx>(s.val.data() + i * n, nb),
                             std::span<cplx>(s.grad.data() + i * n * dim, nb * dim),
                             std::span<cplx>(s.hess.data() + i * n * dim * dim, nb * dim * dim));
    }
}

bool ComplexBackflowSlaterDeterminant::invert_blocks(Scratch& s, cplx* inv, double* log_det,
                                                     double* arg_det) const {
    const std::size_t n_max = max_block_;
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const std::size_t n = blocks_[b].last - blocks_[b].first;
        const std::size_t row0 = blocks_[b].first - first_row_;
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t m = 0; m < n; ++m)
                s.work[r * n + m] = s.val[(row0 + r) * n_max + m];   // M[r][m] = φ_m(x_r)
        if (!invert(s.work.data(), inv + inv_offset_[b], n, log_det[b], arg_det[b]))
            return false;
    }
    return true;
}

void ComplexBackflowSlaterDeterminant::split_f(Scratch& s, const cplx* f) const {
    for (std::size_t e = 0; e < n_rows_ * dim_; ++e) {
        s.f_re[e] = f[e].real();
        s.f_im[e] = f[e].imag();
    }
}

void ComplexBackflowSlaterDeterminant::compute_f(Scratch& s, const cplx* inv, cplx* f) const {
    const std::size_t dim = dim_, n_max = max_block_;
    for (std::size_t i = 0; i < n_rows_; ++i) {
        const std::size_t b = block_of_row_[i];
        const std::size_t n = blocks_[b].last - blocks_[b].first;
        const std::size_t r = first_row_ + i - blocks_[b].first;
        const cplx* A = inv + inv_offset_[b];
        for (std::size_t a = 0; a < dim; ++a) {
            cplx acc(0.0, 0.0);
            for (std::size_t m = 0; m < n; ++m) acc += A[m * n + r] * s.grad[(i * n_max + m) * dim + a];
            f[i * dim + a] = acc;
        }
    }
    split_f(s, f);
}

// J is real, so Σ_t F_t · ∂x_t/∂r_k splits into the real and imaginary parts of F.
void ComplexBackflowSlaterDeterminant::particle_gradient(Scratch& s, std::size_t k,
                                                         cplx* out) const {
    double gr[SimulationCell::kMaxDim] = {}, gi[SimulationCell::kMaxDim] = {};
    transform_.gradient(s.tables, s.f_re.data(), k, gr);
    transform_.gradient(s.tables, s.f_im.data(), k, gi);
    for (std::size_t d = 0; d < dim_; ++d) out[d] = cplx(gr[d], gi[d]);
}

void ComplexBackflowSlaterDeterminant::hessian_traces(Scratch& s, const cplx* inv) const {
    const std::size_t dd = dim_ * dim_, n_max = max_block_;
    for (std::size_t i = 0; i < n_rows_; ++i) {
        const std::size_t b = block_of_row_[i];
        const std::size_t n = blocks_[b].last - blocks_[b].first;
        const std::size_t r = first_row_ + i - blocks_[b].first;
        const cplx* A = inv + inv_offset_[b];
        for (std::size_t ag = 0; ag < dd; ++ag) {
            cplx acc(0.0, 0.0);
            for (std::size_t m = 0; m < n; ++m) acc += A[m * n + r] * s.hess[(i * n_max + m) * dd + ag];
            s.t[i * dd + ag] = acc;
        }
    }
}

void ComplexBackflowSlaterDeterminant::particle_laplacian(Scratch& s, const cplx* inv,
                                                          std::size_t k, cplx* out) const {
    const std::size_t dim = dim_, dd = dim * dim, n_max = max_block_;
    hessian_traces(s, inv);
    for (std::size_t beta = 0; beta < dim; ++beta) {
        // Σ F·S by parts; both calls write the same (real) column v.
        const double fs_re = transform_.jacobian_column(s.tables, s.f_re.data(), k, beta, s.v.data());
        const double fs_im = transform_.jacobian_column(s.tables, s.f_im.data(), k, beta, s.v.data());
        cplx lap(fs_re, fs_im);
        for (std::size_t i = 0; i < n_rows_; ++i) {
            const double* v = s.v.data() + i * dim;
            const cplx* T = s.t.data() + i * dd;
            for (std::size_t a = 0; a < dim; ++a)
                for (std::size_t g = 0; g < dim; ++g) lap += v[a] * v[g] * T[a * dim + g];
        }
        // −Σ_{i,j} U_ji U_ij per block, U = Aᵀ W, W_im = Σ_α v_iα ∂_α φ_m(x_i).
        for (std::size_t b = 0; b < blocks_.size(); ++b) {
            const std::size_t n = blocks_[b].last - blocks_[b].first;
            const std::size_t row0 = blocks_[b].first - first_row_;
            const cplx* A = inv + inv_offset_[b];
            bool any = false;
            for (std::size_t r = 0; r < n; ++r) {
                const double* v = s.v.data() + (row0 + r) * dim;
                for (std::size_t m = 0; m < n; ++m) {
                    cplx acc(0.0, 0.0);
                    for (std::size_t a = 0; a < dim; ++a)
                        acc += v[a] * s.grad[((row0 + r) * n_max + m) * dim + a];
                    s.w[r * n + m] = acc;
                    any = any || acc != cplx(0.0, 0.0);
                }
            }
            if (!any) continue;
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t m = 0; m < n; ++m) s.at[j * n + m] = A[m * n + j];
            for (std::size_t j = 0; j < n; ++j) {
                const cplx* at_j = s.at.data() + j * n;
                for (std::size_t r = 0; r < n; ++r) {
                    const cplx* w_r = s.w.data() + r * n;
                    cplx acc(0.0, 0.0);
                    for (std::size_t m = 0; m < n; ++m) acc += at_j[m] * w_r[m];
                    s.u[j * n + r] = acc;
                }
            }
            cplx uu(0.0, 0.0);
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t r = 0; r < n; ++r) uu += s.u[j * n + r] * s.u[r * n + j];
            lap -= uu;
        }
        out[beta] = lap;
    }
}

void ComplexBackflowSlaterDeterminant::gradient_matrices(Scratch& s, const cplx* inv) const {
    const std::size_t dim = dim_, n_max = max_block_;
    hessian_traces(s, inv);
    // G^α[j][r] = Σ_m A_mj ∂_α φ_m(x_r), at gmat[α*inv_size + off + j*n + r].
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const std::size_t n = blocks_[b].last - blocks_[b].first;
        const std::size_t row0 = blocks_[b].first - first_row_;
        const cplx* A = inv + inv_offset_[b];
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t m = 0; m < n; ++m) s.at[j * n + m] = A[m * n + j];
        for (std::size_t a = 0; a < dim; ++a) {
            for (std::size_t r = 0; r < n; ++r)
                for (std::size_t m = 0; m < n; ++m)
                    s.grad_a[r * n + m] = s.grad[((row0 + r) * n_max + m) * dim + a];
            cplx* G = s.gmat.data() + a * inv_size_ + inv_offset_[b];
            for (std::size_t j = 0; j < n; ++j) {
                const cplx* at_j = s.at.data() + j * n;
                for (std::size_t r = 0; r < n; ++r) {
                    const cplx* g_r = s.grad_a.data() + r * n;
                    cplx acc(0.0, 0.0);
                    for (std::size_t m = 0; m < n; ++m) acc += at_j[m] * g_r[m];
                    G[j * n + r] = acc;
                }
            }
        }
    }
}

void ComplexBackflowSlaterDeterminant::all_derivatives(Scratch& s, const cplx* inv) const {
    const std::size_t dim = dim_, dd = dim * dim;
    gradient_matrices(s, inv);
    std::fill(s.out_grad.begin(), s.out_grad.end(), cplx(0.0, 0.0));
    std::fill(s.out_lap.begin(), s.out_lap.end(), cplx(0.0, 0.0));
    for (std::size_t k = 0; k < np_; ++k) {
        if (!affects(k)) continue;
        particle_gradient(s, k, s.out_grad.data() + k * dim);
        for (std::size_t beta = 0; beta < dim; ++beta) {
            const double fs_re =
                transform_.jacobian_column(s.tables, s.f_re.data(), k, beta, s.v.data());
            const double fs_im =
                transform_.jacobian_column(s.tables, s.f_im.data(), k, beta, s.v.data());
            cplx l(fs_re, fs_im);
            for (std::size_t i = 0; i < n_rows_; ++i) {
                const double* v = s.v.data() + i * dim;
                const cplx* T = s.t.data() + i * dd;
                for (std::size_t a = 0; a < dim; ++a)
                    for (std::size_t g = 0; g < dim; ++g) l += v[a] * v[g] * T[a * dim + g];
            }
            // −Σ_{i,j} U_ji U_ij with U_ji = Σ_α v_iα G^α_ji (block by block).
            for (std::size_t b = 0; b < blocks_.size(); ++b) {
                const std::size_t n = blocks_[b].last - blocks_[b].first;
                const std::size_t row0 = blocks_[b].first - first_row_;
                for (std::size_t j = 0; j < n; ++j)
                    for (std::size_t r = 0; r < n; ++r) {
                        cplx acc(0.0, 0.0);
                        for (std::size_t a = 0; a < dim; ++a)
                            acc += s.v[(row0 + r) * dim + a] *
                                   s.gmat[a * inv_size_ + inv_offset_[b] + j * n + r];
                        s.u[j * n + r] = acc;
                    }
                cplx uu(0.0, 0.0);
                for (std::size_t j = 0; j < n; ++j)
                    for (std::size_t r = 0; r < n; ++r) uu += s.u[j * n + r] * s.u[r * n + j];
                l -= uu;
            }
            s.out_lap[k * dim + beta] = l;
        }
    }
}

// ── Full evaluation ─────────────────────────────────────────────────────────

void ComplexBackflowSlaterDeterminant::evaluate_batch(WalkerSet& batch) {
    const std::size_t nw = batch.num_walkers();
    const std::size_t np = batch.n_particles();
    const std::size_t dim = batch.dim();
    prepare(nw, np, dim);
    const std::size_t nb = blocks_.size(), xs = n_rows_ * dim;
    long singular = -1;

#pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t ws = 0; ws < static_cast<std::ptrdiff_t>(nw); ++ws) {
        const auto w = static_cast<std::size_t>(ws);
        Scratch& s = scratch();
        load_positions(batch, w, s);
        transform_.full(s.pos.data(), s.x.data(), s.tables);
        evaluate_rows(s);
        cplx* inv = inv_.data() + w * inv_size_;
        if (!invert_blocks(s, inv, log_det_.data() + w * nb, arg_det_.data() + w * nb)) {
#pragma omp critical(complex_backflow_singular)
            singular = static_cast<long>(w);
            continue;
        }
        std::copy(s.x.begin(), s.x.end(), x_.begin() + static_cast<std::ptrdiff_t>(w * xs));
        compute_f(s, inv, f_.data() + w * xs);
        all_derivatives(s, inv);
        double log_psi = 0.0;
        for (std::size_t b = 0; b < nb; ++b) log_psi += log_det_[w * nb + b];
        log_psi_c_[w] = log_psi;
        for (std::size_t k = 0; k < np; ++k)
            for (std::size_t d = 0; d < dim; ++d) {
                const std::size_t j = (k * dim + d) * nw + w;
                const cplx g = s.out_grad[k * dim + d], l = s.out_lap[k * dim + d];
                grad_c_[j] = g.real();
                lap_c_[j] = l.real();
                phase_grad_c_[j] = g.imag();
                phase_lap_c_[j] = l.imag();
            }
    }
    if (singular >= 0)
        throw std::runtime_error("complex backflow determinant: singular at the configuration "
                                 "of walker " + std::to_string(singular) +
                                 " (a zero of the trial wavefunction)");

    std::span<double> b_logpsi = batch.log_psi();
    for (std::size_t w = 0; w < nw; ++w) b_logpsi[w] += log_psi_c_[w];
    const auto add = [](std::span<double> dst, const std::vector<double>& src) {
        for (std::size_t k = 0; k < dst.size(); ++k) dst[k] += src[k];
    };
    add(batch.gradients(), grad_c_);
    add(batch.laplacians(), lap_c_);
    add(batch.phase_gradients(), phase_grad_c_);
    add(batch.phase_laplacians(), phase_lap_c_);
}

// ── Moves ───────────────────────────────────────────────────────────────────

void ComplexBackflowSlaterDeterminant::ratio_batch(WalkerSet& batch, std::size_t p,
                                                   std::size_t w_begin, std::size_t w_end) {
    const std::size_t nw = batch.num_walkers();
    const std::size_t dim = dim_, nb = blocks_.size(), xs = n_rows_ * dim_;
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end || !affects(p)) return;

    std::span<double> log_ratios = batch.log_ratios();
    std::span<double> phase = batch.phase_diffs();
    std::span<double> new_grad = batch.new_grad();
    std::span<double> new_lap = batch.new_lap();
    std::span<double> new_pgrad = batch.new_phase_grad();
    std::span<double> new_plap = batch.new_phase_lap();
    const std::span<const double> new_pos = batch.new_pos();
    Scratch& s = scratch();
    for (std::size_t w = w_begin; w < w_end; ++w) {
        load_positions(batch, w, s);
        double old_r[SimulationCell::kMaxDim] = {};
        for (std::size_t d = 0; d < dim; ++d) {
            old_r[d] = s.pos[p * dim + d];
            s.pos[p * dim + d] = new_pos[d * nw + w];
        }
        // x' from the committed x: O(N) pairs, not the O(N²) of a rebuild.
        std::copy_n(x_.begin() + static_cast<std::ptrdiff_t>(w * xs), xs, s.x.begin());
        transform_.shift(s.pos.data(), p, old_r, s.x.data());
        transform_.particle_tables(s.pos.data(), p, s.x.data(), s.tables);
        evaluate_rows(s);
        cplx* pinv = pinv_.data() + w * inv_size_;
        double* plog = plog_det_.data() + w * nb;
        double* parg = parg_det_.data() + w * nb;
        if (!invert_blocks(s, pinv, plog, parg)) {
            pvalid_[w] = 0;
            log_ratios[w] += kSingularLogRatio;
            for (std::size_t d = 0; d < dim; ++d)
                pgrad_[w * dim + d] = plap_[w * dim + d] = cplx(0.0, 0.0);
            continue;
        }
        pvalid_[w] = 1;
        double dlog = 0.0, darg = 0.0;
        for (std::size_t b = 0; b < nb; ++b) {
            dlog += plog[b] - log_det_[w * nb + b];
            darg += parg[b] - arg_det_[w * nb + b];
        }
        log_ratios[w] += dlog;
        // arg(D'/D) ∈ (−π, π]: a continuous phase, no node gate (fixed phase).
        phase[w] += std::remainder(darg, 2.0 * kPi);

        std::copy(s.x.begin(), s.x.end(), px_.begin() + static_cast<std::ptrdiff_t>(w * xs));
        cplx* pf = pf_.data() + w * xs;
        compute_f(s, pinv, pf);
        cplx g[SimulationCell::kMaxDim] = {}, l[SimulationCell::kMaxDim] = {};
        particle_gradient(s, p, g);
        // Skipped, l = 0, when the driver refreshes before reading it.
        if (proposal_laplacian_) particle_laplacian(s, pinv, p, l);
        for (std::size_t d = 0; d < dim; ++d) {
            pgrad_[w * dim + d] = g[d];
            plap_[w * dim + d] = l[d];
            new_grad[d * nw + w] += g[d].real();
            new_lap[d * nw + w] += l[d].real();
            new_pgrad[d * nw + w] += g[d].imag();
            new_plap[d * nw + w] += l[d].imag();
        }
    }
}

void ComplexBackflowSlaterDeterminant::commit(WalkerSet& batch, std::size_t p, std::size_t w) {
    if (!pvalid_[w])
        throw std::logic_error(
            "complex backflow determinant: a move to a singular matrix was accepted");
    const std::size_t nw = batch.num_walkers(), dim = dim_, nb = blocks_.size();
    const auto copy = [w](const auto& from, auto& to, std::size_t width) {
        std::copy_n(from.begin() + static_cast<std::ptrdiff_t>(w * width), width,
                    to.begin() + static_cast<std::ptrdiff_t>(w * width));
    };
    copy(px_, x_, n_rows_ * dim);
    copy(pinv_, inv_, inv_size_);
    copy(pf_, f_, n_rows_ * dim);
    copy(plog_det_, log_det_, nb);
    copy(parg_det_, arg_det_, nb);
    double log_psi = 0.0;
    for (std::size_t b = 0; b < nb; ++b) log_psi += log_det_[w * nb + b];
    batch.log_psi()[w] += log_psi - log_psi_c_[w];
    log_psi_c_[w] = log_psi;
    // The moved particle's derivatives are already in new_grad/new_lap and
    // their phase counterparts, which WalkerSet::apply_move writes into its totals.
    for (std::size_t d = 0; d < dim; ++d) {
        const std::size_t j = (p * dim + d) * nw + w;
        grad_c_[j] = pgrad_[w * dim + d].real();
        lap_c_[j] = plap_[w * dim + d].real();
        phase_grad_c_[j] = pgrad_[w * dim + d].imag();
        phase_lap_c_[j] = plap_[w * dim + d].imag();
    }
}

void ComplexBackflowSlaterDeterminant::apply_move_batch_deferred(WalkerSet& batch,
                                                                 std::size_t p,
                                                                 std::span<const bool> accept,
                                                                 std::size_t w_begin,
                                                                 std::size_t w_end) {
    const std::size_t nw = batch.num_walkers();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end || !affects(p)) return;
    for (std::size_t w = w_begin; w < w_end; ++w)
        if (accept[w]) commit(batch, p, w);
}

void ComplexBackflowSlaterDeterminant::apply_move_batch(WalkerSet& batch, std::size_t p,
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
//
// Unlike BackflowSlaterDeterminant, the orbitals are evaluated at the
// COMMITTED x, not at x rebuilt from the positions: after a wrap of r_j the
// rebuilt x_j differs from the committed one by a lattice vector, so its row
// carries the twist phase e^{iθ·a} that the committed inverse does not, and
// G^α_ji, T_j would come out multiplied by it.  The pair tables (B, s) depend
// on minimum-image separations only, so they are rebuilt.  The committed x_j
// is rebuilt from scratch at every move of j (particle_tables), so its
// incremental rounding never accumulates beyond one sweep.
void ComplexBackflowSlaterDeterminant::refresh_walker(WalkerSet& batch, std::size_t w,
                                                      std::size_t moved) {
    const std::size_t nw = batch.num_walkers(), dim = dim_, xs = n_rows_ * dim;
    Scratch& s = scratch();
    load_positions(batch, w, s);
    if (moved != npos) {
        const std::span<const double> new_pos = batch.new_pos();
        for (std::size_t d = 0; d < dim; ++d) s.pos[moved * dim + d] = new_pos[d * nw + w];
    }
    transform_.full(s.pos.data(), s.x.data(), s.tables);
    std::copy_n(x_.begin() + static_cast<std::ptrdiff_t>(w * xs), xs, s.x.begin());
    evaluate_rows(s);
    split_f(s, f_.data() + w * xs);
    all_derivatives(s, inv_.data() + w * inv_size_);
    std::span<double> bg = batch.gradients();
    std::span<double> bl = batch.laplacians();
    std::span<double> bpg = batch.phase_gradients();
    std::span<double> bpl = batch.phase_laplacians();
    // Already-current fields stay untouched (subtracting and re-adding could
    // cancel a smaller component's bits).
    const auto update = [](double& total, double& mine, double value) {
        if (value != mine) total = (total - mine) + value;
        mine = value;
    };
    for (std::size_t k = 0; k < np_; ++k) {
        if (k == moved) continue;
        for (std::size_t d = 0; d < dim; ++d) {
            const std::size_t j = (k * dim + d) * nw + w;
            const cplx g = s.out_grad[k * dim + d], l = s.out_lap[k * dim + d];
            update(bg[j], grad_c_[j], g.real());
            update(bl[j], lap_c_[j], l.real());
            update(bpg[j], phase_grad_c_[j], g.imag());
            update(bpl[j], phase_lap_c_[j], l.imag());
        }
    }
}

void ComplexBackflowSlaterDeterminant::refresh_particle_derivatives(WalkerSet& batch,
                                                                    std::size_t p,
                                                                    std::size_t w_begin,
                                                                    std::size_t w_end) {
    const std::size_t nw = batch.num_walkers(), dim = dim_, xs = n_rows_ * dim;
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end || !affects(p)) return;
    std::span<double> bg = batch.gradients();
    std::span<double> bpg = batch.phase_gradients();
    Scratch& s = scratch();
    // The gradient only (modulus and phase): F is committed and gauge-free,
    // and p's table rows depend on separations only.
    for (std::size_t w = w_begin; w < w_end; ++w) {
        load_positions(batch, w, s);
        transform_.particle_tables(s.pos.data(), p, s.x.data(), s.tables);
        split_f(s, f_.data() + w * xs);
        cplx g[SimulationCell::kMaxDim] = {};
        particle_gradient(s, p, g);
        for (std::size_t d = 0; d < dim; ++d) {
            const std::size_t j = (p * dim + d) * nw + w;
            if (g[d].real() != grad_c_[j]) bg[j] = (bg[j] - grad_c_[j]) + g[d].real();
            if (g[d].imag() != phase_grad_c_[j])
                bpg[j] = (bpg[j] - phase_grad_c_[j]) + g[d].imag();
            grad_c_[j] = g[d].real();
            phase_grad_c_[j] = g[d].imag();
        }
    }
}

void ComplexBackflowSlaterDeterminant::refresh_derivatives(WalkerSet& batch,
                                                           std::size_t w_begin,
                                                           std::size_t w_end) {
    const std::size_t nw = batch.num_walkers();
    if (w_end > nw) w_end = nw;
    for (std::size_t w = w_begin; w < w_end; ++w) refresh_walker(batch, w, npos);
}

// ── Parameters ──────────────────────────────────────────────────────────────

void ComplexBackflowSlaterDeterminant::compute_parameter_derivatives(
    const WalkerSet& batch, std::span<double> derivs) const {
    compute_complex_parameter_derivatives(batch, derivs, {});
}

// O = Σ_{iα} F_iα ∂x_iα/∂p with ∂x/∂p real: BackflowTransformation contracts
// the real and the imaginary part of F in turn.
void ComplexBackflowSlaterDeterminant::compute_complex_parameter_derivatives(
    const WalkerSet& batch, std::span<double> derivs, std::span<double> phase_derivs) const {
    const std::size_t nw = batch.num_walkers(), xs = n_rows_ * dim_;
    std::vector<std::size_t> column(params_.size(), npos);   // [k_opt * nw + w]
    std::size_t n_opt = 0;
    for (std::size_t q = 0; q < params_.size(); ++q)
        if (!params_[q].is_fixed) column[q] = n_opt++;
    if (n_opt == 0) return;
    const bool phase = !phase_derivs.empty();
    Scratch& s = scratch();
    for (std::size_t w = 0; w < nw; ++w) {
        load_positions(batch, w, s);
        split_f(s, f_.data() + w * xs);
        transform_.parameter_derivatives(s.pos.data(), s.f_re.data(), s.pderiv_re, s.param);
        if (phase)
            transform_.parameter_derivatives(s.pos.data(), s.f_im.data(), s.pderiv_im, s.param);
        for (std::size_t q = 0; q < params_.size(); ++q) {
            if (column[q] == npos) continue;
            derivs[column[q] * nw + w] = s.pderiv_re[q];
            if (phase) phase_derivs[column[q] * nw + w] = s.pderiv_im[q];
        }
    }
}

// ── Parameter derivatives of the gradient and Laplacian ─────────────────────

bool ComplexBackflowSlaterDeterminant::supports_parameter_gradient_derivatives() const {
    for (const Block& b : blocks_)
        if (!b.orbitals->supports_third_derivatives()) return false;
    for (std::size_t c = 0; c < transform_.n_channels(); ++c)
        if (transform_.eta(c).truncation() < 3) return false;
    return true;
}

void ComplexBackflowSlaterDeterminant::size_tangent(Tangent& t) const {
    const std::size_t dim = dim_, dd = dim * dim, n = max_block_, rows = n_rows_;
    t.third.assign(rows * n * dd * dim, cplx(0.0, 0.0));
    t.x_dot.assign(rows * dim, 0.0);
    transform_.size_tables(t.tables);
    t.eta.assign(3 * std::max<std::size_t>(transform_.max_channel_parameters(), 1), 0.0);
    t.m_dot.assign(rows * n, cplx(0.0, 0.0));
    t.g_dot.assign(rows * n * dim, cplx(0.0, 0.0));
    t.h_dot.assign(rows * n * dd, cplx(0.0, 0.0));
    t.a_dot.assign(inv_size_, cplx(0.0, 0.0));
    t.prod.assign(n * n, cplx(0.0, 0.0));
    t.f_dot.assign(rows * dim, cplx(0.0, 0.0));
    t.f_dot_re.assign(rows * dim, 0.0);
    t.f_dot_im.assign(rows * dim, 0.0);
    t.t_dot.assign(rows * dd, cplx(0.0, 0.0));
    t.gmat_dot.assign(dim * inv_size_, cplx(0.0, 0.0));
    t.v_dot.assign(rows * dim, 0.0);
    t.v_tmp.assign(rows * dim, 0.0);
    t.u.assign(n * n, cplx(0.0, 0.0));
    t.u_dot.assign(n * n, cplx(0.0, 0.0));
    t.out_grad.assign(np_ * dim, cplx(0.0, 0.0));
    t.out_lap.assign(np_ * dim, cplx(0.0, 0.0));
}

// BackflowSlaterDeterminant::tangent_derivatives for a backflow parameter,
// complex where the orbitals are.  T stays symmetric (a Hessian contracted
// with A) but not Hermitian, so ∂(vᵀTv) = 2 v̇ᵀTv + vᵀṪv and
// ∂(Σ U_ji U_ij) = 2 Σ U̇_ji U_ij hold without conjugation.
void ComplexBackflowSlaterDeterminant::tangent_derivatives(Scratch& s, Tangent& t,
                                                           const cplx* inv) const {
    const std::size_t dim = dim_, dd = dim * dim, n_max = max_block_;
    // Ȧ = −A Ṁ A per block, with Ṁ[r][m] = m_dot.
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const std::size_t n = blocks_[b].last - blocks_[b].first;
        const std::size_t row0 = blocks_[b].first - first_row_;
        const cplx* A = inv + inv_offset_[b];
        cplx* Ad = t.a_dot.data() + inv_offset_[b];
        for (std::size_t m = 0; m < n; ++m)
            for (std::size_t mp = 0; mp < n; ++mp) {
                cplx acc(0.0, 0.0);
                for (std::size_t r = 0; r < n; ++r)
                    acc += A[m * n + r] * t.m_dot[(row0 + r) * n_max + mp];
                t.prod[m * n + mp] = acc;
            }
        for (std::size_t m = 0; m < n; ++m)
            for (std::size_t r = 0; r < n; ++r) {
                cplx acc(0.0, 0.0);
                for (std::size_t mp = 0; mp < n; ++mp) acc += t.prod[m * n + mp] * A[mp * n + r];
                Ad[m * n + r] = -acc;
            }
    }
    // Ḟ_iα = Σ_m [Ȧ_mi ∂_αφ_m + A_mi (∂_αφ_m)˙], Ṫ_i likewise with the Hessian.
    for (std::size_t i = 0; i < n_rows_; ++i) {
        const std::size_t b = block_of_row_[i];
        const std::size_t n = blocks_[b].last - blocks_[b].first;
        const std::size_t r = first_row_ + i - blocks_[b].first;
        const cplx* A = inv + inv_offset_[b];
        const cplx* Ad = t.a_dot.data() + inv_offset_[b];
        for (std::size_t a = 0; a < dim; ++a) {
            cplx acc(0.0, 0.0);
            for (std::size_t m = 0; m < n; ++m) {
                const std::size_t e = (i * n_max + m) * dim + a;
                acc += Ad[m * n + r] * s.grad[e] + A[m * n + r] * t.g_dot[e];
            }
            t.f_dot[i * dim + a] = acc;
            t.f_dot_re[i * dim + a] = acc.real();
            t.f_dot_im[i * dim + a] = acc.imag();
        }
        for (std::size_t ag = 0; ag < dd; ++ag) {
            cplx acc(0.0, 0.0);
            for (std::size_t m = 0; m < n; ++m) {
                const std::size_t e = (i * n_max + m) * dd + ag;
                acc += Ad[m * n + r] * s.hess[e] + A[m * n + r] * t.h_dot[e];
            }
            t.t_dot[i * dd + ag] = acc;
        }
    }
    // Ġ^α[j][r] = Σ_m [Ȧ_mj ∂_αφ_m(x_r) + A_mj (∂_αφ_m)˙(x_r)].
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const std::size_t n = blocks_[b].last - blocks_[b].first;
        const std::size_t row0 = blocks_[b].first - first_row_;
        const cplx* A = inv + inv_offset_[b];
        const cplx* Ad = t.a_dot.data() + inv_offset_[b];
        for (std::size_t a = 0; a < dim; ++a) {
            cplx* Gd = t.gmat_dot.data() + a * inv_size_ + inv_offset_[b];
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t r = 0; r < n; ++r) {
                    cplx acc(0.0, 0.0);
                    for (std::size_t m = 0; m < n; ++m) {
                        const std::size_t e = ((row0 + r) * n_max + m) * dim + a;
                        acc += Ad[m * n + j] * s.grad[e] + A[m * n + j] * t.g_dot[e];
                    }
                    Gd[j * n + r] = acc;
                }
        }
    }

    std::fill(t.out_grad.begin(), t.out_grad.end(), cplx(0.0, 0.0));
    std::fill(t.out_lap.begin(), t.out_lap.end(), cplx(0.0, 0.0));
    for (std::size_t k = 0; k < np_; ++k) {
        if (!affects(k)) continue;
        // ∂_p(∂_k ln D) = Σ_t (Ḟ_t·J_tk + F_t·J̇_tk); J̇ has no identity part.
        double g1r[SimulationCell::kMaxDim] = {}, g1i[SimulationCell::kMaxDim] = {};
        double g2r[SimulationCell::kMaxDim] = {}, g2i[SimulationCell::kMaxDim] = {};
        transform_.gradient(s.tables, t.f_dot_re.data(), k, g1r);
        transform_.gradient(s.tables, t.f_dot_im.data(), k, g1i);
        transform_.gradient(t.tables, s.f_re.data(), k, g2r, /*with_identity=*/false);
        transform_.gradient(t.tables, s.f_im.data(), k, g2i, /*with_identity=*/false);
        for (std::size_t beta = 0; beta < dim; ++beta)
            t.out_grad[k * dim + beta] = cplx(g1r[beta] + g2r[beta], g1i[beta] + g2i[beta]);

        for (std::size_t beta = 0; beta < dim; ++beta) {
            transform_.jacobian_column(s.tables, s.f_re.data(), k, beta, s.v.data());   // v
            // Σ Ḟ·S by parts (each call rewrites the same v into v_tmp).
            cplx l(transform_.jacobian_column(s.tables, t.f_dot_re.data(), k, beta,
                                              t.v_tmp.data()),
                   transform_.jacobian_column(s.tables, t.f_dot_im.data(), k, beta,
                                              t.v_tmp.data()));
            // Σ F·Ṡ by parts, and v̇.
            l += cplx(transform_.jacobian_column(t.tables, s.f_re.data(), k, beta,
                                                 t.v_dot.data(), /*with_identity=*/false),
                      transform_.jacobian_column(t.tables, s.f_im.data(), k, beta,
                                                 t.v_dot.data(), /*with_identity=*/false));
            // ∂(vᵀ T v) = 2 v̇ᵀ T v + vᵀ Ṫ v.
            for (std::size_t i = 0; i < n_rows_; ++i) {
                const double* v = s.v.data() + i * dim;
                const double* vd = t.v_dot.data() + i * dim;
                const cplx* T = s.t.data() + i * dd;
                const cplx* Td = t.t_dot.data() + i * dd;
                for (std::size_t a = 0; a < dim; ++a)
                    for (std::size_t g = 0; g < dim; ++g)
                        l += 2.0 * vd[a] * v[g] * T[a * dim + g] + v[a] * v[g] * Td[a * dim + g];
            }
            // ∂(−Σ U_ji U_ij) = −2 Σ U̇_ji U_ij, U_ji = Σ_α v_iα G^α_ji.
            for (std::size_t b = 0; b < blocks_.size(); ++b) {
                const std::size_t n = blocks_[b].last - blocks_[b].first;
                const std::size_t row0 = blocks_[b].first - first_row_;
                for (std::size_t j = 0; j < n; ++j)
                    for (std::size_t r = 0; r < n; ++r) {
                        cplx acc(0.0, 0.0), acc_dot(0.0, 0.0);
                        for (std::size_t a = 0; a < dim; ++a) {
                            const std::size_t gi = a * inv_size_ + inv_offset_[b] + j * n + r;
                            const double v = s.v[(row0 + r) * dim + a];
                            const double vd = t.v_dot[(row0 + r) * dim + a];
                            acc += v * s.gmat[gi];
                            acc_dot += vd * s.gmat[gi] + v * t.gmat_dot[gi];
                        }
                        t.u[j * n + r] = acc;
                        t.u_dot[j * n + r] = acc_dot;
                    }
                cplx uu(0.0, 0.0);
                for (std::size_t j = 0; j < n; ++j)
                    for (std::size_t r = 0; r < n; ++r) uu += t.u_dot[j * n + r] * t.u[r * n + j];
                l -= 2.0 * uu;
            }
            t.out_lap[k * dim + beta] = l;
        }
    }
}

bool ComplexBackflowSlaterDeterminant::compute_parameter_gradient_derivatives(
    const WalkerSet& batch, std::span<double> dgrad, std::span<double> dlap) const {
    std::vector<double> dpgrad(dgrad.size()), dplap(dlap.size());
    return compute_complex_parameter_gradient_derivatives(batch, dgrad, dlap, dpgrad, dplap);
}

bool ComplexBackflowSlaterDeterminant::compute_complex_parameter_gradient_derivatives(
    const WalkerSet& batch, std::span<double> dgrad, std::span<double> dlap,
    std::span<double> dpgrad, std::span<double> dplap) const {
    if (!supports_parameter_gradient_derivatives()) return false;
    const std::size_t nw = batch.num_walkers(), dim = dim_, dd = dim * dim, n_max = max_block_;
    const std::size_t field = np_ * dim * nw, xs = n_rows_ * dim;
    for (std::span<double> o : {dgrad, dlap, dpgrad, dplap}) std::fill(o.begin(), o.end(), 0.0);
    std::vector<std::size_t> column(params_.size(), npos);
    std::size_t n_opt = 0;
    for (std::size_t q = 0; q < params_.size(); ++q)
        if (!params_[q].is_fixed) column[q] = n_opt++;
    if (n_opt == 0) return true;

    Tangent t;   // measurement frequency: sized once per call
    size_tangent(t);
    Scratch& s = scratch();
    for (std::size_t w = 0; w < nw; ++w) {
        // Pair tables from the positions (separations only), orbitals at the
        // committed x, in the gauge of the committed inverse and F.
        load_positions(batch, w, s);
        transform_.full(s.pos.data(), s.x.data(), s.tables);
        std::copy_n(x_.begin() + static_cast<std::ptrdiff_t>(w * xs), xs, s.x.begin());
        evaluate_rows(s);
        for (std::size_t i = 0; i < n_rows_; ++i) {
            const ComplexOrbitalSet& orb = *blocks_[block_of_row_[i]].orbitals;
            orb.evaluate_third_derivatives(
                std::span<const double>(s.x.data() + i * dim, dim), dim,
                std::span<cplx>(t.third.data() + i * n_max * dd * dim, orb.size() * dd * dim));
        }
        const cplx* inv = inv_.data() + w * inv_size_;
        split_f(s, f_.data() + w * xs);
        gradient_matrices(s, inv);

        for (std::size_t q = 0; q < params_.size(); ++q) {
            if (column[q] == npos) continue;
            // The orbitals ride along x: ∂_p(∂^n φ(x_i)) = ∇(∂^n φ)·ẋ_i.
            transform_.parameter_tangent(s.pos.data(), q, t.x_dot.data(), t.tables, t.eta);
            std::fill(t.m_dot.begin(), t.m_dot.end(), cplx(0.0, 0.0));
            std::fill(t.g_dot.begin(), t.g_dot.end(), cplx(0.0, 0.0));
            std::fill(t.h_dot.begin(), t.h_dot.end(), cplx(0.0, 0.0));
            for (std::size_t i = 0; i < n_rows_; ++i) {
                const std::size_t n = blocks_[block_of_row_[i]].orbitals->size();
                const double* xd = t.x_dot.data() + i * dim;
                for (std::size_t m = 0; m < n; ++m) {
                    const std::size_t e = i * n_max + m;
                    cplx mv(0.0, 0.0);
                    for (std::size_t c = 0; c < dim; ++c) mv += s.grad[e * dim + c] * xd[c];
                    t.m_dot[e] = mv;
                    for (std::size_t a = 0; a < dim; ++a) {
                        cplx gv(0.0, 0.0);
                        for (std::size_t c = 0; c < dim; ++c)
                            gv += s.hess[(e * dim + a) * dim + c] * xd[c];
                        t.g_dot[e * dim + a] = gv;
                        for (std::size_t g = 0; g < dim; ++g) {
                            cplx hv(0.0, 0.0);
                            for (std::size_t c = 0; c < dim; ++c)
                                hv += t.third[((e * dim + a) * dim + g) * dim + c] * xd[c];
                            t.h_dot[(e * dim + a) * dim + g] = hv;
                        }
                    }
                }
            }
            tangent_derivatives(s, t, inv);
            const std::size_t off = column[q] * field;
            for (std::size_t k = 0; k < np_; ++k)
                for (std::size_t d = 0; d < dim; ++d) {
                    const std::size_t j = off + (k * dim + d) * nw + w;
                    const cplx g = t.out_grad[k * dim + d], l = t.out_lap[k * dim + d];
                    dgrad[j] = g.real();
                    dlap[j] = l.real();
                    dpgrad[j] = g.imag();
                    dplap[j] = l.imag();
                }
        }
    }
    return true;
}

}  // namespace alea
