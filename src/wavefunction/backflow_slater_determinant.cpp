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

    if (channels.empty())
        throw std::invalid_argument("backflow determinant: no backflow channel");
    for (std::size_t c = 0; c < channels.size(); ++c) {
        Channel& ch = channels[c];
        const std::string label = "backflow channel " + std::to_string(c);
        if (ch.source_last <= ch.source_first)
            throw std::invalid_argument(label + ": the source group is empty");
        for (const ChannelState& other : channels_)
            if (ch.source_first < other.source_last && other.source_first < ch.source_last)
                throw std::invalid_argument(label + ": two channels share source particles; "
                                            "give each source group one channel");
        // A source overlapping the rows must be the group itself, of one spin:
        // its pairs are like-spin, and η'(0) = 0 follows (backflow_function.h).
        const bool overlaps = ch.source_first < first_row_ + n_rows_ &&
                              first_row_ < ch.source_last;
        if (overlaps && (ch.source_first != first_row_ || ch.source_last != first_row_ + n_rows_))
            throw std::invalid_argument(label + ": a source overlapping the determinant's "
                                        "group must be that whole group");
        if (overlaps && blocks_.size() != 1)
            throw std::invalid_argument(
                label + ": a same-group channel needs a group of one spin; with both "
                "spins in one group its pairs are partly like-spin (eta'(0) = 0) and "
                "partly not.  Put each spin in its own group");
        if (ch.params.size() < 2)
            throw std::invalid_argument(label + ": needs c_0 and the cutoff L");
        std::vector<double> free;
        for (std::size_t k = 0; k + 1 < ch.params.size(); ++k) free.push_back(ch.params[k].value);
        const double L = ch.params.back().value;
        if (cell_.is_periodic() && L > cell_.ws_radius())
            throw std::invalid_argument(
                label + ": cutoff " + std::to_string(L) + " exceeds the Wigner-Seitz "
                "inradius " + std::to_string(cell_.ws_radius()) + "; beyond it the "
                "minimum image is not the periodic sum");
        BackflowFunction eta(ch.truncation, L, std::move(free), overlaps);
        // Names [c_0, (c_1), c_2.., L], the cutoff positive and, under PBC,
        // clamped at the inradius by the optimizer.
        std::size_t power = 0;
        for (std::size_t k = 0; k + 1 < ch.params.size(); ++k, ++power) {
            if (overlaps && power == 1) ++power;
            ch.params[k].name = "bf" + std::to_string(c) + "_c" + std::to_string(power);
        }
        Parameter& cut = ch.params.back();
        cut.name = "bf" + std::to_string(c) + "_L";
        cut.sign_locked = true;
        if (cell_.is_periodic()) cut.upper_bound = cell_.ws_radius();
        channels_.push_back({ch.source_first, ch.source_last, std::move(eta), params_.size()});
        params_.insert(params_.end(), ch.params.begin(), ch.params.end());
    }
}

BackflowSlaterDeterminant::BackflowSlaterDeterminant(const BackflowSlaterDeterminant& o)
    : WavefunctionComponent(o), channels_(o.channels_), cell_(o.cell_),
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
    for (ChannelState& ch : channels_)
        for (std::size_t k = 0; k < ch.eta.n_parameters(); ++k) {
            const Parameter& p = params_[ch.param_offset + k];
            if (k + 1 == ch.eta.n_parameters() && p.value > p.upper_bound)
                throw std::runtime_error("backflow: cutoff " + p.name + " = " +
                                         std::to_string(p.value) + " exceeds its bound " +
                                         std::to_string(p.upper_bound));
            ch.eta.set_parameter(k, p.value);
        }
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
    for (const ChannelState& ch : channels_)
        if (ch.source_last > np)
            throw std::invalid_argument("backflow determinant: source beyond the particle count");
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
    std::size_t max_channel_params = 1, max_orbital_params = 1;
    for (const ChannelState& ch : channels_)
        max_channel_params = std::max(max_channel_params, ch.eta.n_parameters());
    for (const Block& b : blocks_)
        max_orbital_params = std::max(max_orbital_params, b.orbitals->n_parameters());
    const std::size_t n = max_block_, rows = n_rows_;
    {
        s.pos.assign(np * dim, 0.0);
        s.x.assign(rows * dim, 0.0);
        s.val.assign(rows * n, 0.0);
        s.grad.assign(rows * n * dim, 0.0);
        s.hess.assign(rows * n * dim * dim, 0.0);
        s.pair_b.assign(rows * np * dim * dim, 0.0);
        s.pair_s.assign(rows * np * dim * dim, 0.0);
        s.linked.assign(rows * np, 0);
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
    if (p >= first_row_ && p < first_row_ + n_rows_) return true;
    for (const ChannelState& ch : channels_)
        if (p >= ch.source_first && p < ch.source_last) return true;
    return false;
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

std::size_t BackflowSlaterDeterminant::channel_of_source(std::size_t j) const noexcept {
    for (std::size_t c = 0; c < channels_.size(); ++c)
        if (j >= channels_[c].source_first && j < channels_[c].source_last) return c;
    return npos;
}

// The pair (row at ri, source at rj) of channel ch: adds sign * eta(r) r_ij to
// x and, when B is given, fills B_ab = eta delta_ab + eta' r_a r_b / r and
// s_ab = d^2/dr_b^2 [eta(r) r_a].  False (nothing added) beyond the cutoff.  At
// r = 0 the r-hat terms of B vanish with r (B -> eta(0) delta) and s is bounded
// and direction-dependent; that single point is left at zero.
bool BackflowSlaterDeterminant::pair_terms(const ChannelState& ch, const double* ri,
                                           const double* rj, double sign, double* x,
                                           double* B, double* S) const {
    const std::size_t dim = dim_;
    double del[SimulationCell::kMaxDim] = {};
    for (std::size_t d = 0; d < dim; ++d) del[d] = ri[d] - rj[d];
    cell_.min_image(del);
    double r2 = 0.0;
    for (std::size_t d = 0; d < dim; ++d) r2 += del[d] * del[d];
    const double r = std::sqrt(r2);
    if (r >= ch.eta.cutoff()) return false;
    double eta = 0.0, e1 = 0.0, e2 = 0.0;
    ch.eta.evaluate(r, eta, e1, e2);
    for (std::size_t d = 0; d < dim; ++d) x[d] += sign * eta * del[d];
    if (!B) return true;
    const double inv_r = r > 0.0 ? 1.0 / r : 0.0;
    for (std::size_t a = 0; a < dim; ++a)
        for (std::size_t b = 0; b < dim; ++b) {
            B[a * dim + b] = (a == b ? eta : 0.0) + e1 * del[a] * del[b] * inv_r;
            const double rb2 = del[b] * del[b] * inv_r * inv_r;
            S[a * dim + b] = (a == b ? 2.0 * e1 * del[b] * inv_r : 0.0) +
                             del[a] * (e2 * rb2 + e1 * inv_r * (1.0 - rb2));
        }
    return true;
}

void BackflowSlaterDeterminant::quasi_coordinates(Scratch& s) const {
    const std::size_t dim = dim_, np = np_, dd = dim * dim;
    std::fill(s.linked.begin(), s.linked.end(), 0);
    std::fill(s.pair_b.begin(), s.pair_b.end(), 0.0);
    std::fill(s.pair_s.begin(), s.pair_s.end(), 0.0);
    for (std::size_t i = 0; i < n_rows_; ++i) {
        const std::size_t P = first_row_ + i;
        double* x = s.x.data() + i * dim;
        for (std::size_t d = 0; d < dim; ++d) x[d] = s.pos[P * dim + d];
        for (const ChannelState& ch : channels_)
            for (std::size_t j = ch.source_first; j < ch.source_last; ++j) {
                if (j == P) continue;
                const std::size_t e = i * np + j;
                s.linked[e] = pair_terms(ch, s.pos.data() + P * dim, s.pos.data() + j * dim,
                                         1.0, x, s.pair_b.data() + e * dd,
                                         s.pair_s.data() + e * dd);
            }
    }
}

// Only the table entries the derivatives of particle k read: its own row
// against every source (rebuilding that row of s.x from scratch), and every row
// against k.  O(N) pairs instead of the O(N^2) of quasi_coordinates; the other
// entries keep whatever they held.
void BackflowSlaterDeterminant::particle_tables(Scratch& s, std::size_t k) const {
    const std::size_t dim = dim_, np = np_, dd = dim * dim;
    const auto clear = [&](std::size_t e) {
        s.linked[e] = 0;
        std::fill_n(s.pair_b.begin() + static_cast<std::ptrdiff_t>(e * dd), dd, 0.0);
        std::fill_n(s.pair_s.begin() + static_cast<std::ptrdiff_t>(e * dd), dd, 0.0);
    };
    if (k >= first_row_ && k < first_row_ + n_rows_) {
        const std::size_t ik = k - first_row_;
        double* x = s.x.data() + ik * dim;
        for (std::size_t d = 0; d < dim; ++d) x[d] = s.pos[k * dim + d];
        for (std::size_t j = 0; j < np; ++j) clear(ik * np + j);
        for (const ChannelState& ch : channels_)
            for (std::size_t j = ch.source_first; j < ch.source_last; ++j) {
                if (j == k) continue;
                const std::size_t e = ik * np + j;
                s.linked[e] = pair_terms(ch, s.pos.data() + k * dim, s.pos.data() + j * dim,
                                         1.0, x, s.pair_b.data() + e * dd,
                                         s.pair_s.data() + e * dd);
            }
    }
    const std::size_t c = channel_of_source(k);
    for (std::size_t i = 0; i < n_rows_; ++i) {
        const std::size_t P = first_row_ + i;
        if (P == k) continue;
        const std::size_t e = i * np + k;
        clear(e);
        if (c == npos) continue;
        double unused[SimulationCell::kMaxDim] = {};
        s.linked[e] = pair_terms(channels_[c], s.pos.data() + P * dim, s.pos.data() + k * dim,
                                 1.0, unused, s.pair_b.data() + e * dd,
                                 s.pair_s.data() + e * dd);
    }
}

// x' of a move of k from old_r to s.pos[k]: s.x holds the committed x, and
// every row that k displaces shifts by eta(r'_ik) r'_ik - eta(r_ik) r_ik.  The
// own row of k is rebuilt by particle_tables.
void BackflowSlaterDeterminant::shift_quasi_coordinates(Scratch& s, std::size_t k,
                                                        const double* old_r) const {
    const std::size_t c = channel_of_source(k);
    if (c == npos) return;
    const std::size_t dim = dim_;
    for (std::size_t i = 0; i < n_rows_; ++i) {
        const std::size_t P = first_row_ + i;
        if (P == k) continue;
        double* x = s.x.data() + i * dim;
        pair_terms(channels_[c], s.pos.data() + P * dim, s.pos.data() + k * dim, 1.0, x,
                   nullptr, nullptr);
        pair_terms(channels_[c], s.pos.data() + P * dim, old_r, -1.0, x, nullptr, nullptr);
    }
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

void BackflowSlaterDeterminant::particle_gradient(const Scratch& s, const double* f,
                                                  std::size_t k, double* out) const {
    const std::size_t dim = dim_, np = np_, dd = dim * dim;
    for (std::size_t b = 0; b < dim; ++b) out[b] = 0.0;
    const bool in_rows = k >= first_row_ && k < first_row_ + n_rows_;
    // Own row: J_{k,k} = 1 + Σ_j B(r_kj).
    if (in_rows) {
        const std::size_t ik = k - first_row_;
        const double* F = f + ik * dim;
        for (std::size_t b = 0; b < dim; ++b) out[b] += F[b];
        for (std::size_t j = 0; j < np; ++j) {
            if (!s.linked[ik * np + j]) continue;
            const double* B = s.pair_b.data() + (ik * np + j) * dd;
            for (std::size_t a = 0; a < dim; ++a)
                for (std::size_t b = 0; b < dim; ++b) out[b] += F[a] * B[a * dim + b];
        }
    }
    // Rows k displaces: J_{i,k} = −B(r_ik).
    for (std::size_t i = 0; i < n_rows_; ++i) {
        if (!s.linked[i * np + k]) continue;
        const double* F = f + i * dim;
        const double* B = s.pair_b.data() + (i * np + k) * dd;
        for (std::size_t a = 0; a < dim; ++a)
            for (std::size_t b = 0; b < dim; ++b) out[b] -= F[a] * B[a * dim + b];
    }
}

double BackflowSlaterDeterminant::jacobian_column(Scratch& s, const double* f, std::size_t k,
                                                  std::size_t beta) const {
    const std::size_t dim = dim_, np = np_, dd = dim * dim;
    std::fill(s.v.begin(), s.v.end(), 0.0);
    double fs = 0.0;   // Σ_{iα} F_iα S_{iα,kβ}
    const bool in_rows = k >= first_row_ && k < first_row_ + n_rows_;
    if (in_rows) {
        const std::size_t ik = k - first_row_;
        double* v = s.v.data() + ik * dim;
        v[beta] = 1.0;
        for (std::size_t j = 0; j < np; ++j) {
            if (!s.linked[ik * np + j]) continue;
            const double* B = s.pair_b.data() + (ik * np + j) * dd;
            const double* S = s.pair_s.data() + (ik * np + j) * dd;
            for (std::size_t a = 0; a < dim; ++a) {
                v[a] += B[a * dim + beta];
                fs += f[ik * dim + a] * S[a * dim + beta];
            }
        }
    }
    for (std::size_t i = 0; i < n_rows_; ++i) {
        if (!s.linked[i * np + k]) continue;
        const double* B = s.pair_b.data() + (i * np + k) * dd;
        const double* S = s.pair_s.data() + (i * np + k) * dd;
        double* v = s.v.data() + i * dim;
        for (std::size_t a = 0; a < dim; ++a) {
            v[a] = -B[a * dim + beta];
            fs += f[i * dim + a] * S[a * dim + beta];
        }
    }
    return fs;
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
        double lap = jacobian_column(s, f, k, beta);
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
        particle_gradient(s, f, k, grad + k * dim);
        for (std::size_t beta = 0; beta < dim; ++beta) {
            double l = jacobian_column(s, f, k, beta);
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
        quasi_coordinates(s);
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
        shift_quasi_coordinates(s, p, old_r);
        particle_tables(s, p);
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
        particle_gradient(s, pf, p, g);
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
    quasi_coordinates(s);
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
        particle_tables(s, p);
        double g[SimulationCell::kMaxDim] = {};
        particle_gradient(s, f_.data() + w * n_rows_ * dim, p, g);
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
        for (const ChannelState& ch : channels_) {
            const std::size_t np_c = ch.eta.n_parameters();
            bool any = false;
            for (std::size_t q = 0; q < np_c; ++q)
                any = any || column[ch.param_offset + q] != npos;
            if (!any) continue;
            for (std::size_t q = 0; q < np_c; ++q)
                if (column[ch.param_offset + q] != npos)
                    derivs[column[ch.param_offset + q] * nw + w] = 0.0;
            for (std::size_t i = 0; i < n_rows_; ++i) {
                const std::size_t P = first_row_ + i;
                for (std::size_t j = ch.source_first; j < ch.source_last; ++j) {
                    if (j == P) continue;
                    double del[SimulationCell::kMaxDim] = {};
                    for (std::size_t d = 0; d < dim; ++d)
                        del[d] = s.pos[P * dim + d] - s.pos[j * dim + d];
                    cell_.min_image(del);
                    double r2 = 0.0, fd = 0.0;
                    for (std::size_t d = 0; d < dim; ++d) {
                        r2 += del[d] * del[d];
                        fd += f[i * dim + d] * del[d];
                    }
                    const double r = std::sqrt(r2);
                    if (r >= ch.eta.cutoff()) continue;
                    ch.eta.parameter_derivatives(r, std::span<double>(s.param.data(), np_c));
                    for (std::size_t q = 0; q < np_c; ++q)
                        if (column[ch.param_offset + q] != npos)
                            derivs[column[ch.param_offset + q] * nw + w] += fd * s.param[q];
                }
            }
        }
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
