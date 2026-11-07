// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "wavefunction/pairing_determinant.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Dense>

#include "population/simulation_cell.h"
#include "population/walker_set.h"

namespace alea {

namespace {

constexpr double kPi       = 3.14159265358979323846;
constexpr double kDetFloor = 1e-300;   // log(0) guard for a walker on a node
constexpr int    kRebuildEvery = 64;   // Sherman–Morrison updates between LU rebuilds

using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

}  // namespace

PairingDeterminant::PairingDeterminant(PairOrbital orbital, std::vector<Block> blocks,
                                       std::vector<Parameter> params)
    : orbital_(std::move(orbital)), blocks_(std::move(blocks)), params_(std::move(params)) {
    if (blocks_.empty()) throw std::invalid_argument("pairing determinant: no block");
    const std::size_t n_par = orbital_.n_stars() + orbital_.n_poly_params();
    if (params_.size() != n_par)
        throw std::invalid_argument("pairing determinant: " + std::to_string(params_.size()) +
                                    " parameters for an orbital that has " +
                                    std::to_string(n_par));
    std::vector<const ParticleGroup*> seen;
    for (const Block& blk : blocks_) {
        Shape s;
        s.n_rows = blk.rows.size();
        s.n_columns = blk.columns.size();
        s.n = std::max(s.n_rows, s.n_columns);
        s.extra_columns = s.n_rows > s.n_columns;
        s.extra_rows = s.n_columns > s.n_rows;
        const std::size_t extra = s.n - std::min(s.n_rows, s.n_columns);
        const std::size_t given = blk.unpaired ? blk.unpaired->size() : 0;
        if (given != extra)
            throw std::invalid_argument(
                "pairing determinant: block (" + blk.rows.name + ", " + blk.columns.name +
                ") has " + std::to_string(s.n_rows) + " rows and " +
                std::to_string(s.n_columns) + " columns, so it needs " +
                std::to_string(extra) + " unpaired orbitals, got " + std::to_string(given));
        if (s.n == 0) throw std::invalid_argument("pairing determinant: empty block");
        for (const ParticleGroup* g : {&blk.rows, &blk.columns}) {
            for (const ParticleGroup* o : seen)
                if (g->first < o->last && o->first < g->last)
                    throw std::invalid_argument("pairing determinant: group '" + g->name +
                                                "' appears in two places");
            seen.push_back(g);
        }
        shape_.push_back(s);
    }
    rebuild();
}

PairingDeterminant::PairingDeterminant(const PairingDeterminant& o)
    : WavefunctionComponent(o), orbital_(o.orbital_), shape_(o.shape_), params_(o.params_),
      max_cutoff_(o.max_cutoff_), nw_(o.nw_), np_(o.np_), dim_(o.dim_), n_max_(o.n_max_),
      n_k_(o.n_k_), n_slots_(o.n_slots_), slot_(o.slot_), d_(o.d_), a_(o.a_), g_(o.g_),
      l_(o.l_), log_(o.log_), updates_(o.updates_), cos_(o.cos_), sin_(o.sin_),
      pend_v_(o.pend_v_), pend_g_(o.pend_g_), pend_l_(o.pend_l_), pend_r_(o.pend_r_), pend_scale_(o.pend_scale_),
      pend_cos_(o.pend_cos_), pend_sin_(o.pend_sin_), work_(o.work_), line_(o.line_),
      unp_v_(o.unp_v_), unp_g_(o.unp_g_), unp_l_(o.unp_l_) {
    blocks_.reserve(o.blocks_.size());
    for (const Block& b : o.blocks_)
        blocks_.push_back({b.rows, b.columns, b.unpaired ? b.unpaired->clone() : nullptr});
}

void PairingDeterminant::rebuild() {
    const std::size_t S = orbital_.n_stars();
    std::vector<double> p(S), c;
    for (std::size_t s = 0; s < S; ++s) p[s] = params_[s].value;
    double L = 0.0;
    if (orbital_.has_polynomial()) {
        const std::size_t M = static_cast<std::size_t>(orbital_.poly_order());
        for (std::size_t j = 0; j < M; ++j) c.push_back(params_[S + j].value);
        L = params_[S + M].value;
        if (L > max_cutoff_)
            throw std::runtime_error(
                "pairing determinant: cutoff L = " + std::to_string(L) +
                " exceeds the Wigner-Seitz inradius " + std::to_string(max_cutoff_) +
                "; beyond it the minimum image no longer sums every periodic image.");
    }
    orbital_.set(p, c, L);
}

PairingDeterminant::Location PairingDeterminant::locate(std::size_t p) const noexcept {
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        if (blocks_[b].rows.contains(p)) return {b, true, p - blocks_[b].rows.first};
        if (blocks_[b].columns.contains(p)) return {b, false, p - blocks_[b].columns.first};
    }
    return {};
}

void PairingDeterminant::trig(const double* x, double* c, double* s) const noexcept {
    for (std::size_t i = 0; i < n_k_; ++i) {
        const double* k = orbital_.k(i);
        double phase = 0.0;
        for (std::size_t d = 0; d < dim_; ++d) phase += k[d] * x[d];
        c[i] = std::cos(phase);
        s[i] = std::sin(phase);
    }
}

void PairingDeterminant::entry(const SimulationCell& cell, const double* x_row,
                               const double* x_col, const double* cos_row,
                               const double* sin_row, const double* cos_col,
                               const double* sin_col, double& value, double* grad,
                               double* lap) const {
    // Cosine part, factorized: cos(k·Δ) = c_r c_c + s_r s_c, sin(k·Δ) =
    // s_r c_c − c_r s_c.  Reciprocal vectors make it periodic, so Δ need not
    // be folded.  ∂_d cos(k·Δ) = −k_d sin(k·Δ), ∂²_d = −k_d² cos(k·Δ).
    value = 0.0;
    for (std::size_t d = 0; d < dim_; ++d) grad[d] = lap[d] = 0.0;
    for (std::size_t i = 0; i < n_k_; ++i) {
        const double w = orbital_.weight(i);
        const double ck = cos_row[i] * cos_col[i] + sin_row[i] * sin_col[i];
        const double sk = sin_row[i] * cos_col[i] - cos_row[i] * sin_col[i];
        const double* k = orbital_.k(i);
        value += w * ck;
        for (std::size_t d = 0; d < dim_; ++d) {
            grad[d] -= w * k[d] * sk;
            lap[d]  -= w * k[d] * k[d] * ck;
        }
    }
    if (!orbital_.has_polynomial()) return;
    // Polynomial part at the minimum image:
    //   ∂_d g(r) = g′ Δ_d/r,  ∂²_d g = g″ (Δ_d/r)² + (g′/r)(1 − (Δ_d/r)²).
    // At r = 0, g′(0) = 0 (no cusp) and every ∂²_d g = g″(0).
    double del[3] = {0.0, 0.0, 0.0};
    for (std::size_t d = 0; d < dim_; ++d) del[d] = x_row[d] - x_col[d];
    if (cell.is_periodic()) cell.min_image(del);
    double r2 = 0.0;
    for (std::size_t d = 0; d < dim_; ++d) r2 += del[d] * del[d];
    const double r = std::sqrt(r2);
    double g, dg, d2g;
    orbital_.radial(r, g, dg, d2g);
    value += g;
    if (r < 1e-12) {
        for (std::size_t d = 0; d < dim_; ++d) lap[d] += d2g;
        return;
    }
    const double inv_r = 1.0 / r;
    for (std::size_t d = 0; d < dim_; ++d) {
        const double u = del[d] * inv_r;
        grad[d] += dg * u;
        lap[d]  += d2g * u * u + dg * inv_r * (1.0 - u * u);
    }
}

void PairingDeterminant::size_state(std::size_t nw, std::size_t np, std::size_t dim) {
    nw_ = nw;
    np_ = np;
    dim_ = dim;
    n_k_ = orbital_.n_k();
    if (dim > 3) throw std::invalid_argument("pairing determinant: dim > 3");
    n_max_ = 0;
    for (const Shape& s : shape_) n_max_ = std::max(n_max_, s.n);
    slot_.assign(np, npos);
    n_slots_ = 0;
    for (const Block& blk : blocks_)
        for (const ParticleGroup* g : {&blk.rows, &blk.columns}) {
            if (g->last > np)
                throw std::invalid_argument("pairing determinant: group '" + g->name +
                                            "' lies beyond the batch's particles");
            for (std::size_t q = g->first; q < g->last; ++q) slot_[q] = n_slots_++;
        }
    const std::size_t B = blocks_.size();
    d_.resize(B); a_.resize(B); g_.resize(B); l_.resize(B); log_.resize(B); updates_.resize(B);
    for (std::size_t b = 0; b < B; ++b) {
        const std::size_t n2 = shape_[b].n * shape_[b].n;
        d_[b].assign(nw * n2, 0.0);
        a_[b].assign(nw * n2, 0.0);
        g_[b].assign(nw * n2 * dim, 0.0);
        l_[b].assign(nw * n2 * dim, 0.0);
        log_[b].assign(nw, 0.0);
        updates_[b].assign(nw, 0);
    }
    cos_.assign(nw * n_slots_ * n_k_, 0.0);
    sin_.assign(nw * n_slots_ * n_k_, 0.0);
    pend_v_.assign(nw * n_max_, 0.0);
    pend_g_.assign(nw * n_max_ * dim, 0.0);
    pend_l_.assign(nw * n_max_ * dim, 0.0);
    pend_r_.assign(nw, 1.0);
    pend_scale_.assign(nw, 0.0);
    pend_cos_.assign(nw * n_k_, 0.0);
    pend_sin_.assign(nw * n_k_, 0.0);
    work_.assign(nw * n_max_, 0.0);
    line_.assign(nw * n_max_, 0.0);
    std::size_t n_unp = 0;
    for (const Block& blk : blocks_)
        if (blk.unpaired) n_unp = std::max(n_unp, blk.unpaired->size());
    unp_v_.assign(n_unp * nw, 0.0);
    unp_g_.assign(n_unp * dim * nw, 0.0);
    unp_l_.assign(n_unp * dim * nw, 0.0);
}

void PairingDeterminant::prepare(std::size_t nw, std::size_t np, std::size_t dim) {
    resize_caches(nw, np, dim);
    size_state(nw, np, dim);
}

void PairingDeterminant::permute_caches(std::span<const std::size_t> src) {
    WavefunctionComponent::permute_caches(src);
    const std::size_t nw_new = src.size();
    auto permute = [&](auto& v, std::size_t stride) {
        auto out = v;
        out.resize(nw_new * stride);
        for (std::size_t c = 0; c < nw_new; ++c)
            std::copy_n(v.begin() + static_cast<std::ptrdiff_t>(src[c] * stride), stride,
                        out.begin() + static_cast<std::ptrdiff_t>(c * stride));
        v = std::move(out);
    };
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const std::size_t n2 = shape_[b].n * shape_[b].n;
        permute(d_[b], n2);
        permute(a_[b], n2);
        permute(g_[b], n2 * dim_);
        permute(l_[b], n2 * dim_);
        permute(log_[b], 1);
        permute(updates_[b], 1);
    }
    permute(cos_, n_slots_ * n_k_);
    permute(sin_, n_slots_ * n_k_);
    // Proposal scratch is not carried; resize it for the new walker count.
    nw_ = nw_new;
    pend_v_.assign(nw_ * n_max_, 0.0);
    pend_g_.assign(nw_ * n_max_ * dim_, 0.0);
    pend_l_.assign(nw_ * n_max_ * dim_, 0.0);
    pend_r_.assign(nw_, 1.0);
    pend_scale_.assign(nw_, 0.0);
    pend_cos_.assign(nw_ * n_k_, 0.0);
    pend_sin_.assign(nw_ * n_k_, 0.0);
    work_.assign(nw_ * n_max_, 0.0);
    line_.assign(nw_ * n_max_, 0.0);
    std::size_t n_unp = 0;
    for (const Block& blk : blocks_)
        if (blk.unpaired) n_unp = std::max(n_unp, blk.unpaired->size());
    unp_v_.assign(n_unp * nw_, 0.0);
    unp_g_.assign(n_unp * dim_ * nw_, 0.0);
    unp_l_.assign(n_unp * dim_ * nw_, 0.0);
}

double PairingDeterminant::rebuild_inverse(std::size_t b, std::size_t w) {
    const std::size_t n = shape_[b].n;
    const Eigen::Map<const RowMatrix> D(d_[b].data() + mat(b, w), static_cast<Eigen::Index>(n),
                                        static_cast<Eigen::Index>(n));
    const Eigen::PartialPivLU<RowMatrix> lu(D);
    Eigen::Map<RowMatrix> A(a_[b].data() + mat(b, w), static_cast<Eigen::Index>(n),
                            static_cast<Eigen::Index>(n));
    A = lu.inverse();
    updates_[b][w] = 0;
    return std::log(std::max(std::abs(lu.determinant()), kDetFloor));
}

void PairingDeterminant::block_derivatives(WalkerSet& batch, std::size_t b, std::size_t w,
                                           bool totals, std::size_t moved) {
    const Shape& s = shape_[b];
    const std::size_t n = s.n, nw = nw_, dim = dim_;
    const double* A = a_[b].data() + mat(b, w);
    const double* G = g_[b].data() + mat(b, w) * dim;
    const double* L = l_[b].data() + mat(b, w) * dim;
    std::span<double> b_grad = batch.gradients();
    std::span<double> b_lap  = batch.laplacians();
    auto store = [&](std::size_t particle, std::size_t d, double grad, double lap_raw) {
        const std::size_t idx = (particle * dim + d) * nw + w;
        const double lap = lap_raw - grad * grad;
        if (totals && particle != moved) {
            b_grad[idx] += grad - grad_c_[idx];
            b_lap[idx]  += lap - lap_c_[idx];
        }
        grad_c_[idx] = grad;
        lap_c_[idx]  = lap;
    };
    // Row particle a: Σ_c A[c][a] ∂D[a][c] (its whole row, extra columns included).
    for (std::size_t a = 0; a < s.n_rows; ++a)
        for (std::size_t d = 0; d < dim; ++d) {
            double grad = 0.0, lap = 0.0;
            for (std::size_t c = 0; c < n; ++c) {
                const double inv = A[c * n + a];
                grad += inv * G[(a * n + c) * dim + d];
                lap  += inv * L[(a * n + c) * dim + d];
            }
            store(blocks_[b].rows.first + a, d, grad, lap);
        }
    // Column particle c: φ entries depend on it through −∂φ (G holds the
    // row-owner derivative); extra rows χ_o(r_c) are its own.
    for (std::size_t c = 0; c < s.n_columns; ++c)
        for (std::size_t d = 0; d < dim; ++d) {
            double grad = 0.0, lap = 0.0;
            for (std::size_t a = 0; a < n; ++a) {
                const double inv = A[c * n + a];
                const double sign = a < s.n_rows ? -1.0 : 1.0;
                grad += inv * sign * G[(a * n + c) * dim + d];
                lap  += inv * L[(a * n + c) * dim + d];
            }
            store(blocks_[b].columns.first + c, d, grad, lap);
        }
}

void PairingDeterminant::evaluate_batch(WalkerSet& batch) {
    const std::size_t nw = batch.num_walkers(), np = batch.n_particles(), dim = batch.dim();
    prepare(nw, np, dim);
    const SimulationCell& cell = batch.cell();
    const std::span<const double> pos = batch.positions();
    double x[3] = {0.0, 0.0, 0.0}, y[3] = {0.0, 0.0, 0.0};
    auto position = [&](std::size_t q, std::size_t w, double* out) {
        for (std::size_t d = 0; d < dim; ++d) out[d] = pos[(q * dim + d) * nw + w];
    };

    // cos/sin(k·r) of every block particle.
    if (n_k_ > 0)
        for (std::size_t q = 0; q < np; ++q) {
            if (slot_[q] == npos) continue;
            for (std::size_t w = 0; w < nw; ++w) {
                position(q, w, x);
                const std::size_t off = (w * n_slots_ + slot_[q]) * n_k_;
                trig(x, cos_.data() + off, sin_.data() + off);
            }
        }

    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const Block& blk = blocks_[b];
        const Shape& s = shape_[b];
        const std::size_t n = s.n;

        // Unpaired orbitals at their owners: extra columns of the row
        // particles, or extra rows of the column particles.
        if (blk.unpaired) {
            const ParticleGroup& owner = s.extra_columns ? blk.rows : blk.columns;
            const std::size_t base = s.extra_columns ? s.n_columns : s.n_rows;
            const std::size_t n_unp = blk.unpaired->size();
            for (std::size_t j = 0; j < owner.size(); ++j) {
                const std::size_t q = owner.first + j;
                blk.unpaired->evaluate(pos.subspan(q * dim * nw, dim * nw), dim, unp_v_,
                                       unp_g_, unp_l_, 0, nw);
                for (std::size_t w = 0; w < nw; ++w)
                    for (std::size_t o = 0; o < n_unp; ++o) {
                        const std::size_t a = s.extra_columns ? j : base + o;
                        const std::size_t c = s.extra_columns ? base + o : j;
                        d_[b][mat(b, w) + a * n + c] = unp_v_[o * nw + w];
                        for (std::size_t d = 0; d < dim; ++d) {
                            const std::size_t e = (mat(b, w) + a * n + c) * dim + d;
                            g_[b][e] = unp_g_[(o * dim + d) * nw + w];
                            l_[b][e] = unp_l_[(o * dim + d) * nw + w];
                        }
                    }
            }
        }

        for (std::size_t w = 0; w < nw; ++w) {
            for (std::size_t a = 0; a < s.n_rows; ++a) {
                const std::size_t qa = blk.rows.first + a;
                position(qa, w, x);
                const double* ca = cos_.data() + (w * n_slots_ + slot_[qa]) * n_k_;
                const double* sa = sin_.data() + (w * n_slots_ + slot_[qa]) * n_k_;
                for (std::size_t c = 0; c < s.n_columns; ++c) {
                    const std::size_t qc = blk.columns.first + c;
                    position(qc, w, y);
                    const double* cc = cos_.data() + (w * n_slots_ + slot_[qc]) * n_k_;
                    const double* sc = sin_.data() + (w * n_slots_ + slot_[qc]) * n_k_;
                    const std::size_t e = mat(b, w) + a * n + c;
                    entry(cell, x, y, ca, sa, cc, sc, d_[b][e], g_[b].data() + e * dim,
                          l_[b].data() + e * dim);
                }
            }
            log_[b][w] = rebuild_inverse(b, w);
            log_psi_c_[w] += log_[b][w];
            block_derivatives(batch, b, w, false, npos);
        }
    }

    std::span<double> b_logpsi = batch.log_psi();
    for (std::size_t w = 0; w < nw; ++w) b_logpsi[w] += log_psi_c_[w];
    std::span<double> b_grad = batch.gradients();
    for (std::size_t k = 0; k < b_grad.size(); ++k) b_grad[k] += grad_c_[k];
    std::span<double> b_lap = batch.laplacians();
    for (std::size_t k = 0; k < b_lap.size(); ++k) b_lap[k] += lap_c_[k];
}

void PairingDeterminant::ratio_batch(WalkerSet& batch, std::size_t p, std::size_t w_begin,
                                     std::size_t w_end) {
    const Location loc = locate(p);
    if (loc.block == npos) return;   // not in any block: ψ unchanged
    const std::size_t nw = batch.num_walkers(), dim = batch.dim();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end) return;
    const std::size_t b = loc.block;
    const Block& blk = blocks_[b];
    const Shape& s = shape_[b];
    const std::size_t n = s.n;
    const SimulationCell& cell = batch.cell();
    const std::span<const double> pos = batch.positions();
    const std::span<const double> new_pos = batch.new_pos();
    std::span<double> log_ratios = batch.log_ratios();
    std::span<double> phase = batch.phase_diffs();
    std::span<double> new_grad = batch.new_grad();
    std::span<double> new_lap = batch.new_lap();

    // The moving particle's line has extra entries only on the larger side.
    const bool extras = loc.row ? s.extra_columns : s.extra_rows;
    const std::size_t base = loc.row ? s.n_columns : s.n_rows;   // first extra index
    if (extras)
        blk.unpaired->evaluate(new_pos, dim, unp_v_, unp_g_, unp_l_, w_begin, w_end);
    const ParticleGroup& partners = loc.row ? blk.columns : blk.rows;

    double xp[3] = {0.0, 0.0, 0.0}, y[3] = {0.0, 0.0, 0.0}, gtmp[3], ltmp[3];
    for (std::size_t w = w_begin; w < w_end; ++w) {
        for (std::size_t d = 0; d < dim; ++d) xp[d] = new_pos[d * nw + w];
        double* pc = pend_cos_.data() + w * n_k_;
        double* ps = pend_sin_.data() + w * n_k_;
        trig(xp, pc, ps);
        double* v  = pend_v_.data() + w * n_max_;
        double* gm = pend_g_.data() + w * n_max_ * dim;   // ∂ wrt the moving particle
        double* lm = pend_l_.data() + w * n_max_ * dim;
        const double* A = a_[b].data() + mat(b, w);

        double ratio = 0.0, scale = 0.0;
        for (std::size_t x = 0; x < n; ++x) {
            if (x < base) {
                const std::size_t q = partners.first + x;
                for (std::size_t d = 0; d < dim; ++d) y[d] = pos[(q * dim + d) * nw + w];
                const double* cq = cos_.data() + (w * n_slots_ + slot_[q]) * n_k_;
                const double* sq = sin_.data() + (w * n_slots_ + slot_[q]) * n_k_;
                if (loc.row) {
                    entry(cell, xp, y, pc, ps, cq, sq, v[x], gtmp, ltmp);
                    for (std::size_t d = 0; d < dim; ++d) gm[x * dim + d] = gtmp[d];
                } else {
                    entry(cell, y, xp, cq, sq, pc, ps, v[x], gtmp, ltmp);
                    for (std::size_t d = 0; d < dim; ++d) gm[x * dim + d] = -gtmp[d];
                }
                for (std::size_t d = 0; d < dim; ++d) lm[x * dim + d] = ltmp[d];
            } else {
                const std::size_t o = x - base;
                v[x] = unp_v_[o * nw + w];
                for (std::size_t d = 0; d < dim; ++d) {
                    gm[x * dim + d] = unp_g_[(o * dim + d) * nw + w];
                    lm[x * dim + d] = unp_l_[(o * dim + d) * nw + w];
                }
            }
            // Row a: A[x][a] (column a of A); column c: A[c][x] (row c).
            const double inv = loc.row ? A[x * n + loc.line] : A[loc.line * n + x];
            ratio += inv * v[x];
            scale += std::abs(inv * v[x]);
        }
        pend_r_[w] = ratio;
        pend_scale_[w] = scale;
        log_ratios[w] += std::log(std::max(std::abs(ratio), kDetFloor));
        if (ratio < 0.0) phase[w] += kPi;   // node crossing → the fixed-node gate rejects

        // The new inverse line is the old one over R.
        const double inv_r = ratio != 0.0 ? 1.0 / ratio : 0.0;
        for (std::size_t d = 0; d < dim; ++d) {
            double grad = 0.0, lap = 0.0;
            for (std::size_t x = 0; x < n; ++x) {
                const double inv = loc.row ? A[x * n + loc.line] : A[loc.line * n + x];
                grad += inv * gm[x * dim + d];
                lap  += inv * lm[x * dim + d];
            }
            grad *= inv_r;
            const double lap_new = lap * inv_r - grad * grad;
            new_grad[d * nw + w] += grad;
            new_lap[d * nw + w]  += lap_new;
            pending_grad_c_[d * nw + w] = grad;
            pending_lap_c_[d * nw + w]  = lap_new;
        }
    }
}

void PairingDeterminant::shifted_pair_ratios_batch(const WalkerSet& batch, std::size_t a_first,
                                                   std::size_t a_last, std::size_t b_first,
                                                   std::size_t b_last,
                                                   std::span<const double> shift,
                                                   std::span<double> log_ratios,
                                                   std::span<double> phases) {
    const std::size_t nw = batch.num_walkers(), dim = batch.dim();
    if (nw != nw_)
        throw std::logic_error(
            "pairing determinant: shifted pair ratios need the batch it evaluated");
    const std::size_t n_a = a_last - a_first, n_b = b_last - b_first, n_t = n_a + n_b;
    const SimulationCell& cell = batch.cell();
    const std::span<const double> pos = batch.positions();
    shift_pos_.resize(dim * nw);
    if (shift_delta_.size() < n_t * nw * n_max_) {
        shift_delta_.resize(n_t * nw * n_max_);
        shift_xa_.resize(n_t * nw * n_max_);
    }
    if (shift_ratio_.size() < n_t * nw) shift_ratio_.resize(n_t * nw);
    const auto particle = [&](std::size_t t) { return t < n_a ? a_first + t : b_first + t - n_a; };

    // Each moved particle alone: its new line at r + s against the unmoved
    // partners, the change x⁰ (row) or y⁰ (column), its ratio R and, for a
    // row, x⁰ᵀA.
    double xp[3] = {0.0, 0.0, 0.0}, y[3] = {0.0, 0.0, 0.0}, gtmp[3], ltmp[3];
    for (std::size_t t = 0; t < n_t; ++t) {
        const std::size_t x = particle(t);
        const Location loc = locate(x);
        if (loc.block == npos) continue;
        const std::size_t b = loc.block;
        const Block& blk = blocks_[b];
        const Shape& s = shape_[b];
        const std::size_t n = s.n;
        const bool extras = loc.row ? s.extra_columns : s.extra_rows;
        const std::size_t base = loc.row ? s.n_columns : s.n_rows;   // first extra index
        const ParticleGroup& partners = loc.row ? blk.columns : blk.rows;
        // r + s folded into the cell, as an accepted move would leave it.
        for (std::size_t w = 0; w < nw; ++w) {
            double r[3] = {0.0, 0.0, 0.0};
            for (std::size_t d = 0; d < dim; ++d)
                r[d] = pos[(x * dim + d) * nw + w] + shift[d * nw + w];
            cell.wrap(r);
            for (std::size_t d = 0; d < dim; ++d) shift_pos_[d * nw + w] = r[d];
        }
        if (extras) blk.unpaired->evaluate(shift_pos_, dim, unp_v_, unp_g_, unp_l_, 0, nw);
        for (std::size_t w = 0; w < nw; ++w) {
            for (std::size_t d = 0; d < dim; ++d) xp[d] = shift_pos_[d * nw + w];
            double* pc = pend_cos_.data() + w * n_k_;
            double* ps = pend_sin_.data() + w * n_k_;
            trig(xp, pc, ps);
            const double* A = a_[b].data() + mat(b, w);
            const double* D = d_[b].data() + mat(b, w);
            double* delta = shift_delta_.data() + (t * nw + w) * n_max_;
            double ratio = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                double v;
                if (k < base) {
                    const std::size_t q = partners.first + k;
                    for (std::size_t d = 0; d < dim; ++d) y[d] = pos[(q * dim + d) * nw + w];
                    const double* cq = cos_.data() + (w * n_slots_ + slot_[q]) * n_k_;
                    const double* sq = sin_.data() + (w * n_slots_ + slot_[q]) * n_k_;
                    if (loc.row)
                        entry(cell, xp, y, pc, ps, cq, sq, v, gtmp, ltmp);
                    else
                        entry(cell, y, xp, cq, sq, pc, ps, v, gtmp, ltmp);
                } else {
                    v = unp_v_[(k - base) * nw + w];
                }
                delta[k] = v - (loc.row ? D[loc.line * n + k] : D[k * n + loc.line]);
                ratio += (loc.row ? A[k * n + loc.line] : A[loc.line * n + k]) * v;
            }
            shift_ratio_[t * nw + w] = ratio;
            if (loc.row) {
                double* xa = shift_xa_.data() + (t * nw + w) * n_max_;
                for (std::size_t k = 0; k < n; ++k) {
                    double acc = 0.0;
                    for (std::size_t l = 0; l < n; ++l) acc += delta[l] * A[l * n + k];
                    xa[k] = acc;
                }
            }
        }
    }

    for (std::size_t ii = 0; ii < n_a; ++ii)
        for (std::size_t jj = 0; jj < n_b; ++jj) {
            const std::size_t ti = ii, tj = n_a + jj;
            const Location li = locate(a_first + ii), lj = locate(b_first + jj);
            if (li.block == npos && lj.block == npos) continue;
            const bool rank2 = li.block != npos && li.block == lj.block;
            if (rank2 && li.row == lj.row)
                throw std::logic_error(
                    "pairing determinant: two particles of one side of a block cannot be "
                    "shifted together");
            double* lr = log_ratios.data() + (ii * n_b + jj) * nw;
            double* ph = phases.data() + (ii * n_b + jj) * nw;
            for (std::size_t w = 0; w < nw; ++w) {
                double ratio;
                if (rank2) {
                    const std::size_t tr = li.row ? ti : tj, tc = li.row ? tj : ti;
                    const std::size_t r = li.row ? li.line : lj.line;
                    const std::size_t c = li.row ? lj.line : li.line;
                    const std::size_t b = li.block, n = shape_[b].n;
                    const double* A = a_[b].data() + mat(b, w);
                    const double* x0 = shift_delta_.data() + (tr * nw + w) * n_max_;
                    const double* y0 = shift_delta_.data() + (tc * nw + w) * n_max_;
                    const double* xa = shift_xa_.data() + (tr * nw + w) * n_max_;
                    const double rr1 = shift_ratio_[tr * nw + w] - 1.0;   // x⁰ᵀA e_r
                    const double rc1 = shift_ratio_[tc * nw + w] - 1.0;   // e_cᵀA y⁰
                    const double acr = A[c * n + r];
                    const double x0c = x0[c], y0r = y0[r];
                    double xay = 0.0;
                    for (std::size_t k = 0; k < n; ++k) xay += xa[k] * y0[k];
                    const double alpha = rr1 - x0c * acr;
                    const double gamma = rc1 - y0r * acr;
                    const double beta = xay - y0r * rr1 - x0c * rc1 + x0c * y0r * acr;
                    ratio = (1.0 + alpha) * (1.0 + gamma) - beta * acr;
                } else {
                    ratio = (li.block != npos ? shift_ratio_[ti * nw + w] : 1.0) *
                            (lj.block != npos ? shift_ratio_[tj * nw + w] : 1.0);
                }
                lr[w] += std::log(std::max(std::abs(ratio), kDetFloor));
                if (ratio < 0.0) ph[w] += kPi;
            }
        }
}

void PairingDeterminant::apply_move_batch(WalkerSet& batch, std::size_t p,
                                          std::span<const bool> accept, std::size_t w_begin,
                                          std::size_t w_end) {
    const Location loc = locate(p);
    if (loc.block == npos) return;
    const std::size_t nw = batch.num_walkers(), dim = batch.dim();
    if (w_end > nw) w_end = nw;
    const std::size_t b = loc.block;
    const Shape& s = shape_[b];
    const std::size_t n = s.n, line = loc.line;
    std::span<double> b_logpsi = batch.log_psi();

    for (std::size_t w = w_begin; w < w_end; ++w) {
        if (!accept[w]) continue;
        const double R = pend_r_[w];
        double* D = d_[b].data() + mat(b, w);
        double* A = a_[b].data() + mat(b, w);
        double* G = g_[b].data() + mat(b, w) * dim;
        double* L = l_[b].data() + mat(b, w) * dim;
        const double* v  = pend_v_.data() + w * n_max_;
        const double* gm = pend_g_.data() + w * n_max_ * dim;
        const double* lm = pend_l_.data() + w * n_max_ * dim;
        double* work = work_.data() + w * n_max_;
        double* old  = line_.data() + w * n_max_;

        // Rebuild instead of updating when the update would lose accuracy: a
        // tiny R (near a node) or one that cancels to roundoff, as in
        // SlaterDeterminant.
        bool rebuild = ++updates_[b][w] >= kRebuildEvery || !std::isfinite(R) ||
                       std::abs(R) < 1e-5 ||
                       std::abs(R) <= 1e4 * std::numeric_limits<double>::epsilon() *
                                          pend_scale_[w];
        if (loc.row) {
            // wᵀ = (v − d_a)ᵀ A;  A′ = A − (A e_a) wᵀ / R.
            if (!rebuild) {
                std::fill_n(work, n, 0.0);
                for (std::size_t x = 0; x < n; ++x) {
                    const double dv = v[x] - D[line * n + x];
                    for (std::size_t c = 0; c < n; ++c) work[c] += dv * A[x * n + c];
                }
                for (std::size_t i = 0; i < n; ++i) {
                    const double f = A[i * n + line] / R;
                    for (std::size_t c = 0; c < n; ++c) A[i * n + c] -= f * work[c];
                }
            }
            for (std::size_t x = 0; x < n; ++x) {
                D[line * n + x] = v[x];
                for (std::size_t d = 0; d < dim; ++d) {
                    G[(line * n + x) * dim + d] = gm[x * dim + d];
                    L[(line * n + x) * dim + d] = lm[x * dim + d];
                }
            }
        } else {
            // z = A (u − d^c);  A′ = A − z (e_cᵀ A) / R.
            if (!rebuild) {
                for (std::size_t j = 0; j < n; ++j) old[j] = A[line * n + j];
                for (std::size_t i = 0; i < n; ++i) {
                    double z = 0.0;
                    for (std::size_t x = 0; x < n; ++x) z += A[i * n + x] * (v[x] - D[x * n + line]);
                    work[i] = z / R;
                }
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j) A[i * n + j] -= work[i] * old[j];
            }
            // G keeps the row-owner derivative: −(∂ wrt the column particle)
            // for φ entries, the column particle's own ∂χ for extra rows.
            for (std::size_t x = 0; x < n; ++x) {
                D[x * n + line] = v[x];
                const double sign = x < s.n_rows ? -1.0 : 1.0;
                for (std::size_t d = 0; d < dim; ++d) {
                    G[(x * n + line) * dim + d] = sign * gm[x * dim + d];
                    L[(x * n + line) * dim + d] = lm[x * dim + d];
                }
            }
        }

        // Accuracy monitor.  Repeated Sherman–Morrison updates amplify the
        // inverse's error by roughly ‖A‖‖v − d‖/|R| each, which compounds
        // geometrically when ‖A‖ is large and small ratios are accepted.  The
        // updated line must give D′A′ = 1 there (row a: Σ_x D′_ax A′_xj = δ_aj;
        // column c: Σ_x A′_ix D′_xc = δ_ic), so its residual measures the error
        // at O(n²), the cost of the update itself; every line is checked when
        // its particle moves.  ‖D‖∞ ‖A‖∞ > 1e8 (ill-conditioned) also rebuilds.
        if (!rebuild) {
            double residual = 0.0, norm_d = 0.0, norm_a = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                double s = 0.0;
                if (loc.row)
                    for (std::size_t x = 0; x < n; ++x) s += D[line * n + x] * A[x * n + j];
                else
                    for (std::size_t x = 0; x < n; ++x) s += A[j * n + x] * D[x * n + line];
                residual = std::max(residual, std::abs(s - (j == line ? 1.0 : 0.0)));
            }
            for (std::size_t i = 0; i < n; ++i) {
                double row_d = 0.0, row_a = 0.0;
                for (std::size_t j = 0; j < n; ++j) {
                    row_d += std::abs(D[i * n + j]);
                    row_a += std::abs(A[i * n + j]);
                }
                norm_d = std::max(norm_d, row_d);
                norm_a = std::max(norm_a, row_a);
            }
            const double cond = norm_d * norm_a;
            rebuild = !(residual <= 1e-9) || !std::isfinite(cond) || cond > 1e8;
        }
        const double old_log = log_[b][w];
        log_[b][w] = rebuild ? rebuild_inverse(b, w) : old_log + std::log(std::abs(R));
        const double dlog = log_[b][w] - old_log;
        log_psi_c_[w] += dlog;
        b_logpsi[w] += dlog;

        if (n_k_ > 0) {
            const std::size_t off = (w * n_slots_ + slot_[p]) * n_k_;
            std::copy_n(pend_cos_.data() + w * n_k_, n_k_, cos_.data() + off);
            std::copy_n(pend_sin_.data() + w * n_k_, n_k_, sin_.data() + off);
        }
        block_derivatives(batch, b, w, true, p);
    }
}

std::size_t PairingDeterminant::walker_state_bytes() const noexcept {
    std::size_t per_walker = WavefunctionComponent::walker_state_bytes() / sizeof(double);
    for (const Shape& s : shape_) per_walker += s.n * s.n * (2 + 2 * dim_) + 2;
    per_walker += 2 * n_slots_ * n_k_;
    return per_walker * sizeof(double);
}

namespace {

// ∂D/∂θ, ∂(∂_d D)/∂θ and ∂(∂²_d D)/∂θ of one φ entry for every optimizable
// parameter θ (its column k in [0, n_opt)), at the minimum-image Δ.
struct EntryDerivs {
    std::vector<double> value, grad, lap;   // [n_opt], [n_opt * dim], [n_opt * dim]
};

}  // namespace

void PairingDeterminant::compute_parameter_derivatives(const WalkerSet& batch,
                                                       std::span<double> derivs) const {
    std::fill(derivs.begin(), derivs.end(), 0.0);
    const std::size_t n_opt = num_optimizable_params();
    if (n_opt == 0) return;
    const std::size_t nw = batch.num_walkers(), dim = batch.dim();
    const std::span<const double> pos = batch.positions();
    const SimulationCell& cell = batch.cell();
    const std::size_t S = orbital_.n_stars();

    // Column of each optimizable parameter, npos for fixed ones.
    std::vector<std::size_t> col(params_.size(), npos);
    for (std::size_t k = 0, j = 0; k < params_.size(); ++k)
        if (!params_[k].is_fixed) col[k] = j++;

    std::vector<double> star_cos(S);
    double x[3] = {0.0, 0.0, 0.0};
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const Block& blk = blocks_[b];
        const Shape& s = shape_[b];
        const std::size_t n = s.n;
        for (std::size_t w = 0; w < nw; ++w) {
            const double* A = a_[b].data() + mat(b, w);
            for (std::size_t a = 0; a < s.n_rows; ++a)
                for (std::size_t c = 0; c < s.n_columns; ++c) {
                    const std::size_t qa = blk.rows.first + a, qc = blk.columns.first + c;
                    const double inv = A[c * n + a];
                    const double* ca = cos_.data() + (w * n_slots_ + slot_[qa]) * n_k_;
                    const double* sa = sin_.data() + (w * n_slots_ + slot_[qa]) * n_k_;
                    const double* cc = cos_.data() + (w * n_slots_ + slot_[qc]) * n_k_;
                    const double* sc = sin_.data() + (w * n_slots_ + slot_[qc]) * n_k_;
                    // ∂φ/∂p_s = Σ_{k ∈ star s} cos(k·Δ).
                    std::fill(star_cos.begin(), star_cos.end(), 0.0);
                    for (std::size_t i = 0; i < n_k_; ++i)
                        star_cos[orbital_.star(i)] +=
                            orbital_.multiplicity(i) * (ca[i] * cc[i] + sa[i] * sc[i]);
                    for (std::size_t st = 0; st < S; ++st)
                        if (col[st] != npos) derivs[col[st] * nw + w] += inv * star_cos[st];
                    if (!orbital_.has_polynomial()) continue;
                    for (std::size_t d = 0; d < dim; ++d)
                        x[d] = pos[(qa * dim + d) * nw + w] - pos[(qc * dim + d) * nw + w];
                    if (cell.is_periodic()) cell.min_image(x);
                    double r2 = 0.0;
                    for (std::size_t d = 0; d < dim; ++d) r2 += x[d] * x[d];
                    const double r = std::sqrt(r2);
                    for (std::size_t j = 0; j < orbital_.n_poly_params(); ++j) {
                        if (col[S + j] == npos) continue;
                        double h, dh, d2h;
                        orbital_.radial_param(j, r, h, dh, d2h);
                        derivs[col[S + j] * nw + w] += inv * h;
                    }
                }
        }
    }
}

bool PairingDeterminant::compute_parameter_gradient_derivatives(const WalkerSet& batch,
                                                                std::span<double> dgrad,
                                                                std::span<double> dlap) const {
    std::fill(dgrad.begin(), dgrad.end(), 0.0);
    std::fill(dlap.begin(), dlap.end(), 0.0);
    const std::size_t n_opt = num_optimizable_params();
    if (n_opt == 0) return true;
    const std::size_t nw = batch.num_walkers(), np = batch.n_particles(), dim = batch.dim();
    const std::span<const double> pos = batch.positions();
    const SimulationCell& cell = batch.cell();
    const std::size_t S = orbital_.n_stars();
    const std::size_t stride = np * dim * nw;

    std::vector<std::size_t> opt;   // parameter index of each optimizable column
    for (std::size_t k = 0; k < params_.size(); ++k)
        if (!params_[k].is_fixed) opt.push_back(k);

    double del[3] = {0.0, 0.0, 0.0};
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const Block& blk = blocks_[b];
        const Shape& s = shape_[b];
        const std::size_t n = s.n;
        // Per walker: ∂D, ∂(∂D), ∂(∂²D) of every optimizable parameter
        // ([k][a][c] and [k][a][c][d]); the extra orbitals have none.
        std::vector<double> dD(n_opt * n * n), dG(n_opt * n * n * dim), dL(n_opt * n * n * dim);
        RowMatrix Am(n, n), dDm(n, n), dAm(n, n);
        for (std::size_t w = 0; w < nw; ++w) {
            std::fill(dD.begin(), dD.end(), 0.0);
            std::fill(dG.begin(), dG.end(), 0.0);
            std::fill(dL.begin(), dL.end(), 0.0);
            for (std::size_t a = 0; a < s.n_rows; ++a)
                for (std::size_t c = 0; c < s.n_columns; ++c) {
                    const std::size_t qa = blk.rows.first + a, qc = blk.columns.first + c;
                    const double* ca = cos_.data() + (w * n_slots_ + slot_[qa]) * n_k_;
                    const double* sa = sin_.data() + (w * n_slots_ + slot_[qa]) * n_k_;
                    const double* cc = cos_.data() + (w * n_slots_ + slot_[qc]) * n_k_;
                    const double* sc = sin_.data() + (w * n_slots_ + slot_[qc]) * n_k_;
                    double r = 0.0;
                    if (orbital_.has_polynomial()) {
                        for (std::size_t d = 0; d < dim; ++d)
                            del[d] = pos[(qa * dim + d) * nw + w] - pos[(qc * dim + d) * nw + w];
                        if (cell.is_periodic()) cell.min_image(del);
                        double r2 = 0.0;
                        for (std::size_t d = 0; d < dim; ++d) r2 += del[d] * del[d];
                        r = std::sqrt(r2);
                    }
                    for (std::size_t k = 0; k < n_opt; ++k) {
                        const std::size_t pk = opt[k];
                        const std::size_t e = (k * n + a) * n + c;
                        if (pk < S) {
                            for (std::size_t i = 0; i < n_k_; ++i) {
                                if (orbital_.star(i) != pk) continue;
                                const double m = orbital_.multiplicity(i);
                                const double ck = ca[i] * cc[i] + sa[i] * sc[i];
                                const double sk = sa[i] * cc[i] - ca[i] * sc[i];
                                const double* kv = orbital_.k(i);
                                dD[e] += m * ck;
                                for (std::size_t d = 0; d < dim; ++d) {
                                    dG[e * dim + d] -= m * kv[d] * sk;
                                    dL[e * dim + d] -= m * kv[d] * kv[d] * ck;
                                }
                            }
                        } else {
                            double h, dh, d2h;
                            orbital_.radial_param(pk - S, r, h, dh, d2h);
                            dD[e] += h;
                            if (r < 1e-12) {
                                for (std::size_t d = 0; d < dim; ++d) dL[e * dim + d] += d2h;
                            } else {
                                for (std::size_t d = 0; d < dim; ++d) {
                                    const double u = del[d] / r;
                                    dG[e * dim + d] += dh * u;
                                    dL[e * dim + d] += d2h * u * u + dh / r * (1.0 - u * u);
                                }
                            }
                        }
                    }
                }

            const double* A = a_[b].data() + mat(b, w);
            const double* G = g_[b].data() + mat(b, w) * dim;
            const double* L = l_[b].data() + mat(b, w) * dim;
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j) Am(i, j) = A[i * n + j];
            for (std::size_t k = 0; k < n_opt; ++k) {
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j) dDm(i, j) = dD[(k * n + i) * n + j];
                dAm.noalias() = -Am * dDm * Am;   // ∂A = −A (∂D) A
                double* og = dgrad.data() + k * stride;
                double* ol = dlap.data() + k * stride;
                // L = Lraw − G², so ∂L = ∂Lraw − 2 G ∂G.
                auto store = [&](std::size_t particle, std::size_t d, double dg_, double dl_) {
                    const std::size_t idx = (particle * dim + d) * nw + w;
                    og[idx] = dg_;
                    ol[idx] = dl_ - 2.0 * grad_c_[idx] * dg_;
                };
                for (std::size_t a = 0; a < s.n_rows; ++a)
                    for (std::size_t d = 0; d < dim; ++d) {
                        double dg_ = 0.0, dl_ = 0.0;
                        for (std::size_t c = 0; c < n; ++c) {
                            const std::size_t e = (a * n + c) * dim + d;
                            const std::size_t ek = ((k * n + a) * n + c) * dim + d;
                            dg_ += dAm(c, a) * G[e] + Am(c, a) * dG[ek];
                            dl_ += dAm(c, a) * L[e] + Am(c, a) * dL[ek];
                        }
                        store(blk.rows.first + a, d, dg_, dl_);
                    }
                for (std::size_t c = 0; c < s.n_columns; ++c)
                    for (std::size_t d = 0; d < dim; ++d) {
                        double dg_ = 0.0, dl_ = 0.0;
                        for (std::size_t a = 0; a < n; ++a) {
                            const double sign = a < s.n_rows ? -1.0 : 1.0;
                            const std::size_t e = (a * n + c) * dim + d;
                            const std::size_t ek = ((k * n + a) * n + c) * dim + d;
                            dg_ += sign * (dAm(c, a) * G[e] + Am(c, a) * dG[ek]);
                            dl_ += dAm(c, a) * L[e] + Am(c, a) * dL[ek];
                        }
                        store(blk.columns.first + c, d, dg_, dl_);
                    }
            }
        }
    }
    return true;
}

}  // namespace alea
