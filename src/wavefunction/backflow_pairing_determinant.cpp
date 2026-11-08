// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "wavefunction/backflow_pairing_determinant.h"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <utility>

#include "population/walker_set.h"

namespace alea {

namespace {

constexpr double kPi = 3.14159265358979323846;
// log|ψ'/ψ| of a singular proposal: never accepted, and no infinity reaches
// the acceptance test (as in the backflow Slater determinant).
constexpr double kSingularLogRatio = -700.0;

// Pivoted Gauss–Jordan: a = m⁻¹ (row-major n×n, m destroyed), log|det m| and
// its sign.  Golub & Van Loan, Matrix Computations, 4th ed., Sec. 3.1 (2013).
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

BackflowPairingDeterminant::BackflowPairingDeterminant(
    PairOrbital orbital, std::vector<Block> blocks, std::vector<Parameter> orbital_params,
    std::vector<BackflowTransformation::Channel> channels, const SimulationCell& cell)
    : orbital_(std::move(orbital)), blocks_(std::move(blocks)), cell_(cell),
      params_(std::move(orbital_params)) {
    if (blocks_.empty()) throw std::invalid_argument("backflow pairing determinant: no block");
    const std::size_t n_par = orbital_.n_stars() + orbital_.n_poly_params();
    if (params_.size() != n_par)
        throw std::invalid_argument("backflow pairing determinant: " +
                                    std::to_string(params_.size()) +
                                    " parameters for an orbital that has " + std::to_string(n_par));
    n_orbital_params_ = n_par;

    std::vector<std::size_t> transformed;
    std::vector<const ParticleGroup*> seen;
    for (const Block& blk : blocks_) {
        for (const ParticleGroup* g : {&blk.rows, &blk.columns}) {
            for (const ParticleGroup* o : seen)
                if (g->first < o->last && o->first < g->last)
                    throw std::invalid_argument("backflow pairing determinant: group '" +
                                                g->name + "' appears in two places");
            seen.push_back(g);
            for (std::size_t p = g->first; p < g->last; ++p) transformed.push_back(p);
        }
        const std::size_t nr = blk.rows.size(), nc = blk.columns.size();
        const std::size_t n = std::max(nr, nc), extra = n - std::min(nr, nc);
        const std::size_t given = blk.unpaired ? blk.unpaired->size() : 0;
        if (n == 0) throw std::invalid_argument("backflow pairing determinant: empty block");
        if (given != extra)
            throw std::invalid_argument(
                "backflow pairing determinant: block (" + blk.rows.name + ", " +
                blk.columns.name + ") needs " + std::to_string(extra) +
                " unpaired orbitals, got " + std::to_string(given));
        if (blk.unpaired && !blk.unpaired->supports_hessian())
            throw std::invalid_argument(
                "backflow pairing determinant: the unpaired orbitals provide no Hessian, "
                "which the backflow Laplacian needs (plane_wave does)");
    }
    std::sort(transformed.begin(), transformed.end());
    const auto t_of = [&](std::size_t p) {
        return static_cast<std::size_t>(
            std::lower_bound(transformed.begin(), transformed.end(), p) - transformed.begin());
    };
    for (const Block& blk : blocks_) {
        Shape sh;
        sh.n_rows = blk.rows.size();
        sh.n_columns = blk.columns.size();
        sh.n = std::max(sh.n_rows, sh.n_columns);
        sh.row_t.assign(sh.n, npos);
        sh.col_t.assign(sh.n, npos);
        for (std::size_t a = 0; a < sh.n_rows; ++a) sh.row_t[a] = t_of(blk.rows.first + a);
        for (std::size_t c = 0; c < sh.n_columns; ++c) sh.col_t[c] = t_of(blk.columns.first + c);
        sh.kind.assign(sh.n * sh.n, Kind::paired);
        for (std::size_t a = 0; a < sh.n; ++a)
            for (std::size_t c = 0; c < sh.n; ++c)
                sh.kind[a * sh.n + c] = a >= sh.n_rows ? Kind::column
                                      : c >= sh.n_columns ? Kind::row : Kind::paired;
        mat_offset_.push_back(mat_size_);
        mat_size_ += sh.n * sh.n;
        n_max_ = std::max(n_max_, sh.n);
        shape_.push_back(std::move(sh));
    }
    transform_ = BackflowTransformation(std::move(channels), std::move(transformed), cell_);
    const auto& bf = transform_.initial_parameters();
    params_.insert(params_.end(), bf.begin(), bf.end());
    rebuild();
}

BackflowPairingDeterminant::BackflowPairingDeterminant(const BackflowPairingDeterminant& o)
    : WavefunctionComponent(o), orbital_(o.orbital_), shape_(o.shape_),
      transform_(o.transform_), cell_(o.cell_), params_(o.params_),
      n_orbital_params_(o.n_orbital_params_), max_cutoff_(o.max_cutoff_), dim_(o.dim_),
      np_(o.np_), nw_(o.nw_), n_k_(o.n_k_), n_t_(o.n_t_), n_max_(o.n_max_),
      mat_offset_(o.mat_offset_), mat_size_(o.mat_size_), x_(o.x_), inv_(o.inv_), f_(o.f_),
      log_det_(o.log_det_), sign_(o.sign_), px_(o.px_), pinv_(o.pinv_), pf_(o.pf_),
      plog_det_(o.plog_det_), psign_(o.psign_), pgrad_(o.pgrad_), plap_(o.plap_),
      pvalid_(o.pvalid_), proposal_laplacian_(o.proposal_laplacian_),
      scratch_(o.scratch_.size()) {
    blocks_.reserve(o.blocks_.size());
    for (const Block& b : o.blocks_)
        blocks_.push_back({b.rows, b.columns, b.unpaired ? b.unpaired->clone() : nullptr});
}

void BackflowPairingDeterminant::bound_cutoff(double max_cutoff) noexcept {
    max_cutoff_ = max_cutoff;
    if (orbital_.has_polynomial()) params_[n_orbital_params_ - 1].upper_bound = max_cutoff;
}

void BackflowPairingDeterminant::rebuild() {
    const std::size_t S = orbital_.n_stars();
    std::vector<double> p(S), c;
    for (std::size_t s = 0; s < S; ++s) p[s] = params_[s].value;
    double L = 0.0;
    if (orbital_.has_polynomial()) {
        const auto M = static_cast<std::size_t>(orbital_.poly_order());
        for (std::size_t j = 0; j < M; ++j) c.push_back(params_[S + j].value);
        L = params_[S + M].value;
        if (L > max_cutoff_)
            throw std::runtime_error(
                "backflow pairing determinant: orbital cutoff L = " + std::to_string(L) +
                " exceeds the Wigner-Seitz inradius " + std::to_string(max_cutoff_));
    }
    orbital_.set(p, c, L);
    transform_.set_parameters(std::span<const Parameter>(params_).subspan(n_orbital_params_));
}

// ── Sizes ───────────────────────────────────────────────────────────────────

void BackflowPairingDeterminant::resize_state(std::size_t nw) {
    nw_ = nw;
    const std::size_t nb = blocks_.size(), xt = n_t_ * dim_;
    for (auto* v : {&x_, &f_, &px_, &pf_}) v->assign(nw * xt, 0.0);
    for (auto* v : {&inv_, &pinv_}) v->assign(nw * mat_size_, 0.0);
    for (auto* v : {&log_det_, &plog_det_}) v->assign(nw * nb, 0.0);
    for (auto* v : {&sign_, &psign_}) v->assign(nw * nb, 1.0);
    for (auto* v : {&pgrad_, &plap_}) v->assign(nw * dim_, 0.0);
    pvalid_.assign(nw, 0);
}

void BackflowPairingDeterminant::prepare(std::size_t nw, std::size_t np, std::size_t dim) {
    for (const Block& blk : blocks_)
        if (blk.rows.last > np || blk.columns.last > np)
            throw std::invalid_argument("backflow pairing determinant: a group beyond np");
    if (dim > SimulationCell::kMaxDim)
        throw std::invalid_argument("backflow pairing determinant: dim > 3");
    transform_.prepare(np, dim);
    resize_caches(nw, np, dim);
    if (np != np_ || dim != dim_) ++scratch_generation_;
    dim_ = dim;
    np_ = np;
    n_k_ = orbital_.n_k();
    n_t_ = transform_.n_transformed();
    resize_state(nw);
    const auto threads = static_cast<std::size_t>(
        std::max({1, omp_get_max_threads(), omp_get_num_procs()}));
    if (scratch_.size() < threads) scratch_.resize(threads);
}

void BackflowPairingDeterminant::size_scratch(Scratch& s) const {
    const std::size_t dim = dim_, n = n_max_, T = n_t_;
    s.pos.assign(np_ * dim, 0.0);
    s.x.assign(T * dim, 0.0);
    transform_.size_tables(s.tables);
    s.cx.assign(T * n_k_, 0.0);
    s.sx.assign(T * n_k_, 0.0);
    s.val.assign(mat_size_, 0.0);
    s.g.assign(mat_size_ * dim, 0.0);
    s.hess.assign(mat_size_ * dim * dim, 0.0);
    s.unp_v.assign(n, 0.0);
    s.unp_g.assign(n * dim, 0.0);
    s.unp_h.assign(n * dim * dim, 0.0);
    s.work.assign(n * n, 0.0);
    s.pq.assign(2 * dim * n * n, 0.0);
    s.xyz.assign((2 * dim + dim * dim) * mat_size_, 0.0);
    s.v.assign(T * dim, 0.0);
    s.va.assign(dim * n, 0.0);
    s.wc.assign(dim * n, 0.0);
    s.dm.assign(n * n, 0.0);
    s.e.assign(n * n, 0.0);
    s.param.assign(std::max(transform_.max_channel_parameters(), n_orbital_params_ + 1), 0.0);
    s.pderiv.assign(transform_.n_parameters(), 0.0);
    s.out_grad.assign(np_ * dim, 0.0);
    s.out_lap.assign(np_ * dim, 0.0);
    s.generation = scratch_generation_;
}

BackflowPairingDeterminant::Scratch& BackflowPairingDeterminant::scratch() const {
    const auto t = static_cast<std::size_t>(omp_get_thread_num());
    if (t >= scratch_.size())
        throw std::logic_error("backflow pairing determinant: OpenMP thread " +
                               std::to_string(t) + " beyond the " +
                               std::to_string(scratch_.size()) + " scratch slots");
    Scratch& s = scratch_[t];
    if (s.generation != scratch_generation_) size_scratch(s);   // once per thread
    return s;
}

std::size_t BackflowPairingDeterminant::walker_state_bytes() const noexcept {
    if (nw_ == 0) return WavefunctionComponent::walker_state_bytes();
    const std::size_t per_walker = 2 * n_t_ * dim_ + mat_size_ + 2 * blocks_.size();
    return WavefunctionComponent::walker_state_bytes() + per_walker * sizeof(double);
}

void BackflowPairingDeterminant::permute_caches(std::span<const std::size_t> src) {
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
    gather(x_, n_t_ * dim_);
    gather(inv_, mat_size_);
    gather(f_, n_t_ * dim_);
    gather(log_det_, blocks_.size());
    gather(sign_, blocks_.size());
    // resize_state sizes everything for the new count; the gathered committed
    // state is kept, the pending state (transient) starts fresh.
    std::vector<double> x = std::move(x_), inv = std::move(inv_), f = std::move(f_),
                        ld = std::move(log_det_), sg = std::move(sign_);
    resize_state(new_nw);
    x_ = std::move(x);
    inv_ = std::move(inv);
    f_ = std::move(f);
    log_det_ = std::move(ld);
    sign_ = std::move(sg);
}

// ── One configuration ───────────────────────────────────────────────────────

void BackflowPairingDeterminant::load_positions(const WalkerSet& batch, std::size_t w,
                                                Scratch& s) const {
    const std::size_t nw = batch.num_walkers();
    const std::span<const double> pos = batch.positions();
    for (std::size_t k = 0; k < np_; ++k)
        for (std::size_t d = 0; d < dim_; ++d) s.pos[k * dim_ + d] = pos[(k * dim_ + d) * nw + w];
}

// Every entry's value, gradient g and Hessian H at its argument.  φ(Δ):
//   cosines  Σ_i w_i cos(k_i·Δ):  g = −Σ w k sin,  H = −Σ w k kᵀ cos;
//   polynomial g(r) at the minimum image:  ∇ = g' Δ̂,
//            H = g'' Δ̂Δ̂ᵀ + (g'/r)(1 − Δ̂Δ̂ᵀ)  → g''(0) 1 at r = 0 (g'(0) = 0).
void BackflowPairingDeterminant::evaluate_entries(Scratch& s) const {
    const std::size_t dim = dim_, dd = dim * dim, nk = n_k_;
    for (std::size_t t = 0; t < n_t_; ++t)
        for (std::size_t i = 0; i < nk; ++i) {
            const double* k = orbital_.k(i);
            double phase = 0.0;
            for (std::size_t d = 0; d < dim; ++d) phase += k[d] * s.x[t * dim + d];
            s.cx[t * nk + i] = std::cos(phase);
            s.sx[t * nk + i] = std::sin(phase);
        }
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const Shape& sh = shape_[b];
        const std::size_t n = sh.n, off = mat_offset_[b];
        for (std::size_t a = 0; a < sh.n_rows; ++a)
            for (std::size_t c = 0; c < sh.n_columns; ++c) {
                const std::size_t e = off + a * n + c;
                const std::size_t ta = sh.row_t[a], tc = sh.col_t[c];
                double value = 0.0, g[SimulationCell::kMaxDim] = {},
                       H[SimulationCell::kMaxDim * SimulationCell::kMaxDim] = {};
                for (std::size_t i = 0; i < nk; ++i) {
                    const double w = orbital_.weight(i);
                    const double ca = s.cx[ta * nk + i], sa = s.sx[ta * nk + i];
                    const double cc = s.cx[tc * nk + i], sc = s.sx[tc * nk + i];
                    const double ck = ca * cc + sa * sc, sk = sa * cc - ca * sc;
                    const double* kv = orbital_.k(i);
                    value += w * ck;
                    for (std::size_t al = 0; al < dim; ++al) {
                        g[al] -= w * kv[al] * sk;
                        for (std::size_t ga = 0; ga < dim; ++ga)
                            H[al * dim + ga] -= w * kv[al] * kv[ga] * ck;
                    }
                }
                if (orbital_.has_polynomial()) {
                    double del[SimulationCell::kMaxDim] = {};
                    for (std::size_t d = 0; d < dim; ++d)
                        del[d] = s.x[ta * dim + d] - s.x[tc * dim + d];
                    cell_.min_image(del);
                    double r2 = 0.0;
                    for (std::size_t d = 0; d < dim; ++d) r2 += del[d] * del[d];
                    const double r = std::sqrt(r2);
                    double gr = 0.0, dg = 0.0, d2g = 0.0;
                    orbital_.radial(r, gr, dg, d2g);
                    value += gr;
                    if (r < 1e-12) {
                        for (std::size_t d = 0; d < dim; ++d) H[d * dim + d] += d2g;
                    } else {
                        for (std::size_t al = 0; al < dim; ++al) {
                            const double ua = del[al] / r;
                            g[al] += dg * ua;
                            for (std::size_t ga = 0; ga < dim; ++ga) {
                                const double ug = del[ga] / r;
                                H[al * dim + ga] += d2g * ua * ug +
                                                    dg / r * ((al == ga ? 1.0 : 0.0) - ua * ug);
                            }
                        }
                    }
                }
                s.val[e] = value;
                for (std::size_t d = 0; d < dim; ++d) s.g[e * dim + d] = g[d];
                for (std::size_t q = 0; q < dd; ++q) s.hess[e * dd + q] = H[q];
            }
        // Unpaired orbitals at their owners' quasi-coordinates.
        const Block& blk = blocks_[b];
        if (!blk.unpaired) continue;
        const bool extra_columns = sh.n_rows > sh.n_columns;
        const std::size_t owners = extra_columns ? sh.n_rows : sh.n_columns;
        const std::size_t base = extra_columns ? sh.n_columns : sh.n_rows;
        const std::size_t n_unp = blk.unpaired->size();
        for (std::size_t j = 0; j < owners; ++j) {
            const std::size_t t = extra_columns ? sh.row_t[j] : sh.col_t[j];
            blk.unpaired->evaluate_hessian(std::span<const double>(s.x.data() + t * dim, dim), dim,
                                           std::span<double>(s.unp_v.data(), n_unp),
                                           std::span<double>(s.unp_g.data(), n_unp * dim),
                                           std::span<double>(s.unp_h.data(), n_unp * dd), 0, 1);
            for (std::size_t o = 0; o < n_unp; ++o) {
                const std::size_t a = extra_columns ? j : base + o;
                const std::size_t c = extra_columns ? base + o : j;
                const std::size_t e = off + a * n + c;
                s.val[e] = s.unp_v[o];
                for (std::size_t d = 0; d < dim; ++d) s.g[e * dim + d] = s.unp_g[o * dim + d];
                for (std::size_t q = 0; q < dd; ++q) s.hess[e * dd + q] = s.unp_h[o * dd + q];
            }
        }
    }
}

bool BackflowPairingDeterminant::invert_blocks(Scratch& s, double* inv, double* log_det,
                                               double* sign) const {
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const std::size_t n = shape_[b].n, off = mat_offset_[b];
        std::copy_n(s.val.begin() + static_cast<std::ptrdiff_t>(off), n * n, s.work.begin());
        if (!invert(s.work.data(), inv + off, n, log_det[b], sign[b])) return false;
    }
    return true;
}

// F_t = Σ_entries A_ca ∂M_ac/∂x_t: +g for the row owner, −g for the column
// owner of a paired entry, +g for the owner of an unpaired one.
void BackflowPairingDeterminant::compute_f(const Scratch& s, const double* inv,
                                           double* f) const {
    const std::size_t dim = dim_;
    std::fill(f, f + n_t_ * dim, 0.0);
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const Shape& sh = shape_[b];
        const std::size_t n = sh.n, off = mat_offset_[b];
        for (std::size_t a = 0; a < n; ++a)
            for (std::size_t c = 0; c < n; ++c) {
                const std::size_t e = off + a * n + c;
                const double A = inv[off + c * n + a];
                const double* g = s.g.data() + e * dim;
                const Kind kind = sh.kind[a * n + c];
                if (kind != Kind::column)
                    for (std::size_t d = 0; d < dim; ++d) f[sh.row_t[a] * dim + d] += A * g[d];
                if (kind == Kind::paired)
                    for (std::size_t d = 0; d < dim; ++d) f[sh.col_t[c] * dim + d] -= A * g[d];
                if (kind == Kind::column)
                    for (std::size_t d = 0; d < dim; ++d) f[sh.col_t[c] * dim + d] += A * g[d];
            }
    }
}

void BackflowPairingDeterminant::direction(const Shape& sh, const Scratch& s, std::size_t a,
                                           std::size_t c, double* u) const noexcept {
    const std::size_t dim = dim_;
    const Kind kind = sh.kind[a * sh.n + c];
    for (std::size_t d = 0; d < dim; ++d) u[d] = 0.0;
    if (kind != Kind::column)
        for (std::size_t d = 0; d < dim; ++d) u[d] += s.v[sh.row_t[a] * dim + d];
    if (kind == Kind::paired)
        for (std::size_t d = 0; d < dim; ++d) u[d] -= s.v[sh.col_t[c] * dim + d];
    if (kind == Kind::column)
        for (std::size_t d = 0; d < dim; ++d) u[d] += s.v[sh.col_t[c] * dim + d];
}

// One particle: fs + tr(A d²M) − tr((A dM)²) with A dM formed directly.
void BackflowPairingDeterminant::particle_laplacian(Scratch& s, const double* inv,
                                                    const double* f, std::size_t k,
                                                    double* out) const {
    const std::size_t dim = dim_, dd = dim * dim;
    for (std::size_t beta = 0; beta < dim; ++beta) {
        double lap = transform_.jacobian_column(s.tables, f, k, beta, s.v.data());
        for (std::size_t b = 0; b < blocks_.size(); ++b) {
            const Shape& sh = shape_[b];
            const std::size_t n = sh.n, off = mat_offset_[b];
            const double* A = inv + off;
            for (std::size_t a = 0; a < n; ++a)
                for (std::size_t c = 0; c < n; ++c) {
                    const std::size_t e = off + a * n + c;
                    double u[SimulationCell::kMaxDim];
                    direction(sh, s, a, c, u);
                    double dmv = 0.0, d2 = 0.0;
                    for (std::size_t al = 0; al < dim; ++al) {
                        dmv += s.g[e * dim + al] * u[al];
                        for (std::size_t ga = 0; ga < dim; ++ga)
                            d2 += u[al] * s.hess[e * dd + al * dim + ga] * u[ga];
                    }
                    s.dm[a * n + c] = dmv;
                    lap += A[c * n + a] * d2;
                }
            // E = A dM, E[c][c'] = Σ_a A[c][a] dM[a][c'].
            for (std::size_t c = 0; c < n; ++c)
                for (std::size_t c2 = 0; c2 < n; ++c2) {
                    double acc = 0.0;
                    for (std::size_t a = 0; a < n; ++a) acc += A[c * n + a] * s.dm[a * n + c2];
                    s.e[c * n + c2] = acc;
                }
            double tr = 0.0;
            for (std::size_t c = 0; c < n; ++c)
                for (std::size_t c2 = 0; c2 < n; ++c2) tr += s.e[c * n + c2] * s.e[c2 * n + c];
            lap -= tr;
        }
        out[beta] = lap;
    }
}

void BackflowPairingDeterminant::all_derivatives(Scratch& s, const double* inv,
                                                 const double* f, double* grad,
                                                 double* lap) const {
    const std::size_t dim = dim_, dd = dim * dim;
    // X_α = P_α A, Y_α = A Q_α, Z_αγ = X_α Q_γ of every block.
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const Shape& sh = shape_[b];
        const std::size_t n = sh.n, nn = n * n, off = mat_offset_[b];
        const double* A = inv + off;
        double* P = s.pq.data();
        double* Q = s.pq.data() + dim * nn;
        for (std::size_t a = 0; a < n; ++a)
            for (std::size_t c = 0; c < n; ++c) {
                const Kind kind = sh.kind[a * n + c];
                const double* g = s.g.data() + (off + a * n + c) * dim;
                for (std::size_t al = 0; al < dim; ++al) {
                    P[al * nn + a * n + c] = kind == Kind::column ? 0.0 : g[al];
                    Q[al * nn + a * n + c] = kind == Kind::paired ? -g[al]
                                           : kind == Kind::column ? g[al] : 0.0;
                }
            }
        double* base = s.xyz.data() + (2 * dim + dd) * off;
        for (std::size_t al = 0; al < dim; ++al) {
            double* X = base + al * nn;
            double* Y = base + (dim + al) * nn;
            for (std::size_t a = 0; a < n; ++a)
                for (std::size_t a2 = 0; a2 < n; ++a2) {
                    double acc = 0.0;
                    for (std::size_t c = 0; c < n; ++c) acc += P[al * nn + a * n + c] * A[c * n + a2];
                    X[a * n + a2] = acc;
                }
            for (std::size_t c = 0; c < n; ++c)
                for (std::size_t c2 = 0; c2 < n; ++c2) {
                    double acc = 0.0;
                    for (std::size_t a = 0; a < n; ++a) acc += A[c * n + a] * Q[al * nn + a * n + c2];
                    Y[c * n + c2] = acc;
                }
        }
        for (std::size_t al = 0; al < dim; ++al)
            for (std::size_t ga = 0; ga < dim; ++ga) {
                const double* X = base + al * nn;
                double* Z = base + (2 * dim + al * dim + ga) * nn;
                for (std::size_t a = 0; a < n; ++a)
                    for (std::size_t c = 0; c < n; ++c) {
                        double acc = 0.0;
                        for (std::size_t a2 = 0; a2 < n; ++a2)
                            acc += X[a * n + a2] * Q[ga * nn + a2 * n + c];
                        Z[a * n + c] = acc;
                    }
            }
    }
    std::fill(grad, grad + np_ * dim, 0.0);
    std::fill(lap, lap + np_ * dim, 0.0);
    for (std::size_t k = 0; k < np_; ++k) {
        if (!transform_.affects(k)) continue;
        transform_.gradient(s.tables, f, k, grad + k * dim);
        for (std::size_t beta = 0; beta < dim; ++beta) {
            double l = transform_.jacobian_column(s.tables, f, k, beta, s.v.data());
            for (std::size_t b = 0; b < blocks_.size(); ++b) {
                const Shape& sh = shape_[b];
                const std::size_t n = sh.n, nn = n * n, off = mat_offset_[b];
                const double* A = inv + off;
                const double* base = s.xyz.data() + (2 * dim + dd) * off;
                for (std::size_t al = 0; al < dim; ++al)
                    for (std::size_t i = 0; i < n; ++i) {
                        s.va[al * n + i] = i < sh.n_rows ? s.v[sh.row_t[i] * dim + al] : 0.0;
                        s.wc[al * n + i] = i < sh.n_columns ? s.v[sh.col_t[i] * dim + al] : 0.0;
                    }
                // tr(A d²M)
                for (std::size_t a = 0; a < n; ++a)
                    for (std::size_t c = 0; c < n; ++c) {
                        const std::size_t e = off + a * n + c;
                        double u[SimulationCell::kMaxDim];
                        direction(sh, s, a, c, u);
                        double d2 = 0.0;
                        for (std::size_t al = 0; al < dim; ++al)
                            for (std::size_t ga = 0; ga < dim; ++ga)
                                d2 += u[al] * s.hess[e * dd + al * dim + ga] * u[ga];
                        l += A[c * n + a] * d2;
                    }
                // tr((A dM)²) = Σ_αγ [T1 + 2 T2 + T3]
                double tr = 0.0;
                for (std::size_t al = 0; al < dim; ++al)
                    for (std::size_t ga = 0; ga < dim; ++ga) {
                        const double* Xa = base + al * nn;
                        const double* Xg = base + ga * nn;
                        const double* Ya = base + (dim + al) * nn;
                        const double* Yg = base + (dim + ga) * nn;
                        const double* Z = base + (2 * dim + al * dim + ga) * nn;
                        const double* Va = s.va.data() + al * n;
                        const double* Vg = s.va.data() + ga * n;
                        const double* Wa = s.wc.data() + al * n;
                        const double* Wg = s.wc.data() + ga * n;
                        for (std::size_t i = 0; i < n; ++i)
                            for (std::size_t j = 0; j < n; ++j) {
                                tr += Va[i] * Xa[i * n + j] * Vg[j] * Xg[j * n + i];
                                tr += 2.0 * Va[i] * Z[i * n + j] * Wg[j] * A[j * n + i];
                                tr += Wa[i] * Yg[i * n + j] * Wg[j] * Ya[j * n + i];
                            }
                    }
                l -= tr;
            }
            lap[k * dim + beta] = l;
        }
    }
}

// ── Full evaluation ─────────────────────────────────────────────────────────

void BackflowPairingDeterminant::evaluate_batch(WalkerSet& batch) {
    const std::size_t nw = batch.num_walkers(), np = batch.n_particles(), dim = batch.dim();
    prepare(nw, np, dim);
    const std::size_t nb = blocks_.size();
    long singular = -1;
#pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t ws = 0; ws < static_cast<std::ptrdiff_t>(nw); ++ws) {
        const auto w = static_cast<std::size_t>(ws);
        Scratch& s = scratch();
        load_positions(batch, w, s);
        transform_.full(s.pos.data(), s.x.data(), s.tables);
        evaluate_entries(s);
        double* inv = inv_.data() + w * mat_size_;
        if (!invert_blocks(s, inv, log_det_.data() + w * nb, sign_.data() + w * nb)) {
#pragma omp critical(backflow_pairing_singular)
            singular = static_cast<long>(w);
            continue;
        }
        std::copy(s.x.begin(), s.x.end(), x_.begin() + static_cast<std::ptrdiff_t>(w * n_t_ * dim));
        double* f = f_.data() + w * n_t_ * dim;
        compute_f(s, inv, f);
        all_derivatives(s, inv, f, s.out_grad.data(), s.out_lap.data());
        double log_psi = 0.0;
        for (std::size_t b = 0; b < nb; ++b) log_psi += log_det_[w * nb + b];
        log_psi_c_[w] = log_psi;
        for (std::size_t k = 0; k < np; ++k)
            for (std::size_t d = 0; d < dim; ++d) {
                grad_c_[(k * dim + d) * nw + w] = s.out_grad[k * dim + d];
                lap_c_[(k * dim + d) * nw + w] = s.out_lap[k * dim + d];
            }
    }
    if (singular >= 0)
        throw std::runtime_error("backflow pairing determinant: singular at the configuration "
                                 "of walker " + std::to_string(singular));
    std::span<double> b_logpsi = batch.log_psi();
    for (std::size_t w = 0; w < nw; ++w) b_logpsi[w] += log_psi_c_[w];
    std::span<double> b_grad = batch.gradients();
    for (std::size_t q = 0; q < b_grad.size(); ++q) b_grad[q] += grad_c_[q];
    std::span<double> b_lap = batch.laplacians();
    for (std::size_t q = 0; q < b_lap.size(); ++q) b_lap[q] += lap_c_[q];
}

// ── Moves ───────────────────────────────────────────────────────────────────

void BackflowPairingDeterminant::ratio_batch(WalkerSet& batch, std::size_t p,
                                             std::size_t w_begin, std::size_t w_end) {
    const std::size_t nw = batch.num_walkers(), dim = dim_, nb = blocks_.size();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end || !transform_.affects(p)) return;
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
        std::copy_n(x_.begin() + static_cast<std::ptrdiff_t>(w * n_t_ * dim), n_t_ * dim,
                    s.x.begin());
        transform_.shift(s.pos.data(), p, old_r, s.x.data());
        transform_.particle_tables(s.pos.data(), p, s.x.data(), s.tables);
        evaluate_entries(s);
        double* pinv = pinv_.data() + w * mat_size_;
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
        std::copy(s.x.begin(), s.x.end(), px_.begin() + static_cast<std::ptrdiff_t>(w * n_t_ * dim));
        double* pf = pf_.data() + w * n_t_ * dim;
        compute_f(s, pinv, pf);
        double g[SimulationCell::kMaxDim] = {}, l[SimulationCell::kMaxDim] = {};
        transform_.gradient(s.tables, pf, p, g);
        if (proposal_laplacian_) particle_laplacian(s, pinv, pf, p, l);
        for (std::size_t d = 0; d < dim; ++d) {
            pgrad_[w * dim + d] = g[d];
            plap_[w * dim + d] = l[d];
            new_grad[d * nw + w] += g[d];
            new_lap[d * nw + w] += l[d];
        }
    }
}

void BackflowPairingDeterminant::commit(WalkerSet& batch, std::size_t p, std::size_t w) {
    if (!pvalid_[w])
        throw std::logic_error("backflow pairing determinant: a singular proposal was accepted");
    const std::size_t nw = batch.num_walkers(), dim = dim_, nb = blocks_.size();
    const auto copy = [w](const std::vector<double>& from, std::vector<double>& to,
                          std::size_t width) {
        std::copy_n(from.begin() + static_cast<std::ptrdiff_t>(w * width), width,
                    to.begin() + static_cast<std::ptrdiff_t>(w * width));
    };
    copy(px_, x_, n_t_ * dim);
    copy(pinv_, inv_, mat_size_);
    copy(pf_, f_, n_t_ * dim);
    copy(plog_det_, log_det_, nb);
    copy(psign_, sign_, nb);
    double log_psi = 0.0;
    for (std::size_t b = 0; b < nb; ++b) log_psi += log_det_[w * nb + b];
    batch.log_psi()[w] += log_psi - log_psi_c_[w];
    log_psi_c_[w] = log_psi;
    for (std::size_t d = 0; d < dim; ++d) {
        grad_c_[(p * dim + d) * nw + w] = pgrad_[w * dim + d];
        lap_c_[(p * dim + d) * nw + w] = plap_[w * dim + d];
    }
}

void BackflowPairingDeterminant::apply_move_batch_deferred(WalkerSet& batch, std::size_t p,
                                                           std::span<const bool> accept,
                                                           std::size_t w_begin,
                                                           std::size_t w_end) {
    const std::size_t nw = batch.num_walkers();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end || !transform_.affects(p)) return;
    for (std::size_t w = w_begin; w < w_end; ++w)
        if (accept[w]) commit(batch, p, w);
}

void BackflowPairingDeterminant::apply_move_batch(WalkerSet& batch, std::size_t p,
                                                  std::span<const bool> accept,
                                                  std::size_t w_begin, std::size_t w_end) {
    const std::size_t nw = batch.num_walkers();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end || !transform_.affects(p)) return;
    for (std::size_t w = w_begin; w < w_end; ++w) {
        if (!accept[w]) continue;
        commit(batch, p, w);
        refresh_walker(batch, w, p);   // positions still hold the old r_p
    }
}

void BackflowPairingDeterminant::refresh_walker(WalkerSet& batch, std::size_t w,
                                                std::size_t moved) {
    const std::size_t nw = batch.num_walkers(), dim = dim_;
    Scratch& s = scratch();
    load_positions(batch, w, s);
    if (moved != npos) {
        const std::span<const double> new_pos = batch.new_pos();
        for (std::size_t d = 0; d < dim; ++d) s.pos[moved * dim + d] = new_pos[d * nw + w];
    }
    transform_.full(s.pos.data(), s.x.data(), s.tables);
    // The rebuilt x replaces the incrementally shifted one.
    std::copy(s.x.begin(), s.x.end(), x_.begin() + static_cast<std::ptrdiff_t>(w * n_t_ * dim));
    evaluate_entries(s);
    all_derivatives(s, inv_.data() + w * mat_size_, f_.data() + w * n_t_ * dim,
                    s.out_grad.data(), s.out_lap.data());
    std::span<double> bg = batch.gradients();
    std::span<double> bl = batch.laplacians();
    for (std::size_t k = 0; k < np_; ++k) {
        if (k == moved) continue;
        for (std::size_t d = 0; d < dim; ++d) {
            const std::size_t j = (k * dim + d) * nw + w;
            const double g = s.out_grad[k * dim + d], l = s.out_lap[k * dim + d];
            if (g != grad_c_[j]) bg[j] = (bg[j] - grad_c_[j]) + g;
            if (l != lap_c_[j]) bl[j] = (bl[j] - lap_c_[j]) + l;
            grad_c_[j] = g;
            lap_c_[j] = l;
        }
    }
}

void BackflowPairingDeterminant::refresh_particle_derivatives(WalkerSet& batch, std::size_t p,
                                                              std::size_t w_begin,
                                                              std::size_t w_end) {
    const std::size_t nw = batch.num_walkers(), dim = dim_;
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end || !transform_.affects(p)) return;
    std::span<double> bg = batch.gradients();
    Scratch& s = scratch();
    for (std::size_t w = w_begin; w < w_end; ++w) {   // the gradient only (the drift)
        load_positions(batch, w, s);
        transform_.particle_tables(s.pos.data(), p, s.x.data(), s.tables);
        double g[SimulationCell::kMaxDim] = {};
        transform_.gradient(s.tables, f_.data() + w * n_t_ * dim, p, g);
        for (std::size_t d = 0; d < dim; ++d) {
            const std::size_t j = (p * dim + d) * nw + w;
            if (g[d] != grad_c_[j]) bg[j] = (bg[j] - grad_c_[j]) + g[d];
            grad_c_[j] = g[d];
        }
    }
}

void BackflowPairingDeterminant::refresh_derivatives(WalkerSet& batch, std::size_t w_begin,
                                                     std::size_t w_end) {
    const std::size_t nw = batch.num_walkers();
    if (w_end > nw) w_end = nw;
    for (std::size_t w = w_begin; w < w_end; ++w) refresh_walker(batch, w, npos);
}

// ── Parameter derivatives ───────────────────────────────────────────────────

void BackflowPairingDeterminant::compute_parameter_derivatives(const WalkerSet& batch,
                                                               std::span<double> derivs) const {
    const std::size_t nw = batch.num_walkers(), dim = dim_;
    std::vector<std::size_t> column(params_.size(), npos);
    std::size_t n_opt = 0;
    for (std::size_t q = 0; q < params_.size(); ++q)
        if (!params_[q].is_fixed) column[q] = n_opt++;
    if (n_opt == 0) return;
    const std::size_t S = orbital_.n_stars(), n_poly = orbital_.n_poly_params();
    Scratch& s = scratch();
    for (std::size_t w = 0; w < nw; ++w) {
        load_positions(batch, w, s);
        const double* f = f_.data() + w * n_t_ * dim;
        transform_.parameter_derivatives(s.pos.data(), f, s.pderiv, s.param);
        for (std::size_t q = 0; q < transform_.n_parameters(); ++q)
            if (column[n_orbital_params_ + q] != npos)
                derivs[column[n_orbital_params_ + q] * nw + w] = s.pderiv[q];
        // Orbital parameters: Σ_b Σ_paired A_ca ∂φ(Δ_ac)/∂θ, Δ from the committed x.
        std::fill_n(s.param.begin(), n_orbital_params_, 0.0);
        const double* x = x_.data() + w * n_t_ * dim;
        for (std::size_t b = 0; b < blocks_.size(); ++b) {
            const Shape& sh = shape_[b];
            const std::size_t n = sh.n, off = mat_offset_[b];
            const double* A = inv_.data() + w * mat_size_ + off;
            for (std::size_t a = 0; a < sh.n_rows; ++a)
                for (std::size_t c = 0; c < sh.n_columns; ++c) {
                    const double Aca = A[c * n + a];
                    double del[SimulationCell::kMaxDim] = {};
                    for (std::size_t d = 0; d < dim; ++d)
                        del[d] = x[sh.row_t[a] * dim + d] - x[sh.col_t[c] * dim + d];
                    for (std::size_t i = 0; i < orbital_.n_k(); ++i) {
                        const double* kv = orbital_.k(i);
                        double phase = 0.0;
                        for (std::size_t d = 0; d < dim; ++d) phase += kv[d] * del[d];
                        s.param[orbital_.star(i)] += Aca * orbital_.multiplicity(i) * std::cos(phase);
                    }
                    if (n_poly == 0) continue;
                    cell_.min_image(del);
                    double r2 = 0.0;
                    for (std::size_t d = 0; d < dim; ++d) r2 += del[d] * del[d];
                    const double r = std::sqrt(r2);
                    for (std::size_t j = 0; j < n_poly; ++j) {
                        double h = 0.0, dh = 0.0, d2h = 0.0;
                        orbital_.radial_param(j, r, h, dh, d2h);
                        s.param[S + j] += Aca * h;
                    }
                }
        }
        for (std::size_t q = 0; q < n_orbital_params_; ++q)
            if (column[q] != npos) derivs[column[q] * nw + w] = s.param[q];
    }
}

}  // namespace alea
