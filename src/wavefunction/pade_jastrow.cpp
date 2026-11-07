// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "wavefunction/pade_jastrow.h"

#include "numerics/pair_radial.h"

#include <algorithm>
#include <vector>

#include "population/walker_set.h"

namespace alea {

// ---------------------------------------------------------------------------
//  PadeFunction implementation
// ---------------------------------------------------------------------------

namespace {

// Accumulate the polynomial Σ_{k=1}^{n} c_k r^k and its first two derivatives.
// r_km1/r_km2 track r^{k-1} and r^{k-2} so derivatives never divide by r;
// r_km2 is unused at k=1 (the k(k-1) factor is 0), so its placeholder value
// is irrelevant as long as it is finite.
inline void accumulate_poly(const double* c, int n, double r,
                            double& v, double& dv, double& d2v) {
    double r_km1 = 1.0;  // r^{k-1}, starts at r^0
    double r_km2 = 0.0;  // r^{k-2}, valid from k=2
    for (int i = 0; i < n; ++i) {
        const int k = i + 1;
        v   += c[i] * r_km1 * r;            // c_k r^k
        dv  += c[i] * k * r_km1;            // k c_k r^{k-1}
        d2v += c[i] * k * (k - 1) * r_km2;  // k(k-1) c_k r^{k-2}
        r_km2 = r_km1;
        r_km1 *= r;
    }
}

// Per-distance Padé kernel shared by the scalar and batched paths. Kept inline
// (and header-free of virtuals) so it folds into the vectorized loop below.
inline void eval_one(int num_order, int den_order, const double* c, double r,
                     double& u, double& du, double& d2u) {
    // P(r) = Σ_{k=1}^N a_k r^k   ;   Q(r) = 1 + Σ_{k=1}^M b_k r^k
    double p = 0.0, dp = 0.0, d2p = 0.0;
    accumulate_poly(c, num_order, r, p, dp, d2p);
    double q = 1.0, dq = 0.0, d2q = 0.0;
    accumulate_poly(c + num_order, den_order, r, q, dq, d2q);

    const double inv_q = 1.0 / q;
    u  = p * inv_q;
    du = (dp * q - p * dq) * inv_q * inv_q;
    // u'' = P''/Q - 2 P'Q'/Q² + 2 P Q'²/Q³ - P Q''/Q²
    d2u = (d2p - (2.0 * dp * dq - 2.0 * p * dq * dq * inv_q + p * d2q) * inv_q)
          * inv_q;
}

}  // namespace

PadeFunction::PadeFunction(int num_order, int den_order,
                           std::vector<double> coeffs)
    : num_order_(num_order), den_order_(den_order), coeffs_(std::move(coeffs)) {}

void PadeFunction::evaluate(double r, double& u, double& du, double& d2u) const noexcept {
    eval_one(num_order_, den_order_, coeffs_.data(), r, u, du, d2u);
}

void PadeFunction::param_derivatives(double r, std::span<double> out) const noexcept {
    // P = Σ a_k r^k,  Q = 1 + Σ b_k r^k,  u = P/Q.
    const double* __restrict c = coeffs_.data();
    double P = 0.0, Q = 1.0, rk = r;
    for (int k = 1; k <= num_order_; ++k) { P += c[k - 1] * rk; rk *= r; }
    rk = r;
    for (int k = 1; k <= den_order_; ++k) { Q += c[num_order_ + k - 1] * rk; rk *= r; }
    const double inv_q = 1.0 / Q;
    const double u = P * inv_q;
    // ∂u/∂a_k = r^k / Q ;  ∂u/∂b_k = −u · r^k / Q.
    rk = r;
    for (int k = 1; k <= num_order_; ++k) { out[k - 1] = rk * inv_q; rk *= r; }
    rk = r;
    for (int k = 1; k <= den_order_; ++k) {
        out[num_order_ + k - 1] = -u * rk * inv_q;
        rk *= r;
    }
}

void PadeFunction::param_radial_derivatives(double r, std::span<double> f1,
                                            std::span<double> f2) const noexcept {
    // With u = P/Q and B_k = r^k:
    //   ∂u/∂a_k = B_k/Q            — the u = P/Q form with P → B_k, so
    //   f' = (B'Q − BQ')/Q²,  f'' = B''/Q − 2B'Q'/Q² + 2BQ'²/Q³ − BQ''/Q²
    //   ∂u/∂b_k = −u · (B_k/Q)     — product rule on the two factors.
    const double* __restrict c = coeffs_.data();
    double p = 0.0, dp = 0.0, d2p = 0.0;
    accumulate_poly(c, num_order_, r, p, dp, d2p);
    double q = 1.0, dq = 0.0, d2q = 0.0;
    accumulate_poly(c + num_order_, den_order_, r, q, dq, d2q);
    const double iq  = 1.0 / q;
    const double u   = p * iq;
    const double du  = (dp * q - p * dq) * iq * iq;
    const double d2u = (d2p - (2.0 * dp * dq - 2.0 * p * dq * dq * iq + p * d2q) * iq) * iq;

    const int max_order = std::max(num_order_, den_order_);
    double r_km2 = 0.0, r_km1 = 1.0;   // r^{k-2}, r^{k-1}
    for (int k = 1; k <= max_order; ++k) {
        const double B   = r_km1 * r;
        const double dB  = k * r_km1;
        const double d2B = k * (k - 1) * r_km2;
        const double g   = B * iq;
        const double dg  = (dB * q - B * dq) * iq * iq;
        const double d2g = (d2B - (2.0 * dB * dq - 2.0 * B * dq * dq * iq + B * d2q) * iq) * iq;
        if (k <= num_order_) {
            f1[k - 1] = dg;
            f2[k - 1] = d2g;
        }
        if (k <= den_order_) {
            f1[num_order_ + k - 1] = -(du * g + u * dg);
            f2[num_order_ + k - 1] = -(d2u * g + 2.0 * du * dg + u * d2g);
        }
        r_km2 = r_km1;
        r_km1 *= r;
    }
}

void PadeFunction::evaluate(std::span<const double> r,
                            std::span<double> u,
                            std::span<double> du,
                            std::span<double> d2u) const noexcept {
    const int            num_order = num_order_;
    const int            den_order = den_order_;
    const double* __restrict c    = coeffs_.data();
    const double* __restrict rin  = r.data();
    double* __restrict       uo   = u.data();
    double* __restrict       duo  = du.data();
    double* __restrict       d2uo = d2u.data();
    const std::size_t n = r.size();

#pragma omp simd
    for (std::size_t w = 0; w < n; ++w) {
        eval_one(num_order, den_order, c, rin[w], uo[w], duo[w], d2uo[w]);
    }
}

// ---------------------------------------------------------------------------
//  PadeJastrow implementation
// ---------------------------------------------------------------------------

namespace {

std::vector<double> coeff_values(const std::vector<Parameter>& params) {
    std::vector<double> c;
    c.reserve(params.size());
    for (const Parameter& p : params) c.push_back(p.value);
    return c;
}

}  // namespace

PadeJastrow::PadeJastrow(int num_order, int den_order,
                         ParticleGroup group_a, ParticleGroup group_b,
                         std::vector<Parameter> params)
    : num_order_(num_order),
      den_order_(den_order),
      group_a_(std::move(group_a)),
      group_b_(std::move(group_b)),
      same_group_(group_a_ == group_b_),
      params_(std::move(params)),
      pade_(num_order, den_order, coeff_values(params_)) {}

void PadeJastrow::prepare(std::size_t nw, std::size_t np, std::size_t dim) {
    // Committed + pending caches (base), then this component's own scratch.
    // Sizing all per-walker buffers up front lets the range-restricted
    // ratio_batch/apply_move_batch run without resizing — safe to call
    // concurrently on disjoint walker ranges of one shared component.
    resize_caches(nw, np, dim);
    u_.assign(nw, 0.0);       du_.assign(nw, 0.0);
    d2u_.assign(nw, 0.0);     inv_r_.assign(nw, 0.0);
    u_old_.assign(nw, 0.0);   du_old_.assign(nw, 0.0);
    d2u_old_.assign(nw, 0.0); inv_r_old_.assign(nw, 0.0);
    mask_.assign(nw, 0.0);
    proposal_radial_.resize(6 * std::max(group_a_.last, group_b_.last) * nw);
    if (image_sum_.active()) image_sum_.prepare(nw);
}

void PadeJastrow::permute_caches(std::span<const std::size_t> src) {
    WavefunctionComponent::permute_caches(src);  // committed (+ pending)
    const std::size_t nw = log_psi_c().size();   // new walker count
    u_.assign(nw, 0.0);       du_.assign(nw, 0.0);
    d2u_.assign(nw, 0.0);     inv_r_.assign(nw, 0.0);
    u_old_.assign(nw, 0.0);   du_old_.assign(nw, 0.0);
    d2u_old_.assign(nw, 0.0); inv_r_old_.assign(nw, 0.0);
    mask_.assign(nw, 0.0);
    proposal_radial_.resize(6 * std::max(group_a_.last, group_b_.last) * nw);
    if (image_sum_.active()) image_sum_.permute(src);   // ρ(k) travels with its walker
}

void PadeJastrow::radial(std::span<const double> r, std::span<double> u,
                         std::span<double> du, std::span<double> d2u) const noexcept {
    pade_.evaluate(r, u, du, d2u);
    if (image_sum_.active()) image_sum_.to_short_range(r, u, du, d2u);
}

void PadeJastrow::sum_over_images(const SimulationCell& cell) {
    image_cell_ = cell;
    build_image_sum();
}

// The split needs u − u∞ and, for the optimiser, ∂u/∂p − ∂u∞/∂p of every
// optimisable parameter.  All of them are rational in r:
//   u          = P/Q,              P = Σ a_k r^k,  Q = 1 + Σ b_k r^k
//   ∂u/∂a_k    = r^k / Q
//   ∂u/∂b_k    = −P r^k / Q²
// so each goes through the same Padé machinery (numerics/pair_radial.h).
void PadeJastrow::build_image_sum() {
    const double rc = image_cell_->ws_radius();
    std::vector<double> a, b;
    for (int k = 0; k < num_order_; ++k) a.push_back(params_[k].value);
    for (int k = 0; k < den_order_; ++k) b.push_back(params_[num_order_ + k].value);

    // Q² − 1 = 2Σ b_j r^j + (Σ b_j r^j)², coefficients of r^1..r^{2M}.
    std::vector<double> q2(2 * b.size(), 0.0);
    for (std::size_t j = 0; j < b.size(); ++j) {
        q2[j] += 2.0 * b[j];
        for (std::size_t l = 0; l < b.size(); ++l) q2[j + l + 1] += b[j] * b[l];
    }

    std::vector<numerics::RadialFunction> dfs;
    std::vector<double> dinf;
    for (std::size_t i = 0; i < params_.size(); ++i) {
        if (params_[i].is_fixed) continue;
        std::vector<double> num, den;
        if (i < static_cast<std::size_t>(num_order_)) {
            num.assign(i + 1, 0.0);
            num[i] = 1.0;                       // r^k, k = i + 1
            den = b;
        } else {
            const std::size_t k = i - num_order_ + 1;
            num.assign(a.size() + k, 0.0);
            for (std::size_t j = 0; j < a.size(); ++j) num[j + k] = -a[j];   // −P r^k
            den = q2;
        }
        dfs.push_back(numerics::pade_radial(num, den, rc));
        dinf.push_back(numerics::pade_limit(num, den));
    }
    image_sum_ = PeriodicPairSum(*image_cell_, group_a_, group_b_,
                                 numerics::pade_radial(a, b, rc),
                                 numerics::pade_limit(a, b), dfs, dinf);
}

void PadeJastrow::evaluate_batch(WalkerSet& batch) {
    const std::size_t nw  = batch.num_walkers();
    const std::size_t np  = batch.n_particles();
    const std::size_t dim = batch.dim();
    prepare(nw, np, dim);

    const BatchedDistanceTable& dt = batch.dist_tables();
    const std::span<const double> pos = batch.positions();

    auto accumulate_pair = [&]<class BC>(std::size_t i, std::size_t j) {
        // Fold this pair's displacements into the Wigner-Seitz cell once for
        // every walker.  The fold mixes Cartesian components on a general
        // Bravais lattice, so it cannot live inside the per-dimension loop.
        if constexpr (BC::periodic)
            batch.cell().min_image_batch(pos.data() + i * dim * nw,
                                         pos.data() + j * dim * nw, nw,
                                         fold_delta_.data(), 0, nw);
        // One batched Padé call per pair (vs. one scalar call per walker).
        radial(dt.row(i, j), u_, du_, d2u_);

        const double* __restrict r    = dt.row(i, j).data();
        const double* __restrict u    = u_.data();
        const double* __restrict du   = du_.data();
        const double* __restrict d2u  = d2u_.data();
        double* __restrict       invr = inv_r_.data();
        double* __restrict       lpc  = log_psi_c_.data();

        // log psi and 1/r (reused below).
#pragma omp simd
        for (std::size_t w = 0; w < nw; ++w) {
            invr[w] = 1.0 / r[w];
            lpc[w] += u[w];
        }

        // Gradient and per-dimension laplacian:
        //   ∂_d u(r) = u'(r) Δ_d / r
        //   ∂²_d u(r) = u''(r) (Δ_d/r)² + (u'(r)/r) (1 - (Δ_d/r)²)
        for (std::size_t d = 0; d < dim; ++d) {
            [[maybe_unused]] const double* __restrict pi =
                pos.data() + (i * dim + d) * nw;
            [[maybe_unused]] const double* __restrict pj =
                pos.data() + (j * dim + d) * nw;
            [[maybe_unused]] const double* __restrict fd = fold_delta_.data() + d * nw;
            double* __restrict       gi = grad_c_.data() + (i * dim + d) * nw;
            double* __restrict       gj = grad_c_.data() + (j * dim + d) * nw;
            double* __restrict       li = lap_c_.data() + (i * dim + d) * nw;
            double* __restrict       lj = lap_c_.data() + (j * dim + d) * nw;
#pragma omp simd
            for (std::size_t w = 0; w < nw; ++w) {
                double del;
                if constexpr (BC::periodic) del = fd[w];
                else                        del = pi[w] - pj[w];
                const double dor = del * invr[w];  // Δ_d / r
                const double g   = du[w] * dor;
                gi[w] += g;
                gj[w] -= g;
                const double lap_d = d2u[w] * dor * dor
                                   + du[w] * invr[w] * (1.0 - dor * dor);
                li[w] += lap_d;
                lj[w] += lap_d;
            }
        }
    };

    // Select the boundary policy once per pair, never inside the walker loop:
    // the open instantiation is byte-identical to the pre-periodic code.
    auto pair = [&](std::size_t i, std::size_t j) {
        if (batch.cell().is_periodic())
            accumulate_pair.template operator()<PeriodicBC>(i, j);
        else
            accumulate_pair.template operator()<OpenBC>(i, j);
    };

    if (same_group_) {
        for (std::size_t i = group_a_.first; i < group_a_.last; ++i)
            for (std::size_t j = i + 1; j < group_a_.last; ++j)
                pair(i, j);
    } else {
        for (std::size_t i = group_a_.first; i < group_a_.last; ++i)
            for (std::size_t j = group_b_.first; j < group_b_.last; ++j)
                pair(i, j);
    }

    // The reciprocal-space half of the image sum.
    if (image_sum_.active()) image_sum_.evaluate(batch, log_psi_c_, grad_c_, lap_c_);

    // Accumulate this component's contribution into the batch totals.
    std::span<double> log_psi = batch.log_psi();
    for (std::size_t w = 0; w < nw; ++w) log_psi[w] += log_psi_c_[w];

    std::span<double> grad = batch.gradients();
    for (std::size_t k = 0; k < grad.size(); ++k) grad[k] += grad_c_[k];

    std::span<double> lap = batch.laplacians();
    for (std::size_t k = 0; k < lap.size(); ++k) lap[k] += lap_c_[k];
}

void PadeJastrow::ratio_batch(WalkerSet& batch, std::size_t p,
                              std::size_t w_begin, std::size_t w_end) {
    const std::size_t nw  = batch.num_walkers();   // global stride
    const std::size_t dim = batch.dim();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end) return;
    const std::size_t cnt = w_end - w_begin;
    // Scratch is normally sized by evaluate_batch/prepare before any range call
    // (so concurrent crowd calls never resize); size it here only if a caller
    // invoked ratio_batch standalone.
    if (u_.size() < nw) prepare(nw, batch.n_particles(), dim);
    const BatchedDistanceTable& dt = batch.dist_tables();   // old distances
    const std::span<const double> pend = batch.pending_dist_rows();  // new [np*nw]
    const std::span<const double> pos  = batch.positions();
    const std::span<const double> new_pos = batch.new_pos();  // proposed pos of p
    std::span<double> log_ratios = batch.log_ratios();
    std::span<double> new_grad   = batch.new_grad();  // [dim*nw]
    std::span<double> new_lap    = batch.new_lap();   // [dim*nw]

    // Scratch is pre-sized by prepare()/evaluate_batch; only zero this
    // component's pending contribution over the active range.
    for (std::size_t d = 0; d < dim; ++d) {
        double* __restrict pg0 = pending_grad_c_.data() + d * nw;
        double* __restrict pl0 = pending_lap_c_.data() + d * nw;
        for (std::size_t w = w_begin; w < w_end; ++w) { pg0[w] = 0.0; pl0[w] = 0.0; }
    }

    // For partner k of the moved particle p, at the proposed position:
    //   log_ratio += u(r'_pk) - u(r_pk)
    //   new_grad  += u'(r'_pk) (r'_p - r_k)/r'_pk
    //   new_lap   += u''(r'_pk) + (dim-1)/r'_pk u'(r'_pk)
    auto add_partner = [&]<class BC>(std::size_t k) {
        // Fold this pair's displacements into the Wigner-Seitz cell once for
        // every walker.  The fold mixes Cartesian components on a general
        // Bravais lattice, so it cannot live inside the per-dimension loop.
        if constexpr (BC::periodic)
            batch.cell().min_image_batch(new_pos.data(),
                                         pos.data() + k * dim * nw, nw,
                                         fold_delta_.data(), w_begin, w_end);
        // Batched Padé over the active range only (subspans offset by w_begin).
        radial(dt.row(p, k).subspan(w_begin, cnt),
               std::span<double>(u_old_.data() + w_begin, cnt),
               std::span<double>(du_old_.data() + w_begin, cnt),
               std::span<double>(d2u_old_.data() + w_begin, cnt));
        radial(pend.subspan(k * nw + w_begin, cnt),
               std::span<double>(u_.data() + w_begin, cnt),
               std::span<double>(du_.data() + w_begin, cnt),
               std::span<double>(d2u_.data() + w_begin, cnt));

        const double* __restrict rnew = pend.data() + k * nw;
        const double* __restrict uo   = u_old_.data();
        const double* __restrict un   = u_.data();
        const double* __restrict dun  = du_.data();
        const double* __restrict d2un = d2u_.data();
        double* __restrict       invr = inv_r_.data();
        double* __restrict       lr   = log_ratios.data();

        // 1/r and log|ψ'/ψ|.
#pragma omp simd
        for (std::size_t w = w_begin; w < w_end; ++w) {
            invr[w] = 1.0 / rnew[w];
            lr[w] += un[w] - uo[w];
            proposal_radial_[(6 * k + 0) * nw + w] = uo[w];
            proposal_radial_[(6 * k + 1) * nw + w] = du_old_[w];
            proposal_radial_[(6 * k + 2) * nw + w] = d2u_old_[w];
            proposal_radial_[(6 * k + 3) * nw + w] = un[w];
            proposal_radial_[(6 * k + 4) * nw + w] = dun[w];
            proposal_radial_[(6 * k + 5) * nw + w] = d2un[w];
        }

        // Gradient and per-dimension laplacian at the proposed position.
        for (std::size_t d = 0; d < dim; ++d) {
            [[maybe_unused]] const double* __restrict np = new_pos.data() + d * nw;
            [[maybe_unused]] const double* __restrict pk =
                pos.data() + (k * dim + d) * nw;
            [[maybe_unused]] const double* __restrict fd = fold_delta_.data() + d * nw;
            double* __restrict       ng = pending_grad_c_.data() + d * nw;
            double* __restrict       nl = pending_lap_c_.data() + d * nw;
#pragma omp simd
            for (std::size_t w = w_begin; w < w_end; ++w) {
                double del;
                if constexpr (BC::periodic) del = fd[w];
                else                        del = np[w] - pk[w];
                const double dor = del * invr[w];  // Δ_d / r
                ng[w] += dun[w] * dor;
                nl[w] += d2un[w] * dor * dor
                       + dun[w] * invr[w] * (1.0 - dor * dor);
            }
        }
    };

    // Boundary policy chosen once per partner, above the walker loop.
    auto partner = [&](std::size_t k) {
        if (batch.cell().is_periodic())
            add_partner.template operator()<PeriodicBC>(k);
        else
            add_partner.template operator()<OpenBC>(k);
    };

    if (same_group_) {
        if (group_a_.contains(p))
            for (std::size_t k = group_a_.first; k < group_a_.last; ++k)
                if (k != p) partner(k);
    } else if (group_a_.contains(p)) {
        for (std::size_t k = group_b_.first; k < group_b_.last; ++k)
            partner(k);
    } else if (group_b_.contains(p)) {
        for (std::size_t k = group_a_.first; k < group_a_.last; ++k)
            partner(k);
    }

    // Reciprocal-space half: the change of Σ w T and p's k-space gradient.
    if (image_sum_.active())
        image_sum_.ratio(batch, p, w_begin, w_end, log_ratios, pending_grad_c_,
                         pending_lap_c_);

    // Add this component's contribution into the (summed) batch totals (range).
    for (std::size_t d = 0; d < dim; ++d) {
        double* __restrict ng_b = new_grad.data() + d * nw;
        double* __restrict nl_b = new_lap.data() + d * nw;
        const double* __restrict pg = pending_grad_c_.data() + d * nw;
        const double* __restrict pl = pending_lap_c_.data() + d * nw;
#pragma omp simd
        for (std::size_t w = w_begin; w < w_end; ++w) {
            ng_b[w] += pg[w];
            nl_b[w] += pl[w];
        }
    }
}

void PadeJastrow::apply_move_batch(WalkerSet& batch, std::size_t p,
                                   std::span<const bool> accept,
                                   std::size_t w_begin, std::size_t w_end) {
    const std::size_t nw  = batch.num_walkers();   // global stride
    const std::size_t dim = batch.dim();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end) return;
    if (mask_.size() < nw) prepare(nw, batch.n_particles(), dim);
    const BatchedDistanceTable& dt = batch.dist_tables();   // OLD distances
    const std::span<const double> pend = batch.pending_dist_rows();  // new [np*nw]
    const std::span<const double> pos      = batch.positions();   // OLD positions
    const std::span<const double> new_pos  = batch.new_pos();     // proposed pos of p

    std::span<double> b_logpsi = batch.log_psi();
    std::span<double> b_grad   = batch.gradients();
    std::span<double> b_lap    = batch.laplacians();

    // Scratch is pre-sized by prepare(); just materialise the accept mask over
    // the active range.
    for (std::size_t w = w_begin; w < w_end; ++w) mask_[w] = accept[w] ? 1.0 : 0.0;

    // Fold the pair (p,k) change into walker w's caches and batch totals.
    // Rejected walkers are masked out (mask_=0) rather than branched, to keep
    // the loops vectorizable. Partner k shifts incrementally; particle p's
    // gradient/laplacian are overwritten with the absolute new values below.
    auto update_partner = [&]<class BC>(std::size_t k) {
        // Fold this pair's displacements into the Wigner-Seitz cell once for
        // every walker.  The fold mixes Cartesian components on a general
        // Bravais lattice, so it cannot live inside the per-dimension loop.
        if constexpr (BC::periodic) {
            batch.cell().min_image_batch(pos.data() + k * dim * nw,
                                         pos.data() + p * dim * nw, nw,
                                         fold_delta_old_.data(), w_begin, w_end);
            batch.cell().min_image_batch(pos.data() + k * dim * nw,
                                         new_pos.data(), nw,
                                         fold_delta_.data(), w_begin, w_end);
        }
        const double* __restrict rold = dt.row(p, k).data();
        const double* __restrict rnew = pend.data() + k * nw;
        const double* __restrict uo   = proposal_radial_.data() + (6 * k + 0) * nw;
        const double* __restrict duo  = proposal_radial_.data() + (6 * k + 1) * nw;
        const double* __restrict d2uo = proposal_radial_.data() + (6 * k + 2) * nw;
        const double* __restrict un   = proposal_radial_.data() + (6 * k + 3) * nw;
        const double* __restrict dun  = proposal_radial_.data() + (6 * k + 4) * nw;
        const double* __restrict d2un = proposal_radial_.data() + (6 * k + 5) * nw;
        const double* __restrict m    = mask_.data();
        double* __restrict iro = inv_r_old_.data();
        double* __restrict irn = inv_r_.data();
        double* __restrict lpc = log_psi_c_.data();
        double* __restrict blp = b_logpsi.data();

        // log ψ and 1/r.
#pragma omp simd
        for (std::size_t w = w_begin; w < w_end; ++w) {
            iro[w] = 1.0 / rold[w];
            irn[w] = 1.0 / rnew[w];
            const double dlog = m[w] * (un[w] - uo[w]);
            lpc[w] += dlog;
            blp[w] += dlog;
        }

        // Partner k gradient and per-dimension laplacian.
        for (std::size_t d = 0; d < dim; ++d) {
            [[maybe_unused]] const double* __restrict xk =
                pos.data() + (k * dim + d) * nw;
            [[maybe_unused]] const double* __restrict xpo =
                pos.data() + (p * dim + d) * nw;
            [[maybe_unused]] const double* __restrict xpn = new_pos.data() + d * nw;
            [[maybe_unused]] const double* __restrict fdo =
                fold_delta_old_.data() + d * nw;
            [[maybe_unused]] const double* __restrict fdn = fold_delta_.data() + d * nw;
            double* __restrict gck = grad_c_.data() + (k * dim + d) * nw;
            double* __restrict bgk = b_grad.data() + (k * dim + d) * nw;
            double* __restrict lck = lap_c_.data() + (k * dim + d) * nw;
            double* __restrict blk = b_lap.data() + (k * dim + d) * nw;
#pragma omp simd
            for (std::size_t w = w_begin; w < w_end; ++w) {
                double del_old, del_new;
                if constexpr (BC::periodic) {
                    del_old = fdo[w];
                    del_new = fdn[w];
                } else {
                    del_old = xk[w] - xpo[w];
                    del_new = xk[w] - xpn[w];
                }
                const double dor_old = del_old * iro[w];
                const double dor_new = del_new * irn[w];
                const double g_old = duo[w] * dor_old;
                const double g_new = dun[w] * dor_new;
                const double dg = m[w] * (g_new - g_old);
                gck[w] += dg;
                bgk[w] += dg;

                const double lap_old = d2uo[w] * dor_old * dor_old
                                     + duo[w] * iro[w] * (1.0 - dor_old * dor_old);
                const double lap_new = d2un[w] * dor_new * dor_new
                                     + dun[w] * irn[w] * (1.0 - dor_new * dor_new);
                const double dlap = m[w] * (lap_new - lap_old);
                lck[w] += dlap;
                blk[w] += dlap;
            }
        }
    };

    // Boundary policy chosen once per partner, above the walker loop.
    auto partner = [&](std::size_t k) {
        if (batch.cell().is_periodic())
            update_partner.template operator()<PeriodicBC>(k);
        else
            update_partner.template operator()<OpenBC>(k);
    };

    if (same_group_) {
        if (group_a_.contains(p))
            for (std::size_t k = group_a_.first; k < group_a_.last; ++k)
                if (k != p) partner(k);
    } else if (group_a_.contains(p)) {
        for (std::size_t k = group_b_.first; k < group_b_.last; ++k)
            partner(k);
    } else if (group_b_.contains(p)) {
        for (std::size_t k = group_a_.first; k < group_a_.last; ++k)
            partner(k);
    } else {
        return;  // p is not in this component's groups: nothing changes.
    }

    // Reciprocal-space half: ρ(k) moves with p, so every particle that sees it
    // shifts; p itself is overwritten from the pending values just below.
    if (image_sum_.active()) {
        const std::span<double> lp[] = {log_psi_c_, b_logpsi};
        const std::span<double> gr[] = {grad_c_, b_grad};
        const std::span<double> lc[] = {lap_c_, b_lap};
        image_sum_.accept(batch, p, mask_, w_begin, w_end, lp, gr, lc);
    }

    // Particle p: overwrite with the absolute new grad/lap (accepted only),
    // pushing the delta to the batch totals.
    {
        const double* __restrict m = mask_.data();
        for (std::size_t d = 0; d < dim; ++d) {
            const double* __restrict ng  = pending_grad_c_.data() + d * nw;
            const double* __restrict nl  = pending_lap_c_.data() + d * nw;
            double* __restrict gcp = grad_c_.data() + (p * dim + d) * nw;
            double* __restrict bgp = b_grad.data() + (p * dim + d) * nw;
            double* __restrict lcp = lap_c_.data() + (p * dim + d) * nw;
            double* __restrict blp = b_lap.data() + (p * dim + d) * nw;
#pragma omp simd
            for (std::size_t w = w_begin; w < w_end; ++w) {
                const double dg_p   = m[w] * (ng[w] - gcp[w]);
                const double dlap_p = m[w] * (nl[w] - lcp[w]);
                gcp[w] += dg_p;
                bgp[w] += dg_p;
                lcp[w] += dlap_p;
                blp[w] += dlap_p;
            }
        }
    }
}

void PadeJastrow::rebuild() {
    pade_ = PadeFunction(num_order_, den_order_, coeff_values(params_));
    if (image_cell_) build_image_sum();   // the split follows the new u
}

void PadeJastrow::compute_parameter_derivatives(const WalkerSet& batch,
                                                std::span<double> derivs) const {
    const std::size_t nw  = batch.num_walkers();
    const std::size_t n_opt = num_optimizable_params();
    if (n_opt == 0) return;

    // Zero output.
    std::fill(derivs.begin(), derivs.end(), 0.0);

    // Summed over images, ln Ψ = Σ u_short + Σ_k w_k T_k, so ∂/∂p takes the
    // short-range part of ∂u/∂p and the k-space weights ∂w_k/∂p.  Optimiser
    // path, not the sampling hot loop: scalar per walker.
    if (image_sum_.active()) {
        const BatchedDistanceTable& dt = batch.dist_tables();
        std::vector<double> dp(params_.size());
        auto pair = [&](std::size_t i, std::size_t j) {
            const double* r = dt.row(i, j).data();
            for (std::size_t w = 0; w < nw; ++w) {
                pade_.param_derivatives(r[w], dp);
                std::size_t s = 0;
                for (std::size_t q = 0; q < params_.size(); ++q) {
                    if (params_[q].is_fixed) continue;
                    double v = dp[q], d1 = 0.0, d2 = 0.0;
                    image_sum_.param_to_short_range(s, r[w], v, d1, d2);
                    derivs[s * nw + w] += v;
                    ++s;
                }
            }
        };
        if (same_group_) {
            for (std::size_t i = group_a_.first; i < group_a_.last; ++i)
                for (std::size_t j = i + 1; j < group_a_.last; ++j) pair(i, j);
        } else {
            for (std::size_t i = group_a_.first; i < group_a_.last; ++i)
                for (std::size_t j = group_b_.first; j < group_b_.last; ++j) pair(i, j);
        }
        for (std::size_t s = 0; s < n_opt; ++s)
            image_sum_.param_log_psi(s, nw, derivs.subspan(s * nw, nw));
        return;
    }

    // Build a mapping: non_fixed_idx → {param_idx, is_numerator}.
    struct ParamInfo {
        std::size_t idx;       // index into params_
        bool        is_num;    // true = numerator (a_k), false = denominator (b_k)
    };
    std::vector<ParamInfo> opt_map;
    opt_map.reserve(n_opt);
    for (std::size_t i = 0; i < params_.size(); ++i) {
        if (!params_[i].is_fixed)
            opt_map.push_back({i, i < static_cast<std::size_t>(num_order_)});
    }

    const std::size_t max_order = static_cast<std::size_t>(
        std::max(num_order_, den_order_));

    // Scratch for per-walker r^k powers and Q.
    // r_pow[k * nw + w] = r(w)^k  for k = 0..max_order.
    std::vector<double> r_pow((max_order + 1) * nw);
    std::vector<double> q_inv(nw);   // 1/Q(r) per walker
    std::vector<double> u_val(nw);   // u(r) per walker

    const BatchedDistanceTable& dt = batch.dist_tables();

    // Helper: process one pair (i, j).
    auto accumulate_pair = [&](std::size_t i, std::size_t j) {
        const double* __restrict r = dt.row(i, j).data();

        // Precompute r^k for k = 0..max_order.
        {
            double* __restrict p0 = r_pow.data();  // k = 0
            double* __restrict p1 = r_pow.data() + nw;  // k = 1
#pragma omp simd
            for (std::size_t w = 0; w < nw; ++w) {
                p0[w] = 1.0;
                p1[w] = r[w];
            }
            for (std::size_t k = 2; k <= max_order; ++k) {
                double* __restrict pk  = r_pow.data() + k * nw;
                double* __restrict pkm = r_pow.data() + (k - 1) * nw;
#pragma omp simd
                for (std::size_t w = 0; w < nw; ++w)
                    pk[w] = pkm[w] * r[w];
            }
        }

        // Compute Q(r) and u(r).
        double* __restrict qinv = q_inv.data();
        double* __restrict uval = u_val.data();
        {
            // Zero u and start Q at 1.0.
#pragma omp simd
            for (std::size_t w = 0; w < nw; ++w) {
                uval[w] = 0.0;
                qinv[w] = 1.0;  // will become Q(w)
            }

            // Numerator: Σ a_k r^k
            for (int k = 1; k <= num_order_; ++k) {
                const double ak = params_[k - 1].value;
                const double* __restrict pk = r_pow.data() + k * nw;
#pragma omp simd
                for (std::size_t w = 0; w < nw; ++w)
                    uval[w] += ak * pk[w];
            }

            // Denominator: Q = 1 + Σ b_k r^k
            for (int k = 1; k <= den_order_; ++k) {
                const double bk = params_[num_order_ + k - 1].value;
                const double* __restrict pk = r_pow.data() + k * nw;
#pragma omp simd
                for (std::size_t w = 0; w < nw; ++w)
                    qinv[w] += bk * pk[w];
            }

            // Now qinv holds Q. Compute u = P/Q and qinv = 1/Q.
#pragma omp simd
            for (std::size_t w = 0; w < nw; ++w) {
                const double inv_q = 1.0 / qinv[w];
                uval[w] *= inv_q;
                qinv[w] = inv_q;
            }
        }

        // Accumulate derivatives.
        for (std::size_t opt_k = 0; opt_k < n_opt; ++opt_k) {
            const ParamInfo& info = opt_map[opt_k];
            // The power index depends on whether this is a numerator or
            // denominator parameter. params_ layout:
            //   [a_1..a_N, b_1..b_M]
            const int k = info.is_num
                ? static_cast<int>(info.idx) + 1                       // a_k
                : static_cast<int>(info.idx) - num_order_ + 1;         // b_k
            const double* __restrict pk = r_pow.data() + k * nw;
            double* __restrict       dk = derivs.data() + opt_k * nw;

            if (info.is_num) {
                // ∂ log ψ / ∂a_k = Σ_pairs r^k / Q
#pragma omp simd
                for (std::size_t w = 0; w < nw; ++w)
                    dk[w] += pk[w] * qinv[w];
            } else {
                // ∂ log ψ / ∂b_k = Σ_pairs -u · r^k / Q
#pragma omp simd
                for (std::size_t w = 0; w < nw; ++w)
                    dk[w] -= uval[w] * pk[w] * qinv[w];
            }
        }
    };

    if (same_group_) {
        for (std::size_t i = group_a_.first; i < group_a_.last; ++i)
            for (std::size_t j = i + 1; j < group_a_.last; ++j)
                accumulate_pair(i, j);
    } else {
        for (std::size_t i = group_a_.first; i < group_a_.last; ++i)
            for (std::size_t j = group_b_.first; j < group_b_.last; ++j)
                accumulate_pair(i, j);
    }
}

// ∂(∂_d log ψ)/∂p_k and ∂(∂²_d log ψ)/∂p_k.  Each f_k(r) = ∂u/∂p_k is a radial
// pair function, so it enters exactly as u does in evaluate_batch:
//   ∂_d f_k  = f_k' Δ_d/r                                  (− for partner j)
//   ∂²_d f_k = f_k'' (Δ_d/r)² + (f_k'/r)(1 − (Δ_d/r)²)     (same for i and j)
bool PadeJastrow::compute_parameter_gradient_derivatives(
    const WalkerSet& batch, std::span<double> dgrad, std::span<double> dlap) const {
    const std::size_t n_opt = num_optimizable_params();
    if (n_opt == 0) return true;
    const std::size_t nw     = batch.num_walkers();
    const std::size_t np     = batch.n_particles();
    const std::size_t dim    = batch.dim();
    const std::size_t stride = np * dim * nw;
    std::fill(dgrad.begin(), dgrad.end(), 0.0);
    std::fill(dlap.begin(), dlap.end(), 0.0);

    // slot[k] = output column of the k-th non-fixed parameter (params_ order).
    std::vector<std::size_t> live;
    live.reserve(n_opt);
    for (std::size_t i = 0; i < params_.size(); ++i)
        if (!params_[i].is_fixed) live.push_back(i);

    const BatchedDistanceTable& dt  = batch.dist_tables();
    const double*               pos = batch.positions().data();
    const bool periodic = batch.cell().is_periodic();
    std::vector<double> delta(dim * nw);
    std::vector<double> f1(params_.size()), f2(params_.size());

    auto accumulate_pair = [&](std::size_t i, std::size_t j) {
        const double* xi = pos + i * dim * nw;
        const double* xj = pos + j * dim * nw;
        if (periodic) {
            batch.cell().min_image_batch(xi, xj, nw, delta.data(), 0, nw);
        } else {
            for (std::size_t k = 0; k < dim * nw; ++k) delta[k] = xi[k] - xj[k];
        }
        const double* r = dt.row(i, j).data();

        for (std::size_t w = 0; w < nw; ++w) {
            const double rr = r[w];
            pade_.param_radial_derivatives(rr, f1, f2);
            if (image_sum_.active()) {   // short-range part of each ∂u/∂p
                for (std::size_t s = 0; s < n_opt; ++s) {
                    double v = 0.0;
                    image_sum_.param_to_short_range(s, rr, v, f1[live[s]], f2[live[s]]);
                }
            }
            const double inv_r = 1.0 / rr;
            for (std::size_t d = 0; d < dim; ++d) {
                const double dor  = delta[d * nw + w] * inv_r;
                const double dor2 = dor * dor;
                const std::size_t gi = (i * dim + d) * nw + w;
                const std::size_t gj = (j * dim + d) * nw + w;
                for (std::size_t s = 0; s < n_opt; ++s) {
                    const std::size_t k = live[s];
                    const double gd = f1[k] * dor;
                    const double ld = f2[k] * dor2 + f1[k] * inv_r * (1.0 - dor2);
                    const std::size_t o = s * stride;
                    dgrad[o + gi] += gd;
                    dgrad[o + gj] -= gd;
                    dlap[o + gi]  += ld;
                    dlap[o + gj]  += ld;
                }
            }
        }
    };

    if (same_group_) {
        for (std::size_t i = group_a_.first; i < group_a_.last; ++i)
            for (std::size_t j = i + 1; j < group_a_.last; ++j)
                accumulate_pair(i, j);
    } else {
        for (std::size_t i = group_a_.first; i < group_a_.last; ++i)
            for (std::size_t j = group_b_.first; j < group_b_.last; ++j)
                accumulate_pair(i, j);
    }
    if (image_sum_.active()) {
        for (std::size_t s = 0; s < n_opt; ++s)
            image_sum_.param_grad_lap(batch, s, dgrad.subspan(s * stride, stride),
                                      dlap.subspan(s * stride, stride));
    }
    return true;
}

}  // namespace alea
