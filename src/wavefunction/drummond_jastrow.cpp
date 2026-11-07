// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "wavefunction/drummond_jastrow.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "population/walker_set.h"
#include "wavefunction/radial_pair_jastrow_impl.h"

namespace alea {

// ---------------------------------------------------------------------------
//  DrummondFunction implementation
// ---------------------------------------------------------------------------

namespace {

// Integer power b^e for e ≥ 0 by repeated multiplication.  Used for the small
// truncation exponents (C, C−1, C−2 with C ≥ 2), so the trip count is tiny and
// the call folds into the vectorized loops below.  Unlike std::pow this is exact
// for a negative base (r−L < 0 inside the cutoff) with an integer exponent.
inline double ipow(double b, int e) {
    double r = 1.0;
    for (int i = 0; i < e; ++i) r *= b;
    return r;
}

// Per-distance DTN kernel shared by the scalar and batched paths (Eq. 19,
// u-term only).  For r < L:
//   u(r) = (r−L)^C · [ α0 + α1 r + Σ_{k=1..n−1} coeffs[k] r^{k+1} ]
// and u ≡ u' ≡ u'' ≡ 0 for r ≥ L (the Θ(L−r) cutoff, applied branchlessly via a
// 0/1 mask so the loop stays vectorizable).  Requires n ≥ 1 and C ≥ 2.
inline void eval_one(int C, double alpha1, double L, const double* coeffs,
                     int n, double r, double& u, double& du, double& d2u) {
    const double t    = r - L;            // < 0 inside the cutoff
    const double tCm2 = ipow(t, C - 2);   // (r−L)^{C−2}
    const double tCm1 = tCm2 * t;         // (r−L)^{C−1}
    const double tC   = tCm1 * t;         // (r−L)^C
    const double g    = tC;
    const double dg   = static_cast<double>(C) * tCm1;
    const double d2g  = static_cast<double>(C) * (C - 1) * tCm2;

    // Polynomial P(r) = α0 + α1 r + Σ_{l=2..N} αl r^l and its two derivatives.
    // r_lm2/r_lm1/r_l track r^{l−2}, r^{l−1}, r^l so no division by r is needed.
    double P     = coeffs[0] + alpha1 * r;
    double dP    = alpha1;
    double d2P   = 0.0;
    double r_lm2 = 1.0;     // r^{l−2}, l starts at 2
    double r_lm1 = r;       // r^{l−1}
    double r_l   = r * r;   // r^l
    for (int k = 1; k < n; ++k) {
        const int    l = k + 1;
        const double c = coeffs[k];
        P   += c * r_l;
        dP  += c * l * r_lm1;
        d2P += c * l * (l - 1) * r_lm2;
        r_lm2 = r_lm1;
        r_lm1 = r_l;
        r_l  *= r;
    }

    const double m = (r < L) ? 1.0 : 0.0;  // Θ(L − r)
    u   = m * (g * P);
    du  = m * (dg * P + g * dP);
    d2u = m * (d2g * P + 2.0 * dg * dP + g * d2P);
}

}  // namespace

DrummondFunction::DrummondFunction(int C, double gamma, double L,
                                   std::vector<double> coeffs)
    : C_(C), gamma_(gamma), L_(L), coeffs_(std::move(coeffs)) {
    // α1 = Γ/(−L)^C + α0·C/L (cusp-derived); coeffs_[0] = α0.
    const double a0 = coeffs_.empty() ? 0.0 : coeffs_[0];
    alpha1_ = gamma_ / ipow(-L_, C_) + a0 * static_cast<double>(C_) / L_;
}

void DrummondFunction::evaluate(double r, double& u, double& du,
                                double& d2u) const noexcept {
    eval_one(C_, alpha1_, L_, coeffs_.data(),
             static_cast<int>(coeffs_.size()), r, u, du, d2u);
}

void DrummondFunction::evaluate(std::span<const double> r,
                                std::span<double> u,
                                std::span<double> du,
                                std::span<double> d2u) const noexcept {
    const int            C   = C_;
    const double         a1  = alpha1_;
    const double         L   = L_;
    const int            n   = static_cast<int>(coeffs_.size());
    const double* __restrict c    = coeffs_.data();
    const double* __restrict rin  = r.data();
    double* __restrict       uo   = u.data();
    double* __restrict       duo  = du.data();
    double* __restrict       d2uo = d2u.data();
    const std::size_t n_w = r.size();

#pragma omp simd
    for (std::size_t w = 0; w < n_w; ++w) {
        eval_one(C, a1, L, c, n, rin[w], uo[w], duo[w], d2uo[w]);
    }
}

void DrummondFunction::param_derivatives(double r,
                                          std::span<double> out) const noexcept {
    const std::size_t n_alpha = coeffs_.size();
    const std::size_t n_total = n_alpha + 1;
    if (r >= L_) {
        for (std::size_t i = 0; i < n_total; ++i) out[i] = 0.0;
        return;
    }

    const double t    = r - L_;
    const double tCm2 = ipow(t, C_ - 2);
    const double tCm1 = tCm2 * t;
    const double tC   = tCm1 * t;

    const double a0 = n_alpha > 0 ? coeffs_[0] : 0.0;
    double P = a0 + alpha1_ * r;
    for (std::size_t k = 1; k < n_alpha; ++k) {
        const int l = static_cast<int>(k) + 1;
        P += coeffs_[k] * ipow(r, l);
    }

    const double dalpha1_dL =
        gamma_ * static_cast<double>(C_) / ipow(-L_, C_ + 1)
        - a0 * static_cast<double>(C_) / (L_ * L_);

    if (n_alpha > 0) {
        const double c_over_L = static_cast<double>(C_) / L_;
        out[0] = tC * (1.0 + c_over_L * r);
    }

    for (std::size_t k = 1; k < n_alpha; ++k) {
        const int l = static_cast<int>(k) + 1;
        out[k] = tC * ipow(r, l);
    }

    out[n_alpha] =
        -static_cast<double>(C_) * tCm1 * P + tC * dalpha1_dL * r;
}

// ---------------------------------------------------------------------------
//  DrummondJastrow implementation
// ---------------------------------------------------------------------------

// The pair-loop machinery for this u, in this translation unit so that
// DrummondFunction::evaluate inlines into it.
template class RadialPairJastrow<DrummondFunction>;

namespace {

// params layout is [α0, α2..αN, L]: the leading entries are the polynomial
// coefficients and the final entry is the cutoff length.
std::vector<double> alpha_values(const std::vector<Parameter>& params) {
    std::vector<double> c;
    if (params.empty()) return c;
    c.reserve(params.size() - 1);
    for (std::size_t i = 0; i + 1 < params.size(); ++i)
        c.push_back(params[i].value);
    return c;
}

double cutoff_value(const std::vector<Parameter>& params) {
    return params.empty() ? 0.0 : params.back().value;
}

}  // namespace

DrummondJastrow::DrummondJastrow(int C, double gamma,
                                 ParticleGroup group_a, ParticleGroup group_b,
                                 std::vector<Parameter> params)
    : DrummondJastrow(C, gamma,
                      std::vector<JastrowGroupPair>{{std::move(group_a), std::move(group_b)}},
                      std::move(params)) {}

DrummondJastrow::DrummondJastrow(int C, double gamma, std::vector<JastrowGroupPair> pairs,
                                 std::vector<Parameter> params)
    : RadialPairJastrow(std::move(pairs),
                        DrummondFunction(C, gamma, cutoff_value(params), alpha_values(params))),
      C_(C),
      gamma_(gamma),
      params_(std::move(params)) {}

void DrummondJastrow::rebuild() {
    const double L = cutoff_value(params_);
    if (L > max_cutoff_)
        throw std::runtime_error(
            "drummond jastrow: cutoff L = " + std::to_string(L) +
            " exceeds the Wigner-Seitz inradius " + std::to_string(max_cutoff_) +
            "; beyond it the minimum image no longer sums every periodic image.");
    radial_ = DrummondFunction(C_, gamma_, L, alpha_values(params_));
}

void DrummondJastrow::compute_parameter_derivatives(const WalkerSet& batch,
                                                    std::span<double> derivs) const {
    const std::size_t nw    = batch.num_walkers();
    const std::size_t n_opt = num_optimizable_params();
    if (n_opt == 0) return;
    std::fill(derivs.begin(), derivs.end(), 0.0);

    const std::size_t n_alpha = params_.size() - 1;  // N: number of α coeffs
    const std::size_t L_idx   = params_.size() - 1;  // index of L in params_
    const int         C       = C_;
    const double      L       = params_[L_idx].value;
    const double      a0      = params_[0].value;

    // α1 and ∂α1/∂L are constant across pairs and walkers.
    const double alpha1     = gamma_ / ipow(-L, C) + a0 * static_cast<double>(C) / L;
    const double dalpha1_dL = gamma_ * static_cast<double>(C) / ipow(-L, C + 1)
                              - a0 * static_cast<double>(C) / (L * L);

    // Map each non-fixed parameter to its output column and kind.
    //   kind 0 → α0,  kind 1 → αl (power = l),  kind 2 → L.
    struct ParamInfo { std::size_t col; int kind; int power; };
    std::vector<ParamInfo> opt_map;
    opt_map.reserve(n_opt);
    {
        std::size_t col = 0;
        for (std::size_t i = 0; i < params_.size(); ++i) {
            if (params_[i].is_fixed) continue;
            ParamInfo info{col++, 0, 0};
            if (i == L_idx)       { info.kind = 2; info.power = 0; }
            else if (i == 0)      { info.kind = 0; info.power = 0; }
            else                  { info.kind = 1; info.power = static_cast<int>(i) + 1; }
            opt_map.push_back(info);
        }
    }

    const std::size_t maxN = n_alpha;  // highest power present is N
    std::vector<double> r_pow((maxN + 1) * nw);  // r_pow[k*nw + w] = r(w)^k
    std::vector<double> gC(nw), gCm1(nw), Pval(nw), cut(nw);

    const BatchedDistanceTable& dt = batch.dist_tables();

    auto accumulate_pair = [&](std::size_t i, std::size_t j) {
        const double* __restrict r = dt.row(i, j).data();

        // Precompute r^k for k = 0..N.
        {
            double* __restrict p0 = r_pow.data();
            double* __restrict p1 = r_pow.data() + nw;
#pragma omp simd
            for (std::size_t w = 0; w < nw; ++w) { p0[w] = 1.0; p1[w] = r[w]; }
            for (std::size_t k = 2; k <= maxN; ++k) {
                double* __restrict pk  = r_pow.data() + k * nw;
                double* __restrict pkm = r_pow.data() + (k - 1) * nw;
#pragma omp simd
                for (std::size_t w = 0; w < nw; ++w) pk[w] = pkm[w] * r[w];
            }
        }

        // Prefactors (r−L)^C, (r−L)^{C−1}, the cutoff mask, and P(r).
        double* __restrict gc   = gC.data();
        double* __restrict gcm1 = gCm1.data();
        double* __restrict pv   = Pval.data();
        double* __restrict cm   = cut.data();
#pragma omp simd
        for (std::size_t w = 0; w < nw; ++w) {
            const double rr   = r[w];
            const double t    = rr - L;
            const double tCm2 = ipow(t, C - 2);
            const double tCm1 = tCm2 * t;
            cm[w]   = (rr < L) ? 1.0 : 0.0;
            gcm1[w] = tCm1;
            gc[w]   = tCm1 * t;
            pv[w]   = a0 + alpha1 * rr;
        }
        for (std::size_t k = 1; k < n_alpha; ++k) {
            const int    l  = static_cast<int>(k) + 1;
            const double al = params_[k].value;
            const double* __restrict rl = r_pow.data() + static_cast<std::size_t>(l) * nw;
#pragma omp simd
            for (std::size_t w = 0; w < nw; ++w) pv[w] += al * rl[w];
        }

        // Accumulate the derivative of every optimisable parameter.
        for (const ParamInfo& info : opt_map) {
            double* __restrict dk = derivs.data() + info.col * nw;
            if (info.kind == 0) {
                // ∂u/∂α0 = (r−L)^C [ 1 + (C/L) r ]
                const double c_over_L = static_cast<double>(C) / L;
#pragma omp simd
                for (std::size_t w = 0; w < nw; ++w)
                    dk[w] += cm[w] * gc[w] * (1.0 + c_over_L * r[w]);
            } else if (info.kind == 1) {
                // ∂u/∂αl = (r−L)^C r^l
                const double* __restrict rl =
                    r_pow.data() + static_cast<std::size_t>(info.power) * nw;
#pragma omp simd
                for (std::size_t w = 0; w < nw; ++w)
                    dk[w] += cm[w] * gc[w] * rl[w];
            } else {
                // ∂u/∂L = −C(r−L)^{C−1} P + (r−L)^C (∂α1/∂L) r
#pragma omp simd
                for (std::size_t w = 0; w < nw; ++w)
                    dk[w] += cm[w] * (-static_cast<double>(C) * gcm1[w] * pv[w]
                                      + gc[w] * dalpha1_dL * r[w]);
            }
        }
    };

    for_each_pair(accumulate_pair);
}

// ∂(∂_d log ψ)/∂p_k and ∂(∂²_d log ψ)/∂p_k.  Each parameter derivative
// f_k(r) = ∂u/∂p_k is itself a radial pair function, so it enters the gradient
// and per-dimension laplacian exactly as u does in evaluate_batch:
//   ∂_d f_k   = f_k' Δ_d/r                                  (−  for partner j)
//   ∂²_d f_k  = f_k'' (Δ_d/r)² + (f_k'/r)(1 − (Δ_d/r)²)     (same for i and j)
// With t = r−L, g = t^C (g' = C t^{C−1}, g'' = C(C−1) t^{C−2}):
//   α0 :  f = g (1 + (C/L) r)
//   αl :  f = g r^l
//   L  :  f = h P + g a r,   h = −C t^{C−1},  a = ∂α1/∂L
// and the product rule gives f', f''.  All vanish for r ≥ L.
bool DrummondJastrow::compute_parameter_gradient_derivatives(
    const WalkerSet& batch, std::span<double> dgrad, std::span<double> dlap) const {
    const std::size_t n_opt = num_optimizable_params();
    if (n_opt == 0) return true;
    const std::size_t nw     = batch.num_walkers();
    const std::size_t np     = batch.n_particles();
    const std::size_t dim    = batch.dim();
    const std::size_t stride = np * dim * nw;
    std::fill(dgrad.begin(), dgrad.end(), 0.0);
    std::fill(dlap.begin(), dlap.end(), 0.0);

    const std::size_t L_idx = params_.size() - 1;
    const std::size_t n_alpha = L_idx;
    const int    C  = C_;
    const double Cd = static_cast<double>(C);
    const double L  = params_[L_idx].value;
    const double a0 = params_[0].value;
    const double alpha1     = gamma_ / ipow(-L, C) + a0 * Cd / L;
    const double dalpha1_dL = gamma_ * Cd / ipow(-L, C + 1) - a0 * Cd / (L * L);

    // kind 0 → α0, 1 → αl (power l), 2 → L; col = output slot.
    struct ParamInfo { std::size_t col; int kind; int power; };
    std::vector<ParamInfo> opt_map;
    opt_map.reserve(n_opt);
    for (std::size_t i = 0, col = 0; i < params_.size(); ++i) {
        if (params_[i].is_fixed) continue;
        if (i == L_idx)  opt_map.push_back({col++, 2, 0});
        else if (i == 0) opt_map.push_back({col++, 0, 0});
        else             opt_map.push_back({col++, 1, static_cast<int>(i) + 1});
    }

    const BatchedDistanceTable& dt  = batch.dist_tables();
    const double*               pos = batch.positions().data();
    const bool periodic = batch.cell().is_periodic();
    std::vector<double> delta(dim * nw);
    std::vector<double> f1(n_opt), f2(n_opt);

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
            if (rr >= L) continue;
            const double t    = rr - L;
            const double tCm2 = ipow(t, C - 2);
            const double tCm1 = tCm2 * t;
            const double g    = tCm1 * t;
            const double dg   = Cd * tCm1;
            const double d2g  = Cd * (C - 1) * tCm2;

            // P(r) and its first two derivatives (see eval_one).
            double P = a0 + alpha1 * rr, dP = alpha1, d2P = 0.0;
            for (std::size_t k = 1; k < n_alpha; ++k) {
                const int    l = static_cast<int>(k) + 1;
                const double c = params_[k].value;
                P   += c * ipow(rr, l);
                dP  += c * l * ipow(rr, l - 1);
                d2P += c * l * (l - 1) * ipow(rr, l - 2);
            }

            for (std::size_t s = 0; s < n_opt; ++s) {
                const ParamInfo& info = opt_map[s];
                if (info.kind == 0) {
                    const double c  = Cd / L;
                    const double Q  = 1.0 + c * rr;
                    f1[s] = dg * Q + g * c;
                    f2[s] = d2g * Q + 2.0 * dg * c;
                } else if (info.kind == 1) {
                    const int    l   = info.power;
                    const double Q   = ipow(rr, l);
                    const double dQ  = l * ipow(rr, l - 1);
                    const double d2Q = l * (l - 1) * ipow(rr, l - 2);
                    f1[s] = dg * Q + g * dQ;
                    f2[s] = d2g * Q + 2.0 * dg * dQ + g * d2Q;
                } else {
                    const double h   = -Cd * tCm1;
                    const double dh  = -Cd * (C - 1) * tCm2;
                    const double d2h = (C >= 3) ? -Cd * (C - 1) * (C - 2) * ipow(t, C - 3)
                                                : 0.0;
                    const double a = dalpha1_dL;
                    f1[s] = dh * P + h * dP + dg * a * rr + g * a;
                    f2[s] = d2h * P + 2.0 * dh * dP + h * d2P + d2g * a * rr + 2.0 * dg * a;
                }
            }

            const double inv_r = 1.0 / rr;
            for (std::size_t d = 0; d < dim; ++d) {
                const double dor  = delta[d * nw + w] * inv_r;
                const double dor2 = dor * dor;
                const std::size_t gi = (i * dim + d) * nw + w;
                const std::size_t gj = (j * dim + d) * nw + w;
                for (std::size_t s = 0; s < n_opt; ++s) {
                    const double gd = f1[s] * dor;
                    const double ld = f2[s] * dor2 + f1[s] * inv_r * (1.0 - dor2);
                    const std::size_t o = opt_map[s].col * stride;
                    dgrad[o + gi] += gd;
                    dgrad[o + gj] -= gd;
                    dlap[o + gi]  += ld;
                    dlap[o + gj]  += ld;
                }
            }
        }
    };

    for_each_pair(accumulate_pair);
    return true;
}

}  // namespace alea
