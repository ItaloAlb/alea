// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "wavefunction/zero_energy_jastrow.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "population/walker_set.h"
#include "wavefunction/radial_pair_jastrow_impl.h"

namespace alea {

// ---------------------------------------------------------------------------
//  ZeroEnergyFunction
// ---------------------------------------------------------------------------

ZeroEnergyFunction::ZeroEnergyFunction(double b, double exponent,
                                       double matching_radius, double cutoff)
    : s_(0.5 * (exponent - 2.0)),
      kappa_(0.0),
      r_m_(matching_radius),
      r_c_(cutoff),
      c3_(0.0),
      log_c1_(0.0),
      log_c2_(0.0),
      dc3_(0.0),
      bessel_(&numerics::ScaledBesselK01::table()) {
    if (!(std::isfinite(b) && b > 0.0))
        throw std::invalid_argument(
            "zero_energy jastrow: the core strength b = 2*mu*C must be finite and > 0 "
            "(a repulsive C/r^p core); got " + std::to_string(b));
    if (!(std::isfinite(exponent) && exponent > 2.0))
        throw std::invalid_argument(
            "zero_energy jastrow: needs a core C/r^p with p > 2; got p = " +
            std::to_string(exponent));
    if (!(std::isfinite(cutoff) && cutoff > 0.0))
        throw std::invalid_argument("zero_energy jastrow: the cutoff r_c must be finite "
                                    "and > 0; got " + std::to_string(cutoff));
    if (!(matching_radius > 0.0 && matching_radius < cutoff))
        throw std::invalid_argument(
            "zero_energy jastrow: the matching radius R_m = " +
            std::to_string(matching_radius) + " must lie inside (0, r_c = " +
            std::to_string(cutoff) + ")");
    kappa_ = std::sqrt(b) / s_;

    // Matching at R_m (header): continuity of u and u′, and u(r_c) = 0.
    double u0 = 0.0, du0 = 0.0, d2u0 = 0.0;
    zero_energy_solution(r_m_, u0, du0, d2u0);
    const double c  = 2.0 * r_c_ - r_m_;
    const double h  = 1.0 / (r_m_ * r_m_) - 1.0 / (c * c);  // > 0 for R_m < r_c
    const double dh = -2.0 / (r_m_ * r_m_ * r_m_) - 2.0 / (c * c * c);
    c3_     = du0 / h;
    log_c2_ = 2.0 * c3_ / r_c_;
    log_c1_ = log_c2_ - c3_ / r_m_ - c3_ / c - u0;
    dc3_    = (d2u0 * h - du0 * dh) / (h * h);
}

void ZeroEnergyFunction::zero_energy_solution(double r, double& u0, double& du0,
                                              double& d2u0) const noexcept {
    // x = κ r^{−s}; ln K₀(x) = ln(e^x K₀(x)) − x keeps the scaled value, which
    // neither underflows nor needs a log of a tiny number.
    const double x = kappa_ * std::pow(r, -s_);
    double k0e = 0.0, k1e = 0.0;
    bessel_->evaluate(x, k0e, k1e);
    const double q  = k1e / k0e;  // K₁(x)/K₀(x)
    const double sx = s_ * x;
    u0   = std::log(k0e) - x;
    du0  = sx * q / r;
    d2u0 = sx / (r * r) * (sx * (1.0 - q) * (1.0 + q) - q);
}

void ZeroEnergyFunction::evaluate(double r, double& u, double& du,
                                  double& d2u) const noexcept {
    if (r < r_m_) {
        zero_energy_solution(r, u, du, d2u);
        u += log_c1_;
        return;
    }
    // Tail; clamped at r_c so 2r_c − r stays positive, then masked to 0.
    const double rr    = std::min(r, r_c_);
    const double inv_r = 1.0 / rr;
    const double inv_c = 1.0 / (2.0 * r_c_ - rr);
    const double m     = (r < r_c_) ? 1.0 : 0.0;
    u   = m * (log_c2_ - c3_ * inv_r - c3_ * inv_c);
    du  = m * (c3_ * inv_r * inv_r - c3_ * inv_c * inv_c);
    d2u = m * (-2.0 * c3_ * inv_r * inv_r * inv_r - 2.0 * c3_ * inv_c * inv_c * inv_c);
}

void ZeroEnergyFunction::evaluate(std::span<const double> r, std::span<double> u,
                                  std::span<double> du,
                                  std::span<double> d2u) const noexcept {
    const std::size_t n = r.size();
    const double a = c3_, rc = r_c_, two_rc = 2.0 * r_c_, log_c2 = log_c2_, rm = r_m_;
    const double* __restrict rin  = r.data();
    double* __restrict       uo   = u.data();
    double* __restrict       duo  = du.data();
    double* __restrict       d2uo = d2u.data();

    // The tail (or 0) for every walker, branch-free; note whether any walker
    // has this pair inside R_m.  Most pairs are not, so the Bessel part runs
    // only for the few walkers that need it.
    int any_inside = 0;
#pragma omp simd reduction(|: any_inside)
    for (std::size_t w = 0; w < n; ++w) {
        const double rr    = std::min(rin[w], rc);
        const double inv_r = 1.0 / rr;
        const double inv_c = 1.0 / (two_rc - rr);
        const double m     = (rin[w] < rc) ? 1.0 : 0.0;
        uo[w]   = m * (log_c2 - a * inv_r - a * inv_c);
        duo[w]  = m * (a * inv_r * inv_r - a * inv_c * inv_c);
        d2uo[w] = m * (-2.0 * a * inv_r * inv_r * inv_r - 2.0 * a * inv_c * inv_c * inv_c);
        any_inside |= (rin[w] < rm) ? 1 : 0;
    }
    if (!any_inside) return;
    for (std::size_t w = 0; w < n; ++w) {
        if (rin[w] >= rm) continue;
        zero_energy_solution(rin[w], uo[w], duo[w], d2uo[w]);
        uo[w] += log_c1_;
    }
}

void ZeroEnergyFunction::matching_radius_derivative(double r, double& du_dp,
                                                    double& d_du_dp,
                                                    double& d_d2u_dp) const noexcept {
    if (r >= r_c_) {
        du_dp = d_du_dp = d_d2u_dp = 0.0;
        return;
    }
    // ∂u/∂R_m = C₃′ φ(max(r, R_m)): constant inside R_m, so its r-derivatives
    // vanish there.
    const double rho = std::max(r, r_m_);
    const double c   = 2.0 * r_c_ - rho;
    du_dp = dc3_ * (2.0 / r_c_ - 1.0 / rho - 1.0 / c);
    if (r < r_m_) {
        d_du_dp = d_d2u_dp = 0.0;
        return;
    }
    d_du_dp  = dc3_ * (1.0 / (r * r) - 1.0 / (c * c));
    d_d2u_dp = dc3_ * (-2.0 / (r * r * r) - 2.0 / (c * c * c));
}

// ---------------------------------------------------------------------------
//  ZeroEnergyJastrow
// ---------------------------------------------------------------------------

// The pair-loop machinery for this u, in this translation unit so that
// ZeroEnergyFunction::evaluate inlines into it.
template class RadialPairJastrow<ZeroEnergyFunction>;

namespace {

double matching_radius_of(const std::vector<Parameter>& params) {
    if (params.size() != 1)
        throw std::invalid_argument("zero_energy jastrow: expects one parameter, the "
                                    "matching radius R_m; got " +
                                    std::to_string(params.size()));
    return params[0].value;
}

}  // namespace

ZeroEnergyJastrow::ZeroEnergyJastrow(double b, double exponent, double cutoff,
                                     ParticleGroup group_a, ParticleGroup group_b,
                                     std::vector<Parameter> params)
    : RadialPairJastrow(std::move(group_a), std::move(group_b),
                        ZeroEnergyFunction(b, exponent, matching_radius_of(params), cutoff)),
      b_(b),
      exponent_(exponent),
      cutoff_(cutoff),
      params_(std::move(params)) {}

void ZeroEnergyJastrow::rebuild() {
    const double r_m = params_[0].value;
    if (!(r_m > 0.0 && r_m < cutoff_))
        throw std::runtime_error(
            "zero_energy jastrow: the matching radius R_m = " + std::to_string(r_m) +
            " left (0, r_c = " + std::to_string(cutoff_) + "); the tail "
            "C2 exp(-C3/r - C3/(2 r_c - r)) is matched inside the cutoff.");
    radial_ = ZeroEnergyFunction(b_, exponent_, r_m, cutoff_);
}

void ZeroEnergyJastrow::compute_parameter_derivatives(const WalkerSet& batch,
                                                      std::span<double> derivs) const {
    if (num_optimizable_params() == 0) return;
    const std::size_t nw = batch.num_walkers();
    std::fill(derivs.begin(), derivs.end(), 0.0);
    const BatchedDistanceTable& dt = batch.dist_tables();
    for_each_pair([&](std::size_t i, std::size_t j) {
        const double* r = dt.row(i, j).data();
        for (std::size_t w = 0; w < nw; ++w) {
            double du_dp = 0.0, d_du_dp = 0.0, d_d2u_dp = 0.0;
            radial_.matching_radius_derivative(r[w], du_dp, d_du_dp, d_d2u_dp);
            derivs[w] += du_dp;
        }
    });
}

// ∂(∂_d log ψ)/∂R_m and ∂(∂²_d log ψ)/∂R_m.  f = ∂u/∂R_m is itself a radial
// pair function, so it enters exactly as u does in evaluate_batch:
//   ∂_d f = f′ Δ_d/r (− for partner j),  ∂²_d f = f″ (Δ_d/r)² + (f′/r)(1 − (Δ_d/r)²).
bool ZeroEnergyJastrow::compute_parameter_gradient_derivatives(
    const WalkerSet& batch, std::span<double> dgrad, std::span<double> dlap) const {
    if (num_optimizable_params() == 0) return true;
    const std::size_t nw  = batch.num_walkers();
    const std::size_t dim = batch.dim();
    std::fill(dgrad.begin(), dgrad.end(), 0.0);
    std::fill(dlap.begin(), dlap.end(), 0.0);

    const BatchedDistanceTable& dt  = batch.dist_tables();
    const double*               pos = batch.positions().data();
    const bool periodic = batch.cell().is_periodic();
    std::vector<double> delta(dim * nw);  // [dim * nw], Δ = r_i − r_j

    for_each_pair([&](std::size_t i, std::size_t j) {
        const double* xi = pos + i * dim * nw;
        const double* xj = pos + j * dim * nw;
        if (periodic)
            batch.cell().min_image_batch(xi, xj, nw, delta.data(), 0, nw);
        else
            for (std::size_t k = 0; k < dim * nw; ++k) delta[k] = xi[k] - xj[k];
        const double* r = dt.row(i, j).data();
        for (std::size_t w = 0; w < nw; ++w) {
            double f = 0.0, f1 = 0.0, f2 = 0.0;
            radial_.matching_radius_derivative(r[w], f, f1, f2);
            if (f1 == 0.0 && f2 == 0.0) continue;
            const double inv_r = 1.0 / r[w];
            for (std::size_t d = 0; d < dim; ++d) {
                const double dor  = delta[d * nw + w] * inv_r;
                const double dor2 = dor * dor;
                const double gd   = f1 * dor;
                const double ld   = f2 * dor2 + f1 * inv_r * (1.0 - dor2);
                const std::size_t gi = (i * dim + d) * nw + w;
                const std::size_t gj = (j * dim + d) * nw + w;
                dgrad[gi] += gd;
                dgrad[gj] -= gd;
                dlap[gi]  += ld;
                dlap[gj]  += ld;
            }
        }
    });
    return true;
}

}  // namespace alea
