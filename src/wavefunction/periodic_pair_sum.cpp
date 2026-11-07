// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "wavefunction/periodic_pair_sum.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "population/walker_set.h"

namespace alea {

PeriodicPairSum::PeriodicPairSum(const SimulationCell& cell, ParticleGroup a,
                                 ParticleGroup b,
                                 const numerics::RadialFunction& u_minus_limit,
                                 double u_inf,
                                 const std::vector<numerics::RadialFunction>& param_minus_limit,
                                 const std::vector<double>& param_inf)
    : active_(true),
      u_inf_(u_inf),
      param_inf_(param_inf) {
    if (cell.dim() != 2)
        throw std::invalid_argument("image-summed jastrow: only 2-D cells are supported");
    if (param_minus_limit.size() != param_inf.size())
        throw std::invalid_argument("image-summed jastrow: one limit per parameter function");

    // A Jastrow's ũ_long(k) falls much faster than Coulomb's 2π/k (u is bounded
    // with at most a cusp), so the fit needs a far shorter k range than the
    // potential's defaults (shells to 60π/r_c, continuum to 6000/r_c).  The
    // retained weights still come from the exact tail transform; this only
    // sets how much of ũ_long the truncation drops.
    numerics::BreakupOptions opt{};
    const double rc = cell.ws_radius();
    opt.k_continuum = 20.0 * 3.14159265358979323846 / rc;
    opt.k_max       = 300.0 / rc;
    opt.n_continuum = 150;
    // Retained k: |k| ≤ 5|b₁| rather than the potential's 8|b₁|.  Measured on
    // a [2/2] Padé (one and two groups) against 12|b₁|: 5|b₁| already sits on
    // the fit's own noise floor (ln Ψ contrasts and gradients within 1e-8,
    // ≤ 1e-6 relative), the same as 8|b₁|; 4|b₁| is 4× worse.  k·r_c is what
    // matters, so this carries over to every cell size.  (5/8)² of the k
    // vectors: the k-space move cost drops ~2.6×.
    opt.k_cut = 5.0 * std::min(std::hypot(cell.reciprocal()[0], cell.reciprocal()[1]),
                               std::hypot(cell.reciprocal()[2], cell.reciprocal()[3]));
    const numerics::LongRangeBreakup br =
        numerics::build_long_range_breakup(cell, u_minus_limit, opt);
    spline_ = br.spline();

    // ∂u/∂p is split on the SAME k set (the set depends only on the cell and
    // the options), so ∂w_k/∂p lines up with w_k.
    std::vector<std::vector<double>> param_w;
    for (const numerics::RadialFunction& f : param_minus_limit) {
        const numerics::LongRangeBreakup bp = numerics::build_long_range_breakup(cell, f, opt);
        if (bp.weight().size() != br.weight().size())
            throw std::logic_error("image-summed jastrow: parameter breakup k set differs");
        param_splines_.push_back(bp.spline());
        param_w.push_back(bp.weight());
    }
    k_ = KSpacePairSum(cell, std::move(a), std::move(b), br.n1(), br.n2(), br.kx(), br.ky(),
                       br.weight(), std::move(param_w));
}

void PeriodicPairSum::to_short_range(std::span<const double> r, std::span<double> u,
                                     std::span<double> du,
                                     std::span<double> d2u) const noexcept {
    // (A branch-free, simd form of this loop measured no faster: 1m15 vs
    // 1m15 on periodic_2d_ehgas.  The k-space move path dominates.)
    const double rc = spline_.r_cut;
    for (std::size_t i = 0; i < r.size(); ++i) {
        if (r[i] >= rc) {
            u[i] = du[i] = d2u[i] = 0.0;
            continue;
        }
        double f, df, d2f;
        spline_.eval(r[i], f, df, d2f);
        u[i]   -= u_inf_ + f;
        du[i]  -= df;
        d2u[i] -= d2f;
    }
}

void PeriodicPairSum::param_to_short_range(std::size_t k, double r, double& v, double& d1,
                                           double& d2) const noexcept {
    if (r >= spline_.r_cut) {
        v = d1 = d2 = 0.0;
        return;
    }
    double f, df, d2f;
    param_splines_[k].eval(r, f, df, d2f);
    v  -= param_inf_[k] + f;
    d1 -= df;
    d2 -= d2f;
}

}  // namespace alea
