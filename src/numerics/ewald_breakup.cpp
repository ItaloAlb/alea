// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "numerics/ewald_breakup.h"

#include "numerics/long_range_breakup.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "numerics/keldysh_kernel.h"
#include "numerics/special_functions.h"

namespace alea::numerics {
namespace {

// Every helper of the fixed erfc splitting (solve_alpha, solve_kcut,
// hankel_long_range, fit_long_range) is gone: the splitting is now the
// optimized breakup, which has no alpha to solve for and fits nothing in real
// space -- its long-range half IS the spline it solved for.

// 20-point Gauss-Legendre, as ten symmetric pairs.
constexpr double kGx[10] = {0.0765265211334973, 0.2277858511416451,
                            0.3737060887154195, 0.5108670019508271,
                            0.6360536807265150, 0.7463319064601508,
                            0.8391169718222188, 0.9122344282513259,
                            0.9639719272779138, 0.9931285991850949};
constexpr double kGw[10] = {0.1527533871307258, 0.1491729864726037,
                            0.1420961093183820, 0.1316886384491766,
                            0.1181945319615184, 0.1019301198172404,
                            0.0832767415767048, 0.0626720483341091,
                            0.0406014298003869, 0.0176140071391521};

// ∫_0^∞ f(t) dt for a smooth f that is regular at 0 and whose features sit at
// the scale `s` or beyond.  Panels [0, h] then [h 2^i, h 2^{i+1}]: each panel
// is as wide as its distance from the origin, so a feature at any scale ≥ h
// falls in a panel of its own size, where 20-point Gauss-Legendre is exact to
// rounding for the smooth, non-oscillatory integrands used here.  Stops once
// the panels pass `t_end`, beyond which f is negligible.
template <class F>
double geometric_integral(F&& f, double s, double t_end) {
    auto panel = [&](double a, double b) {
        const double c = 0.5 * (a + b), h = 0.5 * (b - a);
        double sum = 0.0;
        for (int i = 0; i < 10; ++i)
            sum += kGw[i] * (f(c + h * kGx[i]) + f(c - h * kGx[i]));
        return sum * h;
    };
    double       a   = 1e-3 * s;
    double       sum = panel(0.0, a);
    for (int i = 0; i < 400 && a < t_end; ++i, a *= 2.0) sum += panel(a, 2.0 * a);
    return sum;
}

// ── Rytova-Keldysh between two planes ───────────────────────────────────────
//
// The pair feels V(r) = V_RK(s), s = sqrt(r² + d²) -- the real-space form the
// open-boundary KeldyshTerm uses.  Its transform has no closed form, but two
// standard identities reduce it to a smooth 1-D integral:
//
//   H0(x) − Y0(x) = (2/π) ∫_0^∞ e^{−x t}/sqrt(1+t²) dt      (DLMF 11.5.2)
//     ⇒  V_RK(s) = (1/ε) ∫_0^∞ e^{−s u} w(u) du,   w(u) = 1/sqrt(1 + r0² u²)
//
// a superposition of e^{−s u} (r0 → 0 leaves ∫e^{−su}du = 1/s, Coulomb); and
//
//   2π ∫_0^∞ r J0(kr) e^{−u sqrt(r²+d²)} dr = 2π u e^{−dq}(1 + dq)/q³,
//   q = sqrt(k² + u²)   (−∂_u of Sommerfeld's ∫ J0 e^{−uR}/R r dr = e^{−dq}/q).
//
// Swapping the integrals and changing variable u → q (u du = q dq):
//
//   Ṽ(k) = (2π/ε) ∫_k^∞ e^{−dq}(1 + dq) / (q² sqrt(1 + r0²(q² − k²))) dq.
//
// Smooth, positive, non-oscillatory, decaying as e^{−dq}; at r0 = 0 it
// integrates to 2π e^{−kd}/(εk) exactly.
double keldysh_offset_fourier(double k, double eps, double r0, double d) {
    const auto f = [=](double t) {
        const double q = k + t;
        return std::exp(-d * q) * (1.0 + d * q) /
               (q * q * std::sqrt(1.0 + r0 * r0 * t * (2.0 * k + t)));
    };
    // Scales where the integrand changes: 1/q² at k, the exponential at 1/d,
    // the square root at 1/r0 or, for large k, at 1/(2 k r0²).
    const double s = std::min({k, 1.0 / d, 1.0 / r0, 1.0 / (2.0 * k * r0 * r0)});
    return 2.0 * PI / eps * geometric_integral(f, s, 60.0 / d);
}

// c = lim_{k→0}[Ṽ(k) − 2π/(εk)] = ∫ (V − 1/(εr)) d²r.  Splitting V − 1/(εr)
// into [V_RK(s) − 1/(εs)] + [1/(εs) − 1/(εr)], the second part integrates to
// −2π d/ε and the first, through the representation above, to
//
//   (2π/ε) ∫_0^∞ (w(u) − 1) e^{−du}(1 + du)/u² du,
//
// with w − 1 = −r0²u²/(sqrt(1+r0²u²)(1 + sqrt(1+r0²u²))) written so that
// nothing cancels as u → 0.  At d = 0 the integral is −r0 in closed form.
double keldysh_offset_k0(double eps, double r0, double d) {
    const auto f = [=](double u) {
        const double root = std::sqrt(1.0 + r0 * r0 * u * u);
        return -r0 * r0 * std::exp(-d * u) * (1.0 + d * u) / (root * (1.0 + root));
    };
    const double s = std::min(1.0 / d, 1.0 / r0);
    return -2.0 * PI * d / eps + 2.0 * PI / eps * geometric_integral(f, s, 60.0 / d);
}

}  // namespace

PotentialSpec PotentialSpec::coulomb(double epsilon, double d_layer) {
    if (!(d_layer >= 0.0))
        throw std::runtime_error("PotentialSpec::coulomb: d_layer must be >= 0");
    PotentialSpec p;
    p.kind    = PotentialSpec::Kind::Coulomb;
    p.epsilon = epsilon;
    p.d_layer = d_layer;
    const double inv_eps = 1.0 / epsilon;
    if (d_layer == 0.0) {
        p.v_real    = [inv_eps](double r) { return inv_eps / r; };
        p.v_fourier = [inv_eps](double k) { return 2.0 * PI * inv_eps / k; };
        p.dv_real   = [inv_eps](double r) { return -inv_eps / (r * r); };
        p.d2v_real  = [inv_eps](double r) { return 2.0 * inv_eps / (r * r * r); };
        return p;
    }
    // With s² = r² + d²:  V = 1/(eps s),  V' = −r/(eps s³),
    // V'' = (2r² − d²)/(eps s⁵).
    const double d2 = d_layer * d_layer;
    p.v_real    = [inv_eps, d2](double r) { return inv_eps / std::sqrt(r * r + d2); };
    p.v_fourier = [inv_eps, d_layer](double k) {
        return 2.0 * PI * inv_eps * std::exp(-k * d_layer) / k;
    };
    p.dv_real   = [inv_eps, d2](double r) {
        const double s2 = r * r + d2;
        return -inv_eps * r / (s2 * std::sqrt(s2));
    };
    p.d2v_real  = [inv_eps, d2](double r) {
        const double s2 = r * r + d2;
        return inv_eps * (2.0 * r * r - d2) / (s2 * s2 * std::sqrt(s2));
    };
    p.k0_offset = -2.0 * PI * d_layer * inv_eps;
    return p;
}

PotentialSpec PotentialSpec::rytova_keldysh(double epsilon, double r0,
                                            double d_layer) {
    if (!(r0 > 0.0))
        throw std::runtime_error(
            "PotentialSpec::rytova_keldysh: r0 must be positive (use "
            "PotentialSpec::coulomb for the unscreened limit)");
    if (!(d_layer >= 0.0))
        throw std::runtime_error("PotentialSpec::rytova_keldysh: d_layer must be >= 0");
    PotentialSpec p;
    p.kind    = PotentialSpec::Kind::Keldysh;
    p.epsilon = epsilon;
    p.r0      = r0;
    p.d_layer = d_layer;
    const double prefactor = PI / (2.0 * epsilon * r0);
    const double inv_r0    = 1.0 / r0;

    if (d_layer > 0.0) {
        // V(r) = V_RK(s), s = sqrt(r² + d²).  With V_RK' and V_RK'' at s:
        //   V'  = V_RK' r/s,   V'' = V_RK'' r²/s² + V_RK' d²/s³.
        const double d2 = d_layer * d_layer;
        p.v_real   = [prefactor, inv_r0, d2](double r) {
            return prefactor * keldysh_kernel(std::sqrt(r * r + d2) * inv_r0);
        };
        p.dv_real  = [prefactor, inv_r0, d2](double r) {
            const double s = std::sqrt(r * r + d2);
            return prefactor * inv_r0 * keldysh_kernel_d1(s * inv_r0) * r / s;
        };
        p.d2v_real = [prefactor, inv_r0, d2](double r) {
            const double s  = std::sqrt(r * r + d2);
            const double v1 = prefactor * inv_r0 * keldysh_kernel_d1(s * inv_r0);
            const double v2 = prefactor * inv_r0 * inv_r0 * keldysh_kernel_d2(s * inv_r0);
            return v2 * r * r / (s * s) + v1 * d2 / (s * s * s);
        };
        p.v_fourier = [epsilon, r0, d_layer](double k) {
            return keldysh_offset_fourier(k, epsilon, r0, d_layer);
        };
        p.k0_offset = keldysh_offset_k0(epsilon, r0, d_layer);
        return p;
    }
    // The SETUP path uses the high-precision kernel, not the Chebyshev table:
    // the table is calibrated for the hot loop at ~5e-9, which would be the
    // limiting noise in a breakup whose own accuracy is ~3e-10.
    p.v_real    = [prefactor, inv_r0](double r) {
        return prefactor * keldysh_kernel(r * inv_r0);
    };
    p.v_fourier = [epsilon, r0](double k) {
        return 2.0 * PI / (k * epsilon * (1.0 + r0 * k));
    };
    p.dv_real   = [prefactor, inv_r0](double r) {
        return prefactor * inv_r0 * keldysh_kernel_d1(r * inv_r0);
    };
    p.d2v_real  = [prefactor, inv_r0](double r) {
        return prefactor * inv_r0 * inv_r0 * keldysh_kernel_d2(r * inv_r0);
    };
    // 2π/(εk(1+r0 k)) − 2π/(εk) = −2π r0/(ε(1+r0 k)) → −2π r0/ε as k → 0.
    p.k0_offset = -2.0 * PI * r0 / epsilon;
    return p;
}

EwaldSetup build_ewald(const SimulationCell& cell, double tol,
                       const PotentialSpec& pot) {
    if (!pot.v_real || !pot.v_fourier || !pot.dv_real || !pot.d2v_real)
        throw std::runtime_error("build_ewald: the potential spec is incomplete");
    if (!cell.is_periodic())
        throw std::runtime_error("build_ewald: the cell is not periodic");
    if (cell.dim() != 2)
        throw std::runtime_error(
            "build_ewald: implemented for dimension 2 only (got " +
            std::to_string(cell.dim()) + ")");
    if (!(tol > 0.0) || !(tol < 1.0) || !std::isfinite(tol))
        throw std::runtime_error("build_ewald: tolerance must lie in (0, 1)");

    // The splitting is the Natoli-Ceperley optimized breakup (see
    // numerics/long_range_breakup.h), which unlike the fixed erfc split it
    // replaced places no analyticity demand on eps(k) and so works for
    // Rytova-Keldysh as well as Coulomb.
    RadialFunction f;
    f.value   = pot.v_real;
    f.d1      = pot.dv_real;
    f.d2      = pot.d2v_real;
    // The engine only ever needs the kernel's transform pair (V, Ṽ); nothing
    // about which physics produced it.
    f.fourier = pot.v_fourier;

    // n_knots is auto-tuned, k_cut is not: in this method accuracy is set by
    // the fit and COST by the reciprocal cutoff, which was verified directly by
    // freezing the fit and sweeping k_cut over a 10x range without the energy
    // moving.  So spend knots (setup only) and leave k_cut at the default.
    //
    // Accuracy is not monotone in n_knots: it bottoms out near 20 and then
    // degrades as cond(C) grows, so the ladder stops rather than climbing.
    LongRangeBreakup br;
    bool             passed = false;
    for (int n_knots : {12, 16, 20, 24}) {
        BreakupOptions opt;
        opt.n_knots = n_knots;
        br = build_long_range_breakup(cell, f, opt);
        if (br.residual_above_cut() < tol) {
            passed = true;
            break;
        }
    }
    if (!passed)
        throw std::runtime_error(
            "build_ewald: the breakup did not reach the requested tolerance "
            "(residual " + std::to_string(br.residual_above_cut()) +
            " against tol " + std::to_string(tol) +
            ").  Loosen ewald_tol, or use a larger cell: r_c is the "
            "Wigner-Seitz inradius and a small cell leaves the fit too little "
            "room.");

    EwaldSetup s;
    s.area         = cell.volume();
    s.r_cut        = br.r_cut();
    s.k_cut        = br.k_cut();
    s.kx           = br.kx();
    s.ky           = br.ky();
    s.n1           = br.n1();
    s.n2           = br.n2();
    s.weight       = br.weight();
    s.n1_max       = br.n1_max();
    s.n2_max       = br.n2_max();
    s.long_range   = br.spline();
    s.v_long_zero  = br.long_range_zero();
    s.sr_k0        = br.short_range_k0();
    s.residual     = br.residual_above_cut();
    s.condition    = br.condition();
    return s;
}

}  // namespace alea::numerics
