// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "numerics/long_range_breakup.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

#include "numerics/bessel_j0.h"
#include "numerics/quadrature.h"

namespace alea::numerics {
namespace {

constexpr double kPi = std::numbers::pi_v<double>;

// c_n(k) = 2π ∫ r J₀(kr) h_n(r) dr.
//
// Integrated as TWO panels split at the knot.  h_n is piecewise polynomial with
// different formulae on [r_{i-1}, r_i] and [r_i, r_{i+1}]; a Gauss-Legendre
// panel straddling r_i integrates across a kink in the third derivative and
// loses several digits without any sign that it did.  This split was worth four
// orders of magnitude in the Madelung constant when it was first put in.
double basis_transform(const LpqhiBasis& basis, int n, double k) {
    const int i = n / 3;
    const double lo = basis.support_lo(n);
    const double mid = basis.knot(i);
    const double hi = basis.support_hi(n);
    const auto integrand = [&](double r) {
        return r * bessel_j0(k * r) * basis.h(n, r);
    };
    double s = 0.0;
    if (mid > lo) {
        s += panel_integral(integrand, lo, mid, k);
    }
    if (hi > mid) {
        s += panel_integral(integrand, mid, hi, k);
    }
    return 2.0 * kPi * s;
}

// 2π ∫₀^{r_c} r f(r) dr — the k = 0 limit of the inner integral below.  Split
// off because X(0) itself does not exist: f̃(0) diverges for any potential with
// a 1/r tail, which is precisely why a charged cell needs a background.
double radial_integral(const RadialFunction& f, double r_cut) {
    const auto integrand = [&](double r) { return r * f.value(r); };
    double s = 0.0, b = r_cut;
    for (int j = 0; j < 60 && b > 1e-14 * r_cut; ++j) {
        const double a = 0.5 * b;
        s += panel_integral(integrand, a, b, 0.0);
        b = a;
    }
    return 2.0 * kPi * s;
}

// X(k) = −∫_{r > r_c} f(r) e^{−ik·r} d²r = −f̃(k) + 2π ∫₀^{r_c} r J₀(kr) f(r) dr.
//
// The inner integral runs on a geometric mesh collapsing toward the origin: for
// a Coulomb f the integrand r·f(r) is constant and any mesh works, but for a
// logarithmic one it is −r log r, which is finite yet has unbounded derivative
// at 0.  Halving panels lets each see a smooth function on its own scale.
double fit_target(const RadialFunction& f, double r_cut, double k) {
    if (f.tail) {
        return -f.tail(k);
    }
    const auto integrand = [&](double r) {
        return r * bessel_j0(k * r) * f.value(r);
    };
    double s = 0.0, b = r_cut;
    for (int j = 0; j < 60 && b > 1e-14 * r_cut; ++j) {
        const double a = 0.5 * b;
        s += panel_integral(integrand, a, b, k);
        b = a;
    }
    return -f.fourier(k) + 2.0 * kPi * s;
}

}  // namespace

double LongRangeBreakup::long_range_real(double r) const noexcept {
    if (r >= basis_.r_cut()) {
        return value_ ? value_(r) : 0.0;
    }
    double s = 0.0;
    for (std::size_t n = 0; n < coefs_.size(); ++n) {
        s += coefs_[n] * basis_.h(static_cast<int>(n), r);
    }
    return s;
}

LongRangeBreakup build_long_range_breakup(const SimulationCell& cell,
                                          const RadialFunction& f,
                                          const BreakupOptions& opt) {
    if (!cell.is_periodic() || cell.dim() != 2) {
        throw std::invalid_argument(
            "long-range breakup: needs a periodic 2-D cell");
    }
    if (opt.n_knots < 4) {
        throw std::invalid_argument("long-range breakup: n_knots must be >= 4");
    }
    if (!f.value || !(f.fourier || f.tail) || !f.d1 || !f.d2) {
        throw std::invalid_argument(
            "long-range breakup: value, d1, d2 and a fourier or tail transform "
            "are all required");
    }

    LongRangeBreakup out;
    // r_c is the Wigner-Seitz inradius: the largest reach for which the
    // real-space sum over images collapses onto the minimum image.
    const double r_cut = cell.ws_radius();
    if (f.tail && std::abs(f.tail_r_cut - r_cut) > 1e-12 * r_cut) {
        throw std::invalid_argument(
            "long-range breakup: the tail transform was built for r_c = " +
            std::to_string(f.tail_r_cut) + ", but this cell's is " +
            std::to_string(r_cut));
    }
    out.basis_ = LpqhiBasis(opt.n_knots, r_cut);
    out.value_ = f.value;

    const auto b = cell.reciprocal();
    const double b1x = b[0], b1y = b[1], b2x = b[2], b2y = b[3];
    const double b_min = std::min(std::hypot(b1x, b1y), std::hypot(b2x, b2y));
    const double area = cell.volume();

    const double k_cut = opt.k_cut > 0.0 ? opt.k_cut : 8.0 * b_min;
    const double k_continuum =
        opt.k_continuum > 0.0 ? opt.k_continuum : 60.0 * kPi / r_cut;
    const double k_max = opt.k_max > 0.0 ? opt.k_max : 6000.0 / r_cut;
    if (k_continuum <= k_cut || k_max <= k_continuum) {
        throw std::invalid_argument(
            "long-range breakup: need k_cut < k_continuum < k_max");
    }
    out.k_cut_ = k_cut;

    // ── k-list: exact lattice shells up to k_continuum, continuum beyond ────
    std::vector<double> kf, kw;   // |k| and its (possibly fractional) count
    {
        const int nmax = static_cast<int>(std::ceil(k_continuum / b_min)) + 1;
        std::vector<double> raw;
        for (int i = -nmax; i <= nmax; ++i) {
            for (int j = -nmax; j <= nmax; ++j) {
                if (i == 0 && j == 0) {
                    continue;
                }
                const double k = std::hypot(i * b1x + j * b2x, i * b1y + j * b2y);
                if (k > k_cut && k <= k_continuum) {
                    raw.push_back(k);
                }
            }
        }
        std::sort(raw.begin(), raw.end());
        for (std::size_t a = 0; a < raw.size();) {
            std::size_t z = a;
            while (z < raw.size() && raw[z] - raw[a] < 1e-9 * b_min) {
                ++z;
            }
            kf.push_back(raw[a]);
            kw.push_back(static_cast<double>(z - a));
            a = z;
        }
        // Continuum tail.  In 2-D the number of lattice points in [k, k+dk] is
        // (A/(2π)²)·2πk dk = (A/2π)·k dk, so each sample carries that as a
        // fractional degeneracy.  Log spacing because |f̃_long|²·k falls steeply.
        const double lo = std::log(k_continuum), hi = std::log(k_max);
        const double dl = (hi - lo) / opt.n_continuum;
        for (int i = 0; i < opt.n_continuum; ++i) {
            const double k = std::exp(lo + (i + 0.5) * dl);
            kf.push_back(k);
            kw.push_back(area * k * (k * dl) / (2.0 * kPi));
        }
    }

    const int nb = out.basis_.size();
    const int n_free = nb - 3;   // last three are assigned, not fitted
    out.coefs_.assign(nb, 0.0);

    // ── boundary condition at r_c ───────────────────────────────────────────
    // f_long must meet f smoothly at r_c so that f_short = f − f_long vanishes
    // there with two derivatives and the compact support is real.  Because the
    // LPQHI coefficients ARE (value, slope, curvature) at their knot, this is an
    // assignment rather than a constraint on the fit.
    out.coefs_[nb - 3] = f.value(r_cut);
    out.coefs_[nb - 2] = f.d1(r_cut);
    out.coefs_[nb - 1] = f.d2(r_cut);

    // ── design matrix ───────────────────────────────────────────────────────
    const std::size_t nrows = kf.size();
    std::vector<double> cnk(nrows * static_cast<std::size_t>(nb));
    std::vector<double> target(nrows);
    for (std::size_t r = 0; r < nrows; ++r) {
        for (int n = 0; n < nb; ++n) {
            cnk[r * nb + n] = basis_transform(out.basis_, n, kf[r]);
        }
        double x = fit_target(f, r_cut, kf[r]);
        for (int n = n_free; n < nb; ++n) {
            x -= out.coefs_[n] * cnk[r * nb + n];
        }
        target[r] = x;
    }

    // ── weighted least squares by Householder QR ────────────────────────────
    // Deliberately NOT the normal equations: CᵀWC squares the condition number,
    // and cond(C) is already ~1e5–1e6 here because neighbouring knots overlap
    // and their transforms are nearly parallel.  Squaring it destroys the
    // information outright — forming the Gram matrix rounds 1 + d² to 1 — so no
    // solver, however careful, can recover it afterwards.
    std::vector<double> d(nrows * static_cast<std::size_t>(n_free));
    std::vector<double> y(nrows);
    for (std::size_t r = 0; r < nrows; ++r) {
        const double sw = std::sqrt(kw[r]);
        for (int n = 0; n < n_free; ++n) {
            d[r * n_free + n] = sw * cnk[r * nb + n];
        }
        y[r] = sw * target[r];
    }
    std::vector<double> diag_r(n_free, 0.0);
    for (int j = 0; j < n_free; ++j) {
        double norm2 = 0.0;
        for (std::size_t r = j; r < nrows; ++r) {
            norm2 += d[r * n_free + j] * d[r * n_free + j];
        }
        const double norm = std::sqrt(norm2);
        if (norm < 1e-300) {
            continue;
        }
        const double head = d[static_cast<std::size_t>(j) * n_free + j];
        const double alpha = head > 0.0 ? -norm : norm;
        const double v0 = head - alpha;
        const double vnorm2 = norm2 - head * head + v0 * v0;
        if (vnorm2 < 1e-300) {
            diag_r[j] = alpha;
            continue;
        }
        d[static_cast<std::size_t>(j) * n_free + j] = v0;
        for (int c = j + 1; c < n_free; ++c) {
            double dot = 0.0;
            for (std::size_t r = j; r < nrows; ++r) {
                dot += d[r * n_free + j] * d[r * n_free + c];
            }
            const double s = 2.0 * dot / vnorm2;
            for (std::size_t r = j; r < nrows; ++r) {
                d[r * n_free + c] -= s * d[r * n_free + j];
            }
        }
        double dot = 0.0;
        for (std::size_t r = j; r < nrows; ++r) {
            dot += d[r * n_free + j] * y[r];
        }
        const double s = 2.0 * dot / vnorm2;
        for (std::size_t r = j; r < nrows; ++r) {
            y[r] -= s * d[r * n_free + j];
        }
        diag_r[j] = alpha;
    }
    for (int i = n_free - 1; i >= 0; --i) {
        double s = y[i];
        for (int j = i + 1; j < n_free; ++j) {
            s -= d[static_cast<std::size_t>(i) * n_free + j] * out.coefs_[j];
        }
        out.coefs_[i] = std::abs(diag_r[i]) > 1e-300 ? s / diag_r[i] : 0.0;
    }
    {
        double hi = 0.0, lo = 1e300;
        for (int i = 0; i < n_free; ++i) {
            const double a = std::abs(diag_r[i]);
            hi = std::max(hi, a);
            if (a > 1e-300) {
                lo = std::min(lo, a);
            }
        }
        out.condition_ = lo < 1e299 ? hi / lo : 0.0;
    }

    // Hot-loop form: coefficients pre-scaled by δ^α so evaluation needs no pow.
    out.spline_.n_knots   = opt.n_knots;
    out.spline_.r_cut     = r_cut;
    out.spline_.inv_delta = 1.0 / out.basis_.delta();
    out.spline_.scaled.resize(out.coefs_.size());
    for (std::size_t n = 0; n < out.coefs_.size(); ++n) {
        const int a = static_cast<int>(n) % 3;
        out.spline_.scaled[n] =
            out.coefs_[n] * std::pow(out.basis_.delta(), a);
    }

    // f_long(0) = t_0: every other basis function vanishes at the origin.
    out.long_range_zero_ = out.coefs_[0];

    // (1/A) ∫ f_short d²r.  The basis half is c_n(0): c_n(k) is
    // 2π∫ r J₀(kr) h_n(r) dr and J₀(0) = 1, so the machinery already exists.
    {
        double integral = radial_integral(f, r_cut);
        for (int n = 0; n < nb; ++n) {
            integral -= out.coefs_[n] * basis_transform(out.basis_, n, 0.0);
        }
        out.sr_k0_ = integral / area;
    }
    for (double t : out.coefs_) {
        out.t_max_ = std::max(out.t_max_, std::abs(t));
    }

    // ── diagnostics: the residual that the truncation actually drops ────────
    double chi2 = 0.0, wsum = 0.0;
    for (std::size_t r = 0; r < nrows; ++r) {
        double resid = target[r];
        for (int n = 0; n < n_free; ++n) {
            resid -= out.coefs_[n] * cnk[r * nb + n];
        }
        chi2 += kw[r] * resid * resid;
        wsum += kw[r];
        // Only the exact lattice shells are physical; the continuum samples are
        // a quadrature device and do not correspond to any retained k.
        if (kf[r] <= k_continuum) {
            out.residual_ = std::max(out.residual_, std::abs(resid));
        }
    }
    out.chi_ = wsum > 0.0 ? std::sqrt(chi2 / wsum) : 0.0;

    // ── retained reciprocal vectors ─────────────────────────────────────────
    // Half space, since |ρ(k)|² = |ρ(−k)|².  The energy is
    //   E = Σ_i weight_i |ρ(k_i)|²   with weight = f̃_long(k)/A,
    // the ½ from the double sum and the ×2 from the fold cancelling.
    {
        const int nmax = static_cast<int>(std::ceil(k_cut / b_min)) + 1;
        for (int i = -nmax; i <= nmax; ++i) {
            for (int j = -nmax; j <= nmax; ++j) {
                if (i < 0 || (i == 0 && j <= 0)) {
                    continue;   // keep one of each ±k pair
                }
                const double kx = i * b1x + j * b2x;
                const double ky = i * b1y + j * b2y;
                const double k = std::hypot(kx, ky);
                if (k > k_cut || k == 0.0) {
                    continue;
                }
                double fs = 0.0;
                for (int n = 0; n < nb; ++n) {
                    fs += out.coefs_[n] * basis_transform(out.basis_, n, k);
                }
                const double f_long = -fit_target(f, r_cut, k) + fs;
                out.kx_.push_back(kx);
                out.ky_.push_back(ky);
                out.weight_.push_back(f_long / area);
                out.n1_.push_back(i);
                out.n2_.push_back(j);
                out.n1_max_ = std::max(out.n1_max_, std::abs(i));
                out.n2_max_ = std::max(out.n2_max_, std::abs(j));
            }
        }
    }
    return out;
}

}  // namespace alea::numerics
