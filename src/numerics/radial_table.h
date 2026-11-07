// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace alea::numerics {

// A smooth radial function g(r) on [0, r_max], tabulated as piecewise
// Chebyshev series on uniform panels, with its first two derivatives.
//
// ── Why piecewise ───────────────────────────────────────────────────────────
// The functions tabulated here (the smooth part of a screened potential) are
// analytic but can vary on a length scale much shorter than r_max -- a thin
// dielectric layer puts a feature at its thickness.  One global series would
// need a degree set by r_max over that scale; uniform panels of a fixed,
// modest degree adapt by adding panels instead, and keep every evaluation to
// one short Clenshaw recurrence.
//
// ── Accuracy ────────────────────────────────────────────────────────────────
// build() doubles the panel count until the table agrees with g itself at
// points BETWEEN the interpolation nodes to `tol_abs` -- measured, not assumed.
// Before checking accuracy, a degree-five Hermite correction gives adjacent
// panels shared values/slopes/curvatures: values come from g at each knot;
// derivatives average the two fitted limits. This makes the sampled table C²
// at panel joins, to roundoff, without another hot-loop polynomial. The
// correction is folded into the Chebyshev coefficients at setup. This is an
// approximation to g, not a proof of physical derivative accuracy.
// Differentiate each corrected series exactly. Coefficient errors can grow
// like degree²/h in the slope and degree⁴/h² in curvature; the value tolerance
// alone does not bound either physical derivative (h is the panel width).
class RadialTable {
public:
    static constexpr int kN = 16;  // coefficients per panel (degree 15)

    RadialTable() = default;

    // Throws std::runtime_error if 4096 panels do not reach tol_abs.
    [[nodiscard]] static RadialTable build(const std::function<double(double)>& g,
                                           double r_max, double tol_abs) {
        if (!g || !std::isfinite(r_max) || r_max <= 0.0 ||
            !std::isfinite(tol_abs) || tol_abs <= 0.0 ||
            !std::isfinite(8192.0/r_max))
            throw std::invalid_argument("RadialTable: finite positive radius and tolerance required");
        for (int m = 8; m <= 4096; m *= 2) {
            RadialTable t = fit(g, r_max, m, tol_abs);
            t.match_endpoints(g);
            double worst = 0.0;
            const double h = r_max / m;
            for (int p = 0; p < m; ++p)
                for (int j = 0; j < 7; ++j) {
                    // Points strictly between Chebyshev nodes of the panel.
                    const double r = h * (p + (j + 0.5) / 7.0);
                    const double e = std::abs(t.eval(r) - g(r));
                    // A NaN compares false and would slip past std::max.
                    worst = std::isfinite(e) ? std::max(worst, e) : INFINITY;
                }
            if (worst < tol_abs) return t;
        }
        throw std::runtime_error(
            "RadialTable: could not reach an absolute accuracy of " +
            std::to_string(tol_abs) + " with 4096 panels");
    }

    [[nodiscard]] double r_max() const noexcept { return r_max_; }
    [[nodiscard]] int    n_panels() const noexcept { return n_panels_; }

    // r is clamped into [0, r_max]: callers that may ask past the edge (a
    // masked hot loop) get a finite number, never garbage memory.
    [[nodiscard]] double eval(double r) const noexcept { return run(c_, r, 1.0); }
    // Ordinary derivatives of the clamped value; exact edges select the
    // interior branch. A default table represents the identically zero function.
    [[nodiscard]] double d1(double r) const noexcept {
        return r < 0.0 || r > r_max_ ? 0.0 : run(c1_, r, 2.0 * inv_h_);
    }
    [[nodiscard]] double d2(double r) const noexcept {
        return r < 0.0 || r > r_max_ ? 0.0 :
            (run(c2_, r, 2.0 * inv_h_) * (2.0 * inv_h_));
    }

private:
    double run(const std::vector<double>& c, double r, double scale) const noexcept {
        if (std::isnan(r)) return std::numeric_limits<double>::quiet_NaN();
        if (n_panels_ == 0) return 0.0;
        double rc = r < 0.0 ? 0.0 : (r > r_max_ ? r_max_ : r);
        int    p  = static_cast<int>(rc * inv_h_);
        p         = p < n_panels_ ? p : n_panels_ - 1;
        const double  x  = 2.0 * (rc * inv_h_ - p) - 1.0;
        const double* cp = c.data() + static_cast<std::size_t>(p) * kN;
        // Clenshaw for Σ_{m} c_m T_m(x), c_0 stored already halved.
        return scale*clenshaw(cp,x);
    }

    static double clenshaw(const double* cp, double x) noexcept {
        double b1 = 0.0, b2 = 0.0;
        for (int m = kN - 1; m >= 1; --m) {
            const double b0 = 2.0 * x * b1 - b2 + cp[m];
            b2 = b1;
            b1 = b0;
        }
        return x * b1 - b2 + cp[0];
    }

    void match_endpoints(const std::function<double(double)>& g) {
        // Knot data use xi derivatives: d/dr=(2/h)d/dxi. The common panel
        // width permits averaging without amplifying setup rounding by 1/h².
        std::vector<std::array<double,3>> knots(n_panels_+1); // [knot][value,d_xi,d2_xi]
        for (int j=0; j<=n_panels_; ++j) {
            auto& knot=knots[j];
            knot[0]=g(r_max_*(double(j)/n_panels_));
            for (int order=1; order<=2; ++order) {
                const auto& coefficients=order == 1 ? c1_ : c2_;
                const double left=j>0 ? clenshaw(coefficients.data()+(j-1)*kN,1.0) : 0.0;
                const double right=j<n_panels_ ? clenshaw(coefficients.data()+j*kN,-1.0) : 0.0;
                knot[order]=j == 0 ? right : (j == n_panels_ ? left : 0.5*left+0.5*right);
            }
            for (double value : knot)
                if (!std::isfinite(value))
                    throw std::runtime_error("RadialTable: nonfinite endpoint data");
        }
        const long double pi=std::acos(-1.0L);
        for (int p=0; p<n_panels_; ++p) {
            double* cp=c_.data()+p*kN;
            std::array<double,3> left{},right{}; // [residual value,d_xi,d2_xi]
            for (int order=0; order<=2; ++order) {
                const auto& coefficients=order == 0 ? c_ : (order == 1 ? c1_ : c2_);
                left[order]=knots[p][order]-clenshaw(coefficients.data()+p*kN,-1.0);
                right[order]=knots[p+1][order]-clenshaw(coefficients.data()+p*kN,1.0);
            }
            // Hermite bases at each end. In u=(xi+1)/2 the slope/curvature
            // residuals scale by 2/4; the mirrored far-end slope changes sign.
            auto endpoint=[](double u,const std::array<double,3>& delta,double sign) {
                const double v=1.0-u;
                return u*u*u*(delta[0]*(10.0+u*(-15.0+6.0*u))+
                    sign*2.0*delta[1]*(-4.0+u*(7.0-3.0*u))+
                    2.0*delta[2]*v*v);
            };
            double sampled[kN]; // [Chebyshev interpolation node]
            for (int j=0; j<kN; ++j) {
                const double u=0.5*(1.0+static_cast<double>(std::cos(pi*(j+0.5L)/kN)));
                sampled[j]=endpoint(1.0-u,left,-1.0)+endpoint(u,right,1.0);
            }
            // The correction has degree five. Project only its six nonzero
            // orders, retaining the existing fixed-degree hot-loop evaluator.
            for (int k=0; k<=5; ++k) {
                long double sum=0.0L;
                for (int j=0; j<kN; ++j) sum+=sampled[j]*std::cos(pi*k*(j+0.5L)/kN);
                cp[k]+=static_cast<double>((k == 0 ? 1.0L : 2.0L)*sum/kN);
            }
            differentiate(cp,c1_.data()+p*kN);
            differentiate(c1_.data()+p*kN,c2_.data()+p*kN);
        }
    }

    // Chebyshev derivative: for f = Σ a_m T_m (a_0 full), f' = Σ d_m T_m with
    // d_{m-1} = d_{m+1} + 2 m a_m (d_n = d_{n+1} = 0) and d_0 halved.
    static void differentiate(const double* a_half, double* d_half) {
        double a[kN], d[kN + 2] = {};
        for (int m = 0; m < kN; ++m) a[m] = a_half[m];
        a[0] *= 2.0;
        for (int m = kN - 1; m >= 1; --m) d[m - 1] = d[m + 1] + 2.0 * m * a[m];
        for (int m = 0; m < kN; ++m) d_half[m] = d[m];
        d_half[0] *= 0.5;
    }

    static RadialTable fit(const std::function<double(double)>& g, double r_max, int m,
                           double tol_abs) {
        RadialTable t;
        t.r_max_    = r_max;
        t.n_panels_ = m;
        t.inv_h_    = m / r_max;
        t.c_.assign(static_cast<std::size_t>(m) * kN, 0.0);
        t.c1_.assign(t.c_.size(), 0.0);
        t.c2_.assign(t.c_.size(), 0.0);
        // Endpoint curvature amplifies tiny spurious high-order coefficients.
        // Project in extended precision, then discard coefficients below the
        // resolution of double g samples. Their total value contribution is
        // bounded by tol_abs/4; post-correction validation still sets admission.
        const long double pi = std::acos(-1.0L);
        const double h = r_max / m;
        double f[kN];
        for (int p = 0; p < m; ++p) {
            double magnitude=0.0;
            for (int j = 0; j < kN; ++j) {
                const double x = static_cast<double>(std::cos(pi * (j + 0.5L) / kN));
                f[j] = g(h * (p + 0.5 * (x + 1.0)));
                magnitude=std::max(magnitude,std::abs(f[j]));
            }
            double* cp = t.c_.data() + static_cast<std::size_t>(p) * kN;
            for (int k = 0; k < kN; ++k) {
                long double s = 0.0L;
                for (int j = 0; j < kN; ++j) s += f[j] * std::cos(pi * k * (j + 0.5L) / kN);
                cp[k] = static_cast<double>(2.0L * s / kN);
            }
            cp[0] *= 0.5;
            const double floor=std::min(4.0*std::numeric_limits<double>::epsilon()*magnitude,
                                        tol_abs/(4.0*kN));
            for (int k=1; k<kN; ++k)
                if (std::abs(cp[k])<floor) cp[k]=0.0;
            differentiate(cp, t.c1_.data() + static_cast<std::size_t>(p) * kN);
            differentiate(t.c1_.data() + static_cast<std::size_t>(p) * kN,
                          t.c2_.data() + static_cast<std::size_t>(p) * kN);
        }
        return t;
    }

    double              r_max_    = 0.0;
    double              inv_h_    = 0.0;
    int                 n_panels_ = 0;
    std::vector<double> c_, c1_, c2_;  // [panel * kN + m]
};

}  // namespace alea::numerics
