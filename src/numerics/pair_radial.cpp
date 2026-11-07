// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "numerics/pair_radial.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "numerics/bessel_j0.h"
#include "numerics/quadrature.h"
#include "numerics/special_functions.h"

namespace alea::numerics {
namespace {

constexpr double kTwoPi = 2.0 * std::numbers::pi_v<double>;

// Terms in the analytic tail model: the remainder is then O(r^{−17}), so it
// is negligible a few model widths out.
constexpr int kModelTerms = 16;

// Largest |root| of Q(r) = 1 + Σ b_k r^k (Durand–Kerner on the monic form):
// the radius beyond which the 1/r expansion of P/Q converges.
double pole_radius(const std::vector<double>& b) {
    const std::size_t M = b.size();
    if (M == 0) return 0.0;
    // monic coefficients c_j of r^j, j = 0..M−1, after dividing by b_M.
    std::vector<double> c(M);
    c[0] = 1.0 / b.back();
    for (std::size_t j = 1; j < M; ++j) c[j] = b[j - 1] / b.back();
    std::vector<std::complex<double>> z(M);
    for (std::size_t i = 0; i < M; ++i) z[i] = std::pow(std::complex<double>(0.4, 0.9), double(i));
    for (int it = 0; it < 500; ++it) {
        double change = 0.0;
        for (std::size_t i = 0; i < M; ++i) {
            std::complex<double> p = 1.0;   // monic: z^M + Σ c_j z^j
            for (std::size_t j = M; j-- > 0;) p = p * z[i] + c[j];
            std::complex<double> den = 1.0;
            for (std::size_t l = 0; l < M; ++l)
                if (l != i) den *= z[i] - z[l];
            const std::complex<double> dz = p / den;
            z[i] -= dz;
            change = std::max(change, std::abs(dz));
        }
        if (change < 1e-14) break;
    }
    double r = 0.0;
    for (const auto& zi : z) r = std::max(r, std::abs(zi));
    return r;
}

// Beyond this k·r_c the smooth (windowed-away) part of the remainder has a
// negligible transform: it is C^∞ on the scale r_c, so its transform falls
// faster than any power.  The retained k (≤ 8|b₁|, k·r_c ≲ 30) are well below
// it; above it only the fit of the truncated tail is affected.
constexpr double kSmoothCutoff = 100.0;

// Rational u = P/Q with its first two radial derivatives.
struct Rational {
    std::vector<double> a, b;   // P = Σ a_k r^k (k ≥ 1), Q = 1 + Σ b_k r^k

    void eval(double r, double& u, double& du, double& d2u) const noexcept {
        double p = 0, dp = 0, d2p = 0, rk = 1.0;   // rk = r^{k−1} at step k
        for (std::size_t i = 0; i < a.size(); ++i) {
            const double k = static_cast<double>(i + 1);
            d2p += (i >= 1 ? k * (k - 1) * a[i] * (rk / r) : 0.0);
            dp  += k * a[i] * rk;
            rk  *= r;
            p   += a[i] * rk;
        }
        double q = 1, dq = 0, d2q = 0;
        rk = 1.0;
        for (std::size_t i = 0; i < b.size(); ++i) {
            const double k = static_cast<double>(i + 1);
            d2q += (i >= 1 ? k * (k - 1) * b[i] * (rk / r) : 0.0);
            dq  += k * b[i] * rk;
            rk  *= r;
            q   += b[i] * rk;
        }
        u   = p / q;
        du  = (dp - u * dq) / q;
        d2u = (d2p - 2.0 * du * dq - u * d2q) / q;
    }
};

// u(1/x) = A(x)/B(x) as a power series Σ_j e_j x^j, j = 0..n−1.  With
// N ≤ M, A(x) = Σ a_k x^{M−k} and B(x) = x^M + Σ b_k x^{M−k}, B(0) = b_M ≠ 0.
std::vector<double> large_r_series(const Rational& f, int n) {
    const std::size_t M = f.b.size();
    std::vector<double> A(M + 1, 0.0), B(M + 1, 0.0);
    for (std::size_t k = 1; k <= f.a.size(); ++k) A[M - k] += f.a[k - 1];
    B[M] = 1.0;
    for (std::size_t k = 1; k <= M; ++k) B[M - k] += f.b[k - 1];
    std::vector<double> e(n, 0.0);
    for (int j = 0; j < n; ++j) {
        double s = j <= static_cast<int>(M) ? A[j] : 0.0;
        for (int i = 1; i <= j && i <= static_cast<int>(M); ++i) s -= B[i] * e[j - i];
        e[j] = s / B[0];
    }
    return e;
}

// Smooth step: 1 for r ≤ lo, 0 for r ≥ hi, C^∞ in between.
double window(double r, double lo, double hi) noexcept {
    if (r <= lo) return 1.0;
    if (r >= hi) return 0.0;
    const double t = (r - lo) / (hi - lo);
    const double p = std::exp(-1.0 / (1.0 - t)), q = std::exp(-1.0 / t);
    return p / (p + q);
}

}  // namespace

RadialFunction pade_radial(std::span<const double> a_in, std::span<const double> b_in,
                           double r_cut, double model_width) {
    // Trailing zero coefficients do not raise the order.
    std::vector<double> a(a_in.begin(), a_in.end()), b(b_in.begin(), b_in.end());
    while (!a.empty() && a.back() == 0.0) a.pop_back();
    while (!b.empty() && b.back() == 0.0) b.pop_back();
    if (a.size() > b.size())
        throw std::invalid_argument(
            "padé jastrow: numerator of order " + std::to_string(a.size()) +
            " over denominator of order " + std::to_string(b.size()) +
            " grows without bound, so its sum over periodic images does not exist");
    // A real positive root of the denominator is a pole the images reach.
    {
        auto Q = [&](double r) {
            double q = 1.0, rk = 1.0;
            for (double bk : b) { rk *= r; q += bk * rk; }
            return q;
        };
        double prev = 1.0;
        for (int i = 0; i <= 4000; ++i) {
            const double r = 1e-6 * std::pow(1e14, i / 4000.0);   // 1e-6 .. 1e8
            const double q = Q(r);
            if (q <= 0.0 || (prev > 0.0) != (q > 0.0))
                throw std::invalid_argument(
                    "padé jastrow: the denominator vanishes near r = " +
                    std::to_string(r) + ", a pole that the periodic images reach");
            prev = q;
        }
    }

    auto f = std::make_shared<const Rational>(Rational{a, b});
    const std::vector<double> e = large_r_series(*f, kModelTerms + 1);
    const double u_inf = e[0];

    RadialFunction out;
    out.value = [f, u_inf](double r) { double u, d1, d2; f->eval(r, u, d1, d2); return u - u_inf; };
    out.d1    = [f](double r) { double u, d1, d2; f->eval(r, u, d1, d2); return d1; };
    out.d2    = [f](double r) { double u, d1, d2; f->eval(r, u, d1, d2); return d2; };
    out.tail_r_cut = r_cut;

    // Model m = Σ_{n=1}^{8} d_n (r²+s²)^{−n/2}.  Expanding
    //     (r²+s²)^{−n/2} = Σ_i C(−n/2, i) s^{2i} r^{−n−2i}
    // and matching r^{−1..−8} to e_m is triangular:
    //     d_m = e_m − Σ_{n<m, m−n even} d_n C(−n/2, (m−n)/2) s^{m−n}.
    // The 1/r series only converges beyond the poles of Q, so s is at least
    // twice the pole radius: that keeps every d_n s^{−n} of the size of u
    // itself instead of exploding.
    const double s = model_width > 0.0 ? model_width
                                       : std::max(0.5 * r_cut, 2.0 * pole_radius(b));
    std::array<double, kModelTerms + 1> d{};
    for (int m = 1; m <= kModelTerms; ++m) {
        double v = e[m];
        for (int n = m - 2; n >= 1; n -= 2) {
            const int i = (m - n) / 2;
            double c = 1.0;   // C(−n/2, i)
            for (int t = 0; t < i; ++t) c *= (-0.5 * n - t) / (t + 1);
            v -= d[n] * c * std::pow(s, m - n);
        }
        d[m] = v;
    }
    auto model = [d, s](double r) {
        const double w = 1.0 / std::sqrt(r * r + s * s);
        double acc = 0.0;
        for (int n = kModelTerms; n >= 1; --n) acc = w * (d[n] + acc);
        return acc;
    };
    auto remainder = [f, u_inf, model](double r) {
        double u, d1, d2;
        f->eval(r, u, d1, d2);
        return u - u_inf - model(r);
    };

    // Reach of the smooth remainder piece: h = O(r^{−17}).  Stop once r²|h| is
    // 1e-12 of the edge scale — this is a trial function, not an energy.
    const double scale = std::max(std::abs(out.value(r_cut)) * r_cut * r_cut, 1e-300);
    const double far = 64.0 * std::max(r_cut, s);
    double R = 2.0 * r_cut;
    while (R < far && std::abs(remainder(R)) * R * R > 1e-12 * scale) R *= 1.2;

    out.tail = [d, s, model, remainder, r_cut, R](double k) {
        // Closed-form transform of the whole model, ∫₀^∞ … (G&R 6.565.4 with
        // μ = n/2 − 1; K_{−μ} = K_μ covers n = 1).
        const double ks = k * s;
        double total = 0.0;
        for (int n = 1; n <= kModelTerms; ++n) {
            if (d[n] == 0.0) continue;
            const double mu = 0.5 * n - 1.0;
            total += d[n] * std::pow(k / s, mu) * bessel_k(mu, ks) /
                     (std::pow(2.0, mu) * std::tgamma(mu + 1.0));
        }
        // … minus the model's own part inside r_c (smooth on the scale s) …
        const int n_in = 1 + static_cast<int>(k * r_cut + 4.0 * r_cut / s);
        total -= gauss20_panels([&](double r) { return r * bessel_j0(k * r) * model(r); },
                                0.0, r_cut, n_in);
        // … plus the remainder beyond r_c: the windowed edge piece at every k,
        total += gauss20_panels(
            [&](double r) {
                return r * bessel_j0(k * r) * remainder(r) * window(r, r_cut, 2.0 * r_cut);
            },
            r_cut, 2.0 * r_cut, 40 + static_cast<int>(k * r_cut));
        // … and the smooth decaying piece where its transform is not negligible.
        if (k * r_cut < kSmoothCutoff) {
            total += gauss20_panels(
                [&](double r) {
                    return r * bessel_j0(k * r) * remainder(r) *
                           (1.0 - window(r, r_cut, 2.0 * r_cut));
                },
                r_cut, R, 40 + static_cast<int>(k * (R - r_cut) + 8.0 * (R - r_cut) / r_cut));
        }
        return kTwoPi * total;
    };
    return out;
}

double pade_limit(std::span<const double> a_in, std::span<const double> b_in) {
    std::vector<double> a(a_in.begin(), a_in.end()), b(b_in.begin(), b_in.end());
    while (!a.empty() && a.back() == 0.0) a.pop_back();
    while (!b.empty() && b.back() == 0.0) b.pop_back();
    if (a.size() > b.size())
        throw std::invalid_argument(
            "padé jastrow: numerator of order " + std::to_string(a.size()) +
            " over denominator of order " + std::to_string(b.size()) +
            " grows without bound, so its sum over periodic images does not exist");
    return large_r_series(Rational{a, b}, 1)[0];
}

}  // namespace alea::numerics
