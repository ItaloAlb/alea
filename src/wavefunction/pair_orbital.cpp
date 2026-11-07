// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "wavefunction/pair_orbital.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "population/simulation_cell.h"

namespace alea {

PairOrbital::PairOrbital(const SimulationCell& cell, std::size_t shells, int poly_order)
    : dim_(cell.dim()), n_stars_(shells), poly_order_(poly_order) {
    if (poly_order != -1 && poly_order < 1)
        throw std::invalid_argument(
            "pair orbital: poly_order must be at least 1 (c_1 is fixed by the "
            "no-cusp condition), or absent; got " + std::to_string(poly_order));
    if (shells == 0 && poly_order < 1)
        throw std::invalid_argument("pair orbital: needs cosine shells or a polynomial");
    if (shells > 0 && (!cell.is_periodic() || cell.dim() != 2))
        throw std::invalid_argument(
            "pair orbital: cosine shells need a periodic 2-D cell (they are the "
            "reciprocal vectors of the cell)");
    if (has_polynomial()) c_.assign(static_cast<std::size_t>(poly_order) + 1, 0.0);
    if (shells == 0) return;

    // Star 0 is k = 0; the others are the half-space vectors (n1 > 0, or
    // n1 = 0 and n2 > 0) sorted by |k|, cut after `shells` distinct lengths.
    // The scanned box doubles until the last star ends well inside it.
    const auto b = cell.reciprocal();
    const double bmin = std::min(std::hypot(b[0], b[1]), std::hypot(b[2], b[3]));
    struct K { double k, kx, ky; };
    std::vector<K> ks;
    for (int nmax = 2;; nmax *= 2) {
        ks.clear();
        for (int i = 0; i <= nmax; ++i)
            for (int j = -nmax; j <= nmax; ++j) {
                if (i == 0 && j <= 0) continue;
                const double kx = i * b[0] + j * b[2], ky = i * b[1] + j * b[3];
                ks.push_back({std::hypot(kx, ky), kx, ky});
            }
        std::sort(ks.begin(), ks.end(), [](const K& x, const K& y) { return x.k < y.k; });
        std::size_t stars = 1;   // k = 0
        double last = 0.0, end_k = 0.0;
        for (const K& q : ks) {
            if (q.k - last > 1e-9 * bmin) {
                if (stars == shells) { end_k = q.k; break; }
                ++stars;
                last = q.k;
            }
        }
        if (stars == shells && end_k > 0.0 && end_k < 0.9 * nmax * bmin) break;
        if (shells == 1) break;
    }

    k_ = {0.0, 0.0};
    star_ = {0};
    mult_ = {1.0};
    star_k_ = {0.0};
    double last = 0.0;
    for (const K& q : ks) {
        if (q.k - last > 1e-9 * bmin) {
            if (star_k_.size() == shells) break;
            star_k_.push_back(q.k);
            last = q.k;
        }
        k_.push_back(q.kx);
        k_.push_back(q.ky);
        star_.push_back(star_k_.size() - 1);
        mult_.push_back(2.0);
    }
    weight_.assign(star_.size(), 0.0);
}

void PairOrbital::set(std::span<const double> p, std::span<const double> c, double L) {
    if (p.size() != n_stars_)
        throw std::invalid_argument("pair orbital: " + std::to_string(p.size()) +
                                    " cosine coefficients for " +
                                    std::to_string(n_stars_) + " shells");
    for (std::size_t i = 0; i < star_.size(); ++i) weight_[i] = mult_[i] * p[star_[i]];
    if (!has_polynomial()) return;
    const std::size_t M = static_cast<std::size_t>(poly_order_);
    if (c.size() != M)
        throw std::invalid_argument("pair orbital: " + std::to_string(c.size()) +
                                    " polynomial coefficients, expected c_0, c_2 … c_" +
                                    std::to_string(M) + " (" + std::to_string(M) + ")");
    if (!(L > 0.0) || !std::isfinite(L))
        throw std::invalid_argument("pair orbital: cutoff L must be positive, got " +
                                    std::to_string(L));
    L_ = L;
    c_[0] = c[0];
    c_[1] = 3.0 * c[0] / L;   // φ′(0) = −3 c_0/L + c_1 = 0
    for (std::size_t m = 2; m <= M; ++m) c_[m] = c[m - 1];
}

namespace {

// P(r) = Σ c_m r^m and its first two derivatives (Horner).
void polynomial(const std::vector<double>& c, double r, double& P, double& P1, double& P2) {
    P = c.back();
    P1 = 0.0;
    P2 = 0.0;
    for (std::size_t m = c.size() - 1; m-- > 0;) {
        P2 = P2 * r + 2.0 * P1;
        P1 = P1 * r + P;
        P = P * r + c[m];
    }
}

}  // namespace

void PairOrbital::radial(double r, double& g, double& dg, double& d2g) const noexcept {
    if (!has_polynomial() || r >= L_) {
        g = dg = d2g = 0.0;
        return;
    }
    // f = t³, t = 1 − r/L:  f′ = −3t²/L,  f″ = 6t/L².
    const double t = 1.0 - r / L_;
    const double f = t * t * t, f1 = -3.0 * t * t / L_, f2 = 6.0 * t / (L_ * L_);
    double P, P1, P2;
    polynomial(c_, r, P, P1, P2);
    g = f * P;
    dg = f1 * P + f * P1;
    d2g = f2 * P + 2.0 * f1 * P1 + f * P2;
}

void PairOrbital::radial_param(std::size_t j, double r, double& h, double& dh,
                               double& d2h) const noexcept {
    const std::size_t M = static_cast<std::size_t>(poly_order_);
    if (!has_polynomial() || r >= L_ || j > M) {
        h = dh = d2h = 0.0;
        return;
    }
    const double t = 1.0 - r / L_;
    const double f = t * t * t, f1 = -3.0 * t * t / L_, f2 = 6.0 * t / (L_ * L_);
    if (j == 0) {
        // c_0 enters as c_0 + (3 c_0/L) r:  ∂g/∂c_0 = f q, q = 1 + 3r/L.
        const double q = 1.0 + 3.0 * r / L_, q1 = 3.0 / L_;
        h = f * q;
        dh = f1 * q + f * q1;
        d2h = f2 * q + 2.0 * f1 * q1;
    } else if (j < M) {
        // ∂g/∂c_m = f r^m, m = j + 1.
        const double m = static_cast<double>(j + 1);
        const double rm2 = std::pow(r, m - 2.0), rm1 = rm2 * r, rm = rm1 * r;
        h = f * rm;
        dh = f1 * rm + f * m * rm1;
        d2h = f2 * rm + 2.0 * f1 * m * rm1 + f * m * (m - 1.0) * rm2;
    } else {
        // ∂g/∂L = (∂f/∂L) P + f (∂c_1/∂L) r, with ∂f/∂L = 3 r t²/L² ≡ A and
        // ∂c_1/∂L = −3 c_0/L², so B ≡ −3 c_0 r/L².
        double P, P1, P2;
        polynomial(c_, r, P, P1, P2);
        const double L2 = L_ * L_;
        const double A = 3.0 * r * t * t / L2;
        const double A1 = 3.0 / L2 * (t * t - 2.0 * r * t / L_);
        const double A2 = 3.0 / L2 * (-4.0 * t / L_ + 2.0 * r / L2);
        const double B = -3.0 * c_[0] * r / L2, B1 = -3.0 * c_[0] / L2;
        h = A * P + f * B;
        dh = A1 * P + A * P1 + f1 * B + f * B1;
        d2h = A2 * P + 2.0 * A1 * P1 + A * P2 + f2 * B + 2.0 * f1 * B1;
    }
}

}  // namespace alea
