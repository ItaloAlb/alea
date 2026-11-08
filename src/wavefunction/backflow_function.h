// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace alea {

// Two-body backflow function of P. López Ríos, A. Ma, N. D. Drummond,
// M. D. Towler and R. J. Needs, Phys. Rev. E 74, 066701 (2006), Eqs. (7)–(8):
//
//   η(r) = (1 − r/L)^C Θ(L − r) Σ_{k=0}^{N} c_k r^k,
//
// the displacement a source particle at separation r causes in a quasi-
// coordinate, x_i = r_i + Σ_j η(r_ij) r_ij.  The cutoff factor makes η and its
// first C − 1 derivatives vanish at L; C ≥ 3 keeps the Laplacian, hence the
// local energy, continuous there (C = 2 leaves a finite jump), C = 1 would put
// a delta function into it and is refused.
//
// ── Like-spin cusp ──────────────────────────────────────────────────────────
// Between identical fermions of one spin the determinant vanishes linearly at
// coalescence, D ∝ |x_i − x_j| ≈ r_ij (1 + 2η(0) + 2η'(0) r_ij + ...).  A
// nonzero η'(0) adds an isotropic linear term 2η'(0) r_ij/(1 + 2η(0)) to
// ln|D|, which shifts the P-wave cusp the Jastrow factor carries and makes the
// local energy diverge as 1/r_ij.  So a like-spin η must have η'(0) = 0, i.e.
// (Ref., Sec. III and App. A)
//
//   c_1 = C c_0 / L     (zero_slope).
//
// For distinguishable or opposite-spin pairs the linear term of ln|D| is
// anisotropic (∝ r_ij, a vector) and does not touch the cusp; c_1 is free.
//
// Free parameters, in this order: [c_0, c_1 (absent when zero_slope), c_2 ..
// c_N, L].  A pure value type: no walker state.
class BackflowFunction {
public:
    BackflowFunction() = default;

    // free_coefficients: [c_0, c_2..c_N] when zero_slope, else [c_0, c_1..c_N].
    BackflowFunction(int truncation, double cutoff, std::vector<double> free_coefficients,
                     bool zero_slope)
        : C_(truncation), L_(cutoff), zero_slope_(zero_slope),
          free_(std::move(free_coefficients)) {
        if (C_ < 2)
            throw std::invalid_argument(
                "backflow: truncation C must be >= 2 (C = 1 leaves a delta function "
                "in the Laplacian at the cutoff)");
        if (!(std::isfinite(L_) && L_ > 0.0))
            throw std::invalid_argument("backflow: the cutoff length L must be positive");
        if (free_.empty())
            throw std::invalid_argument("backflow: eta needs at least the coefficient c_0");
        rebuild_coefficients();
    }

    [[nodiscard]] double cutoff() const noexcept { return L_; }
    [[nodiscard]] int truncation() const noexcept { return C_; }
    [[nodiscard]] bool zero_slope() const noexcept { return zero_slope_; }
    // Free coefficients plus L.
    [[nodiscard]] std::size_t n_parameters() const noexcept { return free_.size() + 1; }
    // Polynomial order N.
    [[nodiscard]] std::size_t order() const noexcept { return c_.size() - 1; }
    [[nodiscard]] std::span<const double> coefficients() const noexcept { return c_; }

    // Parameter k ([c_0, (c_1), c_2.., L]) set to value; the derived c_1 follows.
    void set_parameter(std::size_t k, double value) {
        if (k < free_.size()) {
            free_[k] = value;
        } else if (k == free_.size()) {
            if (!(std::isfinite(value) && value > 0.0))
                throw std::invalid_argument("backflow: the cutoff length L must be positive");
            L_ = value;
        } else {
            throw std::out_of_range("backflow: no parameter " + std::to_string(k));
        }
        rebuild_coefficients();
    }

    // η, η' and η'' at r (all 0 for r ≥ L).  With t = 1 − r/L, f = t^C:
    //   f' = −(C/L) t^{C−1},  f'' = C(C−1)/L² t^{C−2},
    //   η = f P,  η' = f'P + fP',  η'' = f''P + 2f'P' + fP''.
    void evaluate(double r, double& eta, double& d1, double& d2) const noexcept {
        if (r >= L_) {
            eta = d1 = d2 = 0.0;
            return;
        }
        const double t = 1.0 - r / L_;
        const double t_c2 = power(t, C_ - 2);
        const double f   = t_c2 * t * t;
        const double fp  = -static_cast<double>(C_) / L_ * t_c2 * t;
        const double fpp = static_cast<double>(C_) * static_cast<double>(C_ - 1) / (L_ * L_) * t_c2;
        double P = 0.0, Pp = 0.0, Ppp = 0.0;   // Horner for P, P', P''
        for (std::size_t k = c_.size(); k-- > 0;) {
            Ppp = Ppp * r + 2.0 * Pp;
            Pp  = Pp * r + P;
            P   = P * r + c_[k];
        }
        eta = f * P;
        d1  = fp * P + f * Pp;
        d2  = fpp * P + 2.0 * fp * Pp + f * Ppp;
    }

    // ∂η/∂p_k at r for every parameter, in the order [c_0, (c_1), c_2.., L];
    // all zero for r ≥ L.  With zero_slope, c_1 = C c_0/L adds C r/L to the
    // c_0 derivative and −C c_0 r/L² to the L derivative.
    //   ∂η/∂c_k = f r^k,   ∂f/∂L = C t^{C−1} r/L².
    void parameter_derivatives(double r, std::span<double> out) const noexcept {
        for (double& v : out) v = 0.0;
        if (r >= L_) return;
        const double t = 1.0 - r / L_;
        const double t_c1 = power(t, C_ - 1);
        const double f = t_c1 * t;
        double P = 0.0;
        for (std::size_t k = c_.size(); k-- > 0;) P = P * r + c_[k];
        std::size_t slot = 0;
        double rk = 1.0;   // r^k
        for (std::size_t k = 0; k < c_.size(); ++k, rk *= r) {
            if (zero_slope_ && k == 1) continue;
            out[slot] = f * rk;
            if (zero_slope_ && k == 0) out[slot] += f * static_cast<double>(C_) * r / L_;
            ++slot;
        }
        double dL = static_cast<double>(C_) * t_c1 * r / (L_ * L_) * P;
        if (zero_slope_) dL -= f * static_cast<double>(C_) * c_[0] * r / (L_ * L_);
        out[slot] = dL;
    }

    // ∂η/∂p_k, ∂η'/∂p_k and ∂η''/∂p_k at r, same order as above (all zero for
    // r ≥ L); what the parameter derivative of a backflow Laplacian needs.
    // A free coefficient multiplies a basis function b(r) (r^k, or 1 + C r/L
    // for c_0 when zero_slope), so with f = t^C:
    //   ∂η/∂c = f b,  ∂η'/∂c = f'b + f b',  ∂η''/∂c = f''b + 2f'b' + f b''.
    // The cutoff enters f and, when zero_slope, c_1 = C c_0/L; with
    //   g = ∂f/∂L = (C/L²) r t^{C−1},
    //   g' = (C/L²) [t^{C−1} − (C−1)(r/L) t^{C−2}],
    //   g'' = (C(C−1)/L³) [−2 t^{C−2} + (C−2)(r/L) t^{C−3}],
    //   ∂P/∂L = a r, ∂P'/∂L = a, a = −C c_0/L² (zero_slope; else 0):
    //   ∂η/∂L   = gP + f a r,
    //   ∂η'/∂L  = g'P + gP' + f'a r + f a,
    //   ∂η''/∂L = g''P + 2g'P' + gP'' + f''a r + 2f'a.
    void parameter_derivatives(double r, std::span<double> d_eta, std::span<double> d_d1,
                               std::span<double> d_d2) const noexcept {
        for (std::size_t k = 0; k < d_eta.size(); ++k) d_eta[k] = d_d1[k] = d_d2[k] = 0.0;
        if (r >= L_) return;
        const double C = static_cast<double>(C_);
        const double t = 1.0 - r / L_;
        const double t_c2 = power(t, C_ - 2);
        const double t_c1 = t_c2 * t;
        const double f   = t_c1 * t;
        const double fp  = -C / L_ * t_c1;
        const double fpp = C * (C - 1.0) / (L_ * L_) * t_c2;
        double P = 0.0, Pp = 0.0, Ppp = 0.0;
        for (std::size_t k = c_.size(); k-- > 0;) {
            Ppp = Ppp * r + 2.0 * Pp;
            Pp  = Pp * r + P;
            P   = P * r + c_[k];
        }
        std::size_t slot = 0;
        for (std::size_t k = 0; k < c_.size(); ++k) {
            if (zero_slope_ && k == 1) continue;
            // b = r^k, b' = k r^{k−1}, b'' = k(k−1) r^{k−2} (zero below k).
            const double kd = static_cast<double>(k);
            double b = power(r, static_cast<int>(k));
            double b1 = k >= 1 ? kd * power(r, static_cast<int>(k) - 1) : 0.0;
            double b2 = k >= 2 ? kd * (kd - 1.0) * power(r, static_cast<int>(k) - 2) : 0.0;
            if (zero_slope_ && k == 0) {   // c_0 also sets c_1 = C c_0/L
                b += C * r / L_;
                b1 += C / L_;
            }
            d_eta[slot] = f * b;
            d_d1[slot]  = fp * b + f * b1;
            d_d2[slot]  = fpp * b + 2.0 * fp * b1 + f * b2;
            ++slot;
        }
        const double pre = C / (L_ * L_);
        const double g   = pre * r * t_c1;
        const double g1  = pre * (t_c1 - (C - 1.0) * (r / L_) * t_c2);
        // (C − 2) t^{C−3}: zero for C = 2, where t^{C−3} would be 1/t.
        const double t_c3 = C_ >= 3 ? power(t, C_ - 3) : 0.0;
        const double g2  = C * (C - 1.0) / (L_ * L_ * L_) *
                          (-2.0 * t_c2 + (C - 2.0) * (r / L_) * t_c3);
        const double a = zero_slope_ ? -C * c_[0] / (L_ * L_) : 0.0;
        d_eta[slot] = g * P + f * a * r;
        d_d1[slot]  = g1 * P + g * Pp + fp * a * r + f * a;
        d_d2[slot]  = g2 * P + 2.0 * g1 * Pp + g * Ppp + fpp * a * r + 2.0 * fp * a;
    }

private:
    static double power(double t, int n) noexcept {
        double p = 1.0;
        for (int i = 0; i < n; ++i) p *= t;
        return p;
    }

    void rebuild_coefficients() {
        // c_ = [c_0 .. c_N] with c_1 inserted (zero_slope) or taken as given.
        c_.clear();
        if (zero_slope_) {
            c_.push_back(free_[0]);
            c_.push_back(static_cast<double>(C_) * free_[0] / L_);
            for (std::size_t k = 1; k < free_.size(); ++k) c_.push_back(free_[k]);
        } else {
            c_ = free_;
        }
    }

    int                 C_ = 3;
    double              L_ = 1.0;
    bool                zero_slope_ = false;
    std::vector<double> free_;   // [c_0, (c_1), c_2..c_N]
    std::vector<double> c_;      // [c_0 .. c_N], c_1 derived when zero_slope
};

}  // namespace alea
