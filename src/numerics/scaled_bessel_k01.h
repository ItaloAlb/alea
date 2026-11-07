// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cmath>
#include <numbers>

#include "special_functions.h"

namespace alea::numerics {

// e^x K₀(x) and e^x K₁(x), x > 0, in double precision and cheap enough for a
// per-walker loop (the zero-energy pair Jastrow needs them for every close pair
// at every move).  Scaled so that neither underflows: K_ν(x) ~ e^{−x}.
//
//   x ≤ 2 : the ascending series (DLMF 10.31.2, and 10.31.1 at n = 1)
//             K₀(x) = −(ln(x/2) + γ) I₀(x) + Σ_{k≥1} H_k y^k/(k!)²
//             K₁(x) = 1/x + ln(x/2) I₁(x) − (x/4) Σ_{k≥0} (H_k + H_{k+1} − 2γ) y^k/(k!(k+1)!)
//           with y = x²/4, I₀ = Σ y^k/(k!)², I₁ = (x/2) Σ y^k/(k!(k+1)!) and H_k
//           the harmonic numbers (H₀ = 0).  15 terms: y^14/(14!)² < 1e-21.
//   x > 2 : √x e^x K_ν(x) is smooth in 1/x (→ √(π/2) as x → ∞), so it is a
//           Chebyshev series in ξ = 4/x − 1 ∈ (−1, 1) — the split and variable
//           of Cephes' k0e/k1e.  The coefficients are projected once per
//           process, at first use, from bessel_k_scaled (an integral
//           representation); they fall below 1e-16 by order ~21.
//
// Checked against libstdc++'s std::cyl_bessel_k (Temme's method, an
// independent algorithm) and the large-x asymptotic series in
// tests/test_zero_energy_jastrow.cpp.
class ScaledBesselK01 {
public:
    static constexpr int kTerms = 24;  // Chebyshev coefficients per order

    // The process-wide table (built on first use; thread-safe initialization).
    [[nodiscard]] static const ScaledBesselK01& table() {
        static const ScaledBesselK01 instance;
        return instance;
    }

    // k0e = e^x K₀(x), k1e = e^x K₁(x).
    void evaluate(double x, double& k0e, double& k1e) const noexcept {
        if (x <= 2.0) {
            series(x, k0e, k1e);
            return;
        }
        // Clenshaw recurrence for Σ' c_k T_k(ξ), both orders in one pass.
        const double xi = 4.0 / x - 1.0, two_xi = 2.0 * xi;
        double b0 = 0.0, b0_prev = 0.0, b1 = 0.0, b1_prev = 0.0;
        for (int k = kTerms - 1; k >= 1; --k) {
            const double t0 = two_xi * b0 - b0_prev + c0_[k];
            const double t1 = two_xi * b1 - b1_prev + c1_[k];
            b0_prev = b0; b0 = t0;
            b1_prev = b1; b1 = t1;
        }
        const double inv_sqrt_x = 1.0 / std::sqrt(x);
        k0e = (xi * b0 - b0_prev + c0_[0]) * inv_sqrt_x;
        k1e = (xi * b1 - b1_prev + c1_[0]) * inv_sqrt_x;
    }

private:
    ScaledBesselK01() {
        // Chebyshev projection on M first-kind nodes ξ_j = cos θ_j, oversampled
        // (M ≫ kTerms) so the discrete coefficients equal the continuous ones
        // to below the series' own truncation.
        constexpr int M = 8 * kTerms;
        std::array<double, M> f0{}, f1{}, theta{};
        for (int j = 0; j < M; ++j) {
            theta[j] = std::numbers::pi * (j + 0.5) / M;
            const double x = 4.0 / (std::cos(theta[j]) + 1.0);
            f0[j] = std::sqrt(x) * bessel_k_scaled(0.0, x);
            f1[j] = std::sqrt(x) * bessel_k_scaled(1.0, x);
        }
        for (int k = 0; k < kTerms; ++k) {
            double s0 = 0.0, s1 = 0.0;
            for (int j = 0; j < M; ++j) {
                const double c = std::cos(k * theta[j]);
                s0 += f0[j] * c;
                s1 += f1[j] * c;
            }
            const double norm = (k == 0 ? 1.0 : 2.0) / M;
            c0_[k] = norm * s0;
            c1_[k] = norm * s1;
        }
    }

    static void series(double x, double& k0e, double& k1e) noexcept {
        constexpr double euler_gamma = std::numbers::egamma;
        const double y = 0.25 * x * x;
        double term0 = 1.0;   // y^k/(k!)²
        double term1 = 1.0;   // y^k/(k!(k+1)!)
        double harmonic = 0.0;  // H_k
        double i0 = 0.0, s0 = 0.0, i1 = 0.0, s1 = 0.0;
        for (int k = 0; k < 15; ++k) {
            if (k > 0) {
                term0 *= y / (static_cast<double>(k) * k);
                term1 *= y / (static_cast<double>(k) * (k + 1));
                harmonic += 1.0 / k;
            }
            const double harmonic_next = harmonic + 1.0 / (k + 1);
            i0 += term0;
            s0 += harmonic * term0;
            i1 += term1;
            s1 += (harmonic + harmonic_next - 2.0 * euler_gamma) * term1;
        }
        const double log_half_x = std::log(0.5 * x);
        const double k0 = -(log_half_x + euler_gamma) * i0 + s0;
        const double k1 = 1.0 / x + log_half_x * (0.5 * x * i1) - 0.25 * x * s1;
        const double e_x = std::exp(x);
        k0e = e_x * k0;
        k1e = e_x * k1;
    }

    std::array<double, kTerms> c0_{}, c1_{};  // [Chebyshev order]
};

}  // namespace alea::numerics
