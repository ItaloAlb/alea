// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Gaskell's RPA pair pseudopotential, the seed of the long-range Jastrow.
//
// For Ψ = exp(−Σ_{i<j} u_C(r_ij)) (Ceperley's sign convention), writing the
// variational energy in k space and closing it with the RPA relation
// S(k)⁻¹ = S₀(k)⁻¹ + 2ρ ũ_C(k) gives (Gaskell, Proc. Phys. Soc. 77, 1182
// (1961); Ceperley, "Introduction to QMC methods applied to the electron gas",
// eq. 14):
//
//     2ρ ũ_C(k) = −1/S₀(k) + [ 1/S₀(k)² + 2ρ ṽ(k) / (λ k²) ]^{1/2},   λ = ħ²/2m,
//
// with ρ the number density, ṽ(k) the 2-D Fourier transform of the pair
// potential and S₀ the structure factor without correlation.  For 2-D Coulomb
// it reproduces the plasmon: S(k) → k²/(2mω_p), ω_p = (2πρk/m)^{1/2}, i.e.
// S ∝ k^{3/2} — "not only the power, but also the coefficient is exactly
// correct" (Ceperley).  ṽ is whatever potential the Hamiltonian holds, so the
// same formula seeds Rytova–Keldysh or screened pairs.

#include <algorithm>
#include <cmath>
#include <numbers>

namespace alea::numerics {

// ũ_C(k) of the Gaskell form (Ceperley sign: Ψ = exp(−Σ u_C)).
[[nodiscard]] inline double gaskell_u(double k, double rho, double s0, double v_k,
                                      double inv_mass) noexcept {
    const double lambda = 0.5 * inv_mass;
    const double a = 1.0 / s0;
    return (-a + std::sqrt(a * a + 2.0 * rho * v_k / (lambda * k * k))) / (2.0 * rho);
}

// The RPA structure factor that ũ_C implies: S⁻¹ = S₀⁻¹ + 2ρ ũ_C.
[[nodiscard]] inline double gaskell_sofk(double k, double rho, double s0, double v_k,
                                         double inv_mass) noexcept {
    return 1.0 / (1.0 / s0 + 2.0 * rho * gaskell_u(k, rho, s0, v_k, inv_mass));
}

// S₀(k) of one spin species of free fermions in 2-D at density n_σ
// (k_F = (4π n_σ)^{1/2}), x = k / 2k_F:
//     S₀ = (2/π)[asin x + x (1 − x²)^{1/2}]  for x ≤ 1,   1 beyond.
[[nodiscard]] inline double free_fermion_sofk_2d(double k, double n_sigma) noexcept {
    if (n_sigma <= 0.0) return 1.0;
    const double x = k / (2.0 * std::sqrt(4.0 * std::numbers::pi * n_sigma));
    if (x >= 1.0) return 1.0;
    return (2.0 / std::numbers::pi) * (std::asin(x) + x * std::sqrt(1.0 - x * x));
}

// S₀ of a group with n_up, n_down free fermions in area `area`: each species
// contributes its own S₀ weighted by its share of the particles (the cross-
// species exchange hole vanishes).
[[nodiscard]] inline double free_fermion_sofk_2d(double k, int n_up, int n_down,
                                                 double area) noexcept {
    const double n = n_up + n_down;
    if (n <= 0) return 1.0;
    return (n_up * free_fermion_sofk_2d(k, n_up / area) +
            n_down * free_fermion_sofk_2d(k, n_down / area)) / n;
}

}  // namespace alea::numerics
