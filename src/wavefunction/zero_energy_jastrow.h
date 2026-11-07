// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "radial_pair_jastrow.h"
#include "../numerics/scaled_bessel_k01.h"
#include "../population/particle_group.h"

namespace alea {

// Pair function of a 2-D pair with a repulsive core V(r) ≈ C/r^p, p > 2: the
// exact two-body scattering solution at zero energy inside a matching radius
// R_m, joined to a phonon-like tail that flattens to u = 0 at the cutoff r_c
// (G. E. Astrakharchik, J. Boronat, I. L. Kurbakov, Yu. E. Lozovik,
//  Phys. Rev. Lett. 98, 060405 (2007), for dipoles, p = 3):
//
//   f(r) = e^{u(r)} =  C₁ K₀(κ r^{−s})                     r < R_m
//                      C₂ exp(−C₃/r − C₃/(2r_c − r))       R_m ≤ r < r_c
//                      1                                   r ≥ r_c
//
//   s = (p − 2)/2,   κ = √b / s,   b = 2μC   (μ the pair's reduced mass)
//
// The first line solves the relative zero-energy equation
//   −(1/2μ)(f″ + f′/r) + (C/r^p) f = 0,  i.e.  f″ + f′/r = b r^{−p} f:
// with x = κ r^{−s}, the modified-Bessel equation of order 0 gives
// f″ + f′/r = s²κ² r^{−2s−2} K₀(x) = b r^{−p} f.  So inside R_m the pair's
// kinetic energy cancels C/r^p exactly — the analogue of the Kato cusp for a
// core stronger than 1/r², where ψ must vanish as exp(−κ r^{−s}).  (A Padé or
// Drummond u stays finite at r = 0 and leaves ⟨C/r^p⟩ divergent for p ≥ 2.)
// For p = 3, b = r₀ this is the paper's K₀(2√(r₀/r)).
//
// The constants follow from three conditions: u and u′ continuous at R_m,
// u(r_c) = 0 (the symmetric tail already has u′(r_c) = 0):
//   C₃   = u₀′(R_m) / (1/R_m² − 1/(2r_c − R_m)²),   u₀ = ln K₀(κ r^{−s})
//   ln C₂ = 2C₃/r_c
//   ln C₁ = ln C₂ − C₃/R_m − C₃/(2r_c − R_m) − u₀(R_m)
// u″ jumps at R_m and at r_c, so E_L has finite steps there (as in the paper);
// ψ itself is C¹.  With q = K₁(x)/K₀(x), inside R_m:
//   u′ = s x q / r,   u″ = (s x / r²) [s x (1 − q)(1 + q) − q],
// using K₀′ = −K₁ and K₀″ = K₀ + K₁/x; then u″ + u′/r + u′² = s²x²/r² = b r^{−p}.
//
// The one variational parameter is R_m.  Its derivative needs no derivative
// of the Bessel part: the matching condition cancels it, leaving
//   ∂u/∂R_m = C₃′ φ(max(r, R_m)) for r < r_c (0 beyond),
//   φ(r) = 2/r_c − 1/r − 1/(2r_c − r),
//   C₃′ = (u₀″ h − u₀′ h′)/h² at R_m,   h = 1/r² − 1/(2r_c − r)².
class ZeroEnergyFunction {
public:
    // b = 2μC > 0, exponent p > 2, 0 < matching_radius < cutoff (finite).
    ZeroEnergyFunction(double b, double exponent, double matching_radius, double cutoff);

    // u(r), u′(r), u″(r) at one distance r > 0.
    void evaluate(double r, double& u, double& du, double& d2u) const noexcept;

    // Batched over a contiguous vector of distances; outputs written, not
    // accumulated, each r.size() long.
    void evaluate(std::span<const double> r, std::span<double> u,
                  std::span<double> du, std::span<double> d2u) const noexcept;

    // ∂u/∂R_m and its first two r-derivatives at r.
    void matching_radius_derivative(double r, double& du_dp, double& d_du_dp,
                                    double& d_d2u_dp) const noexcept;

    [[nodiscard]] double matching_radius() const noexcept { return r_m_; }
    [[nodiscard]] double cutoff() const noexcept { return r_c_; }

private:
    // Inside R_m without ln C₁: u₀ = ln K₀(x), u₀′, u₀″.
    void zero_energy_solution(double r, double& u0, double& du0,
                              double& d2u0) const noexcept;

    double s_;       // (p − 2)/2
    double kappa_;   // √b / s
    double r_m_;     // matching radius R_m
    double r_c_;     // cutoff r_c (u = 0 beyond)
    double c3_;      // C₃
    double log_c1_;  // ln C₁
    double log_c2_;  // ln C₂
    double dc3_;     // dC₃/dR_m
    const numerics::ScaledBesselK01* bessel_;
};

// Zero-energy pair Jastrow log ψ = Σ u(r_ij) over the pairs of
// (group_a, group_b), evaluated by RadialPairJastrow.
//
// params layout: [R_m].  b, p and r_c are fixed at construction: b and p come
// from the Hamiltonian's core on this pair (HamiltonianComponent::
// power_law_core), r_c is the Wigner–Seitz inradius of a periodic cell, where
// the minimum image is the exact image sum because u is flat beyond it.
extern template class RadialPairJastrow<ZeroEnergyFunction>;

class ZeroEnergyJastrow final : public RadialPairJastrow<ZeroEnergyFunction> {
public:
    ZeroEnergyJastrow(double b, double exponent, double cutoff,
                      ParticleGroup group_a, ParticleGroup group_b,
                      std::vector<Parameter> params);

    std::string name() const override { return "zero_energy_jastrow"; }

    [[nodiscard]] std::unique_ptr<WavefunctionComponent> clone() const override {
        return std::make_unique<ZeroEnergyJastrow>(*this);
    }

    std::span<Parameter>       parameters()       override { return params_; }
    std::span<const Parameter> parameters() const override { return params_; }

    // Throws if R_m left (0, r_c): the matching needs R_m inside the cutoff.
    void rebuild() override;

    void compute_parameter_derivatives(const WalkerSet& batch,
                                       std::span<double> derivs) const override;

    [[nodiscard]] bool supports_parameter_gradient_derivatives() const override {
        return true;
    }
    bool compute_parameter_gradient_derivatives(const WalkerSet& batch,
                                                std::span<double> dgrad,
                                                std::span<double> dlap) const override;

    [[nodiscard]] double core_strength() const noexcept { return b_; }  // b = 2μC
    [[nodiscard]] double exponent() const noexcept { return exponent_; }
    [[nodiscard]] double cutoff() const noexcept { return cutoff_; }

private:
    double b_;
    double exponent_;
    double cutoff_;
    std::vector<Parameter> params_;  // [R_m]; radial_ is rebuilt from it
};

}  // namespace alea
