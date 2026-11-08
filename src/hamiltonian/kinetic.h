// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <span>
#include <vector>

#include "hamiltonian_component.h"
#include "../population/particle_group.h"

namespace alea {

// Kinetic energy term (supports anisotropic effective mass):
//   K_w = -½ Σ_i Σ_d (1/m_i_d) [ ∂²_d log ψ(R_w) + (∂_d log ψ(R_w))² ]
//
// Reads batch.gradients() and batch.laplacians(), both per-dimension.
// inv_mass[d] from the particle groups provides the per-dimension 1/m.
//
// For a complex trial ψ = e^{u + iφ} (batch.has_phase()) it returns the REAL
// part, the fixed-phase local energy, which adds +½ Σ (1/m)(∂_d φ)², and
// writes Im E_L to batch.local_energies_imag() (see add_phase_terms).
// Uniform external fields entering the kinetic energy through a vector
// potential, both along ẑ (2-D, perpendicular to the plane):
//   B      magnetic field; symmetric gauge A = ½ B ẑ×r about the origin, and
//          particle i couples through its charge: (p − q_i A)²/2m_i.
//   omega  angular velocity of a rotating frame: H − Ω L_z =
//          Σ (p − m_i Ω ẑ×r)²/2m_i − ½ m_i Ω² r², i.e. the same form with
//          coupling m_i Ω; the −½mΩ²r² piece is a separate HarmonicTrap term.
// Both add to one per-particle vector potential a_i = c_i ẑ×r_i with
// c_i = q_i B/2 + m_i Ω.
struct GaugeField {
    double B     = 0.0;
    double omega = 0.0;
    [[nodiscard]] bool active() const noexcept { return B != 0.0 || omega != 0.0; }
};

class Kinetic : public HamiltonianComponent {
public:
    explicit Kinetic(std::vector<ParticleGroup> groups, GaugeField gauge = {});

    std::string name() const override { return "kinetic"; }

    // Reads ∇ log ψ and ∇² log ψ, so it is not a function of R alone.
    [[nodiscard]] bool is_potential() const noexcept override { return false; }

    void evaluate_batch(WalkerSet& batch,
                        std::size_t w_begin = 0,
                        std::size_t w_end = npos) override;

    // ∂K/∂p_k = -½ Σ_i Σ_d (1/m_i_d) [ ∂_p(∂²_d) + 2 (∂_d logψ) ∂_p(∂_d) ].
    void local_energy_parameter_derivatives(
        const WalkerSet& batch, std::span<const double> dgrad,
        std::span<const double> dlap, std::size_t n_opt,
        std::span<double> out) const override;

    // Complex trial; see the derivation at the definition.
    void complex_local_energy_parameter_derivatives(
        const WalkerSet& batch, std::span<const double> dgrad, std::span<const double> dlap,
        std::span<const double> dpgrad, std::span<const double> dplap, std::size_t n_opt,
        std::span<double> out_re, std::span<double> out_im) const override;

    // K_w into out[w] (accumulated), read-only on the batch.
    void accumulate_psi_dependent_energy(const WalkerSet& batch,
                                         std::span<double> out) const override;

    [[nodiscard]] std::unique_ptr<HamiltonianComponent> clone() const override {
        return std::make_unique<Kinetic>(*this);
    }

private:
    // Phase part of the kinetic energy: +½(1/m)(∂φ)² into re[w], and Im E_L
    // into im[w] when im != nullptr, over [w_begin, w_end).
    void add_phase_terms(const WalkerSet& batch, double* __restrict re,
                         std::size_t w_begin, std::size_t w_end,
                         double* __restrict im) const;

    // Same, with the vector potential: (∂φ − a)² in place of (∂φ)², where the
    // phase planes may be absent (real trial → ∂φ = 0).
    void add_gauge_terms(const WalkerSet& batch, double* __restrict re,
                         std::size_t w_begin, std::size_t w_end,
                         double* __restrict im) const;

    // Per-particle gauge coupling c_i (a_i = c_i ẑ×r_i); empty when no field.
    std::vector<double> gauge_coupling_;

    // Per-particle, per-dimension inverse mass [i * dim + d].
    std::vector<double> inv_mass_;
};

}  // namespace alea
