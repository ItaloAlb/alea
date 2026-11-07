// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "population/particle_group.h"
#include "wavefunction_component.h"

namespace alea {

// Galilean boost of the whole system by total momentum K (a pure phase):
//
//   ψ_K = e^{iK·R_cm} ψ₀,     R_cm_d = Σ_i m_{i,d} r_{i,d} / M_d,   M_d = Σ_i m_{i,d}
//
// so φ = Σ_i k_i·r_i with k_{i,d} = m_{i,d} K_d / M_d (per dimension, so an
// anisotropic mass tensor is boosted correctly).  Then ∇_i φ = k_i, ∇²φ = 0 and
// |ψ_K| = |ψ₀|: the modulus planes are untouched.
//
// For a translation-invariant H (no one-body potential) the centre of mass
// separates, and H ψ_K = (E₀ + Σ_d K_d²/2M_d) ψ_K exactly whenever H ψ₀ = E₀ ψ₀.
// The phase is therefore EXACT, and fixed-phase DMC on ψ_K must return
// E₀ + K²/2M — the exact oracle the fixed-phase machinery is tested against
// (Ortiz, Ceperley & Martin, PRL 71, 2777 (1993): E_FP = E₀ when φ is exact).
// Physically it gives the dispersion of a moving complex (exciton, trion, ...).
//
// Open cells only: on a periodic cell e^{ik_i·r_i} is single-valued only for
// k_i on the reciprocal lattice, which the config layer must enforce.
class CenterOfMassMomentum final : public WavefunctionComponent {
public:
    // K: [dim].  groups: every particle of the system (defines m_i and M).
    CenterOfMassMomentum(const std::vector<ParticleGroup>& groups,
                         std::vector<double> K);

    std::string name() const override { return "center_of_mass_momentum"; }

    [[nodiscard]] bool is_complex() const override { return true; }

    void evaluate_batch(WalkerSet& batch) override;
    void ratio_batch(WalkerSet& batch, std::size_t p, std::size_t w_begin = 0,
                     std::size_t w_end = npos) override;
    void apply_move_batch(WalkerSet& batch, std::size_t p,
                          std::span<const bool> accept_mask,
                          std::size_t w_begin = 0,
                          std::size_t w_end = npos) override;

    [[nodiscard]] std::unique_ptr<WavefunctionComponent> clone() const override {
        return std::make_unique<CenterOfMassMomentum>(*this);
    }

    // K is a quantum number, not a variational parameter.
    std::span<Parameter>       parameters() override { return {}; }
    std::span<const Parameter> parameters() const override { return {}; }
    void rebuild() override {}
    void compute_parameter_derivatives(const WalkerSet&,
                                       std::span<double>) const override {}

    // k_{i,d} = ∇_{i,d} φ, [np * dim].
    [[nodiscard]] std::span<const double> particle_wavevectors() const noexcept {
        return k_;
    }

private:
    std::size_t         dim_;
    std::vector<double> K_;   // [dim]
    std::vector<double> k_;   // [np * dim]
};

}  // namespace alea
