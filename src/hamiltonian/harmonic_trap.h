// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "hamiltonian_component.h"
#include "../population/particle_group.h"

namespace alea {

// Isotropic harmonic one-body potential (a parabolic quantum dot):
//
//   V = Σ_i Σ_d ½ m_{i,d} ω² (r_{i,d} − c_d)²
//
// ω² is taken signed on purpose: ω² < 0 is the centrifugal term −½mΩ²r² of a
// frame rotating at Ω (see GaugeField), so the rotating frame is a trap plus a
// vector potential with no special-case code.  Per-dimension mass keeps an
// anisotropic effective mass consistent with Kinetic.
class HarmonicTrap final : public HamiltonianComponent {
public:
    // omega2 = ω² (may be negative); center: [dim], default the origin.
    HarmonicTrap(const std::vector<ParticleGroup>& groups, double omega2,
                 std::vector<double> center = {});

    std::string name() const override { return "harmonic_trap"; }
    PairActionRole pair_action_role() const noexcept override { return PairActionRole::one_body; }

    bool supports_potential_gradient() const noexcept override { return true; }
    bool supports_open_centroid_virial() const noexcept override;
    bool confines_open_coordinate(std::size_t p, std::size_t d,
                                  std::size_t dim) const noexcept override;
    std::unique_ptr<PotentialGradientWorkspace>
    make_potential_gradient_workspace(const ParticleBatch&) const override;
    // G_{i,d}=m_{i,d} omega^2 (r_{i,d}-c_d), including signed curvature.
    void evaluate_potential_gradient(const ParticleBatch&, std::span<double>,
        PotentialGradientWorkspace&, std::size_t = 0, std::size_t = npos) const override;

    double delta_potential_particle(ParticleBatch&, std::size_t, std::size_t,
        std::span<const double>, std::span<double>) const override;

    void evaluate_potential(const ParticleBatch& batch, std::span<double> v_out,
                            std::size_t w_begin = 0,
                            std::size_t w_end = npos) const override;

    [[nodiscard]] std::unique_ptr<HamiltonianComponent> clone() const override {
        return std::make_unique<HarmonicTrap>(*this);
    }

private:
    std::vector<double> half_k_;   // [np * dim] ½ m_{i,d} ω²  (0: not trapped)
    std::vector<double> center_;   // [dim]
};

}  // namespace alea
