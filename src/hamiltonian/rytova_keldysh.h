// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "hamiltonian_component.h"
#include "../population/particle_group.h"

namespace alea {

class WalkerSet;

namespace numerics { struct KeldyshChebyshev; }

// Rytova–Keldysh interaction between two particle groups (atomic units, e² = 1).
//
//   V_ab(r) = (π q_a q_b) / (2 ε r₀) · [ H₀(r/r₀) − Y₀(r/r₀) ]
//
// H₀ is the Struve function of order 0, Y₀ the Bessel function of the second
// kind of order 0, and r₀ the 2D screening length.  The sign is carried by the
// charge product q_a q_b, exactly like the Coulomb term.  For r ≫ r₀ this
// recovers the Coulomb 1/r tail; near the origin the divergence is only
// logarithmic.
//
//   Intra-group (a == b):  V = Σ_{i<j} (π q²)/(2 ε r₀) · K(r_ij / r₀)
//   Inter-group (a != b):  V = Σ_{i,j} (π q_a q_b)/(2 ε r₀) · K(r_ij / r₀)
class KeldyshTerm : public HamiltonianComponent {
public:
    // d_layer: out-of-plane separation between the two groups' planes.  A pair's
    // 3D distance is √(r_in-plane² + d_layer²); d_layer = 0 (default) → strictly
    // 2D, exactly the original behaviour.  Lets a bilayer e–h exciton be solved
    // in real coordinates.
    KeldyshTerm(ParticleGroup group_a,
                ParticleGroup group_b,
                double        epsilon,
                double        r0,
                double        d_layer = 0.0);

    std::string name() const override { return "rytova_keldysh"; }
    PairActionRole pair_action_role() const noexcept override { return PairActionRole::radial_pairs; }
    std::vector<RadialPairPotential> radial_pair_potentials() const override;

    bool supports_potential_gradient() const noexcept override { return true; }
    // The admitted identity uses the continuous, lower-bounded SAMPLED kernel:
    // frozen low core, fitted middle, C2 blend and inverse-power tail. No new
    // regularization or singular-attraction primitive admission is introduced.
    // Physical-reference budgets are recorded in docs/estimators.md.
    bool supports_open_centroid_virial() const noexcept override;
    bool supports_periodic_virial(const SimulationCell&, std::size_t) const noexcept override;
    std::string potential_approximation_notes() const override;
    std::unique_ptr<PotentialGradientWorkspace>
    make_potential_gradient_workspace(const ParticleBatch&) const override;
    // G_i=q_i*q_j*prefactor*K'(rho/r0)*Delta/(r0*rho), G_j=-G_i.
    // Differentiate the fitted/clamped branches and their C² tail transition.
    // Gradient capability alone does not establish a virial estimator identity.
    void evaluate_potential_gradient(const ParticleBatch&, std::span<double>,
        PotentialGradientWorkspace&, std::size_t = 0, std::size_t = npos) const override;

    bool is_translation_invariant() const noexcept override { return true; }
    bool primitive_pair_admissible(const ParticleGroup& a,
                                   const ParticleGroup& b) const noexcept override {
        const bool matches = (a == group_a_ && b == group_b_) ||
                             (a == group_b_ && b == group_a_);
        return !matches || d2_ > 0.0 || group_a_.charge * group_b_.charge >= 0.0;
    }

    double delta_potential_particle(ParticleBatch&, std::size_t, std::size_t,
        std::span<const double>, std::span<double>) const override;

    void evaluate_potential(const ParticleBatch& batch, std::span<double> v_out,
                            std::size_t w_begin = 0,
                            std::size_t w_end = npos) const override;

    [[nodiscard]] std::unique_ptr<HamiltonianComponent> clone() const override {
        return std::make_unique<KeldyshTerm>(*this);
    }

private:
    ParticleGroup group_a_;
    ParticleGroup group_b_;
    double        epsilon_;
    double        r0_;
    double        inv_r0_;     // 1 / r₀
    double        prefactor_;  // π / (2 ε r₀)
    double        d2_;         // layer offset squared (0 → strictly 2D)
    bool          is_intragroup_;

    // Shared, read-only Chebyshev kernel K(x) = H₀(x) − Y₀(x).  Points at a
    // process-wide static (see numerics::keldysh_cheb), so the default copy used
    // by clone() simply duplicates the pointer.
    const numerics::KeldyshChebyshev* cheb_;
};

}  // namespace alea
