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

// Coulomb interaction between two particle groups.
//
//   Intra-group (a == b):  V = Σ_{i<j} q² / (ε |r_i - r_j|)
//   Inter-group (a != b):  V = Σ_{i,j} q_a * q_b / (ε |r_i - r_j|)
class CoulombTerm : public HamiltonianComponent {
public:
    // d_layer: out-of-plane separation between the two groups' planes.  A pair's
    // 3D distance is √(r_in-plane² + d_layer²); d_layer = 0 (default) → strictly
    // 2D, exactly the original behaviour.  Lets a bilayer e–h exciton (e and h in
    // different planes) be solved in real coordinates.
    CoulombTerm(ParticleGroup group_a,
                ParticleGroup group_b,
                double        epsilon,
                double        d_layer = 0.0);

    std::string name() const override { return "coulomb"; }
    PairActionRole pair_action_role() const noexcept override { return PairActionRole::radial_pairs; }
    std::vector<RadialPairPotential> radial_pair_potentials() const override;

    bool supports_potential_gradient() const noexcept override { return true; }
    bool supports_open_centroid_virial() const noexcept override {
        return std::isfinite(epsilon_) && epsilon_ > 0.0 && std::isfinite(d2_) &&
            std::isfinite(group_a_.charge*group_b_.charge) &&
            (d2_ > 0.0 || group_a_.charge*group_b_.charge >= 0.0);
    }
    // Scalar nearest-image radius agrees on Voronoi faces. This admits the
    // sampled truncated interaction, not an Ewald sum of periodic images.
    bool supports_periodic_virial(const SimulationCell& cell, std::size_t np) const noexcept override {
        return cell.is_periodic() && supports_open_centroid_virial() &&
            std::isfinite(group_a_.charge*group_b_.charge*inv_epsilon_) &&
            group_a_.first <= group_a_.last && group_b_.first <= group_b_.last &&
            group_a_.last <= np && group_b_.last <= np;
    }
    std::unique_ptr<PotentialGradientWorkspace>
    make_potential_gradient_workspace(const ParticleBatch&) const override;
    // G_i=-q_i q_j Delta/[epsilon (|Delta|^2+d_layer^2)^(3/2)], G_j=-G_i.
    // Delta is the same minimum-image displacement used by the sampled V.
    void evaluate_potential_gradient(const ParticleBatch&, std::span<double>,
        PotentialGradientWorkspace&, std::size_t = 0, std::size_t = npos) const override;

    bool is_translation_invariant() const noexcept override { return true; }
    double delta_potential_particle(ParticleBatch&, std::size_t, std::size_t,
        std::span<const double>, std::span<double>) const override;

    void evaluate_potential(const ParticleBatch& batch, std::span<double> v_out,
                            std::size_t w_begin = 0,
                            std::size_t w_end = npos) const override;

    [[nodiscard]] std::unique_ptr<HamiltonianComponent> clone() const override {
        return std::make_unique<CoulombTerm>(*this);
    }

    // 1/r residue for the pair (a,b): q_a q_b/ε when (a,b) are this term's
    // groups (either order), else 0.  V = q_a q_b/(ε r) ⇒ lim_{r→0} r V = q_a q_b/ε.
    double coulomb_cusp_residue(const ParticleGroup& a,
                                const ParticleGroup& b) const override;

private:
    ParticleGroup group_a_;
    ParticleGroup group_b_;
    double        epsilon_;
    double        inv_epsilon_;
    double        d2_;  // layer offset squared (0 → strictly 2D)
    bool          is_intragroup_;
};

}  // namespace alea
