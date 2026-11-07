// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "hamiltonian_component.h"

namespace alea {

class WalkerSet;

// First-shell reciprocal-lattice vectors of a triangular moiré superlattice of
// period a_M, rotated by theta0.  The triangular lattice has a single first-shell
// magnitude |g| = 4π/(√3 a_M); the three vectors sit 120° apart:
//   g_i = |g| · (cos(theta0 + 120°·i), sin(theta0 + 120°·i)),  i = 0,1,2.
// Returned as dim-length vectors (components beyond x,y are zero) so they dot
// directly against the SoA particle coordinates.
[[nodiscard]] std::vector<std::vector<double>>
moire_reciprocal_vectors(double a_M, double theta0, std::size_t dim);

// One-body moiré potential (Wu–Lovorn–MacDonald continuum model, PRL 121,
// 026402 (2018)):
//
//     V(r) = V0 + 2 Ṽ Σ_{i=1}^{3} cos(g_i · r + φ)
//
// An external scalar potential felt by every particle of the listed groups
// independently (no pair coupling, no charge factor): the term adds V(r_p) to
// the local energy for each particle p in any supplied index range [first,last).
// Different species use separate MoireTerm instances with their own V0, Ṽ, φ.
class MoireTerm : public HamiltonianComponent {
public:
    // dim       : spatial dimension.
    // V0        : constant offset (added once per particle).
    // Vtilde    : modulation amplitude Ṽ.
    // phi       : phase φ.
    // g_vectors : the three reciprocal vectors, each of length dim.
    // ranges    : half-open particle index ranges [first,last) this term acts on
    //             (one per listed group).
    MoireTerm(std::size_t dim, double V0, double Vtilde, double phi,
              std::vector<std::vector<double>> g_vectors,
              std::vector<std::pair<std::size_t, std::size_t>> ranges);

    std::string name() const override { return "moire"; }
    PairActionRole pair_action_role() const noexcept override { return PairActionRole::one_body; }

    bool supports_potential_gradient() const noexcept override { return true; }
    // Finite Fourier sum: continuous and bounded; workspace validates constants.
    bool supports_open_centroid_virial() const noexcept override { return true; }
    bool supports_periodic_virial(const SimulationCell&, std::size_t) const noexcept override;
    std::unique_ptr<PotentialGradientWorkspace>
    make_potential_gradient_workspace(const ParticleBatch&) const override;
    // G_p=-2 Vtilde sum_g g sin(g.r_p+phi); V0 has zero derivative.
    void evaluate_potential_gradient(const ParticleBatch&, std::span<double>,
        PotentialGradientWorkspace&, std::size_t = 0, std::size_t = npos) const override;

    double delta_potential_particle(ParticleBatch&, std::size_t, std::size_t,
        std::span<const double>, std::span<double>) const override;

    void evaluate_potential(const ParticleBatch& batch, std::span<double> v_out,
                            std::size_t w_begin = 0,
                            std::size_t w_end = npos) const override;

    [[nodiscard]] std::unique_ptr<HamiltonianComponent> clone() const override {
        return std::make_unique<MoireTerm>(*this);
    }

private:
    std::size_t                                      dim_;
    double                                           V0_;
    double                                           two_Vtilde_;  // 2 Ṽ
    double                                           phi_;
    std::vector<std::vector<double>>                 g_;       // [3][dim]
    std::vector<std::pair<std::size_t, std::size_t>> ranges_;  // groups
};

}  // namespace alea
