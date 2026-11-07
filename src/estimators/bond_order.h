// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "estimators/estimator.h"
#include "population/particle_group.h"
#include "population/simulation_cell.h"

namespace alea {

// Six-nearest-neighbor diagnostic for a periodic 2D particle group:
// psi6(i) = (1/6) sum_j w_ij exp(6 i theta_ij), sum_j w_ij=6.
// Outputs <|mean_i psi6(i)|^2>, <mean_i |psi6(i)|^2>, and
// <(1/6N) sum_ij w_ij (r_ij/a_ref-1)^2>. The last is bond-length
// distortion, not site displacement or a universal Lindemann criterion.
// Neighbor distances use minimum images; cutoff ties share the remaining
// neighbor weight equally, preserving particle-label invariance.
// Nelson & Halperin, Phys. Rev. B 19, 2457 (1979), sixfold orientational
// order; this specified six-nearest convention is not a Voronoi topology.
// All three values and their signed denominator keep synchronized blocks.
// A diagonal observable of R, so every sampler measures it: VMC in |psi_T|^2,
// DMC as the mixed estimate (walker weights), PIMC over every slice at 1/beta.
// PIMC reblocks replica-synchronous histories; VMC/DMC keep constant-space
// independent-block moments (BlockMoments).
class BondOrder final : public Estimator {
public:
    BondOrder(std::span<const ParticleGroup> groups, const std::string& group,
              const SimulationCell& cell, double reference_spacing);
    void configure_reblocking(std::size_t blocks) override;
    void configure_signed_weights() override { reblocking_.configure_signed(); }
    [[nodiscard]] std::size_t reblocking_storage_bytes(std::size_t blocks) const override;
    void skip_measurement(std::size_t n) override;
    void start_block() override;
    using Estimator::accumulate;
    void accumulate(const ParticleBatch&, std::span<const double> weights) override;
    void stop_block() override;
    void collect(const Estimator&) override;
    void finalize(RunResult&) const override;
    [[nodiscard]] std::unique_ptr<Estimator> clone() const override {
        return std::make_unique<BondOrder>(*this);
    }
    [[nodiscard]] std::string type() const override { return "bond_order"; }
    // bond_<group>_global6, _local6, _distortion, _reference_spacing, ...
    [[nodiscard]] std::vector<std::string> output_namespaces() const override {
        return {"bond_" + group_};
    }
private:
    std::string group_;
    std::size_t first_ = 0, last_ = 0;
    SimulationCell cell_;
    double spacing_ = 0.;
    std::vector<double> dx_, dy_, distance2_;  // [group particle], clone-local scratch
    std::array<double, 3> accumulated_{};  // [global psi6^2, local psi6^2, distortion]
    double weight_ = 0., absolute_weight_ = 0.;
    EstimatorReblocking reblocking_;  // PIMC: [block, the three joint observables]
    BlockMoments moments_;            // VMC/DMC: [the three joint observables]
};

}  // namespace alea
