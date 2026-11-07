// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "estimators/estimator.h"
#include "population/particle_group.h"
#include "population/simulation_cell.h"

namespace alea {

// Selected-star crystal order B = (1/n_G) sum_G <|sum_j exp(i G.r_j)|^2>/N^2.
// This is the full equal-time intensity, invariant under global translation,
// not |<rho>|^2. See Drummond & Needs, PRL 102, 126402 (2009), crystal geometry;
// Flyvbjerg & Petersen, JCP 91, 461 (1989), for block-error analysis.
// Each supplied Cartesian G must be a nonzero reciprocal vector of the cell.
// Supply one member of each +/- pair; duplicate intensities are refused.
// A diagonal observable of R, so every sampler measures it: VMC in |psi_T|^2,
// DMC as the mixed estimate <psi_T|B|Phi>/<psi_T|Phi> (walker weights), PIMC
// over every slice at temperature 1/beta.  Each block combines directions
// BEFORE its error analysis, so directional covariance is retained in the
// scalar B: PIMC reblocks replica-synchronous histories, VMC/DMC keep
// constant-space independent-block moments (BlockMoments).
class BraggOrder final : public Estimator {
public:
    BraggOrder(std::span<const ParticleGroup> groups, const std::string& group,
               const SimulationCell& cell, std::span<const double> wavevectors);
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
        return std::make_unique<BraggOrder>(*this);
    }
    [[nodiscard]] std::string type() const override { return "bragg"; }
    // bragg_<group>, bragg_<group>_directions, _kx, _ky, ...
    [[nodiscard]] std::vector<std::string> output_namespaces() const override {
        return {"bragg_" + group_};
    }
private:
    std::string group_;
    std::size_t first_ = 0, last_ = 0;
    std::vector<double> kx_, ky_;  // [direction], fixed at construction
    std::vector<double> accumulated_;  // [direction], no measurement allocation
    std::vector<double> block_values_; // [mean-star + direction intensities], per-block scratch
    double weight_ = 0., absolute_weight_ = 0.;
    EstimatorReblocking reblocking_;  // PIMC: [block, mean-star + direction intensities]
    BlockMoments moments_;            // VMC/DMC: [mean-star + direction intensities]
};

}  // namespace alea
