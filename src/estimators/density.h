// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "estimators/estimator.h"
#include "population/particle_group.h"
#include "population/simulation_cell.h"

namespace alea {

struct RunResult;

// One-body spatial density ρ_g(r) per particle group, histogrammed on a
// cartesian grid [−rmax, +rmax]^dim with n_bins per axis (lab frame, origin
// centred).  This is the walker cloud made visible: in VMC the walkers sample
// |ψ_T|², so ρ = ⟨ψ_T|ρ̂|ψ_T⟩; in DMC accumulate() runs pre-branch with the
// branch weights, so ρ is the mixed estimator ⟨ψ_T|ρ̂|Φ⟩/⟨ψ_T|Φ⟩ and the pure
// density can be extrapolated in post-processing as ρ_pure ≈ 2ρ_DMC − ρ_VMC
// (Ceperley & Kalos 1979; Foulkes et al., Rev. Mod. Phys. 73, 33 (2001)).
//
// Normalization: each block's cell estimate is counts/(W_block·V_cell) with
// W_block = Σ config weights and V_cell = Δ^dim, so ∫ρ_g dV equals the mean
// number of group-g particles inside the grid (= the group count when nothing
// leaves the box).  Out-of-grid particles drop from the numerator only — the
// config weight still enters W_block, since it is the measure of ⟨ψ_T|Φ⟩.
//
// Statistics: unlike PairCorrelation's per-block history (which for a 2-D
// grid would cost n_blocks × G·n_bins² doubles ≈ 0.5 GB at 2048 × 128² × 2),
// per-block estimates are streamed into per-cell Welford accumulators
// (mean_, m2_, n_blocks_); collect() merges clones with the Chan et al.
// parallel update.  Both are algebraically equivalent to
// numerics::block_mean_error over the full block history.
//
// Lab-frame caveat: for a free complex the centre of mass diffuses and the
// density smears toward uniform — pin a species (e.g. fixed holes) or a
// one-body potential for a meaningful image.
class Density : public Estimator {
public:
    // Throws std::runtime_error if n_bins^dim exceeds 2^24 cells per group.
    // per_particle: one histogram per particle, named "<group>_p<i>" (i local
    // to the group), instead of one per group.  Labels of same-spin particles
    // are exchangeable in the sampling, so per-particle densities are physical
    // only through channel sums/differences (e.g. lithium 1s²2s¹: the up
    // channel ρ_↑ = p0+p1 holds |1s|²+|2s|², the lone down electron ρ_↓ = p2
    // is pure |1s|², and ρ_↑ − ρ_↓ ≈ |2s|² isolates the excited orbital).
    Density(std::span<const ParticleGroup> groups, std::size_t dim,
            double rmax, int n_bins, bool per_particle = false,
            const SimulationCell& cell = SimulationCell{});

    void configure_reblocking(std::size_t num_blocks) override;
    void configure_signed_weights() override { reblocking_.configure_signed(); }
    // Worm runs: an open path is a measurement of n zero weights; it counts
    // in the signed normalization without being binned.
    void skip_measurement(std::size_t n) override {
        if (!reblocking_.signed_mode())
            throw std::logic_error("density: skip_measurement needs signed weights");
        block_abs_weight_sum_ += static_cast<double>(n);
    }
    [[nodiscard]] std::size_t reblocking_storage_bytes(std::size_t num_blocks) const override;
    void start_block() override;
    using Estimator::accumulate;
    void accumulate(const ParticleBatch& configs,
                    std::span<const double> weights) override;
    void stop_block() override;
    void collect(const Estimator& other) override;
    void finalize(RunResult& out) const override;

    [[nodiscard]] std::unique_ptr<Estimator> clone() const override;
    [[nodiscard]] std::string type() const override { return "density"; }
    // density_edges, density_<group>[_p<i>][_error], grid scalars.
    [[nodiscard]] std::vector<std::string> output_namespaces() const override {
        return {"density"};
    }

    // ── Introspection (used by callers / tests) ──────────────────────────
    std::size_t n_cells()     const noexcept { return n_cells_; }  // per group
    std::size_t n_bins()      const noexcept { return n_bins_; }
    std::size_t n_blocks()    const noexcept { return n_blocks_; }
    double      cell_volume() const noexcept { return cell_volume_; }

private:
    struct GroupRange {
        std::string name;
        std::size_t first, last;
    };
    std::vector<GroupRange> groups_;

    std::size_t dim_;
    std::size_t n_bins_;
    std::size_t n_cells_;      // n_bins^dim, per group
    std::size_t n_groups_;
    double      rmax_;
    double      delta_;        // 2·rmax / n_bins
    double      inv_delta_;
    double      cell_volume_;  // Δ^dim (open) or V_cell/n_bins^dim (periodic)
    // Periodic runs bin fractional coordinates over the actual cell instead of
    // a fixed [−rmax, rmax]^dim cartesian window, so no particle is ever
    // outside the grid and the bin volume follows the cell's shape.
    SimulationCell cell_;

    // Current-block accumulation (flat: [g * n_cells + cell]).
    std::vector<double> block_hist_;
    double              block_weight_sum_ = 0.0;  // Σ per-config sample weight
    double              block_abs_weight_sum_ = 0.0;  // signed normalization: number of weights

    // Streaming block statistics (Welford), flat [g * n_cells + cell].
    std::size_t         n_blocks_ = 0;  // finished blocks
    std::vector<double> mean_;
    std::vector<double> m2_;
    EstimatorReblocking reblocking_; // PIMC only; legacy Welford is untouched
};

}  // namespace alea
