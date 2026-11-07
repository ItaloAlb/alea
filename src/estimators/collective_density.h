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

namespace alea {

struct RunResult;

// One term of a collective coordinate: every particle of `group` enters the
// linear combination with weight `coeff`.
struct CollectiveTerm {
    std::string group;
    double      coeff = 1.0;
};

// Density of a COLLECTIVE coordinate: per walker, the linear combination
//
//     v = Σ_terms c_t · Σ_{p ∈ group(t)} r_p            (one dim-vector)
//
// is histogrammed on a cartesian grid [−rmax, +rmax)^dim, exactly one entry
// per configuration.  Unlike Density (which bins each particle separately and
// therefore only sees marginals), this resolves correlations that live in a
// combination of positions.  Canonical choices of coefficients:
//
//   c_i = q_i             → the total-dipole (polarization) distribution
//                           P(μ), μ = Σ q_i r_i, in any system;
//   +1/−1 on two groups   → a separation / internal coordinate, e.g. the
//                           exciton internal wavefunction r_e − r_h in the
//                           lab frame;
//
// When the coefficients sum to zero (counting multiplicity), v is invariant
// under global translations, so the image stays sharp even when the complex's
// centre of mass diffuses freely — no pinning needed (contrast the lab-frame
// caveat in density.h).
//
// Sampling meaning matches Density: VMC walkers give ⟨ψ_T|δ(v̂−v)|ψ_T⟩; DMC
// accumulates pre-branch with the branch weights, giving the mixed estimator
// ⟨ψ_T|δ(v̂−v)|Φ⟩/⟨ψ_T|Φ⟩, extrapolable as ρ_pure ≈ 2ρ_DMC − ρ_VMC (Ceperley
// & Kalos 1979; Foulkes et al., Rev. Mod. Phys. 73, 33 (2001)).
//
// Normalization: one coordinate per configuration, weighted by the config
// weight, so ∫ρ_v dV = 1 when v stays inside the grid (out-of-grid drops
// from the numerator only; the weight still measures ⟨ψ_T|Φ⟩).
//
// Output keys are namespaced by the instance `name` (results_writer does not
// deduplicate, so distinct instances MUST NOT share keys): series
// cdensity_<name>, cdensity_<name>_error, cdensity_<name>_edges and scalars
// cdensity_<name>_{dim,n_bins,rmax,n_blocks}.
//
// Statistics: per-block estimates stream into per-cell Welford accumulators;
// collect() merges clones with the Chan et al. parallel update — keep in sync
// with estimators/density.cpp, whose machinery this mirrors.
class CollectiveDensity : public Estimator {
public:
    // Throws std::runtime_error on: empty name, empty terms, a term group
    // name not present in `groups`, rmax <= 0, n_bins <= 0, or n_bins^dim
    // exceeding 2^24 cells.
    CollectiveDensity(std::span<const ParticleGroup> groups, std::size_t dim,
                      double rmax, int n_bins, std::string name,
                      std::span<const CollectiveTerm> terms);

    void configure_reblocking(std::size_t num_blocks) override;
    void configure_signed_weights() override { reblocking_.configure_signed(); }
    // Worm runs: an open path is a measurement of n zero weights; it counts
    // in the signed normalization without being binned.
    void skip_measurement(std::size_t n) override {
        if (!reblocking_.signed_mode())
            throw std::logic_error("collective_density: skip_measurement needs signed weights");
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
    [[nodiscard]] std::string type() const override {
        return "collective_density";
    }
    // The instance name is the namespace: cdensity_<name>[_edges|_error|...].
    [[nodiscard]] std::vector<std::string> output_namespaces() const override {
        return {"cdensity_" + name_};
    }

    // ── Introspection (used by callers / tests) ──────────────────────────
    const std::string& name() const noexcept { return name_; }
    std::size_t n_cells()     const noexcept { return n_cells_; }
    std::size_t n_bins()      const noexcept { return n_bins_; }
    std::size_t n_blocks()    const noexcept { return n_blocks_; }
    double      cell_volume() const noexcept { return cell_volume_; }

private:
    // A term with its group resolved to the particle range [first, last).
    struct ResolvedTerm {
        double      coeff;
        std::size_t first, last;
    };

    std::string               name_;
    std::vector<ResolvedTerm> terms_;

    std::size_t dim_;
    std::size_t n_bins_;
    std::size_t n_cells_;      // n_bins^dim
    double      rmax_;
    double      delta_;        // 2·rmax / n_bins
    double      inv_delta_;
    double      cell_volume_;  // Δ^dim

    // Current-block accumulation (one grid — one coordinate per walker).
    std::vector<double> block_hist_;
    double              block_weight_sum_ = 0.0;  // Σ per-config sample weight
    double              block_abs_weight_sum_ = 0.0;  // signed normalization: number of weights

    // Streaming block statistics (Welford).
    std::size_t         n_blocks_ = 0;  // finished blocks
    std::vector<double> mean_;
    std::vector<double> m2_;
    EstimatorReblocking reblocking_; // PIMC only; legacy Welford is untouched
};

}  // namespace alea
