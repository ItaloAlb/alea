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

// Pair-correlation function g_αβ(r) for ALL group-pair combinations — general,
// never hardcoded to a specific species.  For G groups it produces G(G+1)/2
// functions: same-group (AA, all unordered i<j within the group) and
// cross-group (AB, all i∈α × j∈β).  Mirrors the logic of QMCpack's
// PairCorrEstimator (src/QMCHamiltonians/PairCorrEstimator.cpp): a triangular
// pair id (gen_pair_id), an AA/AB split, and a shell-volume normalization.
//
// Layout note (GPU-aware): the per-block histogram is one flat, contiguous
// buffer indexed [pair_id * n_bins + bin], so it is trivially a device buffer,
// and accumulate() walks the distance table's row(i,j) — unit-stride over
// walkers — which vectorizes (the only wart is the data-dependent bin scatter).
class PairCorrelation : public Estimator {
public:
    PairCorrelation(std::span<const ParticleGroup> groups, std::size_t dim,
                    double rmax, int n_bins,
                    const SimulationCell& cell = SimulationCell{});

    void configure_reblocking(std::size_t num_blocks) override;
    void configure_signed_weights() override { reblocking_.configure_signed(); }
    // Worm runs: an open path is a measurement of n zero weights; it counts
    // in the signed normalization without being binned.
    void skip_measurement(std::size_t n) override {
        if (!reblocking_.signed_mode())
            throw std::logic_error("gofr: skip_measurement needs signed weights");
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
    [[nodiscard]] std::string type() const override { return "gofr"; }
    // gofr_r, gofr_<a>_<b>[_error].
    [[nodiscard]] std::vector<std::string> output_namespaces() const override {
        return {"gofr"};
    }

    // ── Introspection (used by callers / tests) ──────────────────────────
    std::size_t n_functions() const noexcept { return n_func_; }
    std::size_t n_bins()      const noexcept { return n_bins_; }
    double      bin_width()   const noexcept { return delta_; }
    // Number of distinct particle pairs contributing to a group-pair function
    // (0 ⇒ that function is empty, e.g. a single-particle group's AA term).
    std::size_t pair_count(std::size_t pid) const noexcept { return pair_count_[pid]; }
    // Triangular index of the (unordered) group pair (ga, gb).
    std::size_t pair_id(std::size_t ga, std::size_t gb) const noexcept;

private:
    struct GroupRange {
        std::string name;
        std::size_t first, last;
    };
    std::vector<GroupRange>  groups_;
    std::vector<std::size_t> group_of_;    // particle index → group index
    std::vector<std::size_t> pair_count_;  // [n_func] distinct pairs per function
    std::vector<std::string> names_;       // [n_func] "gofr_<A>_<B>"

    std::size_t dim_;
    double      rmax_;
    std::size_t n_bins_;
    double      delta_;
    double      inv_delta_;
    std::size_t np_;
    // Cell volume for the ρ = N/V factor; 0 on open boundaries, where a bound
    // complex has no meaningful bulk density and the historical per-pair
    // normalisation is kept.
    double      volume_ = 0.0;
    std::size_t n_groups_;
    std::size_t n_func_;

    // Current-block accumulation (flat: [pid * n_bins + bin]).
    std::vector<double> block_hist_;
    double              block_weight_sum_ = 0.0;  // Σ per-config sample weight
    double              block_abs_weight_sum_ = 0.0;  // signed normalization: number of weights

    // One entry per finished block: its per-bin estimate (block_hist / weight),
    // flat [n_func * n_bins].  Averaged per bin in finalize() for error bars.
    std::vector<std::vector<double>> block_history_;
    EstimatorReblocking reblocking_;
};

}  // namespace alea
