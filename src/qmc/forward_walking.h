// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "../estimators/estimator.h"
#include "../hamiltonian/hamiltonian.h"
#include "../population/configuration_batch.h"
#include "../population/walker_set.h"
#include "run_result.h"

namespace alea {

// Pure (forward-walking) estimators of a DMC run
// (J. Casulleras & J. Boronat, Phys. Rev. B 52, 3654 (1995)).
//
// DMC samples the mixed distribution f(R) = ψ_T(R)Φ(R).  The descendants of a
// configuration R carried for an imaginary time T have, on average, the total
// weight
//
//   D_T(R) = [e^{−T(H−E_0)} ψ_T](R) / ψ_T(R)  →  Φ(R)⟨Φ|ψ_T⟩/ψ_T(R)   (T → ∞),
//
// so weighting the configuration sampled at t by the weight its descendants
// carry at t + T gives
//
//   ⟨O⟩_T = Σ_w W_w(t+T) O(R_{a(w)}(t)) / Σ_w W_w(t+T)
//         = ⟨Φ|O e^{−T(H−E_0)}|ψ_T⟩ / ⟨Φ|ψ_T⟩  →  ⟨Φ|O|Φ⟩/⟨Φ|Φ⟩,
//
// with a(w) the ancestor at t of walker w at t + T and W_w the walker weight
// the mixed estimator uses (after the step's reweighting, before branching).
// T = 0 is the mixed estimate itself.  The bias of a finite T is
// Σ_n c_n⟨Φ|O|n⟩e^{−T(E_n−E_0)}/c_0, c_n = ⟨n|ψ_T⟩: it decays with the gaps
// of the excitations O connects to Φ, while the variance grows with T because
// fewer and fewer ancestors carry the weight.  E_T and the population-control
// factors are common to every walker of a step and cancel in the ratio.  Under
// the fixed-node (fixed-phase) constraint H is the constrained Hamiltonian and
// Φ its ground state.
//
// ── Implementation ──────────────────────────────────────────────────────────
// Every `interval` steps the population's positions are stored once (a
// snapshot); each snapshot keeps, for every live walker, the index of its
// ancestor in that snapshot, and branch() carries those indices through the
// branching map — integers, not positions.  When a snapshot is lag steps old,
// the current weights are summed per ancestor, D[a] = Σ_{w: a(w)=a} W_w, and
// the ancestors with descendants are measured with weights D[a] through the
// ordinary Estimator::accumulate (any diagonal estimator works unchanged) and
// Hamiltonian::evaluate_potential (the pure potential energy, from which the
// pure kinetic energy is E − ⟨V⟩_pure).
//
// Steps are numbered from 0 at the first equilibration step; accumulation
// starts at step K0.  Snapshots start at K0 − max lag, so every accumulation
// step whose snapshot exists measures, and every block holds the same number
// of measurements.  Snapshots still in flight when the run ends are dropped.
class ForwardWalking {
public:
    // lag_steps: strictly increasing lags in DMC steps (a step is n_substeps
    // sweeps); step_time: the imaginary time of one step (τ·n_substeps), for
    // the output only; interval: steps between snapshots; estimators[i]: the
    // pure copies measured at lag i (each set may be empty; the potential
    // energy is always measured).  Throws on an inconsistent schedule or an
    // estimator that is not diagonal (one that needs ψ_T).
    ForwardWalking(std::vector<std::size_t> lag_steps, double step_time,
                   std::size_t interval,
                   std::vector<std::vector<std::unique_ptr<Estimator>>> estimators);

    // Size every buffer for populations of up to `capacity` walkers (no
    // allocation afterwards) and fix the schedule: accumulation starts at step
    // first_measured_step, which must be at least the largest lag.
    void prepare(std::size_t capacity, std::size_t np, std::size_t dim,
                 const SimulationCell& cell, std::size_t first_measured_step);

    // Step `step` of the run, after the moves and the reweighting and before
    // branching: store the population if a snapshot is due, then, at an
    // accumulation step, measure every lag whose snapshot is that old.
    void observe(const WalkerSet& population, std::size_t step,
                 const Hamiltonian& hamiltonian);

    // After branching: new walker c descends from old walker src[c].
    void branch(std::span<const std::size_t> src);

    void start_block();
    void stop_block();

    // Per lag i: the pure estimators (keys labelled fw<i>), potential_fw<i>
    // with its reblocked error and diagnostics, acc_block_potential_fw<i>, and
    // fw<i>_lag_time / _lag_steps / _measurements / _surviving_fraction; plus
    // fw_interval and the series fw_lag_times.
    void write_results(RunResult& out) const;

    [[nodiscard]] std::size_t n_lags() const noexcept { return lag_steps_.size(); }
    [[nodiscard]] std::size_t n_slots() const noexcept { return slots_.size(); }
    // Bytes held by the snapshot ring and the measurement batch after prepare().
    [[nodiscard]] std::size_t memory_bytes() const noexcept;

private:
    static constexpr std::size_t kFree = static_cast<std::size_t>(-1);

    struct Snapshot {
        std::size_t              step = kFree;  // step it was taken at; kFree: unused
        std::size_t              nw   = 0;      // population size at that step
        std::vector<double>      positions;     // [np * dim * nw]
        std::vector<std::size_t> ancestor;      // [live nw]: index in positions
    };

    void record(const WalkerSet& population, std::size_t step);
    void measure(std::size_t lag, const Snapshot& snapshot, const WalkerSet& population,
                 const Hamiltonian& hamiltonian);
    [[nodiscard]] bool snapshot_step(std::size_t step) const noexcept;
    [[nodiscard]] std::size_t slot_of(std::size_t step) const noexcept;

    std::vector<std::size_t> lag_steps_;   // [n_lags]
    double                   step_time_;
    std::size_t              interval_;
    std::vector<std::vector<std::unique_ptr<Estimator>>> estimators_;  // [n_lags][...]

    std::size_t first_measured_step_ = 0;  // K0
    std::size_t first_snapshot_step_ = 0;  // K0 − max lag
    bool        prepared_ = false;

    std::vector<Snapshot> slots_;          // ring, floor(max lag / interval) + 1
    std::unique_ptr<ConfigurationBatch> batch_;

    // Measurement scratch, [capacity] each.
    std::vector<double>      descendant_weight_;  // D[a], a over the snapshot
    std::vector<std::size_t> kept_;               // ancestors with descendants
    std::vector<double>      kept_weight_;        // their D[a]
    std::vector<double>      potential_;          // V of each kept ancestor
    std::vector<std::size_t> gather_;             // branch() destination

    // Pure potential energy per lag: the open block's Σ D V and Σ D, and the
    // block means.
    std::vector<double>              block_weighted_potential_;  // [n_lags]
    std::vector<double>              block_weight_;              // [n_lags]
    std::vector<std::vector<double>> block_potential_;           // [n_lags][block]
    // Diagnostics: measurements per lag and Σ of the fraction of each
    // snapshot's walkers that still have descendants.
    std::vector<std::size_t> measurements_;       // [n_lags]
    std::vector<double>      surviving_sum_;      // [n_lags]
};

}  // namespace alea
