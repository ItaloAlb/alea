// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "qmc/forward_walking.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include "numerics/statistics.h"
#include "qmc/reblock_diagnostics.h"

namespace alea {

ForwardWalking::ForwardWalking(
    std::vector<std::size_t> lag_steps, double step_time, std::size_t interval,
    std::vector<std::vector<std::unique_ptr<Estimator>>> estimators)
    : lag_steps_(std::move(lag_steps)), step_time_(step_time), interval_(interval),
      estimators_(std::move(estimators)) {
    if (lag_steps_.empty())
        throw std::invalid_argument("forward walking needs at least one lag");
    for (std::size_t i = 1; i < lag_steps_.size(); ++i)
        if (lag_steps_[i] <= lag_steps_[i - 1])
            throw std::invalid_argument("forward walking lags must be strictly increasing");
    if (interval_ == 0)
        throw std::invalid_argument("forward walking interval must be at least one step");
    if (!(std::isfinite(step_time_) && step_time_ > 0.0))
        throw std::invalid_argument("forward walking needs a positive step time (tau * sub_steps)");
    if (estimators_.empty()) estimators_.resize(lag_steps_.size());
    if (estimators_.size() != lag_steps_.size())
        throw std::invalid_argument("forward walking: one estimator set per lag");
    for (const auto& set : estimators_)
        for (const auto& e : set)
            if (e->needs_wavefunction())
                throw std::invalid_argument(
                    "forward walking measures diagonal observables only; '" + e->type() +
                    "' takes ratios of psi_T at displaced configurations");

    const std::size_t n = lag_steps_.size();
    block_weighted_potential_.assign(n, 0.0);
    block_weight_.assign(n, 0.0);
    block_potential_.resize(n);
    measurements_.assign(n, 0);
    surviving_sum_.assign(n, 0.0);
}

void ForwardWalking::prepare(std::size_t capacity, std::size_t np, std::size_t dim,
                             const SimulationCell& cell,
                             std::size_t first_measured_step) {
    const std::size_t max_lag = lag_steps_.back();
    if (first_measured_step < max_lag)
        throw std::invalid_argument(
            "forward walking: the largest lag (" + std::to_string(max_lag) +
            " steps) exceeds the equilibration (" + std::to_string(first_measured_step) +
            " steps); the first accumulation block would have no snapshot that old");
    first_measured_step_ = first_measured_step;
    first_snapshot_step_ = first_measured_step - max_lag;

    // A snapshot taken at s is last read at s + max lag, before the snapshot
    // of that step is recorded into the ring: floor(max/interval) + 1 slots.
    slots_.assign(max_lag / interval_ + 1, Snapshot{});
    for (Snapshot& s : slots_) {
        s.positions.reserve(np * dim * capacity);
        s.ancestor.reserve(capacity);
    }
    batch_ = std::make_unique<ConfigurationBatch>(capacity, np, dim, cell);
    descendant_weight_.assign(capacity, 0.0);
    kept_.assign(capacity, 0);
    kept_weight_.assign(capacity, 0.0);
    potential_.assign(capacity, 0.0);
    gather_.reserve(capacity);
    prepared_ = true;
}

std::size_t ForwardWalking::memory_bytes() const noexcept {
    std::size_t bytes = 0;
    for (const Snapshot& s : slots_)
        bytes += s.positions.capacity() * sizeof(double) +
                 s.ancestor.capacity() * sizeof(std::size_t);
    if (batch_) {
        const std::size_t c = batch_->capacity();
        const std::size_t np = batch_->n_particles(), dim = batch_->dim();
        bytes += (np * np + 2 * np * dim + np + dim) * c * sizeof(double);
    }
    return bytes;
}

bool ForwardWalking::snapshot_step(std::size_t step) const noexcept {
    return step >= first_snapshot_step_ && (step - first_snapshot_step_) % interval_ == 0;
}

std::size_t ForwardWalking::slot_of(std::size_t step) const noexcept {
    return ((step - first_snapshot_step_) / interval_) % slots_.size();
}

void ForwardWalking::record(const WalkerSet& population, std::size_t step) {
    Snapshot& s = slots_[slot_of(step)];
    const std::size_t nw = population.num_walkers();
    const std::span<const double> positions = population.positions();
    s.step = step;
    s.nw = nw;
    s.positions.assign(positions.begin(),
                       positions.begin() + static_cast<std::ptrdiff_t>(
                           population.n_particles() * population.dim() * nw));
    s.ancestor.resize(nw);
    std::iota(s.ancestor.begin(), s.ancestor.end(), std::size_t{0});
}

void ForwardWalking::observe(const WalkerSet& population, std::size_t step,
                             const Hamiltonian& hamiltonian) {
    if (!prepared_) throw std::logic_error("ForwardWalking::observe before prepare");
    // Record first: a lag of 0 measures the snapshot of this very step.  The
    // slot it overwrites belongs to step − (max/interval + 1)·interval, older
    // than every lag.
    if (snapshot_step(step)) record(population, step);
    if (step < first_measured_step_) return;
    for (std::size_t i = 0; i < lag_steps_.size(); ++i) {
        const std::size_t taken = step - lag_steps_[i];
        if (!snapshot_step(taken)) continue;
        const Snapshot& s = slots_[slot_of(taken)];
        if (s.step != taken)
            throw std::logic_error("ForwardWalking: snapshot ring overwritten too early");
        measure(i, s, population, hamiltonian);
    }
}

void ForwardWalking::measure(std::size_t lag, const Snapshot& snapshot,
                             const WalkerSet& population,
                             const Hamiltonian& hamiltonian) {
    // D[a] = Σ_{w: a(w)=a} W_w: the weight the descendants of ancestor a carry
    // now, before this step's branching (the mixed estimator's weights).
    const std::span<const double> weights = population.weights();
    const std::size_t nw = population.num_walkers();
    if (snapshot.ancestor.size() != nw)
        throw std::logic_error("ForwardWalking: ancestry out of step with the population");
    std::fill(descendant_weight_.begin(),
              descendant_weight_.begin() + static_cast<std::ptrdiff_t>(snapshot.nw), 0.0);
    for (std::size_t w = 0; w < nw; ++w) descendant_weight_[snapshot.ancestor[w]] += weights[w];

    // Only ancestors with descendants are measured, in snapshot order, so a
    // lag of 0 with every walker kept repeats the mixed measurement exactly.
    std::size_t n_kept = 0;
    for (std::size_t a = 0; a < snapshot.nw; ++a) {
        if (!(descendant_weight_[a] > 0.0)) continue;
        kept_[n_kept] = a;
        kept_weight_[n_kept] = descendant_weight_[a];
        ++n_kept;
    }
    if (n_kept == 0) return;   // only if every weight vanished; nothing to bin

    const std::span<const std::size_t> kept(kept_.data(), n_kept);
    const std::span<const double> kept_weight(kept_weight_.data(), n_kept);
    batch_->load(snapshot.positions, snapshot.nw, kept);
    for (const auto& e : estimators_[lag]) e->accumulate(*batch_, kept_weight);

    const std::span<double> v(potential_.data(), n_kept);
    hamiltonian.evaluate_potential(*batch_, v);
    double weighted = 0.0, total = 0.0;
    for (std::size_t c = 0; c < n_kept; ++c) {
        weighted += kept_weight[c] * v[c];
        total += kept_weight[c];
    }
    block_weighted_potential_[lag] += weighted;
    block_weight_[lag] += total;
    ++measurements_[lag];
    surviving_sum_[lag] += static_cast<double>(n_kept) / static_cast<double>(snapshot.nw);
}

void ForwardWalking::branch(std::span<const std::size_t> src) {
    for (Snapshot& s : slots_) {
        if (s.step == kFree) continue;
        gather_.resize(src.size());
        for (std::size_t c = 0; c < src.size(); ++c) gather_[c] = s.ancestor[src[c]];
        s.ancestor.swap(gather_);
    }
}

void ForwardWalking::start_block() {
    std::fill(block_weighted_potential_.begin(), block_weighted_potential_.end(), 0.0);
    std::fill(block_weight_.begin(), block_weight_.end(), 0.0);
    for (auto& set : estimators_)
        for (auto& e : set) e->start_block();
}

void ForwardWalking::stop_block() {
    for (std::size_t i = 0; i < lag_steps_.size(); ++i)
        if (block_weight_[i] > 0.0)
            block_potential_[i].push_back(block_weighted_potential_[i] / block_weight_[i]);
    for (auto& set : estimators_)
        for (auto& e : set) e->stop_block();
}

void ForwardWalking::write_results(RunResult& out) const {
    std::vector<double> lag_times(lag_steps_.size());
    for (std::size_t i = 0; i < lag_steps_.size(); ++i) {
        const std::string tag = "fw" + std::to_string(i);
        const double lag_time = static_cast<double>(lag_steps_[i]) * step_time_;
        lag_times[i] = lag_time;
        out.add_scalar(tag + "_lag_time", lag_time);
        out.add_scalar(tag + "_lag_steps", static_cast<double>(lag_steps_[i]));
        out.add_scalar(tag + "_measurements", static_cast<double>(measurements_[i]));
        out.add_scalar(tag + "_surviving_fraction",
                       measurements_[i] > 0
                           ? surviving_sum_[i] / static_cast<double>(measurements_[i])
                           : 0.0);
        const std::vector<double>& chain = block_potential_[i];
        const std::string name = "potential_" + tag;
        if (!chain.empty()) {
            const numerics::ReblockResult stats = numerics::reblock(chain);
            out.add_scalar(name, stats.mean);
            out.add_scalar(name + "_error", stats.error);
            add_reblock_diagnostics(out, name, stats, chain.size());
        }
        out.add_series("acc_block_" + name, chain);
        for (const auto& e : estimators_[i]) e->write_results(out);
    }
    out.add_scalar("fw_interval", static_cast<double>(interval_));
    out.add_series("fw_lag_times", std::move(lag_times));
}

}  // namespace alea
