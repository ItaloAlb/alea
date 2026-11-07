// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "numerics/statistics.h"
#include "qmc/run_result.h"

#include "population/walker_set.h"

namespace alea {

struct RunResult;
class TrialWavefunction;

// Abstract base for per-block Monte Carlo estimators other than the always-on
// local energy.
//
// The lifecycle mirrors how the drivers already accumulate the energy, and is
// deliberately split into two levels so it maps cleanly onto a future GPU
// backend (one estimator per thread/crowd accumulating into its own contiguous
// buffer, then a cheap reduce):
//
//   per block:   start_block() → accumulate(configs, weights) × n_steps → stop_block()
//   reduce:      collect(other)     merges another clone's finished blocks
//   output:      finalize(out)      normalizes and writes named series
//
// Estimators see configurations only (ParticleBatch) and an explicit weight per
// configuration, so the same accumulate path serves unweighted VMC (weights ≡
// 1), weighted (mixed) DMC and PIMC, whose configurations are imaginary-time
// slices (weights ≡ 1: every slice of a closed path samples the diagonal
// density matrix).
class Estimator {
public:
    virtual ~Estimator() = default;

    // Reset the running buffer for a new block.
    virtual void start_block() = 0;
    // Bin this measurement step's samples: configuration w of `configs`
    // counts with weight weights[w] ([nw]).
    virtual void accumulate(const ParticleBatch& configs,
                            std::span<const double> weights) = 0;
    // Walkers carry their own weights (≡ 1 in VMC, the DMC weights otherwise).
    void accumulate(const WalkerSet& wset) { accumulate(wset, wset.weights()); }
    // Close the current block, storing its per-bin estimate for error bars.
    virtual void stop_block() = 0;
    // Merge another estimator clone's finished blocks into this one (thread
    // reduce).  `other` must be the same concrete type.
    virtual void collect(const Estimator& other) = 0;
    // Normalize and append the result as named 1-D series to `out`.
    virtual void finalize(RunResult& out) const = 0;

    // PIMC opts in before cloning. Unsupported estimators refuse rather than
    // silently returning independence-assuming errors. VMC/DMC never opt in.
    virtual void configure_reblocking(std::size_t) {
        throw std::runtime_error("estimator '" + type() + "' does not support PIMC reblocking");
    }
    // History bytes per replica, checked before the driver's aggregate budget.
    [[nodiscard]] virtual std::size_t reblocking_storage_bytes(std::size_t) const {
        throw std::runtime_error("estimator '" + type() + "' does not support PIMC reblocking");
    }
    // Signed PIMC (fermions): weights carry the permutation sign s = +/-1
    // (times [closed] in worm runs).  Each block is then normalized by its
    // number of weights (= sum|w| without a worm) and its mean sign is kept, so
    // every output is the ratio of means <O s>/<s> with a jackknife error
    // (numerics::ratio_of_means).  Called once, before configure_reblocking.
    virtual void configure_signed_weights() {
        throw std::runtime_error("estimator '" + type() + "' does not support signed weights");
    }
    // Worm runs (signed mode): a measurement on an open path has weight 0 for
    // every one of its n configurations.  Signed blocks are normalized by the
    // number of weights (= sum|w| for s = +/-1), so it is counted, not binned.
    virtual void skip_measurement(std::size_t) {
        throw std::logic_error("estimator '" + type() + "' cannot skip worm measurements");
    }
    // Off-diagonal (open-path) estimators of a PIMC worm, e.g. the one-body
    // density matrix: the group whose worm they follow (empty for the usual
    // diagonal estimators), the extra-bead samples wanted per open
    // measurement, the worm's particle count (set once by the driver), and
    // the samples themselves: displacement [n * dim], weight [n].
    [[nodiscard]] virtual std::string open_path_group() const { return {}; }
    [[nodiscard]] virtual std::size_t open_path_samples() const noexcept { return 0; }
    virtual void configure_open_paths(std::size_t) {
        throw std::logic_error("estimator '" + type() + "' does not measure open paths");
    }
    virtual void accumulate_open(std::span<const double>, std::span<const double>) {
        throw std::logic_error("estimator '" + type() + "' does not measure open paths");
    }

    // Estimators of ψ_T ratios at virtual configurations (the density
    // matrices) need the trial wavefunction itself, not only the sampled R.
    // The VMC and DMC drivers bind every such estimator once per run, after
    // cloning it and before its first accumulate: it keeps its own copies of
    // ψ_T (one per worker of its internal walker loop, n_workers ≥ 1), so its
    // virtual moves never touch the sampling state, and draws its random
    // numbers from the stream `seed`.  PIMC never binds (the builders refuse
    // such kinds there).
    [[nodiscard]] virtual bool needs_wavefunction() const noexcept { return false; }
    virtual void bind_wavefunction(const TrialWavefunction&, std::uint64_t /*seed*/,
                                   int /*n_workers*/) {
        throw std::logic_error("estimator '" + type() + "' does not use the trial wavefunction");
    }

    // Deep copy for per-thread isolation (like TrialWavefunction::clone()).
    [[nodiscard]] virtual std::unique_ptr<Estimator> clone() const = 0;

    // Config tag (e.g. "gofr").
    [[nodiscard]] virtual std::string type() const = 0;

    // ── Result keys ─────────────────────────────────────────────────────────
    // Every key finalize() writes (series, scalar or metadata) equals one of
    // these namespaces or extends one with '_': "gofr" covers gofr_r,
    // gofr_e_h and gofr_e_h_error.  Instances whose namespaces coincide would
    // overwrite each other in the results file; builders/estimator_builders.h
    // keeps them apart with output labels.
    [[nodiscard]] virtual std::vector<std::string> output_namespaces() const = 0;

    // Inserted after the namespace of every result key of this instance:
    // label "fine" turns gofr_r into gofr_fine_r.  Empty → the plain keys.  Set
    // once by the builder; clones carry it.
    void set_output_label(std::string label) { output_label_ = std::move(label); }
    [[nodiscard]] const std::string& output_label() const noexcept { return output_label_; }

    // finalize() under this instance's output label — what every driver calls
    // to emit results.  Cold path (once per run).
    void write_results(RunResult& out) const {
        if (output_label_.empty()) {
            finalize(out);
            return;
        }
        RunResult mine;
        finalize(mine);
        const std::vector<std::string> namespaces = output_namespaces();
        // The longest namespace `key` equals or extends with '_' gets the label.
        const auto relabel = [&](const std::string& key) {
            const std::string* match = nullptr;
            for (const std::string& ns : namespaces)
                if ((key == ns || (key.size() > ns.size() && key.starts_with(ns) &&
                                   key[ns.size()] == '_')) &&
                    (!match || ns.size() > match->size()))
                    match = &ns;
            if (!match)
                throw std::logic_error("estimator '" + type() + "': result key '" + key +
                                       "' lies outside its output namespaces");
            return *match + "_" + output_label_ + key.substr(match->size());
        };
        for (auto& [key, values] : mine.series) out.add_series(relabel(key), std::move(values));
        for (auto& [key, value] : mine.scalars) out.add_scalar(relabel(key), value);
        for (auto& [key, value] : mine.metadata) out.add_metadata(relabel(key), std::move(value));
    }

private:
    std::string output_label_;
};

// Constant-space block statistics of `width` components, for the VMC/DMC path
// of estimators whose PIMC path is EstimatorReblocking.  Blocks are treated as
// independent samples (as Density, CollectiveDensity and
// ElectronHoleProximity do).  Welford: μ_n = μ_{n−1} + (x−μ_{n−1})/n,
// M2_n = M2_{n−1} + (x−μ_{n−1})(x−μ_n); clones merge by the parallel rule of
// Chan, Golub & LeVeque (1979): δ = μ_b − μ_a, μ = μ_a + δ n_b/n,
// M2 = M2_a + M2_b + δ² n_a n_b/n.  SEM = √(M2 / ((n−1) n)).
class BlockMoments {
public:
    // Size once, at estimator construction (no allocation while sampling).
    void configure(std::size_t width) {
        if (width == 0) throw std::logic_error("BlockMoments: width must be positive");
        mean_.assign(width, 0.0);
        m2_.assign(width, 0.0);
        n_blocks_ = 0;
    }
    [[nodiscard]] std::size_t width() const noexcept { return mean_.size(); }
    [[nodiscard]] std::size_t blocks() const noexcept { return n_blocks_; }
    // One block's normalized estimate of every component.
    void append(std::span<const double> values) {
        if (values.size() != mean_.size())
            throw std::logic_error("BlockMoments: block width mismatch");
        ++n_blocks_;
        const double n = static_cast<double>(n_blocks_);
        for (std::size_t i = 0; i < mean_.size(); ++i) {
            const double delta = values[i] - mean_[i];
            mean_[i] += delta / n;
            m2_[i] += delta * (values[i] - mean_[i]);
        }
    }
    void collect(const BlockMoments& other) {
        if (other.mean_.size() != mean_.size())
            throw std::logic_error("BlockMoments: cannot merge different widths");
        if (other.n_blocks_ == 0) return;
        if (n_blocks_ == 0) { *this = other; return; }
        if (n_blocks_ > std::numeric_limits<std::size_t>::max() - other.n_blocks_)
            throw std::runtime_error("BlockMoments: collected block count overflow");
        const double n_a = static_cast<double>(n_blocks_);
        const double n_b = static_cast<double>(other.n_blocks_);
        const double n = n_a + n_b;
        for (std::size_t i = 0; i < mean_.size(); ++i) {
            const double delta = other.mean_[i] - mean_[i];
            mean_[i] += delta * (n_b / n);
            m2_[i] += other.m2_[i] + delta * delta * n_a * n_b / n;
        }
        n_blocks_ += other.n_blocks_;
    }
    [[nodiscard]] double mean(std::size_t i) const { return mean_.at(i); }
    // 0 with a single block (no scatter to estimate), as the other estimators.
    [[nodiscard]] double sem(std::size_t i) const {
        if (n_blocks_ < 2) return 0.0;
        const double n = static_cast<double>(n_blocks_);
        return std::sqrt(m2_.at(i) / ((n - 1.0) * n));
    }

private:
    std::size_t n_blocks_ = 0;
    std::vector<double> mean_;  // [width]
    std::vector<double> m2_;    // [width]
};

// Opt-in bounded, flat [block * width + bin] history. collect() adds equal
// temporal indices across independent replicas; no artificial transition is
// introduced by concatenating chain boundaries. Flyvbjerg & Petersen,
// J. Chem. Phys. 91, 461 (1989), applied to the synchronized grand-block means.
class EstimatorReblocking {
public:
    static constexpr std::size_t kMaxHistoryBytes = 64 * 1024 * 1024;

    [[nodiscard]] static std::size_t storage_bytes(std::size_t blocks,
                                                 std::size_t width) {
        if (width > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
            (width != 0 && blocks > std::numeric_limits<std::size_t>::max() /
                                       (width * sizeof(double))))
            throw std::runtime_error("PIMC estimator reblocking history size overflow");
        return blocks * width * sizeof(double);
    }
    // Configure a pristine estimator; history never grows beyond this budget.
    void configure(std::size_t blocks, std::size_t width) {
        if (blocks == 0 || width == 0)
            throw std::runtime_error("PIMC estimator reblocking requires nonempty blocks and bins");
        if (storage_bytes(blocks, width) > kMaxHistoryBytes)
            throw std::runtime_error("PIMC estimator reblocking history exceeds 64 MiB per replica; reduce bins or blocks");
        if (enabled() || !history_.empty())
            throw std::runtime_error("PIMC estimator reblocking must be configured once before accumulation");
        expected_blocks_ = blocks;
        width_ = width;
        // Do not allocate on the template: replicas reserve on first append.
    }
    [[nodiscard]] bool enabled() const noexcept { return expected_blocks_ != 0; }
    // Signed blocks: values are sum(w O)/sum|w| and each block records its
    // mean sign sum(w)/sum|w|; emit() forms ratios of means.  Set once, before
    // configure() and any append.
    void configure_signed() {
        if (enabled() || !history_.empty())
            throw std::runtime_error("PIMC estimator signed mode must precede reblocking");
        signed_ = true;
    }
    [[nodiscard]] bool signed_mode() const noexcept { return signed_; }
    [[nodiscard]] std::size_t blocks() const noexcept {
        return width_ == 0 ? 0 : history_.size() / width_;
    }
    // One normalized per-bin estimate from a single replica's block; a signed
    // block also records its mean sign (1 for unsigned blocks).
    void append(std::span<const double> values, double sign_mean = 1.0) {
        if (values.size() != width_ || blocks() >= expected_blocks_ || replicas_ > 1)
            throw std::runtime_error("PIMC estimator reblocking block shape/count mismatch");
        // Copying a pristine template need not copy vector capacity. Reserve
        // the exact bounded history on first append, avoiding geometric growth
        // beyond the configured memory budget in its replica clones.
        if (history_.capacity() < expected_blocks_ * width_)
            history_.reserve(expected_blocks_ * width_);
        replicas_ = 1;
        history_.insert(history_.end(), values.begin(), values.end());
        if (signed_) {
            if (sign_history_.capacity() < expected_blocks_)
                sign_history_.reserve(expected_blocks_);
            sign_history_.push_back(sign_mean);
        }
    }
    // Fixed-order indexwise reduction; every replica must finish equal blocks.
    void collect(const EstimatorReblocking& other) {
        if (!enabled() || !other.enabled() || width_ != other.width_ ||
            expected_blocks_ != other.expected_blocks_)
            throw std::runtime_error("PIMC estimator reblocking configuration mismatch");
        if (other.blocks() != expected_blocks_ || signed_ != other.signed_)
            throw std::runtime_error("PIMC estimator reblocking requires all configured blocks");
        if (history_.empty()) {
            history_ = other.history_;
            sign_history_ = other.sign_history_;
            replicas_ = other.replicas_;
            return;
        }
        if (blocks() != other.blocks())
            throw std::runtime_error("PIMC estimator replica block counts differ");
        for (std::size_t i = 0; i < history_.size(); ++i)
            history_[i] += other.history_[i];
        for (std::size_t i = 0; i < sign_history_.size(); ++i)
            sign_history_[i] += other.sign_history_[i];
        replicas_ += other.replicas_;
    }
    // Export a signed grand-block numerator and its denominator for paired
    // cross-estimator checks. Ratios must be formed after averaging blocks;
    // per-block division would bias the mean when a block's sign is small.
    void emit_ratio_history(RunResult& out, const std::string& name,
                            std::size_t offset, double norm = 1.0,
                            const std::string& denominator_name = "") const {
        if (!signed_ || blocks() != expected_blocks_ || replicas_ == 0 ||
            offset >= width_ || !std::isfinite(norm) || !(norm > 0.0))
            throw std::runtime_error("PIMC ratio history requires complete signed blocks and positive norm");
        std::vector<double> numerator(blocks()), denominator(blocks());
        for (std::size_t block = 0; block < blocks(); ++block) {
            numerator[block] = history_[block * width_ + offset] /
                               (static_cast<double>(replicas_) * norm);
            denominator[block] = sign_history_[block] / static_cast<double>(replicas_);
        }
        out.add_series(name + "_numerator_blocks", std::move(numerator));
        if (!denominator_name.empty())
            out.add_series(denominator_name, std::move(denominator));
    }

    // Export synchronized unsigned block means for covariance/drift analysis.
    // Signed observables must use emit_ratio_history, never blockwise division.
    void emit_unsigned_history(RunResult& out, const std::string& name,
                               std::size_t offset) const {
        if (signed_ || blocks() != expected_blocks_ || replicas_ == 0 || offset >= width_)
            throw std::runtime_error("PIMC unsigned history requires complete unsigned blocks");
        std::vector<double> values(blocks());
        for (std::size_t block = 0; block < blocks(); ++block)
            values[block] = history_[block * width_ + offset] / static_cast<double>(replicas_);
        out.add_series(name, std::move(values));
    }

    // Emit the observable and correlation diagnostics per bin; the naive SEM
    // is that of grand-block means, not the concatenation of replica samples.
    // level/count/length refer to the actual max-SEM row. The original onset
    // and selected SEM's error-on-error are separate diagnostics; reliability
    // needs both a verified plateau and at least 16 surviving blocks.
    void emit(RunResult& out, const std::string& name, std::size_t offset,
              std::size_t count, std::span<const double> norms = {}) const {
        if (blocks() != expected_blocks_ || replicas_ == 0 || offset > width_ ||
            count > width_ - offset || (!norms.empty() && norms.size() != count))
            throw std::runtime_error("PIMC estimator reblocking incomplete history/output shape");
        std::vector<double> chain(blocks());
        std::vector<double> mean(count), error(count), naive(count), plateau(count),
            level(count), n_blocks(count), tau(count), reliable(count), length(count),
            error_on_error(count), onset(count);
        // Signed: grand-block mean signs, the common denominator of every bin.
        std::vector<double> sign_chain;
        if (signed_)
            for (double s : sign_history_)
                sign_chain.push_back(s / static_cast<double>(replicas_));
        for (std::size_t bin = 0; bin < count; ++bin) {
            const double norm = norms.empty() ? 1.0 : norms[bin];
            for (std::size_t block = 0; block < blocks(); ++block)
                chain[block] = history_[block * width_ + offset + bin] /
                               (static_cast<double>(replicas_) * norm);
            if (signed_) {
                // <O s>/<s>: plain ratio of means, jackknife error over the
                // reblocking's decorrelated coarse blocks.
                const auto ratio = numerics::ratio_of_means(chain, sign_chain);
                mean[bin] = ratio.mean;
                error[bin] = ratio.error;
                naive[bin] = ratio.numerator.naive_error;
                plateau[bin] = ratio.numerator.plateau_found && ratio.denominator.plateau_found ? 1.0 : 0.0;
                error_on_error[bin] = ratio.numerator.error_on_error;
                onset[bin] = ratio.numerator.level;
                level[bin] = ratio.level;
                n_blocks[bin] = static_cast<double>(ratio.n_coarse_blocks);
                tau[bin] = std::max(ratio.numerator.tau_int, ratio.denominator.tau_int);
                length[bin] = static_cast<double>(std::size_t{1} << ratio.level);
                reliable[bin] = ratio.reliable ? 1.0 : 0.0;
                continue;
            }
            const auto stats = numerics::reblock(chain);
            mean[bin] = stats.mean;
            error[bin] = stats.error;
            naive[bin] = stats.naive_error;
            plateau[bin] = stats.plateau_found ? 1.0 : 0.0;
            error_on_error[bin] = stats.error_on_error;
            onset[bin] = stats.level;
            level[bin] = stats.selected_level;
            n_blocks[bin] = static_cast<double>(stats.selected_n_blocks);
            tau[bin] = stats.tau_int;
            length[bin] = static_cast<double>(std::size_t{1} << stats.selected_level);
            reliable[bin] = stats.plateau_found && n_blocks[bin] >= 16 ? 1.0 : 0.0;
        }
        out.add_series(name, std::move(mean));
        out.add_series(name + "_error", std::move(error));
        out.add_series(name + "_error_naive", std::move(naive));
        out.add_series(name + "_reblock_plateau", std::move(plateau));
        out.add_series(name + "_reblock_level", std::move(level));
        out.add_series(name + "_reblock_onset_level", std::move(onset));
        out.add_series(name + "_error_on_error", std::move(error_on_error));
        out.add_series(name + "_reblock_n_blocks", std::move(n_blocks));
        out.add_series(name + "_reblock_length", std::move(length));
        out.add_series(name + "_tau_int", std::move(tau));
        out.add_series(name + "_error_reliable", std::move(reliable));
        out.add_scalar(name + "_reblock_replicas", static_cast<double>(replicas_));
    }

private:
    std::size_t expected_blocks_ = 0;
    std::size_t width_ = 0;
    std::size_t replicas_ = 0;
    bool signed_ = false;
    std::vector<double> history_;
    std::vector<double> sign_history_;  // [block], signed mode only
};

}  // namespace alea
