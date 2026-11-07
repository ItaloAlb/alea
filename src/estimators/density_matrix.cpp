// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "estimators/density_matrix.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <numbers>
#include <stdexcept>
#include <utility>

#include <omp.h>

#include "numerics/random.h"
#include "numerics/xoshiro_random.h"
#include "qmc/run_result.h"

namespace alea {

namespace {

// Configurations per worker batch: enough walkers for the SoA loops of the
// components, few enough that a worker's copy of ψ_T stays small.
constexpr std::size_t kChunkWalkers = 32;

// A direction uniform on the unit sphere of `dim` dimensions.
void unit_vector(numerics::XoshiroRandom& rng, std::size_t dim, double* u) {
    if (dim == 1) {
        u[0] = rng.uniform() < 0.5 ? -1.0 : 1.0;
        return;
    }
    if (dim == 2) {
        const double theta = 2.0 * std::numbers::pi * rng.uniform();
        u[0] = std::cos(theta);
        u[1] = std::sin(theta);
        return;
    }
    // Archimedes: z uniform on [−1, 1] gives a uniform point on the sphere.
    const double z = 2.0 * rng.uniform() - 1.0;
    const double phi = 2.0 * std::numbers::pi * rng.uniform();
    const double s = std::sqrt(std::max(0.0, 1.0 - z * z));
    u[0] = s * std::cos(phi);
    u[1] = s * std::sin(phi);
    u[2] = z;
}

}  // namespace

std::size_t VirtualMovePlan::n_two_body() const noexcept {
    std::size_t n = 0;
    for (const auto& [a, b] : pairs) n += a.size() * b.size();
    return n;
}

std::size_t VirtualMovePlan::max_channel() const noexcept {
    std::size_t n = 0;
    for (const auto& [a, b] : pairs) n = std::max(n, a.size() * b.size());
    return n;
}

void virtual_move_ratios(TrialWavefunction& psi, WalkerSet& batch, const SimulationCell& cell,
                         const VirtualMovePlan& plan, std::span<const double> displacement,
                         std::span<double> one_body, std::span<double> two_body,
                         std::span<double> scratch) {
    const std::size_t nw = batch.num_walkers();
    const std::size_t dim = batch.dim();
    if (displacement.size() != dim * nw || scratch.size() != plan.max_channel() * nw ||
        one_body.size() != plan.singles.size() * nw ||
        two_body.size() != plan.n_two_body() * nw)
        throw std::logic_error("virtual_move_ratios: buffer shapes do not match the batch");

    // One body: the first half of a Metropolis move at r_p + r′_w, then the
    // second half with nothing accepted, in the order of a sweep.
    const std::span<bool> mask = batch.accept_mask();
    std::fill(mask.begin(), mask.end(), false);
    for (std::size_t q = 0; q < plan.singles.size(); ++q) {
        const std::size_t p = plan.singles[q];
        const std::span<double> new_pos = batch.new_pos();
        const std::span<const double> pos = batch.positions();
        for (std::size_t w = 0; w < nw; ++w) {
            double r[SimulationCell::kMaxDim];
            for (std::size_t d = 0; d < dim; ++d)
                r[d] = pos[(p * dim + d) * nw + w] + displacement[d * nw + w];
            cell.wrap(r);
            for (std::size_t d = 0; d < dim; ++d) new_pos[d * nw + w] = r[d];
        }
        batch.propose_move_batch(p, new_pos);
        psi.ratio_batch(batch, p);
        const std::span<const double> lr = batch.log_ratios();
        const std::span<const double> ph = batch.phase_diffs();
        for (std::size_t w = 0; w < nw; ++w)
            one_body[q * nw + w] = std::exp(lr[w]) * std::cos(ph[w]);
        psi.apply_move_batch(batch, p, mask);
        batch.apply_move(p, batch.new_pos(), batch.new_grad(), batch.new_lap(), mask);
    }

    // Two body: every pair of a channel from R alone (logs into two_body,
    // phases into scratch, then the signed ratio in place).
    std::size_t row0 = 0;   // first two_body row of the channel
    for (const auto& [a, b] : plan.pairs) {
        const std::size_t rows = a.size() * b.size();
        const std::span<double> logs = two_body.subspan(row0 * nw, rows * nw);
        const std::span<double> phases = scratch.first(rows * nw);
        psi.shifted_pair_ratios_batch(batch, a.first, a.last, b.first, b.last, displacement,
                                      logs, phases);
        for (std::size_t k = 0; k < rows * nw; ++k)
            logs[k] = std::exp(logs[k]) * std::cos(phases[k]);
        row0 += rows;
    }
}

DensityMatrix::DensityMatrix(std::span<const ParticleGroup> groups,
                             std::vector<std::string> one_body,
                             std::vector<std::pair<std::string, std::string>> pairs,
                             double rmax, int n_bins, int every, const SimulationCell& cell,
                             std::size_t dim)
    : cell_(cell), dim_(dim), rmax_(rmax) {
    if (!std::isfinite(rmax) || !(rmax > 0.0))
        throw std::runtime_error("density_matrix: rmax must be finite and positive");
    if (n_bins < 1) throw std::runtime_error("density_matrix: n_bins must be at least 1");
    if (every < 1) throw std::runtime_error("density_matrix: every must be at least 1");
    if (dim < 1 || dim > SimulationCell::kMaxDim)
        throw std::runtime_error("density_matrix: needs a 1-, 2- or 3-D system");
    if (cell.is_periodic() && cell.dim() != dim)
        throw std::logic_error("density_matrix: cell and system dimensions differ");
    if (one_body.empty() && pairs.empty())
        throw std::runtime_error("density_matrix: give groups, pairs or both");
    n_bins_ = static_cast<std::size_t>(n_bins);
    every_ = static_cast<std::size_t>(every);
    delta_ = rmax / static_cast<double>(n_bins_);
    for (const ParticleGroup& g : groups) np_ = std::max(np_, g.last);

    // Index of a group in groups_, appending it on first use.
    const auto index_of = [&](const std::string& name) -> std::size_t {
        for (std::size_t k = 0; k < groups_.size(); ++k)
            if (groups_[k].name == name) return k;
        const auto it = std::find_if(groups.begin(), groups.end(),
                                     [&](const ParticleGroup& g) { return g.name == name; });
        if (name.empty() || it == groups.end() || it->last <= it->first)
            throw std::runtime_error("density_matrix: unknown or empty group '" + name + "'");
        groups_.push_back({name, {it->first, it->last}});
        return groups_.size() - 1;
    };
    for (const std::string& name : one_body) {
        if (std::any_of(groups_.begin(), groups_.end(),
                        [&](const Group& g) { return g.name == name; }))
            throw std::runtime_error("density_matrix: group '" + name + "' is listed twice");
        index_of(name);
    }
    for (const auto& [a, b] : pairs) {
        if (a == b)
            throw std::runtime_error("density_matrix: pair [" + a + ", " + b +
                                     "] needs two different groups");
        const Channel c{index_of(a), index_of(b)};
        for (const Channel& o : channels_)
            if ((o.a == c.a && o.b == c.b) || (o.a == c.b && o.b == c.a))
                throw std::runtime_error("density_matrix: pair [" + a + ", " + b +
                                         "] is listed twice");
        channels_.push_back(c);
    }
    for (const Group& g : groups_)
        for (std::size_t p = g.range.first; p < g.range.last; ++p) plan_.singles.push_back(p);
    for (const Channel& c : channels_)
        plan_.pairs.emplace_back(groups_[c.a].range, groups_[c.b].range);

    block_sums_.assign(sum_width(), 0.0);
    block_values_.assign((groups_.size() + 2 * channels_.size()) * n_bins_, 0.0);
    moments_.configure(block_values_.size());
}

DensityMatrix::DensityMatrix(const DensityMatrix& other)
    : Estimator(other),
      groups_(other.groups_),
      channels_(other.channels_),
      plan_(other.plan_),
      cell_(other.cell_),
      dim_(other.dim_),
      np_(other.np_),
      rmax_(other.rmax_),
      delta_(other.delta_),
      n_bins_(other.n_bins_),
      every_(other.every_),
      calls_(other.calls_),
      samples_(other.samples_),
      block_sums_(other.block_sums_),
      carry_(other.carry_),
      block_samples_(other.block_samples_),
      closed_samples_(other.closed_samples_),
      block_values_(other.block_values_),
      moments_(other.moments_) {}

void DensityMatrix::bind_wavefunction(const TrialWavefunction& psi, std::uint64_t seed,
                                      int n_workers) {
    if (psi.is_complex())
        throw std::runtime_error(
            "density_matrix: a complex (fixed-phase) trial wavefunction is not "
            "supported; its ratios would be complex");
    if (n_workers < 1) throw std::logic_error("density_matrix: n_workers must be at least 1");
    if (!channels_.empty()) {
        const std::vector<std::string> missing = psi.components_without_shifted_pairs();
        if (!missing.empty()) {
            std::string list;
            for (const std::string& name : missing) list += (list.empty() ? "" : ", ") + name;
            throw std::runtime_error(
                "density_matrix: `pairs` needs the ratios of a pair moved together, which "
                "these trial-wavefunction components do not supply: " + list);
        }
    }
    psi_copy_ = std::make_unique<TrialWavefunction>(psi.clone());
    seed_ = seed;
    n_workers_ = static_cast<std::size_t>(n_workers);
    workers_.clear();
}

void DensityMatrix::allocate_workers(std::size_t chunk) {
    workers_.clear();
    workers_.reserve(n_workers_);
    for (std::size_t k = 0; k < n_workers_; ++k) {
        workers_.push_back(Worker{psi_copy_->clone(),
                                  WalkerSet(chunk, np_, dim_, cell_),
                                  std::vector<double>(dim_ * chunk, 0.0),
                                  std::vector<double>(plan_.max_channel() * chunk, 0.0),
                                  std::vector<double>(plan_.singles.size() * chunk, 0.0),
                                  std::vector<double>(plan_.n_two_body() * chunk, 0.0),
                                  std::vector<std::size_t>(chunk, 0),
                                  std::vector<double>(sum_width(), 0.0)});
        workers_.back().psi.prepare(chunk, np_, dim_);
    }
}

void DensityMatrix::start_block() {
    if (carry_) return;   // the open block continues
    std::fill(block_sums_.begin(), block_sums_.end(), 0.0);
    block_samples_ = 0;
}

void DensityMatrix::accumulate(const ParticleBatch& configs, std::span<const double> weights) {
    if (!psi_copy_)
        throw std::logic_error(
            "density_matrix: accumulate before bind_wavefunction (only VMC and DMC bind it)");
    if (++calls_ % every_ != 0) return;
    const std::size_t nw = configs.num_walkers();
    if (configs.n_particles() != np_ || configs.dim() != dim_ || weights.size() != nw)
        throw std::runtime_error(
            "density_matrix: configurations do not match the system it was built for");
    if (nw == 0) return;
    for (const double w : weights)
        if (!std::isfinite(w) || w < 0.0)
            throw std::runtime_error("density_matrix: weights must be finite and nonnegative");
    if (workers_.empty()) allocate_workers(std::min(kChunkWalkers, nw));

    const std::size_t chunk = workers_.front().batch.num_walkers();
    const std::size_t n_chunks = (nw + chunk - 1) / chunk;
    const std::uint64_t sample0 = samples_;
    for (Worker& worker : workers_) std::fill(worker.sums.begin(), worker.sums.end(), 0.0);

    const std::size_t n_active = std::min(n_workers_, n_chunks);
    if (n_active <= 1) {
        for (std::size_t c = 0; c < n_chunks; ++c)
            measure_chunk(workers_.front(), configs, weights, c * chunk, sample0);
    } else {
        // Chunk c goes to worker t = thread; an exception must not escape the
        // parallel region, so the first one is kept and rethrown after it.
        std::exception_ptr error;
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(n_active))
        for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
            try {
                measure_chunk(workers_[static_cast<std::size_t>(omp_get_thread_num())],
                              configs, weights, static_cast<std::size_t>(c) * chunk, sample0);
            } catch (...) {
#pragma omp critical(density_matrix_error)
                if (!error) error = std::current_exception();
            }
        }
        if (error) std::rethrow_exception(error);
    }
    // Worker sums in worker order: the same total for a given worker count;
    // other counts agree to rounding.
    for (const Worker& worker : workers_)
        for (std::size_t i = 0; i < block_sums_.size(); ++i) block_sums_[i] += worker.sums[i];
    samples_ += nw;
    block_samples_ += nw;
}

void DensityMatrix::measure_chunk(Worker& worker, const ParticleBatch& configs,
                                  std::span<const double> weights, std::size_t w0,
                                  std::uint64_t sample0) {
    const std::size_t nw = configs.num_walkers();
    const std::size_t chunk = worker.batch.num_walkers();
    const std::size_t count = std::min(chunk, nw - w0);

    // Column j holds configuration w0 + j; padding columns repeat the last one
    // (evaluated, never counted), so the batch keeps its size.
    const std::span<const double> src = configs.positions();   // [p * dim * nw + d * nw + w]
    const std::span<double> dst = worker.batch.positions();    // [p * dim * chunk + d * chunk + j]
    for (std::size_t p = 0; p < np_; ++p)
        for (std::size_t d = 0; d < dim_; ++d)
            for (std::size_t j = 0; j < chunk; ++j)
                dst[(p * dim_ + d) * chunk + j] =
                    src[(p * dim_ + d) * nw + w0 + std::min(j, count - 1)];
    worker.batch.dist_tables().build_all(worker.batch.positions());
    worker.psi.evaluate_batch(worker.batch);

    for (std::size_t j = 0; j < chunk; ++j) {
        const std::uint64_t s = sample0 + w0 + std::min(j, count - 1);
        numerics::XoshiroRandom rng(numerics::stream_seed(seed_, kDensityMatrixSampleStream, s));
        const std::size_t bin = static_cast<std::size_t>(s % n_bins_);
        const double r = (static_cast<double>(bin) + rng.uniform()) * delta_;
        double u[SimulationCell::kMaxDim] = {0.0, 0.0, 0.0};
        unit_vector(rng, dim_, u);
        for (std::size_t d = 0; d < dim_; ++d) worker.displacement[d * chunk + j] = r * u[d];
        worker.bin[j] = bin;
    }
    virtual_move_ratios(worker.psi, worker.batch, cell_, plan_, worker.displacement,
                        worker.one_body, worker.two_body, worker.scratch);

    double* num1 = worker.sums.data();                    // [g * n_bins + b]
    double* num2 = num1 + groups_.size() * n_bins_;       // [c * n_bins + b]
    double* den = num2 + channels_.size() * n_bins_;      // [b]
    for (std::size_t j = 0; j < count; ++j) {
        const double weight = weights[w0 + j];
        const std::size_t b = worker.bin[j];
        den[b] += weight;
        std::size_t q = 0;
        for (std::size_t g = 0; g < groups_.size(); ++g) {
            const std::size_t n = groups_[g].range.size();
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k, ++q) sum += worker.one_body[q * chunk + j];
            num1[g * n_bins_ + b] += weight * sum / static_cast<double>(n);
        }
        std::size_t row = 0;
        for (std::size_t c = 0; c < channels_.size(); ++c) {
            const std::size_t n = plan_.pairs[c].first.size() * plan_.pairs[c].second.size();
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k, ++row) sum += worker.two_body[row * chunk + j];
            num2[c * n_bins_ + b] += weight * sum / static_cast<double>(n);
        }
    }
}

void DensityMatrix::stop_block() {
    for (const double s : block_sums_)
        if (!std::isfinite(s))
            throw std::runtime_error(
                "density_matrix: a wavefunction ratio was not finite (a sampled "
                "configuration on a node of psi_T?)");
    const std::size_t G = groups_.size(), K = channels_.size();
    const double* num1 = block_sums_.data();
    const double* num2 = num1 + G * n_bins_;
    const double* den = num2 + K * n_bins_;
    // A bin without a sample has no estimate (a zero would bias it): the block
    // stays open and merges into the next one.
    for (std::size_t b = 0; b < n_bins_; ++b)
        if (!(den[b] > 0.0)) {
            carry_ = true;
            return;
        }
    double* rho1 = block_values_.data();
    double* rho2 = rho1 + G * n_bins_;
    double* frac = rho2 + K * n_bins_;
    for (std::size_t g = 0; g < G; ++g)
        for (std::size_t b = 0; b < n_bins_; ++b)
            rho1[g * n_bins_ + b] = num1[g * n_bins_ + b] / den[b];
    for (std::size_t c = 0; c < K; ++c) {
        const Channel& ch = channels_[c];
        const double n_a = static_cast<double>(groups_[ch.a].range.size());
        const double n_b = static_cast<double>(groups_[ch.b].range.size());
        const double factor = n_a * n_b / std::min(n_a, n_b);   // N_a N_b / N_p (Eq. S8)
        for (std::size_t b = 0; b < n_bins_; ++b) {
            rho2[c * n_bins_ + b] = num2[c * n_bins_ + b] / den[b];
            frac[c * n_bins_ + b] =
                factor * (rho2[c * n_bins_ + b] -
                          rho1[ch.a * n_bins_ + b] * rho1[ch.b * n_bins_ + b]);
        }
    }
    moments_.append(block_values_);
    closed_samples_ += block_samples_;
    block_samples_ = 0;
    carry_ = false;
}

void DensityMatrix::collect(const Estimator& other) {
    const auto* source = dynamic_cast<const DensityMatrix*>(&other);
    if (!source) throw std::runtime_error("density_matrix: estimator type mismatch");
    const auto same_groups = [&] {
        if (groups_.size() != source->groups_.size()) return false;
        for (std::size_t g = 0; g < groups_.size(); ++g)
            if (groups_[g].name != source->groups_[g].name ||
                groups_[g].range.first != source->groups_[g].range.first ||
                groups_[g].range.last != source->groups_[g].range.last)
                return false;
        return true;
    };
    const auto same_channels = [&] {
        if (channels_.size() != source->channels_.size()) return false;
        for (std::size_t c = 0; c < channels_.size(); ++c)
            if (channels_[c].a != source->channels_[c].a ||
                channels_[c].b != source->channels_[c].b)
                return false;
        return true;
    };
    if (!same_groups() || !same_channels() || n_bins_ != source->n_bins_ ||
        rmax_ != source->rmax_ || every_ != source->every_)
        throw std::runtime_error("density_matrix: cannot collect a different configuration");
    moments_.collect(source->moments_);
    closed_samples_ += source->closed_samples_;
}

void DensityMatrix::finalize(RunResult& out) const {
    out.add_metadata("density_matrix_error_model", "independent blocks (no reblocking)");
    out.add_scalar("density_matrix_blocks", static_cast<double>(moments_.blocks()));
    out.add_scalar("density_matrix_measurements", static_cast<double>(closed_samples_));
    if (moments_.blocks() == 0) return;

    std::vector<double> r(n_bins_);
    for (std::size_t b = 0; b < n_bins_; ++b) r[b] = (static_cast<double>(b) + 0.5) * delta_;
    out.add_series("density_matrix_r", std::move(r));
    const auto emit = [&](const std::string& key, std::size_t offset) {
        std::vector<double> mean(n_bins_), error(n_bins_);
        for (std::size_t b = 0; b < n_bins_; ++b) {
            mean[b] = moments_.mean(offset + b);
            error[b] = moments_.sem(offset + b);
        }
        out.add_series(key, std::move(mean));
        out.add_series(key + "_error", std::move(error));
    };
    const std::size_t G = groups_.size(), K = channels_.size();
    for (std::size_t g = 0; g < G; ++g)
        emit("density_matrix_rho1_" + groups_[g].name, g * n_bins_);
    for (std::size_t c = 0; c < K; ++c) {
        const std::string pair = groups_[channels_[c].a].name + "_" + groups_[channels_[c].b].name;
        emit("density_matrix_rho2_" + pair, (G + c) * n_bins_);
        emit("density_matrix_pair_condensate_fraction_" + pair, (G + K + c) * n_bins_);
    }
}

}  // namespace alea
