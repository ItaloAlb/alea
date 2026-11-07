// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "estimators/bragg_order.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace alea {

BraggOrder::BraggOrder(std::span<const ParticleGroup> groups, const std::string& group,
                       const SimulationCell& cell, std::span<const double> wavevectors)
    : group_(group) {
    if (!cell.is_periodic() || cell.dim() != 2)
        throw std::runtime_error("bragg: requires a periodic 2-D cell");
    const auto selected = std::find_if(groups.begin(), groups.end(),
        [&](const auto& g) { return g.name == group; });
    if (group.empty() || selected == groups.end() || selected->last <= selected->first)
        throw std::runtime_error("bragg: requires a named nonempty particle group");
    first_ = selected->first; last_ = selected->last;
    if (wavevectors.empty() || wavevectors.size() % 2 != 0)
        throw std::runtime_error("bragg: wavevectors must contain nonempty Cartesian [kx,ky] pairs");
    const auto cell_vectors = cell.vectors();
    const auto reciprocal = cell.reciprocal();
    std::vector<std::pair<double, double>> indices;
    for (std::size_t i = 0; i < wavevectors.size(); i += 2) {
        const double x = wavevectors[i], y = wavevectors[i + 1];
        if (!std::isfinite(x) || !std::isfinite(y))
            throw std::runtime_error("bragg: wavevectors must be finite");
        const double n1 = (x * cell_vectors[0] + y * cell_vectors[1]) / (2 * std::numbers::pi);
        const double n2 = (x * cell_vectors[2] + y * cell_vectors[3]) / (2 * std::numbers::pi);
        const double a = std::round(n1), b = std::round(n2);
        if (!std::isfinite(n1) || !std::isfinite(n2) ||
            std::abs(n1 - a) > 1e-9 || std::abs(n2 - b) > 1e-9 ||
            (a == 0. && b == 0.))
            throw std::runtime_error("bragg: each wavevector must be a nonzero cell reciprocal vector");
        for (const auto& [old_a, old_b] : indices)
            if ((a == old_a && b == old_b) || (a == -old_a && b == -old_b))
                throw std::runtime_error("bragg: duplicate wavevector intensity (including +/- pair)");
        indices.emplace_back(a, b);
        // Canonical reciprocal representation guarantees exact cell periodicity
        // to floating-point precision; input may contain printed roundoff.
        const double kx = a * reciprocal[0] + b * reciprocal[2];
        const double ky = a * reciprocal[1] + b * reciprocal[3];
        if (!std::isfinite(kx) || !std::isfinite(ky))
            throw std::runtime_error("bragg: reciprocal wavevector overflow");
        kx_.push_back(kx); ky_.push_back(ky);
    }
    accumulated_.resize(kx_.size());
    block_values_.resize(1 + kx_.size());
    moments_.configure(1 + kx_.size());
}

void BraggOrder::configure_reblocking(std::size_t blocks) {
    if (reblocking_storage_bytes(blocks) > EstimatorReblocking::kMaxHistoryBytes)
        throw std::runtime_error("bragg: history including signs exceeds 64 MiB per replica");
    reblocking_.configure(blocks, 1 + kx_.size());
}

std::size_t BraggOrder::reblocking_storage_bytes(std::size_t blocks) const {
    // The driver's aggregate admission happens before configure_signed_weights.
    // Reserve the possible denominator even for unsigned runs, a conservative
    // bound that remains valid if the driver subsequently enables signed mode.
    return EstimatorReblocking::storage_bytes(blocks, 2 + kx_.size());
}

void BraggOrder::skip_measurement(std::size_t n) {
    if (!reblocking_.signed_mode())
        throw std::logic_error("bragg: skip_measurement needs signed weights");
    absolute_weight_ += static_cast<double>(n);
}

void BraggOrder::start_block() {
    std::fill(accumulated_.begin(), accumulated_.end(), 0.);
    weight_ = absolute_weight_ = 0.;
}

void BraggOrder::accumulate(const ParticleBatch& configs, std::span<const double> weights) {
    const std::size_t nw = configs.num_walkers();
    if (configs.dim() != 2 || configs.n_particles() < last_ || weights.size() != nw)
        throw std::runtime_error("bragg: incompatible configuration/weight shape");
    // Signed (fermionic) PIMC passes the permutation sign as the weight.
    for (const double w : weights)
        if (!std::isfinite(w) || (w < 0. && !reblocking_.signed_mode()))
            throw std::runtime_error("bragg: weights must be finite and nonnegative");
    const auto positions = configs.positions();  // [particle, dimension, walker]
    const double inverse_n2 = 1. / std::pow(static_cast<double>(last_ - first_), 2);
    for (std::size_t k = 0; k < kx_.size(); ++k) {
        for (std::size_t w = 0; w < nw; ++w) {
            double re = 0., im = 0.;
            for (std::size_t p = first_; p < last_; ++p) {
                const double phase = kx_[k] * positions[(p * 2) * nw + w] +
                                     ky_[k] * positions[(p * 2 + 1) * nw + w];
                re += std::cos(phase); im += std::sin(phase);
            }
            accumulated_[k] += weights[w] * (re * re + im * im) * inverse_n2;
        }
    }
    for (const double w : weights) weight_ += w;
    absolute_weight_ += static_cast<double>(nw);
}

void BraggOrder::stop_block() {
    const double normalization = reblocking_.signed_mode() ? absolute_weight_ : weight_;
    std::fill(block_values_.begin(), block_values_.end(), 0.);
    if (normalization > 0.) {
        for (std::size_t k = 0; k < kx_.size(); ++k) {
            block_values_[1 + k] = accumulated_[k] / normalization;
            block_values_[0] += block_values_[1 + k] / static_cast<double>(kx_.size());
        }
    }
    if (reblocking_.enabled()) {
        reblocking_.append(block_values_, reblocking_.signed_mode() && absolute_weight_ > 0.
                                              ? weight_ / absolute_weight_ : 1.);
        return;
    }
    // VMC/DMC: a block without weight has no estimate (a zero would bias B).
    if (!(normalization > 0.) || !std::isfinite(normalization))
        throw std::runtime_error("bragg: block requires positive finite accumulated weight");
    moments_.append(block_values_);
}

void BraggOrder::collect(const Estimator& other) {
    const auto* source = dynamic_cast<const BraggOrder*>(&other);
    if (!source) throw std::runtime_error("bragg: estimator type mismatch");
    if (group_ != source->group_ || first_ != source->first_ || last_ != source->last_ ||
        kx_ != source->kx_ || ky_ != source->ky_ ||
        reblocking_.enabled() != source->reblocking_.enabled())
        throw std::runtime_error("bragg: cannot collect different groups, wavevectors or samplers");
    if (reblocking_.enabled())
        reblocking_.collect(source->reblocking_);
    else
        moments_.collect(source->moments_);
}

void BraggOrder::finalize(RunResult& out) const {
    const std::string name = "bragg_" + group_;
    if (!reblocking_.enabled()) {
        if (moments_.blocks() == 0) return;
        std::vector<double> directions(kx_.size()), errors(kx_.size());
        for (std::size_t k = 0; k < kx_.size(); ++k) {
            directions[k] = moments_.mean(1 + k);
            errors[k] = moments_.sem(1 + k);
        }
        out.add_series(name, {moments_.mean(0)});
        out.add_series(name + "_error", {moments_.sem(0)});
        out.add_series(name + "_directions", std::move(directions));
        out.add_series(name + "_directions_error", std::move(errors));
        out.add_series(name + "_kx", kx_);
        out.add_series(name + "_ky", ky_);
        out.add_metadata(name + "_error_model", "independent blocks (no reblocking)");
        return;
    }
    reblocking_.emit(out, name, 0, 1);
    reblocking_.emit(out, name + "_directions", 1, kx_.size());
    out.add_series(name + "_kx", kx_);
    out.add_series(name + "_ky", ky_);
    if (reblocking_.signed_mode())
        reblocking_.emit_ratio_history(out, name, 0, 1., name + "_denominator_blocks");
    else
        reblocking_.emit_unsigned_history(out, name + "_blocks", 0);
}

}  // namespace alea
