// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "estimators/bond_order.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>

namespace alea {

BondOrder::BondOrder(std::span<const ParticleGroup> groups, const std::string& group,
                     const SimulationCell& cell, double reference_spacing)
    : group_(group), cell_(cell), spacing_(reference_spacing) {
    if (!cell.is_periodic() || cell.dim() != 2)
        throw std::runtime_error("bond_order: requires a periodic 2-D cell");
    if (!std::isfinite(spacing_) || spacing_ <= 0. || !std::isfinite(spacing_*spacing_) || spacing_*spacing_ == 0.)
        throw std::runtime_error("bond_order: reference_spacing must be finite and positive with finite square");
    const auto selected = std::find_if(groups.begin(), groups.end(),
        [&](const auto& g) { return g.name == group; });
    if (group.empty() || selected == groups.end() || selected->last < selected->first ||
        selected->last-selected->first < 7)
        throw std::runtime_error("bond_order: requires a named group with at least seven particles");
    first_ = selected->first; last_ = selected->last;
    dx_.resize(last_-first_); dy_.resize(last_-first_); distance2_.resize(last_-first_);
    moments_.configure(3);
}

void BondOrder::configure_reblocking(std::size_t blocks) {
    if (reblocking_storage_bytes(blocks) > EstimatorReblocking::kMaxHistoryBytes)
        throw std::runtime_error("bond_order: history including signs exceeds 64 MiB per replica");
    reblocking_.configure(blocks, 3);
}

std::size_t BondOrder::reblocking_storage_bytes(std::size_t blocks) const {
    // Driver admission precedes signed-mode setup; reserve its denominator.
    return EstimatorReblocking::storage_bytes(blocks, 4);
}

void BondOrder::skip_measurement(std::size_t n) {
    if (!reblocking_.signed_mode())
        throw std::logic_error("bond_order: skip_measurement needs signed weights");
    absolute_weight_ += static_cast<double>(n);
}

void BondOrder::start_block() {
    accumulated_.fill(0.);
    weight_ = absolute_weight_ = 0.;
}

void BondOrder::accumulate(const ParticleBatch& configs, std::span<const double> weights) {
    if (configs.dim() != 2 || configs.n_particles() < last_ || weights.size() != configs.num_walkers())
        throw std::runtime_error("bond_order: incompatible configuration/weight shape");
    const auto nw = configs.num_walkers(), n = last_-first_;
    const auto positions = configs.positions();  // [particle, dimension, walker]
    for (std::size_t w = 0; w < nw; ++w) {
        // Signed (fermionic) PIMC passes the permutation sign as the weight.
        if (!std::isfinite(weights[w]) || (weights[w] < 0. && !reblocking_.signed_mode()))
            throw std::runtime_error("bond_order: weights must be finite and nonnegative");
        if (weights[w] == 0.) continue;
        std::complex<double> global{};
        double local = 0., distortion = 0.;
        for (std::size_t p = first_; p < last_; ++p) {
            std::array<double, 6> nearest;  // six distance order statistics
            nearest.fill(std::numeric_limits<double>::infinity());
            for (std::size_t j = first_; j < last_; ++j) {
                if (j == p) continue;
                double delta[2]{positions[(j*2)*nw+w]-positions[(p*2)*nw+w],
                                positions[(j*2+1)*nw+w]-positions[(p*2+1)*nw+w]};
                cell_.min_image(delta);
                const double r2 = delta[0]*delta[0]+delta[1]*delta[1];
                if (!std::isfinite(r2) || r2 <= 0.)
                    throw std::runtime_error("bond_order: coincident or nonfinite particles have undefined bond angles");
                const auto index = j-first_;
                dx_[index] = delta[0]; dy_[index] = delta[1]; distance2_[index] = r2;
                if (r2 < nearest.back()) {
                    nearest.back() = r2;
                    std::sort(nearest.begin(), nearest.end());
                }
            }
            const double cutoff = nearest.back();
            const double tolerance = 64*std::numeric_limits<double>::epsilon()*cutoff;
            std::size_t closer = 0, tied = 0;
            for (std::size_t j = first_; j < last_; ++j) {
                if (j == p) continue;
                const double r2 = distance2_[j-first_];
                closer += r2 < cutoff-tolerance;
                tied += std::abs(r2-cutoff) <= tolerance;
            }
            const double tie_weight = (6.-static_cast<double>(closer))/static_cast<double>(tied);
            std::complex<double> psi{};
            for (std::size_t j = first_; j < last_; ++j) {
                if (j == p) continue;
                const auto index = j-first_;
                const double r2 = distance2_[index];
                const double neighbor_weight = r2 < cutoff-tolerance ? 1. :
                    (std::abs(r2-cutoff) <= tolerance ? tie_weight : 0.);
                if (neighbor_weight == 0.) continue;
                const double radius = std::sqrt(r2);
                const std::complex<double> direction(dx_[index]/radius, dy_[index]/radius);
                const auto squared = direction*direction;
                psi += neighbor_weight*squared*squared*squared/6.;
                const double relative = radius/spacing_-1.;
                if (!std::isfinite(relative*relative))
                    throw std::runtime_error("bond_order: relative bond distortion overflow");
                distortion += neighbor_weight*relative*relative/(6.*static_cast<double>(n));
            }
            global += psi/static_cast<double>(n);
            local += std::norm(psi)/static_cast<double>(n);
        }
        accumulated_[0] += weights[w]*std::norm(global);
        accumulated_[1] += weights[w]*local;
        accumulated_[2] += weights[w]*distortion;
        weight_ += weights[w];
    }
    absolute_weight_ += static_cast<double>(nw);
}

void BondOrder::stop_block() {
    const double normalization = reblocking_.signed_mode() ? absolute_weight_ : weight_;
    std::array<double, 3> values{};
    if (normalization > 0.)
        for (std::size_t i = 0; i < values.size(); ++i) values[i] = accumulated_[i]/normalization;
    if (reblocking_.enabled()) {
        reblocking_.append(values, reblocking_.signed_mode() && absolute_weight_ > 0. ? weight_/absolute_weight_ : 1.);
        return;
    }
    // VMC/DMC: a block without weight has no estimate (zeros would bias it).
    if (!(normalization > 0.) || !std::isfinite(normalization))
        throw std::runtime_error("bond_order: block requires positive finite accumulated weight");
    moments_.append(values);
}

void BondOrder::collect(const Estimator& other) {
    const auto* source = dynamic_cast<const BondOrder*>(&other);
    if (!source) throw std::runtime_error("bond_order: estimator type mismatch");
    if (group_ != source->group_ || first_ != source->first_ || last_ != source->last_ ||
        spacing_ != source->spacing_ ||
        !std::equal(cell_.vectors().begin(), cell_.vectors().end(), source->cell_.vectors().begin()) ||
        reblocking_.enabled() != source->reblocking_.enabled())
        throw std::runtime_error("bond_order: cannot collect different groups, cells, reference spacings or samplers");
    if (reblocking_.enabled())
        reblocking_.collect(source->reblocking_);
    else
        moments_.collect(source->moments_);
}

void BondOrder::finalize(RunResult& out) const {
    if (!reblocking_.enabled() && moments_.blocks() == 0) return;
    const std::string prefix = "bond_"+group_;
    out.add_scalar(prefix+"_reference_spacing", spacing_);
    out.add_metadata(prefix+"_neighbors", "six nearest distinct particles, equal sharing of cutoff ties; minimum images");
    const std::array<std::string, 3> suffix{"global6", "local6", "distortion"};
    if (!reblocking_.enabled()) {
        for (std::size_t i = 0; i < suffix.size(); ++i) {
            const auto name = prefix+"_"+suffix[i];
            out.add_series(name, {moments_.mean(i)});
            out.add_series(name+"_error", {moments_.sem(i)});
        }
        out.add_metadata(prefix+"_error_model", "independent blocks (no reblocking)");
        return;
    }
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        const auto name = prefix+"_"+suffix[i];
        reblocking_.emit(out, name, i, 1);
        if (reblocking_.signed_mode())
            reblocking_.emit_ratio_history(out, name, i, 1., name+"_denominator_blocks");
        else
            reblocking_.emit_unsigned_history(out, name+"_blocks", i);
    }
}

}  // namespace alea
