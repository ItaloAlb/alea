// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Electron-hole proximity is the diagonal configuration-averaged indicator
// U_e = (1/(M N_e)) sum_{k,e} 1[min_{h in named hole group} r_eh(k) >= r_cut]
// over the M configurations k of a measurement (VMC/DMC walkers, weighted by
// their weights; PIMC path slices); C_e = 1-U_e. The indicator is geometric,
// not an exciton binding diagnostic. PIMC block SEMs are synchronized over
// independent replicas, rather than obtained by concatenating their Markov
// chains (Flyvbjerg and Petersen, J. Chem. Phys. 91, 461 (1989)).
// Minimum-image distances come from ParticleBatch's table.
#include "estimators/electron_hole_proximity.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "numerics/statistics.h"
#include "population/particle_batch.h"
#include "qmc/run_result.h"

namespace alea {
namespace {

// Exact boundary geometry comparison, so a prebuilt table cannot accidentally
// change a periodic PBC observable into an open-boundary one during sampling.
bool same_cell(const SimulationCell& left, const SimulationCell& right) noexcept {
    if (left.boundary() != right.boundary()) return false;
    // Open cells carry no geometric data. A default-constructed 0-D cell is
    // equivalent to ParticleBatch's explicit open cell in the physical dim.
    if (!left.is_periodic()) return true;
    if (left.dim() != right.dim()) return false;
    const auto a = left.vectors();
    const auto b = right.vectors();
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}
}  // namespace

// Resolve the two charged species by exact names. All group ranges must be
// nonempty, uniquely named and disjoint, so min_h never includes an electron.
ElectronHoleProximity::ElectronHoleProximity(
    std::span<const ParticleGroup> groups, std::string electron_group,
    std::string hole_group, double r_cut, const SimulationCell& cell)
    : electron_group_(std::move(electron_group)),
      hole_group_(std::move(hole_group)),
      output_prefix_(type() + "_" + electron_group_ + "_" + hole_group_),
      r_cut_(r_cut), cell_(cell) {
    if (!std::isfinite(r_cut_) || r_cut_ <= 0.0)
        throw std::runtime_error("electron_hole_proximity: r_cut must be positive and finite");
    if (cell_.is_periodic() && r_cut_ > cell_.ws_radius())
        throw std::runtime_error("electron_hole_proximity: r_cut exceeds periodic Wigner-Seitz inradius");
    if (electron_group_.empty() || hole_group_.empty() ||
        electron_group_ == hole_group_)
        throw std::runtime_error("electron_hole_proximity: distinct named electron and hole groups required");

    const ParticleGroup* electrons = nullptr;
    const ParticleGroup* holes = nullptr;
    for (std::size_t g = 0; g < groups.size(); ++g) {
        const ParticleGroup& current = groups[g];
        if (current.name.empty() || current.first >= current.last)
            throw std::runtime_error("electron_hole_proximity: group names and ranges must be nonempty");
        for (std::size_t j = 0; j < g; ++j) {
            const ParticleGroup& previous = groups[j];
            if (previous.name == current.name)
                throw std::runtime_error("electron_hole_proximity: duplicate group name");
            if (current.first < previous.last && previous.first < current.last)
                throw std::runtime_error("electron_hole_proximity: overlapping group ranges");
        }
        if (current.name == electron_group_) electrons = &current;
        if (current.name == hole_group_) holes = &current;
        required_particles_ = std::max(required_particles_, current.last);
    }
    if (!electrons || !holes)
        throw std::runtime_error("electron_hole_proximity: requested electron or hole group missing");
    if (!std::isfinite(electrons->charge) || electrons->charge >= 0.0 ||
        !std::isfinite(holes->charge) || holes->charge <= 0.0)
        throw std::runtime_error("electron_hole_proximity: electron charge must be negative and hole charge positive");
    electron_first_ = electrons->first;
    electron_last_ = electrons->last;
    hole_first_ = holes->first;
    hole_last_ = holes->last;
}

// Opt-in bounded flat [block, U_e/C_e] histories only before accumulation.
void ElectronHoleProximity::configure_reblocking(std::size_t num_blocks) {
    if (n_blocks_ != 0 || block_open_)
        throw std::runtime_error("electron_hole_proximity: configure reblocking before accumulation");
    reblocking_.configure(num_blocks, 2);
}

// Report bytes per independent replica, checking arithmetic overflow (the
// actual 64-MiB cap is checked when configure_reblocking reserves the budget).
std::size_t ElectronHoleProximity::reblocking_storage_bytes(
    std::size_t num_blocks) const {
    return EstimatorReblocking::storage_bytes(num_blocks, 2);
}

// Initialize block sums; no allocation or global mutable state.
void ElectronHoleProximity::start_block() {
    if (block_open_)
        throw std::runtime_error("electron_hole_proximity: previous block is not finished");
    block_free_weight_ = 0.0;
    block_config_weight_ = 0.0;
    block_config_abs_weight_ = 0.0;
    block_open_ = true;
}

// Check batch shape, exact cell geometry and weights before touching any
// per-pair distance, so malformed inputs cannot index outside the table.
void ElectronHoleProximity::validate_batch(
    const ParticleBatch& configs, std::span<const double> weights) const {
    const auto& distances = configs.dist_tables();
    if (!block_open_ || configs.n_particles() < required_particles_ ||
        configs.num_walkers() == 0 ||
        configs.dim() == 0 || configs.dim() != distances.dim() ||
        configs.num_walkers() != distances.nw() ||
        configs.n_particles() != distances.np() ||
        weights.size() != configs.num_walkers() ||
        !same_cell(configs.cell(), cell_) ||
        !same_cell(distances.cell(), cell_) ||
        (cell_.dim() != 0 && cell_.dim() != configs.dim()))
        throw std::runtime_error("electron_hole_proximity: measurement batch/weights/cell mismatch");
    // Signed (fermionic) runs pass the permutation sign as the weight.
    for (double weight : weights)
        if (!std::isfinite(weight) || (weight < 0.0 && !reblocking_.signed_mode()))
            throw std::runtime_error("electron_hole_proximity: weights must be finite and nonnegative");
}

// For every slice k and electron e, count exactly one if NO named hole h has
// r_eh(k) < r_cut. The distance table implements the nearest periodic image.
// Weighted samples reduce to the advertised 1/(M*N_e) average when weights=1;
// otherwise they use the standard estimator-weighted configuration mean.
// Walkers are fastest in each table row but small-group early exits avoid an
// allocated O(M) occupancy buffer in this per-measurement path. Thread-local
// estimator clones and replica paths allow independent-replica weak scaling;
// both weak- and strong-scaling behavior remain unmeasured.
void ElectronHoleProximity::accumulate(
    const ParticleBatch& configs, std::span<const double> weights) {
    validate_batch(configs, weights);
    const auto& distances = configs.dist_tables();
    const std::size_t slices = configs.num_walkers();
    const double inv_electrons =
        1.0 / static_cast<double>(electron_last_ - electron_first_);
    for (std::size_t k = 0; k < slices; ++k) {
        const double weight = weights[k];
        if (weight == 0.0) continue;
        std::size_t free_electrons = 0;
        for (std::size_t e = electron_first_; e < electron_last_; ++e) {
            bool no_hole_inside = true;
            for (std::size_t h = hole_first_; h < hole_last_; ++h) {
                if (distances(k, e, h) < r_cut_) {
                    no_hole_inside = false;
                    break;
                }
            }
            free_electrons += no_hole_inside;
        }
        block_free_weight_ += weight *
                              static_cast<double>(free_electrons) * inv_electrons;
        block_config_weight_ += weight;
        block_config_abs_weight_ += 1.0;  // = |w| for s = +/-1
    }
}

// Close a nonempty block. Append the two normalized indicators as one bounded
// record, or update legacy constant-space Welford moments if not opted in.
void ElectronHoleProximity::stop_block() {
    const bool signed_block = reblocking_.signed_mode();
    const double norm_weight = signed_block ? block_config_abs_weight_ : block_config_weight_;
    if (!block_open_ || !std::isfinite(norm_weight) ||
        norm_weight <= 0.0 || !std::isfinite(block_free_weight_))
        throw std::runtime_error("electron_hole_proximity: block requires positive finite accumulated weight");
    const double u = block_free_weight_ / norm_weight;
    // Signed: C = sum w (1 - free)/sum|w| = <s>_block - u, so C/<s> = 1 - U.
    const double sign_mean = signed_block ? block_config_weight_ / norm_weight : 1.0;
    const double values[] = {u, sign_mean - u};
    if (reblocking_.enabled()) {
        reblocking_.append(values, sign_mean);
    } else {
        ++n_blocks_;
        const double delta = u - mean_u_;
        mean_u_ += delta / static_cast<double>(n_blocks_);
        m2_u_ += delta * (u - mean_u_);
    }
    block_open_ = false;
}

// Reject unlike physical observables before statistical reduction.
void ElectronHoleProximity::validate_compatible(
    const ElectronHoleProximity& other) const {
    if (electron_group_ != other.electron_group_ ||
        hole_group_ != other.hole_group_ ||
        electron_first_ != other.electron_first_ ||
        electron_last_ != other.electron_last_ ||
        hole_first_ != other.hole_first_ || hole_last_ != other.hole_last_ ||
        required_particles_ != other.required_particles_ ||
        r_cut_ != other.r_cut_ || !same_cell(cell_, other.cell_) ||
        block_open_ || other.block_open_ ||
        reblocking_.enabled() != other.reblocking_.enabled())
        throw std::runtime_error("electron_hole_proximity: incompatible estimator clones");
}

// Reblocking merges equal temporal indices of independent replicas, never
// concatenated chains. Legacy output merges moments by Chan's parallel rule.
void ElectronHoleProximity::collect(const Estimator& other) {
    const auto* source = dynamic_cast<const ElectronHoleProximity*>(&other);
    if (!source) throw std::runtime_error("electron_hole_proximity: estimator type mismatch");
    validate_compatible(*source);
    if (reblocking_.enabled()) {
        reblocking_.collect(source->reblocking_);
        return;
    }
    if (source->n_blocks_ == 0) return;
    if (n_blocks_ == 0) {
        n_blocks_ = source->n_blocks_;
        mean_u_ = source->mean_u_;
        m2_u_ = source->m2_u_;
        return;
    }
    if (n_blocks_ > std::numeric_limits<std::size_t>::max() - source->n_blocks_)
        throw std::runtime_error("electron_hole_proximity: collected block count overflow");
    const double n_left = static_cast<double>(n_blocks_);
    const double n_right = static_cast<double>(source->n_blocks_);
    const double delta = source->mean_u_ - mean_u_;
    const double n_total = n_left + n_right;
    mean_u_ += delta * (n_right / n_total);
    m2_u_ += source->m2_u_ + delta * delta * n_left * n_right / n_total;
    n_blocks_ += source->n_blocks_;
}

// Emit threshold, both exact named species, finite-size and non-binding caveats
// alongside U_e/C_e. Correlation-corrected SEM and reliability diagnostics are
// emitted only after an explicitly configured PIMC reblocking history completes.
void ElectronHoleProximity::finalize(RunResult& out) const {
    if (!reblocking_.enabled() && n_blocks_ == 0) return;
    if (block_open_)
        throw std::runtime_error("electron_hole_proximity: cannot finalize an open block");
    const std::string u_name = output_prefix_ + "_U_e";
    const std::string c_name = output_prefix_ + "_C_e";
    if (reblocking_.enabled()) {
        // emit validates complete, equal-length histories before output.
        reblocking_.emit(out, u_name, 0, 1);
        reblocking_.emit(out, c_name, 1, 1);
    } else {
        const double sem = n_blocks_ > 1
            ? std::sqrt(m2_u_ / (static_cast<double>(n_blocks_ - 1) *
                                 static_cast<double>(n_blocks_))) : 0.0;
        out.add_series(u_name, {mean_u_});
        out.add_series(u_name + "_error", {sem});
        out.add_series(c_name, {1.0 - mean_u_});
        out.add_series(c_name + "_error", {sem});
    }
    out.add_scalar(output_prefix_ + "_r_cut", r_cut_);
    out.add_metadata(output_prefix_ + "_electron_group", electron_group_);
    out.add_metadata(output_prefix_ + "_hole_group", hole_group_);
    out.add_metadata(output_prefix_ + "_boundary",
                     cell_.is_periodic() ? "periodic" : "open");
    out.add_metadata(output_prefix_ + "_measure",
                     "U_e: fraction of electron configurations (walkers or path "
                     "slices) without any hole inside r_cut; C_e=1-U_e; "
                     "proximity is not bound exciton fraction");
    out.add_metadata(output_prefix_ + "_finite_size_caveat",
                     "finite simulation cell/particle number and explicit r_cut "
                     "alter the geometric baseline; statistical SEM excludes "
                     "finite-size and time-step bias");
}

// Deep-copy the configured template (zero history until first block) or an
// existing independent replica, with no shared scratch or mutable state.
std::unique_ptr<Estimator> ElectronHoleProximity::clone() const {
    return std::make_unique<ElectronHoleProximity>(*this);
}

}  // namespace alea
