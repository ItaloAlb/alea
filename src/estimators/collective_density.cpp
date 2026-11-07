// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "estimators/collective_density.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "population/walker_set.h"
#include "qmc/run_result.h"

namespace alea {

CollectiveDensity::CollectiveDensity(std::span<const ParticleGroup> groups,
                                     std::size_t dim, double rmax, int n_bins,
                                     std::string name,
                                     std::span<const CollectiveTerm> terms)
    : name_(std::move(name)),
      dim_(dim),
      n_bins_(static_cast<std::size_t>(n_bins)),
      rmax_(rmax) {
    // Hard validation here (not in the parser): a bad spec must abort the run
    // loudly — a silently skipped estimator or a duplicate output key loses
    // data (results_writer drops the whole .h5 on a duplicate name).
    if (name_.empty())
        throw std::runtime_error(
            "collective_density: a non-empty `name` is required (it "
            "namespaces the cdensity_<name> output keys)");
    if (terms.empty())
        throw std::runtime_error("collective_density '" + name_ +
                                 "': `terms` is empty — give at least one "
                                 "{group, coeff} entry");
    if (rmax <= 0.0 || n_bins <= 0)
        throw std::runtime_error("collective_density '" + name_ +
                                 "': rmax and n_bins must be positive");

    delta_     = 2.0 * rmax / static_cast<double>(n_bins);
    inv_delta_ = static_cast<double>(n_bins) / (2.0 * rmax);

    // n_cells = n_bins^dim, capped so a config typo cannot silently allocate
    // gigabytes (same cap as Density).
    constexpr std::size_t max_cells = std::size_t{1} << 24;
    n_cells_ = 1;
    for (std::size_t d = 0; d < dim_; ++d) {
        n_cells_ *= n_bins_;
        if (n_cells_ > max_cells)
            throw std::runtime_error(
                "collective_density: n_bins^dim exceeds 2^24 cells — reduce "
                "n_bins");
    }

    cell_volume_ = 1.0;
    for (std::size_t d = 0; d < dim_; ++d) cell_volume_ *= delta_;

    // Resolve each term's group name into a particle range so accumulate is
    // string-free; copying the ranges keeps the estimator self-contained and
    // clone-able (same pattern as Density::GroupRange).
    for (const CollectiveTerm& t : terms) {
        const ParticleGroup* found = nullptr;
        for (const ParticleGroup& g : groups)
            if (g.name == t.group) {
                found = &g;
                break;
            }
        if (!found) {
            std::string available;
            for (const ParticleGroup& g : groups)
                available += (available.empty() ? "" : ", ") + g.name;
            throw std::runtime_error("collective_density '" + name_ +
                                     "': unknown group '" + t.group +
                                     "' (available: " + available + ")");
        }
        terms_.push_back({t.coeff, found->first, found->last});
    }

    block_hist_.assign(n_cells_, 0.0);
    mean_.assign(n_cells_, 0.0);
    m2_.assign(n_cells_, 0.0);
}

void CollectiveDensity::start_block() {
    std::fill(block_hist_.begin(), block_hist_.end(), 0.0);
    block_weight_sum_ = 0.0;
    block_abs_weight_sum_ = 0.0;
}

void CollectiveDensity::accumulate(const ParticleBatch& configs,
                                   std::span<const double> weights) {
    const std::span<const double> pos      = configs.positions();
    const std::size_t             nw       = configs.num_walkers();
    const double                  two_rmax = 2.0 * rmax_;

    // One collective coordinate per walker: the reduction over terms and
    // particles happens per walker, so the walker loop is outermost (unlike
    // Density's particle-outer nest).  SoA positions [p*dim*nw + d*nw + w]
    // keep each (p, d) stream unit-stride in w and cached across the small
    // terms×dim inner nest.  dim ≤ 3 → stack scratch, no allocations.
    double wsum = 0.0;
    for (std::size_t w = 0; w < nw; ++w) {
        double v[3] = {0.0, 0.0, 0.0};
        for (const ResolvedTerm& t : terms_)
            for (std::size_t p = t.first; p < t.last; ++p) {
                const double* __restrict base = pos.data() + p * dim_ * nw;
                for (std::size_t d = 0; d < dim_; ++d)
                    v[d] += t.coeff * base[d * nw + w];
            }

        std::size_t cell   = 0;
        bool        in_box = true;
        for (std::size_t d = 0; d < dim_; ++d) {
            // Shift [−rmax, +rmax) → [0, 2·rmax) before binning.
            const double x = v[d] + rmax_;
            if (x < 0.0 || x >= two_rmax) {
                in_box = false;
                break;
            }
            std::size_t b = static_cast<std::size_t>(x * inv_delta_);
            if (b >= n_bins_) b = n_bins_ - 1;  // FP rounding at edge
            cell = cell * n_bins_ + b;          // C row-major
        }
        if (in_box) block_hist_[cell] += weights[w];
        wsum += weights[w];
    }

    // Per-configuration sample weight — counted even when v falls outside
    // the grid: the weight measures ⟨ψ_T|Φ⟩, not the grid occupancy.
    block_weight_sum_ += wsum;
    block_abs_weight_sum_ += static_cast<double>(nw);  // = sum|w| for s = +/-1
}

void CollectiveDensity::configure_reblocking(std::size_t num_blocks) {
    if (n_blocks_ != 0) throw std::runtime_error("collective_density: configure reblocking before accumulation");
    reblocking_.configure(num_blocks, mean_.size());
}

std::size_t CollectiveDensity::reblocking_storage_bytes(std::size_t num_blocks) const {
    return EstimatorReblocking::storage_bytes(num_blocks, mean_.size());
}

void CollectiveDensity::stop_block() {
    // Stream the block estimate x = counts/(W_block·V_cell) into per-cell
    // Welford accumulators — keep in sync with Density::stop_block.
    // Welford: μ_n = μ_{n−1} + (x−μ_{n−1})/n;  M2_n = M2_{n−1} + (x−μ_{n−1})(x−μ_n).
    // A signed block is normalized by sum|w| and carries its mean sign.
    const bool signed_block = reblocking_.signed_mode();
    const double norm_weight = signed_block ? block_abs_weight_sum_ : block_weight_sum_;
    const double norm = norm_weight > 0.0
                            ? 1.0 / (norm_weight * cell_volume_)
                            : 0.0;
    if (reblocking_.enabled()) {
        std::vector<double> block(mean_.size());
        for (std::size_t k = 0; k < block.size(); ++k)
            block[k] = block_hist_[k] * norm;
        reblocking_.append(block, signed_block && block_abs_weight_sum_ > 0.0
                                      ? block_weight_sum_ / block_abs_weight_sum_ : 1.0);
        ++n_blocks_;
        return;
    }
    ++n_blocks_;
    const double inv_n = 1.0 / static_cast<double>(n_blocks_);
    for (std::size_t k = 0; k < mean_.size(); ++k) {
        const double x = block_hist_[k] * norm;
        const double d = x - mean_[k];
        mean_[k] += d * inv_n;
        m2_[k] += d * (x - mean_[k]);
    }
}

void CollectiveDensity::collect(const Estimator& other) {
    // Chan et al. parallel merge of two Welford states — algebraically the
    // same mean/M2 as concatenating the two block histories (keep in sync
    // with Density::collect):
    //   δ = μ_b − μ_a;  μ = μ_a + δ·n_b/n;  M2 = M2_a + M2_b + δ²·n_a·n_b/n.
    const auto& o = dynamic_cast<const CollectiveDensity&>(other);
    if (reblocking_.enabled() || o.reblocking_.enabled()) {
        reblocking_.collect(o.reblocking_);
        n_blocks_ = reblocking_.blocks();
        return;
    }
    if (o.n_blocks_ == 0) return;
    if (n_blocks_ == 0) {
        mean_     = o.mean_;
        m2_       = o.m2_;
        n_blocks_ = o.n_blocks_;
        return;
    }
    const double na = static_cast<double>(n_blocks_);
    const double nb = static_cast<double>(o.n_blocks_);
    const double n  = na + nb;
    for (std::size_t k = 0; k < mean_.size(); ++k) {
        const double d = o.mean_[k] - mean_[k];
        mean_[k] += d * (nb / n);
        m2_[k] += o.m2_[k] + d * d * (na * nb / n);
    }
    n_blocks_ += o.n_blocks_;
}

void CollectiveDensity::finalize(RunResult& out) const {
    if (n_blocks_ == 0 && !reblocking_.enabled()) return;

    const std::string prefix = "cdensity_" + name_;

    // Bin edges (n_bins+1 values from −rmax to +rmax); one axis serves every
    // dimension of the cubic grid.
    std::vector<double> edges(n_bins_ + 1);
    for (std::size_t b = 0; b <= n_bins_; ++b)
        edges[b] = -rmax_ + static_cast<double>(b) * delta_;
    out.add_series(prefix + "_edges", std::move(edges));

    // Grid metadata, namespaced per instance so several collective densities
    // (and the plain density estimator) coexist in one results file.
    out.add_scalar(prefix + "_dim", static_cast<double>(dim_));
    out.add_scalar(prefix + "_n_bins", static_cast<double>(n_bins_));
    out.add_scalar(prefix + "_rmax", rmax_);
    out.add_scalar(prefix + "_n_blocks", static_cast<double>(n_blocks_));

    if (reblocking_.enabled()) {
        reblocking_.emit(out, prefix, 0, n_cells_);
        return;
    }

    // Per-cell block SEM = sqrt(M2/((n−1)·n)) — same convention as
    // numerics::block_mean_error; not estimable from a single block.
    const double n = static_cast<double>(n_blocks_);
    const double err_norm = n_blocks_ >= 2 ? 1.0 / ((n - 1.0) * n) : 0.0;

    std::vector<double> rho(n_cells_);
    std::vector<double> rho_err(n_cells_);
    for (std::size_t c = 0; c < n_cells_; ++c) {
        rho[c]     = mean_[c];
        rho_err[c] = std::sqrt(m2_[c] * err_norm);
    }
    out.add_series(prefix, std::move(rho));
    out.add_series(prefix + "_error", std::move(rho_err));
}

std::unique_ptr<Estimator> CollectiveDensity::clone() const {
    return std::make_unique<CollectiveDensity>(*this);
}

}  // namespace alea
