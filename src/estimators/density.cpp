// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "estimators/density.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "population/walker_set.h"
#include "qmc/run_result.h"

namespace alea {

Density::Density(std::span<const ParticleGroup> groups, std::size_t dim,
                 double rmax, int n_bins, bool per_particle,
                 const SimulationCell& cell)
    : dim_(dim),
      n_bins_(static_cast<std::size_t>(n_bins)),
      n_groups_(groups.size()),
      rmax_(rmax),
      delta_(2.0 * rmax / static_cast<double>(n_bins)),
      inv_delta_(static_cast<double>(n_bins) / (2.0 * rmax)) {
    // n_cells = n_bins^dim, with a hard cap so a config typo (say n_bins: 512
    // in 3-D) cannot silently allocate gigabytes of histogram.
    constexpr std::size_t max_cells = std::size_t{1} << 24;
    n_cells_ = 1;
    for (std::size_t d = 0; d < dim_; ++d) {
        n_cells_ *= n_bins_;
        if (n_cells_ > max_cells)
            throw std::runtime_error(
                "density estimator: n_bins^dim exceeds 2^24 cells — reduce "
                "n_bins");
    }

    cell_ = cell;
    if (cell_.is_periodic()) {
        // The grid IS the simulation cell: n_bins^dim equal-volume boxes tiling
        // it, so ∫ρ dV recovers the group count exactly (nothing can leave).
        cell_volume_ = cell_.volume() / static_cast<double>(n_cells_);
    } else {
        cell_volume_ = 1.0;
        for (std::size_t d = 0; d < dim_; ++d) cell_volume_ *= delta_;
    }

    // Copy group ranges so the estimator is self-contained (and clone-able).
    // per_particle expands each group into single-particle entries: the
    // accumulate/normalize machinery is unchanged, only the ranges shrink.
    for (const ParticleGroup& g : groups) {
        if (per_particle) {
            for (std::size_t p = g.first; p < g.last; ++p)
                groups_.push_back(
                    {g.name + "_p" + std::to_string(p - g.first), p, p + 1});
        } else {
            groups_.push_back({g.name, g.first, g.last});
        }
    }
    n_groups_ = groups_.size();

    block_hist_.assign(n_groups_ * n_cells_, 0.0);
    mean_.assign(n_groups_ * n_cells_, 0.0);
    m2_.assign(n_groups_ * n_cells_, 0.0);
}

void Density::start_block() {
    std::fill(block_hist_.begin(), block_hist_.end(), 0.0);
    block_weight_sum_ = 0.0;
    block_abs_weight_sum_ = 0.0;
}

void Density::accumulate(const ParticleBatch& configs,
                         std::span<const double> weights) {
    const std::span<const double> pos     = configs.positions();
    const std::size_t             nw      = configs.num_walkers();
    const double                  two_rmax = 2.0 * rmax_;
    const bool                    periodic = cell_.is_periodic();

    for (std::size_t g = 0; g < n_groups_; ++g) {
        double* __restrict h = block_hist_.data() + g * n_cells_;
        for (std::size_t p = groups_[g].first; p < groups_[g].last; ++p) {
            // SoA positions [p*dim*nw + d*nw + w]: each dimension is a
            // unit-stride stream over walkers; the cell index is composed in
            // registers, so no scratch buffer is needed.  The data-dependent
            // cell write is the scatter (cf. PairCorrelation).
            const double* __restrict base = pos.data() + p * dim_ * nw;
            if (periodic) {
                // Bin the fractional coordinates: f ∈ [−½, ½) maps onto the
                // cell whatever its shape, and wrapping means every particle
                // lands in some bin — there is no outside to reject.
                for (std::size_t w = 0; w < nw; ++w) {
                    double r[SimulationCell::kMaxDim] = {};
                    double f[SimulationCell::kMaxDim] = {};
                    for (std::size_t d = 0; d < dim_; ++d) r[d] = base[d * nw + w];
                    cell_.wrap(r);
                    cell_.fractional(r, f);
                    std::size_t cell = 0;
                    for (std::size_t d = 0; d < dim_; ++d) {
                        double      x = f[d] + 0.5;  // [−½, ½) → [0, 1)
                        std::size_t b =
                            static_cast<std::size_t>(x * static_cast<double>(n_bins_));
                        if (b >= n_bins_) b = n_bins_ - 1;  // FP rounding at edge
                        cell = cell * n_bins_ + b;          // C row-major
                    }
                    h[cell] += weights[w];
                }
                continue;
            }
            for (std::size_t w = 0; w < nw; ++w) {
                std::size_t cell   = 0;
                bool        in_box = true;
                for (std::size_t d = 0; d < dim_; ++d) {
                    // Shift [−rmax, +rmax) → [0, 2·rmax) before binning.
                    const double x = base[d * nw + w] + rmax_;
                    if (x < 0.0 || x >= two_rmax) {
                        in_box = false;
                        break;
                    }
                    std::size_t b = static_cast<std::size_t>(x * inv_delta_);
                    if (b >= n_bins_) b = n_bins_ - 1;  // FP rounding at edge
                    cell = cell * n_bins_ + b;          // C row-major
                }
                if (in_box) h[cell] += weights[w];
            }
        }
    }

    // Per-configuration sample weight, shared across all groups.  Counted
    // even when a particle falls outside the grid: the weight measures
    // ⟨ψ_T|Φ⟩, not the grid occupancy.
    double wsum = 0.0;
    for (std::size_t w = 0; w < nw; ++w) wsum += weights[w];
    block_weight_sum_ += wsum;
    block_abs_weight_sum_ += static_cast<double>(nw);  // = sum|w| for s = +/-1
}

void Density::configure_reblocking(std::size_t num_blocks) {
    if (n_blocks_ != 0) throw std::runtime_error("density: configure reblocking before accumulation");
    reblocking_.configure(num_blocks, mean_.size());
}

std::size_t Density::reblocking_storage_bytes(std::size_t num_blocks) const {
    return EstimatorReblocking::storage_bytes(num_blocks, mean_.size());
}

void Density::stop_block() {
    // Stream the block estimate x = counts/(W_block·V_cell) into per-cell
    // Welford accumulators (n, mean, M2) instead of keeping a block history —
    // a 128² grid × 2048 blocks would otherwise cost ~0.5 GB per group.
    // Singleton Chan update: δ=x−μ_{n−1}, μ_n=μ_{n−1}+δ/n,
    // M2_n=M2_{n−1}+δ²(n−1)/n (Chan, Golub & LeVeque,
    // The American Statistician 37, 242–247 (1983), updating formula).
    // This is Welford's update with nonnegative increments. In the product
    // δ*(x−μ_n), FMA contraction of counts*norm−μ_n can leave a negative
    // roundoff residual even in the first, exactly zero-variance block.
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
    if (n_blocks_ == 1) {
        // Initialize the singleton directly; do not form δ²*0, which could
        // overflow for a large but finite first density.
        for (std::size_t k = 0; k < mean_.size(); ++k) {
            mean_[k] = block_hist_[k] * norm;
            m2_[k] = 0.0;
        }
        return;
    }
    const double inv_n = 1.0 / static_cast<double>(n_blocks_);
    const double variance_weight = static_cast<double>(n_blocks_ - 1) * inv_n;
    for (std::size_t k = 0; k < mean_.size(); ++k) {
        const double x = block_hist_[k] * norm;
        const double d = x - mean_[k];
        mean_[k] += d * inv_n;
        // Scale one factor first: δ² can overflow even when δ²*(n−1)/n fits.
        m2_[k] += d * (d * variance_weight);
    }
}

void Density::collect(const Estimator& other) {
    // Chan et al. parallel merge of two Welford states — algebraically the
    // same mean/M2 as concatenating the two block histories:
    //   δ = μ_b − μ_a;  μ = μ_a + δ·n_b/n;  M2 = M2_a + M2_b + δ²·n_a·n_b/n.
    const auto& o = dynamic_cast<const Density&>(other);
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
        // For two singletons na*nb/n=1/2: scale before squaring so a
        // representable M2 is not lost to an overflowing intermediate δ².
        m2_[k] += o.m2_[k] + d * (d * (na * nb / n));
    }
    n_blocks_ += o.n_blocks_;
}

void Density::finalize(RunResult& out) const {
    if (n_blocks_ == 0 && !reblocking_.enabled()) return;

    // Bin edges, n_bins+1 values along one axis (the grid is uniform, so one
    // axis serves every dimension).
    //
    // Open:     cartesian, −rmax … +rmax.
    // Periodic: FRACTIONAL, −½ … +½, because the grid is the cell itself rather
    //           than a fixed cartesian window.  The lattice vectors go out
    //           alongside them so a reader can map f → r = Σ_i f_i a_i; without
    //           that the edges would be silently misinterpreted as Bohr.
    std::vector<double> edges(n_bins_ + 1);
    if (cell_.is_periodic()) {
        for (std::size_t b = 0; b <= n_bins_; ++b)
            edges[b] = -0.5 + static_cast<double>(b) / static_cast<double>(n_bins_);
        const auto a = cell_.vectors();
        out.add_series("density_lattice", std::vector<double>(a.begin(), a.end()));
    } else {
        for (std::size_t b = 0; b <= n_bins_; ++b)
            edges[b] = -rmax_ + static_cast<double>(b) * delta_;
    }
    out.add_series("density_edges", std::move(edges));
    out.add_scalar("density_periodic", cell_.is_periodic() ? 1.0 : 0.0);

    // Grid metadata so the Python reader is self-describing.
    out.add_scalar("density_dim", static_cast<double>(dim_));
    out.add_scalar("density_n_bins", static_cast<double>(n_bins_));
    out.add_scalar("density_rmax", rmax_);
    out.add_scalar("density_n_blocks", static_cast<double>(n_blocks_));

    if (reblocking_.enabled()) {
        for (std::size_t g = 0; g < n_groups_; ++g)
            reblocking_.emit(out, "density_" + groups_[g].name, g * n_cells_, n_cells_);
        return;
    }

    // Per-cell block SEM = sqrt(M2/((n−1)·n)) — the same convention as
    // numerics::block_mean_error; not estimable from a single block.
    const double n = static_cast<double>(n_blocks_);
    const double err_norm = n_blocks_ >= 2 ? 1.0 / ((n - 1.0) * n) : 0.0;

    for (std::size_t g = 0; g < n_groups_; ++g) {
        std::vector<double> rho(n_cells_);
        std::vector<double> rho_err(n_cells_);
        for (std::size_t c = 0; c < n_cells_; ++c) {
            const std::size_t k = g * n_cells_ + c;
            rho[c]     = mean_[k];
            rho_err[c] = std::sqrt(m2_[k] * err_norm);
        }
        out.add_series("density_" + groups_[g].name, std::move(rho));
        out.add_series("density_" + groups_[g].name + "_error",
                       std::move(rho_err));
    }
}

std::unique_ptr<Estimator> Density::clone() const {
    return std::make_unique<Density>(*this);
}

}  // namespace alea
