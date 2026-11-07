// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "estimators/pair_correlation.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#include "numerics/statistics.h"
#include "population/walker_set.h"
#include "qmc/run_result.h"

namespace alea {

namespace {
constexpr double kPi = 3.14159265358979323846;

// Geometric measure of the radial bin [bin·Δ, (bin+1)·Δ): the d-dimensional
// shell volume.  2D → annulus π(r_hi²−r_lo²); 3D → spherical shell
// (4π/3)(r_hi³−r_lo³).  This is the ideal-gas factor that turns a raw pair
// histogram into g(r) (cf. PairCorrEstimator::norm_factor).
double shell_volume(std::size_t dim, double delta, std::size_t bin) noexcept {
    const double r_lo = static_cast<double>(bin) * delta;
    const double r_hi = static_cast<double>(bin + 1) * delta;
    if (dim == 2) return kPi * (r_hi * r_hi - r_lo * r_lo);
    if (dim == 3)
        return (4.0 / 3.0) * kPi *
               (r_hi * r_hi * r_hi - r_lo * r_lo * r_lo);
    return r_hi - r_lo;  // 1D / fallback: bin width
}
}  // namespace

PairCorrelation::PairCorrelation(std::span<const ParticleGroup> groups,
                                 std::size_t dim, double rmax, int n_bins,
                                 const SimulationCell& cell)
    : dim_(dim),
      rmax_(rmax),
      n_bins_(static_cast<std::size_t>(n_bins)),
      delta_(rmax / static_cast<double>(n_bins)),
      inv_delta_(static_cast<double>(n_bins) / rmax),
      n_groups_(groups.size()),
      n_func_(groups.size() * (groups.size() + 1) / 2) {
    // Copy group ranges so the estimator is self-contained (and clone-able).
    np_ = 0;
    for (const ParticleGroup& g : groups) {
        groups_.push_back({g.name, g.first, g.last});
        np_ = std::max(np_, g.last);
    }

    // particle → group map for the binning kernel.
    group_of_.assign(np_, 0);
    for (std::size_t gi = 0; gi < n_groups_; ++gi)
        for (std::size_t p = groups_[gi].first; p < groups_[gi].last; ++p)
            group_of_[p] = gi;

    // Per-function pair counts (AA: nα(nα−1)/2, AB: nα·nβ) and series names.
    pair_count_.assign(n_func_, 0);
    names_.assign(n_func_, std::string());
    for (std::size_t a = 0; a < n_groups_; ++a)
        for (std::size_t b = a; b < n_groups_; ++b) {
            const std::size_t pid = pair_id(a, b);
            const std::size_t na  = groups_[a].last - groups_[a].first;
            const std::size_t nb  = groups_[b].last - groups_[b].first;
            pair_count_[pid] = (a == b) ? na * (na - 1) / 2 : na * nb;
            names_[pid]      = "gofr_" + groups_[a].name + "_" + groups_[b].name;
        }

    block_hist_.assign(n_func_ * n_bins_, 0.0);

    if (cell.is_periodic()) {
        // Beyond the Wigner-Seitz inradius the minimum-image separation is no
        // longer unique — the histogram would mix shells and the ideal-gas
        // measure below would stop being the right one.  Reject rather than
        // quietly report a distorted tail.
        if (rmax > cell.ws_radius())
            throw std::runtime_error(
                "gofr rmax " + std::to_string(rmax) +
                " exceeds the cell's Wigner-Seitz inradius " +
                std::to_string(cell.ws_radius()) +
                "; minimum-image distances are not unique beyond it");
        volume_ = cell.volume();
    }
}

// Triangular index for the unordered group pair, mirroring
// PairCorrEstimator::gen_pair_id: row a holds (n_groups − a) entries.
std::size_t PairCorrelation::pair_id(std::size_t ga,
                                     std::size_t gb) const noexcept {
    if (gb < ga) std::swap(ga, gb);
    return ga * n_groups_ - ga * (ga - 1) / 2 + (gb - ga);
}

void PairCorrelation::start_block() {
    std::fill(block_hist_.begin(), block_hist_.end(), 0.0);
    block_weight_sum_ = 0.0;
    block_abs_weight_sum_ = 0.0;
}

void PairCorrelation::accumulate(const ParticleBatch& configs,
                                 std::span<const double> weights) {
    const BatchedDistanceTable&   dt      = configs.dist_tables();
    const std::size_t             nw      = configs.num_walkers();

    // All unordered particle pairs (i<j); each maps to its group-pair function.
    for (std::size_t i = 0; i < np_; ++i) {
        const std::size_t gi = group_of_[i];
        for (std::size_t j = i + 1; j < np_; ++j) {
            const std::size_t pid = pair_id(gi, group_of_[j]);
            double* __restrict h  = block_hist_.data() + pid * n_bins_;
            const std::span<const double> row = dt.row(i, j);
            // Unit-stride over walkers; data-dependent bin write is the scatter.
            for (std::size_t w = 0; w < nw; ++w) {
                const double r = row[w];
                if (r < rmax_) {
                    const std::size_t bin =
                        static_cast<std::size_t>(r * inv_delta_);
                    if (bin < n_bins_) h[bin] += weights[w];
                }
            }
        }
    }

    // Per-configuration sample weight, shared across all group-pair functions.
    double wsum = 0.0;
    for (std::size_t w = 0; w < nw; ++w) wsum += weights[w];
    block_weight_sum_ += wsum;
    block_abs_weight_sum_ += static_cast<double>(nw);  // = sum|w| for s = +/-1
}

void PairCorrelation::configure_reblocking(std::size_t num_blocks) {
    if (!block_history_.empty()) throw std::runtime_error("gofr: configure reblocking before accumulation");
    reblocking_.configure(num_blocks, n_func_ * n_bins_);
}

std::size_t PairCorrelation::reblocking_storage_bytes(std::size_t num_blocks) const {
    return EstimatorReblocking::storage_bytes(num_blocks, n_func_ * n_bins_);
}

void PairCorrelation::stop_block() {
    std::vector<double> blk(n_func_ * n_bins_, 0.0);
    // A signed block is normalized by sum|w| and carries its mean sign.
    const bool signed_block = reblocking_.signed_mode();
    const double norm_weight = signed_block ? block_abs_weight_sum_ : block_weight_sum_;
    if (norm_weight > 0.0) {
        const double inv_w = 1.0 / norm_weight;
        for (std::size_t k = 0; k < blk.size(); ++k)
            blk[k] = block_hist_[k] * inv_w;
    }
    if (reblocking_.enabled()) {
        reblocking_.append(blk, signed_block && block_abs_weight_sum_ > 0.0
                                    ? block_weight_sum_ / block_abs_weight_sum_ : 1.0);
        return;
    }
    block_history_.push_back(std::move(blk));
}

void PairCorrelation::collect(const Estimator& other) {
    const auto& o = dynamic_cast<const PairCorrelation&>(other);
    if (reblocking_.enabled() || o.reblocking_.enabled()) {
        reblocking_.collect(o.reblocking_);
        return;
    }
    block_history_.insert(block_history_.end(), o.block_history_.begin(),
                          o.block_history_.end());
}

void PairCorrelation::finalize(RunResult& out) const {
    if (block_history_.empty() && !reblocking_.enabled()) return;
    const std::size_t n_blocks = block_history_.size();

    // Bin centers (emitted once, shared by every gofr_* series).
    std::vector<double> r_axis(n_bins_);
    for (std::size_t b = 0; b < n_bins_; ++b)
        r_axis[b] = (static_cast<double>(b) + 0.5) * delta_;
    out.add_series("gofr_r", std::move(r_axis));

    std::vector<double> per_block(n_blocks);
    for (std::size_t pid = 0; pid < n_func_; ++pid) {
        if (pair_count_[pid] == 0) continue;  // empty function (no pairs)

        const double pc = static_cast<double>(pair_count_[pid]);
        if (reblocking_.enabled()) {
            std::vector<double> norms(n_bins_);
            for (std::size_t b = 0; b < n_bins_; ++b) {
                const double shell = shell_volume(dim_, delta_, b);
                norms[b] = volume_ > 0.0 ? pc * shell / volume_ : pc * shell;
            }
            reblocking_.emit(out, names_[pid], pid * n_bins_, n_bins_, norms);
            continue;
        }
        std::vector<double> g(n_bins_, 0.0);
        std::vector<double> g_err(n_bins_, 0.0);
        for (std::size_t b = 0; b < n_bins_; ++b) {
            for (std::size_t blk = 0; blk < n_blocks; ++blk)
                per_block[blk] = block_history_[blk][pid * n_bins_ + b];
            // g(r) = ⟨pairs-per-config in bin⟩ / (n_pairs · ideal-gas shell).
            // Periodic: also divide by the cell volume, i.e. measure against the
            // actual density ρ = N/V — that factor is what makes g(r) → 1 at
            // large r for a homogeneous fluid.  Open: no volume is defined (and
            // a bound complex has no bulk density), so the historical per-pair
            // normalisation stands and existing results are unchanged.
            const double shell = shell_volume(dim_, delta_, b);
            const double norm  = (volume_ > 0.0) ? pc * shell / volume_ : pc * shell;
            const numerics::MeanError stats   = numerics::block_mean_error(per_block);
            g[b]     = stats.mean / norm;
            g_err[b] = stats.error / norm;
        }
        out.add_series(names_[pid], std::move(g));
        out.add_series(names_[pid] + "_error", std::move(g_err));
    }
}

std::unique_ptr<Estimator> PairCorrelation::clone() const {
    return std::make_unique<PairCorrelation>(*this);
}

}  // namespace alea
