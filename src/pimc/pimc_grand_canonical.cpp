// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Grand-canonical stage of the PIMC driver: sampling, sector bookkeeping and
// outputs (Boninsegni, Prokof'ev & Svistunov, Phys. Rev. E 74, 036701 (2006),
// "BPS"; the moves are GrandCanonicalWormMove).
//
// ── What is sampled ─────────────────────────────────────────────────────────
// Closed paths of N bosons in [n_min, n_max] with weight
//     P(N) ∝ e^{beta mu_in N} Z_N,
// Z_N the canonical finite-M primitive partition function, and open paths
// (BPS's G sector) that carry no diagonal measurement.  In Z the driver draws
// a move by weight: the worm runs with the capacity action (its bead energies
// count the beads present on each slice), every other move with the action of
// the current N, built for the group range [0, N), so that it is exactly the
// canonical move of N.  In G only the worm runs.
//
// ── What is measured ────────────────────────────────────────────────────────
// After each step (sub_steps attempts) the path is measured once; an open
// path counts with weight 0.  With w_b(N) the fraction of a block's
// measurements that were closed in N and O_b(N) the block mean of O [closed
// in N], every sector observable is the ratio estimator
//     <O>_N = sum_b O_b(N) / sum_b w_b(N),
// with a jackknife error over decorrelated coarse blocks
// (numerics::ratio_of_means): a block that never visits N has w_b(N) = 0
// instead of an undefined block mean.  The sector fractions
// P(N) = sum_b w_b(N) / sum_b w_b(closed) give
//     F(N + 1) - F(N) = mu_in - T ln[P(N + 1)/P(N)]
// (the canonical chemical potential at N + 1/2; BPS Sec. III), whose error
// follows from the jackknife error of the ratio of the two sector weights by
// the delta method, sigma_mu = T sigma_R / R.  The all-sector aggregates are ratios of sums:
// <E>, <K>, <V> over every closed path (grand-canonical averages), and the
// per-particle winding <N W^2>/<N>, never <W^2/N>.
#include "pimc/pimc.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <string>

#include "io/logger.h"
#include "pimc/grand_canonical_worm.h"
#include "pimc/winding.h"

namespace alea {

namespace {

// Per-replica block fields of the all-sector aggregate (closed paths only).
constexpr std::size_t kAggEnergy = 0, kAggKinetic = 1, kAggPotential = 2, kAggWinding2 = 3,
                      kAggClosed = 4, kAggN = 5, kAggN2 = 6, kAggTensor = 7,
                      kAggFields = 7 + kMaxWindingTensor;

std::string sector_label(std::size_t n) { return "N" + std::to_string(n); }

}  // namespace

void Pimc::setup_grand_canonical(const PimcGrandCanonicalSectors& sectors,
                                 const Action& base_action) {
    // The move set: exactly one worm (the grand-canonical one, see
    // build_moves) and no permutation move, which would reconnect the spare
    // rows of the capacity's exchange class.
    const PimcConfig::MoveSpec* worm_spec = nullptr;
    for (const PimcConfig::MoveSpec& spec : move_specs_) {
        if (spec.type == "permutation" || spec.type == "joint_permutation")
            throw std::runtime_error("pimc: a grand-canonical stage samples exchange with its "
                                     "worm; remove the `" + spec.type + "` move");
        if (spec.type != "worm") continue;
        if (worm_spec)
            throw std::runtime_error("pimc: a grand-canonical stage needs exactly one worm move");
        worm_spec = &spec;
    }
    if (!worm_spec || !worm_spec->grand_canonical)
        throw std::runtime_error("pimc: a grand-canonical stage needs a worm move with a "
                                 "grand_canonical block");
    const auto& window = *worm_spec->grand_canonical;
    gc_mu_ = window.chemical_potential;
    gc_n_min_ = static_cast<std::size_t>(window.n_min);
    gc_n_max_ = static_cast<std::size_t>(window.n_max);
    if (np_ != gc_n_max_)
        throw std::logic_error("pimc: a grand-canonical stage is built for its capacity n_max = " +
                               std::to_string(gc_n_max_) + ", not " + std::to_string(np_));

    // One bosonic species whose particles form one exchange class.
    if (groups_.size() != 1)
        throw std::runtime_error("pimc: a grand-canonical run samples one particle group; this "
                                 "system has " + std::to_string(groups_.size()));
    if (cfg_.exchange_classes.size() != 1 || cfg_.exchange_classes.front().statistics != "bosons" ||
        cfg_.exchange_classes.front().particles.size() != np_ ||
        cfg_.exchange_classes.front().group != worm_spec->group)
        throw std::runtime_error("pimc: a grand-canonical run samples one bosonic species whose "
                                 "particles form one exchange class: give the worm's group "
                                 "`type: bosons` and every particle in `up`");
    if (cfg_.fixed_permutation_sector)
        throw std::logic_error("pimc: the fixed permutation sector has no grand-canonical meaning");
    if (!base_action.supports_bead_action())
        throw std::runtime_error("pimc: the grand-canonical worm needs a bead energy for every "
                                 "potential term (moire, inverse_power, harmonic_trap, coulomb)");

    gc_initial_ = sectors.initial_particles;
    if (gc_initial_ < gc_n_min_ || gc_initial_ > gc_n_max_)
        throw std::runtime_error("pimc: the initial particle count " + std::to_string(gc_initial_) +
                                 " lies outside the grand-canonical window [" +
                                 std::to_string(gc_n_min_) + ", " + std::to_string(gc_n_max_) + "]");
    const std::size_t n_sectors = gc_n_max_ - gc_n_min_ + 1;
    if (sectors.actions.size() != n_sectors)
        throw std::logic_error("pimc: one sector action per N in the window is required");
    std::vector<Action> prepared;
    prepared.reserve(n_sectors);
    for (std::size_t i = 0; i < n_sectors; ++i) {
        const std::size_t n = gc_n_min_ + i;
        if (n == 0) {
            prepared.emplace_back();
            continue;
        }
        const Action& a = sectors.actions[i];
        if (a.dim() != dim_ || a.np() != n)
            throw std::logic_error("pimc: the sector action of N = " + std::to_string(n) +
                                   " must act on N particles");
        if (!a.supports_free_bridge() || !a.slice_local_deltas() || !a.restricted_blocks().empty())
            throw std::runtime_error("pimc: a grand-canonical run needs the primitive action in "
                                     "every sector");
        for (std::size_t d = 0; d < dim_; ++d)
            if (a.mass(0, d) != base_action.mass(0, d))
                throw std::logic_error("pimc: the sector actions must share the species' mass");
        a.validate_sampling(cell_, cfg_.time_step);
        prepared.push_back(a.clone());
        prepared.back().prepare_time_step(cfg_.time_step);
    }
    sector_actions_.clear();
    sector_actions_.resize(static_cast<std::size_t>(cfg_.samples));
    for (auto& replica : sector_actions_) {
        replica.reserve(n_sectors);
        for (const Action& a : prepared) replica.push_back(a.clone());
    }
}

void Pimc::set_sector_estimators(std::vector<std::vector<std::unique_ptr<Estimator>>> per_sector) {
    if (!gc_)
        throw std::logic_error("pimc: sector estimators belong to a grand-canonical run");
    const std::size_t n_sectors = gc_n_max_ - gc_n_min_ + 1;
    if (per_sector.size() != n_sectors)
        throw std::logic_error("pimc: one estimator set per N in the window is required");
    // The aggregate history budget of set_estimators, counting every sector.
    constexpr std::size_t kMaxAllReplicaHistoryBytes = 256 * 1024 * 1024;
    const auto blocks = static_cast<std::size_t>(cfg_.accumulation_blocks);
    std::size_t per_replica = 0;
    for (const auto& ests : per_sector)
        for (const auto& est : ests) {
            if (!est) throw std::runtime_error("pimc: null estimator");
            const auto bytes = est->reblocking_storage_bytes(blocks);
            if (bytes > kMaxAllReplicaHistoryBytes - per_replica)
                throw std::runtime_error("pimc: optional-observable history of the " +
                                         std::to_string(n_sectors) + " sectors exceeds the "
                                         "aggregate 256 MiB budget; reduce bins, blocks, "
                                         "replicas or the window");
            per_replica += bytes;
        }
    if (per_replica > kMaxAllReplicaHistoryBytes / (static_cast<std::size_t>(cfg_.samples) + 1))
        throw std::runtime_error("pimc: optional-observable histories of the " +
                                 std::to_string(n_sectors) + " sectors exceed the aggregate "
                                 "256 MiB budget; reduce bins, blocks, replicas or the window");
    for (std::size_t i = 0; i < n_sectors; ++i)
        for (auto& est : per_sector[i]) {
            // Off-diagonal estimators measure open paths, whose particle count
            // is not a sector of the window.
            if (!est->open_path_group().empty())
                throw std::runtime_error("pimc: estimator '" + est->type() + "' measures open "
                                         "paths; a grand-canonical run measures diagonal "
                                         "observables only");
            // Weight 1 [closed in N], 0 otherwise: the signed-weight mode's
            // ratio of means is then the sector's ratio estimator.
            est->configure_signed_weights();
            const std::string label = sector_label(gc_n_min_ + i);
            est->set_output_label(est->output_label().empty() ? label
                                                              : est->output_label() + "_" + label);
            est->configure_reblocking(blocks);
        }
    sector_est_templates_ = std::move(per_sector);
}

void Pimc::run_grand_canonical() {
    const int n_threads = cfg_.threads > 0 ? cfg_.threads : omp_get_max_threads();
    const int samples = cfg_.samples;
    const int n_eq = cfg_.equilibration_blocks;
    const int n_acc = cfg_.accumulation_blocks;
    const int steps = cfg_.steps_per_block;
    const int sub_steps = cfg_.sub_steps;
    const std::size_t n_sectors = gc_n_max_ - gc_n_min_ + 1;
    const std::size_t worm_index = worm_moves_.front();

    std::vector<double> cum_weights;
    {
        double total = 0.0;
        for (const auto& m : moves_[0]) {
            total += m->weight();
            cum_weights.push_back(total);
        }
        for (double& w : cum_weights) w /= total;
    }
    const auto select_move = [&](numerics::Random& rng) -> std::size_t {
        const double u = rng.uniform();
        for (std::size_t i = 0; i < cum_weights.size(); ++i)
            if (u < cum_weights[i]) return i;
        return cum_weights.size() - 1;
    };
    // One attempt: the worm with the capacity action; every other move with
    // the action of the current N (nothing to move at N = 0).
    const auto attempt = [&](std::size_t r, std::size_t move_index) -> bool {
        PathSet& path = paths_[r];
        if (move_index == worm_index)
            return moves_[r][move_index]->attempt(path, actions_[r], rngs_[r]);
        const std::size_t n = path.active_particles();
        if (n == 0) return false;
        return moves_[r][move_index]->attempt(path, sector_actions_[r][n - gc_n_min_], rngs_[r]);
    };

    const std::size_t n_moves = cum_weights.size();
    const auto n_replicas = static_cast<std::size_t>(samples);
    move_proposals_.assign(n_moves, 0);
    move_accepts_.assign(n_moves, 0);
    move_infinite_rejections_.assign(n_moves, 0);
    // Padded per-replica counters, as in run() (no false sharing).
    constexpr std::size_t kCountersPerLine = 64 / sizeof(std::size_t);
    const std::size_t counter_stride =
        (2 * n_moves + kCountersPerLine - 1) / kCountersPerLine * kCountersPerLine +
        kCountersPerLine;
    std::vector<std::size_t> move_counters(n_replicas * counter_stride, 0);

    std::vector<std::exception_ptr> replica_errors(n_replicas);
    std::atomic<bool> stop_replicas{false};
    const auto stopped = [&stop_replicas]() {
        return stop_replicas.load(std::memory_order_relaxed);
    };

    io::Logger::milestone("pimc: grand canonical, mu_in=" + std::to_string(gc_mu_) +
                          " window=[" + std::to_string(gc_n_min_) + ", " +
                          std::to_string(gc_n_max_) + "] M=" + std::to_string(M_) +
                          " tau=" + std::to_string(cfg_.time_step) + " beta=" +
                          std::to_string(cfg_.beta) + " samples=" + std::to_string(samples) +
                          " threads=" + std::to_string(n_threads));

    // Initial rings collapsed onto walkers of the capacity ensemble; the
    // active count (the initial N) is kept.
    if (ensemble_ && !imported_paths_) {
        const auto walkers = ensemble_->walkers();
        if (walkers.empty())
            throw std::runtime_error("pimc: the ensemble handed to the driver has no walkers");
        for (std::size_t s = 0; s < n_replicas; ++s)
            paths_[s].collapse_to(walkers[s % walkers.size()].positions());
    }
    for (std::size_t s = 0; s < n_replicas; ++s)
        if (gc_initial_ > 0 &&
            !std::isfinite(sector_actions_[s][gc_initial_ - gc_n_min_].total_action(paths_[s])))
            throw std::runtime_error("pimc: initial action is nonfinite for replica " +
                                     std::to_string(s) + "; provide a finite-action initial path");

    // ── Equilibration: tune C towards a 50 % closed fraction ──
    io::Logger::milestone("pimc: equilibration " + std::to_string(n_eq) + " blocks");
    #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
    for (int s = 0; s < samples; ++s) {
        const auto r = static_cast<std::size_t>(s);
        try {
            auto& worm = static_cast<WormBase&>(*moves_[r][worm_index]);
            for (int b = 0; b < n_eq && !stopped(); ++b) {
                std::size_t closed_attempts = 0;
                for (int st = 0; st < steps && !stopped(); ++st)
                    for (int ss = 0; ss < sub_steps; ++ss) {
                        if (paths_[r].is_open()) {
                            attempt(r, worm_index);
                            continue;
                        }
                        ++closed_attempts;
                        attempt(r, select_move(rngs_[r]));
                    }
                const double total = static_cast<double>(steps) * sub_steps;
                const double closed = static_cast<double>(closed_attempts) / total;
                worm.tune_constant(closed, 1.0 - closed, 1.0);
            }
        } catch (...) {
            replica_errors[r] = std::current_exception();
            stop_replicas.store(true, std::memory_order_relaxed);
        }
    }
    detail::rethrow_replica_error(replica_errors, "equilibration");

    // ── Accumulation ──
    io::Logger::milestone("pimc: accumulation " + std::to_string(n_acc) + " blocks");
    const auto n_blocks = static_cast<std::size_t>(n_acc);
    // Block means of one replica: aggregate [replica][block][kAgg*] and per
    // sector [replica][block][sector][kSector*], each divided by the block's
    // measurement count.
    std::vector<double> agg(n_replicas * n_blocks * kAggFields, 0.0);
    std::vector<double> sec(n_replicas * n_blocks * n_sectors * kSectorFields, 0.0);
    // Closed measurements per sector and closed N <-> N + 1 changes per
    // adjacent pair, [replica][N - n_min]; cycle lengths, [replica][L - 1],
    // weighted by L, and the particles they were counted over.
    std::vector<std::vector<double>> replica_counts(n_replicas, std::vector<double>(n_sectors, 0.0));
    std::vector<std::vector<double>> replica_pairs(n_replicas, std::vector<double>(n_sectors - 1, 0.0));
    std::vector<std::vector<double>> replica_cycles(n_replicas, std::vector<double>(np_, 0.0));
    std::vector<double> replica_particles(n_replicas, 0.0);

    for (auto& replica_moves : moves_)
        for (auto& move : replica_moves) move->reset_diagnostics();

    // Per-replica sector estimator clones, [replica][sector][estimator].
    std::vector<std::vector<std::vector<std::unique_ptr<Estimator>>>> replica_ests(n_replicas);
    for (auto& replica : replica_ests) {
        replica.resize(n_sectors);
        for (std::size_t i = 0; i < sector_est_templates_.size(); ++i)
            for (const auto& tmpl : sector_est_templates_[i]) replica[i].push_back(tmpl->clone());
    }

    #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
    for (int s = 0; s < samples; ++s) {
        const auto r = static_cast<std::size_t>(s);
        try {
            PathSet& path = paths_[r];
            numerics::Random& rng = rngs_[r];
            auto& ests = replica_ests[r];
            std::size_t* proposals = move_counters.data() + r * counter_stride;
            std::size_t* accepts = proposals + n_moves;
            // Bosons: every slice of a closed path has weight +1.
            const std::vector<double> weights(path.num_slices(), 1.0);
            constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();
            std::size_t last_closed = path.is_open() ? kNone : path.active_particles();

            for (int b = 0; b < n_acc && !stopped(); ++b) {
                for (auto& sector_ests : ests)
                    for (auto& e : sector_ests) e->start_block();
                double* a = agg.data() + (r * n_blocks + static_cast<std::size_t>(b)) * kAggFields;
                double* blk = sec.data() +
                    (r * n_blocks + static_cast<std::size_t>(b)) * n_sectors * kSectorFields;
                int e_count = 0;
                for (int st = 0; st < steps && !stopped(); ++st) {
                    for (int ss = 0; ss < sub_steps; ++ss) {
                        const std::size_t move_index = path.is_open() ? worm_index : select_move(rng);
                        ++proposals[move_index];
                        if (attempt(r, move_index)) ++accepts[move_index];
                        // A closed N changes by one per excursion through G.
                        if (path.is_open()) continue;
                        const std::size_t n = path.active_particles();
                        if (n != last_closed && last_closed != kNone)
                            replica_pairs[r][std::min(n, last_closed) - gc_n_min_] += 1.0;
                        last_closed = n;
                    }
                    ++e_count;
                    if (path.is_open()) {
                        for (auto& sector_ests : ests)
                            for (auto& e : sector_ests) e->skip_measurement(weights.size());
                        continue;
                    }
                    const std::size_t n = path.active_particles(), i = n - gc_n_min_;
                    double* x = blk + i * kSectorFields;
                    const auto dn = static_cast<double>(n);
                    x[kSectorWeight] += 1.0;
                    a[kAggClosed] += 1.0;
                    a[kAggN] += dn;
                    a[kAggN2] += dn * dn;
                    replica_counts[r][i] += 1.0;
                    if (n > 0) {
                        const Action::ThermoMeasure m = sector_actions_[r][i].measure_thermo(path);
                        double wt[kMaxWindingTensor];
                        const double w2 = winding_measure(path, wt);
                        x[kSectorEnergy] += m.energy;
                        x[kSectorKinetic] += m.kinetic;
                        x[kSectorPotential] += m.potential;
                        x[kSectorWinding2] += w2;
                        a[kAggEnergy] += m.energy;
                        a[kAggKinetic] += m.kinetic;
                        a[kAggPotential] += m.potential;
                        a[kAggWinding2] += dn * w2;
                        for (std::size_t c = 0; c < winding_tensor_size(dim_); ++c) {
                            x[kSectorTensor + c] += wt[c];
                            a[kAggTensor + c] += dn * wt[c];
                        }
                        for (const std::size_t leader : path.cycle_leaders()) {
                            std::size_t length = 1;
                            for (std::size_t q = path.next_particle(leader); q != leader;
                                 q = path.next_particle(q))
                                ++length;
                            replica_cycles[r][length - 1] += static_cast<double>(length);
                        }
                        replica_particles[r] += dn;
                    }
                    for (std::size_t j = 0; j < n_sectors; ++j)
                        for (auto& e : ests[j]) {
                            if (j == i) e->accumulate(path, std::span<const double>(weights));
                            else e->skip_measurement(weights.size());
                        }
                }
                for (auto& sector_ests : ests)
                    for (auto& e : sector_ests) e->stop_block();
                const double inv = 1.0 / static_cast<double>(e_count);
                for (std::size_t f = 0; f < kAggFields; ++f) a[f] *= inv;
                for (std::size_t f = 0; f < n_sectors * kSectorFields; ++f) blk[f] *= inv;
            }
        } catch (...) {
            replica_errors[r] = std::current_exception();
            stop_replicas.store(true, std::memory_order_relaxed);
        }
    }
    detail::rethrow_replica_error(replica_errors, "accumulation");

    for (std::size_t r = 0; r < n_replicas; ++r)
        for (std::size_t i = 0; i < n_moves; ++i) {
            move_proposals_[i] += move_counters[r * counter_stride + i];
            move_accepts_[i] += move_counters[r * counter_stride + n_moves + i];
            move_infinite_rejections_[i] += moves_[r][i]->infinite_action_rejections();
        }
    move_extra_.assign(n_moves, {});
    for (std::size_t i = 0; i < n_moves; ++i)
        for (std::size_t r = 0; r < n_replicas; ++r) {
            const auto extra = moves_[r][i]->extra_diagnostics();
            if (r == 0) move_extra_[i] = extra;
            else
                for (std::size_t e = 0; e < extra.size(); ++e)
                    move_extra_[i][e].second += extra[e].second;
        }
    worm_constants_.assign(1, {0.0, 0.0, 0.0});
    for (std::size_t r = 0; r < n_replicas; ++r) {
        const double c = static_cast<const WormBase&>(*moves_[r][worm_index]).worm_constant();
        auto& [mean, low, high] = worm_constants_[0];
        mean += c / static_cast<double>(n_replicas);
        low = r == 0 ? c : std::min(low, c);
        high = r == 0 ? c : std::max(high, c);
    }
    // Leave every replica closed (outside the statistics), as run() does.
    constexpr std::size_t kMaxClosingAttempts = 10'000'000;
    #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
    for (int s = 0; s < samples; ++s) {
        const auto r = static_cast<std::size_t>(s);
        try {
            for (std::size_t t = 0; paths_[r].is_open(); ++t) {
                if (t == kMaxClosingAttempts)
                    throw std::runtime_error("the worm did not close; lower C");
                attempt(r, worm_index);
            }
        } catch (...) {
            replica_errors[r] = std::current_exception();
        }
    }
    detail::rethrow_replica_error(replica_errors, "worm closing");

    // Reduce the sector estimators in replica order (thread-count independent).
    sector_ests_.clear();
    sector_ests_.resize(n_sectors);
    for (std::size_t i = 0; i < n_sectors; ++i)
        for (std::size_t e = 0; e < replica_ests[0][i].size(); ++e) {
            for (std::size_t r = 1; r < n_replicas; ++r)
                replica_ests[0][i][e]->collect(*replica_ests[r][i][e]);
            sector_ests_[i].push_back(std::move(replica_ests[0][i][e]));
        }

    // Grand blocks: the replica average of each block, as run() does.
    const double inv_replicas = 1.0 / static_cast<double>(n_replicas);
    const std::size_t n_tensor = winding_tensor_size(dim_);
    block_energies_.assign(n_blocks, 0.0);
    block_kinetic_.assign(n_blocks, 0.0);
    block_potential_.assign(n_blocks, 0.0);
    block_winding2_.assign(n_blocks, 0.0);
    block_sign_.assign(n_blocks, 0.0);
    block_closed_.assign(n_blocks, 0.0);
    block_gc_n_.assign(n_blocks, 0.0);
    block_gc_n2_.assign(n_blocks, 0.0);
    for (std::size_t c = 0; c < kMaxWindingTensor; ++c)
        block_winding_tensor_[c].assign(c < n_tensor ? n_blocks : 0, 0.0);
    gc_sector_blocks_.assign(n_sectors, {});
    for (auto& fields : gc_sector_blocks_)
        for (auto& series : fields) series.assign(n_blocks, 0.0);
    for (std::size_t r = 0; r < n_replicas; ++r)
        for (std::size_t b = 0; b < n_blocks; ++b) {
            const double* a = agg.data() + (r * n_blocks + b) * kAggFields;
            block_energies_[b] += a[kAggEnergy] * inv_replicas;
            block_kinetic_[b] += a[kAggKinetic] * inv_replicas;
            block_potential_[b] += a[kAggPotential] * inv_replicas;
            block_winding2_[b] += a[kAggWinding2] * inv_replicas;
            block_closed_[b] += a[kAggClosed] * inv_replicas;
            block_gc_n_[b] += a[kAggN] * inv_replicas;
            block_gc_n2_[b] += a[kAggN2] * inv_replicas;
            for (std::size_t c = 0; c < n_tensor; ++c)
                block_winding_tensor_[c][b] += a[kAggTensor + c] * inv_replicas;
            const double* blk = sec.data() + (r * n_blocks + b) * n_sectors * kSectorFields;
            for (std::size_t i = 0; i < n_sectors; ++i)
                for (std::size_t f = 0; f < kSectorFields; ++f)
                    gc_sector_blocks_[i][f][b] += blk[i * kSectorFields + f] * inv_replicas;
        }
    // Bosons: the sign of every closed path is +1.
    block_sign_ = block_closed_;

    // Sector counts, transitions and cycle fractions, summed in replica order.
    gc_result_ = PimcGrandCanonicalResult{};
    gc_result_.chemical_potential_input = gc_mu_;
    gc_result_.n_min = gc_n_min_;
    gc_result_.n_max = gc_n_max_;
    gc_result_.sectors.assign(n_sectors, {});
    gc_result_.pair_transitions.assign(n_sectors - 1, 0.0);
    gc_result_.replica_transitions.assign(n_replicas, 0.0);
    for (std::size_t r = 0; r < n_replicas; ++r) {
        for (std::size_t i = 0; i < n_sectors; ++i)
            gc_result_.sectors[i].measurements += replica_counts[r][i];
        for (std::size_t i = 0; i + 1 < n_sectors; ++i) {
            gc_result_.pair_transitions[i] += replica_pairs[r][i];
            gc_result_.replica_transitions[r] += replica_pairs[r][i];
        }
    }
    cycle_fractions_.assign(1, std::vector<double>(np_, 0.0));
    double particles = 0.0;
    for (std::size_t r = 0; r < n_replicas; ++r) {
        particles += replica_particles[r];
        for (std::size_t l = 0; l < np_; ++l) cycle_fractions_[0][l] += replica_cycles[r][l];
    }
    if (particles > 0.0)
        for (double& f : cycle_fractions_[0]) f /= particles;

    finalize_grand_canonical();
    finalize_moves();
}

void Pimc::finalize_grand_canonical() {
    // All-sector aggregates: <E>, <K>, <V> over closed paths, and the
    // per-particle winding as <N W^2>/<N> (a ratio of sums, not <W^2/N>).
    signed_stats_[0] = numerics::ratio_of_means(block_energies_, block_closed_);
    signed_stats_[1] = numerics::ratio_of_means(block_kinetic_, block_closed_);
    signed_stats_[2] = numerics::ratio_of_means(block_potential_, block_closed_);
    signed_stats_[3] = numerics::ratio_of_means(block_winding2_, block_gc_n_);
    energy_stats_ = signed_stats_[0].numerator;
    kinetic_stats_ = signed_stats_[1].numerator;
    potential_stats_ = signed_stats_[2].numerator;
    winding_stats_ = signed_stats_[3].numerator;
    energy_ = signed_stats_[0].mean;
    energy_error_ = signed_stats_[0].error;
    naive_error_ = std::numeric_limits<double>::quiet_NaN();
    tau_int_ = std::max(signed_stats_[0].numerator.tau_int, signed_stats_[0].denominator.tau_int);
    plateau_found_ = signed_stats_[0].reliable;
    kinetic_ = signed_stats_[1].mean;
    kinetic_error_ = signed_stats_[1].error;
    potential_ = signed_stats_[2].mean;
    potential_error_ = signed_stats_[2].error;
    winding2_ = signed_stats_[3].mean;
    winding2_error_ = signed_stats_[3].error;
    for (std::size_t c = 0; c < winding_tensor_size(dim_); ++c) {
        signed_tensor_stats_[c] = numerics::ratio_of_means(block_winding_tensor_[c], block_gc_n_);
        winding_tensor_stats_[c] = signed_tensor_stats_[c].numerator;
        winding_tensor_[c] = signed_tensor_stats_[c].mean;
        winding_tensor_error_[c] = signed_tensor_stats_[c].error;
    }
    sign_stats_ = numerics::reblock(block_closed_);
    average_sign_ = 1.0;
    average_sign_error_ = 0.0;
    closed_fraction_ = sign_stats_.mean;
    closed_fraction_error_ = sign_stats_.error;
    sign_unreliable_ = !(closed_fraction_ >= cfg_.sign_threshold * closed_fraction_error_) ||
                       !std::isfinite(energy_);
    if (sign_unreliable_)
        io::Logger::warn("pimc: the closed fraction " + std::to_string(closed_fraction_) + " +/- " +
                         std::to_string(closed_fraction_error_) +
                         " is not resolved; the grand-canonical averages are unreliable");

    // Sector fractions, sector observables and F(N + 1) - F(N).
    const std::size_t n_sectors = gc_sector_blocks_.size();
    for (std::size_t i = 0; i < n_sectors; ++i) {
        PimcSectorResult& out = gc_result_.sectors[i];
        const auto& fields = gc_sector_blocks_[i];
        out.n = gc_n_min_ + i;
        const auto fraction = numerics::ratio_of_means(fields[kSectorWeight], block_closed_);
        out.fraction = fraction.mean;
        out.fraction_error = fraction.error;
        const auto ratio = [&](std::size_t field, double& mean, double& error) {
            const auto stats = numerics::ratio_of_means(fields[field], fields[kSectorWeight]);
            mean = stats.mean;
            error = stats.error;
        };
        ratio(kSectorEnergy, out.energy, out.energy_error);
        ratio(kSectorKinetic, out.kinetic, out.kinetic_error);
        ratio(kSectorPotential, out.potential, out.potential_error);
        ratio(kSectorWinding2, out.winding2, out.winding2_error);
        for (std::size_t c = 0; c < winding_tensor_size(dim_); ++c)
            ratio(kSectorTensor + c, out.winding_tensor[c], out.winding_tensor_error[c]);
        if (out.measurements == 0.0)
            io::Logger::warn("pimc: grand-canonical sector N = " + std::to_string(out.n) +
                             " was never visited; mu_in is far from the window");
    }
    const double temperature = 1.0 / cfg_.beta;
    gc_result_.mu.assign(n_sectors - 1, std::numeric_limits<double>::quiet_NaN());
    gc_result_.mu_error.assign(n_sectors - 1, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i + 1 < n_sectors; ++i) {
        // R = P(N + 1)/P(N) as a ratio of block sums; ln R by the delta method.
        const auto weights = numerics::ratio_of_means(gc_sector_blocks_[i + 1][kSectorWeight],
                                                      gc_sector_blocks_[i][kSectorWeight]);
        if (!(weights.mean > 0.0) || !std::isfinite(weights.mean)) continue;
        gc_result_.mu[i] = gc_mu_ - temperature * std::log(weights.mean);
        gc_result_.mu_error[i] = temperature * weights.error / weights.mean;
    }
    const auto n_mean = numerics::ratio_of_means(block_gc_n_, block_closed_);
    const auto n2_mean = numerics::ratio_of_means(block_gc_n2_, block_closed_);
    gc_result_.n_mean = n_mean.mean;
    gc_result_.n_mean_error = n_mean.error;
    gc_result_.n2_mean = n2_mean.mean;
    gc_result_.n2_mean_error = n2_mean.error;
    gc_result_.n_tau_int = std::max(n_mean.numerator.tau_int, n_mean.denominator.tau_int);
}

void Pimc::write_grand_canonical(RunResult& r) const {
    const std::size_t n_sectors = gc_result_.sectors.size();
    r.add_metadata("grand_canonical",
                   "worm with spare rows and gaps (Boninsegni, Prokof'ev & Svistunov, PRE 74, "
                   "036701 (2006)); closed N in [gc_n_min, gc_n_max] sampled with weight "
                   "exp(beta mu_in N) Z_N");
    r.add_metadata("gc_aggregates",
                   "energy, kinetic, potential: grand-canonical averages over every closed path; "
                   "winding2 and winding_tensor_*: <N O>/<N> over closed paths (acc_block_winding* "
                   "hold N O, acc_block_gc_N_mean holds N); per-sector values under gc_N<n>_*");
    r.add_metadata("gc_mu_definition",
                   "gc_mu[i] = mu_in - T ln[P(N+1)/P(N)] = F(N+1) - F(N) at N = gc_mu_N[i] - 1/2; "
                   "gc_mu_reliable needs >= 100 transitions between the two sectors");
    r.add_scalar("gc_chemical_potential_input", gc_result_.chemical_potential_input);
    r.add_scalar("gc_n_min", static_cast<double>(gc_result_.n_min));
    r.add_scalar("gc_n_max", static_cast<double>(gc_result_.n_max));
    r.add_scalar("gc_initial_N", static_cast<double>(gc_initial_));
    r.add_scalar("gc_N_mean", gc_result_.n_mean);
    r.add_scalar("gc_N_mean_error", gc_result_.n_mean_error);
    r.add_scalar("gc_N_squared_mean", gc_result_.n2_mean);
    r.add_scalar("gc_N_squared_mean_error", gc_result_.n2_mean_error);
    r.add_scalar("gc_N_tau_int", gc_result_.n_tau_int);
    double transitions = 0.0;
    for (const double t : gc_result_.pair_transitions) transitions += t;
    r.add_scalar("gc_transitions", transitions);
    r.add_series("gc_sector_transitions", gc_result_.replica_transitions);

    std::vector<double> sector_n, fraction, fraction_error, measurements;
    for (const PimcSectorResult& s : gc_result_.sectors) {
        sector_n.push_back(static_cast<double>(s.n));
        fraction.push_back(s.fraction);
        fraction_error.push_back(s.fraction_error);
        measurements.push_back(s.measurements);
    }
    r.add_series("gc_sector_N", std::move(sector_n));
    r.add_series("gc_sector_fraction", std::move(fraction));
    r.add_series("gc_sector_fraction_error", std::move(fraction_error));
    r.add_series("gc_sector_measurements", std::move(measurements));
    std::vector<double> mu_n, reliable;
    for (std::size_t i = 0; i + 1 < n_sectors; ++i) {
        mu_n.push_back(static_cast<double>(gc_result_.n_min + i) + 0.5);
        reliable.push_back(gc_result_.pair_transitions[i] >= 100.0 &&
                           std::isfinite(gc_result_.mu_error[i]) ? 1.0 : 0.0);
    }
    r.add_series("gc_mu_N", std::move(mu_n));
    r.add_series("gc_mu", gc_result_.mu);
    r.add_series("gc_mu_error", gc_result_.mu_error);
    r.add_series("gc_mu_reliable", std::move(reliable));
    r.add_series("gc_pair_transitions", gc_result_.pair_transitions);
    r.add_series("acc_block_gc_N_mean", block_gc_n_);
    r.add_series("acc_block_gc_N_squared_mean", block_gc_n2_);

    for (std::size_t i = 0; i < n_sectors; ++i) {
        const PimcSectorResult& s = gc_result_.sectors[i];
        const auto& fields = gc_sector_blocks_[i];
        const std::string prefix = "gc_" + sector_label(s.n) + "_";
        r.add_series("acc_block_" + prefix + "weight", fields[kSectorWeight]);
        if (s.n == 0) continue;
        const auto scalar = [&](const std::string& name, double mean, double error) {
            r.add_scalar(prefix + name, mean);
            r.add_scalar(prefix + name + "_error", error);
        };
        scalar("energy", s.energy, s.energy_error);
        scalar("kinetic", s.kinetic, s.kinetic_error);
        scalar("potential", s.potential, s.potential_error);
        scalar("winding2", s.winding2, s.winding2_error);
        r.add_series("acc_block_" + prefix + "energy", fields[kSectorEnergy]);
        r.add_series("acc_block_" + prefix + "kinetic", fields[kSectorKinetic]);
        r.add_series("acc_block_" + prefix + "potential", fields[kSectorPotential]);
        r.add_series("acc_block_" + prefix + "winding2", fields[kSectorWinding2]);
        for (std::size_t c = 0; c < winding_tensor_size(dim_); ++c) {
            const std::string name = std::string("winding_tensor_") + winding_tensor_label(c);
            scalar(name, s.winding_tensor[c], s.winding_tensor_error[c]);
            r.add_series("acc_block_" + prefix + name, fields[kSectorTensor + c]);
        }
    }
    for (const auto& sector : sector_ests_)
        for (const auto& e : sector) e->write_results(r);
}

void Pimc::summarize_grand_canonical(std::ostringstream& ss) const {
    ss << "\npimc: grand canonical, mu_in = " << gc_result_.chemical_potential_input
       << ", window [" << gc_result_.n_min << ", " << gc_result_.n_max << "], <N> = "
       << gc_result_.n_mean << " +/- " << gc_result_.n_mean_error
       << "; E, K, V above are grand-canonical averages";
    for (const PimcSectorResult& s : gc_result_.sectors) {
        ss << "\npimc:   N = " << s.n << ": P = " << s.fraction << " +/- " << s.fraction_error
           << " (" << s.measurements << " closed measurements)";
        if (s.n > 0 && s.measurements > 0)
            ss << ", E = " << s.energy << " +/- " << s.energy_error;
    }
    for (std::size_t i = 0; i < gc_result_.mu.size(); ++i) {
        const double t = gc_result_.pair_transitions[i];
        ss << "\npimc:   F(" << gc_result_.n_min + i + 1 << ") - F(" << gc_result_.n_min + i
           << ") = " << gc_result_.mu[i] << " +/- " << gc_result_.mu_error[i] << " ("
           << t << " transitions)";
        if (t < 100.0) ss << "  WARNING: fewer than 100 transitions; not a quotable value";
    }
}

}  // namespace alea
