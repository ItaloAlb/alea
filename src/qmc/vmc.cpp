// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "qmc/vmc.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <omp.h>
#include <sstream>

#include "io/logger.h"
#include "numerics/green_function.h"
#include "numerics/statistics.h"
#include "qmc/reblock_diagnostics.h"
#include "qmc/sweep_blocking.h"
#include "population/walker.h"
#include "population/walker_ensemble.h"

namespace alea {

VMC::VMC(const VMCConfig& cfg, WalkerEnsemble& ensemble,
         const TrialWavefunction& psi, const Hamiltonian& hamiltonian)
    : cfg_(cfg),
      ensemble_(&ensemble),
      psi_template_(&psi),
      ham_template_(&hamiltonian) {
    const std::size_t nw  = ensemble.num_walkers();
    const std::size_t np  = ensemble.n_particles();
    const std::size_t dim = ensemble.dim();

    const int n_threads = std::min(cfg_.n_threads, static_cast<int>(nw));

    // Per-particle, per-dimension proposal width.
    tau_inv_mass_.assign(np * dim, 0.0);
    sqrt_tau_inv_mass_.assign(np * dim, 0.0);
    for (const ParticleGroup& g : ensemble.groups())
        for (std::size_t i = g.first; i < g.last; ++i)
            for (std::size_t d = 0; d < dim; ++d) {
                tau_inv_mass_[i * dim + d]      = cfg_.tau * g.inv_mass[d];
                sqrt_tau_inv_mass_[i * dim + d] = std::sqrt(tau_inv_mass_[i * dim + d]);
            }

    // Distribute walkers evenly across threads.
    const std::size_t nw_base = nw / static_cast<std::size_t>(n_threads);
    const std::size_t nw_rem  = nw % static_cast<std::size_t>(n_threads);

    auto walker_it = ensemble.walkers().begin();
    for (int t = 0; t < n_threads; ++t) {
        const std::size_t nw_t =
            nw_base + (static_cast<std::size_t>(t) < nw_rem ? 1 : 0);

        wsets_.emplace_back(nw_t, np, dim, ensemble.cell());
        for (std::size_t i = 0; i < nw_t; ++i)
            wsets_[t].add_walker(*walker_it++);
        wsets_[t].load_all();

        rngs_.emplace_back(numerics::stream_seed(cfg_.seed, kVmcThreadStream,
                                                 static_cast<std::uint64_t>(t)));
    }
}

bool VMC::accept(double log_ratio, double u) noexcept {
    // Symmetric proposal: P_acc = min(1, |ψ'/ψ|²) = min(1, exp(2·log_ratio)).
    return u < std::exp(2.0 * log_ratio);
}

std::size_t VMC::sweep(WalkerSet& wset, TrialWavefunction& psi,
                       std::span<const double> sigma, numerics::XoshiroRandom& rng,
                       std::span<double> randoms, std::size_t block) {
    const std::size_t nw = wset.num_walkers();
    // Same draw order as the unblocked sweep, so a symmetric (no-drift) sweep
    // reproduces its trajectory exactly; see sweep_blocking.h.
    draw_sweep_randoms(rng, wset.n_particles(), wset.dim(), nw, 0, nw, randoms);
    std::size_t n_accepted = 0;
    for (std::size_t w0 = 0; w0 < nw; w0 += block)
        n_accepted += sweep_walkers(wset, psi, sigma, randoms, w0, std::min(nw, w0 + block));
    return n_accepted;
}

std::size_t VMC::sweep_walkers(WalkerSet& wset, TrialWavefunction& psi,
                               std::span<const double> sigma,
                               std::span<const double> randoms,
                               std::size_t w_begin, std::size_t w_end) {
    const std::size_t nw  = wset.num_walkers();
    const std::size_t np  = wset.n_particles();
    const std::size_t dim = wset.dim();

    // Use WalkerSet's internal scratch — zero per-sweep allocation.
    const std::span<double> new_pos = wset.new_pos();
    std::span<bool> mask = wset.accept_mask();
    const std::span<const double> pos = wset.positions();
    const std::span<const double> grad = wset.gradients();
    const bool use_drift = cfg_.use_drift;
    std::size_t n_accepted = 0;

    for (std::size_t p = 0; p < np; ++p) {
        // Determinant partners may be deferred; Langevin drift reads only p.
        if (use_drift) psi.refresh_particle_derivatives(wset, p, w_begin, w_end);
        const std::size_t pd_off = p * dim;  // → tau_inv_mass_[pd_off + d]
        const double* gaussian = randoms.data() + p * (dim + 1) * nw;  // [dim * nw]
        const double* uniform  = gaussian + dim * nw;                   // [nw]

        // ── Proposal ──────────────────────────────────────────────────
        for (std::size_t d = 0; d < dim; ++d) {
            const std::size_t off = p * dim * nw + d * nw;
            for (std::size_t w = w_begin; w < w_end; ++w) {
                new_pos[d * nw + w] = use_drift ?
                pos[off + w] + tau_inv_mass_[pd_off + d] * grad[off + w] + sigma[pd_off + d] * gaussian[d * nw + w] :
                pos[off + w] + sigma[pd_off + d] * gaussian[d * nw + w];
            }
        }

        wset.propose_move_batch(p, new_pos, w_begin, w_end);
        psi.ratio_batch(wset, p, w_begin, w_end);

        // ── Accept / reject ───────────────────────────────────────────
        const std::span<const double> log_ratios = wset.log_ratios();

        if (use_drift) {
            // Metropolis-Hastings acceptance with asymmetric Langevin proposal.
            // Correction term: see numerics::langevin_log_ratio_d.
            const std::span<const double> new_grad = wset.new_grad();

            for (std::size_t w = w_begin; w < w_end; ++w) {
                double log_corr = 0.0;
                for (std::size_t d = 0; d < dim; ++d) {
                    const std::size_t off = p * dim * nw + d * nw;
                    log_corr += numerics::langevin_log_ratio_d(
                        new_pos[d * nw + w] - pos[off + w],
                        grad[off + w],
                        new_grad[d * nw + w],
                        tau_inv_mass_[pd_off + d]);
                }
                const double log_accept = 2.0 * log_ratios[w] + log_corr;
                mask[w] = log_accept >= 0.0 || uniform[w] < std::exp(log_accept);
                if (mask[w]) ++n_accepted;
            }
        } else {
            for (std::size_t w = w_begin; w < w_end; ++w) {
                mask[w] = accept(log_ratios[w], uniform[w]);
                if (mask[w]) ++n_accepted;
            }
        }

        // ── Commit accepted moves ──────────────────────────────────────
        // Order matters: the wavefunction commits its caches (reading the old
        // distances/positions) before the WalkerSet overwrites them.
        psi.apply_move_batch_deferred(wset, p, mask, w_begin, w_end);
        wset.apply_move(p, new_pos, wset.new_grad(), wset.new_lap(), mask,
                        w_begin, w_end);
    }
    return n_accepted;
}

VMC::StepResult VMC::run_step(WalkerSet& wset, TrialWavefunction& psi,
                               Hamiltonian& hamiltonian, numerics::XoshiroRandom& rng,
                               std::span<double> randoms, std::size_t block,
                               bool measure) {
    const std::span<const double> sigma = sqrt_tau_inv_mass_;
    StepResult r;

    // n_substeps sweeps total; only the last one is followed by a measurement.
    for (int s = 0; s < cfg_.n_substeps - 1; ++s)
        r.n_accepted += sweep(wset, psi, sigma, rng, randoms, block);

    // Final substep: sweep then measure the local energy.
    r.n_accepted += sweep(wset, psi, sigma, rng, randoms, block);

    // Derivatives enter H and every measurement hook at the current state.
    psi.refresh_derivatives(wset);
    hamiltonian.evaluate_batch(wset);

    if (measure && measure_hook_)
        measure_hook_(wset, psi);

    const std::span<const double> le = wset.local_energies();
    for (double e : le) {
        r.energy_sum += e;
        r.energy_sq_sum += e * e;
    }
    if (wset.has_phase())
        for (double e : wset.local_energies_imag()) r.energy_imag_sum += e;
    r.n_samples = wset.num_walkers();
    return r;
}

void VMC::run() {
    const int n_threads = static_cast<int>(wsets_.size());

    // Clone psi and hamiltonian lazily — the template may have been updated
    // by a previous driver (e.g. optimiser).
    psis_.clear();
    hamiltonians_.clear();
    for (int t = 0; t < n_threads; ++t) {
        psis_.push_back(psi_template_->clone());
        hamiltonians_.push_back(ham_template_->clone());
    }

    // Clone observable estimators per thread so each accumulates independently
    // (reduced after the accumulation region, like the per-thread block means).
    std::vector<std::vector<std::unique_ptr<Estimator>>> thread_ests(
        static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t)
        for (const auto& tmpl : est_templates_)
            thread_ests[static_cast<std::size_t>(t)].push_back(tmpl->clone());
    // Estimators of ψ_T ratios copy this thread's ψ_T; they run inside the
    // thread's own loop, so each gets one worker.
    for (int t = 0; t < n_threads; ++t)
        for (const auto& e : thread_ests[static_cast<std::size_t>(t)])
            if (e->needs_wavefunction())
                e->bind_wavefunction(psis_[static_cast<std::size_t>(t)],
                                     numerics::stream_seed(cfg_.seed, kVmcEstimatorStream,
                                                           static_cast<std::uint64_t>(t)),
                                     1);
    // The sweeps commit deferred and refresh every derivative before H, so no
    // proposal Laplacian is read (set after binding: the estimators' own
    // copies keep the exact one).
    for (TrialWavefunction& psi_t : psis_) psi_t.set_proposal_laplacian(false);

    using Clock = std::chrono::steady_clock;
    auto t0      = Clock::now();
    auto t_last  = t0;

    auto format_duration = [](double secs) {
        if (secs < 60.0) {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%.0fs", secs);
            return std::string(buf);
        }
        if (secs < 3600.0) {
            int m = static_cast<int>(secs) / 60;
            int s = static_cast<int>(secs) % 60;
            return std::to_string(m) + "m" + std::to_string(s) + "s";
        }
        int h = static_cast<int>(secs) / 3600;
        int m = (static_cast<int>(secs) % 3600) / 60;
        return std::to_string(h) + "h" + std::to_string(m) + "m";
    };

    // Per-thread sweep randoms and cache-block sizes, filled by each thread.
    sweep_randoms_.assign(static_cast<std::size_t>(n_threads), {});
    sweep_blocks_.assign(static_cast<std::size_t>(n_threads), 1);

    // ── Equilibration: each thread runs independently. ─────────────────
    const int n_eq         = cfg_.n_equilibration_blocks;
    const int eq_milestone = std::max(1, n_eq / 2);  // 2 milestones

    io::Logger::milestone("vmc: equilibration " + std::to_string(n_eq) + " blocks");

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int t = 0; t < n_threads; ++t) {
        WalkerSet&               wset  = wsets_[t];
        TrialWavefunction&       psi_t = psis_[t];
        Hamiltonian&             ham_t = hamiltonians_[t];
        numerics::XoshiroRandom& rng_t = rngs_[t];

        psi_t.evaluate_batch(wset);
        // Sized by the owning thread (first touch places it near that core).
        sweep_randoms_[t].assign(wset.n_particles() * (wset.dim() + 1) * wset.num_walkers(), 0.0);
        sweep_blocks_[t] = sweep_block_walkers(cfg_.sweep_block_walkers,
            psi_t.walker_state_bytes() + wset.walker_state_bytes(), wset.num_walkers());
        const std::span<double> randoms = sweep_randoms_[t];
        const std::size_t block = sweep_blocks_[t];

        for (int b = 0; b < n_eq; ++b) {
            for (int s = 0; s < cfg_.n_steps; ++s)
                run_step(wset, psi_t, ham_t, rng_t, randoms, block, /*measure=*/false);

            // Milestone (thread 0 only — all threads run in lockstep).
            if (t == 0) {
                int done = b + 1;
                if (done % eq_milestone == 0 || done == n_eq) {
                    auto now  = Clock::now();
                    auto dt   = std::chrono::duration<double>(now - t_last);
                    auto total = std::chrono::duration<double>(now - t0);
                    int blks  = done % eq_milestone == 0
                                ? eq_milestone
                                : done - (n_eq / eq_milestone) * eq_milestone;
                    double ms_per_blk = dt.count() * 1000.0 / static_cast<double>(blks);
                    char ms_buf[16];
                    std::snprintf(ms_buf, sizeof(ms_buf), "%.0f", ms_per_blk);
                    io::Logger::milestone(
                        "vmc: equilibration " + std::to_string(done) + "/"
                        + std::to_string(n_eq)
                        + "  time=" + format_duration(dt.count())
                        + "  average=" + ms_buf + "ms/block"
                        + "  remaining=" + format_duration(ms_per_blk * static_cast<double>(n_eq - done) / 1000.0));
                    t_last = now;
                }
            }
        }
    }

    // ── Accumulation: each thread collects its own block energies. ─────
    const int n_acc         = cfg_.n_accumulation_blocks;
    const int acc_milestone = std::max(1, n_acc / 4);  // 4 milestones
    // per-thread block energies, indexed by omp_get_thread_num().
    std::vector<std::vector<double>> t_blocks(static_cast<std::size_t>(n_threads));
    std::vector<std::vector<double>> t_blocks_imag(static_cast<std::size_t>(n_threads));

    double      e_sum  = 0.0;
    double      e2_sum = 0.0;
    std::size_t n_tot  = 0;
    std::size_t acc    = 0;
    std::size_t prop   = 0;

    t_last = Clock::now();
    io::Logger::milestone("vmc: accumulation " + std::to_string(n_acc) + " blocks");

    #pragma omp parallel reduction(+ : e_sum, e2_sum, n_tot, acc, prop) num_threads(n_threads)
    {
        const int t = omp_get_thread_num();

        WalkerSet&               wset  = wsets_[t];
        TrialWavefunction&       psi_t = psis_[t];
        Hamiltonian&             ham_t = hamiltonians_[t];
        numerics::XoshiroRandom& rng_t = rngs_[t];
        auto&                    ests  = thread_ests[static_cast<std::size_t>(t)];

        const std::size_t pp_step = static_cast<std::size_t>(cfg_.n_substeps) *
                                    wset.n_particles() * wset.num_walkers();

        auto& blocks = t_blocks[static_cast<std::size_t>(t)];
        blocks.reserve(static_cast<std::size_t>(cfg_.n_accumulation_blocks));
        auto& blocks_imag = t_blocks_imag[static_cast<std::size_t>(t)];
        if (wset.has_phase())
            blocks_imag.reserve(static_cast<std::size_t>(cfg_.n_accumulation_blocks));

        for (int b = 0; b < cfg_.n_accumulation_blocks; ++b) {
            for (auto& e : ests) e->start_block();
            double      blk_sum  = 0.0;
            double      blk_imag = 0.0;
            std::size_t blk_n    = 0;
            for (int s = 0; s < cfg_.n_steps; ++s) {
                const StepResult r = run_step(wset, psi_t, ham_t, rng_t,
                    sweep_randoms_[t], sweep_blocks_[t]);
                // Distance table is current after the step's measurement sweep.
                for (auto& e : ests) e->accumulate(wset);
                blk_sum  += r.energy_sum;
                blk_imag += r.energy_imag_sum;
                blk_n    += r.n_samples;
                e_sum   += r.energy_sum;
                e2_sum  += r.energy_sq_sum;
                n_tot   += r.n_samples;
                acc     += r.n_accepted;
                prop    += pp_step;
            }
            for (auto& e : ests) e->stop_block();
            blocks.push_back(blk_sum / static_cast<double>(blk_n));
            if (wset.has_phase())
                blocks_imag.push_back(blk_imag / static_cast<double>(blk_n));

            // Milestone (thread 0 only — all threads run in lockstep).
            if (t == 0) {
                int done = b + 1;
                if (done % acc_milestone == 0 || done == n_acc) {
                    auto now  = Clock::now();
                    auto dt   = std::chrono::duration<double>(now - t_last);
                    int blks  = done % acc_milestone == 0
                                ? acc_milestone
                                : done - (n_acc / acc_milestone) * acc_milestone;
                    double ms_per_blk = dt.count() * 1000.0 / static_cast<double>(blks);
                    char ms_buf[16];
                    std::snprintf(ms_buf, sizeof(ms_buf), "%.0f", ms_per_blk);
                    io::Logger::milestone(
                        "vmc: accumulation " + std::to_string(done) + "/"
                        + std::to_string(n_acc)
                        + "  time=" + format_duration(dt.count())
                        + "  average=" + ms_buf + "ms/block"
                        + "  remaining=" + format_duration(ms_per_blk * static_cast<double>(n_acc - done) / 1000.0));
                    t_last = now;
                }
            }
        }
    }

    // ── Reduce per-thread estimator clones into the final estimators. ──
    ests_.clear();
    if (!est_templates_.empty()) {
        for (std::size_t e = 0; e < thread_ests[0].size(); ++e) {
            for (int t = 1; t < n_threads; ++t)
                thread_ests[0][e]->collect(
                    *thread_ests[static_cast<std::size_t>(t)][e]);
            ests_.push_back(std::move(thread_ests[0][e]));
        }
    }

    // ── Combine block energies across all threads. ─────────────────────
    block_energies_.clear();
    for (auto& blks : t_blocks)
        block_energies_.insert(block_energies_.end(), blks.begin(), blks.end());

    // ── Statistics ─────────────────────────────────────────────────────
    // Each thread is an independent chain of n_acc blocks: reblock them
    // separately and pool (numerics::reblock_chains), so the error accounts
    // for correlation between blocks without ever pairing two threads' blocks.
    const auto n_chains = static_cast<std::size_t>(n_threads);
    energy_stats_ = numerics::reblock_chains(block_energies_, n_chains);
    energy_       = energy_stats_.mean;
    energy_error_ = energy_stats_.error;

    // ⟨Im E_L⟩ (complex trial only): zero for a Hermitian H, so a value many
    // error bars away from 0 flags a bug in a complex component's derivatives.
    std::vector<double> blocks_imag;
    for (auto& blks : t_blocks_imag)
        blocks_imag.insert(blocks_imag.end(), blks.begin(), blks.end());
    is_complex_ = !blocks_imag.empty();
    if (is_complex_) {
        energy_imag_stats_ = numerics::reblock_chains(blocks_imag, n_chains);
        energy_imag_       = energy_imag_stats_.mean;
        energy_imag_error_ = energy_imag_stats_.error;
    }
    // An unreliable error bar is flagged in summary(), not logged here: the
    // optimizer samples with short VMC runs whose energy error it never uses.

    const double mean = e_sum / static_cast<double>(n_tot);
    energy_variance_   = e2_sum / static_cast<double>(n_tot) - mean * mean;
    acceptance_ratio_  = static_cast<double>(acc) / static_cast<double>(prop);

    {
        auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0);
        walltime_s_ = dt.count();
        io::Logger::milestone("vmc: complete  E=" + std::to_string(energy_)
                             + " +/- " + std::to_string(energy_error_)
                             + "  var=" + std::to_string(energy_variance_)
                             + "  acc=" + std::to_string(acceptance_ratio_)
                             + "  time=" + format_duration(dt.count()));
    }
}

std::string VMC::summary() const {
    std::stringstream ss;
    ss << "vmc: E = " << energy_ << " +/- " << energy_error_
       << "  var = " << energy_variance_
       << "  acc = " << acceptance_ratio_;
    if (is_complex_)
        ss << "  ImE = " << energy_imag_ << " +/- " << energy_imag_error_;
    if (!error_reliable(energy_stats_, block_energies_.size()))
        ss << "\nvmc: WARNING: no reliable reblocking plateau (or < " << kMinEffectiveBlocks
           << " effective blocks); the error bar may be too small, increase accumulation_blocks";
    return ss.str();
}

RunResult VMC::run_result() const {
    RunResult r;
    r.method = "vmc";

    r.add_scalar("energy", energy_);
    r.add_scalar("energy_error", energy_error_);
    add_reblock_diagnostics(r, "energy", energy_stats_, block_energies_.size());
    r.add_scalar("energy_variance", energy_variance_);
    r.add_scalar("acceptance_ratio", acceptance_ratio_);
    if (is_complex_) {
        r.add_scalar("energy_imag", energy_imag_);
        r.add_scalar("energy_imag_error", energy_imag_error_);
    }
    r.add_scalar("walltime_s", walltime_s_);
    r.add_scalar("tau", cfg_.tau);
    r.add_scalar("n_walkers",
                 static_cast<double>(ensemble_->target_walkers()));
    r.add_scalar("n_threads", static_cast<double>(wsets_.size()));
    r.add_scalar("n_equilibration_blocks",
                 static_cast<double>(cfg_.n_equilibration_blocks));
    r.add_scalar("n_accumulation_blocks",
                 static_cast<double>(cfg_.n_accumulation_blocks));
    r.add_scalar("n_steps", static_cast<double>(cfg_.n_steps));
    r.add_scalar("n_substeps", static_cast<double>(cfg_.n_substeps));

    r.add_metadata("method", "vmc");
    r.add_metadata("seed", std::to_string(cfg_.seed));
    r.add_metadata("rng_streams",
                   "xoshiro256++ with ziggurat Gaussians per thread t, seeded "
                   "with splitmix64-hashed (seed, vmc thread stream, t)");
    r.add_metadata("energy_error_model",
                   "Flyvbjerg-Petersen reblocking, per-thread chains pooled");

    // Concatenated per (thread, block) — reshape to [n_threads, n_acc] using
    // the scalars above.
    r.add_series("acc_block_energy", block_energies_);

    // Observable estimators (e.g. g(r)) append their own named series, each
    // instance under its output label.
    for (const auto& e : ests_) e->write_results(r);
    r.check_unique_keys();

    return r;
}

}  // namespace alea
