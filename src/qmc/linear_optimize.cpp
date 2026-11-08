// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "qmc/linear_optimize.h"

#include <algorithm>
#include <bitset>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <span>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "io/logger.h"
#include "numerics/random.h"
#include "population/walker.h"
#include "population/walker_ensemble.h"
#include "population/walker_set.h"
#include "qmc/vmc.h"
#include "wavefunction/trial_wavefunction.h"
#include "hamiltonian/hamiltonian.h"

namespace alea {

LinearOptimize::LinearOptimize(OptimizeConfig     cfg,
                               WalkerEnsemble&    ensemble,
                               TrialWavefunction& psi,
                               const Hamiltonian& hamiltonian)
    : cfg_(std::move(cfg)),
      ensemble_(ensemble),
      psi_(psi),
      hamiltonian_(hamiltonian) {
    if (cfg_.convergence_window < 3)
        throw std::invalid_argument(
            "optimizer: convergence_window must be at least 3 (a line through "
            "the energies needs n - 2 > 0 residual degrees of freedom), got " +
            std::to_string(cfg_.convergence_window));
}

// ── Parameter access ─────────────────────────────────────────────────────────

std::vector<double> LinearOptimize::current_params() const {
    std::vector<double> vals;
    for (const auto& comp : psi_.components()) {
        for (const Parameter& p : comp->parameters())
            if (!p.is_fixed) vals.push_back(p.value);
    }
    return vals;
}

void LinearOptimize::apply_update(const std::vector<double>& update) {
    std::size_t k = 0;
    for (auto& comp : psi_.components()) {
        for (Parameter& p : comp->parameters()) {
            if (p.is_fixed) continue;
            // Sign-locked params are optimised in θ-space (p = s·exp(θ)), so the
            // step Δθ maps multiplicatively and keeps the sign; free params take
            // the additive step; a bounded one (a periodic cutoff) is clamped.
            p.value = stepped_value(p, update[k++]);
        }
    }
    psi_.rebuild();
}

void LinearOptimize::set_params(const std::vector<double>& values) {
    std::size_t k = 0;
    for (auto& comp : psi_.components())
        for (Parameter& p : comp->parameters())
            if (!p.is_fixed) p.value = values[k++];
    psi_.rebuild();
}

std::vector<double> LinearOptimize::deriv_scales() const {
    std::vector<double> scales;
    for (const auto& comp : psi_.components()) {
        for (const Parameter& p : comp->parameters()) {
            if (!p.is_fixed) scales.push_back(p.sign_locked ? p.value : 1.0);
        }
    }
    return scales;
}

// ── VMC sampling (delegates to the VMC driver) ────────────────────────────────

std::size_t LinearOptimize::SampleStore::n_samples() const noexcept {
    std::size_t n = 0;
    for (const auto& frames : per_thread)
        for (const SampleFrame& f : frames) n += f.nw;
    return n;
}

double LinearOptimize::sample_vmc(CostFunction& cost_fn, SampleStore* store) const {
    // Build a VMCConfig from our OptimizeConfig.
    VMCConfig vmc_cfg;
    vmc_cfg.n_equilibration_blocks = cfg_.n_equilibration_blocks;
    vmc_cfg.n_accumulation_blocks  = cfg_.n_accumulation_blocks;
    vmc_cfg.n_steps                = cfg_.n_steps;
    vmc_cfg.n_substeps             = cfg_.n_substeps;
    vmc_cfg.tau                    = cfg_.tau;
    vmc_cfg.use_drift              = cfg_.use_drift;
    vmc_cfg.seed                   = cfg_.seed;
    vmc_cfg.n_threads              = cfg_.n_threads;

    const std::size_t n_opt = psi_.total_optimizable_params();
    const int n_threads = (vmc_cfg.n_threads > 0) ? vmc_cfg.n_threads
                                                   : omp_get_max_threads();

    // Chain-rule scales for θ-reparameterised (sign-locked) params. Constant for
    // the whole sampling pass (parameters don't change during one VMC run), so
    // compute once and share read-only across threads.
    const std::vector<double> scales = deriv_scales();

    // Per-thread CostFunction (the hook is called from multiple OpenMP threads).
    std::vector<CostFunction> t_cost(static_cast<std::size_t>(n_threads));

    // Correlated-sampling store: each thread appends only to its own slot.
    // Thinning stride per thread, fixed at its first call from the budget.
    std::vector<std::size_t> t_calls(static_cast<std::size_t>(n_threads), 0);
    std::vector<std::size_t> t_stride(static_cast<std::size_t>(n_threads), 0);
    // Each thread's geometry, copied into the store after the run — writing
    // store->cell/np/dim from the hook would race between threads.
    struct Geometry { bool set = false; SimulationCell cell; std::size_t np = 0, dim = 0; };
    std::vector<Geometry> t_geo(static_cast<std::size_t>(n_threads));
    if (store) {
        *store = SampleStore{};
        store->per_thread.resize(static_cast<std::size_t>(n_threads));
    }
    const std::size_t calls_per_thread =
        static_cast<std::size_t>(cfg_.n_accumulation_blocks) *
        static_cast<std::size_t>(std::max(cfg_.n_steps, 1));

    VMC vmc(vmc_cfg, ensemble_, psi_, hamiltonian_);
    vmc.set_measure_hook([&](const WalkerSet& wset,
                              const TrialWavefunction& psi_thread) {
        const int tid = omp_get_thread_num();
        auto& cf = t_cost[static_cast<std::size_t>(tid)];
        const std::size_t call = t_calls[static_cast<std::size_t>(tid)]++;

        const std::size_t nw_thread = wset.num_walkers();

        // Per-thread scratch for derivatives.
        std::vector<double> derivs_block(n_opt * nw_thread);
        const std::size_t field = wset.n_particles() * wset.dim() * nw_thread;
        std::vector<double> dgrad(n_opt * field), dlap(n_opt * field);
        std::vector<double> el_derivs(n_opt * nw_thread, 0.0);
        std::vector<char>   supported(n_opt, 0);

        const auto le = wset.local_energies();
        std::vector<double> energies(nw_thread);
        bool any_bad = false;
        for (std::size_t w = 0; w < nw_thread; ++w) {
            const double val = le[w];
            if (std::isfinite(val)) {
                energies[w] = val;
            } else {
                energies[w] = 0.0;
                any_bad = true;
            }
        }
        if (any_bad) return;

        if (n_opt > 0) {
            psi_thread.compute_parameter_derivatives(wset, derivs_block);
            // Convert ∂logψ/∂p → ∂logψ/∂θ for sign-locked params: scale each
            // such column by dp/dθ = p. Free params have scale 1.0.
            for (std::size_t k = 0; k < n_opt; ++k) {
                const double sk = scales[k];
                if (sk != 1.0) {
                    double* __restrict col = derivs_block.data() + k * nw_thread;
                    for (std::size_t w = 0; w < nw_thread; ++w) col[w] *= sk;
                }
            }

            // ∂E_L/∂p_k, the curvature term of Hbar.  Components that cannot
            // supply ∂(∇logψ)/∂p leave their slices zero, which drops the term
            // for their parameters only.  The Hamiltonian is read-only here.
            if (psi_thread.compute_parameter_gradient_derivatives(
                    wset, dgrad, dlap, supported)) {
                hamiltonian_.local_energy_parameter_derivatives(
                    wset, dgrad, dlap, n_opt, el_derivs);
                for (std::size_t k = 0; k < n_opt; ++k) {
                    const double sk = scales[k];
                    if (sk != 1.0) {
                        double* __restrict col = el_derivs.data() + k * nw_thread;
                        for (std::size_t w = 0; w < nw_thread; ++w) col[w] *= sk;
                    }
                }
            } else {
                std::fill(el_derivs.begin(), el_derivs.end(), 0.0);
            }
        }

        cf.add_samples(energies, derivs_block, el_derivs, n_opt, nw_thread);

        if (store) {
            const std::size_t t  = static_cast<std::size_t>(tid);
            const std::size_t np = wset.n_particles(), dim = wset.dim();
            std::size_t& stride = t_stride[t];
            if (stride == 0) {
                const std::size_t per_call = nw_thread * (np * dim + 3) * sizeof(double);
                const std::size_t budget =
                    std::max<std::size_t>(cfg_.sample_store_bytes /
                                          static_cast<std::size_t>(n_threads), 1);
                stride = std::max<std::size_t>(
                    1, (calls_per_thread * per_call + budget - 1) / budget);
            }
            if (call % stride == 0) {
                if (!t_geo[t].set) t_geo[t] = Geometry{true, wset.cell(), np, dim};
                SampleFrame f;
                f.nw     = nw_thread;
                f.block  = call / static_cast<std::size_t>(std::max(cfg_.n_steps, 1));
                f.thread = t;
                f.pos.assign(wset.positions().begin(), wset.positions().end());
                f.log_psi.assign(wset.log_psi().begin(), wset.log_psi().end());
                f.e_loc = energies;
                // V = E_L − K: the part of E_L no parameter can change.
                std::vector<double> kin(nw_thread);
                hamiltonian_.psi_dependent_energy(wset, kin);
                f.v_fixed.resize(nw_thread);
                for (std::size_t w = 0; w < nw_thread; ++w)
                    f.v_fixed[w] = energies[w] - kin[w];
                store->per_thread[t].push_back(std::move(f));
            }
        }
    });

    // Suppress the VMC's own log output — the optimiser reports progress
    // at a higher level (one line per iteration).
    io::Logger::pause();
    vmc.run();
    io::Logger::resume();

    // Merge per-thread CostFunctions.
    for (auto& cf : t_cost) cost_fn.merge(cf);

    if (store) {
        for (const Geometry& g : t_geo) {
            if (!g.set) continue;
            store->cell = g.cell;
            store->np   = g.np;
            store->dim  = g.dim;
            break;
        }
    }
    return vmc.energy_error();
}

// ── Adaptive shift helpers ───────────────────────────────────────────────────

std::vector<double> LinearOptimize::prepare_shifts(double central, int num_shifts,
                                                    double factor) {
    std::vector<double> shifts(num_shifts);
    // For odd num_shifts, the central value goes at the middle index.
    // shifts[i] = central / factor^{(mid - i)}
    const int mid = num_shifts / 2;
    for (int i = 0; i < num_shifts; ++i) {
        const int exp = mid - i;
        double multiplier = 1.0;
        for (int e = 0; e < std::abs(exp); ++e)
            multiplier *= factor;
        if (exp >= 0)
            shifts[i] = central / multiplier;
        else
            shifts[i] = central * multiplier;
    }
    return shifts;
}

bool LinearOptimize::is_best_cost(int /*shift_idx*/,
                                   const std::vector<double>& /*shifts*/,
                                   double si, double cost,
                                   double best_cost, double best_si) const {
    // If this is the first valid candidate, accept it.
    if (best_cost >= std::numeric_limits<double>::max())
        return true;

    // Cost-increase tolerance: accept a slightly higher cost if it brings
    // us closer to the target shift.
    const double cost_diff = cost - best_cost;
    const bool cost_ok = cost_diff <= cfg_.cost_increase_tol;

    if (!cost_ok)
        return false;

    // If target_shift_i >= 0, prefer shifts that are closer to the target.
    if (cfg_.target_shift_i >= 0.0) {
        const double dist_to_target = std::abs(si - cfg_.target_shift_i);
        const double best_dist_to_target = std::abs(best_si - cfg_.target_shift_i);
        if (dist_to_target < best_dist_to_target)
            return true;
        if (dist_to_target > best_dist_to_target)
            return false;
    }

    // Prefer lower cost.
    return cost < best_cost;
}

// ── Correlated sampling ──────────────────────────────────────────────────────
//
// Reweighting (the QMCPACK `correlatedSampling`; Umrigar, Wilson & Wilkins,
// PRL 60, 1719 (1988)): the configurations R_k were drawn from |ψ_old|².  For
// new parameters
//   E_new ≈ Σ_k w_k E_L,new(R_k) / Σ_k w_k,     w_k = |ψ_new(R_k)/ψ_old(R_k)|²,
// with E_L,new = V(R_k) + K_new(R_k): only the kinetic part is recomputed.
// Candidates and the current ψ share every R_k, so ΔE carries almost none of
// the sampling noise that two independent VMC runs would, and a runaway step
// shows up as a collapse of N_eff = (Σw)²/Σw² instead of a spuriously low E.
LinearOptimize::Reweighted
LinearOptimize::reweight(const SampleStore& store, const std::vector<double>& update) {
    Reweighted out;
    std::vector<const SampleFrame*> frames;
    for (const auto& per : store.per_thread)
        for (const SampleFrame& f : per) frames.push_back(&f);
    if (frames.empty() || store.np == 0 || store.dim == 0) return out;

    // Per-thread clones at the candidate parameters; ψ itself is restored
    // bit-exactly afterwards (values saved, not an inverse step).
    const int n_threads = (cfg_.n_threads > 0) ? cfg_.n_threads : omp_get_max_threads();
    std::vector<double> saved;
    for (const auto& comp : psi_.components())
        for (const Parameter& p : comp->parameters()) saved.push_back(p.value);
    apply_update(update);
    std::vector<TrialWavefunction> psis;
    psis.reserve(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) psis.push_back(psi_.clone());
    {
        std::size_t k = 0;
        for (auto& comp : psi_.components())
            for (Parameter& p : comp->parameters()) p.value = saved[k++];
        psi_.rebuild();
    }

    std::vector<std::vector<double>> weight(frames.size()), e_new(frames.size());
    const long n_frames = static_cast<long>(frames.size());
#pragma omp parallel num_threads(n_threads)
    {
        TrialWavefunction& psi_t = psis[static_cast<std::size_t>(omp_get_thread_num())];
        std::unique_ptr<WalkerSet> ws;
        std::vector<double> kin;
#pragma omp for schedule(dynamic)
        for (long i = 0; i < n_frames; ++i) {
            const SampleFrame& f = *frames[static_cast<std::size_t>(i)];
            if (!ws || ws->num_walkers() != f.nw) {
                ws = std::make_unique<WalkerSet>(f.nw, store.np, store.dim, store.cell);
                psi_t.prepare(f.nw, store.np, store.dim);
                kin.assign(f.nw, 0.0);
            }
            std::copy(f.pos.begin(), f.pos.end(), ws->positions().begin());
            ws->dist_tables().build_all(ws->positions());
            psi_t.evaluate_batch(*ws);
            hamiltonian_.psi_dependent_energy(*ws, kin);
            auto& lw = weight[static_cast<std::size_t>(i)];   // log-weight for now
            auto& en = e_new[static_cast<std::size_t>(i)];
            lw.resize(f.nw);
            en.resize(f.nw);
            const std::span<const double> lp = ws->log_psi();
            for (std::size_t w = 0; w < f.nw; ++w) {
                lw[w] = 2.0 * (lp[w] - f.log_psi[w]);
                en[w] = f.v_fixed[w] + kin[w];
            }
        }
    }

    // Weights, normalised against the largest log-weight for stability.
    double lw_max = -std::numeric_limits<double>::infinity();
    std::size_t n = 0;
    double e_old_sum = 0.0;
    for (std::size_t i = 0; i < frames.size(); ++i) {
        for (double v : weight[i]) lw_max = std::max(lw_max, v);
        for (double e : frames[i]->e_loc) e_old_sum += e;
        n += frames[i]->nw;
    }
    if (!std::isfinite(lw_max) || n == 0) return out;
    double sw = 0.0, sw2 = 0.0, swe = 0.0;
    for (std::size_t i = 0; i < frames.size(); ++i)
        for (std::size_t w = 0; w < frames[i]->nw; ++w) {
            const double wk = std::exp(weight[i][w] - lw_max);
            weight[i][w] = wk;
            if (wk == 0.0) continue;   // underflowed: no weight, and its E_L
                                       // may be non-finite (0·inf = NaN)
            sw  += wk;
            sw2 += wk * wk;
            swe += wk * e_new[i][w];
        }
    const double nd    = static_cast<double>(n);
    const double e_rw  = swe / sw;
    const double e_old = e_old_sum / nd;
    out.dE             = e_rw - e_old;
    out.n_eff_fraction = sw * sw / (nd * sw2);

    // Error of dE by the delta method: its fluctuation is the mean of
    //   y_k = w̃_k (E_L,new − E_new) − (E_L,old − <E_L,old>),  w̃ = N w / Σw,
    // and the serial correlation is handled by averaging y within each
    // (thread, block) group first — the same blocks the VMC error bar uses.
    std::size_t n_thr = 0, n_blocks = 0;
    for (const SampleFrame* f : frames) {
        n_thr    = std::max(n_thr, f->thread + 1);
        n_blocks = std::max(n_blocks, f->block + 1);
    }
    std::vector<double>      g_sum(n_thr * n_blocks, 0.0);
    std::vector<std::size_t> g_cnt(n_thr * n_blocks, 0);
    for (std::size_t i = 0; i < frames.size(); ++i) {
        const SampleFrame& f = *frames[i];
        const std::size_t  g = f.thread * n_blocks + f.block;
        for (std::size_t w = 0; w < f.nw; ++w) {
            const double wt = nd * weight[i][w] / sw;
            g_sum[g] += (wt == 0.0 ? 0.0 : wt * (e_new[i][w] - e_rw))
                        - (f.e_loc[w] - e_old);
            ++g_cnt[g];
        }
    }
    double m_sum = 0.0, m_sq = 0.0;
    std::size_t n_g = 0;
    for (std::size_t g = 0; g < g_sum.size(); ++g) {
        if (g_cnt[g] == 0) continue;
        const double m = g_sum[g] / static_cast<double>(g_cnt[g]);
        m_sum += m;
        m_sq  += m * m;
        ++n_g;
    }
    if (n_g >= 2) {
        const double gm  = m_sum / static_cast<double>(n_g);
        const double var = (m_sq / static_cast<double>(n_g) - gm * gm) *
                           static_cast<double>(n_g) / static_cast<double>(n_g - 1);
        out.dE_error = std::sqrt(std::max(var, 0.0) / static_cast<double>(n_g));
    }
    out.finite = std::isfinite(out.dE) && std::isfinite(out.dE_error) &&
                 std::isfinite(out.n_eff_fraction);
    return out;
}

LinearOptimize::Reweighted
LinearOptimize::correlated_step_energy(const std::vector<double>& update) {
    CostFunction cf;
    SampleStore  store;
    sample_vmc(cf, &store);
    return reweight(store, update);
}

LinearOptimize::EnergyTrend
LinearOptimize::energy_trend(std::span<const double> energies) {
    const std::size_t n = energies.size();
    if (n < 3)
        throw std::invalid_argument("energy_trend: needs at least 3 energies");
    const double k_bar = 0.5 * static_cast<double>(n - 1);
    double e_bar = 0.0;
    for (double e : energies) e_bar += e;
    e_bar /= static_cast<double>(n);
    double sxx = 0.0, sxy = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        const double dk = static_cast<double>(k) - k_bar;
        sxx += dk * dk;
        sxy += dk * (energies[k] - e_bar);
    }
    EnergyTrend t;
    t.slope = sxy / sxx;
    double rss = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        const double r = energies[k] - e_bar - t.slope * (static_cast<double>(k) - k_bar);
        rss += r * r;
    }
    t.slope_error = std::sqrt(rss / (static_cast<double>(n - 2) * sxx));
    return t;
}

bool LinearOptimize::energy_has_settled(std::span<const double> fresh_energies,
                                        int window, double energy_tol) {
    if (window < 3 || fresh_energies.size() < static_cast<std::size_t>(window))
        return false;
    const EnergyTrend t = energy_trend(fresh_energies.last(static_cast<std::size_t>(window)));
    return t.slope >= -std::max(energy_tol, 2.0 * t.slope_error);
}

// ‖δψ‖/‖ψ‖ alone can lie.  A spurious eigenvector along a near-null direction
// of S̄ gives a huge Δp whose Δpᵀ S̄ Δp rounds to zero or below, so the step
// reads as zero.  On the e–h bilayer at r_s = 2 such a step had reweighted
// ΔE = −2e28 with N_eff = 0 and ended a stage after 6 of 10 iterations.  A
// genuine step of size ε changes log ψ by O(ε) on every configuration, so its
// N_eff is 1 − O(ε²); a collapsed reweighting can only mean a large step.
bool LinearOptimize::step_is_small(double psi_step, const Reweighted& rw,
                                   double psi_step_tol, double min_effective_fraction) {
    return rw.finite && rw.n_eff_fraction >= min_effective_fraction &&
           psi_step < psi_step_tol;
}

bool LinearOptimize::step_within_bounds(std::span<const double> update,
                                        std::span<const double> params,
                                        double max_param_change) {
    double max_param = 1.0;
    for (double p : params) max_param = std::max(max_param, std::abs(p));
    for (double u : update)
        if (!(std::abs(u) <= max_param_change * max_param)) return false;   // NaN too
    return true;
}

// Two independent samples, so their errors add in quadrature.  Comparing with
// the sample the step started from, not with the lowest energy of the stage,
// keeps the test unbiased: the minimum of many noisy energies lies below the
// plateau (by ~2.5σ for 100 of them) and would undo good final steps.
bool LinearOptimize::step_raised_energy(const MeasuredEnergy& before,
                                        const MeasuredEnergy& after,
                                        double reject_sigmas) {
    return after.energy - before.energy >
           reject_sigmas * std::hypot(before.error, after.error);
}

bool LinearOptimize::adaptive_iteration(const Eigen::MatrixXd& Sbar,
                                         const Eigen::MatrixXd& Hbar,
                                         std::size_t n_opt,
                                         const SampleStore& store,
                                         double& best_shift_i,
                                         double& best_shift_s,
                                         std::vector<double>& update,
                                         Reweighted& chosen) {
    const std::vector<double> params_before = current_params();

    // Damp harder after a failure.  shift_i alone is NOT scale-invariant: a
    // parameter whose derivative is huge (Drummond α0 ~ (r−L)^C) has an H̄
    // diagonal that dwarfs any identity shift, so its step never shrinks.  The
    // overlap shift adds shift_s·S̄, which damps every direction in its own
    // metric.  A zero shift_s would stay zero under multiplication, so a failure
    // switches it on at QMCPACK's default overlap shift (shift_s = 1).
    auto grow_shifts = [&] {
        best_shift_i *= cfg_.shift_growth_factor;
        best_shift_s = (best_shift_s > 0.0) ? best_shift_s * cfg_.shift_growth_factor
                                            : kOverlapShiftSeed;
    };

    const std::vector<double> shifts_i =
        prepare_shifts(best_shift_i, cfg_.num_shifts, cfg_.shift_growth_factor);

    double              best_cost = std::numeric_limits<double>::max();
    double              best_si   = best_shift_i;
    std::vector<double> best_update;
    Reweighted          best_rw;
    int                 best_idx  = -1;

    // Solve at every shift and score each admissible candidate on the SAME
    // stored sample, so the ranking compares energies, not noise.  (The
    // shifted eigenvalue is no ranking: it rises monotonically with the shift,
    // so it always crowned the smallest one.)
    for (int idx = 0; idx < cfg_.num_shifts; ++idx) {
        const double si = shifts_i[idx];

        LinearMethodConfig lm_cfg;
        lm_cfg.shift_i = si;
        lm_cfg.shift_s = best_shift_s;
        lm_cfg.rescale = cfg_.lm_config.rescale;
        lm_cfg.xi      = cfg_.lm_config.xi;
        const LinearMethodResult lm_result = LinearMethod(lm_cfg).solve(Sbar, Hbar);

        double max_update = 0.0;
        for (std::size_t k = 0; k < n_opt; ++k)
            max_update = std::max(max_update, std::abs(lm_result.update[k]));
        if (!step_within_bounds(std::span<const double>(lm_result.update).first(n_opt),
                                params_before, cfg_.max_param_change)) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                          "optimize:   shift_i=%.3g shift_s=%.3g  step too large (%.3g)",
                          si, best_shift_s, max_update);
            io::Logger::info(buf);
            continue;
        }

        const Reweighted rw = reweight(store, lm_result.update);
        {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "optimize:   shift_i=%.3g shift_s=%.3g  dE=%+.3e +- %.1e  N_eff=%.2f",
                          si, best_shift_s, rw.dE, rw.dE_error, rw.n_eff_fraction);
            io::Logger::info(buf);
        }
        if (!rw.finite || rw.n_eff_fraction < cfg_.min_effective_fraction) continue;

        if (is_best_cost(idx, shifts_i, si, rw.dE, best_cost, best_si)) {
            best_cost   = rw.dE;
            best_si     = si;
            best_update = lm_result.update;
            best_rw     = rw;
            best_idx    = idx;
        }
    }

    if (best_idx < 0) {
        grow_shifts();
        return false;
    }

    // Reject only a SIGNIFICANT increase: dE is measured on a shared sample,
    // so its error bar is honest and small, unlike the old comparison of two
    // independent VMC runs against a naive σ/√N.
    const double bar = std::max(cfg_.cost_increase_tol,
                                cfg_.reject_sigmas * best_rw.dE_error);
    if (best_rw.dE > bar) {
        grow_shifts();
        return false;
    }

    apply_update(best_update);
    best_shift_i = best_si;
    // Relax the overlap shift by the same factor a failure grows it (QMCPACK:
    // shift_s /= shift_s_base while above 1e-2).  Halving it instead let a run
    // of failures ratchet shift_s up for good.
    if (best_shift_s > 1e-2) best_shift_s /= cfg_.shift_growth_factor;
    update = std::move(best_update);
    chosen = best_rw;
    return true;
}

// ── Main optimisation loop ───────────────────────────────────────────────────

// Pins every parameter outside the stage's scope and restores the flags on
// destruction, so a later optimizer driver sees the wavefunction as it was.
class LinearOptimize::ScopeGuard {
public:
    ScopeGuard(TrialWavefunction& psi, const std::string& scope) : psi_(psi) {
        const bool all = (scope != "nodal" && scope != "jastrow");
        for (auto& comp : psi_.components()) {
            const bool in_scope = all || (comp->is_nodal() == (scope == "nodal"));
            for (Parameter& p : comp->parameters()) {
                saved_.push_back(p.is_fixed);
                if (!in_scope) p.is_fixed = true;
            }
        }
    }
    ~ScopeGuard() {
        std::size_t k = 0;
        for (auto& comp : psi_.components())
            for (Parameter& p : comp->parameters()) p.is_fixed = saved_[k++];
    }
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

private:
    TrialWavefunction& psi_;
    std::vector<char>  saved_;
};

void LinearOptimize::run() {
    result_ = OptimizeResult{};

    if (cfg_.parameters != "all" && cfg_.parameters != "nodal" &&
        cfg_.parameters != "jastrow")
        throw std::runtime_error(
            "optimizer: unknown `parameters: " + cfg_.parameters +
            "` (expected all, nodal or jastrow)");
    const ScopeGuard scope(psi_, cfg_.parameters);

    // A complex factor's O = ∂ln ψ/∂p is complex, and the energy gradient
    // ∂E/∂p = 2⟨Re[O*(E_L − E)]⟩ = 2⟨Re O (Re E_L − E) + Im O Im E_L⟩ couples
    // Im O to Im E_L; the cost function here accumulates real O and Re E_L.
    // (A real factor of a complex trial, a Jastrow, has Im O = 0 and is fine.)
    for (const auto& comp : psi_.components())
        if (comp->is_complex() && comp->num_optimizable_params() > 0)
            throw std::runtime_error(
                "optimizer: " + comp->name() + " is complex and this stage would move " +
                std::to_string(comp->num_optimizable_params()) +
                " of its parameters, whose log-derivatives are complex; the linear method "
                "takes real ones.  Optimize them at the Gamma twist (a real determinant "
                "with the same parameters) and keep them out of this stage "
                "(`parameters: jastrow`, or fix them)");

    const std::size_t n_opt = psi_.total_optimizable_params();
    result_.final_params = current_params();

    if (n_opt == 0) {
        result_.converged = true;
        CostFunction cf;
        result_.final_energy_error = sample_vmc(cf);
        result_.final_energy = cf.mean_energy();
        return;
    }

    auto t0 = std::chrono::steady_clock::now();
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
    io::Logger::info("optimize: linear  max_iterations=" + std::to_string(cfg_.max_iterations));

    // Parameters whose component cannot supply ∂(∇logψ)/∂p lose the curvature
    // term of Hbar: their direction stays right, their step length does not —
    // and it can OVERSHOOT, not only fall short (Drummond α0 did, ~2.5×, which
    // made every step fail the energy check before it gained the term).
    for (const auto& comp : psi_.components()) {
        if (comp->num_optimizable_params() == 0) continue;
        if (comp->supports_parameter_gradient_derivatives()) continue;
        io::Logger::warn("optimize: " + comp->name() + " supplies no "
            "dE_L/dp, so its " +
            std::to_string(comp->num_optimizable_params()) +
            " parameter(s) are optimised without the curvature term of Hbar "
            "— direction right, step length unreliable (may overshoot).");
    }

    // Adaptive state.
    double best_shift_i = cfg_.start_shift_i;
    double best_shift_s = cfg_.start_shift_s;
    // Last 2 iterations: 00 = both rejected, which is the give-up signal.  It
    // starts all ONES: a default-constructed 00 makes the first failure look
    // like two in a row and ends the run on the spot.
    std::bitset<2> accept_history;
    accept_history.set();

    // The sample of the CURRENT wavefunction: matrices and stored
    // configurations.  One VMC per ACCEPTED step; a rejected iteration leaves ψ
    // where this sample was taken, so it is reused as is.
    CostFunction cf;
    SampleStore  store;
    bool need_sample = true;
    // Energy of each FRESH sample of this stage, in order (see 2b).
    std::vector<double> fresh_energies;
    // The latest fresh sample and the parameters it was taken at (see 7).
    MeasuredEnergy      last_fresh;
    std::vector<double> last_fresh_params;

    for (int iter = 0; iter < cfg_.max_iterations; ++iter) {
        // 1. Statistics of the current wavefunction.
        const bool fresh_sample = need_sample;
        if (need_sample) {
            cf = CostFunction{};
            const double error = sample_vmc(cf, &store);
            need_sample = false;
            last_fresh = {cf.mean_energy(), error};
            last_fresh_params = current_params();
        }
        const double cost_before = cf.mean_energy();
        result_.energy_history.push_back(cost_before);
        if (fresh_sample) fresh_energies.push_back(cost_before);

        {
            auto dt = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0);
            io::Logger::info("optimize: iteration " + std::to_string(iter + 1) + "/"
                             + std::to_string(cfg_.max_iterations)
                             + "  E=" + std::to_string(cost_before)
                             + "  time=" + format_duration(dt.count()));
        }

        // 2. Build S̄ and H̄ matrices.
        Eigen::MatrixXd Sbar, Hbar;
        cf.build_matrices(Sbar, Hbar, n_opt);

        // 2b. Convergence.  Converged when EITHER
        //       the UNDAMPED step is small:  ‖δψ‖/‖ψ‖ = √(Δpᵀ S̄ Δp) < psi_step_tol.
        //               At the optimum the energy gradient vanishes, so the
        //               linear method's own step is ≈ 0 whatever the rescale or
        //               the shifts; a step that got through only after heavy
        //               damping is small by construction and says nothing.  S̄
        //               is the covariance of ∂logψ/∂p, so α0 ~ 1e-4 counts like
        //               a parameter of size 1 (a bare max|Δp| cannot).  The
        //               step's reweighting must agree (step_is_small).
        //     OR the energy has settled: the least-squares slope of the energies
        //               of the last convergence_window FRESH samples is no
        //               descent beyond max(energy_tol, 2σ_slope).
        //     Why measured energies, and not the step's correlated-sampling ΔE?
        //     The step is fitted to the same sample that scores it, so its ΔE
        //     is biased low: on the e–h bilayer plateau at r_s = 5 every
        //     iteration predicted −1e-3 to −7e-3 Ha (2–14σ) while the measured
        //     energy stayed flat within ±0.003.  Judging one step at a time
        //     failed the other way: at r_s = 10 the energy fell by 3–5e-4 per
        //     iteration, just under each step's 2σ, so a two-step "futile" rule
        //     stopped after 14 of 60 iterations, 0.006 Ha above where 26 more
        //     iterations went.  Each fresh sample is independent, and a line
        //     through 15 of them resolves that drift at 5.8σ.  The noise of
        //     one measurement is taken from the residuals, not assumed.  A
        //     rejected iteration reuses its sample and adds no new energy.
        {
            LinearMethodConfig lm_cfg;
            lm_cfg.shift_i = cfg_.start_shift_i;
            lm_cfg.shift_s = 0.0;
            lm_cfg.rescale = cfg_.lm_config.rescale;
            lm_cfg.xi      = cfg_.lm_config.xi;
            const LinearMethodResult full = LinearMethod(lm_cfg).solve(Sbar, Hbar);
            double D = 0.0;
            for (std::size_t i = 0; i < n_opt; ++i)
                for (std::size_t j = 0; j < n_opt; ++j)
                    D += full.update[i] * Sbar(i + 1, j + 1) * full.update[j];
            const double psi_step = std::sqrt(std::max(D, 0.0));
            // Only a step within bounds is scored; a larger one is not small
            // either (rw stays non-finite, so step_is_small is false).
            const bool scorable = step_within_bounds(
                std::span<const double>(full.update).first(n_opt), current_params(),
                cfg_.max_param_change);
            const Reweighted rw = scorable ? reweight(store, full.update) : Reweighted{};
            char buf[160];
            if (scorable)
                std::snprintf(buf, sizeof(buf),
                              "optimize:   full step  |dpsi|/|psi|=%.2e  dE=%+.3e +- %.1e  N_eff=%.2f",
                              psi_step, rw.dE, rw.dE_error, rw.n_eff_fraction);
            else
                std::snprintf(buf, sizeof(buf),
                              "optimize:   full step  |dpsi|/|psi|=%.2e  beyond max_param_change, not scored",
                              psi_step);
            io::Logger::info(buf);
            bool settled = false;
            if (fresh_sample &&
                fresh_energies.size() >= static_cast<std::size_t>(cfg_.convergence_window)) {
                const EnergyTrend t = energy_trend(std::span<const double>(fresh_energies)
                    .last(static_cast<std::size_t>(cfg_.convergence_window)));
                std::snprintf(buf, sizeof(buf),
                              "optimize:   trend  dE/iteration=%+.2e +- %.1e  over %d samples",
                              t.slope, t.slope_error, cfg_.convergence_window);
                io::Logger::info(buf);
                settled = energy_has_settled(fresh_energies, cfg_.convergence_window,
                                             cfg_.energy_tol);
            }
            if (step_is_small(psi_step, rw, cfg_.psi_step_tol,
                              cfg_.min_effective_fraction) ||
                settled) {
                result_.converged          = true;
                result_.final_energy       = cost_before;
                result_.final_energy_error = last_fresh.error;
                result_.final_params       = current_params();
                break;
            }
        }

        // 3. Adaptive iteration: try multiple shifts, select the best.
        std::vector<double> update;
        Reweighted chosen;
        int attempts = 0;

        while (attempts < cfg_.max_failed_attempts) {
            if (adaptive_iteration(Sbar, Hbar, n_opt, store,
                                   best_shift_i, best_shift_s, update, chosen))
                break;
            ++attempts;
        }

        // 4. Update result tracking.
        result_.final_params = current_params();
        result_.shift_history.push_back(best_shift_i);
        ++result_.iterations;

        const bool accepted = (attempts < cfg_.max_failed_attempts);
        if (accepted) {
            // Reweighted estimate at the new parameters (the next VMC measures
            // it directly).
            result_.final_energy = cost_before + chosen.dE;
            need_sample = true;
            accept_history <<= 1;
            accept_history.set(0, true);

            char buf[128];
            std::snprintf(buf, sizeof(buf), "optimize:   accepted  dE=%+.3e +- %.1e",
                          chosen.dE, chosen.dE_error);
            io::Logger::info(buf);
        } else {
            result_.final_energy       = cost_before;
            result_.final_energy_error = last_fresh.error;
            accept_history <<= 1;
            accept_history.set(0, false);
        }

        // 6. If two consecutive iterations failed to find a valid update,
        //    the problem may be ill-conditioned.
        if (accept_history == 0x0) {
            break;
        }
    }

    // 7. Check the last step.  Every accepted step is chosen by correlated
    //    sampling on the configurations of the ψ it starts from, and the next
    //    iteration's fresh sample measures it.  Correlated sampling cannot see
    //    where the new ψ puts weight that the old sample never visited, and
    //    rarely it misjudges a step badly: on the e–h bilayer at r_s = 5 (N − 1
    //    pairs) a step scored ΔE = −3e-4 at N_eff = 1.00 and the fresh sample
    //    after it came out 0.02–0.03 Ha* higher (7–11σ), in each of three
    //    trajectories.  Mid-stage the next iteration undoes such a step; a
    //    stage that ends on one keeps it, because nothing measures the last
    //    step.  So it is measured here, and undone when it raised the energy
    //    by more than reject_sigmas combined σ (step_raised_energy).  An undo
    //    restores parameters the stage already measured; the walkers stay
    //    where the check sample left them, and the next stage re-equilibrates
    //    them.  Cost: one sample per stage, ~7% of an 8-iteration stage.
    //    (need_sample without any fresh sample means no iteration ran.)
    if (need_sample && !fresh_energies.empty()) {
        CostFunction check_cf;
        const double error = sample_vmc(check_cf);
        const MeasuredEnergy check{check_cf.mean_energy(), error};
        result_.last_step_checked  = true;
        result_.check_energy       = check.energy;
        result_.check_energy_error = check.error;
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "optimize:   check of the last step  E=%.6f +- %.1e  (from E=%.6f +- %.1e)",
                      check.energy, check.error, last_fresh.energy, last_fresh.error);
        io::Logger::info(buf);
        if (step_raised_energy(last_fresh, check, cfg_.reject_sigmas)) {
            set_params(last_fresh_params);
            result_.final_params       = last_fresh_params;
            result_.final_energy       = last_fresh.energy;
            result_.final_energy_error = last_fresh.error;
            result_.last_step_undone   = true;
            std::snprintf(buf, sizeof(buf),
                          "optimize: the last step raised the energy by %.3e (%.1f sigma); "
                          "undone, back to the parameters it started from",
                          check.energy - last_fresh.energy,
                          (check.energy - last_fresh.energy) /
                              std::hypot(check.error, last_fresh.error));
            io::Logger::warn(buf);
        } else {
            result_.final_energy       = check.energy;
            result_.final_energy_error = check.error;
        }
    }

    {
        auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0);
        io::Logger::milestone("optimize: complete  iter=" + std::to_string(result_.iterations)
                             + "  converged=" + (result_.converged ? "yes" : "no")
                             + "  E=" + std::to_string(result_.final_energy)
                             + " +- " + std::to_string(result_.final_energy_error)
                             + (result_.last_step_undone ? "  last step undone" : "")
                             + "  time=" + format_duration(dt.count()));
    }

}

std::string LinearOptimize::summary() const {
    std::stringstream ss;
    ss << "optimize: linear  iter=" << result_.iterations
       << "  conv=" << (result_.converged ? "yes" : "no")
       << "  E=" << result_.final_energy;
    ss << "  params=[";
    for (std::size_t i = 0; i < result_.final_params.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << result_.final_params[i];
    }
    ss << "]";
    if (!result_.shift_history.empty())
        ss << "  shift_final=" << result_.shift_history.back();
    return ss.str();
}

RunResult LinearOptimize::run_result() const {
    RunResult r;
    r.method = "optimizer";

    r.add_scalar("final_energy", result_.final_energy);
    r.add_scalar("final_energy_error", result_.final_energy_error);
    r.add_scalar("iterations", static_cast<double>(result_.iterations));
    r.add_scalar("converged", result_.converged ? 1.0 : 0.0);
    r.add_scalar("max_iterations", static_cast<double>(cfg_.max_iterations));
    r.add_scalar("last_step_checked", result_.last_step_checked ? 1.0 : 0.0);
    r.add_scalar("last_step_undone", result_.last_step_undone ? 1.0 : 0.0);
    if (result_.last_step_checked) {
        r.add_scalar("check_energy", result_.check_energy);
        r.add_scalar("check_energy_error", result_.check_energy_error);
    }

    r.add_metadata("method", "optimizer");
    r.add_metadata("type", "linear");
    r.add_metadata("seed", std::to_string(cfg_.seed));

    r.add_series("energy_history", result_.energy_history);
    r.add_series("shift_history", result_.shift_history);
    r.add_series("final_params", result_.final_params);

    return r;
}

}  // namespace alea
