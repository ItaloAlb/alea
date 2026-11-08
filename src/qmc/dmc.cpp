// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "qmc/dmc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "io/logger.h"
#include "numerics/green_function.h"
#include "numerics/statistics.h"
#include "qmc/reblock_diagnostics.h"
#include "qmc/sweep_blocking.h"
#include "population/walker.h"
#include "population/walker_ensemble.h"

namespace alea {

NodeConstraint node_constraint(const TrialWavefunction& psi) {
    if (psi.is_complex()) return NodeConstraint::FixedPhase;
    for (const auto& c : psi.components())
        if (c->is_nodal()) return NodeConstraint::FixedNode;
    return NodeConstraint::None;
}

const char* to_string(NodeConstraint nc) noexcept {
    switch (nc) {
        case NodeConstraint::FixedNode:  return "fixed_node";
        case NodeConstraint::FixedPhase: return "fixed_phase";
        case NodeConstraint::None:       break;
    }
    return "none";
}

namespace {
// Fraction of the cut walker sweeps above which the run warns: the cutoff is
// then shaping the projection, not just catching divergences.
constexpr double kMaxCutFraction = 1e-3;
// Mean population below this fraction of the target triggers a warning:
// walkers die faster than E_T compensates.
constexpr double kMinPopulationFraction = 0.8;

// N of the universal branch cutoff: the particles that diffuse.  A clamped
// nucleus (`initialization: fixed` with a huge mass, inv_mass ~1e-6 of an
// electron's) neither moves nor adds E_L fluctuations, so it is not counted:
// a particle counts when its largest inverse mass is at least 1e-3 of the
// largest in the system.
std::size_t diffusing_particles(const WalkerEnsemble& ensemble) {
    double max_inv = 0.0;
    for (const ParticleGroup& g : ensemble.groups())
        for (double im : g.inv_mass) max_inv = std::max(max_inv, im);
    std::size_t n = 0;
    for (const ParticleGroup& g : ensemble.groups()) {
        double group_inv = 0.0;
        for (double im : g.inv_mass) group_inv = std::max(group_inv, im);
        if (group_inv >= 1e-3 * max_inv) n += g.size();
    }
    return std::max<std::size_t>(n, 1);
}
}  // namespace

DMC::DMC(const DMCConfig& cfg, WalkerEnsemble& ensemble,
         const TrialWavefunction& psi, const Hamiltonian& hamiltonian)
    : cfg_(cfg),
      psi_template_(&psi),
      ham_template_(&hamiltonian),
      population_(cfg.tau, cfg.n_substeps, cfg.population_generations,
                  ensemble.target_walkers(), cfg.branch_cutoff,
                  cfg.branch_cutoff_alpha, diffusing_particles(ensemble)),
      pop_(ensemble.num_walkers(), ensemble.n_particles(), ensemble.dim(),
           ensemble.cell()),
      branch_rng_(numerics::stream_seed(cfg.seed, kDmcBranchStream, 0)) {
    const std::size_t nw  = ensemble.num_walkers();
    const std::size_t np  = ensemble.n_particles();
    const std::size_t dim = ensemble.dim();

    if (cfg_.reweighting == Reweighting::suppressed) {
        if (!(std::isfinite(cfg_.suppression_kappa) && cfg_.suppression_kappa > 0.0))
            throw std::invalid_argument(
                "dmc: the suppressed reweighting needs a finite kappa > 0");
        if (!(std::isfinite(cfg_.correlation_time) && cfg_.correlation_time >= 0.0))
            throw std::invalid_argument(
                "dmc: the correlation time of the suppressed reweighting must be "
                "finite and positive (0 measures it)");
        if (cfg_.correlation_min_sweeps < 2 ||
            cfg_.correlation_max_sweeps < cfg_.correlation_min_sweeps)
            throw std::invalid_argument(
                "dmc: correlation sampling needs 2 <= min_sweeps <= max_sweeps");
    }

    // Forward walking: every lag a whole number of steps of τ·n_substeps, the
    // snapshot interval dividing a block (so every block holds the same number
    // of pure measurements), and the largest lag inside the equilibration (so
    // the first accumulation block already has snapshots that old).
    if (!cfg_.forward_walking_lags.empty()) {
        const double step_time = cfg_.tau * static_cast<double>(cfg_.n_substeps);
        if (!(std::isfinite(step_time) && step_time > 0.0))
            throw std::invalid_argument(
                "dmc: forward walking needs time_step > 0 and sub_steps >= 1 (a lag "
                "is an imaginary time)");
        if (cfg_.forward_walking_interval == 0 || cfg_.n_steps <= 0 ||
            static_cast<std::size_t>(cfg_.n_steps) % cfg_.forward_walking_interval != 0)
            throw std::invalid_argument(
                "dmc: forward_walking interval (" +
                std::to_string(cfg_.forward_walking_interval) +
                ") must be >= 1 and divide steps_per_block (" +
                std::to_string(cfg_.n_steps) + ")");
        for (std::size_t i = 0; i < cfg_.forward_walking_lags.size(); ++i) {
            const double lag = cfg_.forward_walking_lags[i];
            if (!(std::isfinite(lag) && lag >= 0.0))
                throw std::invalid_argument("dmc: forward_walking lags must be finite and >= 0");
            if (i > 0 && !(lag > cfg_.forward_walking_lags[i - 1]))
                throw std::invalid_argument(
                    "dmc: forward_walking lags must be strictly increasing");
            const double steps = lag / step_time;
            const double whole = std::round(steps);
            if (std::abs(steps - whole) > 1e-9 * std::max(1.0, steps))
                throw std::invalid_argument(
                    "dmc: forward_walking lag " + std::to_string(lag) +
                    " is not a whole number of steps of time_step * sub_steps = " +
                    std::to_string(step_time));
            fw_lag_steps_.push_back(static_cast<std::size_t>(whole));
        }
        const std::size_t eq_steps = static_cast<std::size_t>(std::max(0, cfg_.n_equilibration_blocks)) *
                                     static_cast<std::size_t>(cfg_.n_steps);
        if (fw_lag_steps_.back() > eq_steps)
            throw std::invalid_argument(
                "dmc: the largest forward_walking lag (" + std::to_string(fw_lag_steps_.back()) +
                " steps) exceeds the equilibration (" + std::to_string(eq_steps) +
                " steps): snapshots start that many steps before accumulation");
        fw_templates_.resize(fw_lag_steps_.size());
    }

    // Per-particle, per-dimension proposal width.
    tau_inv_mass_.assign(np * dim, 0.0);
    sqrt_tau_inv_mass_.assign(np * dim, 0.0);
    for (const ParticleGroup& g : ensemble.groups())
        for (std::size_t i = g.first; i < g.last; ++i)
            for (std::size_t d = 0; d < dim; ++d) {
                tau_inv_mass_[i * dim + d]      = cfg_.tau * g.inv_mass[d];
                sqrt_tau_inv_mass_[i * dim + d] = std::sqrt(tau_inv_mass_[i * dim + d]);
            }

    // One global population (the source of truth for every per-walker datum).
    for (Walker& w : ensemble.walkers()) pop_.add_walker(w);
    pop_.load_all();

    // Worker count: one crowd (contiguous walker range) per worker.
    const int requested =
        cfg_.n_threads > 0 ? cfg_.n_threads : omp_get_max_threads();
    n_threads_ = std::max(1, std::min(requested, static_cast<int>(nw)));
    rngs_.reserve(static_cast<std::size_t>(n_threads_));
    for (int t = 0; t < n_threads_; ++t)
        rngs_.emplace_back(numerics::stream_seed(cfg_.seed, kDmcThreadStream,
                                                 static_cast<std::uint64_t>(t)));
}

void DMC::set_forward_walking_estimators(
    std::vector<std::vector<std::unique_ptr<Estimator>>> per_lag) {
    if (fw_lag_steps_.empty())
        throw std::logic_error("dmc: forward-walking estimators without forward_walking lags");
    if (per_lag.size() != fw_lag_steps_.size())
        throw std::logic_error("dmc: one forward-walking estimator set per lag");
    for (const auto& set : per_lag)
        for (const auto& e : set)
            if (e->needs_wavefunction())
                throw std::logic_error("dmc: forward walking measures diagonal observables "
                                       "only, not '" + e->type() + "'");
    fw_templates_ = std::move(per_lag);
}

bool DMC::dmc_accept(double log_ratio, double log_gf, double log_gb,
                     double u) noexcept {
    // P_acc = min(1, |ψ'/ψ|² · G_b / G_f)
    const double log_accept = 2.0 * log_ratio + log_gb - log_gf;
    return log_accept >= 0.0 || u < std::exp(log_accept);
}

std::size_t DMC::sweep(std::span<const double> sigma,
                       std::span<const double> tau_over_mass,
                       numerics::XoshiroRandom& rng,
                       std::size_t w0, std::size_t w_end) {
    // Unconditional uniforms (one per particle and walker) replace the old
    // draw-only-when-needed acceptance, so DMC trajectories differ from the
    // unblocked sweep while sampling the same Green's function; after that,
    // they are identical for every block size.  See sweep_blocking.h.
    draw_sweep_randoms(rng, pop_.n_particles(), pop_.dim(), sweep_random_stride_,
                       w0, w_end, sweep_randoms_);
    std::size_t n_accepted = 0;
    for (std::size_t b0 = w0; b0 < w_end; b0 += sweep_block_)
        n_accepted += sweep_walkers(sigma, tau_over_mass, b0,
                                    std::min(w_end, b0 + sweep_block_));
    return n_accepted;
}

std::size_t DMC::sweep_walkers(std::span<const double> sigma,
                               std::span<const double> tau_over_mass,
                               std::size_t w0, std::size_t w_end) {
    const std::size_t nw  = pop_.num_walkers();   // global stride
    const std::size_t np  = pop_.n_particles();
    const std::size_t dim = pop_.dim();
    const std::size_t stride = sweep_random_stride_;

    std::span<double> new_pos = pop_.new_pos();
    std::span<bool> mask = pop_.accept_mask();
    std::span<const double> pos = pop_.positions();
    std::span<const double> grad = pop_.gradients();
    std::size_t n_accepted = 0;
    // Fixed phase has no gate: phase_diffs() then holds a continuous arg(ψ'/ψ),
    // not a node-crossing flag (see NodeConstraint).
    const bool node_gate = node_constraint_ != NodeConstraint::FixedPhase;
    // τ_eff (Anderson, Per & Umrigar, JCP 160, 104110 (2024), Eq. 10):
    //   τ_eff = τ·Σ_i p_i m_i δr_i² / Σ_i m_i δr_i²,
    // p_i the acceptance probability of particle i's move (0 for a node
    // crossing) and δr_i its diffusion displacement √(τ/m)·ξ, so m δr² = τ ξ²
    // and the masses drop out: each particle weighs by its Gaussian draw ξ_i².
    // For unit masses this is their Eq. 10. No extra RNG draws or allocations.
    const bool track_tau_eff = cfg_.reweighting != Reweighting::bare;

    for (std::size_t p = 0; p < np; ++p) {
        // Refresh the current drift before the unchanged Green-function test.
        psi_.refresh_particle_derivatives(pop_, p, w0, w_end);
        const std::size_t pd_off = p * dim;
        const double tm_p = tau_over_mass[pd_off];  // isotropic τ/m for particle p
        const double* gaussian = sweep_randoms_.data() + p * (dim + 1) * stride; // [dim * stride]
        const double* uniform  = gaussian + dim * stride;                       // [stride]

        // ── DMC proposal: Umrigar-rescaled drift + diffusion ──────────
        // s_w = umrigar(τ/m, |∇log ψ_p|²) caps the drift step near a node
        // (|∇log ψ| → ∞) so a walker can't be flung into the nodal singularity;
        // s_w → τ/m away from nodes, recovering the bare drift.  drift_scale_ is
        // pre-sized in run() (no resize here → safe under crowd parallelism).
        for (std::size_t w = w0; w < w_end; ++w) drift_scale_[w] = 0.0;
        for (std::size_t d = 0; d < dim; ++d) {
            const std::size_t off = p * dim * nw + d * nw;
            for (std::size_t w = w0; w < w_end; ++w)
                drift_scale_[w] += grad[off + w] * grad[off + w];  // |∇log ψ_p|²
        }
        for (std::size_t w = w0; w < w_end; ++w)
            drift_scale_[w] = numerics::umrigar_drift_scale(tm_p, drift_scale_[w]);

        for (std::size_t d = 0; d < dim; ++d) {
            const std::size_t off = p * dim * nw + d * nw;
            const double sigma_pd = sigma[pd_off + d];
            for (std::size_t w = w0; w < w_end; ++w) {
                new_pos[d * nw + w] = pos[off + w]
                    + drift_scale_[w] * grad[off + w]
                    + sigma_pd * gaussian[d * stride + w];
            }
        }

        pop_.propose_move_batch(p, new_pos, w0, w_end);
        psi_.ratio_batch(pop_, p, w0, w_end);

        // ── Langevin Metropolis-Hastings acceptance ───────────────────
        // log(G_b/G_f) = Σ_d [ -Δ_d·(F_d+F'_d) + (τ/2m_d)(F_d²−F'_d²) ]
        const std::span<const double> log_ratios = pop_.log_ratios();
        const std::span<const double> new_grad = pop_.new_grad();
        const std::span<const double> phase = pop_.phase_diffs();

        auto gaussian_norm = [&](std::size_t w) {
            double xi2 = 0.0;
            for (std::size_t d = 0; d < dim; ++d) {
                const double g = gaussian[d * stride + w];
                xi2 += g * g;
            }
            return xi2;
        };
        for (std::size_t w = w0; w < w_end; ++w) {
            // ── Fixed-node gate (QMCPACK DMCUpdatePbyPFast.cpp:94) ─────────
            // Reject any move that crossed a node of ψ_T (the proposed-move
            // ratio changed sign → phase π).  This runs BEFORE the Metropolis
            // test, mirroring QMCPACK.  For nodeless trials the phase is 0, so
            // crosses_node() is false.  A rejected crossing leaves its
            // pre-drawn uniform unused.
            if (node_gate && crosses_node(phase[w])) {
                mask[w] = false;
                if (track_tau_eff) tau_eff_den_[w] += gaussian_norm(w);
                continue;
            }
            // Backward Umrigar scale from the proposed-position gradient; the
            // forward scale was stored in drift_scale_ during the proposal.  Using
            // the SAME scaled drift in this Metropolis–Hastings correction keeps
            // detailed balance exact, so the rescaling leaves the sampled
            // distribution unbiased (it only tames the near-node time-step error).
            double vsq_new = 0.0;
            for (std::size_t d = 0; d < dim; ++d) {
                const double ng = new_grad[d * nw + w];
                vsq_new += ng * ng;
            }
            const double sc_fwd = drift_scale_[w];
            const double sc_bwd = numerics::umrigar_drift_scale(tm_p, vsq_new);
            double log_corr = 0.0;
            for (std::size_t d = 0; d < dim; ++d) {
                const std::size_t off = p * dim * nw + d * nw;
                log_corr += numerics::langevin_log_ratio_drift_d(
                    new_pos[d * nw + w] - pos[off + w],
                    sc_fwd * grad[off + w],
                    sc_bwd * new_grad[d * nw + w],
                    tau_over_mass[pd_off + d]);
            }
            const double log_accept = 2.0 * log_ratios[w] + log_corr;
            const double p_accept = log_accept >= 0.0 ? 1.0 : std::exp(log_accept);
            mask[w] = log_accept >= 0.0 || uniform[w] < p_accept;
            if (mask[w]) ++n_accepted;
            if (track_tau_eff) {
                const double xi2 = gaussian_norm(w);
                tau_eff_num_[w] += p_accept * xi2;
                tau_eff_den_[w] += xi2;
            }
        }

        // ── Age-tracking: rr_proposed = Σ_d (τ/m_d)·ΔR_d² ──────────
        for (std::size_t w = w0; w < w_end; ++w) {
            double rr_w = 0.0;
            for (std::size_t d = 0; d < dim; ++d) {
                const std::size_t off = p * dim * nw + d * nw;
                const double dr = new_pos[d * nw + w] - pos[off + w];
                rr_w += tau_over_mass[pd_off + d] * dr * dr;
            }
            pop_.rr_proposed()[w] += rr_w;
            if (mask[w]) pop_.rr_accepted()[w] += rr_w;
        }

        psi_.apply_move_batch_deferred(pop_, p, mask, w0, w_end);
        pop_.apply_move(p, new_pos, pop_.new_grad(), pop_.new_lap(), mask,
                        w0, w_end);
    }
    return n_accepted;
}

DMC::StepResult DMC::advance_measure(numerics::XoshiroRandom& rng,
                                     std::size_t w0, std::size_t w_end) {
    const std::span<const double> sigma = sqrt_tau_inv_mass_;
    const std::span<const double> tau_m = tau_inv_mass_;

    std::span<double> rr_p    = pop_.rr_proposed();
    std::span<double> rr_a    = pop_.rr_accepted();
    std::span<double> weights = pop_.weights();
    std::span<double> e_prev  = pop_.old_energies();   // E_L before each sweep
    const std::span<const double> le = pop_.local_energies();
    for (std::size_t w = w0; w < w_end; ++w) {
        rr_p[w] = 0.0;
        rr_a[w] = 0.0;
        weights[w] = 1.0;        // accumulate the branch weight over the sweeps
        e_prev[w] = le[w];       // E_L before the first sweep
    }

    StepResult r;
    const bool suppressed = cfg_.reweighting == Reweighting::suppressed;
    const bool effective = cfg_.reweighting != Reweighting::bare;
    // f(R) of the suppressed weight (population_control.h): x = scale·τ_eff·√u.
    // No drift (u = 0) or no elapsed time (τ_eff = 0) is x = 0, also when
    // θ = 0 makes the scale infinite.
    auto suppression = [&](double u, double tau_eff) {
        return u > 0.0 && tau_eff > 0.0
            ? suppression_factor(suppression_scale_ * tau_eff * std::sqrt(u))
            : 1.0;
    };
    // Each drift-diffusion sweep evolves imaginary time τ and so contributes
    // its OWN branching factor.  Applying a single factor for the whole step
    // (as before) under-projects by a factor ~n_substeps and converges to a
    // wrong, VMC-biased steady state.  Evaluate H after every sweep and
    // multiply the per-walker weight by that sweep's branch factor.
    for (int s = 0; s < cfg_.n_substeps; ++s) {
        if (effective) {
            if (suppressed) drift_norms(drift_u_old_, w0, w_end);
            std::fill(tau_eff_num_.begin() + static_cast<std::ptrdiff_t>(w0),
                      tau_eff_num_.begin() + static_cast<std::ptrdiff_t>(w_end), 0.0);
            std::fill(tau_eff_den_.begin() + static_cast<std::ptrdiff_t>(w0),
                      tau_eff_den_.begin() + static_cast<std::ptrdiff_t>(w_end), 0.0);
        }
        r.n_accepted += sweep(sigma, tau_m, rng, w0, w_end);
        psi_.refresh_derivatives(pop_, w0, w_end);
        ham_.evaluate_batch(pop_, w0, w_end);
        const std::span<const double> enew = pop_.local_energies();
        if (effective) {
            if (suppressed) drift_norms(drift_u_new_, w0, w_end);
            for (std::size_t w = w0; w < w_end; ++w) {
                const double tau_eff = tau_eff_den_[w] > 0.0
                    ? cfg_.tau * tau_eff_num_[w] / tau_eff_den_[w]
                    : cfg_.tau;
                const double f_old = suppressed ? suppression(drift_u_old_[w], tau_eff) : 1.0;
                const double f_new = suppressed ? suppression(drift_u_new_[w], tau_eff) : 1.0;
                weights[w] *= population_.branch_weight(enew[w], e_prev[w], f_new,
                                                        f_old, tau_eff);
                r.n_cut += population_.is_cut(enew[w], e_prev[w], f_new, f_old) ? 1 : 0;
                r.tau_eff_ratio_sum += cfg_.tau > 0.0 ? tau_eff / cfg_.tau : 1.0;
                r.suppression_sum   += f_new;
                e_prev[w] = enew[w];
            }
            continue;
        }
        for (std::size_t w = w0; w < w_end; ++w) {
            weights[w] *= population_.branch_weight(enew[w], e_prev[w]);
            r.n_cut += population_.is_cut(enew[w], e_prev[w]) ? 1 : 0;
            e_prev[w] = enew[w];   // E_old for the next sweep
        }
    }

    std::span<std::size_t> ages = pop_.ages();
    const std::span<const double> efinal = pop_.local_energies();
    for (std::size_t w = w0; w < w_end; ++w) {
        if (cfg_.tau > 0.0) {
            if (rr_p[w] > 0.0) ages[w] = 0;
            else               ++ages[w];
        }
        r.energy_sum += weights[w] * efinal[w];
        r.weight_sum += weights[w];
    }
    return r;
}

void DMC::drift_norms(std::span<double> u, std::size_t w0, std::size_t w_end) const {
    const std::size_t nw  = pop_.num_walkers();   // global stride
    const std::size_t np  = pop_.n_particles();
    const std::size_t dim = pop_.dim();
    const std::span<const double> grad = pop_.gradients();   // [np * dim * nw]
    for (std::size_t w = w0; w < w_end; ++w) u[w] = 0.0;
    for (std::size_t p = 0; p < np; ++p)
        for (std::size_t d = 0; d < dim; ++d) {
            const double tm = tau_inv_mass_[p * dim + d];
            const std::size_t off = p * dim * nw + d * nw;
            for (std::size_t w = w0; w < w_end; ++w) u[w] += tm * grad[off + w] * grad[off + w];
        }
}

void DMC::measure_correlation_time() {
    // Up to 256 walkers: ρ(k) averaged over them is precise after a few
    // windows, and the sweeps of the rest would only add cost.  Their E_L is
    // checked after 256, 512, … sweeps until the window converges with ten
    // windows of data (or the cap is reached).
    //
    // θ belongs to the continuous drift-diffusion: Anderson et al. count
    // T_corr at one small step (τ0 = 0.01 Ha⁻¹) and use that c at every τ.
    // A step that is long against θ also changes the sampled dynamics
    // (lithium: θ = 0.118 at τ = 0.005, 0.153 at τ = 0.04), so the chains
    // move with a step τ_θ ≤ τ that leaves at least kSweepsPerTheta sweeps
    // per correlation time: if θ/τ_θ falls short, τ_θ becomes
    // min(τ_θ/2, θ/kSweepsPerTheta) and θ is measured again.
    constexpr std::size_t kCorrelationChains    = 256;
    constexpr std::size_t kWindowsOfData        = 10;
    constexpr double      kSweepsPerTheta       = 10.0;
    constexpr int         kMaxStepReductions    = 3;
    const std::size_t K = std::min(pop_.num_walkers(), kCorrelationChains);
    const int n_active = std::max(1, std::min(n_threads_, static_cast<int>(K)));
    std::vector<double> tau_m(tau_inv_mass_.size());   // τ_θ/m, [np * dim]
    std::vector<double> sigma(tau_m.size());           // √(τ_θ/m)

    auto advance_chains = [&] {
#pragma omp parallel for schedule(static) num_threads(n_active)
        for (int t = 0; t < n_active; ++t) {
            const std::size_t base = K / static_cast<std::size_t>(n_active);
            const std::size_t rem  = K % static_cast<std::size_t>(n_active);
            const std::size_t tt   = static_cast<std::size_t>(t);
            const std::size_t w0   = tt * base + std::min(tt, rem);
            const std::size_t w1   = w0 + base + (tt < rem ? 1 : 0);
            sweep(sigma, tau_m, rngs_[tt], w0, w1);
            psi_.refresh_derivatives(pop_, w0, w1);
            ham_.evaluate_batch(pop_, w0, w1);
        }
    };
    std::vector<double> series;   // [sweep * K + walker]
    series.reserve(cfg_.correlation_max_sweeps * K);
    numerics::AutocorrelationResult ac;
    double tau_theta = cfg_.tau;
    for (int reduction = 0;; ++reduction) {
        for (std::size_t i = 0; i < tau_m.size(); ++i) {
            tau_m[i] = tau_inv_mass_[i] * (tau_theta / cfg_.tau);
            sigma[i] = std::sqrt(tau_m[i]);
        }
        // Sokal's covariance estimator assumes stationary chains. These
        // discarded sweeps do not establish normalizability of |ψ_T|²; the
        // trial must have a stationary relative distribution (or a finite
        // periodic cell).
        for (std::size_t n = 0; n < cfg_.correlation_equilibration_sweeps; ++n)
            advance_chains();
        series.clear();
        std::size_t next_check = cfg_.correlation_min_sweeps;
        for (std::size_t n = 1; n <= cfg_.correlation_max_sweeps; ++n) {
            advance_chains();
            const std::span<const double> le = pop_.local_energies();
            series.insert(series.end(), le.begin(), le.begin() + static_cast<std::ptrdiff_t>(K));
            if (n == next_check || n == cfg_.correlation_max_sweeps) {
                ac = numerics::integrated_autocorrelation(series, K);
                if (ac.window_found && n >= kWindowsOfData * ac.window) break;
                next_check = next_check > cfg_.correlation_max_sweeps / 2
                    ? cfg_.correlation_max_sweeps : 2 * next_check;
            }
        }
        ac.tau_int = std::max(0.0, ac.tau_int);
        // Trapezoid rule: τ_θ·Σ_{k≥1}ρ(k) alone misses the half sweep at lag
        // 0 and made θ shrink with τ (r_s = 5 bilayer: 0.45, 0.41, 0.32 at
        // τ = 0.09, 0.18, 0.36, against 0.49, 0.50, 0.50 with it).
        correlation_time_ = numerics::integrated_time(ac, tau_theta);
        if (0.5 + ac.tau_int >= kSweepsPerTheta || reduction == kMaxStepReductions)
            break;
        tau_theta = std::min(0.5 * tau_theta, correlation_time_ / kSweepsPerTheta);
    }
    correlation_step_         = tau_theta;
    correlation_tau_int_      = ac.tau_int;
    correlation_sweeps_       = ac.n_steps;
    correlation_window_       = ac.window;
    correlation_window_found_ = ac.window_found;
    correlation_converged_ = ac.window_found &&
        correlation_sweeps_ >= kWindowsOfData * ac.window &&
        0.5 + ac.tau_int >= kSweepsPerTheta;
    if (!correlation_converged_)
        io::Logger::warn(
            "dmc: the correlation time of E_L did not converge within " +
            std::to_string(correlation_sweeps_) + " sweeps of step " +
            std::to_string(tau_theta) + " (window " + std::to_string(ac.window) +
            ", theta/step " + std::to_string(0.5 + ac.tau_int) + "); theta is not "
            "certified. Increase reweighting.correlation_sampling or set "
            "correlation_time explicitly");
}

void DMC::run() {
    // Components parallelise their branching gathers with default-sized
    // OpenMP teams; make that default the configured worker count for this
    // run (and restore the caller's afterwards, also on an exception).
    struct TeamSizeGuard {
        int saved = omp_get_max_threads();
        explicit TeamSizeGuard(int n) { omp_set_num_threads(n); }
        ~TeamSizeGuard() { omp_set_num_threads(saved); }
        TeamSizeGuard(const TeamSizeGuard&) = delete;
        TeamSizeGuard& operator=(const TeamSizeGuard&) = delete;
    } team_size(n_threads_);

    // Clone the (possibly optimiser-updated) wavefunction/Hamiltonian once —
    // a single shared instance drives the whole global population.
    psi_ = psi_template_->clone();
    ham_ = ham_template_->clone();
    node_constraint_ = alea::node_constraint(psi_);
    io::Logger::milestone(std::string("dmc: node constraint ") +
                          to_string(node_constraint_));

    // Instantiate observable estimators (one instance each; the global
    // population is accumulated sequentially → no per-thread clones needed).
    ests_.clear();
    for (const auto& tmpl : est_templates_) ests_.push_back(tmpl->clone());
    // Estimators of ψ_T ratios are measured on the whole population from this
    // serial point, so they split the walkers over the run's workers.
    for (const auto& e : ests_)
        if (e->needs_wavefunction())
            e->bind_wavefunction(psi_, numerics::stream_seed(cfg_.seed, kDmcEstimatorStream, 0),
                                 n_threads_);
    // Sweeps commit deferred and refresh every derivative before H: no
    // proposal Laplacian is read (set after binding, so the estimators'
    // copies keep the exact one).
    psi_.set_proposal_laplacian(false);

    // ── Initial evaluation (full population) ──────────────────────────
    psi_.evaluate_batch(pop_);
    ham_.evaluate_batch(pop_);

    const std::size_t target = population_.target_walkers();
    const std::size_t np     = pop_.n_particles();

    // Per-walker scratch, sized once to the largest population this run can
    // hold, so sweep() never resizes it inside the crowd-parallel region:
    // branching caps the population at n_max = 2·target+1, but the starting
    // ensemble (e.g. carried over from a stage with a larger target) can be
    // larger until the first branch.
    const std::size_t capacity = std::max(2 * target + 2, pop_.num_walkers());
    drift_scale_.assign(capacity, 0.0);   // Umrigar drift scale
    const bool suppressed = cfg_.reweighting == Reweighting::suppressed;
    const bool effective = cfg_.reweighting != Reweighting::bare;
    for (std::vector<double>* v : {&drift_u_old_, &drift_u_new_})
        v->assign(suppressed ? capacity : 0, 0.0);
    for (std::vector<double>* v : {&tau_eff_num_, &tau_eff_den_})
        v->assign(effective ? capacity : 0, 0.0);
    // The per-sweep random numbers use the capacity as walker stride, indexed
    // by global walker so crowds write disjoint entries; the cache-block size
    // comes from the per-walker state the initial evaluation just sized.
    sweep_random_stride_ = capacity;
    sweep_randoms_.assign(np * (pop_.dim() + 1) * sweep_random_stride_, 0.0);
    sweep_block_ = sweep_block_walkers(cfg_.sweep_block_walkers,
        psi_.walker_state_bytes() + pop_.walker_state_bytes(), sweep_random_stride_);

    // Forward walking: fresh copies of the per-lag estimators, every buffer
    // sized to the population capacity; steps count from the first
    // equilibration step, accumulation starts at n_eq · n_steps.
    fw_.reset();
    if (!fw_lag_steps_.empty()) {
        std::vector<std::vector<std::unique_ptr<Estimator>>> per_lag(fw_templates_.size());
        for (std::size_t i = 0; i < fw_templates_.size(); ++i)
            for (const auto& tmpl : fw_templates_[i]) per_lag[i].push_back(tmpl->clone());
        fw_ = std::make_unique<ForwardWalking>(
            fw_lag_steps_, cfg_.tau * static_cast<double>(cfg_.n_substeps),
            cfg_.forward_walking_interval, std::move(per_lag));
        fw_->prepare(capacity, np, pop_.dim(), pop_.cell(),
                     static_cast<std::size_t>(cfg_.n_equilibration_blocks) *
                         static_cast<std::size_t>(cfg_.n_steps));
        io::Logger::milestone("dmc: forward walking " + std::to_string(fw_->n_lags()) +
                              " lags, " + std::to_string(fw_->n_slots()) + " snapshots of " +
                              std::to_string(capacity) + " walkers (" +
                              std::to_string(fw_->memory_bytes() >> 20) + " MiB)");
    }

    // Suppressed reweighting: θ from the config or measured on the starting
    // walkers (whose positions those sweeps move, as more VMC equilibration
    // with the DMC move would), then x = κ·τ_eff·√(u/(N·τ·θ)).
    if (suppressed) {
        if (cfg_.correlation_time > 0.0) {
            correlation_time_ = cfg_.correlation_time;
        } else if (cfg_.tau > 0.0) {   // τ = 0 has no suppression to scale
            measure_correlation_time();
        }
        const double n_part = population_.n_particles();
        suppression_scale_ = suppression_scale(cfg_.suppression_kappa, n_part, cfg_.tau,
                                               correlation_time_);
        std::string line = "dmc: reweighting suppressed: theta=" +
                           std::to_string(correlation_time_) + " (kappa=" +
                           std::to_string(cfg_.suppression_kappa) + ", N=" +
                           std::to_string(static_cast<long>(n_part));
        if (correlation_sweeps_ > 0)
            line += ", tau_int=" + std::to_string(correlation_tau_int_) +
                    " sweeps from " + std::to_string(correlation_sweeps_);
        io::Logger::milestone(line + ")");
    }

    {
        // E_T = E_ref = mean local energy of the starting population.
        const std::span<const double> le = pop_.local_energies();
        const std::size_t n = pop_.num_walkers();
        double e_sum = 0.0;
        for (std::size_t w = 0; w < n; ++w) e_sum += le[w];
        population_.start(e_sum / static_cast<double>(n));
    }
    // The cutoff's E_L spread: from the starting population, refreshed after
    // every equilibration block, frozen for accumulation.
    auto refresh_energy_spread = [&] {
        population_.set_energy_spread(numerics::robust_spread(
            pop_.local_energies().first(pop_.num_walkers()), spread_scratch_));
    };
    refresh_energy_spread();

    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();

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

    // Contiguous walker range [w0, w_end) handled by worker t of nt workers.
    auto crowd_range = [](std::size_t nw, int nt,
                          int t) -> std::pair<std::size_t, std::size_t> {
        const std::size_t base = nw / static_cast<std::size_t>(nt);
        const std::size_t rem  = nw % static_cast<std::size_t>(nt);
        const std::size_t tt   = static_cast<std::size_t>(t);
        const std::size_t w0   = tt * base + std::min(tt, rem);
        const std::size_t cnt  = base + (tt < rem ? 1 : 0);
        return {w0, w0 + cnt};
    };

    std::size_t total_accepted = 0, total_attempts = 0;
    // Walker sweeps tapered by the branch cutoff, and all walker sweeps, per
    // phase (index 0 equilibration, 1 accumulation).
    std::size_t n_cut[2] = {0, 0}, n_sweeps[2] = {0, 0};
    // Suppressed reweighting during accumulation: Σ τ_eff/τ and Σ f(R').
    double tau_eff_ratio_acc = 0.0, suppression_acc = 0.0;
    // Team size can shrink with population, but result storage need not be
    // allocated every global step. Only the active prefix is read below.
    std::vector<StepResult> rs(static_cast<std::size_t>(n_threads_));
    std::vector<std::size_t> src;
    src.reserve(2 * target + 2);
    // Steps since the first equilibration step (the forward-walking schedule).
    std::size_t step_index = 0;

    // One global step: parallel advance over disjoint crowds → join → global
    // branch (QMCPACK-batched) → carry wavefunction caches → adapt E_trial.
    // Returns this step's (Σ w·E_L, Σ w) for the energy estimator.
    auto run_one_step = [&](bool measure) -> std::pair<double, double> {
        const std::size_t nw = pop_.num_walkers();
        const int n_active = std::max(1, std::min(n_threads_, static_cast<int>(nw)));
#pragma omp parallel for schedule(static) num_threads(n_active)
        for (int t = 0; t < n_active; ++t) {
            const auto range = crowd_range(nw, n_active, t);
            rs[static_cast<std::size_t>(t)] =
                advance_measure(rngs_[static_cast<std::size_t>(t)],
                                range.first, range.second);
        }
        if (measure) {
            // Mixed estimator: accumulate the global population before the
            // branch, weighted by each walker's current DMC weight.
            for (auto& e : ests_) e->accumulate(pop_);
            if (measure_hook_) measure_hook_(pop_, psi_);
        }
        // Pure estimators: snapshot and measure with the same pre-branch
        // weights (during equilibration it only stores snapshots).
        if (fw_) fw_->observe(pop_, step_index, ham_);

        double e = 0.0, wsum = 0.0;
        for (int t = 0; t < n_active; ++t) {
            const StepResult& r = rs[static_cast<std::size_t>(t)];
            e    += r.energy_sum;
            wsum += r.weight_sum;
            total_accepted += r.n_accepted;
            n_cut[measure ? 1 : 0] += r.n_cut;
            if (measure) {
                tau_eff_ratio_acc += r.tau_eff_ratio_sum;
                suppression_acc   += r.suppression_sum;
            }
        }
        total_attempts += nw * static_cast<std::size_t>(cfg_.n_substeps) * np;
        n_sweeps[measure ? 1 : 0] += nw * static_cast<std::size_t>(cfg_.n_substeps);

        // Reference energy follows the measured ensemble energy (instantaneous
        // during equilibration, cumulative average during accumulation) so the
        // branch-weight window stays centred and the population is unbiased.
        const double ens_e = wsum > 0.0 ? e / wsum : population_.etrial();

        pop_.branch(branch_rng_, target, src);
        psi_.permute_caches(src);
        if (fw_) fw_->branch(src);
        population_.update(ens_e, wsum, /*accumulate=*/measure);
        ++step_index;
        return {e, wsum};
    };

    // ── Equilibration ─────────────────────────────────────────────────
    const int n_eq = cfg_.n_equilibration_blocks;
    eq_etrial_.reserve(static_cast<std::size_t>(n_eq));
    eq_population_.reserve(static_cast<std::size_t>(n_eq));
    eq_block_energies_.reserve(static_cast<std::size_t>(n_eq));
    io::Logger::milestone("dmc: equilibration " + std::to_string(n_eq) + " blocks");

    for (int b = 0; b < n_eq; ++b) {
        // The same mixed estimator as accumulation: sum the already computed
        // pre-branch weighted energies, with no extra H evaluations or draws.
        // Recording the transient is necessary to diagnose equilibration;
        // these samples never enter the reported production energy.
        double blk_e = 0.0, blk_w = 0.0;
        for (int s = 0; s < cfg_.n_steps; ++s) {
            const auto step = run_one_step(/*measure=*/false);
            blk_e += step.first;
            blk_w += step.second;
        }
        eq_block_energies_.push_back(blk_w > 0.0 ? blk_e / blk_w : 0.0);
        eq_etrial_.push_back(population_.etrial());
        eq_population_.push_back(static_cast<double>(pop_.num_walkers()));
        refresh_energy_spread();
    }
    {
        const double sigma = population_.energy_spread();
        const double e_cut = population_.energy_cutoff();
        std::string line = std::string("dmc: branch cutoff ") + to_string(population_.cutoff());
        if (population_.cutoff() == BranchCutoff::universal)
            line += " (alpha=" + std::to_string(population_.alpha()) + ")";
        line += ": sigma(E_L)=" + std::to_string(sigma) + "  e_cut=" + std::to_string(e_cut);
        if (sigma > 0.0 && std::isfinite(e_cut))
            line += " = " + std::to_string(e_cut / sigma) + " sigma";
        io::Logger::milestone(line);
    }

    // Milestone checkpoint: the equilibrated population, before any
    // accumulation — a later run can redo/extend accumulation from here
    // without re-paying equilibration.
    if (checkpoint_hook_) checkpoint_hook_(pop_, "eq", 0);

    // ── Accumulation ──────────────────────────────────────────────────
    const int n_acc = cfg_.n_accumulation_blocks;
    block_energies_.reserve(static_cast<std::size_t>(n_acc));
    block_etrial_.reserve(static_cast<std::size_t>(n_acc));
    block_population_.reserve(static_cast<std::size_t>(n_acc));
    block_total_weight_.reserve(static_cast<std::size_t>(n_acc));
    io::Logger::milestone("dmc: accumulation " + std::to_string(n_acc) + " blocks");

    // Start the cumulative reference-energy average fresh for accumulation.
    population_.begin_accumulation();

    for (int b = 0; b < n_acc; ++b) {
        for (auto& e : ests_) e->start_block();
        if (fw_) fw_->start_block();
        double blk_e = 0.0, blk_w = 0.0;
        for (int s = 0; s < cfg_.n_steps; ++s) {
            const auto step = run_one_step(/*measure=*/true);
            blk_e += step.first;
            blk_w += step.second;
        }
        for (auto& e : ests_) e->stop_block();
        if (fw_) fw_->stop_block();
        // One weighted-mean energy per block (global population).
        block_energies_.push_back(blk_w > 0.0 ? blk_e / blk_w : 0.0);
        block_etrial_.push_back(population_.etrial());
        block_population_.push_back(static_cast<double>(pop_.num_walkers()));
        block_total_weight_.push_back(
            blk_w / static_cast<double>(std::max(1, cfg_.n_steps)));

        // Quarter milestones (crash recovery + population-evolution
        // diagnostics).  Integer division; the > guards drop milestones that
        // coincide when n_acc < 4 so each tag fires at most once.
        if (checkpoint_hook_) {
            const int done = b + 1;
            if (done == n_acc / 4)
                checkpoint_hook_(pop_, "a25", done);
            else if (done == n_acc / 2 && n_acc / 2 > n_acc / 4)
                checkpoint_hook_(pop_, "a50", done);
            else if (done == 3 * n_acc / 4 && 3 * n_acc / 4 > n_acc / 2)
                checkpoint_hook_(pop_, "a75", done);
        }
    }

    // ── Acceptance ratio ──────────────────────────────────────────────
    acceptance_ratio_ = total_attempts > 0
        ? static_cast<double>(total_accepted) / static_cast<double>(total_attempts)
        : 0.0;

    // ── Branch-cutoff and population diagnostics ──────────────────────
    auto fraction = [](std::size_t num, std::size_t den) {
        return den > 0 ? static_cast<double>(num) / static_cast<double>(den) : 0.0;
    };
    cut_fraction_eq_  = fraction(n_cut[0], n_sweeps[0]);
    cut_fraction_acc_ = fraction(n_cut[1], n_sweeps[1]);
    if (effective && n_sweeps[1] > 0) {
        tau_eff_ratio_mean_ = tau_eff_ratio_acc / static_cast<double>(n_sweeps[1]);
        suppression_mean_   = suppression_acc / static_cast<double>(n_sweeps[1]);
    }
    if (cut_fraction_acc_ > kMaxCutFraction)
        io::Logger::warn("dmc: the branch cutoff tapered " +
                         std::to_string(cut_fraction_acc_) +
                         " of the walker sweeps during accumulation: it shapes the "
                         "projection and biases the energy");
    if (population_mean_fraction() < kMinPopulationFraction)
        io::Logger::warn("dmc: the mean population is " +
                         std::to_string(population_mean_fraction()) +
                         " of the target: walkers die faster than E_T compensates "
                         "and the energy is biased");

    // ── Statistics ────────────────────────────────────────────────────
    // The block energies form one branched chain, correlated over several
    // blocks through the walkers and E_T; the reblocked error accounts for it.
    energy_stats_ = numerics::reblock(block_energies_);
    energy_       = energy_stats_.mean;
    energy_error_ = energy_stats_.error;
    if (!error_reliable(energy_stats_, block_energies_.size()))
        io::Logger::warn(
            "dmc: the energy error bar is not reliable (no reblocking plateau or "
            "fewer than " + std::to_string(kMinEffectiveBlocks) +
            " effective blocks); run more accumulation blocks");

    energy_variance_ = 0.0;
    if (block_energies_.size() >= 2) {
        double var_sum = 0.0;
        for (double blk : block_energies_) {
            const double d = blk - energy_;
            var_sum += d * d;
        }
        energy_variance_ = var_sum
            / static_cast<double>(block_energies_.size() - 1);
    }

    etrial_final_ = population_.etrial();

    {
        const auto dt = std::chrono::duration<double>(Clock::now() - t0);
        walltime_s_ = dt.count();
        io::Logger::milestone("dmc: complete  E=" + std::to_string(energy_)
                             + " +/- " + std::to_string(energy_error_)
                             + "  var=" + std::to_string(energy_variance_)
                             + "  acc=" + std::to_string(acceptance_ratio_)
                             + "  time=" + format_duration(dt.count()));
    }
}

double DMC::population_mean_fraction() const noexcept {
    const std::size_t target = population_.target_walkers();
    if (block_population_.empty() || target == 0) return 1.0;
    double sum = 0.0;
    for (double n : block_population_) sum += n;
    return sum / static_cast<double>(block_population_.size()) / static_cast<double>(target);
}

std::string DMC::summary() const {
    std::stringstream ss;
    ss << "dmc: E = " << energy_ << " +/- " << energy_error_
       << "  var = " << energy_variance_
       << "  acc = " << acceptance_ratio_;
    if (!error_reliable(energy_stats_, block_energies_.size()))
        ss << "\ndmc: WARNING: no reliable reblocking plateau (or < " << kMinEffectiveBlocks
           << " effective blocks); the error bar may be too small, increase accumulation_blocks";
    if (cut_fraction_acc_ > kMaxCutFraction)
        ss << "\ndmc: WARNING: the branch cutoff tapered " << cut_fraction_acc_
           << " of the walker sweeps (e_cut = " << population_.energy_cutoff()
           << "); the energy is biased";
    if (population_mean_fraction() < kMinPopulationFraction)
        ss << "\ndmc: WARNING: mean population " << population_mean_fraction()
           << " of the target; the energy is biased";
    return ss.str();
}

RunResult DMC::run_result() const {
    RunResult r;
    r.method = "dmc";

    // ── Scalars (aggregate results + run parameters) ──────────────────────
    r.add_scalar("energy", energy_);
    r.add_scalar("energy_error", energy_error_);
    add_reblock_diagnostics(r, "energy", energy_stats_, block_energies_.size());
    r.add_scalar("energy_variance", energy_variance_);
    r.add_scalar("acceptance_ratio", acceptance_ratio_);
    r.add_scalar("etrial_final", etrial_final_);
    r.add_scalar("walltime_s", walltime_s_);
    r.add_scalar("tau", cfg_.tau);
    r.add_scalar("n_walkers",
                 static_cast<double>(population_.target_walkers()));
    r.add_scalar("n_threads", static_cast<double>(n_threads_));
    r.add_scalar("n_equilibration_blocks",
                 static_cast<double>(cfg_.n_equilibration_blocks));
    r.add_scalar("n_accumulation_blocks",
                 static_cast<double>(cfg_.n_accumulation_blocks));
    r.add_scalar("n_steps", static_cast<double>(cfg_.n_steps));
    r.add_scalar("n_substeps", static_cast<double>(cfg_.n_substeps));
    r.add_scalar("population_generations", cfg_.population_generations);
    // Branch cutoff: the frozen spread, the diffusing-particle count N and
    // e_cut (+inf when nothing is cut; alpha is written for the universal
    // scheme only), and the fraction of walker sweeps it tapered per phase.
    r.add_scalar("branch_cutoff_sigma", population_.energy_spread());
    r.add_scalar("branch_cutoff_particles", population_.n_particles());
    r.add_scalar("branch_cutoff_energy", population_.energy_cutoff());
    if (population_.cutoff() == BranchCutoff::universal)
        r.add_scalar("branch_cutoff_alpha", population_.alpha());
    r.add_scalar("branch_cutoff_fraction_eq", cut_fraction_eq_);
    r.add_scalar("branch_cutoff_fraction", cut_fraction_acc_);
    // Suppressed reweighting: κ, θ and how θ was obtained (the step τ_θ, tau_int
    // in sweeps of it and the sweeps measured, all 0 when the config gave θ),
    // and the accumulation means of τ_eff/τ and of f(R').
    if (cfg_.reweighting == Reweighting::suppressed) {
        r.add_scalar("suppression_kappa", cfg_.suppression_kappa);
        r.add_scalar("correlation_time", correlation_time_);
        r.add_scalar("correlation_step", correlation_step_);
        r.add_scalar("correlation_tau_int", correlation_tau_int_);
        r.add_scalar("correlation_sweeps", static_cast<double>(correlation_sweeps_));
        r.add_scalar("correlation_window_found", correlation_window_found_ ? 1.0 : 0.0);
        r.add_scalar("correlation_window", static_cast<double>(correlation_window_));
        r.add_scalar("correlation_converged", correlation_converged_ ? 1.0 : 0.0);
        r.add_scalar("correlation_equilibration_sweeps", correlation_sweeps_ > 0
            ? static_cast<double>(cfg_.correlation_equilibration_sweeps) : 0.0);
    }
    if (cfg_.reweighting != Reweighting::bare) {
        r.add_scalar("tau_eff_ratio", tau_eff_ratio_mean_);
        r.add_scalar("suppression_mean", suppression_mean_);
    }

    // ── String metadata ───────────────────────────────────────────────────
    r.add_metadata("method", "dmc");
    r.add_metadata("branch_cutoff", to_string(population_.cutoff()));
    r.add_metadata("reweighting", to_string(cfg_.reweighting));
    r.add_metadata("seed", std::to_string(cfg_.seed));
    r.add_metadata("rng_streams",
                   "xoshiro256++ with ziggurat Gaussians per worker t and for "
                   "branching, seeded with splitmix64-hashed "
                   "(seed, dmc thread stream, t) and (seed, dmc branch stream, 0)");
    r.add_metadata("energy_error_model", "Flyvbjerg-Petersen reblocking");
    r.add_metadata("node_constraint", to_string(node_constraint_));

    // ── Time series (one value per block) ─────────────────────────────────
    r.add_series("acc_block_energy", block_energies_);
    r.add_series("acc_population", block_population_);
    r.add_series("acc_total_weight", block_total_weight_);
    r.add_series("acc_etrial", block_etrial_);
    r.add_series("eq_population", eq_population_);
    r.add_series("eq_etrial", eq_etrial_);
    r.add_series("eq_block_energy", eq_block_energies_);

    // Observable estimators (e.g. g(r)) append their own named series, each
    // instance under its output label.
    for (const auto& e : ests_) e->write_results(r);
    // Pure (forward-walking) estimates, every key carrying its lag index fw<i>.
    if (fw_) fw_->write_results(r);
    r.check_unique_keys();

    return r;
}

}  // namespace alea
