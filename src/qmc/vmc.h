// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "../estimators/estimator.h"
#include "../numerics/xoshiro_random.h"
#include "../numerics/statistics.h"
#include "../population/walker_set.h"
#include "../wavefunction/trial_wavefunction.h"
#include "../hamiltonian/hamiltonian.h"
#include "driver.h"

namespace alea {

class WalkerEnsemble;

// Stream kind for numerics::stream_seed(seed, kind, index): thread t of a VMC
// stage samples with stream_seed(seed, kVmcThreadStream, t).  Hashing keeps
// every thread of every seed on its own stream; seed + t would hand seed s + 1
// the streams of seed s, shifted by one thread.
inline constexpr std::uint64_t kVmcThreadStream = 0x766d632d74687264ULL;  // "vmc-thrd"
// The estimators of thread t that evaluate ψ_T (Estimator::bind_wavefunction)
// draw from stream_seed(seed, kVmcEstimatorStream, t).
inline constexpr std::uint64_t kVmcEstimatorStream = 0x766d632d6573746dULL;  // "vmc-estm"

struct VMCConfig {
    int           n_equilibration_blocks = 100;
    int           n_accumulation_blocks  = 100;
    int           n_steps                = 10;
    int           n_substeps             = 1;
    double        tau                    = 0.01;
    bool          use_drift              = false;  // true → Langevin dynamics
    int           n_threads              = 1;
    std::uint64_t seed                   = 42;     // stage seed (kVmcThreadStream)
    // Walkers swept together through all particles (sweep_blocking.h);
    // 0 sizes the block from the per-walker state.  Never changes results.
    std::size_t   sweep_block_walkers    = 0;
};

class VMC : public Driver {
public:
    VMC(const VMCConfig&     cfg,
        WalkerEnsemble&      ensemble,
        const TrialWavefunction& psi,
        const Hamiltonian&       hamiltonian);

    void run() override;

    [[nodiscard]] std::string summary() const override;
    [[nodiscard]] RunResult   run_result() const override;

    // Carry-over: the |ψ_T|²-distributed configurations every thread's chain
    // ended on (thread 0's walkers first), and a reload of the ensemble's
    // positions into the per-thread WalkerSets before run().
    [[nodiscard]] std::vector<double> final_configurations() const override {
        std::vector<double> out;
        for (const WalkerSet& ws : wsets_) ws.append_configurations(out);
        return out;
    }
    void reload_configurations() override {
        for (WalkerSet& ws : wsets_) ws.load_all();
    }

    double energy()           const noexcept { return energy_; }
    double energy_error()     const noexcept { return energy_error_; }
    double energy_variance()  const noexcept { return energy_variance_; }
    double acceptance_ratio() const noexcept { return acceptance_ratio_; }
    // ⟨Im E_L⟩ ± error for a complex trial (0 otherwise); should vanish.
    double energy_imag()       const noexcept { return energy_imag_; }
    double energy_imag_error() const noexcept { return energy_imag_error_; }

    std::span<const double> block_energies() const noexcept { return block_energies_; }
    std::span<const double> tau_inv_mass()   const noexcept { return tau_inv_mass_; }

    // Set a callback invoked after each hamiltonian.evaluate_batch() during
    // accumulation.  Receives the per-thread WalkerSet (with local_energies()
    // populated) and the per-thread wavefunction clone.  Used by the optimiser
    // to collect parameter derivatives on the same configurations.
    using MeasureHook = std::function<void(const WalkerSet&,
                                           const TrialWavefunction&)>;
    void set_measure_hook(MeasureHook hook) { measure_hook_ = std::move(hook); }

    // Opt-in observable estimators (e.g. g(r)).  Cloned per thread in run();
    // their results are appended to run_result().
    void set_estimators(std::vector<std::unique_ptr<Estimator>> ests) {
        est_templates_ = std::move(ests);
    }

private:
    // Accumulated measurements from one VMC sampling step.
    struct StepResult {
        double      energy_sum    = 0.0;  // Σ_w E_L(w)
        double      energy_sq_sum = 0.0;  // Σ_w E_L(w)²
        double      energy_imag_sum = 0.0;  // Σ_w Im E_L(w)  (complex ψ only)
        std::size_t n_samples     = 0;    // walkers measured this step
        std::size_t n_accepted    = 0;    // accepted single-particle moves
    };

    VMCConfig cfg_;

    // Templates — cloned lazily in run() so a previous driver (optimiser)
    // can update the wavefunction before VMC sees it.
    WalkerEnsemble*           ensemble_     = nullptr;
    const TrialWavefunction*  psi_template_ = nullptr;
    const Hamiltonian*        ham_template_ = nullptr;

    // One WalkerSet, TrialWavefunction, Hamiltonian, and RNG per thread.
    // psis_ and hamiltonians_ are populated at the start of run().
    std::vector<WalkerSet>               wsets_;
    std::vector<TrialWavefunction>       psis_;
    std::vector<Hamiltonian>             hamiltonians_;
    std::vector<numerics::XoshiroRandom> rngs_;
    // One sweep's random numbers per thread, [np * (dim + 1) * nw_t]
    // (layout in sweep_blocking.h), sized at the start of run().
    std::vector<std::vector<double>> sweep_randoms_;
    std::vector<std::size_t>         sweep_blocks_;   // walkers per cache block, per thread

    // tau / m_i_d  and  sqrt(tau / m_i_d) — per particle, per dimension.
    // Layout: [p * dim + d]. Supports anisotropic effective mass.
    std::vector<double> tau_inv_mass_;
    std::vector<double> sqrt_tau_inv_mass_;

    double energy_           = 0.0;
    double energy_error_     = 0.0;  // reblocked (energy_stats_.error)
    double energy_variance_  = 0.0;
    // Reblocking of the per-thread block energies, one chain per thread.
    numerics::ReblockResult energy_stats_, energy_imag_stats_;
    // Wall time of run(), mirroring DMC::walltime_s_ (dmc.h:124).  VMC computed
    // it for the completion log but never reported it, so a VMC run left no
    // machine-readable timing behind.
    double walltime_s_       = 0.0;
    double acceptance_ratio_ = 0.0;
    bool   is_complex_        = false;
    double energy_imag_       = 0.0;
    double energy_imag_error_ = 0.0;

    std::vector<double> block_energies_;

    // Observable estimators: templates set via set_estimators(), and the final
    // per-estimator results after the per-thread clones are reduced in run().
    std::vector<std::unique_ptr<Estimator>> est_templates_;
    std::vector<std::unique_ptr<Estimator>> ests_;

    MeasureHook measure_hook_;

    // Symmetric Metropolis acceptance: accept iff u < |ψ'/ψ|² = exp(2·log_ratio).
    static bool accept(double log_ratio, double u) noexcept;

    // One sweep over all particles: draw the sweep's random numbers, then move
    // walker blocks of `block` through every particle (sweep_blocking.h).
    // `psi` must already have been evaluated on `wset`.
    // Returns the number of accepted single-particle moves across the sweep.
    std::size_t sweep(WalkerSet& wset, TrialWavefunction& psi,
                      std::span<const double> sigma, numerics::XoshiroRandom& rng,
                      std::span<double> randoms, std::size_t block);

    // propose → ratio → accept → apply for every particle of walkers
    // [w_begin, w_end), reading their entries of `randoms`.  Langevin
    // dynamics when cfg_.use_drift is true, symmetric Metropolis otherwise.
    std::size_t sweep_walkers(WalkerSet& wset, TrialWavefunction& psi,
                              std::span<const double> sigma,
                              std::span<const double> randoms,
                              std::size_t w_begin, std::size_t w_end);

    // One measured step: n_substeps Metropolis sweeps followed by a
    // local-energy measurement. Returns the accumulated per-walker sums.
    // `measure` = false during equilibration: the measure hook must not see
    // the transient (the optimiser would build H̄/S̄ from it).
    StepResult run_step(WalkerSet& wset, TrialWavefunction& psi,
                        Hamiltonian& hamiltonian, numerics::XoshiroRandom& rng,
                        std::span<double> randoms, std::size_t block,
                        bool measure = true);
};

}  // namespace alea
