// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "qmc/population_control.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace alea {

PopulationControl::PopulationControl(double tau, int n_substeps, double generations,
                                     std::size_t target_walkers, BranchCutoff cutoff,
                                     double alpha, std::size_t n_particles)
    : tau_(tau), cutoff_(cutoff), alpha_(alpha),
      n_particles_(static_cast<double>(n_particles)), target_walkers_(target_walkers) {
    if (cutoff == BranchCutoff::universal && !(std::isfinite(alpha) && alpha > 0.0))
        throw std::invalid_argument(
            "dmc: the universal branch cutoff needs a finite alpha > 0");
    if (n_particles == 0)
        throw std::invalid_argument("dmc: the branch cutoff needs at least one particle");
    const double t_step = tau * static_cast<double>(std::max(1, n_substeps));
    feedback_rate_ = (t_step > 0.0 && generations > 0.0)
                         ? 1.0 / (generations * t_step)
                         : 0.0;
    // drv is fixed by τ alone; universal waits for σ; none never cuts.
    if (cutoff == BranchCutoff::drv && tau > 0.0) set_cutoff(2.0 / std::sqrt(tau));
}

void PopulationControl::set_cutoff(double e_cut) noexcept {
    e_cut_ = e_cut;
    e_max_ = 1.5 * e_cut;
}

void PopulationControl::set_energy_spread(double sigma) noexcept {
    sigma_ = sigma;
    if (cutoff_ != BranchCutoff::universal) return;
    // e_cut = α·√(σ√N/τ).  σ = 0 (every walker at the same E_L, an exact
    // trial) or τ = 0 leaves nothing to cut: e_cut = ∞.
    if (!(std::isfinite(sigma) && sigma > 0.0 && tau_ > 0.0)) {
        set_cutoff(kInf);
        return;
    }
    set_cutoff(alpha_ * std::sqrt(sigma * std::sqrt(n_particles_) / tau_));
}

void PopulationControl::start(double energy) noexcept {
    etrial_    = energy;
    eref_      = energy;
    eref_sum_  = 0.0;
    eref_wsum_ = 0.0;
}

double PopulationControl::branch_weight(double e_new, double e_old) const noexcept {
    double half_tau = 0.5 * tau_;
    const double x = std::max(eref_ - e_new, eref_ - e_old);
    if (x > e_max_)
        half_tau = 0.0;
    else if (x > e_cut_)
        half_tau *= 1.0 - (x - e_cut_) / (e_max_ - e_cut_);
    return std::exp(half_tau * (2.0 * etrial_ - e_new - e_old));
}

double PopulationControl::branch_weight(double e_new, double e_old, double f_new,
                                        double f_old, double tau_eff) const noexcept {
    const double a_new = (eref_ - e_new) * f_new;
    const double a_old = (eref_ - e_old) * f_old;
    double half_tau = 0.5 * tau_eff;
    const double x = std::max(a_new, a_old);
    if (x > e_max_)
        half_tau = 0.0;
    else if (x > e_cut_)
        half_tau *= 1.0 - (x - e_cut_) / (e_max_ - e_cut_);
    return std::exp(half_tau * (2.0 * (etrial_ - eref_) + a_new + a_old));
}

double suppression_factor(double x) noexcept {
    if (x < 1e-4) return 1.0 - x * x / 3.0;   // the x⁴/10 term is below 1e-17
    return 0.5 * std::sqrt(std::numbers::pi) * std::erf(x) / x;
}

double suppression_scale(double kappa, double n_particles, double tau,
                         double correlation_time) noexcept {
    if (!(tau > 0.0)) return 0.0;
    if (!(correlation_time > 0.0)) return std::numeric_limits<double>::infinity();
    return kappa / std::sqrt(n_particles * tau * correlation_time);
}

void PopulationControl::update(double ensemble_energy, double total_weight,
                               bool accumulate) noexcept {
    if (accumulate) {
        eref_sum_  += ensemble_energy * total_weight;
        eref_wsum_ += total_weight;
        if (eref_wsum_ > 0.0) eref_ = eref_sum_ / eref_wsum_;
    } else {
        eref_ = ensemble_energy;
    }
    if (total_weight > 0.0)
        etrial_ = eref_ - feedback_rate_ *
                              std::log(total_weight / static_cast<double>(target_walkers_));
}

void PopulationControl::begin_accumulation() noexcept {
    eref_sum_  = 0.0;
    eref_wsum_ = 0.0;
}

const char* to_string(BranchCutoff cutoff) noexcept {
    switch (cutoff) {
        case BranchCutoff::universal: return "universal";
        case BranchCutoff::drv:       return "drv";
        case BranchCutoff::none:      break;
    }
    return "none";
}

const char* to_string(Reweighting reweighting) noexcept {
    switch (reweighting) {
        case Reweighting::effective: return "effective";
        case Reweighting::suppressed: return "suppressed";
        case Reweighting::bare: return "bare";
    }
    return "bare";
}

}  // namespace alea
