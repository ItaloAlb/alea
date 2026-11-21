// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Takahashi–Imada potential action; the formulas are in the header.
#include "pimc/takahashi_imada_action.h"
#include "pimc/path_set.h"
#include "pimc/primitive_action.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace alea {
TakahashiImadaPotentialAction::TakahashiImadaPotentialAction(
    std::span<const ParticleGroup> groups, const Hamiltonian& ham, std::size_t dim)
    : ham_(ham.clone()), dim_(dim) {
    if (dim == 0 || dim > SimulationCell::kMaxDim)
        throw std::invalid_argument("takahashi_imada: invalid spatial dimension");
    refuse_unnormalizable_pairs(groups, ham_, "takahashi_imada");
    for (const auto& component : ham_.components())
        if (component->is_potential() && !component->supports_configuration_gradient())
            throw std::invalid_argument(
                "takahashi_imada: the potential term " + component->name() +
                " has no configuration gradient, which the |grad V|^2 term of the action needs");
    for (const auto& g : groups) np_ = std::max(np_, g.last);
    inv_mass_.assign(np_ * dim_, 0.0);
    for (const auto& g : groups) {
        if (g.inv_mass.size() != 1 && g.inv_mass.size() != dim_)
            throw std::invalid_argument("takahashi_imada: mass must be scalar or dimension-sized");
        for (std::size_t p = g.first; p < g.last; ++p)
            for (std::size_t d = 0; d < dim_; ++d) {
                const double inv_m = g.inv_mass[g.inv_mass.size() == 1 ? 0 : d];
                if (!std::isfinite(inv_m) || inv_m <= 0.0)
                    throw std::invalid_argument("takahashi_imada: masses must be finite and positive");
                inv_mass_[p * dim_ + d] = inv_m;
            }
    }
    for (double inv_m : inv_mass_)
        if (inv_m == 0.0)
            throw std::invalid_argument("takahashi_imada: uncovered particle mass");
}

TakahashiImadaPotentialAction::TakahashiImadaPotentialAction(
    const TakahashiImadaPotentialAction& other)
    : ham_(other.ham_.clone()), dim_(other.dim_), np_(other.np_), inv_mass_(other.inv_mass_),
      scratch_(other.scratch_), force_rows_(other.force_rows_),
      bead_rows_old_(other.bead_rows_old_), bead_rows_new_(other.bead_rows_new_) {}

std::unique_ptr<ActionComponent> TakahashiImadaPotentialAction::clone() const {
    return std::make_unique<TakahashiImadaPotentialAction>(*this);
}

void TakahashiImadaPotentialAction::prepare(std::size_t M) {
    if (scratch_.size() != M) scratch_.resize(M);
}

void TakahashiImadaPotentialAction::size_scratch(const PathSet& path) const {
    const std::size_t size = path.n_particles() * dim_;
    if (force_rows_.size() < size) {
        force_rows_.resize(size);
        bead_rows_old_.resize(size);
        bead_rows_new_.resize(size);
    }
}

double TakahashiImadaPotentialAction::rows_correction(const PathSet& path, std::size_t k,
    std::size_t p, std::size_t n_rows, std::size_t skip) const {
    size_scratch(path);
    ham_.configuration_gradient(path, k, p, n_rows, skip, force_rows_);
    const std::size_t rows = std::min(n_rows, np_);
    double sum = 0.0;
    for (std::size_t j = 0; j < rows; ++j) {
        const double* f = force_rows_.data() + j * dim_;
        sum += weighted_dot(j, f, f);
    }
    return sum / 24.0;
}

double TakahashiImadaPotentialAction::slice_correction(const PathSet& path, std::size_t k) const {
    return rows_correction(path, k, HamiltonianComponent::npos, np_, HamiltonianComponent::npos);
}

double TakahashiImadaPotentialAction::potential_action_slice(const PathSet& path,
                                                             std::size_t k) {
    prepare(path.num_slices());
    ham_.evaluate_potential(path, scratch_, k, k + 1);
    const double tau = path.tau();
    return tau * scratch_[k] + tau * tau * tau * slice_correction(path, k);
}

double TakahashiImadaPotentialAction::total_potential(const PathSet& path) {
    prepare(path.num_slices());
    ham_.evaluate_potential(path, scratch_);
    double result = 0.0;
    for (double v : scratch_) result += v;
    return result;
}

double TakahashiImadaPotentialAction::total_action(const PathSet& path) {
    const double tau = path.tau();
    const double potential = total_potential(path);
    double correction = 0.0;
    for (std::size_t k = 0; k < path.num_slices(); ++k) correction += slice_correction(path, k);
    return tau * potential + tau * tau * tau * correction;
}

// d_beta S_V at fixed M: tau = beta/M, so d_beta (tau V + tau^3 W) = (V + 3 tau^2 W)/M.
double TakahashiImadaPotentialAction::beta_derivative(const PathSet& path) {
    return measure_thermo(path).energy;
}

ActionThermoMeasure TakahashiImadaPotentialAction::measure_thermo(const PathSet& path) {
    const double M = static_cast<double>(path.num_slices()), tau2 = path.tau() * path.tau();
    const double potential = total_potential(path);
    double correction = 0.0;
    for (std::size_t k = 0; k < path.num_slices(); ++k) correction += slice_correction(path, k);
    // Hellmann–Feynman split (header): V gets 2 tau^2 W, K gets tau^2 W.
    return {tau2 * correction / M, (potential + 2.0 * tau2 * correction) / M,
            (potential + 3.0 * tau2 * correction) / M};
}

void TakahashiImadaPotentialAction::validate_open_centroid(std::size_t, std::size_t) const {
    throw std::invalid_argument(
        "centroid_virial: the Takahashi-Imada action has no virial estimator (it would need "
        "the Hessian of V); use energy: thermodynamic");
}

void TakahashiImadaPotentialAction::validate_periodic_virial(const SimulationCell&, std::size_t,
                                                             std::size_t) const {
    throw std::invalid_argument(
        "periodic_virial: the Takahashi-Imada action has no virial estimator (it would need "
        "the Hessian of V); use energy: thermodynamic");
}

std::string TakahashiImadaPotentialAction::potential_approximation_notes() const {
    std::string result =
        "takahashi_imada: O(tau^4) for Z and the thermodynamic energy; observables at single "
        "slices (density, S(k), Bragg) keep O(tau^2)";
    for (const auto& component : ham_.components()) {
        if (!component->is_potential()) continue;
        const auto notes = component->potential_approximation_notes();
        if (!notes.empty()) result += "; " + component->name() + ": " + notes;
    }
    return result;
}

// tau dV + tau^3 dW for bead (p, k) moving to `proposed`.  With F_j the
// forces of the other rows R (p left out) and b_j, s the changes a bead at
// the old (o) or new (n) position makes to them and its own force,
//   24 dW = sum_{j in R} [(F + b^n)^2 - (F + b^o)^2]_j / m_j + (s^n)^2/m_p - (s^o)^2/m_p
//         = sum_{j in R} (2 F + b^n + b^o)_j . (b^n - b^o)_j / m_j + ...,
// the second form without the cancellation of two large squares.
double TakahashiImadaPotentialAction::delta_potential_particle(PathSet& path, std::size_t p,
    std::size_t k, std::span<const double> proposed) {
    prepare(path.num_slices());
    const double tau = path.tau();
    const double delta_v = ham_.delta_potential_particle(path, p, k, proposed, scratch_);
    if (!std::isfinite(delta_v)) return tau * delta_v;
    size_scratch(path);
    const auto none = HamiltonianComponent::npos;
    ham_.configuration_gradient(path, k, p, np_, none, force_rows_);
    double old_position[SimulationCell::kMaxDim], self_old[SimulationCell::kMaxDim],
        self_new[SimulationCell::kMaxDim];
    for (std::size_t d = 0; d < dim_; ++d) old_position[d] = path.pos(p, d, k);
    ham_.bead_gradient(path, k, p, std::span<const double>(old_position, dim_), np_, none,
                       std::span<double>(self_old, dim_), bead_rows_old_);
    ham_.bead_gradient(path, k, p, proposed, np_, none, std::span<double>(self_new, dim_),
                       bead_rows_new_);
    double sum = weighted_dot(p, self_new, self_new) - weighted_dot(p, self_old, self_old);
    for (std::size_t j = 0; j < np_; ++j) {
        if (j == p) continue;
        double plus[SimulationCell::kMaxDim], minus[SimulationCell::kMaxDim];
        for (std::size_t d = 0; d < dim_; ++d) {
            const std::size_t i = j * dim_ + d;
            plus[d] = 2.0 * force_rows_[i] + bead_rows_new_[i] + bead_rows_old_[i];
            minus[d] = bead_rows_new_[i] - bead_rows_old_[i];
        }
        sum += weighted_dot(j, plus, minus);
    }
    return tau * delta_v + tau * tau * tau * sum / 24.0;
}

// tau e + tau^3 [W(R + r) - W(R)], with 24 [W(R + r) - W(R)] =
// sum_{j in R} (2 F + b)_j . b_j / m_j + s^2 / m_p (notation as above).
double TakahashiImadaPotentialAction::bead_action(const PathSet& path, std::size_t k,
    std::size_t p, std::span<const double> position, std::size_t n_rows,
    std::size_t skip) const {
    if (p >= np_)
        throw std::logic_error("takahashi_imada: a bead row outside the action's particles");
    const double tau = path.tau();
    const double energy = ham_.bead_energy(path, k, p, position, n_rows, skip);
    if (!std::isfinite(energy)) return tau * energy;
    size_scratch(path);
    ham_.configuration_gradient(path, k, p, n_rows, skip, force_rows_);
    double self[SimulationCell::kMaxDim];
    ham_.bead_gradient(path, k, p, position, n_rows, skip, std::span<double>(self, dim_),
                       bead_rows_new_);
    double sum = weighted_dot(p, self, self);
    const std::size_t rows = std::min(n_rows, np_);
    for (std::size_t j = 0; j < rows; ++j) {
        if (!counted_row(j, p, n_rows, skip)) continue;
        double plus[SimulationCell::kMaxDim];
        for (std::size_t d = 0; d < dim_; ++d)
            plus[d] = 2.0 * force_rows_[j * dim_ + d] + bead_rows_new_[j * dim_ + d];
        sum += weighted_dot(j, plus, bead_rows_new_.data() + j * dim_);
    }
    return tau * energy + tau * tau * tau * sum / 24.0;
}
} // namespace alea
