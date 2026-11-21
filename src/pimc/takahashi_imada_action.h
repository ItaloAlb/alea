// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Takahashi–Imada potential action (M. Takahashi and M. Imada, J. Phys. Soc.
// Jpn. 53, 3765 (1984)), with hbar = 1 and tau = beta/M.  The trace of the
// symmetric product
//
//   Z_M = Tr [ e^{-tau T} e^{-tau V - (tau^3/24) [V, [T, V]]} ]^M,
//   [V, [T, V]] = sum_{j,d} (1/m_{jd}) (d V / d r_{jd})^2,
//
// agrees with Tr e^{-beta H} to O(tau^4) (the primitive product, without the
// double commutator, to O(tau^2)).  Next to PrimitiveKineticAction it
// replaces PrimitivePotentialAction by
//
//   S_V = sum_k [ tau V(R_k) + tau^3 W(R_k) ],
//   W(R) = (1/24) sum_{j,d} (1/m_{jd}) (d V(R) / d r_{jd})^2,
//
// i.e. V_TI = V + (tau^2 / 24) sum (1/m) |grad V|^2 (also Brualla, Sakkos,
// Boronat and Casulleras, J. Chem. Phys. 121, 636 (2004), Sec. II).  V is the
// TOTAL potential: |grad_j V|^2 squares the sum of every term's force on j,
// so it holds cross terms (external x pair) and three-body terms
// (F_ij . F_ik).  The action is still slice-local; moving one bead changes
// the force on every particle it interacts with, so a bead delta costs
// O(N^2) here (the forces of the other rows are recomputed).
//
// Energy (thermodynamic, -d_beta ln Z_M at fixed M, the primitive kinetic
// part unchanged):  d_beta S_V = (1/M) sum_k [V_k + 3 tau^2 W_k].  The split
// into kinetic and potential follows Hellmann–Feynman in a coupling g of V
// (H = T + g V, so S_V(g) = sum_k [tau g V_k + tau^3 g^2 W_k]):
//   <V> = (1/beta) <d_g S_V> = < (1/M) sum_k [V_k + 2 tau^2 W_k] >,
//   K   = primitive kinetic + < (1/M) sum_k tau^2 W_k >.
// Both are exact derivatives of ln Z_M, hence also O(tau^4).
//
// APPROXIMATIONS (reported in the run's notes): observables measured at
// single slices (density, S(k), Bragg intensities, pair correlations) keep an
// O(tau^2) error, since the TI product is fourth order only under the trace;
// the virial energy estimators would need the Hessian of V and are refused.
// Pair forces are taken at the minimum image, so in a periodic cell they
// jump where a pair crosses a Wigner–Seitz face; the order in tau must then
// be checked by a time-step scan.
//
// Every potential term must offer HamiltonianComponent::configuration_gradient
// and bead_gradient; a grand-canonical run also needs bead_energy.
#include <vector>
#include "pimc/action_component.h"
#include "hamiltonian/hamiltonian.h"
#include "population/particle_group.h"

namespace alea {
class TakahashiImadaPotentialAction final : public ActionComponent {
public:
    // Throws invalid_argument when a potential term has no configuration
    // gradient, on the attractive singularities PrimitivePotentialAction
    // refuses, and on masses that are not finite and positive.
    TakahashiImadaPotentialAction(std::span<const ParticleGroup> groups,
                                  const Hamiltonian& ham, std::size_t dim);
    TakahashiImadaPotentialAction(const TakahashiImadaPotentialAction& other);

    void prepare(std::size_t M) override;
    double total_action(const PathSet& path) override;
    double normalization_energy(const PathSet&) const override { return 0.0; }
    double beta_derivative(const PathSet& path) override;
    ActionThermoMeasure measure_thermo(const PathSet& path) override;
    void validate_open_centroid(std::size_t, std::size_t) const override;
    void validate_periodic_virial(const SimulationCell&, std::size_t, std::size_t) const override;
    std::string potential_approximation_notes() const override;
    double potential_action_slice(const PathSet& path, std::size_t k) override;
    // sum_k V(R_k): the plain potential, for reports.
    double total_potential(const PathSet& path) override;
    double delta_potential_particle(PathSet& path, std::size_t p, std::size_t k,
                                    std::span<const double> proposed) override;
    bool supports_free_bridge() const noexcept override { return true; }
    bool slice_local_deltas() const noexcept override { return true; }
    // tau e(r) + tau^3 [W(R + r) - W(R)], e the Hamiltonian's bead energy.
    bool supports_bead_action() const noexcept override { return ham_.supports_bead_energy(); }
    double bead_action(const PathSet& path, std::size_t k, std::size_t p,
                       std::span<const double> position, std::size_t n_rows,
                       std::size_t skip) const override;
    std::unique_ptr<ActionComponent> clone() const override;

    // W(R_k) of slice k, R_k every row of this action (n_particles of its
    // groups); also W of a configuration with rows left out, as bead_action
    // sees it.  Public for tests.
    [[nodiscard]] double slice_correction(const PathSet& path, std::size_t k) const;

private:
    Hamiltonian ham_;
    std::size_t dim_ = 0, np_ = 0;
    std::vector<double> inv_mass_; // [np*dim], 1/m of each particle and axis
    std::vector<double> scratch_;  // [M], V of each slice
    // Per-call force scratch [rows of the path * dim]: the forces of the
    // other rows, and a bead's change of them (old and new position).  Sized
    // on first use by the path (a grand-canonical sector action acts on the
    // first N rows of a larger path), never in the steady state.  Mutable
    // because bead_action is const; each replica owns its clone, so no two
    // threads share them.
    mutable std::vector<double> force_rows_;    // [np*dim]
    mutable std::vector<double> bead_rows_old_; // [np*dim]
    mutable std::vector<double> bead_rows_new_; // [np*dim]

    // W of the rows of R = [0, n_rows) \ {p, skip} at slice k, the forces of R
    // left in force_rows_.
    double rows_correction(const PathSet& path, std::size_t k, std::size_t p,
                           std::size_t n_rows, std::size_t skip) const;
    void size_scratch(const PathSet& path) const;
    // sum_d (1/m_{jd}) x_d y_d for row j.
    [[nodiscard]] double weighted_dot(std::size_t j, const double* x, const double* y) const noexcept {
        double s = 0.0;
        for (std::size_t d = 0; d < dim_; ++d) s += inv_mass_[j * dim_ + d] * x[d] * y[d];
        return s;
    }
};
} // namespace alea
