// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Primitive approximation to the imaginary-time density matrix, split into two
// independent components (Ceperley, Rev. Mod. Phys. 67, 279 (1995), Sec. II),
// with hbar = 1, tau = beta/M and slice M identified with slice 0:
//
//   S   = S_K + S_V
//   S_K = sum_k sum_p sum_d m_pd (r_pd^{k+1} - r_pd^k)^2 / (2 tau)
//         (minimum-image links under PBC; the free normalization
//          (d N M / 2) ln(2 pi tau / m) is kept out of S)
//   S_V = tau sum_k V(R_k)
//
// The thermodynamic estimator is -d_beta ln Z at fixed M (Barker, J. Chem.
// Phys. 70, 2914 (1979)):
//
//   E_T = d N / (2 tau) - <S_K> / beta + < (1/M) sum_k V(R_k) >,
//
// the first two terms from PrimitiveKineticAction::measure_thermo, the last
// from PrimitivePotentialAction::measure_thermo.  Its systematic error is
// O(tau^2); its variance grows with M, which the centroid and periodic virial
// estimators avoid.  Definitions share action.cpp.
#include <vector>
#include "pimc/action_component.h"
#include "hamiltonian/hamiltonian.h"
#include "population/particle_group.h"

namespace alea {
class PrimitiveKineticAction final : public ActionComponent {
public:
    PrimitiveKineticAction(std::span<const ParticleGroup> groups, std::size_t dim);
    double total_action(const PathSet& path) override;
    double normalization_energy(const PathSet& path) const override;
    double beta_derivative(const PathSet& path) override;
    ActionThermoMeasure measure_thermo(const PathSet& path) override;
    void validate_open_centroid(std::size_t np, std::size_t dim) const override;
    ActionThermoMeasure measure_open_centroid(const PathSet&, std::span<const double>) override;
    void validate_periodic_virial(const SimulationCell&, std::size_t, std::size_t) const override;
    ActionThermoMeasure measure_periodic_virial(const PathSet&, std::span<const double>, double) override;
    double spring_action(const PathSet& path, std::size_t p1, std::size_t k1,
                         std::size_t p2, std::size_t k2) const override;
    // Kinetic differences are handled by spring/proposal-density terms.
    double delta_potential_particle(PathSet&, std::size_t, std::size_t,
                                    std::span<const double>) override { return 0.0; }
    bool supports_free_bridge() const noexcept override { return true; }
    bool slice_local_deltas() const noexcept override { return true; }
    bool provides_free_kinetic() const noexcept override { return true; }
    double mass(std::size_t p, std::size_t d) const override {
        return mass_[p * dim_ + d];
    }
    std::unique_ptr<ActionComponent> clone() const override;
private:
    std::vector<double> mass_;
    std::size_t dim_ = 0, np_ = 0;
};

class PrimitivePotentialAction final : public ActionComponent {
public:
    PrimitivePotentialAction(std::span<const ParticleGroup> groups,
                             const Hamiltonian& ham);
    PrimitivePotentialAction(const PrimitivePotentialAction& other);
    void prepare(std::size_t M) override;
    double total_action(const PathSet& path) override;
    double normalization_energy(const PathSet& path) const override;
    double beta_derivative(const PathSet& path) override;
    ActionThermoMeasure measure_thermo(const PathSet& path) override;
    void validate_open_centroid(std::size_t np, std::size_t dim) const override;
    bool confines_open_coordinate(std::size_t p, std::size_t d,
                                  std::size_t dim) const noexcept override;
    void prepare_open_centroid(const PathSet&) override;
    ActionThermoMeasure measure_open_centroid(const PathSet&, std::span<const double>) override;
    void validate_periodic_virial(const SimulationCell&, std::size_t, std::size_t) const override;
    void prepare_periodic_virial(const PathSet&) override;
    ActionThermoMeasure measure_periodic_virial(const PathSet&, std::span<const double>, double) override;
    std::string potential_approximation_notes() const override;
    double potential_action_slice(const PathSet& path, std::size_t k) override;
    double total_potential(const PathSet& path) override;
    double delta_potential_particle(PathSet& path, std::size_t p, std::size_t k,
                                    std::span<const double> proposed) override;
    bool supports_free_bridge() const noexcept override { return true; }
    bool slice_local_deltas() const noexcept override { return true; }
    std::unique_ptr<ActionComponent> clone() const override;
private:
    Hamiltonian ham_;
    std::vector<double> scratch_;
    std::vector<double> gradient_; // [np*dim*M], distinct from wavefunction derivatives
    std::unique_ptr<HamiltonianGradientWorkspace> gradient_workspace_;
};
} // namespace alea
