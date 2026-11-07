// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// rho(R,R';tau) ~= rho_free(R,R';tau) product_{i<j} exp(-u_ij).
// u_ij=-log(rho_pair/rho_relative_free), using the finite-grid backend.
// Pollock & Ceperley, PRB 30, 2555 (1984), Appendix A. The pair product
// omits irreducible three-body short-time terms. Numerical grid and domain
// approximations must be refined separately from the many-body time step.
#include "pimc/action_component.h"
#include "hamiltonian/hamiltonian.h"
#include "numerics/finite_grid_pair.h"
namespace alea {
class PairProductAction final : public ActionComponent {
public:
    PairProductAction(std::span<const ParticleGroup>, const Hamiltonian&,
                      std::size_t dim, double extent, int grid_points);
    void validate_sampling(const SimulationCell&, double,
                           std::size_t) const override;
    void prepare_time_step(double tau) override;
    double total_action(const PathSet&) override;
    double potential_action_slice(const PathSet&, std::size_t) override;
    double total_potential(const PathSet&) override;
    double beta_derivative(const PathSet&) override;
    ActionThermoMeasure measure_thermo(const PathSet&) override;
    // One bead (count = 1) or a staged cyclic window of particle p: the change
    // of u over the min(count+1, M) links touching the window, for every pair
    // containing p — O(N * count) table lookups, never the whole path.
    double delta_potential_particle(PathSet&, std::size_t, std::size_t,
                                    std::span<const double>) override;
    bool link_local_window_deltas() const noexcept override { return true; }
    double delta_potential_window(const PathSet&, std::size_t p,
                                  std::size_t first,
                                  std::span<const double> positions) override;
    bool supports_free_bridge() const noexcept override { return true; }
    bool slice_local_deltas() const noexcept override { return false; }
    std::string potential_approximation_notes() const override;
    std::unique_ptr<ActionComponent> clone() const override {
        return std::make_unique<PairProductAction>(*this);
    }

private:
    struct Channel {
        ParticleGroup a, b;
        double mu = 0, pole = 0;
        // [potential contribution]; called at setup and by the bead-potential
        // observable, never inside a move.
        std::vector<std::function<double(double)>> radial;
        // Immutable tables (up to ~38 MB per channel at 33 points) shared
        // read-only by every replica clone instead of copied per replica;
        // const shared ownership is the deliberate exception to unique_ptr.
        std::shared_ptr<const numerics::FiniteGridPair> table;
    };
    void check(const PathSet&) const;
    double link(const PathSet&, const Channel&, std::size_t, std::size_t,
                std::size_t, bool thermal) const;
    std::vector<Channel> channels_; // stable unordered group pairs
    double extent_ = 0, tau_ = 0;
    int points_ = 0;
    std::size_t np_ = 0;
    std::string source_notes_;
};
} // namespace alea
