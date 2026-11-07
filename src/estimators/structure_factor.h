// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "estimators/estimator.h"
#include "numerics/reciprocal_phases.h"
#include "population/particle_group.h"
#include "population/simulation_cell.h"

namespace alea {

struct RunResult;

// Static structure factor of a periodic cell, for every group pair:
//
//     S_ab(k) = ⟨ Re[ρ_a(k) ρ_b(k)*] ⟩ / √(N_a N_b),   ρ_g(k) = Σ_{i∈g} e^{ik·r_i},
//
// on every reciprocal-lattice vector 0 < |k| ≤ kmax (half space: S(−k) = S(k)),
// averaged over each |k| shell.  S_aa → 1 at large k (each particle with
// itself); for an ideal gas S_aa = 1 at every k ≠ 0.
//
// It is the FULL structure factor.  For a crystal, the Bragg (disconnected)
// part |⟨ρ_g(k)⟩|² must be subtracted to get the fluctuations, so ⟨ρ_g(k)⟩ is
// also recorded per vector (it is not rotation-averaged: a crystal's peaks sit
// on particular vectors of a shell).
//
// Why it is here: the small-k behaviour of S(k) (∝ k^{3/2} for 2-D Coulomb,
// fixed by the plasmon) is the input of the Chiesa finite-size correction
// (Chiesa, Ceperley, Martin & Holzmann, PRL 97, 076404 (2006)); and Bragg
// peaks are a direct crystal/fluid diagnostic.
//
// Sampling: weighted by the walker weights, so VMC gives ⟨ψ|·|ψ⟩ and DMC the
// mixed estimator ⟨ψ|·|Φ⟩ (extrapolate as 2·DMC − VMC), like g(r).
//
// Output series (one block-averaged value per entry, with `_error`):
//   sofk_k, sofk_shell_count             |k| and the number of half-space
//                                        vectors in each shell
//   sofk_<A>_<B>                         S_ab per shell (A ≤ B in group order)
//   sofk_kx, sofk_ky, sofk_shell         every vector and its shell index
//   rhok_<G>_re, rhok_<G>_im             ⟨ρ_g(k)⟩ per vector
//
// Cost per measurement: O(n_w·N·n_k), the cost of building ρ(k).
class StructureFactor : public Estimator {
public:
    // Throws if the cell is not periodic and 2-D, or no vector has |k| ≤ kmax.
    StructureFactor(std::span<const ParticleGroup> groups, const SimulationCell& cell,
                    double kmax);

    void configure_reblocking(std::size_t num_blocks) override;
    void configure_signed_weights() override { reblocking_.configure_signed(); }
    // Worm runs: an open path is a measurement of n zero weights; it counts
    // in the signed normalization without being binned.
    void skip_measurement(std::size_t n) override {
        if (!reblocking_.signed_mode())
            throw std::logic_error("sofk: skip_measurement needs signed weights");
        acc_abs_w_ += static_cast<double>(n);
    }
    [[nodiscard]] std::size_t reblocking_storage_bytes(std::size_t num_blocks) const override;
    void start_block() override;
    using Estimator::accumulate;
    void accumulate(const ParticleBatch& configs,
                    std::span<const double> weights) override;
    void stop_block() override;
    void collect(const Estimator& other) override;
    void finalize(RunResult& out) const override;

    [[nodiscard]] std::unique_ptr<Estimator> clone() const override {
        return std::make_unique<StructureFactor>(*this);
    }
    [[nodiscard]] std::string type() const override { return "sofk"; }
    // sofk_k, sofk_<a>_<b>[_error], ... and the per-group ρ_k planes rhok_<g>_re/_im.
    [[nodiscard]] std::vector<std::string> output_namespaces() const override {
        return {"sofk", "rhok"};
    }

private:
    struct Group {
        std::string name;
        std::size_t first, last;
    };
    [[nodiscard]] std::size_t n_pairs() const noexcept {
        return groups_.size() * (groups_.size() + 1) / 2;
    }

    std::vector<Group> groups_;
    std::size_t dim_ = 2;

    // Half-space vectors, their shell, and the shells.
    std::vector<double> kx_, ky_;
    std::vector<std::size_t> shell_of_;
    std::vector<double> shell_k_;
    std::vector<std::size_t> shell_count_;
    numerics::ReciprocalPhases phases_;

    // Current block: Σ_w weight·Re ρ_aρ_b* per (pair, k), Σ_w weight·ρ_g per
    // (group, k), and Σ_w weight.
    std::vector<double> acc_s_, acc_re_, acc_im_;
    double acc_w_ = 0.0;
    double acc_abs_w_ = 0.0;  // signed normalization: number of weights
    // Scratch: ρ_g of one walker, [g·n_k + k], and one particle's phases.
    std::vector<double> rho_re_, rho_im_, c_, s_;

    // Finished blocks: S per (pair, shell) and ⟨ρ⟩ per (group, k).
    std::vector<std::vector<double>> hist_s_, hist_re_, hist_im_;
    EstimatorReblocking reblocking_;
};

}  // namespace alea
