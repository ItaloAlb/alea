// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// One-body density matrix n(r) and momentum distribution n(k) of one particle
// group, measured on the open paths of a PIMC worm (Boninsegni, Prokof'ev &
// Svistunov, Phys. Rev. E 74, 036701 (2006); PIMC only).
//
// At an open measurement the driver passes, per extra-bead sample, the
// displacement x - t (head extended by one free link to x, tail t; minimum
// image) and c = s w / (n C N_w M) (WormMove::sample_open_ends); at a closed
// measurement the path sign s.  With E over all measurement points and N_w
// the worm's particles,
//   n(r)  = (1/N_w) ∫ ρ₁(R + r, R) dR = E[Σ c δ(x - t - r)] / E[[closed] s],
//   ñ(k)  = ∫ n(r) e^{-ik·r} dr      = E[Σ c A_d(k, x - t)] / E[[closed] s],
// so n(0) = 1, with A_d the angular average of e^{-ik·r} (cos kr, J₀(kr),
// sin(kr)/kr for d = 1, 2, 3) on a uniform k grid in an open cell.  Under PBC
// the occupation per state ⟨n_k⟩ = (N_w / V) ñ(k), averaged over each shell of
// reciprocal-lattice vectors with |k| <= kmax (k = 0 included; Σ_k ⟨n_k⟩ = N_w).
// The finite-cutoff number and kinetic sums are blocked jointly as kernels
// Σ_k cos(k.r) and Σ_k [Σ_d k_d²/(2 m_d)] cos(k.r). Their ratio errors retain
// correlations between shells (and the shared closed-sector denominator).
// Missing reciprocal tails are excluded from those statistical errors.
// In an open cell the angular Fourier grid is integrated with radial
// trapezoidal weights Ω_d k^(d-1) dk/(2π)^d. The kinetic kernel also carries
// the directional inverse mass; it is not inferred from angular n(k) alone
// for anisotropic masses. Cutoff and quadrature errors require refinements.
// Every value is a ratio of means with a jackknife error (signed reblocking).
// n(r) is binned radially on [0, rmax) (rmax <= the Wigner-Seitz inradius
// under PBC).
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "estimators/estimator.h"
#include "population/particle_group.h"
#include "population/simulation_cell.h"

namespace alea {

class MomentumDistribution final : public Estimator {
public:
    // n_k: number of k points of the open-cell grid k_j = j kmax / (n_k - 1);
    // must be 0 (unused) under PBC.  samples: extra beads per open measurement.
    MomentumDistribution(std::span<const ParticleGroup> groups, const std::string& group,
                         double rmax, int n_bins, double kmax, int n_k, int samples,
                         const SimulationCell& cell, std::size_t dim);

    void start_block() override;
    // Closed measurement: counts it and its sign s (the slice weights).
    void accumulate(const ParticleBatch& configs, std::span<const double> weights) override;
    void stop_block() override;
    void collect(const Estimator& other) override;
    void finalize(RunResult& out) const override;
    void configure_reblocking(std::size_t num_blocks) override;
    [[nodiscard]] std::size_t reblocking_storage_bytes(std::size_t num_blocks) const override;
    void configure_signed_weights() override { reblocking_.configure_signed(); }
    // An open path of another worm: counted, not binned.
    void skip_measurement(std::size_t) override;
    [[nodiscard]] std::unique_ptr<Estimator> clone() const override {
        return std::make_unique<MomentumDistribution>(*this);
    }
    [[nodiscard]] std::string type() const override { return "momentum_distribution"; }
    // obdm_<group>_* (n(r)) and nk_<group>_* (n(k)).
    [[nodiscard]] std::vector<std::string> output_namespaces() const override {
        return {"obdm_" + group_, "nk_" + group_};
    }

    [[nodiscard]] std::string open_path_group() const override { return group_; }
    [[nodiscard]] std::size_t open_path_samples() const noexcept override { return samples_; }
    void configure_open_paths(std::size_t worm_particles) override;
    void accumulate_open(std::span<const double> displacement,
                         std::span<const double> weight) override;

private:
    std::string group_;
    double rmax_, delta_;
    std::size_t n_bins_, dim_, samples_;
    SimulationCell cell_;
    bool periodic_;
    double worm_particles_ = 0.0;
    std::vector<double> k_;             // [n_shells]: shell |k| (PBC) or grid k_j (open)
    std::vector<double> kvec_;          // PBC: [n_kvec * dim] reciprocal-lattice vectors
    std::vector<std::size_t> shell_of_; // PBC: [n_kvec] -> shell
    std::vector<double> shell_count_;   // PBC: [n_shells] vectors per shell
    std::vector<double> k_energy_;      // PBC: [n_kvec] sum_d k_d²/(2 m_d)
    std::vector<double> inverse_mass_;  // [dimension]: group inverse masses
    std::vector<double> radial_weights_;// open: [n_k] Ω_d k^(d-1) dk/(2π)^d
    bool isotropic_mass_ = true;
    double trace_inverse_mass_ = 0.0;
    std::vector<double> block_;         // [n_bins + n_shells + 2] kernels, number/kinetic sums
    double block_sign_ = 0.0;           // Σ s over closed measurements
    double block_count_ = 0.0;          // measurement points
    EstimatorReblocking reblocking_;
};

}  // namespace alea
