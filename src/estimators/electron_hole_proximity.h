// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Opt-in diagonal proximity observable for electron-hole configurations:
// U_e = <(1/N_e) sum_e 1[min_{h in the named hole group} r_eh >= r_cut]>.
// C_e = 1 - U_e is a complement of local proximity, NOT a bound-exciton
// fraction.  A function of the configuration R only, so every sampler
// measures it: VMC walkers give <psi_T|.|psi_T>, DMC walkers (weighted) the
// mixed estimate <psi_T|.|Phi>/<psi_T|Phi>, PIMC's imaginary-time slices the
// thermal average at 1/beta.  Replica-synchronous reblocking (PIMC) follows
// Flyvbjerg and Petersen, J. Chem. Phys. 91, 461 (1989); VMC/DMC keep
// constant-space, independence-assuming block moments.
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "estimators/estimator.h"
#include "population/particle_group.h"
#include "population/simulation_cell.h"

namespace alea {

// A configuration of M slices gives M electron counts; no electron-to-hole
// pairing is inferred. PBC distances are provided by the minimum-image table.
class ElectronHoleProximity final : public Estimator {
public:
    // Select distinct, nonempty named negative-electron/positive-hole groups;
    // r_cut is a strictly positive finite length, at most the periodic cell's
    // Wigner-Seitz inradius (no upper bound on an open cell). Invalid input throws.
    ElectronHoleProximity(std::span<const ParticleGroup> groups,
                          std::string electron_group, std::string hole_group,
                          double r_cut, const SimulationCell& cell = SimulationCell{});

    // Bound each replica to at most num_blocks * two double-valued estimates;
    // call once, before accumulating, for correlation-corrected PIMC SEMs.
    void configure_reblocking(std::size_t num_blocks) override;
    void configure_signed_weights() override { reblocking_.configure_signed(); }
    // Worm runs: an open path is a measurement of n zero weights; it counts
    // in the signed normalization without being binned.
    void skip_measurement(std::size_t n) override {
        if (!reblocking_.signed_mode())
            throw std::logic_error("electron_hole_proximity: skip_measurement needs signed weights");
        block_config_abs_weight_ += static_cast<double>(n);
    }
    [[nodiscard]] std::size_t reblocking_storage_bytes(
        std::size_t num_blocks) const override;

    // Start the next block and clear only its counters.
    void start_block() override;
    using Estimator::accumulate;
    // Ingest weighted independent configurations (PIMC: slices) of one replica;
    // weights must match num_walkers() and be finite, nonnegative doubles.
    // The table must represent the same boundary and particle groups. No heap
    // allocations take place in the measurement loop.
    void accumulate(const ParticleBatch& configs,
                    std::span<const double> weights) override;
    // Record the normalized local absence and its complement for this block;
    // the block must have positive total weight.
    void stop_block() override;
    // Indexwise reduction of equal-length, independent replica histories;
    // mismatched parameters, modes or unfinished histories throw.
    void collect(const Estimator& other) override;
    // Append values, SEMs, reblocking diagnostics and physical provenance to
    // the method-agnostic result. Incomplete configured histories throw.
    void finalize(RunResult& out) const override;

    // Copy independent block and reblocking state for thread-local estimators.
    [[nodiscard]] std::unique_ptr<Estimator> clone() const override;
    [[nodiscard]] std::string type() const override {
        return "electron_hole_proximity";
    }
    // electron_hole_proximity_<e>_<h>_{U_e,C_e,r_cut,...}.
    [[nodiscard]] std::vector<std::string> output_namespaces() const override {
        return {output_prefix_};
    }

private:
    // Validate a measurement batch's shape and cell before indexing any row.
    void validate_batch(const ParticleBatch& configs,
                        std::span<const double> weights) const;
    // Verify a collector and its input use the exact same physical definition.
    void validate_compatible(const ElectronHoleProximity& other) const;

    std::string electron_group_;
    std::string hole_group_;
    std::string output_prefix_;
    std::size_t electron_first_ = 0;
    std::size_t electron_last_ = 0;
    std::size_t hole_first_ = 0;
    std::size_t hole_last_ = 0;
    std::size_t required_particles_ = 0;
    double r_cut_ = 0.0;
    SimulationCell cell_;

    double block_free_weight_ = 0.0;
    double block_config_weight_ = 0.0;
    double block_config_abs_weight_ = 0.0;  // signed normalization: number of weights
    std::size_t n_blocks_ = 0;
    bool block_open_ = false;
    double mean_u_ = 0.0;
    double m2_u_ = 0.0;
    EstimatorReblocking reblocking_;
};

}  // namespace alea
