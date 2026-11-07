// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "estimators/estimator.h"
#include "population/particle_group.h"
#include "population/simulation_cell.h"
#include "population/walker_set.h"
#include "wavefunction/trial_wavefunction.h"

namespace alea {

// ── Ratios of virtual displacements ─────────────────────────────────────────
// The particles whose displacements a measurement evaluates: `singles` are
// moved one at a time, and in each channel (a, b) of `pairs` one particle of
// group a and one of group b are moved together.
struct ParticleRange {
    std::size_t first = 0, last = 0;   // particles [first, last)
    [[nodiscard]] std::size_t size() const noexcept { return last - first; }
};
struct VirtualMovePlan {
    std::vector<std::size_t> singles;                            // [q] particle index
    std::vector<std::pair<ParticleRange, ParticleRange>> pairs;  // [channel] (a, b)
    // Σ over channels of N_a N_b: the rows of two_body below.
    [[nodiscard]] std::size_t n_two_body() const noexcept;
    // max over channels of N_a N_b: the rows of the scratch below.
    [[nodiscard]] std::size_t max_channel() const noexcept;
};

// For one displacement r′_w per configuration w of `batch` (displacement
// [d * nw + w]), the wavefunction ratios
//
//   one_body[q * nw + w]           = ψ(…, r_p + r′_w, …) / ψ(R_w),   p = singles[q]
//   two_body[(o_c + i N_b + j) * nw + w]
//                                  = ψ(…, r_{a_i} + r′_w, …, r_{b_j} + r′_w, …) / ψ(R_w)
//
// with o_c the rows of the channels before c, as signed real numbers
// exp(log|ψ′/ψ|)·cos(arg ψ′/ψ) (a real trial: the phase is 0 or π).
//
// A one-body ratio is a rejected move of a Metropolis sweep (propose →
// ratio_batch → apply_move_batch with nothing accepted), the move code the
// sampling trusts; the displaced position is wrapped into `cell` first.  The
// two-body ratios of a channel come from TrialWavefunction::
// shifted_pair_ratios_batch, which every component computes from R alone:
// moving a_i and then b_j would pass through R with a_i moved, where ψ can
// vanish (a compact pair orbital leaves the moved a_i with no partner in
// range, a row of zeros) and the second ratio is then 0/0.
// `batch` must hold R and have been evaluated by `psi`; it is left as it was.
// scratch: [max_channel() * nw], the phases of one channel.
void virtual_move_ratios(TrialWavefunction& psi, WalkerSet& batch, const SimulationCell& cell,
                         const VirtualMovePlan& plan, std::span<const double> displacement,
                         std::span<double> one_body, std::span<double> two_body,
                         std::span<double> scratch);

// Stream kind of the per-sample generators below (numerics::stream_seed).
inline constexpr std::uint64_t kDensityMatrixSampleStream = 0x646d61742d736d70ULL;  // "dmat-smp"

// ── One- and two-body density matrices, pair condensate fraction ────────────
// Translational-rotational averages of the density matrices from ratios of the
// trial wavefunction at virtual displacements (López Ríos, Perali, Needs &
// Neilson, PRL 120, 177701 (2018), Supplement Eqs. S6–S8; Astrakharchik,
// Boronat, Casulleras & Giorgini, PRL 95, 230405 (2005)).
//
// One-body density matrix of group g (N_g particles, density n_g = N_g/A),
//     ρ⁽¹⁾_g(r) = n_g ⟨ ψ(…, r_i + r′, …) / ψ(R) ⟩,   |r′| = r,        (S6)
// averaged over the sampled R, the particles i of g and the directions of r′.
// Reported as rho1_g(r) = ρ⁽¹⁾_g(r)/n_g, which is 1 at r = 0; for bosons its
// large-r plateau is the condensate fraction (Penrose & Onsager, Phys. Rev.
// 104, 576 (1956)).
//
// Two-body density matrix of a channel (a, b), one particle of each group
// displaced by the SAME r′,
//     ρ⁽²⁾_ab(r) = n_a n_b ⟨ ψ(…, r_i + r′, …, r_j + r′, …) / ψ(R) ⟩,  (S7)
// averaged also over the N_a N_b pairs (i, j); reported as rho2_ab = ρ⁽²⁾/(n_a n_b).
//
// Pair condensate fraction of the channel,
//     n_c(r) = (A²/N_p) [ρ⁽²⁾_ab(r) − ρ⁽¹⁾_a(r) ρ⁽¹⁾_b(r)]                (S8)
//            = (N_a N_b / N_p) [rho2_ab(r) − rho1_a(r) rho1_b(r)],   N_p = min(N_a, N_b),
// whose large-r plateau is the fraction of the N_p pairs in the condensate
// (1 for a single bound pair, whose ψ depends on r_a − r_b only).
//
// What is sampled.  VMC: R ~ |ψ_T|², the variational density matrix.  DMC: R
// ~ ψ_T Φ with the walker weights, the mixed estimate ⟨Φ|ρ̂|ψ_T⟩/⟨Φ|ψ_T⟩, whose
// error is first order in Φ − ψ_T; combine as 2·DMC − VMC.  Each measured
// configuration gets one displacement r′: its length uniform within one bin
// (bins taken in turn, so every bin is sampled equally), its direction uniform.
// Every particle of each listed group, and every pair of each channel, is
// evaluated at that r′.  The random numbers of sample s come from
// stream_seed(seed, kDensityMatrixSampleStream, s), s counting the measured
// configurations of this instance in order, so the samples do not depend on
// how the walkers are split among workers.
//
// Cost.  A measurement evaluates ψ_T once per configuration, Σ_g N_g
// one-body ratios, and per channel the components' pair ratios: O(N_a N_b n)
// for a determinant of size n, O((N_a + N_b) N) for a pair Jastrow — for the
// bilayer of the paper (two channels of 29 × 29 pairs) a few sweeps — so
// `every` sets how many sampling steps pass between measurements.  Every
// component of ψ_T must supply shifted_pair_ratios_batch when `pairs` is
// given; bind_wavefunction refuses a trial that does not.
//
// Errors.  Blocks are treated as independent (BlockMoments), like the other
// VMC/DMC estimators; a block closes only once every bin holds a sample, until
// then it merges into the next, and one still open when the run ends is
// dropped (density_matrix_measurements counts the configurations kept).  The
// products in n_c(r) are formed per block.
class DensityMatrix final : public Estimator {
public:
    // one_body: groups whose rho1 is measured; the groups of `pairs` are
    // added after them.  pairs: channels (a, b) of two different groups.
    DensityMatrix(std::span<const ParticleGroup> groups, std::vector<std::string> one_body,
                  std::vector<std::pair<std::string, std::string>> pairs, double rmax,
                  int n_bins, int every, const SimulationCell& cell, std::size_t dim);
    // Configuration and statistics; never the bound copies of ψ_T, which each
    // instance gets from bind_wavefunction.
    DensityMatrix(const DensityMatrix& other);
    DensityMatrix& operator=(const DensityMatrix&) = delete;

    [[nodiscard]] bool needs_wavefunction() const noexcept override { return true; }
    void bind_wavefunction(const TrialWavefunction& psi, std::uint64_t seed,
                           int n_workers) override;

    void start_block() override;
    using Estimator::accumulate;
    void accumulate(const ParticleBatch& configs, std::span<const double> weights) override;
    void stop_block() override;
    void collect(const Estimator& other) override;
    void finalize(RunResult& out) const override;

    [[nodiscard]] std::unique_ptr<Estimator> clone() const override {
        return std::make_unique<DensityMatrix>(*this);
    }
    [[nodiscard]] std::string type() const override { return "density_matrix"; }
    // density_matrix_r, _rho1_<g>, _rho2_<a>_<b>, _pair_condensate_fraction_<a>_<b>, ...
    [[nodiscard]] std::vector<std::string> output_namespaces() const override {
        return {"density_matrix"};
    }

    [[nodiscard]] std::size_t n_bins() const noexcept { return n_bins_; }
    [[nodiscard]] double bin_width() const noexcept { return delta_; }

private:
    struct Group {
        std::string name;
        ParticleRange range;
    };
    struct Channel {
        std::size_t a = 0, b = 0;   // indices into groups_
    };
    // One worker of the walker loop: its own ψ_T copy and a batch of
    // chunk_walkers_ configurations to move virtually.
    struct Worker {
        TrialWavefunction psi;
        WalkerSet batch;
        std::vector<double> displacement;  // [dim * chunk]
        std::vector<double> scratch;       // [max_channel * chunk]
        std::vector<double> one_body;      // [n_singles * chunk]
        std::vector<double> two_body;      // [n_two_body * chunk]
        std::vector<std::size_t> bin;      // [chunk]
        std::vector<double> sums;          // [sum_width()] this measurement's partial sums
    };

    // Per-measurement sums: [rho1 numerators g·n_bins + b | rho2 numerators
    // c·n_bins + b | weight b].
    [[nodiscard]] std::size_t sum_width() const noexcept {
        return (groups_.size() + channels_.size() + 1) * n_bins_;
    }
    void allocate_workers(std::size_t chunk);
    void measure_chunk(Worker& worker, const ParticleBatch& configs,
                       std::span<const double> weights, std::size_t w0, std::uint64_t sample0);

    std::vector<Group>   groups_;    // one-body groups: listed, then those of the channels
    std::vector<Channel> channels_;
    VirtualMovePlan      plan_;      // singles: every particle of groups_, in order
    SimulationCell       cell_;
    std::size_t dim_ = 0, np_ = 0;
    double      rmax_ = 0.0, delta_ = 0.0;
    std::size_t n_bins_ = 0;
    std::size_t every_ = 1;

    // Bound state (bind_wavefunction); never copied.  The workers are made
    // from psi_copy_ at the first measurement, when the walker count fixes
    // their chunk size: once per run, never per step.
    std::unique_ptr<TrialWavefunction> psi_copy_;
    std::uint64_t            seed_ = 0;
    std::size_t              n_workers_ = 0;
    std::vector<Worker>      workers_;

    std::size_t   calls_ = 0;          // accumulate calls (every_-th one measures)
    std::uint64_t samples_ = 0;        // configurations measured so far (RNG and bin index)
    std::vector<double> block_sums_;   // [sum_width()] open block, possibly carried over
    bool          carry_ = false;      // the open block lacked a bin and continues
    std::size_t   block_samples_ = 0;  // configurations in the open block
    std::size_t   closed_samples_ = 0; // configurations in closed blocks
    std::vector<double> block_values_; // [n_groups + 2 n_channels] · n_bins, per-block scratch
    BlockMoments  moments_;            // [rho1 g | rho2 c | n_c c] · n_bins
};

}  // namespace alea
