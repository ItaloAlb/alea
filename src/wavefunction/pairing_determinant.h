// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "orbital_set.h"
#include "pair_orbital.h"
#include "wavefunction_component.h"
#include "../population/particle_group.h"

namespace alea {

// Pairing (geminal) determinant: one determinant per block of a row group and
// a column group,
//
//   ψ = Π_b det D_b,   (D_b)_{ac} = φ(r_a − r_c),
//
// with r_a a particle of the block's row group and r_c one of its column
// group (López Ríos, Perali, Needs & Neilson, PRL 120, 177701 (2018), Eq. S2;
// for e–h: rows e↑, columns h↓, and a second block e↓, h↑).  All blocks share
// one pair orbital φ (pair_orbital.h).  det D_b is antisymmetric under the
// exchange of two row particles (two rows) or two column particles (two
// columns), so a fully polarized row or column group needs the P-wave cusp
// from the Jastrow factor; φ itself is cuspless.
//
// A block with more row than column particles (an N+½ system, Eq. S4) is
// completed by single-particle orbitals χ_o of the row particles in extra
// columns, (D)_{a, N_c + o} = χ_o(r_a); with more column particles, extra rows
// χ_o(r_c).  The extra orbitals carry no parameters.
//
// Derivatives, for a move of row particle a (one row of D changes) or column
// particle c (one column changes), with A = D⁻¹ and ∂ log det/∂D_{ac} = A_{ca}:
//   ratio                R = Σ_c A_{ca} v_c            (row)  /  Σ_a A_{ca} u_a  (column)
//   ∂_d log|ψ| (row a)    = Σ_c A_{ca} ∂_d D_{ac}
//   ∂_d log|ψ| (column c) = Σ_a A_{ca} ∂_{c,d} D_{ac},  ∂_{c,d} φ(r_a − r_c) = −∂_d φ
//   ∂²_d log|ψ|           = Σ A ∂²_d D − (∂_d log|ψ|)²
// and the proposal's values use the updated inverse line A′ = A/R.  Accepted
// moves update A by Sherman–Morrison (row: A′ = A − (A e_a) wᵀ/R with
// wᵀ = (v − d_a)ᵀ A; column: A′ = A − z (e_cᵀ A)/R with z = A (u − d^c)),
// rebuilt by LU every 64 updates, on |R| < 1e-5, or on a non-finite ratio.
//
// A move ratio can be negative: its sign goes into batch.phase_diffs() as π,
// and the DMC fixed-node gate rejects the node crossing (is_nodal()).
//
// Storage is walker-major (one contiguous matrix per walker): the work per
// move is dense O(n²) linear algebra on that walker's matrix, as in the large
// blocks of SlaterDeterminant.  Per walker: D, A (n² each), ∂D and ∂²D
// (n²·dim each) per block, and cos/sin(k·r) of every block particle for the
// cosine part of φ.
class PairingDeterminant final : public WavefunctionComponent {
public:
    struct Block {
        ParticleGroup               rows;
        ParticleGroup               columns;
        std::unique_ptr<OrbitalSet> unpaired;   // |N_rows − N_columns| orbitals, or null
    };

    // params: [p_0 … p_{S−1}, c_0, c_2 … c_M, L] of `orbital`.  Throws
    // std::invalid_argument when a block is unbalanced without the matching
    // unpaired orbitals, a group appears in two blocks, or the parameter
    // count does not match the orbital.
    PairingDeterminant(PairOrbital orbital, std::vector<Block> blocks,
                       std::vector<Parameter> params);
    PairingDeterminant(const PairingDeterminant& other);
    PairingDeterminant& operator=(const PairingDeterminant&) = delete;

    std::string name() const override { return "pairing_determinant"; }

    void prepare(std::size_t nw, std::size_t np, std::size_t dim) override;
    void permute_caches(std::span<const std::size_t> src) override;

    void evaluate_batch(WalkerSet& batch) override;
    void ratio_batch(WalkerSet& batch, std::size_t p, std::size_t w_begin = 0,
                     std::size_t w_end = npos) override;
    void apply_move_batch(WalkerSet& batch, std::size_t p, std::span<const bool> accept,
                          std::size_t w_begin = 0, std::size_t w_end = npos) override;

    [[nodiscard]] std::unique_ptr<WavefunctionComponent> clone() const override {
        return std::make_unique<PairingDeterminant>(*this);
    }

    [[nodiscard]] bool is_nodal() const override { return true; }

    // A row particle r and a column particle c of one block, shifted by the
    // same s: D changes by a row and a column, D″ = D + e_r xᵀ + y e_cᵀ, and
    // the corner φ(r_r + s − r_c − s) is unchanged, so x_c = y_r = 0.  By the
    // matrix determinant lemma, with A = D⁻¹ (A[c][a] = (D⁻¹)_{ca}),
    //   det D″/det D = (1 + xᵀA e_r)(1 + e_cᵀA y) − (xᵀA y)(e_cᵀA e_r),
    // computed from the full new row and column x⁰, y⁰ (corner included) as
    //   xᵀA e_r = (R_r − 1) − x⁰_c A_cr,   e_cᵀA y = (R_c − 1) − y⁰_r A_cr,
    //   xᵀA y = (x⁰ᵀA)·y⁰ − y⁰_r (R_r − 1) − x⁰_c (R_c − 1) + x⁰_c y⁰_r A_cr,
    // R the one-particle ratios (Harville, Matrix Algebra From a Statistician's
    // Perspective (1997), Thm. 18.1.1).  O(n²) per particle, O(n) per pair.
    // Particles in different blocks (or none) multiply their one-particle
    // ratios; two of the same side of one block are refused.
    [[nodiscard]] bool supports_shifted_pairs() const override { return true; }
    void shifted_pair_ratios_batch(const WalkerSet& batch, std::size_t a_first,
                                   std::size_t a_last, std::size_t b_first, std::size_t b_last,
                                   std::span<const double> shift, std::span<double> log_ratios,
                                   std::span<double> phases) override;

    std::span<Parameter>       parameters() override { return params_; }
    std::span<const Parameter> parameters() const override { return params_; }
    void rebuild() override;

    // Periodic cells: the polynomial cutoff must stay within the Wigner–Seitz
    // inradius, where the minimum image is the exact image sum.  The optimizer
    // clamps L there (Parameter::upper_bound) and rebuild() throws if anything
    // else takes it past (as DrummondJastrow::bound_cutoff).
    void bound_cutoff(double max_cutoff) noexcept {
        max_cutoff_ = max_cutoff;
        if (orbital_.has_polynomial()) params_.back().upper_bound = max_cutoff;   // [p, c, L]
    }

    // ∂ log|ψ|/∂θ = Σ_b Tr(A_b ∂D_b/∂θ).
    void compute_parameter_derivatives(const WalkerSet& batch,
                                       std::span<double> derivs) const override;
    [[nodiscard]] bool supports_parameter_gradient_derivatives() const override {
        return true;
    }
    // ∂_θ of the gradient and laplacian above, with ∂A = −A (∂D) A.
    bool compute_parameter_gradient_derivatives(const WalkerSet& batch,
                                                std::span<double> dgrad,
                                                std::span<double> dlap) const override;

    [[nodiscard]] std::size_t walker_state_bytes() const noexcept override;

    [[nodiscard]] const PairOrbital& orbital() const noexcept { return orbital_; }
    // log|det D_b| of walker w as tracked by the updates (tests, diagnostics).
    [[nodiscard]] double block_log(std::size_t b, std::size_t w) const noexcept {
        return log_[b][w];
    }
    // Inverse A_b = D_b⁻¹ of walker w, row-major [n * n] (tests, diagnostics).
    [[nodiscard]] std::span<const double> inverse(std::size_t b, std::size_t w) const noexcept {
        return {a_[b].data() + mat(b, w), shape_[b].n * shape_[b].n};
    }
    // Committed matrix D_b of walker w, row-major [n * n] (tests, diagnostics).
    [[nodiscard]] std::span<const double> matrix(std::size_t b, std::size_t w) const noexcept {
        return {d_[b].data() + mat(b, w), shape_[b].n * shape_[b].n};
    }
    [[nodiscard]] std::size_t n_blocks() const noexcept { return blocks_.size(); }
    [[nodiscard]] const Block& block(std::size_t b) const noexcept { return blocks_[b]; }

private:
    struct Shape {
        std::size_t n_rows = 0, n_columns = 0, n = 0;  // n = max(n_rows, n_columns)
        bool extra_columns = false;   // n_rows > n_columns: χ_o(r_a) in columns
        bool extra_rows    = false;   // n_columns > n_rows: χ_o(r_c) in rows
    };
    struct Location {
        std::size_t block = npos;
        bool        row = true;      // row particle (else column)
        std::size_t line = 0;        // its row (or column) index
    };

    [[nodiscard]] Location locate(std::size_t p) const noexcept;

    // φ at one entry: value, ∂_d φ(Δ) and ∂²_d φ(Δ), Δ = x_row − x_col folded
    // to the minimum image.  cos/sin tables of both particles, [n_k] each.
    void entry(const SimulationCell& cell, const double* x_row, const double* x_col,
               const double* cos_row, const double* sin_row, const double* cos_col,
               const double* sin_col, double& value, double* grad, double* lap) const;

    // cos/sin(k_i·x) for every half-space k_i of the orbital.
    void trig(const double* x, double* c, double* s) const noexcept;

    // Pivoted LU of D_b for walker w into A_b; returns log|det D_b|.
    double rebuild_inverse(std::size_t b, std::size_t w);

    // ∇ and ∂² log|det D_b| of every particle of block b for walker w, into
    // grad_c_/lap_c_; with `totals`, the change also goes into the batch
    // totals of every particle except `moved` (apply_move overwrites its own).
    void block_derivatives(WalkerSet& batch, std::size_t b, std::size_t w, bool totals,
                           std::size_t moved);

    void size_state(std::size_t nw, std::size_t np, std::size_t dim);

    // Per-walker offsets.
    [[nodiscard]] std::size_t mat(std::size_t b, std::size_t w) const noexcept {
        return w * shape_[b].n * shape_[b].n;
    }

    PairOrbital            orbital_;
    std::vector<Block>     blocks_;
    std::vector<Shape>     shape_;
    std::vector<Parameter> params_;
    double max_cutoff_ = std::numeric_limits<double>::infinity();

    std::size_t nw_ = 0, np_ = 0, dim_ = 0, n_max_ = 0, n_k_ = 0, n_slots_ = 0;
    std::vector<std::size_t> slot_;   // [np] cos/sin table slot of particle, or npos

    // Committed state, per block.
    std::vector<std::vector<double>> d_;      // [nw * n * n]        D[a][c]
    std::vector<std::vector<double>> a_;      // [nw * n * n]        A = D⁻¹
    std::vector<std::vector<double>> g_;      // [nw * n * n * dim]  ∂_d D[a][c] (row-owner sign)
    std::vector<std::vector<double>> l_;      // [nw * n * n * dim]  ∂²_d D[a][c]
    std::vector<std::vector<double>> log_;    // [nw]                log|det D_b|
    std::vector<std::vector<int>>    updates_;// [nw]                Sherman–Morrison count
    std::vector<double> cos_, sin_;           // [nw * n_slots * n_k]

    // Pending proposal (one moving particle), per walker.
    std::vector<double> pend_v_;     // [nw * n_max]        new line values
    std::vector<double> pend_g_;     // [nw * n_max * dim]  ∂ wrt the moving particle
    std::vector<double> pend_l_;     // [nw * n_max * dim]
    std::vector<double> pend_r_;     // [nw]                ratio R
    std::vector<double> pend_scale_; // [nw]                Σ |A_line v|, for the cancellation test
    std::vector<double> pend_cos_, pend_sin_;   // [nw * n_k]
    std::vector<double> work_;       // [nw * n_max]        Sherman–Morrison vector
    std::vector<double> line_;       // [nw * n_max]        copy of the old inverse line

    // Unpaired-orbital scratch at one particle, walker-fastest (OrbitalSet layout).
    std::vector<double> unp_v_, unp_g_, unp_l_;

    // shifted_pair_ratios_batch, per moved particle t (a, then b); they grow
    // to the largest request (an estimator call, not a sweep).
    std::vector<double> shift_pos_;     // [dim * nw]                shifted position
    std::vector<double> shift_delta_;   // [(t * nw + w) * n_max]    new line − old line
    std::vector<double> shift_xa_;      // [(t * nw + w) * n_max]    x⁰ᵀA (row particles)
    std::vector<double> shift_ratio_;   // [t * nw + w]              one-particle ratio R
};

}  // namespace alea
