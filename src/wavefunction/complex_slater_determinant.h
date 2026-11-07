// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <complex>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "complex_orbital_set.h"
#include "wavefunction_component.h"

namespace alea {

// Slater determinant of COMPLEX orbitals: ψ_det = Π_b det(D_b), D_b[j][i] =
// φ_i(r_{first+j}), φ_i ∈ ℂ (e.g. twisted plane waves).  Complex counterpart of
// SlaterDeterminant, kept separate so the real determinant — the fixed-node hot
// path — stays exactly as it is.
//
// Writing ln ψ = u + iφ, every derivative splits into the modulus part (the
// batch's usual planes) and the phase part (the batch's phase planes):
//   ∇ ln ψ   (particle j) = G = Σ_m (D⁻¹)_{m,j} ∇φ_m(r_j)             ∈ ℂ
//   ∂²_d ln ψ(particle j) = Σ_m (D⁻¹)_{m,j} ∂²_d φ_m(r_j) − G_d²       ∈ ℂ
//   move ratio  R         = Σ_m (D⁻¹)_{m,j} φ_m(r_j')                  ∈ ℂ
// Re → gradients/laplacians/log_ratios(ln|R|), Im → phase_gradients/
// phase_laplacians/phase_diffs(arg R).  The trial is complex, so DMC runs
// fixed phase (Ortiz, Ceperley & Martin, PRL 71, 2777 (1993)); with twisted
// plane waves this is the twist-averaged-boundary method of Lin, Zong &
// Ceperley, PRE 64, 016702 (2001).
//
// Per-walker dense N×N complex algebra (Eigen); no shared mutable state across
// walkers, so crowds run concurrently on disjoint walker ranges.
class ComplexSlaterDeterminant final : public WavefunctionComponent {
public:
    struct Block {
        std::size_t                        first;
        std::size_t                        last;   // N = last - first == orbitals->size()
        std::unique_ptr<ComplexOrbitalSet> orbitals;
    };

    explicit ComplexSlaterDeterminant(std::vector<Block> blocks);

    std::string name() const override { return "complex_slater_determinant"; }

    [[nodiscard]] bool is_complex() const override { return true; }
    [[nodiscard]] bool is_nodal() const override { return true; }

    void prepare(std::size_t nw, std::size_t np, std::size_t dim) override;
    void permute_caches(std::span<const std::size_t> src) override;

    void evaluate_batch(WalkerSet& batch) override;
    void ratio_batch(WalkerSet& batch, std::size_t p, std::size_t w_begin = 0,
                     std::size_t w_end = npos) override;
    void apply_move_batch(WalkerSet& batch, std::size_t p,
                          std::span<const bool> accept_mask,
                          std::size_t w_begin = 0,
                          std::size_t w_end = npos) override;

    [[nodiscard]] std::unique_ptr<WavefunctionComponent> clone() const override;

    // Orbitals come from the cell and the twist: nothing to optimise.
    std::span<Parameter>       parameters() override { return {}; }
    std::span<const Parameter> parameters() const override { return {}; }
    void rebuild() override {}
    void compute_parameter_derivatives(const WalkerSet&,
                                       std::span<double>) const override {}

private:
    using cplx = std::complex<double>;

    [[nodiscard]] std::size_t block_of(std::size_t p) const noexcept;
    void resize_state(std::size_t nw, std::size_t dim);

    // Orbital values/derivatives of block b at every block particle's position
    // (the moved particle `moved` at batch.new_pos() when moved != npos) into
    // the row scratch, over [w_begin, w_end).
    void fill_rows(const WalkerSet& batch, std::size_t b, std::size_t moved,
                   std::size_t w_begin, std::size_t w_end);

    std::vector<Block> blocks_;
    std::size_t        dim_ = 0;
    std::size_t        nw_  = 0;

    // Committed inverse per block: d_inv_[b][(a*N + c)*nw + w] = (D_b⁻¹)_{a,c}.
    std::vector<std::vector<cplx>> d_inv_;
    // Committed phase part of this component's ∇ln ψ / ∂²ln ψ, [np*dim*nw]
    // (the modulus part lives in the base class's grad_c_/lap_c_).
    std::vector<double> phase_grad_c_;
    std::vector<double> phase_lap_c_;

    // Row scratch, sized to the largest block N: row kr = orbitals at block
    // particle kr's position, split re/im planes.
    struct Rows {
        std::vector<double> vr, vi, gr, gi, lr, li;
    };
    std::vector<Rows> rows_;   // [maxN]
    Rows              nrow_;   // orbitals at new_pos (ratio)
};

}  // namespace alea
