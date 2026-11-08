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

#include "../population/simulation_cell.h"
#include "backflow_slater_determinant.h"   // Channel
#include "backflow_transformation.h"
#include "complex_orbital_set.h"
#include "wavefunction_component.h"

namespace alea {

// Slater determinant with backflow and COMPLEX orbitals: the twisted-boundary
// counterpart of BackflowSlaterDeterminant (twisted plane waves away from Γ).
//
//   ψ_det = Π_b det M_b,   M_b[i][m] = φ_m(x_i) ∈ ℂ,
//   x_i   = r_i + Σ_c Σ_{j∈S_c, j≠i} η_c(r_ij) r_ij      (real, minimum image),
//
// with the quasi-coordinates, channels and rules of BackflowSlaterDeterminant
// (P. López Ríos, A. Ma, N. D. Drummond, M. D. Towler and R. J. Needs,
//  Phys. Rev. E 74, 066701 (2006), Eqs. (2)–(4) and (7)–(8)).
//
// ── Derivatives ─────────────────────────────────────────────────────────────
// Every formula of BackflowSlaterDeterminant's Derivatives section holds with
// ln D in place of ln|D|: they follow from ∂ ln det M = tr(M⁻¹ ∂M) and
// ∂A = −A ∂M A, which hold for a complex M.  F_iα, G^α_ji, T_i^{αγ} and U_ji
// become complex; the Jacobian J = ∂x/∂r and S = ∂²x/∂r² stay real, so
// BackflowTransformation applies to the real and imaginary parts of F in turn.
// Writing ln ψ = u + iφ, as in ComplexSlaterDeterminant:
//   Re ∂_kβ ln D, Re ∂²_kβ ln D  → the batch's gradients / laplacians,
//   Im ∂_kβ ln D, Im ∂²_kβ ln D  → its phase gradients / phase laplacians,
//   ln|D'/D| and arg(D'/D)       → log_ratios / phase_diffs.
// The trial is complex, so DMC runs fixed phase (G. Ortiz, D. M. Ceperley and
// R. M. Martin, PRL 71, 2777 (1993)); with twisted plane waves this is the
// twist-averaged boundary method of C. Lin, F. H. Zong and D. M. Ceperley,
// PRE 64, 016702 (2001), with backflow nodes and phase.
//
// ── Moves ───────────────────────────────────────────────────────────────────
// As in BackflowSlaterDeterminant: a move rebuilds x' and M' and inverts every
// block by pivoted Gauss–Jordan, O(N³); the moved particle's Laplacian (and
// phase Laplacian) is skipped under set_proposal_laplacian(false), and the
// deferred commit leaves the partners' derivatives stale until a refresh.
//
// ── Twisted boundaries ──────────────────────────────────────────────────────
// x_i is not wrapped, and r_ij is the minimum image with every cutoff within
// the Wigner–Seitz inradius.  A wrap of r_j by a lattice vector a leaves every
// r_ij, so every x_i with i ≠ j, and shifts x_j by a: row j picks up the
// constant phase e^{iθ·a} of φ(x + a) = e^{iθ·a} φ(x).  |ψ| and ∇ ln ψ do not
// change.  The committed x, inverse and arg D stay in the gauge they were
// computed in (refresh_walker evaluates the orbitals at the committed x for
// that reason); the next move of j rebuilds x_j from its wrapped position, so
// that one ratio's arg(D'/D) also carries θ·a, as ComplexSlaterDeterminant's
// does.  No fixed-phase quantity reads that arg.
//
// ── Parameters ──────────────────────────────────────────────────────────────
// Each channel's [c_0, (c_1), c_2.., L], named "bf<c>_c<k>" and "bf<c>_L" and
// ordered as BackflowSlaterDeterminant lists them for plane waves (which have
// no orbital parameters), so the values a Γ-point optimization writes carry
// over to a twisted run unchanged.  Their log-derivative is complex,
//   O = ∂ ln D/∂p = Σ_{iα} F_iα ∂x_iα/∂p,   ∂x_i/∂p = Σ_j ∂η(r_ij)/∂p r_ij (real),
// and the optimizer takes both parts (CostFunction's complex accumulation).
// The curvature term differentiates every quantity of the Derivatives
// section along p, as BackflowSlaterDeterminant does, with complex Ȧ, Ḟ, Ṫ,
// Ġ, U̇ and real ẋ, Ḃ, ṡ, v̇; the orbitals' third derivatives enter because
// p moves x.
class ComplexBackflowSlaterDeterminant final : public WavefunctionComponent {
public:
    struct Block {
        std::size_t                        first;
        std::size_t                        last;      // N = last − first == orbitals->size()
        std::unique_ptr<ComplexOrbitalSet> orbitals;  // must support_hessian()
    };
    using Channel = BackflowSlaterDeterminant::Channel;

    ComplexBackflowSlaterDeterminant(std::vector<Block> blocks, std::vector<Channel> channels,
                                     const SimulationCell& cell = SimulationCell{});
    ComplexBackflowSlaterDeterminant(const ComplexBackflowSlaterDeterminant& other);
    ComplexBackflowSlaterDeterminant& operator=(const ComplexBackflowSlaterDeterminant&) = delete;

    std::string name() const override { return "complex_backflow_slater_determinant"; }

    [[nodiscard]] bool is_complex() const override { return true; }
    [[nodiscard]] bool is_nodal() const override { return true; }

    void prepare(std::size_t nw, std::size_t np, std::size_t dim) override;
    void permute_caches(std::span<const std::size_t> src) override;

    void evaluate_batch(WalkerSet& batch) override;
    void ratio_batch(WalkerSet& batch, std::size_t p, std::size_t w_begin = 0,
                     std::size_t w_end = npos) override;
    void apply_move_batch(WalkerSet& batch, std::size_t p, std::span<const bool> accept_mask,
                          std::size_t w_begin = 0, std::size_t w_end = npos) override;
    void apply_move_batch_deferred(WalkerSet& batch, std::size_t p,
                                   std::span<const bool> accept, std::size_t w_begin = 0,
                                   std::size_t w_end = npos) override;
    void refresh_particle_derivatives(WalkerSet& batch, std::size_t p,
                                      std::size_t w_begin = 0, std::size_t w_end = npos) override;
    void refresh_derivatives(WalkerSet& batch, std::size_t w_begin = 0,
                             std::size_t w_end = npos) override;
    void set_proposal_laplacian(bool needed) override { proposal_laplacian_ = needed; }

    [[nodiscard]] std::unique_ptr<WavefunctionComponent> clone() const override;

    std::span<Parameter>       parameters() override { return params_; }
    std::span<const Parameter> parameters() const override { return params_; }
    void rebuild() override;
    // Re O = ∂ ln|D|/∂p; compute_complex_parameter_derivatives gives both parts.
    void compute_parameter_derivatives(const WalkerSet& batch,
                                       std::span<double> derivs) const override;
    // O = Σ_{iα} F_iα ∂x_iα/∂p from the committed F: Re into derivs, Im into
    // phase_derivs (skipped when empty).  F and ∂x/∂p are gauge-free (see
    // Twisted boundaries), so a wrapped walker needs nothing special.
    void compute_complex_parameter_derivatives(const WalkerSet& batch, std::span<double> derivs,
                                               std::span<double> phase_derivs) const override;

    // ∂_p(∂_kβ ln D) and ∂_p(∂²_kβ ln D) (see Parameters), O(d² N³) per walker
    // and parameter, at measurement frequency.  The orbitals are evaluated at
    // the COMMITTED x, for the gauge reason refresh_walker gives.  Not
    // supplied when an orbital set has no third derivatives or a channel has
    // C < 3 (η'' then jumps at L: BackflowSlaterDeterminant explains why).
    [[nodiscard]] bool supports_parameter_gradient_derivatives() const override;
    // The modulus parts only (allocates the phase parts it discards).
    bool compute_parameter_gradient_derivatives(const WalkerSet& batch, std::span<double> dgrad,
                                                std::span<double> dlap) const override;
    bool compute_complex_parameter_gradient_derivatives(const WalkerSet& batch,
                                                        std::span<double> dgrad,
                                                        std::span<double> dlap,
                                                        std::span<double> dpgrad,
                                                        std::span<double> dplap) const override;

    [[nodiscard]] std::size_t walker_state_bytes() const noexcept override;

    [[nodiscard]] std::size_t n_rows() const noexcept { return n_rows_; }
    [[nodiscard]] std::size_t n_channels() const noexcept { return transform_.n_channels(); }
    [[nodiscard]] const BackflowFunction& eta(std::size_t c) const { return transform_.eta(c); }
    // Committed quasi-coordinates of walker w: [n_rows * dim], row i at
    // [i*dim + d] (row i is particle first_row + i).
    [[nodiscard]] std::span<const double> quasi_coordinates(std::size_t w) const {
        return std::span<const double>(x_).subspan(w * n_rows_ * dim_, n_rows_ * dim_);
    }

private:
    using cplx = std::complex<double>;

    // One thread's scratch for one configuration; layouts as in
    // BackflowSlaterDeterminant::Scratch, complex where the orbitals are.
    struct Scratch {
        std::vector<double> pos;     // [np * dim] AoS positions
        std::vector<double> x;       // [n_rows * dim] quasi-coordinates
        std::vector<cplx>   val;     // [n_rows * maxN]              val[i*maxN + m]
        std::vector<cplx>   grad;    // [n_rows * maxN * dim]        [(i*maxN + m)*dim + α]
        std::vector<cplx>   hess;    // [n_rows * maxN * dim * dim]  [((i*maxN + m)*dim + α)*dim + γ]
        BackflowTransformation::Tables tables;
        std::vector<cplx>   work;    // [maxN * maxN] Gauss–Jordan input
        std::vector<cplx>   at;      // [maxN * maxN] Aᵀ
        std::vector<cplx>   gmat;    // [dim * inv_size] G^α[j][i] of every block
        std::vector<cplx>   grad_a;  // [maxN * maxN] ∂_α φ_m(x_i) of one α
        std::vector<cplx>   t;       // [n_rows * dim * dim] T_i^{αγ}
        std::vector<double> v;       // [n_rows * dim] J column v_iα (real)
        std::vector<cplx>   u;       // [maxN * maxN] U_ji (or the product Aᵀ W)
        std::vector<cplx>   w;       // [maxN * maxN] W_im = Σ_α v_iα ∂_α φ_m(x_i)
        std::vector<double> f_re, f_im;           // [n_rows * dim] the parts of one F
        std::vector<cplx>   out_grad, out_lap;    // [np * dim] one configuration's ∂, ∂²
        std::vector<double> param;                // [max channel parameters] ∂η/∂p
        std::vector<double> pderiv_re, pderiv_im; // [n parameters] Re, Im ∂ln D/∂p
        std::size_t generation = 0;  // the sizes it was built for (scratch_generation_)
    };

    // ∂/∂p of one configuration's Scratch quantities (same layouts as the
    // fields they differentiate), for the curvature term.
    struct Tangent {
        std::vector<cplx>   third;     // [n_rows * maxN * dim³] ∂_α∂_γ∂_δ φ_m(x_i)
        std::vector<double> x_dot;     // [n_rows * dim]
        BackflowTransformation::Tables tables;  // Ḃ, ṡ
        std::vector<double> eta;       // [3 * max channel parameters] ∂η, ∂η', ∂η''
        std::vector<cplx>   m_dot;     // [n_rows * maxN]          Ṁ_im
        std::vector<cplx>   g_dot;     // [n_rows * maxN * dim]    (∂_αφ_m)˙ at x_i
        std::vector<cplx>   h_dot;     // [n_rows * maxN * dim²]   (∂_α∂_γφ_m)˙ at x_i
        std::vector<cplx>   a_dot;     // [inv_size]  Ȧ, layout of an inverse
        std::vector<cplx>   prod;      // [maxN * maxN] A Ṁ of one block
        std::vector<cplx>   f_dot;     // [n_rows * dim]
        std::vector<double> f_dot_re, f_dot_im;  // [n_rows * dim] its parts
        std::vector<cplx>   t_dot;     // [n_rows * dim²]
        std::vector<cplx>   gmat_dot;  // [dim * inv_size] Ġ^α
        std::vector<double> v_dot, v_tmp;  // [n_rows * dim] v̇, and a discarded copy of v
        std::vector<cplx>   u, u_dot;  // [maxN * maxN] U and U̇ of one block
        std::vector<cplx>   out_grad, out_lap;  // [np * dim] the result
    };
    void size_tangent(Tangent& t) const;
    // Ȧ, Ḟ, Ṫ, Ġ from t.m_dot/g_dot/h_dot, then ∂_p of every particle's
    // gradient and Laplacian into t.out_grad/out_lap; t.tables holds Ḃ, ṡ.
    // s holds T, G (gradient_matrices) and the split committed F.
    void tangent_derivatives(Scratch& s, Tangent& t, const cplx* inv) const;

    [[nodiscard]] bool affects(std::size_t p) const noexcept { return transform_.affects(p); }
    [[nodiscard]] Scratch& scratch() const;
    void size_scratch(Scratch& s) const;
    void load_positions(const WalkerSet& batch, std::size_t w, Scratch& s) const;
    // Orbital values, gradients and Hessians at s.x for every row.
    void evaluate_rows(Scratch& s) const;
    // Invert every block of s.val into inv ([inv_size]); ln|det| and arg det
    // per block.  False when a block is singular.
    bool invert_blocks(Scratch& s, cplx* inv, double* log_det, double* arg_det) const;
    // F_iα of every row from inv and s.grad, and its parts into s.f_re/f_im.
    void compute_f(Scratch& s, const cplx* inv, cplx* f) const;
    void split_f(Scratch& s, const cplx* f) const;
    // ∂_k ln D ([dim]) = Σ_t F_t · ∂x_t/∂r_k, F already split into s.f_re/f_im.
    void particle_gradient(Scratch& s, std::size_t k, cplx* out) const;
    // T_i^{αγ} of every row into s.t.
    void hessian_traces(Scratch& s, const cplx* inv) const;
    // ∂²_k ln D ([dim]) by one Aᵀ W product per β (a single particle).
    void particle_laplacian(Scratch& s, const cplx* inv, std::size_t k, cplx* out) const;
    // T into s.t and every block's G^α into s.gmat.
    void gradient_matrices(Scratch& s, const cplx* inv) const;
    // ∂ and ∂² ln D of every particle into s.out_grad/out_lap (zero for the
    // particles this factor ignores); F already split.
    void all_derivatives(Scratch& s, const cplx* inv) const;
    // Commit pending walker w (accepted).
    void commit(WalkerSet& batch, std::size_t p, std::size_t w);
    void refresh_walker(WalkerSet& batch, std::size_t w, std::size_t skip);
    void resize_state(std::size_t nw);

    std::vector<Block>        blocks_;
    BackflowTransformation    transform_;
    SimulationCell            cell_;
    std::size_t first_row_ = 0, n_rows_ = 0, max_block_ = 0;
    std::vector<std::size_t>  block_of_row_;   // [n_rows]
    std::vector<std::size_t>  inv_offset_;     // [n_blocks]
    std::size_t               inv_size_ = 0;   // Σ_b N_b²
    std::size_t dim_ = 0, np_ = 0, nw_ = 0;
    std::vector<Parameter>    params_;

    // Committed state, walker-major (as BackflowSlaterDeterminant).
    std::vector<double> x_;        // [nw * n_rows * dim]
    std::vector<cplx>   inv_;      // [nw * inv_size]
    std::vector<cplx>   f_;        // [nw * n_rows * dim]
    std::vector<double> log_det_;  // [nw * n_blocks]  ln|det M_b|
    std::vector<double> arg_det_;  // [nw * n_blocks]  arg det M_b
    // Committed phase part of this factor's ∇ ln ψ / ∂² ln ψ, SoA [np*dim*nw]
    // (the modulus part lives in the base class's grad_c_/lap_c_).
    std::vector<double> phase_grad_c_, phase_lap_c_;
    // Pending state of the move in flight (ratio_batch → apply_move).
    std::vector<double> px_, plog_det_, parg_det_;
    std::vector<cplx>   pinv_, pf_;
    std::vector<cplx>   pgrad_, plap_;  // [nw * dim] moved particle, this factor
    std::vector<char>   pvalid_;        // [nw] the proposal was invertible
    bool                proposal_laplacian_ = true;

    mutable std::vector<Scratch> scratch_;   // [threads], each sized on first use
    std::size_t scratch_generation_ = 1;
};

}  // namespace alea
