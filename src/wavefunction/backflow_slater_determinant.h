// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "../population/simulation_cell.h"
#include "backflow_function.h"
#include "backflow_transformation.h"
#include "orbital_set.h"
#include "wavefunction_component.h"

namespace alea {

// Slater determinant with backflow: the orbitals are evaluated at quasi-
// coordinates that depend on the whole configuration
// (P. López Ríos, A. Ma, N. D. Drummond, M. D. Towler and R. J. Needs,
//  Phys. Rev. E 74, 066701 (2006), Eqs. (2)–(4) and (7)–(8);
//  Y. Kwon, D. M. Ceperley and R. M. Martin, Phys. Rev. B 48, 12037 (1993)):
//
//   ψ_det = Π_b det M_b,   M_b[i][m] = φ_m(x_i),
//   x_i   = r_i + Σ_c Σ_{j∈S_c, j≠i} η_c(r_ij) r_ij,     r_ij = r_i − r_j (minimum image),
//
// over the rows i of one fermion group G (spin blocks b as in SlaterDeterminant)
// and channels c, each a source group S_c with its own η_c (backflow_function.h).
// A channel whose source is G itself is like-spin, so its η has η'(0) = 0; the
// group must then hold one spin only (one block).  The backflow changes the
// NODES: this factor is nodal, and its η parameters move a fixed-node energy.
//
// ── Derivatives ─────────────────────────────────────────────────────────────
// With A = M⁻¹ (A[m][i]) per block and ∂_α φ at x_i:
//   F_iα            = ∂ ln|D| / ∂x_iα            = Σ_m A_mi ∂_α φ_m(x_i)
//   G^α_ji          = Σ_m A_mj ∂_α φ_m(x_i)      (i, j in one block; F_iα = G^α_ii)
//   ∂²ln|D|/∂x_iα∂x_jγ = δ_ij T_i^{αγ} − G^α_ji G^γ_ij,  T_i^{αγ} = Σ_m A_mi ∂_α∂_γ φ_m(x_i)
// (∂A_mi/∂M_jn = −A_mj A_ni), zero between blocks.  The Jacobian J_{iα,kβ} =
// ∂x_iα/∂r_kβ, with B_c(r)_{αβ} = η δ_αβ + η' r_α r_β / r,
//   J_{i,k} = −B_c(r_ik)          (k a source of row i),
//   J_{i,i} = 1 + Σ_j B_c(r_ij),
// and S_{iα,kβ} = ∂²x_iα/∂r_kβ², s_c(r)_{αβ} = ∂²_β[η(r) r_α]
//           = 2η' r_β δ_αβ / r + r_α [η'' r_β²/r² + η'(1/r − r_β²/r³)],
//   S_{i,k} = s_c(r_ik),   S_{i,i} = Σ_j s_c(r_ij),
// give the per-dimension derivatives of every particle k (G and the sources):
//   ∂_kβ ln|D|  = Σ_{iα} F_iα J_{iα,kβ}
//   ∂²_kβ ln|D| = Σ_{iα} F_iα S_{iα,kβ} + Σ_i Σ_{αγ} v_iα v_iγ T_i^{αγ}
//                 − Σ_b Σ_{i,j∈b} U_ji U_ij,       v_iα = J_{iα,kβ},  U_ji = Σ_α v_iα G^α_ji.
//
// ── Moves ───────────────────────────────────────────────────────────────────
// Moving one particle changes every quasi-coordinate within the cutoffs of its
// old and new positions, so the rank-one Sherman–Morrison update of
// SlaterDeterminant does not apply.  ratio_batch rebuilds x', M' and inverts
// every block by pivoted Gauss–Jordan (as López Ríos et al. recompute the
// determinants by LU), O(N³) per move; it keeps x', A', F' pending, and the
// moved particle's exact new gradient and Laplacian (one extra N³ product,
// U = Aᵀ W; skipped when the driver declares set_proposal_laplacian(false),
// as VMC and DMC do).  apply_move commits the pending state.  The deferred commit leaves
// the other particles' derivatives stale; refresh_particle_derivatives
// refreshes the gradient only (all the drift needs, O(N)), and
// refresh_derivatives everything (O(d N³)).
//
// ── Periodic cells ──────────────────────────────────────────────────────────
// r_ij is the minimum image, and every cutoff L ≤ the Wigner–Seitz inradius
// (Ref., Sec. III), so the sum over the nearest image is the exact periodic
// sum and η is continuous across the cell boundary.  x_i is not wrapped: the
// orbitals (plane waves, minimum-image Gaussians) are periodic in it.
//
// ── Storage ─────────────────────────────────────────────────────────────────
// Committed and pending state are walker-major (one walker's matrices
// contiguous): every kernel is dense linear algebra on one configuration.
// Per-thread scratch (one configuration's orbital rows and products) has one
// slot per processor, indexed by the OpenMP thread number so concurrent DMC
// crowds never share it; a slot is sized the first time its thread uses it
// (once), so a VMC per-thread clone holds one sized slot, not one per core.
class BackflowSlaterDeterminant final : public WavefunctionComponent {
public:
    struct Block {
        std::size_t                 first;
        std::size_t                 last;      // N = last − first == orbitals->size()
        std::unique_ptr<OrbitalSet> orbitals;  // must support_hessian()
    };
    // One backflow channel: the particles [source_first, source_last) displace
    // every row by η(r_ij) r_ij.  params: [c_0, (c_1), c_2.., L] as in
    // BackflowFunction; c_1 is derived (absent) when the source is the group.
    struct Channel {
        std::size_t            source_first = 0;
        std::size_t            source_last  = 0;
        int                    truncation   = 3;
        std::vector<Parameter> params;
    };

    BackflowSlaterDeterminant(std::vector<Block> blocks, std::vector<Channel> channels,
                              const SimulationCell& cell = SimulationCell{});
    BackflowSlaterDeterminant(const BackflowSlaterDeterminant& other);
    BackflowSlaterDeterminant& operator=(const BackflowSlaterDeterminant&) = delete;

    std::string name() const override { return "backflow_slater_determinant"; }

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
    // false: ratio_batch skips the moved particle's Laplacian (one N³ product
    // per dimension, about half its dense algebra) and contributes 0 to new_lap;
    // that Laplacian is stale until refresh_derivatives.
    void set_proposal_laplacian(bool needed) override { proposal_laplacian_ = needed; }

    [[nodiscard]] std::unique_ptr<WavefunctionComponent> clone() const override;

    [[nodiscard]] bool is_nodal() const override { return true; }
    // No nodal_blocks(): the nodes are not those of det[φ(r)], and the PIMC
    // fixed-node restriction evaluates plain orbitals, so it refuses this factor.

    // Orbital parameters first (block-major, "b<block>_<name>", as in
    // SlaterDeterminant), then each channel's [c_0, (c_1), c_2.., L] as
    // "bf<channel>_c<k>" and "bf<channel>_L".
    std::span<Parameter>       parameters() override { return params_; }
    std::span<const Parameter> parameters() const override { return params_; }
    void rebuild() override;

    // ∂ln|D|/∂p = Σ_{iα} F_iα ∂x_iα/∂p for a backflow parameter (∂x_i/∂p =
    // Σ_j ∂η(r_ij)/∂p r_ij), and Σ_i Σ_m A_mi ∂φ_m(x_i)/∂p for an orbital one.
    void compute_parameter_derivatives(const WalkerSet& batch,
                                       std::span<double> derivs) const override;

    [[nodiscard]] std::size_t walker_state_bytes() const noexcept override;

    // The orbital parameters come first in parameters(); the channels follow.
    [[nodiscard]] std::size_t n_orbital_parameters() const noexcept { return n_orbital_params_; }
    [[nodiscard]] std::size_t n_rows() const noexcept { return n_rows_; }
    [[nodiscard]] std::size_t n_channels() const noexcept { return transform_.n_channels(); }
    [[nodiscard]] const BackflowFunction& eta(std::size_t c) const { return transform_.eta(c); }
    // Committed quasi-coordinates of walker w: [n_rows * dim], row i at
    // [i*dim + d] (row i is particle first_row + i).
    [[nodiscard]] std::span<const double> quasi_coordinates(std::size_t w) const {
        return std::span<const double>(x_).subspan(w * n_rows_ * dim_, n_rows_ * dim_);
    }

private:
    // One thread's scratch for one configuration; see prepare().
    struct Scratch {
        std::vector<double> pos;     // [np * dim] AoS positions of the configuration
        std::vector<double> x;       // [n_rows * dim] quasi-coordinates
        std::vector<double> val;     // [n_rows * maxN]              val[i*maxN + m]
        std::vector<double> grad;    // [n_rows * maxN * dim]        [(i*maxN + m)*dim + α]
        std::vector<double> hess;    // [n_rows * maxN * dim * dim]  [((i*maxN + m)*dim + α)*dim + γ]
        BackflowTransformation::Tables tables;  // B, s and links of every (row, source)
        std::vector<double> work;    // [maxN * maxN] Gauss–Jordan input
        std::vector<double> at;      // [maxN * maxN] Aᵀ: at[j*N + m] = A[m][j]
        std::vector<double> gmat;    // [dim * maxN * maxN] G^α[j][i] of one block
        std::vector<double> grad_a;  // [maxN * maxN] ∂_α φ_m(x_i) of one α, [i*N + m]
        std::vector<double> t;       // [n_rows * dim * dim] T_i^{αγ}
        std::vector<double> v;       // [n_rows * dim] J column v_iα
        std::vector<double> u;       // [maxN * maxN] U_ji (or the product Aᵀ W)
        std::vector<double> w;       // [maxN * maxN] W_im = Σ_α v_iα ∂_α φ_m(x_i)
        std::vector<double> param;   // [max channel parameters] ∂η/∂p
        std::vector<double> pderiv;  // [backflow parameters] ∂ln|D|/∂p
        std::vector<double> dval;    // [maxN * max orbital parameters] orbital ∂φ/∂p
        std::vector<double> out_grad, out_lap;  // [np * dim] one configuration's ∂, ∂²
        std::size_t generation = 0;  // the sizes it was built for (scratch_generation_)
    };

    [[nodiscard]] bool affects(std::size_t p) const noexcept;
    [[nodiscard]] Scratch& scratch() const;
    void size_scratch(Scratch& s) const;
    void load_positions(const WalkerSet& batch, std::size_t w, Scratch& s) const;
    // Orbital values, gradients and Hessians at s.x for every row.
    void evaluate_rows(Scratch& s) const;
    // Invert every block of s.val into inv ([inv_size]); log|det| and sign
    // per block.  False when a block is singular.
    bool invert_blocks(Scratch& s, double* inv, double* log_det, double* sign) const;
    // F_iα of every row from inv and s.grad.
    void compute_f(const Scratch& s, const double* inv, double* f) const;
    // T_i^{αγ} of every row into s.t.
    void hessian_traces(Scratch& s, const double* inv) const;
    // ∂²_k ln|D| ([dim]) by one Aᵀ W product per β (a single particle).
    void particle_laplacian(Scratch& s, const double* inv, const double* f, std::size_t k,
                            double* out) const;
    // T and the G^α of every block, then ∂ and ∂² of every particle into
    // grad/lap ([np * dim], zero for particles this factor ignores).
    void all_derivatives(Scratch& s, const double* inv, const double* f, double* grad,
                         double* lap) const;
    // Commit pending walker w (accepted), then optionally refresh partners.
    void commit(WalkerSet& batch, std::size_t p, std::size_t w);
    void refresh_walker(WalkerSet& batch, std::size_t w, std::size_t skip);
    void resize_state(std::size_t nw);

    std::vector<Block>        blocks_;
    BackflowTransformation    transform_;   // the quasi-coordinates of the rows
    SimulationCell            cell_;
    std::size_t first_row_ = 0, n_rows_ = 0, max_block_ = 0;
    std::vector<std::size_t>  block_of_row_;   // [n_rows]
    std::vector<std::size_t>  inv_offset_;     // [n_blocks]  offset of block b in an inverse
    std::size_t               inv_size_ = 0;   // Σ_b N_b²
    std::size_t dim_ = 0, np_ = 0, nw_ = 0;

    std::vector<Parameter>                           params_;
    std::vector<std::pair<std::size_t, std::size_t>> orbital_owner_;  // {block, index}
    std::size_t n_orbital_params_ = 0;

    // Committed state, walker-major.
    std::vector<double> x_;        // [nw * n_rows * dim]
    std::vector<double> inv_;      // [nw * inv_size]   A_b[m][i] at inv_offset_[b] + m*N + i
    std::vector<double> f_;        // [nw * n_rows * dim]
    std::vector<double> log_det_;  // [nw * n_blocks]
    std::vector<double> sign_;     // [nw * n_blocks]
    // Pending state of the move in flight (ratio_batch → apply_move).
    std::vector<double> px_, pinv_, pf_, plog_det_, psign_;
    std::vector<double> pgrad_, plap_;  // [nw * dim] moved particle, this factor
    std::vector<char>   pvalid_;        // [nw] the proposal was invertible
    bool                proposal_laplacian_ = true;  // see set_proposal_laplacian

    mutable std::vector<Scratch> scratch_;   // [threads], each sized on first use
    std::size_t scratch_generation_ = 1;     // bumped when np or dim changes
};

}  // namespace alea
