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

#include "../population/particle_group.h"
#include "../population/simulation_cell.h"
#include "backflow_transformation.h"
#include "orbital_set.h"
#include "pair_orbital.h"
#include "wavefunction_component.h"

namespace alea {

// Pairing (geminal) determinant with backflow: the pair orbital and the
// unpaired orbitals are evaluated at quasi-coordinates of both the row and the
// column particles (pairing_determinant.h for the determinant; P. López Ríos,
// A. Ma, N. D. Drummond, M. D. Towler and R. J. Needs, Phys. Rev. E 74, 066701
// (2006) and backflow_transformation.h for the transformation):
//
//   ψ = Π_b det M_b,
//   M[a][c] = φ(x_a − y_c)        a a row particle, c a column particle,
//   M[a][N_c + o] = χ_o(x_a)       extra columns (more rows than columns),
//   M[N_r + o][c] = χ_o(y_c)       extra rows (more columns than rows),
//
// x, y the quasi-coordinates of the row and column particles.  Every entry is
// a function f of one argument: Δ = x_a − y_c (paired), x_a or y_c (unpaired),
// with gradient g and Hessian H there.  Along the Jacobian column v of a
// particle k and dimension β (v_t = ∂x_t/∂r_kβ), with u_ac = v_a − v_c
// (paired), v_a (extra column) or v_c (extra row):
//   dM_ac  = g_ac · u_ac,      d²M_ac = u_acᵀ H_ac u_ac,
//   ∂_kβ ln|D|  = Σ_t F_t · v_t,  F the derivative of ln|D| with respect to
//                 each quasi-coordinate (F_a = Σ_c A_ca ∂_a M_ac, likewise F_c),
//   ∂²_kβ ln|D| = Σ_t F_t · ∂²x_t/∂r_kβ² + tr(A d²M) − tr((A dM)²),
// A = M⁻¹ (∂ ln det/∂M_ac = A_ca, ∂A = −A ∂M A).
//
// tr((A dM)²) for every (k, β) at O(N²) each: with P_α (∂M/∂x of the row
// owner) and Q_α (∂M/∂y of the column owner), dM = Σ_α (V_α P_α + Q_α W_α),
// V_α = diag(v_{a,α}) over the rows and W_α = diag(v_{c,α}) over the columns,
// so with X_α = P_α A, Y_α = A Q_α and Z_αγ = X_α Q_γ, computed once,
//   tr((A dM)²) = Σ_αγ [ Σ_{aa'} V_αa X_α[a][a'] V_γa' X_γ[a'][a]
//                     + 2 Σ_{ac} V_αa Z_αγ[a][c] W_γc A[c][a]
//                     + Σ_{cc'} W_αc Y_γ[c][c'] W_γc' Y_α[c'][c] ].
// A single particle (a proposal) forms A dM directly, one N³ product per β.
//
// Moves rebuild and invert every block (pivoted Gauss–Jordan, O(N³) per
// move); x' is the committed x shifted in O(N), as in the backflow Slater
// determinant, whose deferred-commit and proposal-Laplacian contract this
// follows.  The cosine part of φ is evaluated through cos/sin(k·x) of every
// quasi-coordinate, cos(k·(x − y)) = c_x c_y + s_x s_y, which needs no fold;
// the polynomial part at the minimum image, with L ≤ the inradius.
class BackflowPairingDeterminant final : public WavefunctionComponent {
public:
    struct Block {
        ParticleGroup               rows;
        ParticleGroup               columns;
        std::unique_ptr<OrbitalSet> unpaired;   // |N_rows − N_columns| orbitals (with Hessians), or null
    };

    // orbital_params: [p_0 … p_{S−1}, c_0, c_2 … c_M, L] of `orbital`.
    // channels: targets among the blocks' row and column groups.  Throws
    // std::invalid_argument as PairingDeterminant and BackflowTransformation do.
    BackflowPairingDeterminant(PairOrbital orbital, std::vector<Block> blocks,
                               std::vector<Parameter> orbital_params,
                               std::vector<BackflowTransformation::Channel> channels,
                               const SimulationCell& cell = SimulationCell{});
    BackflowPairingDeterminant(const BackflowPairingDeterminant& other);
    BackflowPairingDeterminant& operator=(const BackflowPairingDeterminant&) = delete;

    std::string name() const override { return "backflow_pairing_determinant"; }

    void prepare(std::size_t nw, std::size_t np, std::size_t dim) override;
    void permute_caches(std::span<const std::size_t> src) override;
    void evaluate_batch(WalkerSet& batch) override;
    void ratio_batch(WalkerSet& batch, std::size_t p, std::size_t w_begin = 0,
                     std::size_t w_end = npos) override;
    void apply_move_batch(WalkerSet& batch, std::size_t p, std::span<const bool> accept,
                          std::size_t w_begin = 0, std::size_t w_end = npos) override;
    void apply_move_batch_deferred(WalkerSet& batch, std::size_t p,
                                   std::span<const bool> accept, std::size_t w_begin = 0,
                                   std::size_t w_end = npos) override;
    void refresh_particle_derivatives(WalkerSet& batch, std::size_t p,
                                      std::size_t w_begin = 0, std::size_t w_end = npos) override;
    void refresh_derivatives(WalkerSet& batch, std::size_t w_begin = 0,
                             std::size_t w_end = npos) override;
    void set_proposal_laplacian(bool needed) override { proposal_laplacian_ = needed; }

    [[nodiscard]] std::unique_ptr<WavefunctionComponent> clone() const override {
        return std::make_unique<BackflowPairingDeterminant>(*this);
    }
    [[nodiscard]] bool is_nodal() const override { return true; }

    // The orbital's [p.., c_0, c_2.., L], then the channels' [c_0, (c_1), c_2.., L].
    std::span<Parameter>       parameters() override { return params_; }
    std::span<const Parameter> parameters() const override { return params_; }
    void rebuild() override;
    // As PairingDeterminant::bound_cutoff, for the orbital's polynomial L.
    void bound_cutoff(double max_cutoff) noexcept;
    [[nodiscard]] std::size_t n_orbital_parameters() const noexcept { return n_orbital_params_; }

    // Σ_b Tr(A_b ∂M_b/∂θ) for an orbital parameter, Σ_t F_t · ∂x_t/∂p for a
    // backflow one.
    void compute_parameter_derivatives(const WalkerSet& batch,
                                       std::span<double> derivs) const override;

    // ∂(∂_kβ ln|D|)/∂p and ∂(∂²_kβ ln|D|)/∂p, the curvature term of the linear
    // method, by differentiating the Derivatives above along p (Q̇ = ∂Q/∂p):
    //   an orbital parameter changes the entries (Ṁ, ġ, Ḣ from ∂φ/∂θ) at fixed
    //   arguments; a backflow parameter moves the arguments, Δ̇ = ẋ_a − ẋ_c (or
    //   ẋ_a, ẋ_c), so Ṁ = g·Δ̇, ġ = H Δ̇, Ḣ = ∇H·Δ̇ (third derivatives), and v, S
    //   move with Ḃ, ṡ;  Ȧ = −A Ṁ A,  Ḟ by the product rule;
    //   ∂_p(∂_kβ ln|D|)  = Σ_t (Ḟ_t·v_t + F_t·v̇_t),
    //   ∂_p(∂²_kβ ln|D|) = Σ (Ḟ·S + F·Ṡ) + Σ_ac [Ȧ_ca d²M_ac + A_ca (2u̇ᵀH u + uᵀḢu)_ac]
    //                      − 2 [tr(R dṀ) − tr(Ṁ E R)],
    // with E = A dM, R = E A (from ∂ tr(E²) = 2 tr(E Ė), Ė = A(dṀ − Ṁ E)) and
    // dṀ_ac = ġ·u + g·u̇.  E, R and E R are three N³ products per (k, β), shared
    // by every parameter.  Every term but those in u̇ is a sum over entries of a
    // parameter-dependent factor (Ȧ, Ḣ, ġ, Ṁ) times a (k, β)-dependent one, so
    // one matrix-vector product Y z gives all parameters at O(d² N²) each; the
    // u̇ terms reduce to per-particle sums, O(N).  Measurement frequency; the
    // batch must be evaluated (or refreshed) at its positions.
    [[nodiscard]] bool supports_parameter_gradient_derivatives() const override;
    bool compute_parameter_gradient_derivatives(const WalkerSet& batch, std::span<double> dgrad,
                                                std::span<double> dlap) const override;
    [[nodiscard]] std::size_t walker_state_bytes() const noexcept override;

    [[nodiscard]] const PairOrbital& orbital() const noexcept { return orbital_; }
    [[nodiscard]] const BackflowTransformation& transformation() const noexcept { return transform_; }

private:
    enum class Kind : char { paired, row, column };   // the argument of an entry
    struct Shape {
        std::size_t n_rows = 0, n_columns = 0, n = 0;
        std::vector<std::size_t> row_t;     // [n] transformed index of row a's owner, or npos
        std::vector<std::size_t> col_t;     // [n] of column c's owner, or npos
        std::vector<Kind>        kind;      // [n * n]
    };
    struct Scratch {
        std::vector<double> pos, x;           // [np * dim], [T * dim]
        BackflowTransformation::Tables tables;
        std::vector<double> cx, sx;           // [T * n_k] cos/sin(k·x)
        std::vector<double> val, g, hess;     // per block, [Σ n²], [Σ n² dim], [Σ n² dim²]
        std::vector<double> unp_v, unp_g, unp_h;   // one particle's unpaired orbitals
        std::vector<double> work;             // [n_max²]
        std::vector<double> pq;               // [2 dim n_max²]  P_α, Q_α of one block
        std::vector<double> xyz;              // [(2 dim + dim²) Σ n²]  X_α, Y_α, Z_αγ per block
        std::vector<double> v;                // [T * dim]
        std::vector<double> va, wc;           // [dim * n_max] V_α, W_α
        std::vector<double> dm, e;            // [n_max²]
        std::vector<double> param, pderiv;    // channel scratch, [transform params]
        std::vector<double> out_grad, out_lap;    // [np * dim]
        std::size_t generation = 0;
    };

    [[nodiscard]] Scratch& scratch() const;
    void size_scratch(Scratch& s) const;
    void load_positions(const WalkerSet& batch, std::size_t w, Scratch& s) const;
    void evaluate_entries(Scratch& s) const;
    bool invert_blocks(Scratch& s, double* inv, double* log_det, double* sign) const;
    void compute_f(const Scratch& s, const double* inv, double* f) const;
    // f ([T * dim]) += Σ_entries A_ca ∂M_ac/∂(owner) for A in the layout of an
    // inverse and g ([Σ n² * dim]) the entries' argument gradients.
    void accumulate_f(const double* inv, const double* g, double* f) const;
    // out ([T * dim]) += Σ_entries ±x_e over each entry's owners, with the
    // signs of ∂(argument)/∂(owner): +1 row, −1 column of a paired entry, +1
    // the owner of an unpaired one; x_e: [Σ n² * dim].  Σ_e u_e·x_e = Σ_t v_t·out_t.
    void accumulate_owner(const double* x_e, double* out) const;
    // The direction u of entry (a, c) along the per-particle vectors vt
    // ([T * dim]; v_a − v_c, v_a or v_c by the entry's kind), into u[dim].
    void direction(const Shape& sh, const double* vt, std::size_t a, std::size_t c,
                   double* u) const noexcept;
    // Every entry's third derivative ∂_α∂_β∂_γ at its argument, [Σ n² * dim³];
    // unpaired_third: [n_max * dim³] scratch for one owner's unpaired orbitals.
    void entry_third(const Scratch& s, std::vector<double>& third,
                     std::vector<double>& unpaired_third) const;
    void particle_laplacian(Scratch& s, const double* inv, const double* f, std::size_t k,
                            double* out) const;
    void all_derivatives(Scratch& s, const double* inv, const double* f, double* grad,
                         double* lap) const;
    void commit(WalkerSet& batch, std::size_t p, std::size_t w);
    void refresh_walker(WalkerSet& batch, std::size_t w, std::size_t moved);
    void resize_state(std::size_t nw);

    PairOrbital            orbital_;
    std::vector<Block>     blocks_;
    std::vector<Shape>     shape_;
    BackflowTransformation transform_;
    SimulationCell         cell_;
    std::vector<Parameter> params_;
    std::size_t n_orbital_params_ = 0;
    double max_cutoff_ = std::numeric_limits<double>::infinity();

    std::size_t dim_ = 0, np_ = 0, nw_ = 0, n_k_ = 0, n_t_ = 0, n_max_ = 0;
    std::vector<std::size_t> mat_offset_;   // [n_blocks] offset of block b in Σ n²
    std::size_t mat_size_ = 0;              // Σ_b n_b²

    // Committed state, walker-major.
    std::vector<double> x_, inv_, f_, log_det_, sign_;   // [nw*T*dim], [nw*Σn²], [nw*T*dim], [nw*nb] ×2
    std::vector<double> px_, pinv_, pf_, plog_det_, psign_, pgrad_, plap_;
    std::vector<char>   pvalid_;
    bool proposal_laplacian_ = true;

    mutable std::vector<Scratch> scratch_;
    std::size_t scratch_generation_ = 1;
};

}  // namespace alea
