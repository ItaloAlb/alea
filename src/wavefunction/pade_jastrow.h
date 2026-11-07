// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <optional>

#include "periodic_pair_sum.h"
#include "wavefunction_component.h"
#include "../population/particle_group.h"
#include "../population/simulation_cell.h"

namespace alea {

// Scalar Padé form used by Jastrow factors:
//
//     u(r) = ( Σ_{k=1}^N a_k r^k ) / ( 1 + Σ_{k=1}^M b_k r^k )
//
// coeffs layout: [a_1 .. a_N, b_1 .. b_M], size == num_order + den_order.
// A pure value type with no virtuals — owns no walker state, just the math.
class PadeFunction {
public:
    PadeFunction(int num_order, int den_order, std::vector<double> coeffs);

    // Evaluate u(r), u'(r), u''(r) at a single distance r.
    void evaluate(double r, double& u, double& du, double& d2u) const noexcept;

    // ∂u/∂p at distance r for every parameter, in coeff order [a_1..a_N, b_1..b_M]:
    //   ∂u/∂a_k = r^k / Q ,   ∂u/∂b_k = −u · r^k / Q .
    // out.size() must equal num_order + den_order.  The single source of the
    // Padé parameter-derivative math, shared by PadeJastrow and the fixed-R
    // one-body / affine Jastrow components.
    void param_derivatives(double r, std::span<double> out) const noexcept;

    // Radial first and second derivatives of every ∂u/∂p_k, same order:
    // f1[k] = d/dr ∂u/∂p_k,  f2[k] = d²/dr² ∂u/∂p_k.  Feeds the linear method's
    // curvature term (∂∇logψ/∂p and ∂∇²logψ/∂p).
    void param_radial_derivatives(double r, std::span<double> f1,
                                  std::span<double> f2) const noexcept;

    // Batched evaluation over a contiguous vector of distances r[w].
    // Outputs are written (not accumulated) into u/du/d2u, which must each be
    // r.size() long. Hot path: the loop over w is vectorizable.
    void evaluate(std::span<const double> r,
                  std::span<double> u,
                  std::span<double> du,
                  std::span<double> d2u) const noexcept;

private:
    int                 num_order_;
    int                 den_order_;
    std::vector<double> coeffs_;
};

// Generalized Padé–Jastrow factor over particle pairs:
//
//     u(r) = ( Σ_{k=1}^N a_k r^k ) / ( 1 + Σ_{k=1}^M b_k r^k )
//
// Contribution to log(ψ), with same_group inferred from group_a == group_b:
//   same group : Σ_{i<j in group_a} u(r_ij)
//   two groups : Σ_{i in group_a, j in group_b} u(r_ij)
//
// params layout: [a_1 .. a_N, b_1 .. b_M], size == num_order + den_order.

class PadeJastrow final : public WavefunctionComponent {
public:
    PadeJastrow(int num_order, int den_order,
                ParticleGroup group_a, ParticleGroup group_b,
                std::vector<Parameter> params);

    std::string name() const override { return "pade_jastrow"; }

    // Sum u over every periodic image of each pair instead of the nearest one
    // (see periodic_pair_sum.h): the minimum-image Padé has a kink on the
    // Wigner–Seitz boundary.  Re-split on every rebuild(), so every parameter
    // stays optimisable.  Throws if the image sum does not exist (numerator of
    // higher order than the denominator, or a pole at some r > 0).
    void sum_over_images(const SimulationCell& cell);
    [[nodiscard]] bool sums_over_images() const noexcept { return image_sum_.active(); }

    void prepare(std::size_t nw, std::size_t np, std::size_t dim) override;

    // Reorder committed caches (base) AND resize this component's scratch to the
    // new walker count, so a subsequent range call never triggers the lazy
    // prepare() (which would wipe the just-permuted committed caches).
    void permute_caches(std::span<const std::size_t> src) override;

    void evaluate_batch(WalkerSet& batch) override;

    void ratio_batch(WalkerSet& batch, std::size_t p,
                     std::size_t w_begin = 0,
                     std::size_t w_end = npos) override;

    void apply_move_batch(WalkerSet& batch,
                           std::size_t p,
                           std::span<const bool> accept_mask,
                           std::size_t w_begin = 0,
                           std::size_t w_end = npos) override;

    [[nodiscard]] std::unique_ptr<WavefunctionComponent> clone() const override {
        return std::make_unique<PadeJastrow>(*this);
    }

    std::span<Parameter>       parameters()       override { return params_; }
    std::span<const Parameter> parameters() const override { return params_; }

    void rebuild() override;

    void compute_parameter_derivatives(const WalkerSet& batch,
                                       std::span<double> derivs) const override;

    // ∂(∇ log ψ)/∂p and ∂(∇² log ψ)/∂p: the curvature term of the linear
    // method's Hbar.  Without it the LM step length is unreliable (it
    // overshot ~2.5× for the analogous Drummond parameter).
    [[nodiscard]] bool supports_parameter_gradient_derivatives() const override {
        return true;
    }
    bool compute_parameter_gradient_derivatives(const WalkerSet& batch,
                                                std::span<double> dgrad,
                                                std::span<double> dlap) const override;

private:
    int           num_order_, den_order_;
    ParticleGroup group_a_, group_b_;
    bool          same_group_;  // inferred: group_a_ == group_b_
    std::vector<Parameter> params_;

    // Padé kernel, built from params_. Rebuild if params_ change (optimization).
    PadeFunction pade_;

    // Image sum (inactive → plain minimum image), and the cell it was built
    // for so rebuild() can redo the split.
    PeriodicPairSum              image_sum_;
    std::optional<SimulationCell> image_cell_;

    // pade_.evaluate, then the short-range part when summing over images.
    void radial(std::span<const double> r, std::span<double> u, std::span<double> du,
                std::span<double> d2u) const noexcept;
    // ∂u/∂p of every optimisable parameter, as the radial functions the split
    // needs (in optimisable order), with their large-r limits.
    void build_image_sum();

    // Per-walker scratch for the batched kernels (resized to nw, no realloc
    // once capacity is reached). The *_old_ buffers hold the Padé values at the
    // pre-move distance; mask_ is the accept flag as 0.0/1.0 for branchless use.
    std::vector<double> u_, du_, d2u_, inv_r_;
    std::vector<double> u_old_, du_old_, d2u_old_, inv_r_old_;
    std::vector<double> mask_;
    // Proposal-local old/new (u,u',u'') for each partner k. Written by ratio,
    // consumed before geometry commits; branching only resizes this scratch.
    std::vector<double> proposal_radial_; // [partner * 6 * nw + field * nw + w]

    // u(r), u'(r), u''(r) for the Padé form.
    void evaluate_u(double r, double& u, double& du, double& d2u) const;
};

}  // namespace alea
