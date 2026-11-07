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

#include "radial_pair_jastrow.h"
#include "../population/particle_group.h"

namespace alea {

// Scalar Drummond–Towler–Needs polynomial Jastrow u-term
// (N. D. Drummond, M. D. Towler, R. J. Needs, Phys. Rev. B 70, 235119 (2004),
//  Eq. 19 — the electron–electron term only):
//
//   u(r) = (r − L)^C · Θ(L − r) · [ α0 + α1 r + Σ_{l=2..N} αl r^l ]
//   α1   = Γ / (−L)^C + α0 · C / L
//
// The linear coefficient α1 is NOT free: it is fixed so that du/dr|_{r=0} = Γ
// (the Kato cusp value) for any α0 and L. The (r−L)^C·Θ(L−r) factor makes u and
// its first C−1 derivatives go to zero at the cutoff length L, and u ≡ 0 beyond.
// C ≥ 2 is required (C=2 → continuous gradient; C=3 → continuous local energy).
//
// coeffs layout: [α0, α2, α3, …, αN], size N (α1 is excluded — cusp-derived).
// A pure value type with no virtuals — owns no walker state, just the math.
class DrummondFunction {
public:
    DrummondFunction(int C, double gamma, double L, std::vector<double> coeffs);

    // Evaluate u(r), u'(r), u''(r) at a single distance r (0 for r ≥ L).
    void evaluate(double r, double& u, double& du, double& d2u) const noexcept;

    // Batched evaluation over a contiguous vector of distances r[w].
    // Outputs are written (not accumulated) into u/du/d2u, each r.size() long.
    void evaluate(std::span<const double> r,
                  std::span<double> u,
                  std::span<double> du,
                  std::span<double> d2u) const noexcept;

    [[nodiscard]] double cutoff() const noexcept { return L_; }

    // Parameter derivatives ∂u/∂p_k at a single distance r.  Output indices map
    // directly to the coeffs_ + L layout used by DrummondJastrow's params vector
    // (see the note beside its params_): out[0..N−1] = ∂u/∂α0…∂u/∂αN,
    // out[N] = ∂u/∂L.  All outputs are 0 for r ≥ L (the cutoff is strict).
    void param_derivatives(double r, std::span<double> out) const noexcept;
    [[nodiscard]] double cusp() const noexcept { return gamma_; }
    [[nodiscard]] double alpha1() const noexcept { return alpha1_; }

private:
    int                 C_;
    double              gamma_;   // Γ: target value of du/dr|_{r=0}
    double              L_;       // cutoff length
    double              alpha1_;  // derived: Γ/(−L)^C + α0 C/L
    std::vector<double> coeffs_;  // [α0, α2..αN]
};

// Drummond–Towler–Needs Jastrow factor over particle pairs: log ψ = Σ u(r_ij)
// over the pairs of one or several group pairs sharing u (all with the same
// cusp Γ), evaluated by RadialPairJastrow.
//
// params layout: [α0, α2, …, αN, L].  The N leading entries are the polynomial
// coefficients (α1 excluded — cusp-derived); the final entry is the cutoff
// length L (a nonlinear variational parameter).  The cusp Γ and truncation C
// are fixed at construction.
extern template class RadialPairJastrow<DrummondFunction>;

class DrummondJastrow final : public RadialPairJastrow<DrummondFunction> {
public:
    DrummondJastrow(int C, double gamma,
                    ParticleGroup group_a, ParticleGroup group_b,
                    std::vector<Parameter> params);
    DrummondJastrow(int C, double gamma, std::vector<JastrowGroupPair> pairs,
                    std::vector<Parameter> params);

    std::string name() const override { return "drummond_jastrow"; }

    // Periodic cells: the cutoff must stay within the Wigner–Seitz inradius.
    // There the minimum image is the exact sum over periodic images (no other
    // image is in reach), so this u needs no Ewald split; beyond it the farther
    // images would be silently dropped.  L remains optimisable: the optimizer
    // clamps it at `max_cutoff` (Parameter::upper_bound), and rebuild() throws
    // if anything else takes it past.
    void bound_cutoff(double max_cutoff) noexcept {
        max_cutoff_                = max_cutoff;
        params_.back().upper_bound = max_cutoff;   // [α0, α2…αN, L]
    }

    [[nodiscard]] std::unique_ptr<WavefunctionComponent> clone() const override {
        return std::make_unique<DrummondJastrow>(*this);
    }

    std::span<Parameter>       parameters()       override { return params_; }
    std::span<const Parameter> parameters() const override { return params_; }

    // The Kato cusp Γ this term was built with: the S-wave value for a nodeless
    // pair, the smaller P-wave one when a determinant already supplies the node.
    [[nodiscard]] double cusp() const noexcept { return gamma_; }

    void rebuild() override;

    void compute_parameter_derivatives(const WalkerSet& batch,
                                       std::span<double> derivs) const override;

    // ∂(∇ log ψ)/∂p and ∂(∇² log ψ)/∂p, the curvature term of the linear
    // method's Hbar.  Without it the Drummond step overshoots badly: the
    // derivative ∂u/∂α0 ~ (r−L)^C is huge, so H̄ − E S̄ is dominated by the
    // missing <D ∂E_L> block.
    [[nodiscard]] bool supports_parameter_gradient_derivatives() const override {
        return true;
    }
    bool compute_parameter_gradient_derivatives(const WalkerSet& batch,
                                                std::span<double> dgrad,
                                                std::span<double> dlap) const override;

private:
    int           C_;
    double        gamma_;
    std::vector<Parameter> params_;  // [α0, α2..αN, L]; radial_ is rebuilt from it

    // Upper bound on L (the inradius of a periodic cell); ∞ when open.
    double max_cutoff_ = std::numeric_limits<double>::infinity();
};

}  // namespace alea
