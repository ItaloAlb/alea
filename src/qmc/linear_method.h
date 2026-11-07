// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <Eigen/Dense>

#include <cstddef>
#include <vector>

namespace alea {

struct LinearMethodConfig {
    double shift_i = 0.01;   // added to diagonal of Hbar (i,i) for i ≥ 1
    double shift_s = 0.0;    // added multiplied by Sbar(i,j) for i,j ≥ 1
    // λ: constant factor on the whole step.  1 = the paper's method; the
    // damping of large steps is ξ's job (below), not a blanket halving —
    // 0.5 (the old default) cost ~4 iterations where 1 took one (moiré N=19).
    double rescale  = 1.0;
    // ξ of the nonlinear-parameter normalisation (Toulouse & Umrigar, JCP 126,
    // 084102 (2007), Eqs. 32 and 34).  1 = no correction (Sorella's choice,
    // "sometimes too large"); 0 = minimal ‖Ψ_lin − Ψ0‖ (safe, sometimes too
    // small); 1/2 = the paper's recommendation, ‖Ψ_lin‖ = ‖Ψ0‖.
    double xi       = 0.5;
};

struct LinearMethodResult {
    std::vector<double> update;       // Δp [N]
    double              optimal_energy = 0.0;  // lowest eigenvalue
};

// Solves the linear-method generalized eigenvalue problem
//   Hbar c = E Sbar c
// and returns the parameter update for N = Sbar.rows() - 1 parameters.
//
// Stabilisation: before solving, Hbar is modified as
//   Hbar(i,i) += shift_i         for i ≥ 1
//   Hbar(i,j) += shift_s·Sbar(i,j)  for i,j ≥ 1
//
// The update is computed from the lowest eigenvector c = (c_0, v) as
//   Δp̄ = v / c_0                                  (semi-orthogonal basis)
//   Δp  = rescale · Δp̄ / (1 + x),
//   x   = (1−ξ) D / [(1−ξ) + ξ √(1+D)],   D = Δp̄ᵀ S̄ Δp̄,
// i.e. Eq. (32) with the N_i of Eq. (34), every parameter treated as
// nonlinear (alea has no CSF-type coefficients that enter ψ linearly; those
// would need Eq. (33) instead).  x ≈ (1−ξ)D/... is O(D), so small steps are
// untouched and only large ones are shortened.  It is evaluated in a form
// that stays finite as c_0 → 0, where Δp̄ → ∞ but, for ξ < 1,
//   Δp → rescale · ξ/(1−ξ) · v / √(vᵀ S̄ v).
class LinearMethod {
public:
    explicit LinearMethod(LinearMethodConfig cfg);

    [[nodiscard]] LinearMethodResult solve(const Eigen::MatrixXd& Sbar,
                                           const Eigen::MatrixXd& Hbar) const;

private:
    LinearMethodConfig cfg_;
};

}  // namespace alea
