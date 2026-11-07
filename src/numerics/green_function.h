// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cmath>

namespace alea::numerics {

// Contribution of one dimension to the Langevin Green's function log-ratio.
//
// Proposal density:  T(R'|R) ∝ Π_d exp(-|Δ_d − drift_d|² / 2σ²_d)
// with drift_d = σ²_d·g_d,  g = ∇log ψ,  σ²_d = τ/m_d.
//
// log[T(R|R') / T(R'|R)] =
//   Σ_d [ −Δ_d·(g_d + g'_d) + (σ²_d/2)·(g_d² − g'_d²) ]
//
// This function returns the per-dimension contribution (the term inside the sum).
inline double langevin_log_ratio_d(double delta,
                                   double grad_old,
                                   double grad_new,
                                   double tau_over_m) {
    return - delta * (grad_old + grad_new)
           + 0.5 * tau_over_m * (grad_old * grad_old - grad_new * grad_new);
}

// Umrigar–Nightingale–Runge drift-velocity rescaling (JCP 99, 2865 (1993)).
// Returns the scalar s that replaces τ/m in a particle's drift displacement
//   d = s·g,   g = ∇log ψ_T (for that particle),   v_sq = |g|².
// Limits:
//   v_sq → 0 : s → τ/m            (inert away from nodes — the bare drift)
//   v_sq → ∞ : |s·g| → √(2(τ/m)/a)  (a BOUNDED step where |g| → ∞, i.e. at a node)
// so the Langevin drift can no longer fling a walker into a nodal singularity —
// the fix that makes fixed-node DMC stable for sign-changing (determinant) trials.
// a = 1 is the standard choice.  Detailed balance is preserved by feeding the SAME
// scaled drift into the acceptance Green's function (langevin_log_ratio_drift_d).
inline double umrigar_drift_scale(double tau_over_m, double v_sq,
                                  double a = 1.0) {
    // Small-velocity guard: the closed form loses precision as v_sq → 0
    // (catastrophic cancellation in √(1+ε)−1); the limit is exactly τ/m.
    if (v_sq < 1e-10) return tau_over_m;
    return (-1.0 + std::sqrt(1.0 + 2.0 * a * tau_over_m * v_sq)) / (a * v_sq);
}

// Per-dimension Langevin Green's-function log-ratio with EXPLICIT forward/backward
// drift displacements:  df = s·g at R,  db = s'·g' at R'.  Generalises
// langevin_log_ratio_d (which assumes df = (τ/m)·g, i.e. s = τ/m):
//   log[T(R|R')/T(R'|R)]_d = [ −Δ_d·(df_d + db_d) + ½·(df_d² − db_d²) ] / σ²_d
// With s = s' = τ/m this reduces exactly to langevin_log_ratio_d.
inline double langevin_log_ratio_drift_d(double delta, double drift_fwd,
                                         double drift_bwd, double tau_over_m) {
    // τ/m == 0 → no diffusion, the walker does not move (Δ = drift = 0): there is
    // no Metropolis–Hastings drift correction.  Guard the 0/0 the division yields.
    if (tau_over_m <= 0.0) return 0.0;
    return (-delta * (drift_fwd + drift_bwd)
            + 0.5 * (drift_fwd * drift_fwd - drift_bwd * drift_bwd)) / tau_over_m;
}

// ── DMC Green's function helpers ───────────────────────────────────────

// Log of the forward Green's function for a Gaussian displacement Δ.
//   G_f(Δ; τ/m) = −½ (m/τ) |Δ|²   ≡  −|Δ|² / (2 τ/m)
inline double log_greens_forward(double delta_sq, double tau_over_m) {
    return -0.5 * delta_sq / tau_over_m;
}

// DMC Metropolis-Hastings acceptance probability:
//   P_acc = min(1, |ψ'/ψ|² · exp(log_gb − log_gf))
// where log_gf and log_gb are the forward and backward Green's function
// logarithms summed over all dimensions.
inline bool dmc_accept(double log_ratio, double log_gf, double log_gb,
                       double u) noexcept {
    const double log_accept = 2.0 * log_ratio + log_gb - log_gf;
    return log_accept >= 0.0 || u < std::exp(log_accept);
}

}  // namespace alea::numerics
