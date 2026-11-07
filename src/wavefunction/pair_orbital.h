// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace alea {

class SimulationCell;

// Pair orbital of a pairing determinant, the function of the separation
// Δ = r_row − r_column of one row and one column particle:
//
//   φ(Δ) = Σ_s p_s Σ_{k ∈ star s} cos(k·Δ)  +  f(r) Σ_{m=0}^{M} c_m r^m,
//   f(r) = (1 − r/L)³ Θ(L − r),   r = |Δ|,   c_1 = 3 c_0 / L.
//
// López Ríos, Perali, Needs & Neilson, PRL 120, 177701 (2018), Eq. (S3);
// Maezono, López Ríos, Ogawa & Needs, PRL 110, 216407 (2013), Eq. (3).
//
// - The cosines carry the pair amplitude φ̃(k) of each star of reciprocal
//   vectors (v_k/u_k in BCS): a star holds every vector of one length, ±k
//   both, and star 0 is k = 0.  Only periodic 2-D cells have them.  They are
//   stored as the half-space vectors (one of each ±k) with multiplicity 2,
//   so the cosine part is Σ_i mult_i p_{star(i)} cos(k_i·Δ).
// - The cutoff polynomial carries the short-range bound pair the truncated
//   cosine sum misses.  It is the Drummond–Towler–Needs family with C = 3,
//   evaluated at the minimum image; L at most the Wigner–Seitz inradius keeps
//   that exact.
// - φ′(0) = 0: e and h of a pair need no cusp from the orbital (any cusp of
//   the pair is the Jastrow factor's), hence c_1 = 3 c_0 / L, derived.
// - φ is linear in (p, c): scaling all of them scales ψ by a constant, so at
//   least one must stay fixed (the builder enforces it).
//
// Parameters, in order: p_0 … p_{S−1}, then c_0, c_2 … c_M, then L.  Either
// part may be absent (S = 0, or no polynomial).
class PairOrbital {
public:
    // shells: number of stars S (0: no cosines; > 0 needs a periodic 2-D
    // cell).  poly_order: M ≥ 1, or −1 for no polynomial.  Throws
    // std::invalid_argument on an impossible combination.
    PairOrbital(const SimulationCell& cell, std::size_t shells, int poly_order);

    // p: [S]; c: [c_0, c_2 … c_M]; L > 0 (ignored without a polynomial).
    void set(std::span<const double> p, std::span<const double> c, double L);

    [[nodiscard]] std::size_t dim() const noexcept { return dim_; }
    [[nodiscard]] std::size_t n_stars() const noexcept { return n_stars_; }
    [[nodiscard]] bool has_polynomial() const noexcept { return poly_order_ >= 1; }
    [[nodiscard]] int poly_order() const noexcept { return poly_order_; }
    [[nodiscard]] double cutoff() const noexcept { return L_; }
    // Number of polynomial parameters: c_0, c_2 … c_M and L (0 without one).
    [[nodiscard]] std::size_t n_poly_params() const noexcept {
        return has_polynomial() ? static_cast<std::size_t>(poly_order_) + 1 : 0;
    }

    // Half-space reciprocal vectors of the stars, k = 0 first.
    [[nodiscard]] std::size_t n_k() const noexcept { return star_.size(); }
    [[nodiscard]] const double* k(std::size_t i) const noexcept { return k_.data() + i * dim_; }
    [[nodiscard]] std::size_t star(std::size_t i) const noexcept { return star_[i]; }
    [[nodiscard]] double multiplicity(std::size_t i) const noexcept { return mult_[i]; }
    // mult_i p_{star(i)}: the coefficient of cos(k_i·Δ).
    [[nodiscard]] double weight(std::size_t i) const noexcept { return weight_[i]; }
    // |k| of star s.
    [[nodiscard]] double star_k(std::size_t s) const noexcept { return star_k_[s]; }

    // g(r) = f(r) Σ c_m r^m and its first two radial derivatives; 0 beyond L
    // and without a polynomial.
    void radial(double r, double& g, double& dg, double& d2g) const noexcept;

    // ∂g/∂θ_j and its radial derivatives for polynomial parameter j
    // (0 → c_0, 1 … M−1 → c_2 … c_M, M → L).
    void radial_param(std::size_t j, double r, double& h, double& dh,
                      double& d2h) const noexcept;

private:
    std::size_t         dim_ = 0;
    std::size_t         n_stars_ = 0;
    int                 poly_order_ = -1;
    std::vector<double> k_;         // [n_k * dim] half-space vectors
    std::vector<std::size_t> star_; // [n_k]
    std::vector<double> mult_;      // [n_k] 1 for k = 0, 2 otherwise
    std::vector<double> star_k_;    // [S]
    std::vector<double> weight_;    // [n_k]
    std::vector<double> c_;         // [M + 1] c_0 … c_M, c_1 derived
    double              L_ = 0.0;
};

}  // namespace alea
