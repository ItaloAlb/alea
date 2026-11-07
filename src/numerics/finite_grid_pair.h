// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Numerical pair action u=-log(rho_V/rho_0), H_rel=-laplacian/(2 mu)+V.
// Pair-product physics: Pollock & Ceperley, PRB 30, 2555 (1984), Appendix A.
// This backend uses a five-point finite-volume Hamiltonian in a 2D square,
// a positive uniformization exponential followed by matrix squaring, and
// bilinear interpolation of BOTH density matrices. It is a finite-grid
// approximation, not an exact infinite-volume Coulomb pair density matrix.
// The free discrete denominator preserves u=0 for V=0 and removes leading
// free lattice error. Finite grid, interpolation and domain biases remain.
#include <array>
#include <cstddef>
#include <functional>
#include <vector>
namespace alea::numerics {
struct PairActionValue {
    double action = 0, tau_derivative = 0;
};
class FiniteGridPair {
public:
    FiniteGridPair(double mu, double tau, double extent, int grid_points,
                   const std::function<double(double)>& radial,
                   double pole_residue);
    PairActionValue evaluate(const std::array<double, 2>& left,
                             const std::array<double, 2>& right) const;
    double tau() const noexcept { return tau_; }
    double extent() const noexcept { return extent_; }
    int grid_points() const noexcept { return points_; }

private:
    struct Stencil {
        std::array<std::size_t, 4> index;
        std::array<double, 4> weight;
    };
    Stencil stencil(const std::array<double, 2>& q) const;
    double tau_ = 0, extent_ = 0, spacing_ = 0;
    int points_ = 0;
    // Interleaved [(state_left*states + state_right)*4 + field], field order
    // {rho_V, d_tau rho_V, rho_0, d_tau rho_0}: one bilinear lookup reads the
    // four values of an index pair from one 32-byte entry instead of four
    // tables far apart (each up to ~9.5 MB at 33 points), cutting cache misses.
    std::vector<double> table_;
};
} // namespace alea::numerics
