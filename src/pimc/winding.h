// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Winding estimators of a closed path (Pollock & Ceperley, PRB 36, 8343 (1987)).
//
// For each closed loop c (a ring when pi = identity, otherwise a whole
// permutation cycle) the loop winding W_c is the sum of the minimum-image link
// vectors around the loop.  It is a lattice vector of the simulation cell,
// W_c = sum_i n_ci a_i, with integer n_ci = the fractional coordinates of the
// sum in the cell basis (SimulationCell::vectors()).  The winding of the path
// is the total over its loops,
//   W = sum_c W_c = sum_i n_i a_i,   n_i = sum_c n_ci,
// the displacement of all particles around imaginary time (Pollock & Ceperley
// Eq. (19), sum_i (r_P(i) - r_i) = W L).  The superfluid fraction needs this
// total: a phase twist phi across the cell multiplies the weight of each path
// by exp(i phi . W) (their Eq. (20)), so the second derivative of the free
// energy in phi is <W W^T>, which gives rho_s / rho = m <|W|^2> / (d beta N)
// (their Eq. (22), W in length units).  The sum over loops of |W_c|^2 is not
// the same quantity: it misses the cross terms sum_{c != c'} <W_c . W_c'>,
// which vanish for ideal particles (their loops wind independently, with
// symmetric distributions) but not with interactions, where one loop's
// winding is correlated with the others'.
//
//  - winding2_of: sum_i n_i^2 / (N dim), the lattice-basis components of the
//    total.  In an orthogonal cell with equal sides L it equals
//    <|W|^2>/(L^2 N dim).  In a skew cell it is NOT proportional to the
//    superfluid fraction: in a hexagonal cell W = a1 - a2 has |W| = L but
//    sum_i n_i^2 = 2.
//  - winding_tensor_of: T_ab = W_a W_b / N, Cartesian components of the total
//    in length units (bohr) in the cell's Cartesian frame, valid for any cell
//    shape.  For one species of mass m (hbar = 1) the superfluid-fraction
//    tensor is rho_s,ab / rho = m T_ab / beta, and in d dimensions
//    rho_s / rho = m tr(T) / (d beta).  A C3- or C4-symmetric fluid has
//    T_xx = T_yy and T_xy = 0; the projection on a unit vector u is u^T T u.

#include <cmath>
#include <cstddef>
#include <span>

#include "pimc/path_set.h"
#include "population/simulation_cell.h"

namespace alea {

// Independent components of the symmetric d x d tensor, ordered
// xx, xy, yy, xz, yz, zz (the 2-D components first).
constexpr std::size_t kMaxWindingTensor = 6;
constexpr std::size_t winding_tensor_size(std::size_t dim) noexcept {
    return dim * (dim + 1) / 2;
}
inline const char* winding_tensor_label(std::size_t c) noexcept {
    static const char* const labels[kMaxWindingTensor] = {"xx", "xy", "yy", "xz", "yz", "zz"};
    return c < kMaxWindingTensor ? labels[c] : "";
}
// Cartesian indices (a, b), a <= b, of component c.
inline void winding_tensor_indices(std::size_t c, std::size_t& a, std::size_t& b) noexcept {
    static constexpr std::size_t ia[kMaxWindingTensor] = {0, 0, 1, 0, 1, 2};
    static constexpr std::size_t ib[kMaxWindingTensor] = {0, 1, 1, 2, 2, 2};
    a = ia[c];
    b = ib[c];
}

namespace detail {

// Calls fn(f) with the fractional winding f[0, dim) of every closed loop of
// a periodic path.
template <class Fn>
void for_each_loop_winding(const PathSet& path, Fn&& fn) {
    const std::size_t M   = path.num_slices();
    const std::size_t np  = path.active_particles();
    const std::size_t dim = path.dim();
    double sum[SimulationCell::kMaxDim];
    double del[SimulationCell::kMaxDim];
    double f[SimulationCell::kMaxDim];
    if (!path.identity_permutation()) {
        for (const std::size_t leader : path.cycle_leaders()) {
            for (std::size_t i = 0; i < dim; ++i) sum[i] = 0.0;
            std::size_t p = leader;
            do {
                for (std::size_t k = 0; k < M; ++k) {
                    const auto after = path.bead_after(p, k);
                    for (std::size_t i = 0; i < dim; ++i)
                        del[i] = path.pos(after.particle, i, after.slice) -
                                 path.pos(p, i, k);
                    path.cell().min_image(del);
                    for (std::size_t i = 0; i < dim; ++i) sum[i] += del[i];
                }
                p = path.next_particle(p);
            } while (p != leader);
            path.cell().fractional(sum, f);
            fn(static_cast<const double*>(f));
        }
        return;
    }
    for (std::size_t p = 0; p < np; ++p) {
        for (std::size_t i = 0; i < dim; ++i) sum[i] = 0.0;
        for (std::size_t k = 0; k < M; ++k) {
            const std::size_t kn = (k + 1) % M;
            for (std::size_t i = 0; i < dim; ++i)
                del[i] = path.pos(p, i, kn) - path.pos(p, i, k);
            path.cell().min_image(del);
            for (std::size_t i = 0; i < dim; ++i) sum[i] += del[i];
        }
        path.cell().fractional(sum, f);
        fn(static_cast<const double*>(f));
    }
}

}  // namespace detail

// One pass over the links of a closed path.  Returns the lattice-basis
// winding2 of the total winding (sum_i n_i^2 / (N dim), n_i = sum over loops of
// n_ci) and writes the Cartesian tensor of the total per particle,
// tensor[c] = W_a W_b / N, for the winding_tensor_size(dim) components c of
// winding_tensor_indices.  For the tensor each loop winding is rounded to its
// exact lattice vector before the sum; winding2 keeps the unrounded fractional
// sum.  Both are zero for open cells, where winding is not defined.  Cost
// O(N M) min-image links, no allocation.
inline double winding_measure(const PathSet& path, double* tensor) {
    const std::size_t dim = path.dim();
    const std::size_t nc  = winding_tensor_size(dim);
    for (std::size_t c = 0; c < nc; ++c) tensor[c] = 0.0;
    if (!path.cell().is_periodic()) return 0.0;
    const std::span<const double> a = path.cell().vectors();   // [i * dim + d]
    double n_total[SimulationCell::kMaxDim] = {};   // lattice basis, unrounded [i]
    double W[SimulationCell::kMaxDim] = {};         // Cartesian total [d]
    detail::for_each_loop_winding(path, [&](const double* f) {
        for (std::size_t i = 0; i < dim; ++i) {
            n_total[i] += f[i];
            const double n = std::nearbyint(f[i]);
            for (std::size_t d = 0; d < dim; ++d) W[d] += n * a[i * dim + d];
        }
    });
    // Per particle of the measured (closed) path; an empty path winds 0.
    const std::size_t np = path.active_particles();
    if (np == 0) return 0.0;
    const double inv = 1.0 / static_cast<double>(np);
    for (std::size_t c = 0; c < nc; ++c) {
        std::size_t p = 0, q = 0;
        winding_tensor_indices(c, p, q);
        tensor[c] = W[p] * W[q] * inv;
    }
    double w2 = 0.0;
    for (std::size_t i = 0; i < dim; ++i) w2 += n_total[i] * n_total[i];
    return w2 / static_cast<double>(np * dim);
}

// The two halves of winding_measure, for callers that need only one.
inline double winding2_of(const PathSet& path) {
    double tensor[kMaxWindingTensor];
    return winding_measure(path, tensor);
}
inline void winding_tensor_of(const PathSet& path, double* tensor) {
    (void)winding_measure(path, tensor);
}

}  // namespace alea
