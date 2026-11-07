// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "population/simulation_cell.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace alea {
namespace {

// Lagrange–Gauss reduction of a 2-D basis, in place on rows a0, a1.
// Terminates with |a0| <= |a1| and |a0·a1| <= |a0|²/2.  Every step is a
// unimodular row operation, so the lattice and |det| are unchanged.
//
// Needed because rounding the fractional coordinates only lands in the correct
// lattice shell when the basis is reduced; on a skewed basis the true nearest
// image can sit further away in index space than the 3^dim scan reaches.
void gauss_reduce_2d(double* a) noexcept {
    for (int guard = 0; guard < 64; ++guard) {
        const double n0 = a[0] * a[0] + a[1] * a[1];
        const double n1 = a[2] * a[2] + a[3] * a[3];
        if (n1 < n0) {
            std::swap(a[0], a[2]);
            std::swap(a[1], a[3]);
            continue;
        }
        if (n0 == 0.0) return;  // singular; the caller rejects it
        const double mu = std::nearbyint((a[0] * a[2] + a[1] * a[3]) / n0);
        if (mu == 0.0) return;
        a[2] -= mu * a[0];
        a[3] -= mu * a[1];
    }
}

}  // namespace

SimulationCell::SimulationCell(std::size_t dim) : boundary_(Boundary::Open), dim_(dim) {
    if (dim > kMaxDim)
        throw std::runtime_error("SimulationCell: dimension " + std::to_string(dim) +
                                 " exceeds the supported maximum of 3");
}

SimulationCell::SimulationCell(std::size_t dim, std::span<const double> vectors)
    : boundary_(Boundary::Periodic), dim_(dim) {
    if (dim != 1 && dim != 2)
        throw std::runtime_error(
            "SimulationCell: periodic boundaries are implemented for dimension 1 "
            "and 2 only (got " +
            std::to_string(dim) +
            ").  A rigorous 3-D minimum image needs an LLL-reduced basis, which "
            "this milestone does not build.");
    if (vectors.size() != dim * dim)
        throw std::runtime_error(
            "SimulationCell: expected " + std::to_string(dim * dim) +
            " lattice-vector components, got " + std::to_string(vectors.size()));
    for (std::size_t k = 0; k < dim * dim; ++k) a_[k] = vectors[k];
    finalize_periodic();
}

// Reorder the rows of an axis-aligned basis so row i carries axis i.
// Returns false when the basis is not EXACTLY axis-aligned (some row has
// other than exactly one nonzero component, or two rows claim the same axis).
// A tiny nonzero shear is still part of the lattice: the orthorhombic fold
// would discard it and break invariance under primitive translations.
//
// This runs BEFORE Gauss reduction on purpose: reduction sorts rows by length,
// which would turn L = [100, 50] into rows {(0,50), (100,0)} and hide a plain
// rectangular cell from the orthorhombic fast path.
static bool axis_align(double* a, std::size_t dim) noexcept {
    std::size_t axis[SimulationCell::kMaxDim];
    bool        used[SimulationCell::kMaxDim] = {};
    for (std::size_t i = 0; i < dim; ++i) {
        std::size_t nonzero = 0, which = 0;
        for (std::size_t d = 0; d < dim; ++d)
            if (a[i * dim + d] != 0.0) {
                ++nonzero;
                which = d;
            }
        if (nonzero != 1 || used[which]) return false;
        axis[i]     = which;
        used[which] = true;
    }
    double reordered[SimulationCell::kMaxDim * SimulationCell::kMaxDim] = {};
    for (std::size_t i = 0; i < dim; ++i)
        for (std::size_t d = 0; d < dim; ++d) reordered[axis[i] * dim + d] = a[i * dim + d];
    for (std::size_t k = 0; k < dim * dim; ++k) a[k] = reordered[k];
    return true;
}

// Derive everything the folds need: a reduced basis, its inverse, the candidate
// shell, the orthorhombic fast path, the volume and the Wigner–Seitz inradius.
void SimulationCell::finalize_periodic() {
    ortho_ = axis_align(a_, dim_);
    if (!ortho_ && dim_ == 2) gauss_reduce_2d(a_);

    // Volume and (Aᵀ)⁻¹.  Only dim 1 and 2 reach here.
    if (dim_ == 1) {
        volume_ = std::abs(a_[0]);
        if (volume_ == 0.0)
            throw std::runtime_error("SimulationCell: lattice vector has zero length");
        ainv_[0] = 1.0 / a_[0];
    } else {
        const double det = a_[0] * a_[3] - a_[1] * a_[2];
        volume_          = std::abs(det);
        if (volume_ == 0.0)
            throw std::runtime_error(
                "SimulationCell: lattice vectors are singular (zero cell area)");
        // f = (Aᵀ)⁻¹ r, with A holding the primitive vectors as rows.
        const double inv_det = 1.0 / det;
        ainv_[0] = a_[3] * inv_det;
        ainv_[1] = -a_[2] * inv_det;
        ainv_[2] = -a_[1] * inv_det;
        ainv_[3] = a_[0] * inv_det;
    }

    // Reciprocal lattice in the 2π convention: b_i · a_j = 2π δ_ij.  Unused in
    // this milestone; it is what the Ewald k-vectors of Milestone 2 iterate.
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    for (std::size_t k = 0; k < dim_ * dim_; ++k) b_[k] = kTwoPi * ainv_[k];

    if (ortho_)
        for (std::size_t d = 0; d < dim_; ++d) {
            box_[d]     = std::abs(a_[d * dim_ + d]);
            inv_box_[d] = 1.0 / box_[d];
        }

    // Candidate translation shell {-1,0,1}^dim in Cartesian coordinates, with
    // the zero translation at index 0 (min_image seeds its running best with it
    // and scans from 1).  Only consulted on the general path.
    for (std::size_t d = 0; d < kMaxDim; ++d) cand_[d] = 0.0;
    n_cand_ = 1;
    if (!ortho_) {
        const int lo1 = (dim_ > 1) ? -1 : 0;
        const int hi1 = (dim_ > 1) ? 1 : 0;
        for (int n0 = -1; n0 <= 1; ++n0)
            for (int n1 = lo1; n1 <= hi1; ++n1) {
                if (n0 == 0 && n1 == 0) continue;
                const double n[2] = {static_cast<double>(n0), static_cast<double>(n1)};
                for (std::size_t d = 0; d < dim_; ++d) {
                    double s = 0.0;
                    for (std::size_t i = 0; i < dim_; ++i) s += n[i] * a_[i * dim_ + d];
                    cand_[n_cand_ * kMaxDim + d] = s;
                }
                ++n_cand_;
            }
    }
    // min_image folds every skewed cell with fold_bravais_2d (a 3×3 shell).
    if (!ortho_ && (dim_ != 2 || n_cand_ != 9))
        throw std::logic_error("SimulationCell: a non-orthogonal cell must be 2-D");

    // Wigner–Seitz inradius = ½ the shortest nonzero lattice vector.  Scan a
    // generous integer window so a mildly skewed basis cannot fool it.
    constexpr int span = 2;
    const int     lo1  = (dim_ > 1) ? -span : 0;
    const int     hi1  = (dim_ > 1) ? span : 0;
    double        shortest2 = 0.0;
    for (int n0 = -span; n0 <= span; ++n0)
        for (int n1 = lo1; n1 <= hi1; ++n1) {
            if (n0 == 0 && n1 == 0) continue;
            const double n[2] = {static_cast<double>(n0), static_cast<double>(n1)};
            double       r2   = 0.0;
            for (std::size_t d = 0; d < dim_; ++d) {
                double s = 0.0;
                for (std::size_t i = 0; i < dim_; ++i) s += n[i] * a_[i * dim_ + d];
                r2 += s * s;
            }
            if (shortest2 == 0.0 || r2 < shortest2) shortest2 = r2;
        }
    ws_radius_ = 0.5 * std::sqrt(shortest2);
}

void SimulationCell::min_image_batch(const double* __restrict xi,
                                     const double* __restrict xj, std::size_t nw,
                                     double* __restrict out, std::size_t w0,
                                     std::size_t w_end) const noexcept {
    if (boundary_ == Boundary::Open) {
        for (std::size_t d = 0; d < dim_; ++d) {
            const double* __restrict pi = xi + d * nw;
            const double* __restrict pj = xj + d * nw;
            double* __restrict o        = out + d * nw;
#pragma omp simd
            for (std::size_t w = w0; w < w_end; ++w) o[w] = pi[w] - pj[w];
        }
        return;
    }
    if (ortho_) {
        // Orthorhombic: the fold decouples, so keep the dimension loop outside
        // and the walker loop unit-stride and vectorized.
        for (std::size_t d = 0; d < dim_; ++d) {
            const double* __restrict pi = xi + d * nw;
            const double* __restrict pj = xj + d * nw;
            double* __restrict o        = out + d * nw;
            const double L    = box_[d];
            const double invL = inv_box_[d];
#pragma omp simd
            for (std::size_t w = w0; w < w_end; ++w) {
                const double del = pi[w] - pj[w];
                o[w]             = del - L * std::nearbyint(del * invL);
            }
        }
        return;
    }
    // General Bravais (always 2-D): the fold mixes components, so every
    // dimension is handled together per walker.  With fixed trip counts the
    // walker loop vectorizes; each lane runs exactly min_image's operations.
    const double* __restrict x0 = xi;
    const double* __restrict x1 = xi + nw;
    const double* __restrict y0 = xj;
    const double* __restrict y1 = xj + nw;
    double* __restrict o0 = out;
    double* __restrict o1 = out + nw;
#pragma omp simd
    for (std::size_t w = w0; w < w_end; ++w) {
        double d0 = x0[w] - y0[w];
        double d1 = x1[w] - y1[w];
        fold_bravais_2d(d0, d1);
        o0[w] = d0;
        o1[w] = d1;
    }
}

void SimulationCell::min_image_batch_from(const double* __restrict xi,
                                          const double* __restrict c, std::size_t nw,
                                          double* __restrict out, std::size_t w0,
                                          std::size_t w_end) const noexcept {
    // The three branches of min_image_batch with the second particle replaced
    // by the fixed point c.
    if (boundary_ == Boundary::Open || ortho_) {
        const bool fold = boundary_ != Boundary::Open;
        for (std::size_t d = 0; d < dim_; ++d) {
            const double* __restrict pi = xi + d * nw;
            double* __restrict o        = out + d * nw;
            const double cd = c[d];
            if (!fold) {
#pragma omp simd
                for (std::size_t w = w0; w < w_end; ++w) o[w] = pi[w] - cd;
                continue;
            }
            const double L    = box_[d];
            const double invL = inv_box_[d];
#pragma omp simd
            for (std::size_t w = w0; w < w_end; ++w) {
                const double del = pi[w] - cd;
                o[w]             = del - L * std::nearbyint(del * invL);
            }
        }
        return;
    }
    const double* __restrict x0 = xi;
    const double* __restrict x1 = xi + nw;
    const double c0 = c[0], c1 = c[1];
    double* __restrict o0 = out;
    double* __restrict o1 = out + nw;
#pragma omp simd
    for (std::size_t w = w0; w < w_end; ++w) {
        double d0 = x0[w] - c0;
        double d1 = x1[w] - c1;
        fold_bravais_2d(d0, d1);
        o0[w] = d0;
        o1[w] = d1;
    }
}

}  // namespace alea
