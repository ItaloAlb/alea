// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cmath>
#include <cstddef>
#include <span>

namespace alea {

// Simulation cell geometry: either open (no cell at all) or periodic on a
// Bravais lattice.
//
// ── Why this exists ─────────────────────────────────────────────────────────
// The thermodynamic limit (N → ∞, V → ∞ at fixed ρ = N/V) is reached by
// simulating N particles in a finite cell tiled periodically.  The cell removes
// the surface (V → ∞); an Ewald sum over the periodic images supplies N → ∞.
//
// ── Wrapping is NOT the minimum image convention ────────────────────────────
// Folding a *position* into the cell and folding a *separation* into the
// Wigner–Seitz cell are different operations, and the first does not imply the
// second.  With L = 10 and both particles legally inside [0, 10):
//
//     x_i = 0.5, x_j = 9.5   →   raw |Δ| = 9.0,  but the nearest image of j
//                                sits at −0.5, so the true r is 1.0.
//
// No choice of wrapped coordinate fixes this, because the correct image of j
// depends on which partner it is paired with.  Equivalently: "use the copy
// inside the box" is not invariant under a shift of the cell origin, and the
// energy cannot depend on where the boundary was drawn.  min_image() is the
// unique translation-invariant answer.
//
// ── Relation to Ewald ───────────────────────────────────────────────────────
// The minimum image is not a placeholder for Ewald — it IS Ewald's real-space
// half.  Ewald sums V over every lattice translation L; with the splitting
// parameter chosen so the short-range part dies beyond the cell inradius, the
// real-space sum Σ_L V_short(|Δ + L|) collapses to the single nearest image.
//
// ── Conventions ─────────────────────────────────────────────────────────────
//   a_[i*dim + d]   d-th Cartesian component of the i-th primitive vector.
//                   A position is r_d = Σ_i f_i a_[i*dim + d], i.e. r = Aᵀf.
//   ainv_           rows of (Aᵀ)⁻¹, so the fractional coordinates are
//                   f_i = Σ_d ainv_[i*dim + d] r_d.
//   b_[i*dim + d]   reciprocal lattice, b_i · a_j = 2π δ_ij  (= 2π·ainv_).
//                   Stored for the Ewald k-vectors of Milestone 2.
class SimulationCell {
public:
    enum class Boundary { Open, Periodic };

    static constexpr std::size_t kMaxDim  = 3;
    static constexpr std::size_t kMaxCand = 27;  // 3^kMaxDim

    // Open cell (no periodicity).  The default is an open 0-dimensional cell so
    // a default-constructed WalkerSet keeps today's open-boundary behaviour.
    SimulationCell() = default;
    explicit SimulationCell(std::size_t dim);

    // Periodic cell from primitive vectors, row-major [i*dim + d].
    // Throws if the vectors are singular, or if dim is not 1 or 2 (a rigorous
    // 3-D minimum image needs an LLL-reduced basis; this project is 2-D).
    SimulationCell(std::size_t dim, std::span<const double> vectors);

    [[nodiscard]] Boundary    boundary() const noexcept { return boundary_; }
    [[nodiscard]] bool        is_periodic() const noexcept {
        return boundary_ == Boundary::Periodic;
    }
    [[nodiscard]] std::size_t dim() const noexcept { return dim_; }

    // Cell volume (area in 2-D, length in 1-D) = |det A|.  Zero when open.
    [[nodiscard]] double volume() const noexcept { return volume_; }

    // Inradius of the Wigner–Seitz cell = ½ min |L| over nonzero lattice
    // vectors.  The largest r for which a minimum-image pair distance is
    // unambiguous, hence the natural cap on g(r) and on an Ewald real-space
    // cutoff.  Zero when open.
    [[nodiscard]] double ws_radius() const noexcept { return ws_radius_; }

    // True when the primitive vectors are mutually orthogonal, enabling the
    // dimension-decoupled per-axis fold.
    [[nodiscard]] bool orthorhombic() const noexcept { return ortho_; }

    [[nodiscard]] std::span<const double> vectors() const noexcept {
        return {a_, dim_ * dim_};
    }
    [[nodiscard]] std::span<const double> reciprocal() const noexcept {
        return {b_, dim_ * dim_};
    }

    // ── Minimum image: fold a separation into the Wigner–Seitz cell ─────────
    // In-place on a single dim-component displacement.  No-op when open.
    // Branch-free in the candidate search so it vectorizes when inlined into a
    // loop over walkers.
    void min_image(double* __restrict del) const noexcept {
        if (boundary_ == Boundary::Open) return;
        if (ortho_) {
            for (std::size_t d = 0; d < dim_; ++d)
                del[d] -= box_[d] * std::nearbyint(del[d] * inv_box_[d]);
            return;
        }
        // General Bravais: round in fractional coordinates to land in the
        // primitive parallelepiped, then scan the surrounding lattice shell.
        // Rounding alone can miss the true nearest image on a skewed basis;
        // on the Gauss-reduced basis built in the ctor, the 3^dim shell is
        // sufficient.  min_image_batch runs these operations on scalar
        // operands (fold_bravais_2d); test_simulation_cell requires the two to
        // agree bit for bit.
        double f[kMaxDim];
        for (std::size_t i = 0; i < dim_; ++i) {
            double s = 0.0;
            for (std::size_t d = 0; d < dim_; ++d) s += ainv_[i * dim_ + d] * del[d];
            f[i] = std::nearbyint(s);
        }
        for (std::size_t d = 0; d < dim_; ++d) {
            double s = 0.0;
            for (std::size_t i = 0; i < dim_; ++i) s += f[i] * a_[i * dim_ + d];
            del[d] -= s;
        }

        double best[kMaxDim];
        double best2 = 0.0;
        for (std::size_t d = 0; d < dim_; ++d) {
            best[d] = del[d];
            best2 += del[d] * del[d];
        }
        // cand_[0] is the zero translation, already accounted for above.
        for (std::size_t c = 1; c < n_cand_; ++c) {
            double t[kMaxDim];
            double r2 = 0.0;
            for (std::size_t d = 0; d < dim_; ++d) {
                t[d] = del[d] + cand_[c * kMaxDim + d];
                r2 += t[d] * t[d];
            }
            // Branch-free select → vminpd/blend rather than a data-dependent jump.
            const double take = (r2 < best2) ? 1.0 : 0.0;
            best2 += take * (r2 - best2);
            for (std::size_t d = 0; d < dim_; ++d) best[d] += take * (t[d] - best[d]);
        }
        for (std::size_t d = 0; d < dim_; ++d) del[d] = best[d];
    }

    // Squared minimum-image distance of a single displacement (fold + norm).
    [[nodiscard]] double min_image_dist2(double* __restrict del) const noexcept {
        min_image(del);
        double r2 = 0.0;
        for (std::size_t d = 0; d < dim_; ++d) r2 += del[d] * del[d];
        return r2;
    }

    // ── Wrapping: fold a position into the primitive cell ───────────────────
    // Centred convention: fractional coordinates land in [−½, ½), matching the
    // open-boundary [−L/2, L/2] placement the code already uses.  Pure hygiene
    // — every observable is already invariant under a lattice translation once
    // separations are folded — but it keeps coordinates bounded and lets the
    // one-body terms read positions modulo the cell.
    void wrap(double* __restrict r) const noexcept {
        if (boundary_ == Boundary::Open) return;
        if (ortho_) {
            for (std::size_t d = 0; d < dim_; ++d)
                r[d] -= box_[d] * std::nearbyint(r[d] * inv_box_[d]);
            return;
        }
        double f[kMaxDim];
        for (std::size_t i = 0; i < dim_; ++i) {
            double s = 0.0;
            for (std::size_t d = 0; d < dim_; ++d) s += ainv_[i * dim_ + d] * r[d];
            f[i] = std::nearbyint(s);
        }
        for (std::size_t d = 0; d < dim_; ++d) {
            double s = 0.0;
            for (std::size_t i = 0; i < dim_; ++i) s += f[i] * a_[i * dim_ + d];
            r[d] -= s;
        }
    }

    // Fractional coordinates f_i of a position, r = Σ_i f_i a_i.  Meaningful
    // only for a periodic cell; used for cell-shaped histogram binning.
    void fractional(const double* __restrict r, double* __restrict f) const noexcept {
        for (std::size_t i = 0; i < dim_; ++i) {
            double s = 0.0;
            for (std::size_t d = 0; d < dim_; ++d) s += ainv_[i * dim_ + d] * r[d];
            f[i] = s;
        }
    }

    // ── Batched fold of one particle pair over a walker range ───────────────
    // xi, xj: SoA bases of the two particles' dim-blocks, element [d*nw + w].
    // out:    [dim * nw], the minimum-image Δ = r_i − r_j, same layout.
    // Only [w0, w_end) is written, so concurrent crowds on disjoint walker
    // ranges are race-free.
    void min_image_batch(const double* __restrict xi, const double* __restrict xj,
                         std::size_t nw, double* __restrict out, std::size_t w0,
                         std::size_t w_end) const noexcept;

    // Same fold for a particle against one fixed point c [dim] (an orbital
    // centre): out[d*nw + w] = minimum-image Δ = r_i − c over [w0, w_end).
    // Every lane runs min_image's operations, so it agrees bit for bit with
    // min_image applied to r_i − c (test_simulation_cell).
    void min_image_batch_from(const double* __restrict xi, const double* __restrict c,
                              std::size_t nw, double* __restrict out, std::size_t w0,
                              std::size_t w_end) const noexcept;

private:
    // min_image's general Bravais fold for the only skewed case the ctor
    // admits (2-D, 3×3 shell), on scalar operands: without local arrays the
    // walker loop of min_image_batch vectorizes.  Same operations in the
    // same order as min_image's loops (s = 0, s += A_i0 x0, s += A_i1 x1, ...),
    // so the results agree bit for bit (checked in test_simulation_cell).
    void fold_bravais_2d(double& x0, double& x1) const noexcept {
        double s0 = 0.0;
        s0 += ainv_[0] * x0;
        s0 += ainv_[1] * x1;
        double s1 = 0.0;
        s1 += ainv_[2] * x0;
        s1 += ainv_[3] * x1;
        const double f0 = std::nearbyint(s0);
        const double f1 = std::nearbyint(s1);
        double t0 = 0.0;
        t0 += f0 * a_[0];
        t0 += f1 * a_[2];
        double t1 = 0.0;
        t1 += f0 * a_[1];
        t1 += f1 * a_[3];
        x0 -= t0;
        x1 -= t1;

        double best0 = x0, best1 = x1;
        double best2 = 0.0;
        best2 += x0 * x0;
        best2 += x1 * x1;
        // cand_[0] is the zero translation, already accounted for above.
        for (std::size_t c = 1; c < 9; ++c) {
            const double y0 = x0 + cand_[c * kMaxDim + 0];
            double r2 = 0.0;
            r2 += y0 * y0;
            const double y1 = x1 + cand_[c * kMaxDim + 1];
            r2 += y1 * y1;
            // Branch-free select → vminpd/blend rather than a data-dependent jump.
            const double take = (r2 < best2) ? 1.0 : 0.0;
            best2 += take * (r2 - best2);
            best0 += take * (y0 - best0);
            best1 += take * (y1 - best1);
        }
        x0 = best0;
        x1 = best1;
    }

    void finalize_periodic();

    Boundary    boundary_ = Boundary::Open;
    std::size_t dim_      = 0;

    double a_[kMaxDim * kMaxDim]{};
    double ainv_[kMaxDim * kMaxDim]{};
    double b_[kMaxDim * kMaxDim]{};
    double cand_[kMaxCand * kMaxDim]{};  // Cartesian candidate translations
    std::size_t n_cand_ = 1;

    double box_[kMaxDim]{};      // orthorhombic fast path: per-axis lengths
    double inv_box_[kMaxDim]{};
    bool   ortho_ = false;

    double volume_    = 0.0;
    double ws_radius_ = 0.0;
};

// Compile-time boundary policies.  Hot kernels are templated on one of these so
// the open path keeps its exact current shape and codegen — OpenBC carries no
// fold at all, so existing open-boundary runs stay bit-identical.
struct OpenBC {
    static constexpr bool periodic = false;
};
struct PeriodicBC {
    static constexpr bool periodic = true;
};

}  // namespace alea
