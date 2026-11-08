// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "../population/simulation_cell.h"
#include "wavefunction_component.h"   // Parameter

namespace alea {

// A set of n single-particle orbitals φ_0..φ_{n-1}, evaluated at ONE particle's
// position across all walkers (SoA, walker index fastest).  A Slater determinant
// uses it to fill one matrix row: row j is { φ_i(r_j) }_i.
//
// All buffers use the project's SoA layout:
//   pos  : [dim * nw]      pos[d * nw + w]            = (r_w)_d
//   val  : [n * nw]        val[i * nw + w]            = φ_i(r_w)
//   grad : [n * dim * nw]  grad[(i*dim + d)*nw + w]   = ∂_d φ_i(r_w)
//   lap  : [n * dim * nw]  lap[(i*dim + d)*nw + w]    = ∂²_d φ_i(r_w)  (per dimension)
class OrbitalSet {
public:
    virtual ~OrbitalSet() = default;

    // Number of orbitals n.
    [[nodiscard]] virtual std::size_t size() const = 0;

    // Evaluate every orbital and its gradient / per-dimension laplacian at the
    // particle position `pos` over the walker range [w_begin, w_end).  Outputs
    // are OVERWRITTEN (not accumulated).  Stride is the global nw = pos.size()/dim.
    virtual void evaluate(std::span<const double> pos, std::size_t dim,
                          std::span<double> val, std::span<double> grad,
                          std::span<double> lap, std::size_t w_begin,
                          std::size_t w_end) const = 0;

    // ── Variational parameters of the orbitals ─────────────────────────────
    // These set the NODES: ψ = D·e^J with J positive, so no Jastrow parameter
    // can move a fixed-node energy and these can.  Default is none, which is
    // what PlaneWaveOrbitalSet wants (its k come from the cell).
    [[nodiscard]] virtual std::size_t n_parameters() const noexcept { return 0; }

    // Descriptor of parameter k.  is_fixed/sign_locked are the orbital's
    // defaults; the config layer may unfix them via SlaterDeterminant.
    [[nodiscard]] virtual Parameter parameter(std::size_t k) const {
        (void)k;
        throw std::out_of_range("orbital set has no parameters");
    }

    virtual void set_parameter(std::size_t k, double value) {
        (void)k;
        (void)value;
        throw std::out_of_range("orbital set has no parameters");
    }

    // ∂φ_i/∂p_k and its gradient/laplacian, mirroring evaluate()'s layout with a
    // parameter axis in front:
    //   dval [(k*n + i)*nw + w]            dgrad/dlap [((k*n + i)*dim + d)*nw + w]
    // Outputs are overwritten.  dgrad/dlap may be empty when only the value
    // derivative is wanted (the ∂lnψ/∂p path); dval never is.  Measurement
    // frequency only — never inside the Metropolis loop.
    virtual void evaluate_parameter_derivatives(std::span<const double> pos,
                                                std::size_t dim,
                                                std::span<double> dval,
                                                std::span<double> dgrad,
                                                std::span<double> dlap,
                                                std::size_t w_begin,
                                                std::size_t w_end) const {
        (void)pos; (void)dim; (void)dval; (void)dgrad; (void)dlap;
        (void)w_begin; (void)w_end;
    }

    // Largest neglected periodic-image term of any orbital relative to its
    // peak, for orbitals made periodic by an approximation; 0 when they are
    // exactly periodic or the cell is open.  A consumer that needs the nodes
    // of the exact lattice-summed orbitals bounds its error with it.
    [[nodiscard]] virtual double periodic_fold_bound() const noexcept { return 0.0; }

    // ── Full Hessian (backflow) ─────────────────────────────────────────────
    // A backflow determinant evaluates the orbitals at quasi-coordinates whose
    // Jacobian mixes Cartesian components, so the Laplacian of log ψ needs the
    // mixed second derivatives ∂_a∂_b φ, not only evaluate()'s ∂²_d φ.  Same
    // layouts as evaluate(), with
    //   hess : [n * dim * dim * nw]  hess[((i*dim + a)*dim + b)*nw + w] = ∂_a ∂_b φ_i
    // Outputs are overwritten.  Orbital sets that do not provide it say so,
    // and their determinants refuse backflow at build time.
    [[nodiscard]] virtual bool supports_hessian() const noexcept { return false; }
    virtual void evaluate_hessian(std::span<const double> pos, std::size_t dim,
                                  std::span<double> val, std::span<double> grad,
                                  std::span<double> hess, std::size_t w_begin,
                                  std::size_t w_end) const {
        (void)pos; (void)dim; (void)val; (void)grad; (void)hess; (void)w_begin; (void)w_end;
        throw std::logic_error("orbital set has no Hessian (supports_hessian() is false)");
    }

    // ── Backflow curvature term ─────────────────────────────────────────────
    // ∂(∇² ln ψ)/∂p of a backflow determinant moves the quasi-coordinates with
    // p, so it differentiates the Hessian once more along x:
    //   third : [n * dim³ * nw]  third[(((i*dim + a)*dim + b)*dim + c)*nw + w] = ∂_a∂_b∂_c φ_i
    // and, for the orbitals' own parameters, needs the full Hessian's derivative:
    //   dval [(k*n + i)*nw + w], dgrad [((k*n + i)*dim + a)*nw + w],
    //   dhess [(((k*n + i)*dim + a)*dim + b)*nw + w] = ∂(∂_a∂_b φ_i)/∂p_k.
    // Outputs are overwritten.  Measurement frequency only (the optimiser).
    [[nodiscard]] virtual bool supports_third_derivatives() const noexcept { return false; }
    virtual void evaluate_third_derivatives(std::span<const double> pos, std::size_t dim,
                                            std::span<double> third, std::size_t w_begin,
                                            std::size_t w_end) const {
        (void)pos; (void)dim; (void)third; (void)w_begin; (void)w_end;
        throw std::logic_error("orbital set has no third derivatives");
    }
    // A set without parameters needs nothing, so it supports this trivially.
    [[nodiscard]] virtual bool supports_parameter_hessian() const noexcept {
        return n_parameters() == 0;
    }
    virtual void evaluate_parameter_hessian(std::span<const double> pos, std::size_t dim,
                                            std::span<double> dval, std::span<double> dgrad,
                                            std::span<double> dhess, std::size_t w_begin,
                                            std::size_t w_end) const {
        (void)pos; (void)dim; (void)dval; (void)dgrad; (void)dhess; (void)w_begin; (void)w_end;
        if (n_parameters() != 0)
            throw std::logic_error("orbital set has no parameter Hessian");
    }

    [[nodiscard]] virtual std::unique_ptr<OrbitalSet> clone() const = 0;
};

// Isotropic Gaussian orbitals at fixed centers:
//   φ_i(r) = exp(-α_i |r - c_i|²)
// with closed-form derivatives (Δ_d = r_d - c_{i,d}):
//   ∂_d φ_i  = φ_i · (-2 α_i Δ_d)
//   ∂²_d φ_i = φ_i · (4 α_i² Δ_d² - 2 α_i)
//
// Centers are fixed (the crystal's sites); the width is variational and, being
// inside the determinant, nodal.  A set with all widths EQUAL exposes the shared
// one as "alpha" (sign_locked: α > 0 by construction).  Per-orbital widths are
// not one shared knob, so such a set exposes nothing.  Equality is tested
// exactly, which is what both construction paths produce.
//
// ── Periodic cells ──────────────────────────────────────────────────────────
// A Gaussian at a fixed center is NOT periodic, and walker positions are wrapped
// into the cell, so the raw difference r − c jumps by a lattice vector when a
// walker crosses the boundary: a site near the cell edge would have its orbital
// truncated by the wrap and ψ would be discontinuous.  Passing a periodic cell
// makes the difference a MINIMUM-IMAGE one, which restores periodicity exactly.
//
// What that approximates is the lattice sum Σ_L exp(-α|r - c + L|²) by its
// largest term.  The largest neglected term is exp(-α R_ws²) relative to the
// orbital's peak, R_ws the inradius of the simulation cell's Wigner–Seitz cell
// (periodic_fold_bound()).  Small, but not always negligible: 2.9e-7 for
// N = 37 at r_s = 40 with the broad α = 0.071 r_s^{-3/2} of Drummond & Needs,
// and far below double precision for larger cells or narrower orbitals.  The
// fold's kink at the WS boundary is suppressed by the same factor.
//
// The default cell is open, where min_image() is a no-op, so the open path keeps
// its exact previous arithmetic.
class GaussianOrbitalSet final : public OrbitalSet {
public:
    // centers: [n * dim], center i dimension d at centers[i*dim + d].
    // alphas : [n].
    GaussianOrbitalSet(std::size_t dim, std::vector<double> centers,
                       std::vector<double> alphas,
                       const SimulationCell& cell = SimulationCell{})
        : dim_(dim), centers_(std::move(centers)), alphas_(std::move(alphas)),
          cell_(cell),
          uniform_width_(!alphas_.empty() &&
                         std::equal(alphas_.begin() + 1, alphas_.end(),
                                    alphas_.begin())) {}

    [[nodiscard]] std::size_t size() const override { return alphas_.size(); }

    // Three walker passes per orbital: the batched fold writes Δ into orbital
    // i's gradient planes [(i*dim + d)*nw + w], a scalar pass takes exp (libm
    // has no vector exp without -ffast-math), and the last pass overwrites the
    // planes with ∂_d φ_i.  The batched fold vectorizes over walkers and agrees
    // with min_image bit for bit; calling the scalar min_image per orbital and
    // walker costs ~10x more and dominated the crystal DMC profile.
    void evaluate(std::span<const double> pos, std::size_t dim,
                  std::span<double> val, std::span<double> grad,
                  std::span<double> lap, std::size_t w_begin,
                  std::size_t w_end) const override {
        const std::size_t n  = alphas_.size();
        const std::size_t nw = dim == 0 ? 0 : pos.size() / dim;
        for (std::size_t i = 0; i < n; ++i) {
            const double alpha = alphas_[i];
            double* __restrict del = grad.data() + i * dim * nw;   // [dim * nw], Δ then ∂φ
            double* __restrict phi = val.data() + i * nw;          // [nw]
            // The default open cell is dimensionless, so the open difference is
            // formed here rather than by the cell.
            if (cell_.is_periodic()) {
                cell_.min_image_batch_from(pos.data(), centers_.data() + i * dim, nw, del,
                                           w_begin, w_end);
            } else {
                for (std::size_t d = 0; d < dim; ++d) {
                    const double c = centers_[i * dim + d];
#pragma omp simd
                    for (std::size_t w = w_begin; w < w_end; ++w)
                        del[d * nw + w] = pos[d * nw + w] - c;
                }
            }
            for (std::size_t w = w_begin; w < w_end; ++w) {
                double r2 = 0.0;
                for (std::size_t d = 0; d < dim; ++d) r2 += del[d * nw + w] * del[d * nw + w];
                phi[w] = std::exp(-alpha * r2);
            }
            for (std::size_t d = 0; d < dim; ++d) {
                double* __restrict g = del + d * nw;
                double* __restrict l = lap.data() + (i * dim + d) * nw;
#pragma omp simd
                for (std::size_t w = w_begin; w < w_end; ++w) {
                    const double delta = g[w];
                    g[w] = phi[w] * (-2.0 * alpha * delta);
                    l[w] = phi[w] * (4.0 * alpha * alpha * delta * delta - 2.0 * alpha);
                }
            }
        }
    }

    // ── The shared width as a variational parameter ─────────────────────────

    [[nodiscard]] std::size_t n_parameters() const noexcept override {
        return uniform_width_ ? 1 : 0;
    }

    [[nodiscard]] Parameter parameter(std::size_t k) const override {
        if (k >= n_parameters())
            throw std::out_of_range("GaussianOrbitalSet: no parameter " +
                                    std::to_string(k));
        // Fixed by default: a determinant that has always been static must not
        // start moving just because the parameter became reachable.  The config
        // layer opts in.  Sign-locked always: α ≤ 0 is not a Gaussian.
        return Parameter{"alpha", alphas_[0], /*is_fixed=*/true,
                         /*sign_locked=*/true};
    }

    void set_parameter(std::size_t k, double value) override {
        if (k >= n_parameters())
            throw std::out_of_range("GaussianOrbitalSet: no parameter " +
                                    std::to_string(k));
        std::fill(alphas_.begin(), alphas_.end(), value);
    }

    // With Δ the minimum-image separation and φ = exp(-α|Δ|²):
    //   ∂φ/∂α        = -|Δ|² φ
    //   ∂(∂_d φ)/∂α  = -2Δ_d φ (1 - α|Δ|²)
    //   ∂(∂²_d φ)/∂α = φ [8αΔ_d² - 2 - |Δ|²(4α²Δ_d² - 2α)]
    // Δ must be the SAME fold the value uses, or the derivative disagrees with
    // the function it differentiates for walkers across the cell boundary.
    void evaluate_parameter_derivatives(std::span<const double> pos,
                                        std::size_t dim,
                                        std::span<double> dval,
                                        std::span<double> dgrad,
                                        std::span<double> dlap,
                                        std::size_t w_begin,
                                        std::size_t w_end) const override {
        if (!uniform_width_) return;
        const std::size_t n  = alphas_.size();
        const std::size_t nw = dim == 0 ? 0 : pos.size() / dim;
        const double alpha = alphas_[0];
        const bool want_derivs = !dgrad.empty();
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t w = w_begin; w < w_end; ++w) {
                double del[SimulationCell::kMaxDim] = {};
                double r2 = 0.0;
                for (std::size_t d = 0; d < dim; ++d)
                    del[d] = pos[d * nw + w] - centers_[i * dim + d];
                cell_.min_image(del);
                for (std::size_t d = 0; d < dim; ++d) r2 += del[d] * del[d];
                const double phi = std::exp(-alpha * r2);
                dval[i * nw + w] = -r2 * phi;
                if (!want_derivs) continue;
                for (std::size_t d = 0; d < dim; ++d) {
                    const double dd = del[d];
                    dgrad[(i * dim + d) * nw + w] =
                        -2.0 * dd * phi * (1.0 - alpha * r2);
                    dlap[(i * dim + d) * nw + w] =
                        phi * (8.0 * alpha * dd * dd - 2.0
                               - r2 * (4.0 * alpha * alpha * dd * dd - 2.0 * alpha));
                }
            }
        }
    }

    // ∂_a∂_b φ_i = φ_i (4α_i² Δ_a Δ_b − 2α_i δ_ab), Δ the same minimum-image
    // separation evaluate() uses.  Scalar over walkers: backflow calls it for
    // one configuration at a time.
    [[nodiscard]] bool supports_hessian() const noexcept override { return true; }
    void evaluate_hessian(std::span<const double> pos, std::size_t dim,
                          std::span<double> val, std::span<double> grad,
                          std::span<double> hess, std::size_t w_begin,
                          std::size_t w_end) const override {
        const std::size_t n  = alphas_.size();
        const std::size_t nw = dim == 0 ? 0 : pos.size() / dim;
        for (std::size_t i = 0; i < n; ++i) {
            const double alpha = alphas_[i];
            for (std::size_t w = w_begin; w < w_end; ++w) {
                double del[SimulationCell::kMaxDim] = {};
                double r2 = 0.0;
                for (std::size_t d = 0; d < dim; ++d)
                    del[d] = pos[d * nw + w] - centers_[i * dim + d];
                cell_.min_image(del);
                for (std::size_t d = 0; d < dim; ++d) r2 += del[d] * del[d];
                const double phi = std::exp(-alpha * r2);
                val[i * nw + w] = phi;
                for (std::size_t a = 0; a < dim; ++a) {
                    grad[(i * dim + a) * nw + w] = -2.0 * alpha * del[a] * phi;
                    for (std::size_t b = 0; b < dim; ++b)
                        hess[((i * dim + a) * dim + b) * nw + w] =
                            phi * (4.0 * alpha * alpha * del[a] * del[b] -
                                   (a == b ? 2.0 * alpha : 0.0));
                }
            }
        }
    }

    // ∂_a∂_b∂_c φ_i = φ_i [−8α³ Δ_aΔ_bΔ_c + 4α² (δ_ab Δ_c + δ_ac Δ_b + δ_bc Δ_a)],
    // on the same minimum-image Δ.
    [[nodiscard]] bool supports_third_derivatives() const noexcept override { return true; }
    void evaluate_third_derivatives(std::span<const double> pos, std::size_t dim,
                                    std::span<double> third, std::size_t w_begin,
                                    std::size_t w_end) const override {
        const std::size_t n  = alphas_.size();
        const std::size_t nw = dim == 0 ? 0 : pos.size() / dim;
        for (std::size_t i = 0; i < n; ++i) {
            const double alpha = alphas_[i];
            for (std::size_t w = w_begin; w < w_end; ++w) {
                double del[SimulationCell::kMaxDim] = {};
                double r2 = 0.0;
                for (std::size_t d = 0; d < dim; ++d)
                    del[d] = pos[d * nw + w] - centers_[i * dim + d];
                cell_.min_image(del);
                for (std::size_t d = 0; d < dim; ++d) r2 += del[d] * del[d];
                const double phi = std::exp(-alpha * r2);
                const double a2 = 4.0 * alpha * alpha, a3 = -8.0 * alpha * alpha * alpha;
                for (std::size_t a = 0; a < dim; ++a)
                    for (std::size_t b = 0; b < dim; ++b)
                        for (std::size_t c = 0; c < dim; ++c) {
                            const double delta = (a == b ? del[c] : 0.0) +
                                                 (a == c ? del[b] : 0.0) +
                                                 (b == c ? del[a] : 0.0);
                            third[(((i * dim + a) * dim + b) * dim + c) * nw + w] =
                                phi * (a3 * del[a] * del[b] * del[c] + a2 * delta);
                        }
            }
        }
    }

    // ∂(∂_a∂_b φ)/∂α = −|Δ|² ∂_a∂_b φ + φ (8α Δ_aΔ_b − 2δ_ab), with the value
    // and gradient derivatives of evaluate_parameter_derivatives().
    [[nodiscard]] bool supports_parameter_hessian() const noexcept override { return true; }
    void evaluate_parameter_hessian(std::span<const double> pos, std::size_t dim,
                                    std::span<double> dval, std::span<double> dgrad,
                                    std::span<double> dhess, std::size_t w_begin,
                                    std::size_t w_end) const override {
        if (!uniform_width_) return;
        const std::size_t n  = alphas_.size();
        const std::size_t nw = dim == 0 ? 0 : pos.size() / dim;
        const double alpha = alphas_[0];
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t w = w_begin; w < w_end; ++w) {
                double del[SimulationCell::kMaxDim] = {};
                double r2 = 0.0;
                for (std::size_t d = 0; d < dim; ++d)
                    del[d] = pos[d * nw + w] - centers_[i * dim + d];
                cell_.min_image(del);
                for (std::size_t d = 0; d < dim; ++d) r2 += del[d] * del[d];
                const double phi = std::exp(-alpha * r2);
                dval[i * nw + w] = -r2 * phi;
                for (std::size_t a = 0; a < dim; ++a) {
                    dgrad[(i * dim + a) * nw + w] = -2.0 * del[a] * phi * (1.0 - alpha * r2);
                    for (std::size_t b = 0; b < dim; ++b) {
                        const double kron = a == b ? 1.0 : 0.0;
                        const double hess = phi * (4.0 * alpha * alpha * del[a] * del[b] -
                                                   2.0 * alpha * kron);
                        dhess[((i * dim + a) * dim + b) * nw + w] =
                            -r2 * hess + phi * (8.0 * alpha * del[a] * del[b] - 2.0 * kron);
                    }
                }
            }
        }
    }

    // exp(-α_min R_ws²): the minimum-image fold's largest neglected image.
    [[nodiscard]] double periodic_fold_bound() const noexcept override {
        if (!cell_.is_periodic() || alphas_.empty()) return 0.0;
        const double alpha_min = *std::min_element(alphas_.begin(), alphas_.end());
        return std::exp(-alpha_min * cell_.ws_radius() * cell_.ws_radius());
    }

    [[nodiscard]] std::unique_ptr<OrbitalSet> clone() const override {
        return std::make_unique<GaussianOrbitalSet>(dim_, centers_, alphas_, cell_);
    }

private:
    std::size_t         dim_;
    std::vector<double> centers_;  // [n * dim]
    std::vector<double> alphas_;   // [n]
    SimulationCell      cell_;     // open by default → min_image is a no-op
    bool                uniform_width_;  // all alphas_ equal → one parameter
};

// Real plane-wave orbitals on the cell's reciprocal lattice:
//   φ_i(r) = cos(k_i · r)   or   sin(k_i · r)
//
// Derivatives are exact and need no fold: k_i · a_j = 2π·integer for every cell
// vector a_j, so each orbital is already periodic and invariant under the wrap.
//
//   cos:  ∂_d = -k_d sin(k·r),  ∂²_d = -k_d² cos(k·r)
//   sin:  ∂_d = +k_d cos(k·r),  ∂²_d = -k_d² sin(k·r)
//
// ── Why real cos/sin instead of complex e^{ik·r} ────────────────────────────
// A filled shell contains k and −k together, and the 2×2 change of basis from
// {e^{ik·r}, e^{-ik·r}} to {cos, sin} is a constant matrix, so it changes the
// determinant only by an overall constant factor — the same state, with real
// arithmetic and no complex determinant machinery.  The k = 0 orbital is the
// constant cos(0) = 1, and its zero gradient/laplacian fall out of the formulas.
//
// Each orbital is one row-uniform pass over walkers: the cos/sin choice is a
// per-orbital branch OUTSIDE the walker loop, so the inner loop stays uniform.
class PlaneWaveOrbitalSet final : public OrbitalSet {
public:
    // kvectors: [n * dim].  is_sin: [n], nonzero → sine, zero → cosine.
    PlaneWaveOrbitalSet(std::size_t dim, std::vector<double> kvectors,
                        std::vector<char> is_sin)
        : dim_(dim), k_(std::move(kvectors)), is_sin_(std::move(is_sin)) {}

    [[nodiscard]] std::size_t size() const override { return is_sin_.size(); }

    void evaluate(std::span<const double> pos, std::size_t dim,
                  std::span<double> val, std::span<double> grad,
                  std::span<double> lap, std::size_t w_begin,
                  std::size_t w_end) const override {
        const std::size_t n  = is_sin_.size();
        const std::size_t nw = dim == 0 ? 0 : pos.size() / dim;
        for (std::size_t i = 0; i < n; ++i) {
            const double* ki = k_.data() + i * dim;
            // The cos/sin choice is hoisted out of the walker loop: the inner
            // loop is then branch-free and uniform across walkers.
            if (is_sin_[i]) {
                for (std::size_t w = w_begin; w < w_end; ++w) {
                    double phase = 0.0;
                    for (std::size_t d = 0; d < dim; ++d)
                        phase += ki[d] * pos[d * nw + w];
                    const double s = std::sin(phase);
                    const double c = std::cos(phase);
                    val[i * nw + w] = s;
                    for (std::size_t d = 0; d < dim; ++d) {
                        grad[(i * dim + d) * nw + w] = ki[d] * c;
                        lap[(i * dim + d) * nw + w]  = -ki[d] * ki[d] * s;
                    }
                }
            } else {
                for (std::size_t w = w_begin; w < w_end; ++w) {
                    double phase = 0.0;
                    for (std::size_t d = 0; d < dim; ++d)
                        phase += ki[d] * pos[d * nw + w];
                    const double s = std::sin(phase);
                    const double c = std::cos(phase);
                    val[i * nw + w] = c;
                    for (std::size_t d = 0; d < dim; ++d) {
                        grad[(i * dim + d) * nw + w] = -ki[d] * s;
                        lap[(i * dim + d) * nw + w]  = -ki[d] * ki[d] * c;
                    }
                }
            }
        }
    }

    // cos: ∂_a∂_b = −k_a k_b cos(k·r);  sin: ∂_a∂_b = −k_a k_b sin(k·r).
    [[nodiscard]] bool supports_hessian() const noexcept override { return true; }
    void evaluate_hessian(std::span<const double> pos, std::size_t dim,
                          std::span<double> val, std::span<double> grad,
                          std::span<double> hess, std::size_t w_begin,
                          std::size_t w_end) const override {
        const std::size_t n  = is_sin_.size();
        const std::size_t nw = dim == 0 ? 0 : pos.size() / dim;
        for (std::size_t i = 0; i < n; ++i) {
            const double* ki = k_.data() + i * dim;
            const bool sine = is_sin_[i] != 0;
            for (std::size_t w = w_begin; w < w_end; ++w) {
                double phase = 0.0;
                for (std::size_t d = 0; d < dim; ++d) phase += ki[d] * pos[d * nw + w];
                const double s = std::sin(phase);
                const double c = std::cos(phase);
                const double v  = sine ? s : c;     // φ
                const double dv = sine ? c : -s;    // ∂φ/∂phase
                val[i * nw + w] = v;
                for (std::size_t a = 0; a < dim; ++a) {
                    grad[(i * dim + a) * nw + w] = ki[a] * dv;
                    for (std::size_t b = 0; b < dim; ++b)
                        hess[((i * dim + a) * dim + b) * nw + w] = -ki[a] * ki[b] * v;
                }
            }
        }
    }

    // ∂_a∂_b∂_c φ = −k_a k_b k_c ∂φ/∂phase: +k_a k_b k_c sin(k·r) for cos,
    // −k_a k_b k_c cos(k·r) for sin.
    [[nodiscard]] bool supports_third_derivatives() const noexcept override { return true; }
    void evaluate_third_derivatives(std::span<const double> pos, std::size_t dim,
                                    std::span<double> third, std::size_t w_begin,
                                    std::size_t w_end) const override {
        const std::size_t n  = is_sin_.size();
        const std::size_t nw = dim == 0 ? 0 : pos.size() / dim;
        for (std::size_t i = 0; i < n; ++i) {
            const double* ki = k_.data() + i * dim;
            const bool sine = is_sin_[i] != 0;
            for (std::size_t w = w_begin; w < w_end; ++w) {
                double phase = 0.0;
                for (std::size_t d = 0; d < dim; ++d) phase += ki[d] * pos[d * nw + w];
                const double dv = sine ? std::cos(phase) : -std::sin(phase);   // ∂φ/∂phase
                for (std::size_t a = 0; a < dim; ++a)
                    for (std::size_t b = 0; b < dim; ++b)
                        for (std::size_t c = 0; c < dim; ++c)
                            third[(((i * dim + a) * dim + b) * dim + c) * nw + w] =
                                -ki[a] * ki[b] * ki[c] * dv;
            }
        }
    }

    // Σ_i |k_i|²/2 — the exact non-interacting energy of this determinant, since
    // every orbital is an eigenfunction of ∇² with eigenvalue −|k_i|².  Used by
    // the closed-shell test as an analytic reference.
    [[nodiscard]] double kinetic_energy_sum(double inv_mass = 1.0) const noexcept {
        double t = 0.0;
        for (std::size_t i = 0; i < is_sin_.size(); ++i) {
            double k2 = 0.0;
            for (std::size_t d = 0; d < dim_; ++d) {
                const double kd = k_[i * dim_ + d];
                k2 += kd * kd;
            }
            t += 0.5 * inv_mass * k2;
        }
        return t;
    }

    [[nodiscard]] std::span<const double> kvectors() const noexcept { return k_; }

    [[nodiscard]] std::unique_ptr<OrbitalSet> clone() const override {
        return std::make_unique<PlaneWaveOrbitalSet>(dim_, k_, is_sin_);
    }

private:
    std::size_t         dim_;
    std::vector<double> k_;       // [n * dim]
    std::vector<char>   is_sin_;  // [n]
};

// ── Reciprocal-lattice shell filling ───────────────────────────────────────
// A plane-wave determinant is only a well-defined ground state when the
// occupied k-points fill COMPLETE shells of equal |k|: a partly filled shell
// leaves the state degenerate, so its energy depends on which of several equally
// good k-points the code happened to pick, and the cell is no longer a sensible
// finite-size approximation to the uniform fluid.
//
// This returns the running list of orbital counts that close a shell, smallest
// first: 1, 7, 13, 19, 31, 37, ... for a triangular (hexagonal) 2-D cell, and
// 1, 5, 9, 13, 21, ... for a square one.
struct ReciprocalShellTable {
    std::vector<std::size_t> closed_counts;  // orbital counts that close a shell
    std::vector<double>      kvectors;       // [n_max * dim], shell-ordered
    std::vector<char>        is_sin;         // [n_max]
};

// Enumerate reciprocal-lattice shells until at least `want` real orbitals are
// available.  `want` is a target, not a constraint: the table is returned whole
// and the caller decides whether `want` closes a shell.
inline ReciprocalShellTable reciprocal_shell_table(const SimulationCell& cell,
                                                   std::size_t want) {
    if (!cell.is_periodic())
        throw std::runtime_error(
            "plane-wave orbitals need a periodic cell: there is no reciprocal "
            "lattice for an open boundary.");
    const std::size_t dim = cell.dim();
    if (dim != 1 && dim != 2)
        throw std::runtime_error(
            "plane-wave orbitals are implemented for 1-D and 2-D cells only.");
    const std::span<const double> b = cell.reciprocal();

    // Spacing between adjacent reciprocal lattice lines: any lattice point
    // within |k| ≤ M·h_min is reachable with coefficients |n_i| ≤ M, so shells
    // below that radius are guaranteed COMPLETE inside the enumeration box.
    double h_min = 0.0;
    if (dim == 1) {
        h_min = std::abs(b[0]);
    } else {
        const double area = std::abs(b[0] * b[3] - b[1] * b[2]);
        const double n0   = std::hypot(b[0], b[1]);
        const double n1   = std::hypot(b[2], b[3]);
        h_min = area / std::max(n0, n1);
    }

    struct Kv { double k2; double k[SimulationCell::kMaxDim]; };
    ReciprocalShellTable out;

    for (int M = 4; M <= 4096; M *= 2) {
        const double r_safe2 = (M * h_min) * (M * h_min) * (1.0 + 1e-12);
        std::vector<Kv> kv;
        const int lo = -M, hi = M;
        for (int i = lo; i <= hi; ++i) {
            for (int j = (dim == 2 ? lo : 0); j <= (dim == 2 ? hi : 0); ++j) {
                Kv e{};
                for (std::size_t d = 0; d < dim; ++d) {
                    e.k[d] = i * b[d];
                    if (dim == 2) e.k[d] += j * b[dim + d];
                }
                e.k2 = 0.0;
                for (std::size_t d = 0; d < dim; ++d) e.k2 += e.k[d] * e.k[d];
                if (e.k2 <= r_safe2) kv.push_back(e);
            }
        }
        // Shell order, then a deterministic order inside a shell.
        std::sort(kv.begin(), kv.end(), [](const Kv& a, const Kv& c) {
            if (a.k2 != c.k2) return a.k2 < c.k2;
            for (std::size_t d = 0; d < SimulationCell::kMaxDim; ++d)
                if (a.k[d] != c.k[d]) return a.k[d] < c.k[d];
            return false;
        });

        out.closed_counts.clear();
        out.kvectors.clear();
        out.is_sin.clear();
        std::size_t s = 0;
        while (s < kv.size()) {
            // Shell = the run of equal |k|² (relative tolerance: the k² values
            // come from sums of products, so exact equality is not guaranteed).
            std::size_t e = s + 1;
            while (e < kv.size() &&
                   std::abs(kv[e].k2 - kv[s].k2) <= 1e-9 * (1.0 + kv[s].k2))
                ++e;
            if (kv[s].k2 <= 1e-12 * (1.0 + kv[s].k2)) {
                // The k = 0 shell: one constant orbital.
                out.kvectors.insert(out.kvectors.end(), dim, 0.0);
                out.is_sin.push_back(0);
            } else {
                // One (cos, sin) pair per ±k pair: keep the lexicographically
                // positive member of each pair.
                for (std::size_t q = s; q < e; ++q) {
                    bool positive = false;
                    for (std::size_t d = 0; d < dim; ++d) {
                        if (kv[q].k[d] > 1e-12) { positive = true; break; }
                        if (kv[q].k[d] < -1e-12) { positive = false; break; }
                    }
                    if (!positive) continue;
                    for (int which = 0; which < 2; ++which) {
                        for (std::size_t d = 0; d < dim; ++d)
                            out.kvectors.push_back(kv[q].k[d]);
                        out.is_sin.push_back(static_cast<char>(which));
                    }
                }
            }
            out.closed_counts.push_back(out.is_sin.size());
            s = e;
        }
        if (!out.is_sin.empty() && out.is_sin.size() >= want) break;
    }
    return out;
}

// Build a closed-shell plane-wave orbital set with exactly `n_orb` orbitals.
// Throws, listing the nearby valid counts, when `n_orb` leaves a shell open.
inline std::unique_ptr<PlaneWaveOrbitalSet> make_plane_wave_orbitals(
    const SimulationCell& cell, std::size_t n_orb) {
    const ReciprocalShellTable t = reciprocal_shell_table(cell, n_orb);
    const bool closed = std::find(t.closed_counts.begin(), t.closed_counts.end(),
                                  n_orb) != t.closed_counts.end();
    if (!closed) {
        std::string list;
        for (std::size_t c : t.closed_counts) {
            if (c > n_orb * 4 + 8) break;
            list += (list.empty() ? "" : ", ") + std::to_string(c);
        }
        throw std::runtime_error(
            "plane-wave orbitals: " + std::to_string(n_orb) +
            " orbitals leave a reciprocal-lattice shell partly filled, so the "
            "determinant is degenerate and its energy depends on which of "
            "several equivalent k-points is picked.  Closed shells for this "
            "cell are: " + list + ".");
    }
    const std::size_t dim = cell.dim();
    std::vector<double> k(t.kvectors.begin(), t.kvectors.begin() + n_orb * dim);
    std::vector<char>   s(t.is_sin.begin(), t.is_sin.begin() + n_orb);
    return std::make_unique<PlaneWaveOrbitalSet>(dim, std::move(k), std::move(s));
}

// Real plane waves listed one by one: orbital o is cos(k_o·r), or sin(k_o·r)
// when is_sin[o], with k_o = Σ_i n_oi b_i for the integer indices n_o of the
// reciprocal vectors b_i dual to the lattice vectors as given
// (SimulationCell::given_reciprocal; is_sin empty: all cosines).  This is how a
// caller fills an OPEN shell with a subset of its degenerate cos/sin of its
// own choosing, which make_plane_wave_orbitals refuses to guess.  Two entries
// that are the same function up to sign (cos or sin at k and at ±k, or sin at
// k = 0, which vanishes) make every determinant zero; they are refused
// (std::invalid_argument), as are indices of the wrong length.
inline std::unique_ptr<PlaneWaveOrbitalSet> make_explicit_plane_wave_orbitals(
    const SimulationCell& cell, const std::vector<std::vector<int>>& k_indices,
    const std::vector<bool>& is_sin) {
    const std::size_t dim = cell.dim();
    if (!cell.is_periodic())
        throw std::invalid_argument("plane-wave orbitals need a periodic cell");
    if (!is_sin.empty() && is_sin.size() != k_indices.size())
        throw std::invalid_argument("`sin:` needs one entry per k");
    const std::span<const double> b = cell.given_reciprocal();
    const std::size_t n = k_indices.size();
    std::vector<double> k(n * dim, 0.0);
    std::vector<char>   s(n, 0);
    for (std::size_t o = 0; o < n; ++o) {
        const std::vector<int>& idx = k_indices[o];
        if (idx.size() != dim)
            throw std::invalid_argument("each k is " + std::to_string(dim) +
                                        " integers, in units of the reciprocal lattice "
                                        "vectors");
        bool zero = true;
        for (std::size_t i = 0; i < dim; ++i) {
            zero = zero && idx[i] == 0;
            for (std::size_t d = 0; d < dim; ++d) k[o * dim + d] += idx[i] * b[i * dim + d];
        }
        s[o] = !is_sin.empty() && is_sin[o];
        if (s[o] && zero) throw std::invalid_argument("sin at k = 0 vanishes everywhere");
        for (std::size_t q = 0; q < o; ++q) {
            bool same = true, opposite = true;
            for (std::size_t i = 0; i < dim; ++i) {
                same = same && k_indices[q][i] == idx[i];
                opposite = opposite && k_indices[q][i] == -idx[i];
            }
            if ((same || opposite) && s[q] == s[o])
                throw std::invalid_argument("two orbitals are the same function (k and ±k), "
                                            "so the determinant vanishes");
        }
    }
    return std::make_unique<PlaneWaveOrbitalSet>(dim, std::move(k), std::move(s));
}

// ── Triangular crystal sites commensurate with the cell ────────────────────
// The N sites of a triangular lattice that tile the simulation cell exactly.
// They exist only when the cell is a superlattice of the site lattice, i.e.
//
//     A₁ = p d₁ + q d₂,   A₂ = −q d₁ + (p+q) d₂,   det = p² + pq + q² = N,
//
// so N must be a Loeschian number (1, 3, 4, 7, 9, 12, 13, 16, 19, 21, 25, 27,
// 28, 31, 36, 37, ...).  Inverting the integer matrix gives the site vectors:
//
//     d₁ = [(p+q) A₁ − q A₂]/N,     d₂ = [q A₁ + p A₂]/N.
//
// Note that q ≠ 0 means the crystal is ROTATED relative to the cell — that is
// the whole point: rotated commensurate crystals (N = 7, 13, 19, 31, 37) are
// exactly the sizes that also close a reciprocal shell, so one N serves both the
// fluid and the crystal phase of the same crossing.
inline std::vector<double> commensurate_triangular_sites(const SimulationCell& cell,
                                                         std::size_t n_sites) {
    if (!cell.is_periodic() || cell.dim() != 2)
        throw std::runtime_error(
            "triangular crystal sites need a periodic 2-D cell.");
    const std::span<const double> a = cell.vectors();
    double a1[2] = {a[0], a[1]};
    double a2[2] = {a[2], a[3]};
    const double n1 = std::hypot(a1[0], a1[1]);
    const double n2 = std::hypot(a2[0], a2[1]);
    const double cosang = (a1[0] * a2[0] + a1[1] * a2[1]) / (n1 * n2);
    if (std::abs(n1 - n2) > 1e-9 * n1)
        throw std::runtime_error(
            "triangular crystal sites need a hexagonal cell (equal-length cell "
            "vectors at 60°); use `shape: hexagonal` or pass the moiré vectors "
            "directly.");
    if (std::abs(cosang + 0.5) < 1e-9) {
        // 120° convention: A₂ → A₁ + A₂ spans the same lattice at 60°.
        a2[0] += a1[0];
        a2[1] += a1[1];
    } else if (std::abs(cosang - 0.5) > 1e-9) {
        throw std::runtime_error(
            "triangular crystal sites need cell vectors at 60° or 120°; this "
            "cell has cos(angle) = " + std::to_string(cosang) + ".");
    }

    const std::size_t N = n_sites;
    std::size_t p = 0, q = 0;
    bool found = false;
    for (std::size_t pp = 0; pp <= N && !found; ++pp)
        for (std::size_t qq = 0; qq <= N && !found; ++qq)
            if (pp * pp + pp * qq + qq * qq == N) { p = pp; q = qq; found = true; }
    if (!found) {
        std::string list;
        for (std::size_t m = 1; m <= 128; ++m) {
            bool ok = false;
            for (std::size_t pp = 0; pp <= m && !ok; ++pp)
                for (std::size_t qq = 0; qq <= m && !ok; ++qq)
                    if (pp * pp + pp * qq + qq * qq == m) ok = true;
            if (ok) list += (list.empty() ? "" : ", ") + std::to_string(m);
        }
        throw std::runtime_error(
            "triangular crystal sites: " + std::to_string(N) +
            " particles cannot tile a hexagonal cell with a triangular lattice "
            "(N must be p²+pq+q²).  Valid sizes are: " + list + ".");
    }

    const double invN = 1.0 / static_cast<double>(N);
    double d1[2], d2[2];
    for (int d = 0; d < 2; ++d) {
        d1[d] = (static_cast<double>(p + q) * a1[d] -
                 static_cast<double>(q) * a2[d]) * invN;
        d2[d] = (static_cast<double>(q) * a1[d] +
                 static_cast<double>(p) * a2[d]) * invN;
    }

    // Enumerate the N distinct classes of i·d₁ + j·d₂ modulo the cell.  Dedup
    // compares MINIMUM-IMAGE separations, not wrapped coordinates: a site whose
    // fractional coordinate lands exactly on ±½ is one lattice point with two
    // legal wrapped forms, and only the fold sees them as equal.
    std::vector<double> sites;
    sites.reserve(N * 2);
    for (std::size_t i = 0; i < N && sites.size() < N * 2; ++i) {
        for (std::size_t j = 0; j < N && sites.size() < N * 2; ++j) {
            double r[SimulationCell::kMaxDim] = {
                static_cast<double>(i) * d1[0] + static_cast<double>(j) * d2[0],
                static_cast<double>(i) * d1[1] + static_cast<double>(j) * d2[1]};
            cell.wrap(r);
            bool dup = false;
            for (std::size_t s = 0; s < sites.size() / 2 && !dup; ++s) {
                double del[SimulationCell::kMaxDim] = {r[0] - sites[s * 2],
                                                       r[1] - sites[s * 2 + 1]};
                if (cell.min_image_dist2(del) < 1e-16 * n1 * n1) dup = true;
            }
            if (!dup) {
                sites.push_back(r[0]);
                sites.push_back(r[1]);
            }
        }
    }
    if (sites.size() != N * 2)
        throw std::runtime_error(
            "triangular crystal sites: expected " + std::to_string(N) +
            " distinct sites, generated " + std::to_string(sites.size() / 2) +
            " — the cell and the site lattice are not commensurate.");
    return sites;
}

}  // namespace alea
