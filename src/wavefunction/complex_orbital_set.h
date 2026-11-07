// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "../population/simulation_cell.h"

namespace alea {

// A complex array stored as two real planes (split re/im).  Keeps the walker
// loop contiguous and SIMD-friendly, the same reason ψ's modulus and phase
// live in separate planes of the WalkerSet.
struct SplitComplexSpan {
    std::span<double> re;
    std::span<double> im;
};

// Complex counterpart of OrbitalSet: n single-particle orbitals φ_i(r) ∈ ℂ,
// evaluated at ONE particle's position across all walkers.  Same SoA layouts
// as OrbitalSet, one plane per real/imaginary part:
//   val  : [n * nw]         val[i * nw + w]           = φ_i(r_w)
//   grad : [n * dim * nw]   grad[(i*dim + d)*nw + w]  = ∂_d φ_i(r_w)
//   lap  : [n * dim * nw]   lap[(i*dim + d)*nw + w]   = ∂²_d φ_i(r_w)
// A ComplexSlaterDeterminant uses it to fill one matrix row.
class ComplexOrbitalSet {
public:
    virtual ~ComplexOrbitalSet() = default;

    [[nodiscard]] virtual std::size_t size() const = 0;

    // Outputs are OVERWRITTEN over the walker range [w_begin, w_end).  Stride is
    // the global nw = pos.size()/dim.
    virtual void evaluate(std::span<const double> pos, std::size_t dim,
                          SplitComplexSpan val, SplitComplexSpan grad,
                          SplitComplexSpan lap, std::size_t w_begin,
                          std::size_t w_end) const = 0;

    [[nodiscard]] virtual std::unique_ptr<ComplexOrbitalSet> clone() const = 0;
};

// Plane waves with a twist, φ_i(r) = e^{i q_i·r},  q_i = k_i + θ:
//   ∂_d φ = i q_d φ,     ∂²_d φ = −q_d² φ.
// With k_i on the reciprocal lattice and θ the twist, φ(r + L) = e^{iθ·L} φ(r):
// the twisted boundary condition of Lin, Zong & Ceperley, PRE 64, 016702
// (2001).  Wrapping a walker back into the cell therefore multiplies a
// determinant row by a constant phase — a gauge change that leaves |ψ|, ∇ln ψ
// and every fixed-phase quantity unchanged.
class TwistedPlaneWaveOrbitalSet final : public ComplexOrbitalSet {
public:
    // q: [n * dim], the full wavevectors k_i + θ.
    TwistedPlaneWaveOrbitalSet(std::size_t dim, std::vector<double> q)
        : dim_(dim), q_(std::move(q)) {}

    [[nodiscard]] std::size_t size() const override {
        return dim_ == 0 ? 0 : q_.size() / dim_;
    }

    void evaluate(std::span<const double> pos, std::size_t dim,
                  SplitComplexSpan val, SplitComplexSpan grad,
                  SplitComplexSpan lap, std::size_t w_begin,
                  std::size_t w_end) const override {
        const std::size_t n  = size();
        const std::size_t nw = dim == 0 ? 0 : pos.size() / dim;
        for (std::size_t i = 0; i < n; ++i) {
            const double* qi = q_.data() + i * dim;
            for (std::size_t w = w_begin; w < w_end; ++w) {
                double arg = 0.0;
                for (std::size_t d = 0; d < dim; ++d) arg += qi[d] * pos[d * nw + w];
                const double c = std::cos(arg), s = std::sin(arg);
                val.re[i * nw + w] = c;
                val.im[i * nw + w] = s;
                for (std::size_t d = 0; d < dim; ++d) {
                    const std::size_t idx = (i * dim + d) * nw + w;
                    grad.re[idx] = -qi[d] * s;          // Re(i q c + i q i s)
                    grad.im[idx] =  qi[d] * c;
                    lap.re[idx]  = -qi[d] * qi[d] * c;
                    lap.im[idx]  = -qi[d] * qi[d] * s;
                }
            }
        }
    }

    // q_i = k_i + θ, [n * dim].
    [[nodiscard]] std::span<const double> wavevectors() const noexcept { return q_; }

    // Σ_i |q_i|²/2m — the exact kinetic energy of a determinant of these
    // orbitals (each is an eigenfunction of −∇²/2m).  Test oracle.
    [[nodiscard]] double kinetic_energy_sum(double inv_mass = 1.0) const noexcept {
        double t = 0.0;
        for (double qd : q_) t += 0.5 * inv_mass * qd * qd;
        return t;
    }

    [[nodiscard]] std::unique_ptr<ComplexOrbitalSet> clone() const override {
        return std::make_unique<TwistedPlaneWaveOrbitalSet>(dim_, q_);
    }

private:
    std::size_t         dim_;
    std::vector<double> q_;   // [n * dim]
};

// Lowest-radial (n = 0) Fock–Darwin orbitals in 2-D, one per angular momentum l:
//
//   φ_l(r) = w^{|l|} e^{−β r²/2},   w = x + i·s·y,   s = sgn(l)  (s = +1 for l = 0)
//
// so φ_l ∝ r^{|l|} e^{ilθ} e^{−βr²/2}: angular momentum L_z = l.  With
// β = mΩ, Ω = √(ω₀² + ω_c²/4), it is an exact eigenstate of a charge in a
// parabolic dot ω₀ and perpendicular field B (symmetric gauge), energy
// (|l|+1)Ω − l ω_c/2 with ω_c = qB/m (Fock, Z. Phys. 47, 446 (1928); Darwin,
// Proc. Camb. Phil. Soc. 27, 86 (1931)); for B = 0 in a frame rotating at Ω_rot
// it is an eigenstate with (|l|+1)ω₀ − Ω_rot l.  For B → strong they become the
// lowest-Landau-level orbitals z^l e^{−|z|²/4ℓ²}.  Centred at the origin — the
// gauge origin of the symmetric-gauge vector potential.
//
// Derivatives with w' = ∂_x w = 1, ∂_y w = i s, g = e^{−βr²/2}, L = |l|:
//   ∂_d φ  = [L w^{L−1} w'_d − β r_d w^L] g
//   ∂²_d φ = [L(L−1) w^{L−2} w'_d² − 2β r_d L w^{L−1} w'_d + (β² r_d² − β) w^L] g
// w^{L−1}, w^{L−2} are built by repeated multiplication and are exactly 0 when
// their prefactor L or L(L−1) vanishes, so the origin is regular (no 0·∞).
class FockDarwinOrbitalSet final : public ComplexOrbitalSet {
public:
    FockDarwinOrbitalSet(std::vector<int> ls, double beta)
        : ls_(std::move(ls)), beta_(beta) {
        if (!(beta_ > 0.0))
            throw std::invalid_argument("Fock-Darwin orbitals need beta > 0");
    }

    [[nodiscard]] std::size_t size() const override { return ls_.size(); }

    void evaluate(std::span<const double> pos, std::size_t dim,
                  SplitComplexSpan val, SplitComplexSpan grad,
                  SplitComplexSpan lap, std::size_t w_begin,
                  std::size_t w_end) const override {
        if (dim != 2)
            throw std::runtime_error("Fock-Darwin orbitals are 2-D");
        using cplx = std::complex<double>;
        const std::size_t nw = pos.size() / dim;
        const double b = beta_;
        for (std::size_t i = 0; i < ls_.size(); ++i) {
            const int    L = std::abs(ls_[i]);
            const double s = ls_[i] < 0 ? -1.0 : 1.0;
            const cplx   dw[2] = {cplx(1.0, 0.0), cplx(0.0, s)};
            for (std::size_t w = w_begin; w < w_end; ++w) {
                const double r[2] = {pos[w], pos[nw + w]};
                const cplx   z(r[0], s * r[1]);
                // w^{L-2}, w^{L-1}, w^L (zero where the exponent is negative).
                cplx wL(1.0, 0.0), wL1(0.0, 0.0), wL2(0.0, 0.0);
                for (int k = 0; k < L; ++k) {
                    wL2 = wL1;
                    wL1 = wL;
                    wL *= z;
                }
                const double g = std::exp(-0.5 * b * (r[0] * r[0] + r[1] * r[1]));
                const cplx   v = wL * g;
                val.re[i * nw + w] = v.real();
                val.im[i * nw + w] = v.imag();
                for (std::size_t d = 0; d < 2; ++d) {
                    const cplx gd = (static_cast<double>(L) * wL1 * dw[d] - b * r[d] * wL) * g;
                    const cplx ld = (static_cast<double>(L * (L - 1)) * wL2 * dw[d] * dw[d]
                                     - 2.0 * b * r[d] * static_cast<double>(L) * wL1 * dw[d]
                                     + (b * b * r[d] * r[d] - b) * wL) * g;
                    const std::size_t idx = (i * dim + d) * nw + w;
                    grad.re[idx] = gd.real();
                    grad.im[idx] = gd.imag();
                    lap.re[idx]  = ld.real();
                    lap.im[idx]  = ld.imag();
                }
            }
        }
    }

    [[nodiscard]] std::span<const int> angular_momenta() const noexcept { return ls_; }
    [[nodiscard]] double beta() const noexcept { return beta_; }

    [[nodiscard]] std::unique_ptr<ComplexOrbitalSet> clone() const override {
        return std::make_unique<FockDarwinOrbitalSet>(ls_, beta_);
    }

private:
    std::vector<int> ls_;
    double           beta_;
};

// Occupation for a twist: the n orbitals with the lowest |k + θ| over the
// cell's reciprocal lattice k, θ = Σ_i t_i b_i given in FRACTIONAL coordinates
// of the reciprocal basis (t and t + integer are the same twist: the set of
// k + θ is identical).  This is the canonical, fixed-N filling used for twist
// averaging (Lin, Zong & Ceperley, PRE 64, 016702 (2001)): each twist is its
// own ground state, and averaging over t removes the single-particle shell
// effects of the finite cell.
//
// Refuses an occupation whose last orbital ties with the first empty one: the
// state is then degenerate and its energy depends on which of the equal-|k+θ|
// points was picked (e.g. θ = b₁/2 with one particle).
inline std::unique_ptr<TwistedPlaneWaveOrbitalSet> make_twisted_plane_wave_orbitals(
    const SimulationCell& cell, std::size_t n, const std::vector<double>& twist) {
    if (!cell.is_periodic())
        throw std::runtime_error("twisted plane waves need a periodic cell");
    const std::size_t dim = cell.dim();
    if (dim != 1 && dim != 2)
        throw std::runtime_error("twisted plane waves are implemented for 1-D and 2-D cells");
    if (twist.size() != dim)
        throw std::runtime_error("twist has " + std::to_string(twist.size()) +
                                 " components, the cell is " + std::to_string(dim) + "-D");
    const std::span<const double> b = cell.reciprocal();

    double theta[SimulationCell::kMaxDim] = {};
    for (std::size_t i = 0; i < dim; ++i)
        for (std::size_t d = 0; d < dim; ++d) theta[d] += twist[i] * b[i * dim + d];
    double theta_norm = 0.0;
    for (std::size_t d = 0; d < dim; ++d) theta_norm += theta[d] * theta[d];
    theta_norm = std::sqrt(theta_norm);

    // Spacing of reciprocal lattice lines (see reciprocal_shell_table): every
    // k with |k| ≤ M·h_min lies in the |i|,|j| ≤ M box, so every q = k + θ with
    // |q| ≤ M·h_min − |θ| is enumerated.
    double h_min = 0.0;
    if (dim == 1) {
        h_min = std::abs(b[0]);
    } else {
        const double area = std::abs(b[0] * b[3] - b[1] * b[2]);
        h_min = area / std::max(std::hypot(b[0], b[1]), std::hypot(b[2], b[3]));
    }

    struct Q { double q2; double q[SimulationCell::kMaxDim]; };
    for (int M = 4; M <= 4096; M *= 2) {
        std::vector<Q> qs;
        for (int i = -M; i <= M; ++i)
            for (int j = (dim == 2 ? -M : 0); j <= (dim == 2 ? M : 0); ++j) {
                Q e{};
                for (std::size_t d = 0; d < dim; ++d) {
                    e.q[d] = i * b[d] + theta[d];
                    if (dim == 2) e.q[d] += j * b[dim + d];
                    e.q2 += e.q[d] * e.q[d];
                }
                qs.push_back(e);
            }
        if (qs.size() <= n) continue;
        std::sort(qs.begin(), qs.end(), [](const Q& a, const Q& c) { return a.q2 < c.q2; });
        const double r_safe = M * h_min - theta_norm;
        if (r_safe <= 0.0 || qs[n].q2 > r_safe * r_safe) continue;   // box too small

        if (n > 0 && std::abs(qs[n].q2 - qs[n - 1].q2) <= 1e-9 * (1.0 + qs[n].q2))
            throw std::runtime_error(
                "twisted plane waves: " + std::to_string(n) + " orbitals split a "
                "degenerate |k+θ| shell at this twist, so the occupation is "
                "ambiguous.  Change the particle count or move the twist off this "
                "symmetric point.");
        std::vector<double> q;
        q.reserve(n * dim);
        for (std::size_t k = 0; k < n; ++k)
            for (std::size_t d = 0; d < dim; ++d) q.push_back(qs[k].q[d]);
        return std::make_unique<TwistedPlaneWaveOrbitalSet>(dim, std::move(q));
    }
    throw std::runtime_error("twisted plane waves: enumeration did not converge");
}

}  // namespace alea
