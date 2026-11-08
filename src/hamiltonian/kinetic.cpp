// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "hamiltonian/kinetic.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "population/walker_set.h"

namespace alea {

Kinetic::Kinetic(std::vector<ParticleGroup> groups, GaugeField gauge) {
    // Flatten groups into per-particle, per-dimension inverse mass.
    std::size_t np = 0;
    for (const ParticleGroup& g : groups) np = std::max(np, g.last);
    // dim is unknown here; resolve from the first group's inv_mass size.
    const std::size_t dim = groups.empty() ? 0 : groups[0].inv_mass.size();
    inv_mass_.assign(np * dim, 0.0);
    for (const ParticleGroup& g : groups)
        for (std::size_t i = g.first; i < g.last; ++i)
            for (std::size_t d = 0; d < dim; ++d)
                inv_mass_[i * dim + d] = g.inv_mass[d];

    if (!gauge.active()) return;
    if (dim != 2)
        throw std::invalid_argument(
            "magnetic field / rotation: implemented for 2-D (field along z)");
    gauge_coupling_.assign(np, 0.0);
    for (const ParticleGroup& g : groups) {
        // A rotating frame couples through the mass, which must then be a
        // scalar: L_z mixes x and y, so an anisotropic mass has no single m.
        const double m = 1.0 / g.inv_mass[0];
        if (gauge.omega != 0.0 && g.inv_mass[0] != g.inv_mass[1])
            throw std::invalid_argument(
                "rotation: group '" + g.name + "' has an anisotropic mass; the "
                "rotating-frame coupling m·Ω needs an isotropic one");
        for (std::size_t i = g.first; i < g.last; ++i)
            gauge_coupling_[i] = 0.5 * g.charge * gauge.B + m * gauge.omega;
    }
}

void Kinetic::evaluate_batch(WalkerSet& batch, std::size_t w_begin,
                             std::size_t w_end) {
    const std::size_t nw  = batch.num_walkers();   // global stride
    const std::size_t np  = batch.n_particles();
    const std::size_t dim = batch.dim();
    if (w_end > nw) w_end = nw;
    const std::span<const double> grad = batch.gradients();   // [p*dim*nw + d*nw + w]
    const std::span<const double> lap  = batch.laplacians();  // [p*dim*nw + d*nw + w]
    double* __restrict le = batch.local_energies().data();

    // K_w = -½ Σ_i Σ_d (1/m_i_d) [ ∂²_d + (∂_d)² ].
    for (std::size_t i = 0; i < np; ++i) {
        for (std::size_t d = 0; d < dim; ++d) {
            const double inv_m = inv_mass_[i * dim + d];
            const std::size_t off = i * dim * nw + d * nw;
            const double* __restrict g = grad.data() + off;
            const double* __restrict l = lap.data() + off;
#pragma omp simd
            for (std::size_t w = w_begin; w < w_end; ++w)
                le[w] += -0.5 * inv_m * (l[w] + g[w] * g[w]);
        }
    }
    double* im = batch.has_phase() ? batch.local_energies_imag().data() : nullptr;
    if (!gauge_coupling_.empty())
        add_gauge_terms(batch, le, w_begin, w_end, im);
    else if (batch.has_phase())
        add_phase_terms(batch, le, w_begin, w_end, im);
}

// Complex trial ψ = e^{u + iφ}.  With ∇ln ψ = ∇u + i∇φ,
//   ψ⁻¹∇²ψ = ∇²u + |∇u|² − |∇φ|² + i(∇²φ + 2∇u·∇φ),
// so the kinetic local energy splits into
//   Re:  −½ Σ (1/m)[∂²u + (∂u)² − (∂φ)²]   (the fixed-phase local energy)
//   Im:  −½ Σ (1/m)[∂²φ + 2 ∂u ∂φ]
// The −(∂φ)² piece is the fixed-phase effective potential +½|∇φ|²/m of
// Ortiz, Ceperley & Martin, PRL 71, 2777 (1993), eq. (5); it matches
// QMCPACK's BareKineticEnergy, real(G·G + L) with a complex G.  Only the
// +½(1/m)(∂φ)² correction is added here: the modulus part was summed above.
// (Im is written to the diagnostic column only when `im` is non-null.)
void Kinetic::add_phase_terms(const WalkerSet& batch, double* __restrict re,
                              std::size_t w_begin, std::size_t w_end,
                              double* __restrict im) const {
    const std::size_t nw  = batch.num_walkers();
    const std::size_t np  = batch.n_particles();
    const std::size_t dim = batch.dim();
    const std::span<const double> grad = batch.gradients();
    const std::span<const double> pgr  = batch.phase_gradients();
    const std::span<const double> plp  = batch.phase_laplacians();
    for (std::size_t i = 0; i < np; ++i) {
        for (std::size_t d = 0; d < dim; ++d) {
            const double inv_m = inv_mass_[i * dim + d];
            const std::size_t off = i * dim * nw + d * nw;
            const double* __restrict g  = grad.data() + off;
            const double* __restrict pg = pgr.data() + off;
            const double* __restrict pl = plp.data() + off;
#pragma omp simd
            for (std::size_t w = w_begin; w < w_end; ++w)
                re[w] += 0.5 * inv_m * pg[w] * pg[w];
            if (im) {
#pragma omp simd
                for (std::size_t w = w_begin; w < w_end; ++w)
                    im[w] += -0.5 * inv_m * (pl[w] + 2.0 * g[w] * pg[w]);
            }
        }
    }
}

void Kinetic::accumulate_psi_dependent_energy(const WalkerSet& batch,
                                              std::span<double> out) const {
    const std::size_t nw  = batch.num_walkers();
    const std::size_t np  = batch.n_particles();
    const std::size_t dim = batch.dim();
    const std::span<const double> grad = batch.gradients();
    const std::span<const double> lap  = batch.laplacians();
    double* __restrict o = out.data();
    for (std::size_t i = 0; i < np; ++i) {
        for (std::size_t d = 0; d < dim; ++d) {
            const double inv_m = inv_mass_[i * dim + d];
            const std::size_t off = i * dim * nw + d * nw;
            const double* __restrict g = grad.data() + off;
            const double* __restrict l = lap.data() + off;
#pragma omp simd
            for (std::size_t w = 0; w < nw; ++w)
                o[w] += -0.5 * inv_m * (l[w] + g[w] * g[w]);
        }
    }
    if (!gauge_coupling_.empty())
        add_gauge_terms(batch, o, 0, nw, nullptr);
    else if (batch.has_phase())
        add_phase_terms(batch, o, 0, nw, nullptr);
}

// Vector potential a_i = c_i ẑ×r_i = c_i(−y_i, x_i).  With ψ = e^{u+iφ} and
// H_kin = Σ (p − a_i)²/2m_i  (p = −i∇; the gauge is Coulomb, ∇·a = 0):
//   ψ⁻¹(p − a)²ψ = −∇²u − |∇u|² + |∇φ − a|² − i[∇²φ + 2∇u·(∇φ − a)]
// so Re E_kin adds +½(1/m)(∂φ − a)² per dimension (the modulus part was summed
// by the caller) and Im E_kin = −½(1/m)[∂²φ + 2 ∂u (∂φ − a)].  This is the
// fixed-phase Hamiltonian of Ortiz, Ceperley & Martin, PRL 71, 2777 (1993),
// eq. (5), with the vector potential: the effective potential is ½|∇φ − a|²/m.
// A real trial (no phase planes) still gets ½|a|²/m — the diamagnetic term.
void Kinetic::add_gauge_terms(const WalkerSet& batch, double* __restrict re,
                              std::size_t w_begin, std::size_t w_end,
                              double* __restrict im) const {
    const std::size_t nw  = batch.num_walkers();
    const std::size_t np  = batch.n_particles();
    const std::size_t dim = batch.dim();
    const bool phase = batch.has_phase();
    const std::span<const double> pos  = batch.positions();
    const std::span<const double> grad = batch.gradients();
    for (std::size_t i = 0; i < np; ++i) {
        const double c = gauge_coupling_[i];
        const double* __restrict x = pos.data() + i * dim * nw;
        const double* __restrict y = x + nw;
        for (std::size_t d = 0; d < dim; ++d) {
            const double inv_m = inv_mass_[i * dim + d];
            const std::size_t off = i * dim * nw + d * nw;
            // a_d = c·(−y, x)_d, written as sign·c·(other coordinate).
            const double* __restrict other = d == 0 ? y : x;
            const double sgn = d == 0 ? -c : c;
            const double* __restrict g  = grad.data() + off;
            const double* __restrict pg = phase ? batch.phase_gradients().data() + off : nullptr;
            const double* __restrict pl = phase ? batch.phase_laplacians().data() + off : nullptr;
            for (std::size_t w = w_begin; w < w_end; ++w) {
                const double v = (pg ? pg[w] : 0.0) - sgn * other[w];   // ∂φ − a
                re[w] += 0.5 * inv_m * v * v;
                if (im) im[w] += -0.5 * inv_m * ((pl ? pl[w] : 0.0) + 2.0 * g[w] * v);
            }
        }
    }
}

void Kinetic::local_energy_parameter_derivatives(
    const WalkerSet& batch, std::span<const double> dgrad,
    std::span<const double> dlap, std::size_t n_opt,
    std::span<double> out) const {
    const std::size_t nw  = batch.num_walkers();
    const std::size_t np  = batch.n_particles();
    const std::size_t dim = batch.dim();
    const std::span<const double> grad = batch.gradients();
    const std::size_t block = np * dim * nw;

    for (std::size_t k = 0; k < n_opt; ++k) {
        double* __restrict o = out.data() + k * nw;
        for (std::size_t i = 0; i < np; ++i) {
            for (std::size_t d = 0; d < dim; ++d) {
                const double inv_m = inv_mass_[i * dim + d];
                const std::size_t off = i * dim * nw + d * nw;
                const double* __restrict g  = grad.data() + off;
                const double* __restrict dg = dgrad.data() + k * block + off;
                const double* __restrict dl = dlap.data() + k * block + off;
#pragma omp simd
                for (std::size_t w = 0; w < nw; ++w)
                    o[w] += -0.5 * inv_m * (dl[w] + 2.0 * g[w] * dg[w]);
            }
        }
    }
}

// Differentiating the split of add_phase_terms / add_gauge_terms, per particle
// and dimension with v = ∂φ − a (a = 0 without a field):
//   Re E_kin = −½(1/m)[∂²u + (∂u)²] + ½(1/m) v²
//   Im E_kin = −½(1/m)[∂²φ + 2 ∂u v]
// and the vector potential a(r) does not depend on p, so with ġ = ∂_p ∂u,
// l̇ = ∂_p ∂²u, ṗ = ∂_p ∂φ, q̇ = ∂_p ∂²φ:
//   ∂_p Re E_kin = −½(1/m)[l̇ + 2 ∂u ġ − 2 v ṗ]
//   ∂_p Im E_kin = −½(1/m)[q̇ + 2 ġ v + 2 ∂u ṗ]
// For ṗ = q̇ = 0 (a real factor) the real part is the formula above.
void Kinetic::complex_local_energy_parameter_derivatives(
    const WalkerSet& batch, std::span<const double> dgrad, std::span<const double> dlap,
    std::span<const double> dpgrad, std::span<const double> dplap, std::size_t n_opt,
    std::span<double> out_re, std::span<double> out_im) const {
    const std::size_t nw  = batch.num_walkers();
    const std::size_t np  = batch.n_particles();
    const std::size_t dim = batch.dim();
    if (!batch.has_phase())
        throw std::logic_error("kinetic: complex parameter derivatives need the phase planes "
                               "of a complex trial");
    const std::span<const double> grad = batch.gradients();
    const std::span<const double> pgr  = batch.phase_gradients();
    const std::span<const double> pos  = batch.positions();
    const std::size_t block = np * dim * nw;
    const bool field = !gauge_coupling_.empty();

    for (std::size_t k = 0; k < n_opt; ++k) {
        double* __restrict ore = out_re.data() + k * nw;
        double* __restrict oim = out_im.data() + k * nw;
        for (std::size_t i = 0; i < np; ++i) {
            for (std::size_t d = 0; d < dim; ++d) {
                const double inv_m = inv_mass_[i * dim + d];
                const std::size_t off = i * dim * nw + d * nw;
                const double* __restrict g   = grad.data() + off;
                const double* __restrict pg  = pgr.data() + off;
                const double* __restrict dg  = dgrad.data() + k * block + off;
                const double* __restrict dl  = dlap.data() + k * block + off;
                const double* __restrict dpg = dpgrad.data() + k * block + off;
                const double* __restrict dpl = dplap.data() + k * block + off;
                // a_d = c·(−y, x)_d, as in add_gauge_terms.
                const double c = field ? gauge_coupling_[i] : 0.0;
                const double* __restrict other =
                    pos.data() + i * dim * nw + (d == 0 ? (dim > 1 ? nw : 0) : 0);
                const double sgn = d == 0 ? -c : c;
                for (std::size_t w = 0; w < nw; ++w) {
                    const double v = pg[w] - sgn * other[w];
                    ore[w] += -0.5 * inv_m * (dl[w] + 2.0 * g[w] * dg[w] - 2.0 * v * dpg[w]);
                    oim[w] += -0.5 * inv_m * (dpl[w] + 2.0 * dg[w] * v + 2.0 * g[w] * dpg[w]);
                }
            }
        }
    }
}

}  // namespace alea
