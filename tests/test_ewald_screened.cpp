// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

#include "fast_screened_coulomb.h"
#include "hamiltonian/ewald_term.h"
#include "hamiltonian/screened_coulomb.h"
#include "numerics/ewald_breakup.h"
#include "population/simulation_cell.h"
#include "population/walker_set.h"

using alea::build_screened_ewald_kernel;
using alea::numerics::PotentialSpec;
using Catch::Approx;

// ── The periodic screened_coulomb kernel ────────────────────────────────────
//
// fast-screened-coulomb supplies eps(k) for a multilayer stack; the kernel
// builds from it an exact transform pair: Ṽ(k) = 2π/(k eps(k)) and a real-space
// table of its inverse transform on (0, r_max].  In a stack with no dielectric
// contrast it must collapse onto plain (or layer-offset) Coulomb exactly.

TEST_CASE("a uniform stack gives the in-plane Coulomb kernel",
          "[ewald][screened]") {
    constexpr double eps = 2.5;
    const PotentialSpec k = build_screened_ewald_kernel(
        {eps, eps, eps}, {-2.0, 2.0}, /*source*/ 2, /*target*/ 0, 0.0, /*r_max*/ 5.0);

    const double pi = std::acos(-1.0);
    CHECK(k.epsilon == eps);  // long-range eps: the outer media's mean
    for (double r : {0.05, 0.7, 2.0, 4.9}) {
        INFO("r = " << r);
        CHECK(k.v_real(r) == Approx(1.0 / (eps * r)).epsilon(1e-10));
    }
    for (double q : {0.01, 0.5, 3.0, 50.0, 2000.0}) {
        INFO("k = " << q);
        CHECK(k.v_fourier(q) == Approx(2 * pi / (eps * q)).epsilon(1e-10));
    }
    CHECK(k.k0_offset == Approx(0.0).margin(1e-8));
}

TEST_CASE("a uniform stack across layers gives the layer-offset Coulomb kernel",
          "[ewald][screened][layers]") {
    // Source in layer 2, observation in layer 3 at z = d: the transform carries
    // e^{-kd} inside eps(k), and the result must be the layer-offset
    // Coulomb kernel coulomb(eps, d).
    constexpr double eps = 2.5, d = 2.0;
    const PotentialSpec k = build_screened_ewald_kernel(
        {eps, eps, eps, eps}, {-1.0, 1.0, 3.0}, 2, 3, d, 5.0);
    const PotentialSpec ref = PotentialSpec::coulomb(eps, d);

    CHECK(k.epsilon == eps);
    for (double r : {0.0 + 1e-3, 0.7, 2.0, 4.9}) {
        INFO("r = " << r);
        CHECK(k.v_real(r) == Approx(ref.v_real(r)).epsilon(1e-10));
    }
    CHECK(k.dv_real(4.9) == Approx(ref.dv_real(4.9)).epsilon(1e-8));
    CHECK(k.d2v_real(4.9) == Approx(ref.d2v_real(4.9)).epsilon(1e-7));
    for (double q : {0.01, 0.5, 3.0, 50.0, 2000.0}) {
        // Past the point where 1/eps(k) has fallen below 1e-15 of the Coulomb
        // scale the kernel sets it to exactly 0 (in the table too, so the pair
        // stays exact); that bounds the absolute difference by 2π·1e-15/(eps k).
        INFO("k = " << q);
        CHECK(k.v_fourier(q) == Approx(ref.v_fourier(q))
                                    .epsilon(1e-10)
                                    .margin(2 * std::acos(-1.0) * 1e-15 / (eps * q)));
    }
    CHECK(k.k0_offset == Approx(ref.k0_offset).epsilon(1e-8));
}

namespace {

using alea::EwaldTerm;
using alea::ParticleGroup;
using alea::SimulationCell;
using alea::WalkerSet;

ParticleGroup grp(const char* n, std::size_t f, std::size_t l, double q) {
    ParticleGroup g;
    g.name = n; g.charge = q; g.first = f; g.last = l; g.count = l - f;
    g.mass = {1.0, 1.0}; g.inv_mass = {1.0, 1.0};
    return g;
}

SimulationCell square(double L) {
    const std::vector<double> a{L, 0.0, 0.0, L};
    return SimulationCell(2, std::span<const double>(a));
}

double energy_with(const EwaldTerm& term, const SimulationCell& cell,
                   const std::vector<double>& pos) {
    const std::size_t np = pos.size() / 2;
    WalkerSet ws(1, np, 2, cell);
    for (std::size_t i = 0; i < 2 * np; ++i) ws.positions()[i] = pos[i];
    ws.dist_tables().build_all(ws.positions());
    ws.local_energies()[0] = 0.0;
    EwaldTerm t = term;
    t.evaluate_batch(ws);
    return ws.local_energies()[0];
}

const std::vector<double> kPos{0.3, 0.4, 2.1, 4.4,   // e
                               1.0, 0.9, 4.2, 2.6};  // h

}  // namespace

TEST_CASE("a uniform-stack screened bilayer sums like the Coulomb bilayer",
          "[ewald][screened][layers][physics]") {
    // Electrons in layer 2, holes in layer 3 (z = 2), no contrast anywhere: the
    // periodic screened sum must equal the Coulomb-with-offset one, which is
    // validated against Parry's slab formula in test_ewald.cpp.
    constexpr double eps = 2.5, d = 2.0, L = 6.0;
    const SimulationCell cell = square(L);
    const double         rc   = cell.ws_radius();
    const std::vector<ParticleGroup> groups{grp("e", 0, 2, -1.0), grp("h", 2, 4, +1.0)};

    const PotentialSpec ee = build_screened_ewald_kernel({eps, eps, eps}, {-1.0, 1.0}, 2, 0, 0.0, rc);
    const PotentialSpec eh = build_screened_ewald_kernel({eps, eps, eps, eps}, {-1.0, 1.0, 3.0},
                                                         2, 3, d, rc);
    const EwaldTerm screened(groups, {{0, 0, ee}, {1, 1, ee}, {0, 1, eh}}, cell, 1e-8);

    const PotentialSpec C = PotentialSpec::coulomb(eps);
    const EwaldTerm coulomb(groups, {{0, 0, C}, {1, 1, C}, {0, 1, PotentialSpec::coulomb(eps, d)}},
                            cell, 1e-8);

    const double e_s = energy_with(screened, cell, kPos);
    const double e_c = energy_with(coulomb, cell, kPos);
    INFO("screened " << e_s << "  coulomb " << e_c);
    CHECK(e_s == Approx(e_c).margin(1e-7));
    // Charged layers, uniform dielectric: the parallel-plate value 2π Q² d/(ε A)
    // with Q = 2, recovered from the ETM kernels' k = 0 remainders alone.
    CHECK(screened.capacitor_energy() ==
          Approx(2.0 * std::acos(-1.0) * 4.0 * d / (eps * L * L)).epsilon(1e-7));
    CHECK(screened.capacitor_energy() == Approx(coulomb.capacitor_energy()).epsilon(1e-7));
}

TEST_CASE("a contrast-stack screened bilayer: capacitor law and periodicity",
          "[ewald][screened][layers][physics]") {
    // TMD-like bilayer: two eps = 14 slabs (6 Bohr each) in vacuum, electrons at
    // the centre of the lower one (z = 0), holes at the centre of the upper one
    // (z = 6).  Every pair has its own ETM kernel; all share the vacuum tail.
    //
    // Reference for the k = 0 physics: two opposite uniform sheets store
    //   E = 2 pi (Q^2/A) int_0^d dz/eps(z)
    // (uniform displacement field between them, none outside) -- plain
    // electrostatics, nothing to do with transfer matrices.  Here both sheets sit
    // inside eps = 14 all the way across: E = 2 pi Q^2 d/(14 A).
    const std::vector<double> eps{1.0, 14.0, 14.0, 1.0};
    constexpr double L = 6.0, d = 6.0, A = 36.0, Q = 2.0;
    const SimulationCell cell = square(L);
    const double         rc   = cell.ws_radius();
    const PotentialSpec ee = build_screened_ewald_kernel(eps, {-3.0, 3.0, 9.0}, 2, 0, 0.0, rc);
    const PotentialSpec hh = build_screened_ewald_kernel(eps, {-9.0, -3.0, 3.0}, 3, 0, 0.0, rc);
    const PotentialSpec eh = build_screened_ewald_kernel(eps, {-3.0, 3.0, 9.0}, 2, 3, d, rc);
    const EwaldTerm term({grp("e", 0, 2, -1.0), grp("h", 2, 4, +1.0)},
                         {{0, 0, ee}, {1, 1, hh}, {0, 1, eh}}, cell, 1e-7);

    const double pi = std::acos(-1.0);
    CHECK(term.capacitor_energy() == Approx(2.0 * pi * Q * Q * d / (14.0 * A)).epsilon(1e-7));

    // Periodicity of the full energy: one hole moved by a lattice vector, and
    // everything shifted by a non-lattice vector.
    const double base = energy_with(term, cell, kPos);
    std::vector<double> wrapped = kPos;
    wrapped[6] -= L;
    wrapped[7] += 2.0 * L;
    CHECK(energy_with(term, cell, wrapped) == Approx(base).margin(1e-9));
    std::vector<double> shifted = kPos;
    for (std::size_t i = 0; i < 4; ++i) { shifted[2 * i] += 0.37; shifted[2 * i + 1] -= 0.81; }
    CHECK(energy_with(term, cell, shifted) == Approx(base).margin(1e-9));

    // The cusp follows the kernels: bulk eps = 14 core within a layer, none across.
    const ParticleGroup e = grp("e", 0, 2, -1.0), h = grp("h", 2, 4, +1.0);
    CHECK(term.coulomb_cusp_residue(e, e) == Approx(1.0 / 14.0));
    CHECK(term.coulomb_cusp_residue(e, h) == 0.0);
}

TEST_CASE("a contrast stack matches the library's exact potential",
          "[ewald][screened]") {
    // A TMD-like slab (eps 14, thickness 6) in vacuum: the table must agree with
    // fast-screened-coulomb's own adaptive Fourier-Bessel integral (Rydberg,
    // hence the 1/2), the independent reference here.
    const std::vector<double> eps{1.0, 14.0, 14.0, 1.0}, iface{-3.0, 3.0, 9.0};
    const PotentialSpec k = build_screened_ewald_kernel(eps, iface, 2, 0, 0.0, 8.0);
    CHECK(k.epsilon == 1.0);  // (eps_bottom + eps_top)/2

    FSCContext* ctx = fsc_create(4, eps.data(), iface.data(), 2, 0, 0.0);
    REQUIRE(ctx != nullptr);
    // The library's own cutoff for a 1e-13 k-integration (its transfer matrices
    // overflow if pushed much further).
    double k_max = 0.0, rho_max = 0.0;
    int    n_quad = 0;
    fsc_optimal_params(ctx, 1e-13, 1e-3, 1.5, &k_max, &n_quad, &rho_max);
    for (double r : {0.3, 1.0, 3.0, 7.5}) {
        const double exact = 0.5 * fsc_potential(ctx, r, k_max, 1e-12);
        INFO("r = " << r << "  table " << k.v_real(r) << "  library " << exact);
        CHECK(k.v_real(r) == Approx(exact).epsilon(1e-7));
    }
    fsc_free(ctx);
}
