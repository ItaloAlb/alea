// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

#include "population/particle_group.h"
#include "population/walker_set.h"
#include "hamiltonian/kinetic.h"

using alea::Kinetic;
using alea::ParticleGroup;
using alea::WalkerSet;

namespace {

ParticleGroup kgroup(std::size_t first, std::size_t last, double inv_m) {
    ParticleGroup g;
    g.name = "g";
    g.first = first;
    g.last = last;
    g.inv_mass = {inv_m, inv_m};
    return g;
}

}  // namespace

TEST_CASE("Kinetic accumulates -1/2 (1/m)(lap + |grad|^2) per particle",
          "[kinetic]") {
    // One particle, dim=2, inv_mass=(1,1). grad=(1,0) → |grad|²=1.
    // Isotropic lap: ∂²_x = ∂²_y = -1.0 (total -2.0).
    // K = -1/2 * [1*(-1 + 1) + 1*(-1 + 0)] = 0.5.
    WalkerSet ws(/*nw=*/1, /*np=*/1, /*dim=*/2);
    auto g = ws.gradients();   // [p*dim*nw + d*nw + w], nw=1
    g[0] = 1.0;                // p0, x
    g[1] = 0.0;                // p0, y
    auto l = ws.laplacians();  // [p*dim*nw + d*nw + w]
    l[0] = -1.0;               // p0, x
    l[1] = -1.0;               // p0, y

    Kinetic kin({kgroup(0, 1, /*inv_m=*/1.0)});
    kin.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(0.5));
}

TEST_CASE("Kinetic scales each particle by its inverse mass", "[kinetic]") {
    // Same as above but inv_mass=(0.5, 0.5) → K = 0.25.
    WalkerSet ws(/*nw=*/1, /*np=*/1, /*dim=*/2);
    auto g = ws.gradients();
    g[0] = 1.0;
    g[1] = 0.0;
    auto l = ws.laplacians();
    l[0] = -1.0;
    l[1] = -1.0;

    Kinetic kin({kgroup(0, 1, /*inv_m=*/0.5)});
    kin.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(0.25));
}

TEST_CASE("Kinetic sums over particles using per-group inverse masses",
          "[kinetic]") {
    // p0: grad=(1,0), |grad|²=1,  lap=(-1,-1),  inv_m=(1,1)    →  0.5
    // p1: grad=(0,2), |grad|²=4,  lap=(-1.5,-1.5), inv_m=(0.5,0.5) → -0.25
    // total = 0.25
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto g = ws.gradients();   // [p*dim*nw + d*nw + w], nw=1
    g[0] = 1.0;   // p0 x
    g[1] = 0.0;   // p0 y
    g[2] = 0.0;   // p1 x
    g[3] = 2.0;   // p1 y
    auto l = ws.laplacians();
    l[0] = -1.0;   // p0 x
    l[1] = -1.0;   // p0 y
    l[2] = -1.5;   // p1 x
    l[3] = -1.5;   // p1 y

    Kinetic kin({kgroup(0, 1, 1.0), kgroup(1, 2, 0.5)});
    kin.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(0.25));
}

TEST_CASE("Kinetic handles anisotropic inverse mass", "[kinetic][anisotropy]") {
    // One particle, dim=2. inv_mass = (1.0, 0.5). grad=(1,2), |grad|²=(1,4).
    // lap = (-0.5, -0.3).
    // K = -1/2 * [1.0*(-0.5 + 1) + 0.5*(-0.3 + 4)] = -1/2*[0.5 + 1.85] = -1.175
    WalkerSet ws(/*nw=*/1, /*np=*/1, /*dim=*/2);
    auto g = ws.gradients();
    g[0] = 1.0;
    g[1] = 2.0;
    auto l = ws.laplacians();
    l[0] = -0.5;
    l[1] = -0.3;

    ParticleGroup pg;
    pg.name = "g";
    pg.first = 0;
    pg.last = 1;
    pg.inv_mass = {1.0, 0.5};
    Kinetic kin({pg});
    kin.evaluate_batch(ws);

    CHECK(ws.local_energies()[0] == Catch::Approx(-1.175));
}

namespace {

// Deterministic, irregular values in [−1, 1).
double fill_value(std::size_t k, double salt) {
    return std::fmod(0.7548776662 * static_cast<double>(k + 1) + salt, 2.0) - 1.0;
}

// The four planes ∇u, ∇²u, ∇φ, ∇²φ of a 2-particle, 3-walker batch, each
// moved by eps times its own direction (dir[plane]), and Re/Im E_kin there.
struct Planes {
    std::vector<double> v[4];
};

void kinetic_at(Kinetic& kin, WalkerSet& ws, const Planes& base, const Planes& dir,
                std::size_t k, std::size_t field, double eps, std::vector<double>& re,
                std::vector<double>& im) {
    const std::span<double> dst[4] = {ws.gradients(), ws.laplacians(), ws.phase_gradients(),
                                      ws.phase_laplacians()};
    for (std::size_t p = 0; p < 4; ++p)
        for (std::size_t e = 0; e < field; ++e)
            dst[p][e] = base.v[p][e] + eps * dir.v[p][k * field + e];
    for (double& e : ws.local_energies()) e = 0.0;
    for (double& e : ws.local_energies_imag()) e = 0.0;
    kin.evaluate_batch(ws);
    re.assign(ws.local_energies().begin(), ws.local_energies().end());
    im.assign(ws.local_energies_imag().begin(), ws.local_energies_imag().end());
}

void check_complex_parameter_derivatives(const alea::GaugeField& gauge) {
    const std::size_t nw = 3, np = 2, dim = 2, n_opt = 2, field = np * dim * nw;
    WalkerSet ws(nw, np, dim);
    ws.enable_phase();
    for (std::size_t e = 0; e < ws.positions().size(); ++e)
        ws.positions()[e] = 2.0 * fill_value(e, 0.31);
    ParticleGroup a = kgroup(0, 1, 1.0), b = kgroup(1, 2, 0.5);
    a.charge = -1.0;
    b.charge = 1.0;
    Kinetic kin({a, b}, gauge);

    Planes base, dir;
    for (std::size_t p = 0; p < 4; ++p) {
        base.v[p].resize(field);
        dir.v[p].resize(n_opt * field);
        for (std::size_t e = 0; e < field; ++e) base.v[p][e] = fill_value(e, 0.1 * p);
        for (std::size_t e = 0; e < n_opt * field; ++e) dir.v[p][e] = fill_value(e, 0.5 + 0.1 * p);
    }
    const std::span<double> dst[4] = {ws.gradients(), ws.laplacians(), ws.phase_gradients(),
                                      ws.phase_laplacians()};
    for (std::size_t p = 0; p < 4; ++p)
        std::copy(base.v[p].begin(), base.v[p].end(), dst[p].begin());
    std::vector<double> out_re(n_opt * nw), out_im(n_opt * nw);
    kin.complex_local_energy_parameter_derivatives(ws, dir.v[0], dir.v[1], dir.v[2], dir.v[3],
                                                   n_opt, out_re, out_im);

    // E_kin is quadratic in the planes, so a central difference is exact up
    // to rounding.
    const double eps = 1e-3;
    for (std::size_t k = 0; k < n_opt; ++k) {
        std::vector<double> rp, ip, rm, im;
        kinetic_at(kin, ws, base, dir, k, field, eps, rp, ip);
        kinetic_at(kin, ws, base, dir, k, field, -eps, rm, im);
        for (std::size_t w = 0; w < nw; ++w) {
            INFO("parameter " << k << " walker " << w);
            CHECK(out_re[k * nw + w] == Catch::Approx((rp[w] - rm[w]) / (2 * eps)).margin(1e-10));
            CHECK(out_im[k * nw + w] == Catch::Approx((ip[w] - im[w]) / (2 * eps)).margin(1e-10));
        }
    }
}

}  // namespace

TEST_CASE("Kinetic complex parameter derivatives are those of Re and Im E_L along the planes",
          "[kinetic][optimize]") {
    SECTION("no field") { check_complex_parameter_derivatives({}); }
    SECTION("magnetic field and rotation") { check_complex_parameter_derivatives({0.7, 0.2}); }
}

TEST_CASE("Kinetic complex parameter derivatives of a real factor reduce to the real ones",
          "[kinetic][optimize]") {
    // ∂_p ∇φ = ∂_p ∇²φ = 0 (a Jastrow of a complex trial): the real part is
    // −½(1/m)[l̇ + 2 ∂u ġ], whatever the phase planes hold.
    const std::size_t nw = 2, np = 2, dim = 2, n_opt = 1, field = np * dim * nw;
    WalkerSet ws(nw, np, dim);
    ws.enable_phase();
    for (std::size_t e = 0; e < field; ++e) {
        ws.gradients()[e] = fill_value(e, 0.2);
        ws.phase_gradients()[e] = fill_value(e, 0.4);
    }
    std::vector<double> dg(field), dl(field), zero(field, 0.0);
    for (std::size_t e = 0; e < field; ++e) {
        dg[e] = fill_value(e, 0.6);
        dl[e] = fill_value(e, 0.8);
    }
    Kinetic kin({kgroup(0, 1, 1.0), kgroup(1, 2, 0.5)});
    std::vector<double> re(n_opt * nw), im(n_opt * nw), real_only(n_opt * nw, 0.0);
    kin.complex_local_energy_parameter_derivatives(ws, dg, dl, zero, zero, n_opt, re, im);
    kin.local_energy_parameter_derivatives(ws, dg, dl, n_opt, real_only);
    for (std::size_t w = 0; w < nw; ++w) CHECK(re[w] == Catch::Approx(real_only[w]).epsilon(1e-14));
}
