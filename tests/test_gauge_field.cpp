// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Magnetic field and rotating frame: a vector potential in the kinetic energy.
//
// Exact oracle: a charged particle in a 2-D parabolic dot (ω₀) with a
// perpendicular field B has the Fock–Darwin states.  The n = 0 member of
// angular momentum l,
//     φ_l = (x + i·sgn(l)·y)^{|l|} e^{−β r²/2},   β = m Ω,  Ω = √(ω₀² + ω_c²/4),
// is an eigenstate with E = (|l|+1) Ω − l ω_c/2 (ω_c = qB/m, signed), so its
// local energy is that constant at EVERY configuration.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <complex>
#include <vector>

#include "hamiltonian/hamiltonian.h"
#include "hamiltonian/harmonic_trap.h"
#include "hamiltonian/kinetic.h"
#include "population/particle_group.h"
#include "population/walker_set.h"
#include "wavefunction/complex_orbital_set.h"
#include "wavefunction/complex_slater_determinant.h"
#include "wavefunction/orbital_set.h"
#include "wavefunction/slater_determinant.h"
#include "wavefunction/trial_wavefunction.h"

using namespace alea;
using cplx = std::complex<double>;

TEST_CASE("Fock-Darwin orbitals match closed form and finite differences",
          "[gauge][fock_darwin]") {
    const std::size_t dim = 2, nw = 3;
    const std::vector<int> ls = {0, 1, -1, 2, -3};
    const double beta = 0.7;
    FockDarwinOrbitalSet orb(ls, beta);
    const std::size_t n = orb.size();
    REQUIRE(n == ls.size());

    std::vector<double> pos(dim * nw);
    for (std::size_t k = 0; k < pos.size(); ++k) pos[k] = 0.4 * static_cast<double>(k) - 0.7;
    std::vector<double> vr(n * nw), vi(n * nw), gr(n * dim * nw), gi(n * dim * nw),
        lr(n * dim * nw), li(n * dim * nw);
    orb.evaluate(pos, dim, {vr, vi}, {gr, gi}, {lr, li}, 0, nw);

    auto phi = [&](std::size_t i, double x, double y) {
        const int l = ls[i];
        const cplx w(x, l < 0 ? -y : y);
        return std::pow(w, std::abs(l)) * std::exp(-0.5 * beta * (x * x + y * y));
    };
    const double h = 1e-4;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t w = 0; w < nw; ++w) {
            const double x = pos[w], y = pos[nw + w];
            const cplx v = phi(i, x, y);
            CHECK(vr[i * nw + w] == Catch::Approx(v.real()).margin(1e-14));
            CHECK(vi[i * nw + w] == Catch::Approx(v.imag()).margin(1e-14));
            for (std::size_t d = 0; d < dim; ++d) {
                const double dx = d == 0 ? h : 0.0, dy = d == 1 ? h : 0.0;
                const cplx fp = phi(i, x + dx, y + dy), fm = phi(i, x - dx, y - dy);
                const cplx g = (fp - fm) / (2.0 * h), l = (fp - 2.0 * v + fm) / (h * h);
                const std::size_t idx = (i * dim + d) * nw + w;
                CHECK(gr[idx] == Catch::Approx(g.real()).margin(1e-6));
                CHECK(gi[idx] == Catch::Approx(g.imag()).margin(1e-6));
                CHECK(lr[idx] == Catch::Approx(l.real()).margin(1e-5));
                CHECK(li[idx] == Catch::Approx(l.imag()).margin(1e-5));
            }
        }
}

TEST_CASE("Fock-Darwin orbitals are regular at the origin", "[gauge][fock_darwin]") {
    // w^{|l|-2} and w^{|l|-1} appear with factors |l|(|l|-1) and |l|; at w = 0
    // they must not turn into 0·∞ = NaN.
    const std::size_t dim = 2, nw = 1;
    FockDarwinOrbitalSet orb({0, 1, -1, 2}, 0.5);
    std::vector<double> pos = {0.0, 0.0};
    std::vector<double> vr(4), vi(4), gr(8), gi(8), lr(8), li(8);
    orb.evaluate(pos, dim, {vr, vi}, {gr, gi}, {lr, li}, 0, nw);
    for (const std::vector<double>* a : {&vr, &vi, &gr, &gi, &lr, &li})
        for (double x : *a) CHECK(std::isfinite(x));
    CHECK(vr[0] == 1.0);                        // l = 0: e^0
    CHECK(gr[1 * 2 + 0] == Catch::Approx(1.0)); // l = 1: ∂x w = 1
    CHECK(gi[1 * 2 + 1] == Catch::Approx(1.0)); // l = 1: ∂y w = i
    CHECK(gi[2 * 2 + 1] == Catch::Approx(-1.0));// l = -1: ∂y w̄ = -i
}

namespace {

ParticleGroup grp(std::string name, std::size_t first, std::size_t last,
                  double charge, double mass) {
    ParticleGroup g;
    g.name = std::move(name);
    g.charge = charge;
    g.first = first;
    g.last = last;
    g.count = last - first;
    g.mass = {mass, mass};
    g.inv_mass = {1.0 / mass, 1.0 / mass};
    return g;
}

// ψ = Π_i φ_{l_i}(r_i): one 1×1 Fock–Darwin block per particle.
std::unique_ptr<WavefunctionComponent> fd_product(const std::vector<int>& ls,
                                                  const std::vector<double>& betas) {
    std::vector<ComplexSlaterDeterminant::Block> bl;
    for (std::size_t p = 0; p < ls.size(); ++p)
        bl.push_back({p, p + 1, std::make_unique<FockDarwinOrbitalSet>(
                                    std::vector<int>{ls[p]}, betas[p])});
    return std::make_unique<ComplexSlaterDeterminant>(std::move(bl));
}

// E_L at a few configurations of nw walkers.
std::vector<double> local_energies(TrialWavefunction& psi, Hamiltonian& H,
                                   std::size_t np, std::vector<double>* im = nullptr) {
    const std::size_t nw = 4, dim = 2;
    WalkerSet ws(nw, np, dim);
    for (std::size_t k = 0; k < ws.positions().size(); ++k)
        ws.positions()[k] = 1.3 * std::sin(1.7 * static_cast<double>(k) + 0.4);
    ws.dist_tables().build_all(ws.positions());
    psi.prepare(nw, np, dim);
    psi.evaluate_batch(ws);
    H.evaluate_batch(ws);
    if (im) im->assign(ws.local_energies_imag().begin(), ws.local_energies_imag().end());
    return {ws.local_energies().begin(), ws.local_energies().end()};
}

}  // namespace

TEST_CASE("Fock-Darwin states in a magnetic field have the exact local energy",
          "[gauge][magnetic]") {
    // Electron (q = −1), m = 0.7, dot ω₀ = 0.5, B = 1.3.
    const double m = 0.7, q = -1.0, w0 = 0.5, B = 1.3;
    const double wc = q * B / m;
    const double Om = std::sqrt(w0 * w0 + 0.25 * wc * wc);
    const std::vector<ParticleGroup> g = {grp("e", 0, 1, q, m)};

    for (int l : {0, 2, -1}) {
        Hamiltonian H;
        H.add_component(std::make_unique<Kinetic>(g, GaugeField{B, 0.0}));
        H.add_component(std::make_unique<HarmonicTrap>(g, w0 * w0));
        TrialWavefunction psi;
        psi.add_component(fd_product({l}, {m * Om}));
        std::vector<double> im;
        const auto e = local_energies(psi, H, 1, &im);
        const double exact = (std::abs(l) + 1) * Om - l * wc / 2.0;
        for (std::size_t w = 0; w < e.size(); ++w) {
            INFO("l = " << l);
            CHECK(e[w] == Catch::Approx(exact).epsilon(1e-10));
            CHECK(im[w] == Catch::Approx(0.0).margin(1e-10));
        }
    }
}

TEST_CASE("A real trial in a magnetic field still sees the |qA|^2/2m term",
          "[gauge][magnetic]") {
    // The l = 0 Fock–Darwin state is real: e^{−βr²/2} = GaussianOrbitalSet with
    // α = β/2.  No phase planes exist, yet E_L must still be Ω (the ½q²A²/m
    // piece is what squeezes the orbital from ω₀ to Ω).
    const double m = 0.7, q = -1.0, w0 = 0.5, B = 1.3;
    const double Om = std::sqrt(w0 * w0 + 0.25 * (q * B / m) * (q * B / m));
    const std::vector<ParticleGroup> g = {grp("e", 0, 1, q, m)};
    Hamiltonian H;
    H.add_component(std::make_unique<Kinetic>(g, GaugeField{B, 0.0}));
    H.add_component(std::make_unique<HarmonicTrap>(g, w0 * w0));
    std::vector<SlaterDeterminant::Block> bl;
    bl.push_back({0, 1, std::make_unique<GaussianOrbitalSet>(
                            2, std::vector<double>{0.0, 0.0}, std::vector<double>{0.5 * m * Om})});
    TrialWavefunction psi;
    psi.add_component(std::make_unique<SlaterDeterminant>(std::move(bl)));
    REQUIRE_FALSE(psi.is_complex());
    for (double e : local_energies(psi, H, 1))
        CHECK(e == Catch::Approx(Om).epsilon(1e-10));
}

TEST_CASE("Fock-Darwin states in a rotating frame have E = (|l|+1) w0 - Omega l",
          "[gauge][rotation]") {
    // H − Ω L_z = Σ(p − mΩ×r)²/2m − ½mΩ²r²: gauge coupling mΩ plus a
    // centrifugal trap of ω² = −Ω².  Rotation leaves the eigenstates alone.
    const double m = 1.3, w0 = 0.6, Om_rot = 0.25;
    const std::vector<ParticleGroup> g = {grp("a", 0, 1, 0.0, m)};
    for (int l : {0, 1, -2, 3}) {
        Hamiltonian H;
        H.add_component(std::make_unique<Kinetic>(g, GaugeField{0.0, Om_rot}));
        H.add_component(std::make_unique<HarmonicTrap>(g, w0 * w0));
        H.add_component(std::make_unique<HarmonicTrap>(g, -Om_rot * Om_rot));
        TrialWavefunction psi;
        psi.add_component(fd_product({l}, {m * w0}));
        std::vector<double> im;
        const auto e = local_energies(psi, H, 1, &im);
        for (std::size_t w = 0; w < e.size(); ++w) {
            INFO("l = " << l);
            CHECK(e[w] == Catch::Approx((std::abs(l) + 1) * w0 - Om_rot * l).epsilon(1e-10));
            CHECK(im[w] == Catch::Approx(0.0).margin(1e-10));
        }
    }
}

TEST_CASE("The gauge coupling is per particle (charge and mass)",
          "[gauge][magnetic]") {
    // Electron (q=−1, m=0.4) and hole (q=+1, m=1.1), non-interacting, same dot
    // and field: E = E_e(l_e) + E_h(l_h), each with its own ω_c and Ω.
    const double w0 = 0.5, B = 0.9;
    const std::vector<ParticleGroup> g = {grp("e", 0, 1, -1.0, 0.4),
                                          grp("h", 1, 2, +1.0, 1.1)};
    auto level = [&](double q, double m, int l) {
        const double wc = q * B / m, Om = std::sqrt(w0 * w0 + 0.25 * wc * wc);
        return std::pair{(std::abs(l) + 1) * Om - l * wc / 2.0, m * Om};
    };
    const auto [Ee, be] = level(-1.0, 0.4, 1);
    const auto [Eh, bh] = level(+1.0, 1.1, -2);
    Hamiltonian H;
    H.add_component(std::make_unique<Kinetic>(g, GaugeField{B, 0.0}));
    H.add_component(std::make_unique<HarmonicTrap>(g, w0 * w0));
    TrialWavefunction psi;
    psi.add_component(fd_product({1, -2}, {be, bh}));
    for (double e : local_energies(psi, H, 2))
        CHECK(e == Catch::Approx(Ee + Eh).epsilon(1e-10));
}
