// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Twisted boundary conditions: complex plane-wave orbitals e^{i(k+θ)·r} and
// the complex Slater determinant that turns them into a fixed-phase trial.
//
// Oracles are exact: plane waves are eigenfunctions of ∇², so a determinant of
// them is an eigenfunction of the free kinetic energy at ANY configuration.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>

#include <cmath>
#include <complex>
#include <vector>

#include "hamiltonian/hamiltonian.h"
#include "hamiltonian/kinetic.h"
#include "population/particle_group.h"
#include "population/simulation_cell.h"
#include "population/walker.h"
#include "population/walker_set.h"
#include "wavefunction/complex_orbital_set.h"
#include "wavefunction/complex_slater_determinant.h"
#include "wavefunction/trial_wavefunction.h"

using namespace alea;
using cplx = std::complex<double>;

namespace {

// Test geometry: 3 particles in 2D.  Block A = particles [0,2) with 2 orbitals,
// block B = particle [2,3) with 1 orbital.  Arbitrary (non-lattice) q: the
// determinant algebra does not care whether the orbitals are periodic.
const std::vector<double> kQA = {0.7, -0.2, 1.3, 0.4};
const std::vector<double> kQB = {-0.5, 0.9};

std::unique_ptr<ComplexSlaterDeterminant> make_cdet() {
    std::vector<ComplexSlaterDeterminant::Block> blocks;
    blocks.push_back({0, 2, std::make_unique<TwistedPlaneWaveOrbitalSet>(2, kQA)});
    blocks.push_back({2, 3, std::make_unique<TwistedPlaneWaveOrbitalSet>(2, kQB)});
    return std::make_unique<ComplexSlaterDeterminant>(std::move(blocks));
}

// Oracle ψ(R) for one walker, R = [p*2 + d].
cplx psi_ref(const std::vector<double>& R) {
    auto pw = [](const std::vector<double>& q, std::size_t i, double x, double y) {
        return std::exp(cplx(0.0, q[2 * i] * x + q[2 * i + 1] * y));
    };
    const cplx detA = pw(kQA, 0, R[0], R[1]) * pw(kQA, 1, R[2], R[3]) -
                      pw(kQA, 1, R[0], R[1]) * pw(kQA, 0, R[2], R[3]);
    const cplx detB = pw(kQB, 0, R[4], R[5]);
    return detA * detB;
}

// ∂_{p,d} ln ψ and ∂²_{p,d} ln ψ by central differences of ln(ψ±/ψ0).
void dlnpsi_fd(const std::vector<double>& R, std::size_t p, std::size_t d,
               cplx& g, cplx& l) {
    const double h = 1e-4;
    std::vector<double> Rp = R, Rm = R;
    Rp[p * 2 + d] += h;
    Rm[p * 2 + d] -= h;
    const cplx f0 = psi_ref(R), fp = psi_ref(Rp), fm = psi_ref(Rm);
    g = std::log(fp / fm) / (2.0 * h);
    l = std::log(fp * fm / (f0 * f0)) / (h * h);
}

// One-walker AoS configuration → WalkerSet column w.
void put(WalkerSet& ws, std::size_t w, const std::vector<double>& R) {
    const std::size_t nw = ws.num_walkers();
    for (std::size_t p = 0; p < 3; ++p)
        for (std::size_t d = 0; d < 2; ++d)
            ws.positions()[p * 2 * nw + d * nw + w] = R[p * 2 + d];
}

std::vector<double> config(std::size_t w) {
    std::vector<double> R(6);
    for (std::size_t k = 0; k < 6; ++k)
        R[k] = 0.9 * std::sin(1.7 * static_cast<double>(k) + 0.8 * static_cast<double>(w));
    return R;
}

}  // namespace

TEST_CASE("Twisted plane waves are e^{iq.r} with closed-form derivatives",
          "[twist]") {
    const std::size_t dim = 2, nw = 3, n = 2;
    const std::vector<double> q = {0.7, -0.2, 1.3, 0.4};   // [n * dim]
    TwistedPlaneWaveOrbitalSet orb(dim, q);
    REQUIRE(orb.size() == n);

    std::vector<double> pos(dim * nw);
    for (std::size_t k = 0; k < pos.size(); ++k) pos[k] = 0.3 * static_cast<double>(k) - 0.4;
    std::vector<double> vr(n * nw), vi(n * nw), gr(n * dim * nw), gi(n * dim * nw),
        lr(n * dim * nw), li(n * dim * nw);
    orb.evaluate(pos, dim, {vr, vi}, {gr, gi}, {lr, li}, 0, nw);

    auto phi = [&](std::size_t i, double x, double y) {
        return std::exp(cplx(0.0, q[i * dim] * x + q[i * dim + 1] * y));
    };
    const double h = 1e-4;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t w = 0; w < nw; ++w) {
            const double x = pos[w], y = pos[nw + w];
            const cplx v = phi(i, x, y);
            CHECK(vr[i * nw + w] == Catch::Approx(v.real()));
            CHECK(vi[i * nw + w] == Catch::Approx(v.imag()));
            for (std::size_t d = 0; d < dim; ++d) {
                const double dx = d == 0 ? h : 0.0, dy = d == 1 ? h : 0.0;
                const cplx fp = phi(i, x + dx, y + dy), fm = phi(i, x - dx, y - dy);
                const cplx g_fd = (fp - fm) / (2.0 * h);
                const cplx l_fd = (fp - 2.0 * v + fm) / (h * h);
                const std::size_t idx = (i * dim + d) * nw + w;
                CHECK(gr[idx] == Catch::Approx(g_fd.real()).margin(1e-6));
                CHECK(gi[idx] == Catch::Approx(g_fd.imag()).margin(1e-6));
                CHECK(lr[idx] == Catch::Approx(l_fd.real()).margin(1e-5));
                CHECK(li[idx] == Catch::Approx(l_fd.imag()).margin(1e-5));
            }
        }
}

TEST_CASE("A complex determinant gives ln|psi|, grad and lap of ln psi (re and im)",
          "[twist][cdet]") {
    const std::size_t nw = 2, np = 3, dim = 2;
    WalkerSet ws(nw, np, dim);
    for (std::size_t w = 0; w < nw; ++w) put(ws, w, config(w));
    ws.dist_tables().build_all(ws.positions());

    TrialWavefunction psi;
    psi.add_component(make_cdet());
    REQUIRE(psi.is_complex());
    psi.prepare(nw, np, dim);
    psi.evaluate_batch(ws);

    for (std::size_t w = 0; w < nw; ++w) {
        const std::vector<double> R = config(w);
        CHECK(ws.log_psi()[w] == Catch::Approx(std::log(std::abs(psi_ref(R)))));
        for (std::size_t p = 0; p < np; ++p)
            for (std::size_t d = 0; d < dim; ++d) {
                cplx g, l;
                dlnpsi_fd(R, p, d, g, l);
                const std::size_t idx = p * dim * nw + d * nw + w;
                CHECK(ws.gradients()[idx]        == Catch::Approx(g.real()).margin(1e-6));
                CHECK(ws.phase_gradients()[idx]  == Catch::Approx(g.imag()).margin(1e-6));
                CHECK(ws.laplacians()[idx]       == Catch::Approx(l.real()).margin(1e-4));
                CHECK(ws.phase_laplacians()[idx] == Catch::Approx(l.imag()).margin(1e-4));
            }
    }
}

TEST_CASE("A complex determinant's move ratio carries |R| and arg R",
          "[twist][cdet]") {
    const std::size_t nw = 2, np = 3, dim = 2;
    WalkerSet ws(nw, np, dim);
    for (std::size_t w = 0; w < nw; ++w) put(ws, w, config(w));
    ws.dist_tables().build_all(ws.positions());

    TrialWavefunction psi;
    psi.add_component(make_cdet());
    psi.prepare(nw, np, dim);
    psi.evaluate_batch(ws);

    for (std::size_t p = 0; p < np; ++p) {
        std::vector<double> new_pos(dim * nw);
        for (std::size_t d = 0; d < dim; ++d)
            for (std::size_t w = 0; w < nw; ++w)
                new_pos[d * nw + w] = config(w)[p * 2 + d] + 0.31 - 0.17 * static_cast<double>(d + w);
        ws.propose_move_batch(p, new_pos);
        psi.ratio_batch(ws, p);
        for (std::size_t w = 0; w < nw; ++w) {
            std::vector<double> R = config(w), Rn = R;
            for (std::size_t d = 0; d < dim; ++d) Rn[p * 2 + d] = new_pos[d * nw + w];
            const cplx ratio = psi_ref(Rn) / psi_ref(R);
            CHECK(ws.log_ratios()[w] ==
                  Catch::Approx(std::log(std::abs(ratio))).margin(1e-12));
            // arg compared through e^{i·arg}: phase_diffs is defined mod 2π.
            const cplx u = std::exp(cplx(0.0, ws.phase_diffs()[w]));
            CHECK(u.real() == Catch::Approx(std::cos(std::arg(ratio))).margin(1e-10));
            CHECK(u.imag() == Catch::Approx(std::sin(std::arg(ratio))).margin(1e-10));
            for (std::size_t d = 0; d < dim; ++d) {
                cplx g, l;
                dlnpsi_fd(Rn, p, d, g, l);
                CHECK(ws.new_grad()[d * nw + w]       == Catch::Approx(g.real()).margin(1e-6));
                CHECK(ws.new_phase_grad()[d * nw + w] == Catch::Approx(g.imag()).margin(1e-6));
                CHECK(ws.new_lap()[d * nw + w]        == Catch::Approx(l.real()).margin(1e-4));
                CHECK(ws.new_phase_lap()[d * nw + w]  == Catch::Approx(l.imag()).margin(1e-4));
            }
        }
    }
}

TEST_CASE("Incremental moves, wraps and branching match a fresh evaluation",
          "[twist][cdet]") {
    // Periodic square cell, twisted lattice plane waves q = k + θ.  Walkers are
    // wrapped back into the cell on acceptance: under twisted BCs that changes
    // a determinant row by a constant phase (a gauge), so ln|ψ|, ∇ln ψ and
    // ∂²ln ψ — everything the cache holds — must still equal a fresh evaluate.
    const std::size_t nw = 3, np = 3, dim = 2;
    const double L = 4.0, b = 2.0 * 3.14159265358979323846 / L;
    const std::vector<double> cellv = {L, 0.0, 0.0, L};
    const SimulationCell cell(dim, cellv);
    const double th[2] = {0.31 * b, -0.17 * b};
    auto twisted = [&](std::vector<std::pair<int, int>> ks) {
        std::vector<double> q;
        for (auto [i, j] : ks) { q.push_back(i * b + th[0]); q.push_back(j * b + th[1]); }
        return std::make_unique<TwistedPlaneWaveOrbitalSet>(dim, q);
    };
    auto make = [&] {
        std::vector<ComplexSlaterDeterminant::Block> bl;
        bl.push_back({0, 2, twisted({{0, 0}, {1, 0}})});
        bl.push_back({2, 3, twisted({{0, 0}})});
        return std::make_unique<ComplexSlaterDeterminant>(std::move(bl));
    };

    WalkerSet ws(nw, np, dim, cell);
    std::vector<Walker> walkers(nw, Walker(np, dim));
    for (Walker& w : walkers) ws.add_walker(w);
    for (std::size_t k = 0; k < ws.positions().size(); ++k)
        ws.positions()[k] = 2.0 + 1.7 * std::sin(0.9 * static_cast<double>(k));
    ws.dist_tables().build_all(ws.positions());

    TrialWavefunction psi;
    psi.add_component(make());
    psi.prepare(nw, np, dim);
    psi.evaluate_batch(ws);

    auto check_fresh = [&] {
        const std::size_t n = ws.num_walkers();
        WalkerSet fresh(n, np, dim, cell);
        for (std::size_t k = 0; k < fresh.positions().size(); ++k)
            fresh.positions()[k] = ws.positions()[k];
        fresh.dist_tables().build_all(fresh.positions());
        TrialWavefunction psi2;
        psi2.add_component(make());
        psi2.prepare(n, np, dim);
        psi2.evaluate_batch(fresh);
        for (std::size_t w = 0; w < n; ++w)
            CHECK(ws.log_psi()[w] == Catch::Approx(fresh.log_psi()[w]).margin(1e-10));
        for (std::size_t k = 0; k < fresh.gradients().size(); ++k) {
            CHECK(ws.gradients()[k]        == Catch::Approx(fresh.gradients()[k]).margin(1e-9));
            CHECK(ws.laplacians()[k]       == Catch::Approx(fresh.laplacians()[k]).margin(1e-8));
            CHECK(ws.phase_gradients()[k]  == Catch::Approx(fresh.phase_gradients()[k]).margin(1e-9));
            CHECK(ws.phase_laplacians()[k] == Catch::Approx(fresh.phase_laplacians()[k]).margin(1e-8));
        }
    };

    bool accept[3] = {true, false, true};
    int n_wraps = 0;
    for (int sweep = 0; sweep < 3; ++sweep) {
        for (std::size_t p = 0; p < np; ++p) {
            const std::size_t n = ws.num_walkers();
            std::vector<double> new_pos(dim * n);
            for (std::size_t d = 0; d < dim; ++d)
                for (std::size_t w = 0; w < n; ++w)
                    // Large steps: most accepted moves leave the cell and wrap.
                    new_pos[d * n + w] = ws.positions()[p * dim * n + d * n + w] +
                                         2.3 * std::cos(1.1 * static_cast<double>(sweep * 7 + p * 3 + d + w));
            for (std::size_t k = 0; k < new_pos.size(); ++k)
                if (accept[k % n] && (new_pos[k] < 0.0 || new_pos[k] >= L)) ++n_wraps;
            ws.propose_move_batch(p, new_pos);
            psi.ratio_batch(ws, p);
            const std::span<const bool> mask(accept, n);
            psi.apply_move_batch(ws, p, mask);
            ws.apply_move(p, new_pos, ws.new_grad(), ws.new_lap(), mask);
            check_fresh();
        }
        std::swap(accept[0], accept[1]);
    }
    REQUIRE(n_wraps > 0);   // the gauge-under-wrap claim was actually exercised

    // Branching: reorder + duplicate, carrying the caches, then keep moving.
    const std::vector<std::size_t> src = {2, 0, 2};
    ws.apply_permutation(src);
    psi.permute_caches(src);
    check_fresh();
    std::vector<double> new_pos(dim * nw);
    for (std::size_t d = 0; d < dim; ++d)
        for (std::size_t w = 0; w < nw; ++w)
            new_pos[d * nw + w] = ws.positions()[1 * dim * nw + d * nw + w] + 0.4 + 0.1 * static_cast<double>(w);
    ws.propose_move_batch(1, new_pos);
    psi.ratio_batch(ws, 1);
    bool all[3] = {true, true, true};
    psi.apply_move_batch(ws, 1, std::span<const bool>(all, nw));
    ws.apply_move(1, new_pos, ws.new_grad(), ws.new_lap(), std::span<const bool>(all, nw));
    check_fresh();
}

TEST_CASE("A twisted free Fermi gas has E_L = sum |k+theta|^2/2 at every configuration",
          "[twist][cdet]") {
    // Plane waves are eigenfunctions of −∇²/2, so their determinant is too:
    // Re E_L is the constant Σ_i |q_i|²/2 and Im E_L ≡ 0, whatever R is.
    const std::size_t nw = 4, np = 3, dim = 2;
    const double L = 4.0, b = 2.0 * 3.14159265358979323846 / L;
    const std::vector<double> cellv = {L, 0.0, 0.0, L};
    const SimulationCell cell(dim, cellv);
    const double th[2] = {0.31 * b, -0.17 * b};
    std::vector<double> q_up = {th[0], th[1], b + th[0], th[1]};
    std::vector<double> q_dn = {th[0], th[1]};
    const double exact = 0.5 * (q_up[0] * q_up[0] + q_up[1] * q_up[1] + q_up[2] * q_up[2] +
                                q_up[3] * q_up[3] + q_dn[0] * q_dn[0] + q_dn[1] * q_dn[1]);

    std::vector<ComplexSlaterDeterminant::Block> bl;
    bl.push_back({0, 2, std::make_unique<TwistedPlaneWaveOrbitalSet>(dim, q_up)});
    bl.push_back({2, 3, std::make_unique<TwistedPlaneWaveOrbitalSet>(dim, q_dn)});
    TrialWavefunction psi;
    psi.add_component(std::make_unique<ComplexSlaterDeterminant>(std::move(bl)));

    ParticleGroup e;
    e.name = "e"; e.charge = -1.0; e.first = 0; e.last = 3; e.count = 3;
    e.mass = {1.0, 1.0}; e.inv_mass = {1.0, 1.0};
    Hamiltonian H;
    H.add_component(std::make_unique<Kinetic>(std::vector<ParticleGroup>{e}));

    WalkerSet ws(nw, np, dim, cell);
    for (std::size_t k = 0; k < ws.positions().size(); ++k)
        ws.positions()[k] = 2.0 + 1.9 * std::sin(2.3 * static_cast<double>(k) + 0.1);
    ws.dist_tables().build_all(ws.positions());
    psi.prepare(nw, np, dim);
    psi.evaluate_batch(ws);
    H.evaluate_batch(ws);

    for (std::size_t w = 0; w < nw; ++w) {
        CHECK(ws.local_energies()[w] == Catch::Approx(exact).epsilon(1e-10));
        CHECK(ws.local_energies_imag()[w] == Catch::Approx(0.0).margin(1e-10));
    }
}

namespace {
std::vector<std::pair<double, double>> sorted_q(const TwistedPlaneWaveOrbitalSet& o) {
    std::vector<std::pair<double, double>> v;
    const auto q = o.wavevectors();
    for (std::size_t i = 0; i < o.size(); ++i) v.emplace_back(q[2 * i], q[2 * i + 1]);
    std::sort(v.begin(), v.end());
    return v;
}
}  // namespace

TEST_CASE("Twisted occupation fills the N lowest |k+theta|", "[twist][occupation]") {
    const double L = 4.0, b = 2.0 * 3.14159265358979323846 / L;
    const std::vector<double> cellv = {L, 0.0, 0.0, L};
    const SimulationCell cell(2, cellv);
    const std::vector<double> t = {0.2, 0.1};   // θ = 0.2 b1 + 0.1 b2

    auto orb = make_twisted_plane_wave_orbitals(cell, 4, t);
    REQUIRE(orb->size() == 4);
    // |k+θ|²/b² for k = (i,j)·b, θ = (0.2,0.1)·b:
    //   (0,0) 0.05   (−1,0) 0.65   (0,−1) 0.85   (0,1) 1.25   (1,0) 1.45 ...
    // so the four occupied are (0,0), (−1,0), (0,−1), (0,1).
    std::vector<std::pair<double, double>> want = {
        {0.2 * b, 0.1 * b}, {-0.8 * b, 0.1 * b}, {0.2 * b, -0.9 * b}, {0.2 * b, 1.1 * b}};
    std::sort(want.begin(), want.end());
    const auto got = sorted_q(*orb);
    for (std::size_t i = 0; i < 4; ++i) {
        CHECK(got[i].first  == Catch::Approx(want[i].first));
        CHECK(got[i].second == Catch::Approx(want[i].second));
    }
    CHECK(orb->kinetic_energy_sum() ==
          Catch::Approx(0.5 * b * b * (0.05 + 0.65 + 0.85 + 1.25)));
}

TEST_CASE("A twist and the same twist shifted by a reciprocal vector are one state",
          "[twist][occupation]") {
    const double L = 4.0;
    const std::vector<double> cellv = {L, 0.0, 0.0, L};
    const SimulationCell cell(2, cellv);
    const auto a = sorted_q(*make_twisted_plane_wave_orbitals(cell, 6, {0.2, 0.1}));
    const auto c = sorted_q(*make_twisted_plane_wave_orbitals(cell, 6, {1.2, -0.9}));
    REQUIRE(a.size() == c.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i].first  == Catch::Approx(c[i].first));
        CHECK(a[i].second == Catch::Approx(c[i].second));
    }
}

TEST_CASE("An occupation that splits a degenerate |k+theta| shell is refused",
          "[twist][occupation]") {
    const std::vector<double> cellv = {4.0, 0.0, 0.0, 4.0};
    const SimulationCell cell(2, cellv);
    // θ = b1/2: k = 0 and k = −b1 give the same |k+θ|, so one orbital is ambiguous.
    REQUIRE_THROWS_WITH(make_twisted_plane_wave_orbitals(cell, 1, {0.5, 0.0}),
                        Catch::Matchers::ContainsSubstring("degenerate"));
    REQUIRE_NOTHROW(make_twisted_plane_wave_orbitals(cell, 2, {0.5, 0.0}));
}
