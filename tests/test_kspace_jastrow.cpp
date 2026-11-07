// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Free-coefficient reciprocal-space Jastrow: ln Ψ ⊃ Σ_s p_s Σ_{k∈s} T_k over the
// first |k| shells of the cell, T_k = |ρ_A|² (one group) or 2 Re ρ_A ρ_B* (two).
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <tuple>
#include <memory>
#include <random>
#include <span>
#include <vector>

#include "numerics/gaskell.h"
#include "population/simulation_cell.h"
#include "population/walker_set.h"
#include "wavefunction/kspace_jastrow.h"
#include "wavefunction/trial_wavefunction.h"

using alea::KSpaceJastrow;
using alea::Parameter;
using alea::ParticleGroup;
using alea::SimulationCell;
using alea::TrialWavefunction;
using alea::WalkerSet;
using Catch::Approx;

namespace {

SimulationCell hexagonal(double L) {
    const std::vector<double> a = {L, 0.0, 0.5 * L, L * std::sqrt(3.0) / 2.0};
    return SimulationCell(2, std::span<const double>(a));
}

ParticleGroup group(const char* name, std::size_t first, std::size_t last) {
    ParticleGroup g;
    g.name = name;
    g.first = first;
    g.last = last;
    return g;
}

constexpr std::size_t kNp = 5;
const ParticleGroup kA = group("a", 0, 3);
const ParticleGroup kB = group("b", 3, 5);

// Three shells, distinct coefficients, all free.
std::vector<Parameter> three_shells() {
    return {{"p1", -0.30, false}, {"p2", 0.12, false}, {"p3", -0.05, false}};
}

void scatter(WalkerSet& ws, const SimulationCell& cell, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const auto a = cell.vectors();
    const std::size_t nw = ws.num_walkers();
    auto pos = ws.positions();
    for (std::size_t w = 0; w < nw; ++w)
        for (std::size_t p = 0; p < ws.n_particles(); ++p) {
            const double f1 = u(rng), f2 = u(rng);
            pos[(p * 2 + 0) * nw + w] = f1 * a[0] + f2 * a[2];
            pos[(p * 2 + 1) * nw + w] = f1 * a[1] + f2 * a[3];
        }
    ws.dist_tables().build_all(ws.positions());
}

// Half-space reciprocal vectors of the cell grouped into the first n shells.
struct Shells { std::vector<std::vector<std::pair<double, double>>> k; };
Shells first_shells(const SimulationCell& cell, std::size_t n) {
    const auto b = cell.reciprocal();
    std::vector<std::tuple<double, double, double>> ks;
    for (int i = 0; i <= 6; ++i)
        for (int j = -6; j <= 6; ++j) {
            if (i == 0 && j <= 0) continue;
            const double kx = i * b[0] + j * b[2], ky = i * b[1] + j * b[3];
            ks.emplace_back(std::hypot(kx, ky), kx, ky);
        }
    std::sort(ks.begin(), ks.end());
    Shells s;
    double last = -1.0;
    for (const auto& [k, kx, ky] : ks) {
        if (k - last > 1e-9) {
            if (s.k.size() == n) break;
            s.k.emplace_back();
            last = k;
        }
        s.k.back().emplace_back(kx, ky);
    }
    return s;
}

}  // namespace

TEST_CASE("kspace jastrow log psi is the shell sum of T_k", "[kspace_jastrow]") {
    const SimulationCell cell = hexagonal(5.0);
    std::mt19937_64 rng(1);
    WalkerSet ws(2, kNp, 2, cell);
    scatter(ws, cell, rng);
    const Shells sh = first_shells(cell, 3);
    const auto p = three_shells();

    for (const bool same : {true, false}) {
        INFO((same ? "same group" : "two groups"));
        const ParticleGroup& gb = same ? kA : kB;
        KSpaceJastrow j(cell, kA, gb, 3, p);
        j.evaluate_batch(ws);
        for (std::size_t w = 0; w < 2; ++w) {
            auto rho = [&](const ParticleGroup& g, double kx, double ky) {
                std::complex<double> r = 0.0;
                for (std::size_t i = g.first; i < g.last; ++i)
                    r += std::exp(std::complex<double>(
                        0.0, kx * ws.positions()[(i * 2) * 2 + w] +
                                 ky * ws.positions()[(i * 2 + 1) * 2 + w]));
                return r;
            };
            double expect = 0.0;
            for (std::size_t s = 0; s < 3; ++s)
                for (const auto& [kx, ky] : sh.k[s])
                    expect += p[s].value * (same ? std::norm(rho(kA, kx, ky))
                                                 : 2.0 * (rho(kA, kx, ky) *
                                                          std::conj(rho(kB, kx, ky))).real());
            CHECK(j.log_psi_c()[w] == Approx(expect).epsilon(1e-12));
        }
    }
}

TEST_CASE("kspace jastrow gradient and laplacian are the derivatives of its log",
          "[kspace_jastrow]") {
    const SimulationCell cell = hexagonal(5.0);
    std::mt19937_64 rng(2);
    WalkerSet base(1, kNp, 2, cell);
    scatter(base, cell, rng);
    const std::vector<double> x0(base.positions().begin(), base.positions().end());
    for (const bool same : {true, false}) {
        INFO((same ? "same group" : "two groups"));
        KSpaceJastrow j(cell, kA, same ? kA : kB, 3, three_shells());
        auto log_at = [&](std::size_t idx, double shift) {
            WalkerSet ws(1, kNp, 2, cell);
            std::copy(x0.begin(), x0.end(), ws.positions().begin());
            ws.positions()[idx] += shift;
            ws.dist_tables().build_all(ws.positions());
            j.evaluate_batch(ws);
            return j.log_psi_c()[0];
        };
        const double l0 = log_at(0, 0.0);
        const std::vector<double> g(j.grad_c().begin(), j.grad_c().end());
        const std::vector<double> l(j.lap_c().begin(), j.lap_c().end());
        const double h = 1e-4;
        for (std::size_t idx = 0; idx < kNp * 2; ++idx) {
            INFO("coordinate " << idx);
            const double lp = log_at(idx, h), lm = log_at(idx, -h);
            CHECK(g[idx] == Approx((lp - lm) / (2 * h)).margin(1e-7));
            CHECK(l[idx] == Approx((lp - 2 * l0 + lm) / (h * h)).margin(1e-5));
        }
    }
}

TEST_CASE("kspace jastrow moves keep the caches equal to a fresh evaluation",
          "[kspace_jastrow]") {
    const SimulationCell cell = hexagonal(5.0);
    const std::size_t nw = 3;
    std::mt19937_64 rng(3);
    auto make = [&] {
        TrialWavefunction psi;
        psi.add_component(std::make_unique<KSpaceJastrow>(cell, kA, kA, 3, three_shells()));
        psi.add_component(std::make_unique<KSpaceJastrow>(cell, kA, kB, 3, three_shells()));
        return psi;
    };
    TrialWavefunction psi = make();
    WalkerSet ws(nw, kNp, 2, cell);
    scatter(ws, cell, rng);
    psi.evaluate_batch(ws);

    std::normal_distribution<double> step(0.0, 0.7);
    std::bernoulli_distribution accept(0.6);
    std::unique_ptr<bool[]> mask(new bool[nw]);
    for (int move = 0; move < 300; ++move) {
        const std::size_t p = static_cast<std::size_t>(move) % kNp;
        std::vector<double> np(2 * nw);
        for (std::size_t d = 0; d < 2; ++d)
            for (std::size_t w = 0; w < nw; ++w)
                np[d * nw + w] = ws.positions()[(p * 2 + d) * nw + w] + step(rng);
        ws.propose_move_batch(p, np);
        psi.ratio_batch(ws, p);
        for (std::size_t w = 0; w < nw; ++w) mask[w] = accept(rng);
        const std::span<const bool> m(mask.get(), nw);
        psi.apply_move_batch(ws, p, m);
        ws.apply_move(p, ws.new_pos(), ws.new_grad(), ws.new_lap(), m);
    }
    TrialWavefunction fresh = make();
    WalkerSet ref(nw, kNp, 2, cell);
    std::copy(ws.positions().begin(), ws.positions().end(), ref.positions().begin());
    ref.dist_tables().build_all(ref.positions());
    fresh.evaluate_batch(ref);
    for (std::size_t w = 0; w < nw; ++w)
        CHECK(ws.log_psi()[w] == Approx(ref.log_psi()[w]).margin(1e-9));
    for (std::size_t i = 0; i < ws.gradients().size(); ++i) {
        INFO("index " << i);
        CHECK(ws.gradients()[i] == Approx(ref.gradients()[i]).margin(1e-9));
        CHECK(ws.laplacians()[i] == Approx(ref.laplacians()[i]).margin(1e-9));
    }
}

TEST_CASE("kspace jastrow parameter derivatives are exact", "[kspace_jastrow][optimize]") {
    // ln Ψ is linear in the p_s, so ∂ ln Ψ/∂p_s is the shell's Σ T_k exactly —
    // no gauge constant — and the curvature terms are ∇, ∇² of that sum.
    const SimulationCell cell = hexagonal(5.0);
    std::mt19937_64 rng(4);
    WalkerSet ws(2, kNp, 2, cell);
    scatter(ws, cell, rng);
    const std::size_t stride = kNp * 2 * 2;
    for (const bool same : {true, false}) {
        INFO((same ? "same group" : "two groups"));
        auto params = three_shells();
        params[1].is_fixed = true;   // a fixed shell is skipped in the output
        KSpaceJastrow j(cell, kA, same ? kA : kB, 3, params);
        j.evaluate_batch(ws);
        REQUIRE(j.num_optimizable_params() == 2);
        REQUIRE(j.supports_parameter_gradient_derivatives());
        std::vector<double> d(2 * 2), dg(2 * stride), dl(2 * stride);
        j.compute_parameter_derivatives(ws, d);
        REQUIRE(j.compute_parameter_gradient_derivatives(ws, dg, dl));

        const double h = 1e-6;
        std::size_t s = 0;
        for (std::size_t q = 0; q < 3; ++q) {
            if (params[q].is_fixed) continue;
            INFO("shell " << q);
            auto at = [&](double shift) {
                auto pp = params;
                pp[q].value += shift;
                KSpaceJastrow c(cell, kA, same ? kA : kB, 3, pp);
                c.evaluate_batch(ws);
                return std::make_tuple(
                    std::vector<double>(c.log_psi_c().begin(), c.log_psi_c().end()),
                    std::vector<double>(c.grad_c().begin(), c.grad_c().end()),
                    std::vector<double>(c.lap_c().begin(), c.lap_c().end()));
            };
            const auto [lp, gp, lap_p] = at(h);
            const auto [lm, gm, lap_m] = at(-h);
            for (std::size_t w = 0; w < 2; ++w)
                CHECK(d[s * 2 + w] == Approx((lp[w] - lm[w]) / (2 * h)).margin(1e-7));
            for (std::size_t i = 0; i < stride; ++i) {
                CHECK(dg[s * stride + i] == Approx((gp[i] - gm[i]) / (2 * h)).margin(1e-6));
                CHECK(dl[s * stride + i] == Approx((lap_p[i] - lap_m[i]) / (2 * h)).margin(1e-5));
            }
            ++s;
        }
    }
}

TEST_CASE("the Gaskell RPA form reproduces the 2-D plasmon at small k", "[gaskell]") {
    // S_RPA → k²/(2mω_p), ω_p = √(2πρk/m), for any mass, from a free-fermion S₀.
    using alea::numerics::gaskell_sofk;
    using alea::numerics::free_fermion_sofk_2d;
    const double pi = 3.14159265358979323846, rho = 1.0 / (pi * 30.0 * 30.0);
    for (const double m : {1.0, 0.35}) {
        INFO("mass " << m);
        for (const double k : {1e-4, 3e-4}) {
            const double s0 = free_fermion_sofk_2d(k, rho);
            const double s  = gaskell_sofk(k, rho, s0, 2 * pi / k, 1.0 / m);
            const double wp = std::sqrt(2 * pi * rho * k / m);
            CHECK(s == Approx(k * k / (2 * m * wp)).epsilon(2e-2));
        }
    }
    // No interaction → no correlation: ũ = 0 and S = S₀.
    CHECK(alea::numerics::gaskell_u(0.3, rho, 0.4, 0.0, 1.0) == 0.0);
    // Free fermions: S₀ is 1 beyond 2k_F, and linear (4x/π) at small k.
    const double kf = std::sqrt(4 * pi * rho);
    CHECK(free_fermion_sofk_2d(2.5 * kf, rho) == 1.0);
    CHECK(free_fermion_sofk_2d(1e-3 * kf, rho) == Approx(4 * 0.5e-3 / pi).epsilon(1e-6));
}

TEST_CASE("kspace jastrow needs a periodic cell and enough shells", "[kspace_jastrow]") {
    CHECK_THROWS_WITH(KSpaceJastrow(SimulationCell{}, kA, kA, 1, {{"p1", 0.0, false}}),
                      Catch::Matchers::ContainsSubstring("periodic"));
    CHECK_THROWS_WITH(KSpaceJastrow(hexagonal(5.0), kA, kA, 2, {{"p1", 0.0, false}}),
                      Catch::Matchers::ContainsSubstring("one coefficient per shell"));
}
