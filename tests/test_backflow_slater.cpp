// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Backflow Slater determinant (wavefunction/backflow_slater_determinant.h):
// the component contract battery against an independent oracle that forms the
// quasi-coordinates and the determinants by its own loops, with central finite
// differences for every derivative.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "population/simulation_cell.h"
#include "population/walker.h"
#include "population/walker_set.h"
#include "wavefunction/backflow_function.h"
#include "wavefunction/backflow_slater_determinant.h"
#include "wavefunction/orbital_set.h"
#include "wavefunction/slater_determinant.h"
#include "wavefunction/trial_wavefunction.h"

using namespace alea;
using Catch::Matchers::ContainsSubstring;

namespace {

constexpr double kPi = 3.14159265358979323846;

// ── The oracle ───────────────────────────────────────────────────────────────

struct OracleChannel {
    std::size_t         source_first, source_last;
    int                 C;
    double              L;
    std::vector<double> c;   // every coefficient c_0..c_N (c_1 already derived)
};

struct System {
    std::size_t dim = 2, np = 0;
    SimulationCell cell = SimulationCell(2);
    // Determinant blocks over one group: [first, last) and their orbitals.
    std::vector<std::pair<std::size_t, std::size_t>> blocks;
    std::vector<std::shared_ptr<OrbitalSet>>         orbitals;
    std::vector<OracleChannel>                       channels;
    // The component's free parameters per channel, [c_0, (c_1), c_2.., L].
    std::vector<std::vector<double>>                 free_params;
};

double oracle_eta(const OracleChannel& ch, double r) {
    if (r >= ch.L) return 0.0;
    double P = 0.0;
    for (std::size_t k = 0; k < ch.c.size(); ++k) P += ch.c[k] * std::pow(r, static_cast<double>(k));
    return std::pow(1.0 - r / ch.L, ch.C) * P;
}

// log|ψ| and sign of ψ at positions pos ([np*dim], AoS).
std::pair<double, double> oracle(const System& sys, const std::vector<double>& pos) {
    const std::size_t dim = sys.dim;
    const std::size_t g0 = sys.blocks.front().first, g1 = sys.blocks.back().second;
    std::vector<double> x((g1 - g0) * dim);
    for (std::size_t P = g0; P < g1; ++P) {
        for (std::size_t d = 0; d < dim; ++d) x[(P - g0) * dim + d] = pos[P * dim + d];
        for (const OracleChannel& ch : sys.channels)
            for (std::size_t j = ch.source_first; j < ch.source_last; ++j) {
                if (j == P) continue;
                double del[3] = {};
                for (std::size_t d = 0; d < dim; ++d) del[d] = pos[P * dim + d] - pos[j * dim + d];
                sys.cell.min_image(del);
                double r = 0.0;
                for (std::size_t d = 0; d < dim; ++d) r += del[d] * del[d];
                const double eta = oracle_eta(ch, std::sqrt(r));
                for (std::size_t d = 0; d < dim; ++d) x[(P - g0) * dim + d] += eta * del[d];
            }
    }
    double log_abs = 0.0, sign = 1.0;
    for (std::size_t b = 0; b < sys.blocks.size(); ++b) {
        const auto [f, l] = sys.blocks[b];
        const std::size_t n = l - f;
        std::vector<double> m(n * n), val(n), grad(n * dim), lap(n * dim);
        for (std::size_t r = 0; r < n; ++r) {
            std::vector<double> xr(x.begin() + static_cast<long>((f - g0 + r) * dim),
                                   x.begin() + static_cast<long>((f - g0 + r + 1) * dim));
            sys.orbitals[b]->evaluate(xr, dim, val, grad, lap, 0, 1);
            for (std::size_t c = 0; c < n; ++c) m[r * n + c] = val[c];
        }
        // Determinant by elimination with partial pivoting.
        for (std::size_t k = 0; k < n; ++k) {
            std::size_t piv = k;
            for (std::size_t r = k + 1; r < n; ++r)
                if (std::abs(m[r * n + k]) > std::abs(m[piv * n + k])) piv = r;
            if (piv != k) {
                for (std::size_t c = 0; c < n; ++c) std::swap(m[k * n + c], m[piv * n + c]);
                sign = -sign;
            }
            const double d = m[k * n + k];
            log_abs += std::log(std::abs(d));
            if (d < 0.0) sign = -sign;
            for (std::size_t r = k + 1; r < n; ++r) {
                const double factor = m[r * n + k] / d;
                for (std::size_t c = k; c < n; ++c) m[r * n + c] -= factor * m[k * n + c];
            }
        }
    }
    return {log_abs, sign};
}

std::unique_ptr<BackflowSlaterDeterminant> make_component(const System& sys) {
    std::vector<BackflowSlaterDeterminant::Block> blocks;
    for (std::size_t b = 0; b < sys.blocks.size(); ++b)
        blocks.push_back({sys.blocks[b].first, sys.blocks[b].second, sys.orbitals[b]->clone()});
    std::vector<BackflowSlaterDeterminant::Channel> channels;
    for (std::size_t c = 0; c < sys.channels.size(); ++c) {
        BackflowSlaterDeterminant::Channel ch;
        ch.source_first = sys.channels[c].source_first;
        ch.source_last = sys.channels[c].source_last;
        ch.truncation = sys.channels[c].C;
        for (double v : sys.free_params[c]) ch.params.push_back(Parameter{"", v, false});
        channels.push_back(std::move(ch));
    }
    return std::make_unique<BackflowSlaterDeterminant>(std::move(blocks), std::move(channels),
                                                       sys.cell);
}

// Open cell: three like-spin particles [0,3) in Gaussians, two "holes" [3,5)
// that displace them; a same-group channel (η'(0) = 0) and a hole channel.
System open_system() {
    System s;
    s.np = 5;
    s.blocks = {{0, 3}};
    s.orbitals = {std::make_shared<GaussianOrbitalSet>(
        2, std::vector<double>{-1.0, 0.0, 1.0, 0.2, 0.1, 1.1}, std::vector<double>{0.5, 0.6, 0.45})};
    const double c0 = 0.2, L = 2.5;
    s.channels.push_back({0, 3, 3, L, {c0, 3.0 * c0 / L, -0.05}});
    s.free_params.push_back({c0, -0.05, L});
    s.channels.push_back({3, 5, 3, 3.0, {-0.15, 0.1, 0.02}});
    s.free_params.push_back({-0.15, 0.1, 0.02, 3.0});
    return s;
}

// Periodic 6×6 cell: five like-spin particles in a closed plane-wave shell,
// a same-group channel with L inside the inradius 3.
System periodic_system() {
    System s;
    s.np = 5;
    s.cell = SimulationCell(2, std::vector<double>{6.0, 0.0, 0.0, 6.0});
    s.blocks = {{0, 5}};
    s.orbitals = {make_plane_wave_orbitals(s.cell, 5)};
    const double c0 = 0.3, L = 2.9;
    s.channels.push_back({0, 5, 3, L, {c0, 3.0 * c0 / L, 0.04, -0.01}});
    s.free_params.push_back({c0, 0.04, -0.01, L});
    return s;
}

// Two spin blocks (up [0,2), down [2,3)) displaced by a third group [3,4).
System two_block_system() {
    System s;
    s.np = 4;
    s.blocks = {{0, 2}, {2, 3}};
    s.orbitals = {std::make_shared<GaussianOrbitalSet>(2, std::vector<double>{-1.0, 0.0, 1.0, 0.0},
                                                      std::vector<double>{0.6, 0.6}),
                  std::make_shared<GaussianOrbitalSet>(2, std::vector<double>{0.0, 0.5},
                                                      std::vector<double>{0.5})};
    s.channels.push_back({3, 4, 3, 2.0, {0.25, -0.1}});
    s.free_params.push_back({0.25, -0.1, 2.0});
    return s;
}

// Hexagonal 6-cell (inradius 3): three like-spin particles in Gaussians at
// three sites, folded to the minimum image as in the Wigner crystal, displaced
// by each other (like-spin) and by two holes [3,5).
System hexagonal_system() {
    System s;
    s.np = 5;
    s.cell = SimulationCell(2, std::vector<double>{6.0, 0.0, 3.0, 3.0 * std::sqrt(3.0)});
    s.blocks = {{0, 3}};
    s.orbitals = {std::make_shared<GaussianOrbitalSet>(
        2, std::vector<double>{1.0, 1.0, 4.0, 1.5, 2.5, 3.5}, std::vector<double>{0.4, 0.4, 0.4},
        s.cell)};
    const double c0 = 0.25, L = 2.5;
    s.channels.push_back({0, 3, 3, L, {c0, 3.0 * c0 / L, -0.03}});
    s.free_params.push_back({c0, -0.03, L});
    s.channels.push_back({3, 5, 3, 2.2, {-0.2, 0.15, 0.04}});
    s.free_params.push_back({-0.2, 0.15, 0.04, 2.2});
    return s;
}

// Cartesian position of fractional coordinates (f1, f2) of the hexagonal cell.
std::pair<double, double> hex_point(double f1, double f2) {
    return {6.0 * f1 + 3.0 * f2, 3.0 * std::sqrt(3.0) * f2};
}

std::vector<std::vector<double>> hexagonal_configs() {
    const std::vector<std::vector<double>> frac = {
        {0.15, 0.2, 0.55, 0.25, 0.3, 0.6, 0.8, 0.7, 0.05, 0.9},
        {0.9, 0.1, 0.45, 0.4, 0.2, 0.75, 0.6, 0.05, 0.35, 0.35}};
    std::vector<std::vector<double>> out;
    for (const auto& f : frac) {
        std::vector<double> c;
        for (std::size_t k = 0; k < f.size(); k += 2) {
            const auto [x, y] = hex_point(f[k], f[k + 1]);
            c.push_back(x);
            c.push_back(y);
        }
        out.push_back(c);
    }
    return out;
}

// AoS configurations, one per walker.
std::vector<std::vector<double>> open_configs() {
    return {{-0.8, 0.1, 0.9, -0.2, 0.2, 0.8, 0.4, -0.5, -0.3, 0.6},
            {-0.5, -0.3, 0.7, 0.4, -0.1, 1.2, 1.1, 0.2, -1.0, -0.4}};
}
std::vector<std::vector<double>> periodic_configs() {
    return {{0.5, 0.7, 2.1, 1.0, 3.9, 4.2, 5.1, 2.0, 1.4, 5.3},
            {1.0, 1.9, 2.6, 0.3, 4.4, 3.1, 5.6, 5.5, 0.2, 4.0}};
}
std::vector<std::vector<double>> two_block_configs() {
    return {{-0.7, 0.2, 0.8, -0.1, 0.1, 0.6, 0.3, -0.4},
            {-0.4, -0.5, 0.9, 0.3, -0.2, 0.2, 0.9, 0.6}};
}

// A WalkerSet holding the configurations, built through Walker objects so
// apply_permutation can gather them.
struct Batch {
    std::vector<Walker> walkers;
    WalkerSet           ws;
    Batch(const System& sys, const std::vector<std::vector<double>>& configs)
        : ws(configs.size(), sys.np, sys.dim, sys.cell) {
        walkers.reserve(configs.size());
        for (const auto& c : configs) {
            walkers.emplace_back(sys.np, sys.dim);
            walkers.back().set_positions(c);
        }
        for (Walker& w : walkers) ws.add_walker(w);
        ws.load_all();
    }
};

std::vector<double> config_of(const WalkerSet& ws, std::size_t w) {
    const std::size_t np = ws.n_particles(), dim = ws.dim(), nw = ws.num_walkers();
    std::vector<double> c(np * dim);
    for (std::size_t k = 0; k < np; ++k)
        for (std::size_t d = 0; d < dim; ++d) c[k * dim + d] = ws.positions()[(k * dim + d) * nw + w];
    return c;
}

// Central finite differences of the oracle's log|ψ|: [np*dim] each.
// Richardson-extrapolated over h and h/2, so the truncation error is O(h⁴):
// near a node, where ln|ψ| ~ ln d, the plain O(h²) stencil errs by ~h²/d⁴.
// grad_err/lap_err: |D(h) − D(h/2)|, which bounds the extrapolated error and is
// ~1e-8 away from nodes (so the checks keep their absolute margins there).
void fd_derivatives(const System& sys, const std::vector<double>& pos, std::vector<double>& grad,
                    std::vector<double>& lap, std::vector<double>& grad_err,
                    std::vector<double>& lap_err) {
    const double f0 = oracle(sys, pos).first;
    grad.assign(pos.size(), 0.0);
    lap.assign(pos.size(), 0.0);
    grad_err.assign(pos.size(), 0.0);
    lap_err.assign(pos.size(), 0.0);
    for (std::size_t q = 0; q < pos.size(); ++q) {
        double g[2] = {}, l[2] = {};
        for (int level = 0; level < 2; ++level) {
            const double h = level == 0 ? 2e-4 : 1e-4;
            std::vector<double> p = pos, m = pos;
            p[q] += h;
            m[q] -= h;
            const double fp = oracle(sys, p).first, fm = oracle(sys, m).first;
            g[level] = (fp - fm) / (2.0 * h);
            l[level] = (fp - 2.0 * f0 + fm) / (h * h);
        }
        grad[q] = (4.0 * g[1] - g[0]) / 3.0;
        lap[q] = (4.0 * l[1] - l[0]) / 3.0;
        grad_err[q] = std::abs(g[0] - g[1]);
        lap_err[q] = std::abs(l[0] - l[1]);
    }
}

void fd_derivatives(const System& sys, const std::vector<double>& pos, std::vector<double>& grad,
                    std::vector<double>& lap) {
    std::vector<double> ge, le;
    fd_derivatives(sys, pos, grad, lap, ge, le);
}

void check_against_oracle(const System& sys, const WalkerSet& ws) {
    const std::size_t nw = ws.num_walkers(), np = sys.np, dim = sys.dim;
    for (std::size_t w = 0; w < nw; ++w) {
        const std::vector<double> pos = config_of(ws, w);
        INFO("walker " << w);
        CHECK(ws.log_psi()[w] == Catch::Approx(oracle(sys, pos).first).epsilon(1e-12));
        std::vector<double> g, l, ge, le;
        fd_derivatives(sys, pos, g, l, ge, le);
        for (std::size_t k = 0; k < np; ++k)
            for (std::size_t d = 0; d < dim; ++d) {
                const std::size_t q = k * dim + d;
                INFO("particle " << k << " dim " << d << "  fd error estimate " << le[q]);
                CHECK(ws.gradients()[(k * dim + d) * nw + w] ==
                      Catch::Approx(g[q]).margin(1e-7 + ge[q]));
                CHECK(ws.laplacians()[(k * dim + d) * nw + w] ==
                      Catch::Approx(l[q]).margin(2e-5 + le[q]));
            }
    }
}

TrialWavefunction trial_of(const System& sys) {
    TrialWavefunction psi;
    psi.add_component(make_component(sys));
    return psi;
}

// Propose particle p to new_r (same for every walker, AoS [dim]).
void propose(WalkerSet& ws, std::size_t p, const std::vector<double>& new_r) {
    const std::size_t nw = ws.num_walkers(), dim = ws.dim();
    std::vector<double> soa(dim * nw);
    for (std::size_t d = 0; d < dim; ++d)
        for (std::size_t w = 0; w < nw; ++w) soa[d * nw + w] = new_r[d];
    ws.propose_move_batch(p, soa);
}

}  // namespace

// ── The backflow function ────────────────────────────────────────────────────

TEST_CASE("backflow eta matches its derivatives and vanishes smoothly at the cutoff",
          "[backflow]") {
    const BackflowFunction eta(3, 2.0, {0.3, -0.2, 0.05}, false);
    const double h = 1e-5;
    for (double r : {0.0 + 2 * h, 0.4, 1.1, 1.9}) {
        double e = 0.0, d1 = 0.0, d2 = 0.0, ep = 0.0, em = 0.0, x = 0.0;
        eta.evaluate(r, e, d1, d2);
        eta.evaluate(r + h, ep, x, x);
        eta.evaluate(r - h, em, x, x);
        CHECK(d1 == Catch::Approx((ep - em) / (2 * h)).margin(1e-8));
        CHECK(d2 == Catch::Approx((ep - 2 * e + em) / (h * h)).margin(1e-5));
    }
    double e = 1.0, d1 = 1.0, d2 = 1.0;
    eta.evaluate(2.0, e, d1, d2);
    CHECK(e == 0.0);
    CHECK(d1 == 0.0);
    eta.evaluate(2.0 - 1e-6, e, d1, d2);   // C = 3: η, η' → 0 and η'' → 0 at L
    CHECK(std::abs(e) < 1e-15);
    CHECK(std::abs(d1) < 1e-10);
    CHECK(std::abs(d2) < 1e-5);
}

TEST_CASE("a like-spin eta has zero slope at contact, with c1 = C c0 / L", "[backflow]") {
    const BackflowFunction eta(3, 2.5, {0.2, -0.05}, true);
    double e = 0.0, d1 = 1.0, d2 = 0.0;
    eta.evaluate(0.0, e, d1, d2);
    CHECK(e == Catch::Approx(0.2));
    CHECK(d1 == Catch::Approx(0.0).margin(1e-15));
    REQUIRE(eta.coefficients().size() == 3);
    CHECK(eta.coefficients()[1] == Catch::Approx(3.0 * 0.2 / 2.5));
    CHECK(eta.n_parameters() == 3);   // c0, c2, L
}

TEST_CASE("backflow eta parameter derivatives match finite differences", "[backflow]") {
    for (bool zero_slope : {false, true}) {
        INFO("zero_slope " << zero_slope);
        const std::vector<double> free = {0.25, -0.12, 0.03};
        const BackflowFunction eta(3, 2.2, free, zero_slope);
        const std::size_t n = eta.n_parameters();
        const double h = 1e-6;
        for (double r : {0.3, 1.0, 1.8}) {
            std::vector<double> d(n);
            eta.parameter_derivatives(r, d);
            for (std::size_t k = 0; k < n; ++k) {
                BackflowFunction plus = eta, minus = eta;
                const double v = k < free.size() ? free[k] : 2.2;
                plus.set_parameter(k, v + h);
                minus.set_parameter(k, v - h);
                double ep = 0.0, em = 0.0, x = 0.0;
                plus.evaluate(r, ep, x, x);
                minus.evaluate(r, em, x, x);
                CHECK(d[k] == Catch::Approx((ep - em) / (2 * h)).margin(1e-8));
            }
        }
    }
}

TEST_CASE("orbital Hessians match finite differences of the gradient", "[backflow][orbitals]") {
    const SimulationCell box(2, std::vector<double>{6.0, 0.0, 0.0, 6.0});
    const std::vector<std::shared_ptr<OrbitalSet>> sets = {
        std::make_shared<GaussianOrbitalSet>(2, std::vector<double>{0.3, -0.2, 1.0, 1.4},
                                             std::vector<double>{0.7, 0.4}),
        std::make_shared<GaussianOrbitalSet>(2, std::vector<double>{0.5, 5.6, 3.0, 3.0},
                                             std::vector<double>{0.3, 0.3}, box),
        make_plane_wave_orbitals(box, 5)};
    const double h = 1e-5;
    for (const auto& orb : sets) {
        REQUIRE(orb->supports_hessian());
        const std::size_t n = orb->size();
        std::vector<double> pos = {0.4, 5.9};
        std::vector<double> val(n), grad(n * 2), hess(n * 4), lap(n * 2), v2(n), g2(n * 2), l2(n * 2);
        orb->evaluate_hessian(pos, 2, val, grad, hess, 0, 1);
        orb->evaluate(pos, 2, v2, g2, l2, 0, 1);
        for (std::size_t m = 0; m < n; ++m) {
            CHECK(val[m] == Catch::Approx(v2[m]));
            for (std::size_t a = 0; a < 2; ++a) {
                CHECK(grad[m * 2 + a] == Catch::Approx(g2[m * 2 + a]).margin(1e-14));
                CHECK(hess[(m * 2 + a) * 2 + a] == Catch::Approx(l2[m * 2 + a]).margin(1e-13));
                for (std::size_t b = 0; b < 2; ++b) {
                    std::vector<double> pp = pos, pm = pos, vp(n), gp(n * 2), lp(n * 2), gm(n * 2);
                    pp[b] += h;
                    pm[b] -= h;
                    orb->evaluate(pp, 2, vp, gp, lp, 0, 1);
                    orb->evaluate(pm, 2, vp, gm, lp, 0, 1);
                    CHECK(hess[(m * 2 + a) * 2 + b] ==
                          Catch::Approx((gp[m * 2 + a] - gm[m * 2 + a]) / (2 * h)).margin(1e-8));
                }
            }
        }
    }
}

// ── Contract battery ─────────────────────────────────────────────────────────

TEST_CASE("backflow determinant: log psi and derivatives match the oracle and finite differences",
          "[backflow]") {
    const std::vector<std::pair<System, std::vector<std::vector<double>>>> cases = {
        {open_system(), open_configs()},
        {periodic_system(), periodic_configs()},
        {two_block_system(), two_block_configs()},
        {hexagonal_system(), hexagonal_configs()}};
    for (std::size_t c = 0; c < cases.size(); ++c) {
        INFO("system " << c);
        const auto& [sys, configs] = cases[c];
        Batch b(sys, configs);
        TrialWavefunction psi = trial_of(sys);
        psi.evaluate_batch(b.ws);
        check_against_oracle(sys, b.ws);
    }
}

TEST_CASE("backflow determinant adds to the batch totals", "[backflow]") {
    const System sys = open_system();
    Batch b(sys, open_configs());
    for (double& v : b.ws.log_psi()) v = 1.5;
    for (double& v : b.ws.gradients()) v = 0.25;
    for (double& v : b.ws.laplacians()) v = -0.5;
    auto det = make_component(sys);
    det->evaluate_batch(b.ws);
    for (std::size_t k = 0; k < b.ws.gradients().size(); ++k) {
        CHECK(b.ws.gradients()[k] == Catch::Approx(0.25 + det->grad_c()[k]));
        CHECK(b.ws.laplacians()[k] == Catch::Approx(-0.5 + det->lap_c()[k]));
    }
    CHECK(b.ws.log_psi()[0] == Catch::Approx(1.5 + det->log_psi_c()[0]));
}

TEST_CASE("backflow ratio_batch agrees with evaluate_batch at the proposal", "[backflow]") {
    const std::vector<std::pair<System, std::vector<std::vector<double>>>> cases = {
        {open_system(), open_configs()},
        {periodic_system(), periodic_configs()},
        {two_block_system(), two_block_configs()},
        {hexagonal_system(), hexagonal_configs()}};
    for (std::size_t c = 0; c < cases.size(); ++c) {
        const auto& [sys, configs] = cases[c];
        // A row of the determinant and, where present, a source particle.
        for (std::size_t p : {std::size_t{1}, sys.np - 1}) {
            INFO("system " << c << " moves particle " << p);
            Batch b(sys, configs);
            TrialWavefunction psi = trial_of(sys);
            psi.evaluate_batch(b.ws);
            const std::vector<double> to = sys.cell.is_periodic() ? std::vector<double>{3.3, 1.2}
                                                                  : std::vector<double>{0.35, -0.6};
            propose(b.ws, p, to);
            psi.ratio_batch(b.ws, p);
            for (std::size_t w = 0; w < b.ws.num_walkers(); ++w) {
                std::vector<double> before = config_of(b.ws, w), after = before;
                for (std::size_t d = 0; d < 2; ++d) after[p * 2 + d] = to[d];
                const auto [l0, s0] = oracle(sys, before);
                const auto [l1, s1] = oracle(sys, after);
                CHECK(b.ws.log_ratios()[w] == Catch::Approx(l1 - l0).margin(1e-11));
                CHECK(b.ws.phase_diffs()[w] == Catch::Approx(s0 * s1 < 0 ? kPi : 0.0));
                std::vector<double> g, l;
                fd_derivatives(sys, after, g, l);
                const std::size_t nw = b.ws.num_walkers();
                for (std::size_t d = 0; d < 2; ++d) {
                    CHECK(b.ws.new_grad()[d * nw + w] == Catch::Approx(g[p * 2 + d]).margin(1e-7));
                    CHECK(b.ws.new_lap()[d * nw + w] == Catch::Approx(l[p * 2 + d]).margin(2e-5));
                }
            }
        }
    }
}

TEST_CASE("backflow commits match a fresh evaluation, eager and deferred", "[backflow]") {
    const std::vector<std::pair<System, std::vector<std::vector<double>>>> cases = {
        {open_system(), open_configs()},
        {periodic_system(), periodic_configs()},
        {two_block_system(), two_block_configs()},
        {hexagonal_system(), hexagonal_configs()}};
    for (std::size_t c = 0; c < cases.size(); ++c) {
        const auto& [sys, configs] = cases[c];
        // Eager; deferred; deferred without the proposal Laplacian (as VMC/DMC run).
        for (int mode = 0; mode < 3; ++mode) {
            const bool deferred = mode > 0;
            INFO("system " << c << " mode " << mode);
            Batch b(sys, configs);
            TrialWavefunction psi = trial_of(sys);
            if (mode == 2) psi.set_proposal_laplacian(false);
            psi.evaluate_batch(b.ws);
            // Two moves; the first walker accepts both, the second rejects the first.
            const std::vector<std::pair<std::size_t, std::vector<double>>> moves = {
                {0, sys.cell.is_periodic() ? std::vector<double>{0.9, 1.4} : std::vector<double>{-0.6, 0.35}},
                {sys.np - 1, sys.cell.is_periodic() ? std::vector<double>{4.1, 5.0} : std::vector<double>{0.2, -0.9}}};
            for (std::size_t mv = 0; mv < moves.size(); ++mv) {
                const auto& [p, to] = moves[mv];
                if (deferred) psi.refresh_particle_derivatives(b.ws, p);
                propose(b.ws, p, to);
                psi.ratio_batch(b.ws, p);
                std::array<bool, 2> accept = {true, mv != 0};
                if (deferred) psi.apply_move_batch_deferred(b.ws, p, accept);
                else          psi.apply_move_batch(b.ws, p, accept);
                b.ws.apply_move(p, b.ws.new_pos(), b.ws.new_grad(), b.ws.new_lap(), accept);
            }
            if (deferred) psi.refresh_derivatives(b.ws);
            check_against_oracle(sys, b.ws);
        }
    }
}

TEST_CASE("backflow refresh_particle_derivatives makes a stale drift current", "[backflow]") {
    const System sys = open_system();
    Batch b(sys, open_configs());
    TrialWavefunction psi = trial_of(sys);
    psi.evaluate_batch(b.ws);
    propose(b.ws, 0, {-0.6, 0.35});
    psi.ratio_batch(b.ws, 0);
    std::array<bool, 2> accept = {true, true};
    psi.apply_move_batch_deferred(b.ws, 0, accept);
    b.ws.apply_move(0, b.ws.new_pos(), b.ws.new_grad(), b.ws.new_lap(), accept);
    // Particle 2's gradient changed with the move; refresh it alone.
    psi.refresh_particle_derivatives(b.ws, 2);
    const std::size_t nw = b.ws.num_walkers();
    for (std::size_t w = 0; w < nw; ++w) {
        std::vector<double> g, l;
        fd_derivatives(sys, config_of(b.ws, w), g, l);
        for (std::size_t d = 0; d < 2; ++d)
            CHECK(b.ws.gradients()[(2 * 2 + d) * nw + w] == Catch::Approx(g[2 * 2 + d]).margin(1e-7));
    }
}

TEST_CASE("backflow ratio over disjoint walker ranges equals the full batch", "[backflow]") {
    const System sys = periodic_system();
    Batch whole(sys, periodic_configs()), split(sys, periodic_configs());
    TrialWavefunction a = trial_of(sys), b = trial_of(sys);
    a.evaluate_batch(whole.ws);
    b.evaluate_batch(split.ws);
    propose(whole.ws, 2, {1.7, 4.4});
    propose(split.ws, 2, {1.7, 4.4});
    a.ratio_batch(whole.ws, 2);
    b.ratio_batch(split.ws, 2, 0, 1);
    b.ratio_batch(split.ws, 2, 1, 2);
    for (std::size_t w = 0; w < 2; ++w) {
        CHECK(split.ws.log_ratios()[w] == whole.ws.log_ratios()[w]);
        for (std::size_t d = 0; d < 2; ++d) {
            CHECK(split.ws.new_grad()[d * 2 + w] == whole.ws.new_grad()[d * 2 + w]);
            CHECK(split.ws.new_lap()[d * 2 + w] == whole.ws.new_lap()[d * 2 + w]);
        }
    }
}

TEST_CASE("backflow permute_caches carries every walker's state", "[backflow]") {
    const System sys = open_system();
    Batch b(sys, open_configs());
    TrialWavefunction psi = trial_of(sys);
    psi.evaluate_batch(b.ws);
    const std::vector<std::size_t> src = {1, 1, 0};   // swap, duplicate, grow
    b.ws.apply_permutation(src);
    psi.permute_caches(src);
    propose(b.ws, 1, {0.1, -0.7});
    psi.ratio_batch(b.ws, 1);
    for (std::size_t w = 0; w < src.size(); ++w) {
        std::vector<double> before = config_of(b.ws, w), after = before;
        after[2] = 0.1;
        after[3] = -0.7;
        CHECK(b.ws.log_ratios()[w] ==
              Catch::Approx(oracle(sys, after).first - oracle(sys, before).first).margin(1e-11));
    }
}

TEST_CASE("backflow clone reproduces the original bit for bit", "[backflow]") {
    const System sys = open_system();
    Batch a(sys, open_configs()), b(sys, open_configs());
    TrialWavefunction psi = trial_of(sys);
    TrialWavefunction copy = psi.clone();
    psi.evaluate_batch(a.ws);
    copy.evaluate_batch(b.ws);
    for (std::size_t k = 0; k < a.ws.gradients().size(); ++k) {
        CHECK(a.ws.gradients()[k] == b.ws.gradients()[k]);
        CHECK(a.ws.laplacians()[k] == b.ws.laplacians()[k]);
    }
    CHECK(a.ws.log_psi()[1] == b.ws.log_psi()[1]);
}

TEST_CASE("backflow parameters: rebuild moves psi and derivatives match finite differences",
          "[backflow][optimize]") {
    const System sys = open_system();
    auto det = make_component(sys);
    // [b0_alpha? no: distinct widths expose none] then [c0, c2, L] and [c0, c1, c2, L].
    const auto params = det->parameters();
    REQUIRE(params.size() == 7);
    CHECK(params[0].name == "bf0_c0");
    CHECK(params[1].name == "bf0_c2");
    CHECK(params[2].name == "bf0_L");
    CHECK(params[3].name == "bf1_c0");
    CHECK(params[4].name == "bf1_c1");
    CHECK(params[6].name == "bf1_L");
    CHECK(det->is_nodal());

    Batch b(sys, open_configs());
    det->evaluate_batch(b.ws);
    std::vector<double> derivs(7 * 2);
    det->compute_parameter_derivatives(b.ws, derivs);

    const double h = 1e-6;
    for (std::size_t q = 0; q < 7; ++q) {
        INFO(params[q].name);
        auto log_psi_at = [&](double shift, std::size_t w) {
            auto copy = make_component(sys);
            copy->parameters()[q].value += shift;
            copy->rebuild();
            Batch fresh(sys, open_configs());
            copy->evaluate_batch(fresh.ws);
            return fresh.ws.log_psi()[w];
        };
        for (std::size_t w = 0; w < 2; ++w)
            CHECK(derivs[q * 2 + w] ==
                  Catch::Approx((log_psi_at(h, w) - log_psi_at(-h, w)) / (2 * h)).margin(1e-7));
    }
    // A changed parameter changes psi; the original stays as it was.
    auto changed = make_component(sys);
    changed->parameters()[0].value = 0.5;
    changed->rebuild();
    Batch c(sys, open_configs());
    changed->evaluate_batch(c.ws);
    CHECK(c.ws.log_psi()[0] != Catch::Approx(b.ws.log_psi()[0]));
}

TEST_CASE("backflow orbital width derivative matches finite differences", "[backflow][optimize]") {
    System sys = two_block_system();   // the up block has one uniform width
    auto det = make_component(sys);
    REQUIRE(det->parameters()[0].name == "b0_alpha");
    for (Parameter& p : det->parameters()) p.is_fixed = true;
    det->parameters()[0].is_fixed = false;
    Batch b(sys, two_block_configs());
    det->evaluate_batch(b.ws);
    std::vector<double> derivs(2);
    det->compute_parameter_derivatives(b.ws, derivs);
    const double h = 1e-6;
    for (std::size_t w = 0; w < 2; ++w) {
        auto at = [&](double a) {
            System s2 = sys;
            s2.orbitals[0] = std::make_shared<GaussianOrbitalSet>(
                2, std::vector<double>{-1.0, 0.0, 1.0, 0.0}, std::vector<double>{a, a});
            return oracle(s2, two_block_configs()[w]).first;
        };
        CHECK(derivs[w] == Catch::Approx((at(0.6 + h) - at(0.6 - h)) / (2 * h)).margin(1e-7));
    }
}

// ── Physical limits ──────────────────────────────────────────────────────────

TEST_CASE("backflow with eta = 0 is the plain Slater determinant", "[backflow]") {
    System sys = open_system();
    for (auto& fp : sys.free_params)
        for (std::size_t k = 0; k + 1 < fp.size(); ++k) fp[k] = 0.0;
    Batch a(sys, open_configs()), b(sys, open_configs());
    TrialWavefunction bf = trial_of(sys);
    bf.evaluate_batch(a.ws);
    std::vector<SlaterDeterminant::Block> blocks;
    blocks.push_back({0, 3, sys.orbitals[0]->clone()});
    TrialWavefunction plain;
    plain.add_component(std::make_unique<SlaterDeterminant>(std::move(blocks)));
    plain.evaluate_batch(b.ws);
    for (std::size_t w = 0; w < 2; ++w)
        CHECK(a.ws.log_psi()[w] == Catch::Approx(b.ws.log_psi()[w]).epsilon(1e-13));
    for (std::size_t k = 0; k < a.ws.gradients().size(); ++k) {
        CHECK(a.ws.gradients()[k] == Catch::Approx(b.ws.gradients()[k]).margin(1e-12));
        CHECK(a.ws.laplacians()[k] == Catch::Approx(b.ws.laplacians()[k]).margin(1e-11));
    }
}

TEST_CASE("the like-spin cusp survives backflow: the kinetic term stays finite at contact",
          "[backflow][physics]") {
    // Two like-spin particles approach along x.  ln|D| ∝ ln r_12 + smooth
    // terms; with η'(0) = 0 the backflow adds no |r_12|-linear term, so
    // ∇²ln|D| + |∇ln|D||² (the determinant's kinetic numerator, whose 1/r²
    // parts cancel exactly for the P-wave) converges as r_12 → 0.
    const System sys = open_system();
    std::vector<double> kinetic;
    for (double eps : {1e-2, 1e-3, 1e-4}) {
        std::vector<std::vector<double>> configs = {open_configs()[0]};
        configs[0][2] = configs[0][0] + eps;   // particle 1 next to particle 0
        configs[0][3] = configs[0][1];
        Batch b(sys, configs);
        TrialWavefunction psi = trial_of(sys);
        psi.evaluate_batch(b.ws);
        double t = 0.0;
        for (std::size_t q = 0; q < 2 * 2; ++q)   // particles 0 and 1
            t += b.ws.laplacians()[q] + b.ws.gradients()[q] * b.ws.gradients()[q];
        kinetic.push_back(t);
    }
    // Converged to a finite limit: successive differences shrink by ~10.
    CHECK(std::abs(kinetic[2] - kinetic[1]) < 0.2 * std::abs(kinetic[1] - kinetic[0]) + 1e-6);
    CHECK(std::isfinite(kinetic[2]));
}

TEST_CASE("backflow refuses what it cannot represent", "[backflow]") {
    // A same-group channel over a group with both spins.
    {
        System sys = two_block_system();
        sys.channels = {{0, 3, 3, 2.0, {0.1, 0.0}}};
        sys.free_params = {{0.1, 2.0}};
        CHECK_THROWS_WITH(make_component(sys), ContainsSubstring("one spin"));
    }
    // A cutoff beyond the Wigner–Seitz inradius.
    {
        System sys = periodic_system();
        sys.free_params[0].back() = 3.5;
        CHECK_THROWS_WITH(make_component(sys), ContainsSubstring("inradius"));
    }
    // Orbitals without a Hessian.
    {
        struct NoHessian final : OrbitalSet {
            std::size_t size() const override { return 1; }
            void evaluate(std::span<const double>, std::size_t, std::span<double> v,
                          std::span<double>, std::span<double>, std::size_t,
                          std::size_t) const override { v[0] = 1.0; }
            std::unique_ptr<OrbitalSet> clone() const override {
                return std::make_unique<NoHessian>();
            }
        };
        std::vector<BackflowSlaterDeterminant::Block> blocks;
        blocks.push_back({0, 1, std::make_unique<NoHessian>()});
        std::vector<BackflowSlaterDeterminant::Channel> ch(1);
        ch[0].source_first = 1;
        ch[0].source_last = 2;
        ch[0].params = {Parameter{"", 0.1, false}, Parameter{"", 1.0, false}};
        CHECK_THROWS_WITH(BackflowSlaterDeterminant(std::move(blocks), std::move(ch)),
                          ContainsSubstring("Hessian"));
    }
}

TEST_CASE("backflow derivatives hold over random configurations and across the cutoff",
          "[backflow]") {
    // Sixty random configurations per system, then configurations with one
    // pair just inside and just outside the cutoff L of its channel, where
    // eta and its first two derivatives go to zero (C = 3).
    std::mt19937_64 rng(20261107);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const std::vector<System> systems = {hexagonal_system(), periodic_system(), open_system()};
    for (std::size_t c = 0; c < systems.size(); ++c) {
        INFO("system " << c);
        const System& sys = systems[c];
        std::vector<std::vector<double>> configs;
        for (int k = 0; k < 60; ++k) {
            std::vector<double> pos;
            for (std::size_t p = 0; p < sys.np; ++p) {
                if (c == 0) {
                    const auto [x, y] = hex_point(unit(rng), unit(rng));
                    pos.push_back(x);
                    pos.push_back(y);
                } else if (c == 1) {
                    pos.push_back(6.0 * unit(rng));
                    pos.push_back(6.0 * unit(rng));
                } else {
                    pos.push_back(3.0 * unit(rng) - 1.5);
                    pos.push_back(3.0 * unit(rng) - 1.5);
                }
            }
            configs.push_back(pos);
        }
        // Particle 1 at L (1 -+ 1e-3) from particle 0 (the like-spin channel).
        const double L = sys.channels[0].L;
        for (double f : {1.0 - 1e-3, 1.0 + 1e-3}) {
            std::vector<double> pos = configs[0];
            pos[2] = pos[0] + f * L * 0.6;
            pos[3] = pos[1] + f * L * 0.8;
            configs.push_back(pos);
        }
        Batch b(sys, configs);
        TrialWavefunction psi = trial_of(sys);
        psi.evaluate_batch(b.ws);
        check_against_oracle(sys, b.ws);
    }
}

TEST_CASE("an electron-hole backflow channel keeps the kinetic term finite at contact",
          "[backflow][physics]") {
    // The hole channel has eta'(0) = c1 - C c0 / L = 0.25, not zero: between
    // distinguishable particles the extra term of ln|D| is anisotropic,
    // proportional to the vector r_eh, and adds no 1/r to the Laplacian.  So
    // sum_k (lap + |grad|^2) of the determinant converges as an electron and
    // a hole meet.
    const System sys = open_system();
    double c0 = sys.channels[1].c[0], c1 = sys.channels[1].c[1];
    REQUIRE(std::abs(c1 - 3.0 * c0 / sys.channels[1].L) > 0.1);
    std::vector<double> kinetic;
    for (double eps : {1e-2, 1e-3, 1e-4}) {
        std::vector<std::vector<double>> configs = {open_configs()[0]};
        configs[0][6] = configs[0][0] + eps;   // hole 3 next to electron 0
        configs[0][7] = configs[0][1];
        Batch b(sys, configs);
        TrialWavefunction psi = trial_of(sys);
        psi.evaluate_batch(b.ws);
        double t = 0.0;
        for (std::size_t q = 0; q < b.ws.gradients().size(); ++q)
            t += b.ws.laplacians()[q] + b.ws.gradients()[q] * b.ws.gradients()[q];
        kinetic.push_back(t);
    }
    CHECK(std::abs(kinetic[2] - kinetic[1]) < 0.2 * std::abs(kinetic[1] - kinetic[0]) + 1e-6);
    CHECK(std::isfinite(kinetic[2]));
}
