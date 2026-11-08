// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Backflow with twisted boundary conditions
// (wavefunction/complex_backflow_slater_determinant.h): the component contract
// battery against an independent oracle that forms the quasi-coordinates and
// the complex determinants by its own loops, with central finite differences
// of ln ψ for both parts of every derivative; and three exact relations —
// η = 0 is the complex Slater determinant, the Γ twist is the real backflow
// determinant up to a constant, and −θ gives ψ*.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "population/simulation_cell.h"
#include "population/walker.h"
#include "population/walker_set.h"
#include "wavefunction/backflow_slater_determinant.h"
#include "wavefunction/complex_backflow_slater_determinant.h"
#include "wavefunction/complex_orbital_set.h"
#include "wavefunction/complex_slater_determinant.h"
#include "wavefunction/orbital_set.h"
#include "wavefunction/trial_wavefunction.h"

using namespace alea;
using cplx = std::complex<double>;
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
    std::vector<double> twist;
    std::vector<std::pair<std::size_t, std::size_t>>        blocks;
    std::vector<std::shared_ptr<TwistedPlaneWaveOrbitalSet>> orbitals;
    std::vector<OracleChannel>                               channels;
    std::vector<std::vector<double>>                         free_params;   // [c_0, (c_1), c_2.., L]
};

double oracle_eta(const OracleChannel& ch, double r) {
    if (r >= ch.L) return 0.0;
    double P = 0.0;
    for (std::size_t k = 0; k < ch.c.size(); ++k) P += ch.c[k] * std::pow(r, static_cast<double>(k));
    return std::pow(1.0 - r / ch.L, ch.C) * P;
}

// ln ψ = ln|ψ| + i arg ψ at positions pos ([np*dim], AoS); arg in (−π, π].
cplx oracle(const System& sys, const std::vector<double>& pos) {
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
    double log_abs = 0.0, arg = 0.0;
    for (std::size_t b = 0; b < sys.blocks.size(); ++b) {
        const auto [f, l] = sys.blocks[b];
        const std::size_t n = l - f;
        std::vector<cplx> m(n * n);
        // Plane waves by hand: e^{i q_c · x_r}.
        const std::span<const double> q = sys.orbitals[b]->wavevectors();
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t c = 0; c < n; ++c) {
                double a = 0.0;
                for (std::size_t d = 0; d < dim; ++d) a += q[c * dim + d] * x[(f - g0 + r) * dim + d];
                m[r * n + c] = std::polar(1.0, a);
            }
        // Determinant by elimination with partial pivoting.
        for (std::size_t k = 0; k < n; ++k) {
            std::size_t piv = k;
            for (std::size_t r = k + 1; r < n; ++r)
                if (std::abs(m[r * n + k]) > std::abs(m[piv * n + k])) piv = r;
            if (piv != k) {
                for (std::size_t c = 0; c < n; ++c) std::swap(m[k * n + c], m[piv * n + c]);
                arg += kPi;
            }
            const cplx d = m[k * n + k];
            log_abs += std::log(std::abs(d));
            arg += std::arg(d);
            for (std::size_t r = k + 1; r < n; ++r) {
                const cplx factor = m[r * n + k] / d;
                for (std::size_t c = k; c < n; ++c) m[r * n + c] -= factor * m[k * n + c];
            }
        }
    }
    return {log_abs, std::remainder(arg, 2.0 * kPi)};
}

// a − b with the phase difference folded into (−π, π].
cplx log_diff(cplx a, cplx b) {
    return {a.real() - b.real(), std::remainder(a.imag() - b.imag(), 2.0 * kPi)};
}

std::unique_ptr<ComplexBackflowSlaterDeterminant> make_component(const System& sys) {
    std::vector<ComplexBackflowSlaterDeterminant::Block> blocks;
    for (std::size_t b = 0; b < sys.blocks.size(); ++b)
        blocks.push_back({sys.blocks[b].first, sys.blocks[b].second, sys.orbitals[b]->clone()});
    std::vector<ComplexBackflowSlaterDeterminant::Channel> channels;
    for (std::size_t c = 0; c < sys.channels.size(); ++c) {
        ComplexBackflowSlaterDeterminant::Channel ch;
        ch.source_first = sys.channels[c].source_first;
        ch.source_last = sys.channels[c].source_last;
        ch.truncation = sys.channels[c].C;
        for (double v : sys.free_params[c]) ch.params.push_back(Parameter{"", v, true});
        channels.push_back(std::move(ch));
    }
    return std::make_unique<ComplexBackflowSlaterDeterminant>(std::move(blocks),
                                                              std::move(channels), sys.cell);
}

std::shared_ptr<TwistedPlaneWaveOrbitalSet> twisted(const SimulationCell& cell, std::size_t n,
                                                    const std::vector<double>& twist) {
    return std::shared_ptr<TwistedPlaneWaveOrbitalSet>(
        make_twisted_plane_wave_orbitals(cell, n, twist));
}

// Square 6×6 cell (inradius 3): five like-spin particles at the twist
// (0.2, 0.13) (at (0.2, 0.1) the fifth |k+θ| ties with the sixth), a
// same-group channel with L inside the inradius.
System square_system(std::vector<double> twist = {0.2, 0.13}) {
    System s;
    s.np = 5;
    s.cell = SimulationCell(2, std::vector<double>{6.0, 0.0, 0.0, 6.0});
    s.twist = twist;
    s.blocks = {{0, 5}};
    s.orbitals = {twisted(s.cell, 5, twist)};
    const double c0 = 0.3, L = 2.9;
    s.channels.push_back({0, 5, 3, L, {c0, 3.0 * c0 / L, 0.04, -0.01}});
    s.free_params.push_back({c0, 0.04, -0.01, L});
    return s;
}

// Two spin blocks of one group (up [0,2), down [2,3)) displaced by a second
// group [3,5) — the electron-hole channel, c_1 free.
System two_block_system() {
    System s;
    s.np = 5;
    s.cell = SimulationCell(2, std::vector<double>{6.0, 0.0, 0.0, 6.0});
    s.twist = {-0.3, 0.15};
    s.blocks = {{0, 2}, {2, 3}};
    s.orbitals = {twisted(s.cell, 2, s.twist), twisted(s.cell, 1, s.twist)};
    s.channels.push_back({3, 5, 3, 2.6, {-0.2, 0.15, 0.04}});
    s.free_params.push_back({-0.2, 0.15, 0.04, 2.6});
    return s;
}

// Hexagonal 6-cell (inradius 3): three like-spin particles displaced by each
// other and by two holes [3,5), at the twist (0.15, −0.35).
System hexagonal_system() {
    System s;
    s.np = 5;
    s.cell = SimulationCell(2, std::vector<double>{6.0, 0.0, 3.0, 3.0 * std::sqrt(3.0)});
    s.twist = {0.15, -0.35};
    s.blocks = {{0, 3}};
    s.orbitals = {twisted(s.cell, 3, s.twist)};
    const double c0 = 0.25, L = 2.5;
    s.channels.push_back({0, 3, 3, L, {c0, 3.0 * c0 / L, -0.03}});
    s.free_params.push_back({c0, -0.03, L});
    s.channels.push_back({3, 5, 3, 2.2, {-0.2, 0.15, 0.04}});
    s.free_params.push_back({-0.2, 0.15, 0.04, 2.2});
    return s;
}

std::pair<double, double> hex_point(double f1, double f2) {
    return {6.0 * f1 + 3.0 * f2, 3.0 * std::sqrt(3.0) * f2};
}

std::vector<std::vector<double>> square_configs() {
    return {{0.5, 0.7, 2.1, 1.0, 3.9, 4.2, 5.1, 2.0, 1.4, 5.3},
            {1.0, 1.9, 2.6, 0.3, 4.4, 3.1, 5.6, 5.5, 0.2, 4.0}};
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

std::vector<std::pair<System, std::vector<std::vector<double>>>> all_cases() {
    return {{square_system(), square_configs()},
            {two_block_system(), square_configs()},
            {hexagonal_system(), hexagonal_configs()}};
}

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

// Central differences of the oracle's ln ψ (both parts): [np*dim] each,
// Richardson-extrapolated over h and h/2; *_err = |D(h) − D(h/2)|.
void fd_derivatives(const System& sys, const std::vector<double>& pos, std::vector<cplx>& grad,
                    std::vector<cplx>& lap, std::vector<double>& grad_err,
                    std::vector<double>& lap_err) {
    const cplx f0 = oracle(sys, pos);
    grad.assign(pos.size(), 0.0);
    lap.assign(pos.size(), 0.0);
    grad_err.assign(pos.size(), 0.0);
    lap_err.assign(pos.size(), 0.0);
    for (std::size_t q = 0; q < pos.size(); ++q) {
        cplx g[2] = {}, l[2] = {};
        for (int level = 0; level < 2; ++level) {
            const double h = level == 0 ? 2e-4 : 1e-4;
            std::vector<double> p = pos, m = pos;
            p[q] += h;
            m[q] -= h;
            const cplx dp = log_diff(oracle(sys, p), f0), dm = log_diff(oracle(sys, m), f0);
            g[level] = (dp - dm) / (2.0 * h);
            l[level] = (dp + dm) / (h * h);
        }
        grad[q] = (4.0 * g[1] - g[0]) / 3.0;
        lap[q] = (4.0 * l[1] - l[0]) / 3.0;
        grad_err[q] = std::abs(g[0] - g[1]);
        lap_err[q] = std::abs(l[0] - l[1]);
    }
}

void fd_derivatives(const System& sys, const std::vector<double>& pos, std::vector<cplx>& grad,
                    std::vector<cplx>& lap) {
    std::vector<double> ge, le;
    fd_derivatives(sys, pos, grad, lap, ge, le);
}

void check_against_oracle(const System& sys, const WalkerSet& ws) {
    const std::size_t nw = ws.num_walkers(), np = sys.np, dim = sys.dim;
    for (std::size_t w = 0; w < nw; ++w) {
        const std::vector<double> pos = config_of(ws, w);
        INFO("walker " << w);
        CHECK(ws.log_psi()[w] == Catch::Approx(oracle(sys, pos).real()).epsilon(1e-12));
        std::vector<cplx> g, l;
        std::vector<double> ge, le;
        fd_derivatives(sys, pos, g, l, ge, le);
        for (std::size_t k = 0; k < np; ++k)
            for (std::size_t d = 0; d < dim; ++d) {
                const std::size_t q = k * dim + d, j = (k * dim + d) * nw + w;
                INFO("particle " << k << " dim " << d << "  fd error estimate " << le[q]);
                CHECK(ws.gradients()[j] == Catch::Approx(g[q].real()).margin(1e-7 + ge[q]));
                CHECK(ws.phase_gradients()[j] == Catch::Approx(g[q].imag()).margin(1e-7 + ge[q]));
                CHECK(ws.laplacians()[j] == Catch::Approx(l[q].real()).margin(2e-5 + le[q]));
                CHECK(ws.phase_laplacians()[j] == Catch::Approx(l[q].imag()).margin(2e-5 + le[q]));
            }
    }
}

TrialWavefunction trial_of(const System& sys) {
    TrialWavefunction psi;
    psi.add_component(make_component(sys));
    return psi;
}

void propose(WalkerSet& ws, std::size_t p, const std::vector<double>& new_r) {
    const std::size_t nw = ws.num_walkers(), dim = ws.dim();
    std::vector<double> soa(dim * nw);
    for (std::size_t d = 0; d < dim; ++d)
        for (std::size_t w = 0; w < nw; ++w) soa[d * nw + w] = new_r[d];
    ws.propose_move_batch(p, soa);
}

void commit(TrialWavefunction& psi, WalkerSet& ws, std::size_t p, std::span<const bool> accept,
            bool deferred) {
    if (deferred) psi.apply_move_batch_deferred(ws, p, accept);
    else          psi.apply_move_batch(ws, p, accept);
    ws.apply_move(p, ws.new_pos(), ws.new_grad(), ws.new_lap(), accept);
}

// ── Parameter derivatives ────────────────────────────────────────────────────

// sys with the component's parameter q (parameters() order: each channel's
// [c_0, (c_1), c_2.., L]) shifted by delta, in the oracle's coefficients too:
// a like-spin channel derives c_1 = C c_0 / L, so c_0 and L both move it.
System with_shifted_parameter(System sys, std::size_t q, double delta) {
    const std::size_t g0 = sys.blocks.front().first, g1 = sys.blocks.back().second;
    for (std::size_t c = 0; c < sys.channels.size(); ++c) {
        std::vector<double>& fp = sys.free_params[c];
        if (q >= fp.size()) {
            q -= fp.size();
            continue;
        }
        fp[q] += delta;
        OracleChannel& ch = sys.channels[c];
        ch.L = fp.back();
        ch.c = {fp[0]};
        if (ch.source_first == g0 && ch.source_last == g1)
            ch.c.push_back(static_cast<double>(ch.C) * fp[0] / ch.L);
        for (std::size_t k = 1; k + 1 < fp.size(); ++k) ch.c.push_back(fp[k]);
        return sys;
    }
    throw std::out_of_range("no backflow parameter " + std::to_string(q));
}

std::size_t n_parameters(const System& sys) {
    std::size_t n = 0;
    for (const auto& fp : sys.free_params) n += fp.size();
    return n;
}

TrialWavefunction free_trial_of(const System& sys) {
    TrialWavefunction psi = trial_of(sys);
    for (Parameter& p : psi.components()[0]->parameters()) p.is_fixed = false;
    return psi;
}

// The four planes (∇u, ∇²u, ∇φ, ∇²φ) of ψ at the configurations, parameter q
// shifted by delta.
std::array<std::vector<double>, 4> planes_at(const System& sys,
                                             const std::vector<std::vector<double>>& configs,
                                             std::size_t q, double delta) {
    const System shifted = with_shifted_parameter(sys, q, delta);
    Batch b(shifted, configs);
    TrialWavefunction psi = trial_of(shifted);
    psi.evaluate_batch(b.ws);
    const auto copy = [](std::span<const double> s) { return std::vector<double>(s.begin(), s.end()); };
    return {copy(b.ws.gradients()), copy(b.ws.laplacians()), copy(b.ws.phase_gradients()),
            copy(b.ws.phase_laplacians())};
}

// The component's O (both parts) and curvature (four planes) at walker set ws,
// every parameter free.
struct ParameterDerivatives {
    std::vector<double> re, im;                   // [n * nw]
    std::array<std::vector<double>, 4> planes;    // dgrad, dlap, dpgrad, dplap [n * field]
};
ParameterDerivatives parameter_derivatives_of(const TrialWavefunction& psi, const WalkerSet& ws) {
    const std::size_t n = psi.total_optimizable_params(), nw = ws.num_walkers();
    const std::size_t field = ws.n_particles() * ws.dim() * nw;
    ParameterDerivatives out;
    out.re.assign(n * nw, 0.0);
    out.im.assign(n * nw, 0.0);
    psi.compute_complex_parameter_derivatives(ws, out.re, out.im);
    for (auto& p : out.planes) p.assign(n * field, 0.0);
    std::vector<char> supported(n, 0);
    REQUIRE(psi.compute_complex_parameter_gradient_derivatives(
        ws, out.planes[0], out.planes[1], out.planes[2], out.planes[3], supported));
    for (char s : supported) CHECK(s == 1);
    return out;
}

ParameterDerivatives fresh_parameter_derivatives(const System& sys,
                                                 const std::vector<std::vector<double>>& configs) {
    Batch b(sys, configs);
    TrialWavefunction psi = free_trial_of(sys);
    psi.evaluate_batch(b.ws);
    return parameter_derivatives_of(psi, b.ws);
}

// O against central differences of the oracle's ln ψ in every parameter, and
// the curvature against central differences of the component's own planes
// (checked against the oracle by the battery above); Richardson over h, h/2,
// with |D(h) − D(h/2)| joining the margin.
void check_parameter_derivatives(const System& sys,
                                 const std::vector<std::vector<double>>& configs) {
    const ParameterDerivatives got = fresh_parameter_derivatives(sys, configs);
    const std::size_t nw = configs.size(), field = sys.np * sys.dim * nw;
    for (std::size_t q = 0; q < n_parameters(sys); ++q) {
        INFO("parameter " << q);
        double value = 0.0;
        {
            std::size_t r = q;
            for (const auto& fp : sys.free_params) {
                if (r < fp.size()) {
                    value = fp[r];
                    break;
                }
                r -= fp.size();
            }
        }
        const double h = 1e-4 * std::max(1.0, std::abs(value));
        std::vector<cplx> o[2];
        std::array<std::vector<double>, 4> fd[2];
        for (int level = 0; level < 2; ++level) {
            const double hl = level == 0 ? h : 0.5 * h;
            const System up = with_shifted_parameter(sys, q, hl);
            const System down = with_shifted_parameter(sys, q, -hl);
            o[level].resize(nw);
            for (std::size_t w = 0; w < nw; ++w)
                o[level][w] = log_diff(oracle(up, configs[w]), oracle(down, configs[w])) / (2.0 * hl);
            const auto pu = planes_at(sys, configs, q, hl), pd = planes_at(sys, configs, q, -hl);
            for (std::size_t k = 0; k < 4; ++k) {
                fd[level][k].resize(field);
                for (std::size_t e = 0; e < field; ++e)
                    fd[level][k][e] = (pu[k][e] - pd[k][e]) / (2.0 * hl);
            }
        }
        for (std::size_t w = 0; w < nw; ++w) {
            const cplx ref = (4.0 * o[1][w] - o[0][w]) / 3.0;
            const double err = std::abs(o[0][w] - o[1][w]);
            CHECK(got.re[q * nw + w] == Catch::Approx(ref.real()).margin(1e-8 + err));
            CHECK(got.im[q * nw + w] == Catch::Approx(ref.imag()).margin(1e-8 + err));
        }
        for (std::size_t k = 0; k < 4; ++k)
            for (std::size_t e = 0; e < field; ++e) {
                const double ref = (4.0 * fd[1][k][e] - fd[0][k][e]) / 3.0;
                const double err = std::abs(fd[0][k][e] - fd[1][k][e]);
                INFO("plane " << k << " entry " << e << "  fd error estimate " << err);
                CHECK(got.planes[k][q * field + e] ==
                      Catch::Approx(ref).margin((k % 2 == 0 ? 1e-7 : 1e-6) + err));
            }
    }
}

}  // namespace

// ── Orbitals ─────────────────────────────────────────────────────────────────

TEST_CASE("twisted plane-wave Hessians match evaluate() and finite differences of the gradient",
          "[backflow][twist][orbitals]") {
    const SimulationCell box(2, std::vector<double>{6.0, 0.0, 0.0, 6.0});
    const auto orb = make_twisted_plane_wave_orbitals(box, 5, {0.2, 0.13});
    REQUIRE(orb->supports_hessian());
    const std::size_t n = orb->size();
    const std::vector<double> pos = {0.4, 5.9};
    std::vector<cplx> val(n), grad(n * 2), hess(n * 4);
    orb->evaluate_hessian(pos, 2, val, grad, hess);
    std::vector<double> vr(n), vi(n), gr(n * 2), gi(n * 2), lr(n * 2), li(n * 2);
    orb->evaluate(pos, 2, {vr, vi}, {gr, gi}, {lr, li}, 0, 1);
    const double h = 1e-5;
    for (std::size_t m = 0; m < n; ++m) {
        CHECK(val[m].real() == Catch::Approx(vr[m]));
        CHECK(val[m].imag() == Catch::Approx(vi[m]).margin(1e-15));
        for (std::size_t a = 0; a < 2; ++a) {
            CHECK(grad[m * 2 + a].real() == Catch::Approx(gr[m * 2 + a]).margin(1e-14));
            CHECK(grad[m * 2 + a].imag() == Catch::Approx(gi[m * 2 + a]).margin(1e-14));
            CHECK(hess[(m * 2 + a) * 2 + a].real() == Catch::Approx(lr[m * 2 + a]).margin(1e-13));
            CHECK(hess[(m * 2 + a) * 2 + a].imag() == Catch::Approx(li[m * 2 + a]).margin(1e-13));
            for (std::size_t b = 0; b < 2; ++b) {
                std::vector<double> pp = pos, pm = pos;
                pp[b] += h;
                pm[b] -= h;
                std::vector<cplx> vp(n), gp(n * 2), hp(n * 4), gm(n * 2);
                orb->evaluate_hessian(pp, 2, vp, gp, hp);
                orb->evaluate_hessian(pm, 2, vp, gm, hp);
                const cplx fd = (gp[m * 2 + a] - gm[m * 2 + a]) / (2 * h);
                CHECK(hess[(m * 2 + a) * 2 + b].real() == Catch::Approx(fd.real()).margin(1e-8));
                CHECK(hess[(m * 2 + a) * 2 + b].imag() == Catch::Approx(fd.imag()).margin(1e-8));
            }
        }
    }
}

TEST_CASE("twisted plane-wave third derivatives match finite differences of the Hessian",
          "[backflow][twist][orbitals]") {
    const SimulationCell box(2, std::vector<double>{6.0, 0.0, 0.0, 6.0});
    const auto orb = make_twisted_plane_wave_orbitals(box, 5, {0.2, 0.13});
    REQUIRE(orb->supports_third_derivatives());
    const std::size_t n = orb->size();
    const std::vector<double> pos = {0.4, 5.9};
    std::vector<cplx> third(n * 8);
    orb->evaluate_third_derivatives(pos, 2, third);
    const double h = 1e-5;
    for (std::size_t c = 0; c < 2; ++c) {
        std::vector<double> pp = pos, pm = pos;
        pp[c] += h;
        pm[c] -= h;
        std::vector<cplx> v(n), g(n * 2), hp(n * 4), hm(n * 4);
        orb->evaluate_hessian(pp, 2, v, g, hp);
        orb->evaluate_hessian(pm, 2, v, g, hm);
        for (std::size_t m = 0; m < n; ++m)
            for (std::size_t a = 0; a < 2; ++a)
                for (std::size_t b = 0; b < 2; ++b) {
                    const std::size_t e = (m * 2 + a) * 2 + b;
                    const cplx fd = (hp[e] - hm[e]) / (2 * h);
                    CHECK(third[e * 2 + c].real() == Catch::Approx(fd.real()).margin(1e-8));
                    CHECK(third[e * 2 + c].imag() == Catch::Approx(fd.imag()).margin(1e-8));
                }
    }
}

// ── The contract battery ─────────────────────────────────────────────────────

TEST_CASE("complex backflow: ln|psi| and both parts of grad and lap of ln psi match the oracle",
          "[backflow][twist]") {
    const auto cases = all_cases();
    for (std::size_t c = 0; c < cases.size(); ++c) {
        INFO("system " << c);
        const auto& [sys, configs] = cases[c];
        Batch b(sys, configs);
        TrialWavefunction psi = trial_of(sys);
        REQUIRE(psi.is_complex());
        psi.evaluate_batch(b.ws);
        check_against_oracle(sys, b.ws);
    }
}

TEST_CASE("complex backflow adds to the batch totals", "[backflow][twist]") {
    const System sys = square_system();
    Batch b(sys, square_configs());
    b.ws.enable_phase();
    for (double& v : b.ws.log_psi()) v = 1.5;
    for (double& v : b.ws.gradients()) v = 0.25;
    for (double& v : b.ws.laplacians()) v = -0.5;
    for (double& v : b.ws.phase_gradients()) v = 0.125;
    for (double& v : b.ws.phase_laplacians()) v = 0.75;
    auto det = make_component(sys);
    det->evaluate_batch(b.ws);
    TrialWavefunction ref = trial_of(sys);
    Batch r(sys, square_configs());
    ref.evaluate_batch(r.ws);
    for (std::size_t k = 0; k < b.ws.gradients().size(); ++k) {
        CHECK(b.ws.gradients()[k] == Catch::Approx(0.25 + r.ws.gradients()[k]));
        CHECK(b.ws.laplacians()[k] == Catch::Approx(-0.5 + r.ws.laplacians()[k]));
        CHECK(b.ws.phase_gradients()[k] == Catch::Approx(0.125 + r.ws.phase_gradients()[k]));
        CHECK(b.ws.phase_laplacians()[k] == Catch::Approx(0.75 + r.ws.phase_laplacians()[k]));
    }
    CHECK(b.ws.log_psi()[0] == Catch::Approx(1.5 + det->log_psi_c()[0]));
}

TEST_CASE("complex backflow ratio_batch: ln|R|, arg R and the moved particle's derivatives",
          "[backflow][twist]") {
    const auto cases = all_cases();
    for (std::size_t c = 0; c < cases.size(); ++c) {
        const auto& [sys, configs] = cases[c];
        // A row of the determinant and a source particle (or the last row).
        for (std::size_t p : {std::size_t{1}, sys.np - 1}) {
            INFO("system " << c << " moves particle " << p);
            Batch b(sys, configs);
            TrialWavefunction psi = trial_of(sys);
            psi.evaluate_batch(b.ws);
            const std::vector<double> to = {3.3, 1.2};
            propose(b.ws, p, to);
            psi.ratio_batch(b.ws, p);
            const std::size_t nw = b.ws.num_walkers();
            for (std::size_t w = 0; w < nw; ++w) {
                std::vector<double> before = config_of(b.ws, w), after = before;
                for (std::size_t d = 0; d < 2; ++d) after[p * 2 + d] = to[d];
                const cplx dlog = log_diff(oracle(sys, after), oracle(sys, before));
                CHECK(b.ws.log_ratios()[w] == Catch::Approx(dlog.real()).margin(1e-11));
                CHECK(b.ws.phase_diffs()[w] == Catch::Approx(dlog.imag()).margin(1e-11));
                std::vector<cplx> g, l;
                fd_derivatives(sys, after, g, l);
                for (std::size_t d = 0; d < 2; ++d) {
                    CHECK(b.ws.new_grad()[d * nw + w] == Catch::Approx(g[p * 2 + d].real()).margin(1e-7));
                    CHECK(b.ws.new_phase_grad()[d * nw + w] ==
                          Catch::Approx(g[p * 2 + d].imag()).margin(1e-7));
                    CHECK(b.ws.new_lap()[d * nw + w] == Catch::Approx(l[p * 2 + d].real()).margin(2e-5));
                    CHECK(b.ws.new_phase_lap()[d * nw + w] ==
                          Catch::Approx(l[p * 2 + d].imag()).margin(2e-5));
                }
            }
        }
    }
}

TEST_CASE("complex backflow commits match a fresh evaluation, eager and deferred, across wraps",
          "[backflow][twist]") {
    const auto cases = all_cases();
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
            // Particle 0 leaves the cell (the WalkerSet wraps it: its row
            // changes gauge), a source/last particle moves, then particle 0
            // moves again from its wrapped position.  The first walker accepts
            // every move, the second rejects the middle one.
            const std::vector<std::pair<std::size_t, std::vector<double>>> moves = {
                {0, {-0.4, 6.3}}, {sys.np - 1, {4.1, 5.0}}, {0, {0.9, 1.4}}};
            for (std::size_t mv = 0; mv < moves.size(); ++mv) {
                const auto& [p, to] = moves[mv];
                if (deferred) psi.refresh_particle_derivatives(b.ws, p);
                const std::size_t nw = b.ws.num_walkers();
                std::vector<std::vector<double>> before(nw);
                for (std::size_t w = 0; w < nw; ++w) before[w] = config_of(b.ws, w);
                propose(b.ws, p, to);
                psi.ratio_batch(b.ws, p);
                // |R| is gauge-free at every move.
                for (std::size_t w = 0; w < nw; ++w) {
                    std::vector<double> after = before[w];
                    for (std::size_t d = 0; d < 2; ++d) after[p * 2 + d] = to[d];
                    CHECK(b.ws.log_ratios()[w] ==
                          Catch::Approx((oracle(sys, after) - oracle(sys, before[w])).real())
                              .margin(1e-11));
                }
                std::array<bool, 2> accept = {true, mv != 1};
                commit(psi, b.ws, p, accept, deferred);
            }
            if (deferred) psi.refresh_derivatives(b.ws);
            check_against_oracle(sys, b.ws);
        }
    }
}

TEST_CASE("complex backflow refresh_particle_derivatives makes a stale drift current",
          "[backflow][twist]") {
    const System sys = hexagonal_system();
    Batch b(sys, hexagonal_configs());
    TrialWavefunction psi = trial_of(sys);
    psi.evaluate_batch(b.ws);
    propose(b.ws, 0, {2.6, 1.1});
    psi.ratio_batch(b.ws, 0);
    std::array<bool, 2> accept = {true, true};
    commit(psi, b.ws, 0, accept, true);
    // Particle 2's gradient changed with the move; refresh it alone.
    psi.refresh_particle_derivatives(b.ws, 2);
    const std::size_t nw = b.ws.num_walkers();
    for (std::size_t w = 0; w < nw; ++w) {
        std::vector<cplx> g, l;
        fd_derivatives(sys, config_of(b.ws, w), g, l);
        for (std::size_t d = 0; d < 2; ++d) {
            const std::size_t j = (2 * 2 + d) * nw + w;
            CHECK(b.ws.gradients()[j] == Catch::Approx(g[2 * 2 + d].real()).margin(1e-7));
            CHECK(b.ws.phase_gradients()[j] == Catch::Approx(g[2 * 2 + d].imag()).margin(1e-7));
        }
    }
}

TEST_CASE("complex backflow ratio over disjoint walker ranges equals the full batch",
          "[backflow][twist]") {
    const System sys = square_system();
    Batch whole(sys, square_configs()), split(sys, square_configs());
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
        CHECK(split.ws.phase_diffs()[w] == whole.ws.phase_diffs()[w]);
        for (std::size_t d = 0; d < 2; ++d) {
            CHECK(split.ws.new_grad()[d * 2 + w] == whole.ws.new_grad()[d * 2 + w]);
            CHECK(split.ws.new_lap()[d * 2 + w] == whole.ws.new_lap()[d * 2 + w]);
            CHECK(split.ws.new_phase_grad()[d * 2 + w] == whole.ws.new_phase_grad()[d * 2 + w]);
            CHECK(split.ws.new_phase_lap()[d * 2 + w] == whole.ws.new_phase_lap()[d * 2 + w]);
        }
    }
}

TEST_CASE("complex backflow permute_caches carries every walker's state", "[backflow][twist]") {
    const System sys = hexagonal_system();
    Batch b(sys, hexagonal_configs());
    TrialWavefunction psi = trial_of(sys);
    psi.evaluate_batch(b.ws);
    const std::vector<std::size_t> src = {1, 1, 0};   // swap, duplicate, grow
    b.ws.apply_permutation(src);
    psi.permute_caches(src);
    // The phase planes travelled with the walkers.
    check_against_oracle(sys, b.ws);
    propose(b.ws, 1, {2.2, 3.1});
    psi.ratio_batch(b.ws, 1);
    for (std::size_t w = 0; w < src.size(); ++w) {
        std::vector<double> before = config_of(b.ws, w), after = before;
        after[2] = 2.2;
        after[3] = 3.1;
        const cplx dlog = log_diff(oracle(sys, after), oracle(sys, before));
        CHECK(b.ws.log_ratios()[w] == Catch::Approx(dlog.real()).margin(1e-11));
        CHECK(b.ws.phase_diffs()[w] == Catch::Approx(dlog.imag()).margin(1e-11));
    }
    // The component's own caches travelled too: a refresh updates the totals
    // by difference from them.
    std::array<bool, 3> accept = {true, true, true};
    commit(psi, b.ws, 1, accept, true);
    psi.refresh_derivatives(b.ws);
    check_against_oracle(sys, b.ws);
}

TEST_CASE("complex backflow clone reproduces the original bit for bit", "[backflow][twist]") {
    const System sys = square_system();
    Batch a(sys, square_configs()), b(sys, square_configs());
    TrialWavefunction psi = trial_of(sys);
    TrialWavefunction copy = psi.clone();
    psi.evaluate_batch(a.ws);
    copy.evaluate_batch(b.ws);
    for (std::size_t k = 0; k < a.ws.gradients().size(); ++k) {
        CHECK(a.ws.gradients()[k] == b.ws.gradients()[k]);
        CHECK(a.ws.laplacians()[k] == b.ws.laplacians()[k]);
        CHECK(a.ws.phase_gradients()[k] == b.ws.phase_gradients()[k]);
        CHECK(a.ws.phase_laplacians()[k] == b.ws.phase_laplacians()[k]);
    }
    CHECK(a.ws.log_psi()[1] == b.ws.log_psi()[1]);
}

TEST_CASE("complex backflow derivatives hold over random configurations and across the cutoff",
          "[backflow][twist]") {
    std::mt19937_64 rng(20261108);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const std::vector<System> systems = {hexagonal_system(), square_system(), two_block_system()};
    for (std::size_t c = 0; c < systems.size(); ++c) {
        INFO("system " << c);
        const System& sys = systems[c];
        std::vector<std::vector<double>> configs;
        for (int k = 0; k < 40; ++k) {
            std::vector<double> pos;
            for (std::size_t p = 0; p < sys.np; ++p) {
                if (c == 0) {
                    const auto [x, y] = hex_point(unit(rng), unit(rng));
                    pos.push_back(x);
                    pos.push_back(y);
                } else {
                    pos.push_back(6.0 * unit(rng));
                    pos.push_back(6.0 * unit(rng));
                }
            }
            configs.push_back(pos);
        }
        // A pair of the first channel at L(1 ∓ 1e-3).
        const OracleChannel& ch = sys.channels[0];
        const std::size_t row = sys.blocks.front().first;
        const std::size_t src = ch.source_first == row ? row + 1 : ch.source_first;
        for (double f : {1.0 - 1e-3, 1.0 + 1e-3}) {
            std::vector<double> pos = configs[0];
            pos[src * 2] = pos[row * 2] + f * ch.L * 0.6;
            pos[src * 2 + 1] = pos[row * 2 + 1] + f * ch.L * 0.8;
            configs.push_back(pos);
        }
        Batch b(sys, configs);
        TrialWavefunction psi = trial_of(sys);
        psi.evaluate_batch(b.ws);
        check_against_oracle(sys, b.ws);
    }
}

// ── Exact relations ──────────────────────────────────────────────────────────

TEST_CASE("complex backflow with eta = 0 is the complex Slater determinant",
          "[backflow][twist]") {
    System sys = hexagonal_system();
    for (auto& fp : sys.free_params)
        for (std::size_t k = 0; k + 1 < fp.size(); ++k) fp[k] = 0.0;
    Batch a(sys, hexagonal_configs()), b(sys, hexagonal_configs());
    TrialWavefunction bf = trial_of(sys);
    bf.evaluate_batch(a.ws);
    std::vector<ComplexSlaterDeterminant::Block> blocks;
    blocks.push_back({0, 3, sys.orbitals[0]->clone()});
    TrialWavefunction plain;
    plain.add_component(std::make_unique<ComplexSlaterDeterminant>(std::move(blocks)));
    plain.evaluate_batch(b.ws);
    for (std::size_t w = 0; w < 2; ++w)
        CHECK(a.ws.log_psi()[w] == Catch::Approx(b.ws.log_psi()[w]).epsilon(1e-13));
    for (std::size_t k = 0; k < a.ws.gradients().size(); ++k) {
        CHECK(a.ws.gradients()[k] == Catch::Approx(b.ws.gradients()[k]).margin(1e-12));
        CHECK(a.ws.laplacians()[k] == Catch::Approx(b.ws.laplacians()[k]).margin(1e-11));
        CHECK(a.ws.phase_gradients()[k] == Catch::Approx(b.ws.phase_gradients()[k]).margin(1e-12));
        CHECK(a.ws.phase_laplacians()[k] == Catch::Approx(b.ws.phase_laplacians()[k]).margin(1e-11));
    }
}

TEST_CASE("complex backflow at the Gamma twist is the real backflow determinant up to a constant",
          "[backflow][twist]") {
    // e^{ik·r} over a closed shell spans the real cos/sin basis, so the two
    // determinants differ by a constant factor at every configuration: the
    // modulus derivatives agree, the phase is constant, and ln|ψ| shifts by
    // one constant across walkers.
    const System sys = square_system({0.0, 0.0});
    std::vector<std::vector<double>> configs = square_configs();
    configs.push_back({2.9, 0.2, 0.4, 3.3, 5.0, 1.7, 3.6, 5.9, 1.1, 2.4});
    Batch a(sys, configs), b(sys, configs);
    TrialWavefunction complex_psi = trial_of(sys);
    complex_psi.evaluate_batch(a.ws);

    std::vector<BackflowSlaterDeterminant::Block> blocks;
    blocks.push_back({0, 5, make_plane_wave_orbitals(sys.cell, 5)});
    std::vector<BackflowSlaterDeterminant::Channel> channels(1);
    channels[0].source_first = 0;
    channels[0].source_last = 5;
    channels[0].truncation = 3;
    for (double v : sys.free_params[0]) channels[0].params.push_back(Parameter{"", v, true});
    TrialWavefunction real_psi;
    real_psi.add_component(std::make_unique<BackflowSlaterDeterminant>(
        std::move(blocks), std::move(channels), sys.cell));
    real_psi.evaluate_batch(b.ws);

    const double shift = a.ws.log_psi()[0] - b.ws.log_psi()[0];
    for (std::size_t w = 1; w < configs.size(); ++w)
        CHECK(a.ws.log_psi()[w] - b.ws.log_psi()[w] == Catch::Approx(shift).margin(1e-11));
    for (std::size_t k = 0; k < a.ws.gradients().size(); ++k) {
        CHECK(a.ws.gradients()[k] == Catch::Approx(b.ws.gradients()[k]).margin(1e-10));
        CHECK(a.ws.laplacians()[k] == Catch::Approx(b.ws.laplacians()[k]).margin(1e-9));
        CHECK(a.ws.phase_gradients()[k] == Catch::Approx(0.0).margin(1e-10));
        CHECK(a.ws.phase_laplacians()[k] == Catch::Approx(0.0).margin(1e-9));
    }
}

TEST_CASE("a twist that shifts a closed shell multiplies the backflow determinant by "
          "e^{i theta . sum r}", "[backflow][twist]") {
    // When the occupied q are the Γ closed shell k plus θ, det[e^{i(k+θ)·x_j}]
    // = e^{iθ·Σ_j x_j} det[e^{ik·x_j}], and a like-spin channel over the whole
    // group leaves Σ_j x_j = Σ_j r_j (the pair terms η(r_ij) r_ij + η(r_ji) r_ji
    // cancel).  So ln ψ_θ = ln ψ_Γ + iθ·Σ r: every particle's phase gradient is
    // θ, the phase Laplacian 0, and the modulus part is the REAL backflow
    // determinant's at Γ — an exact check against independent code.
    const std::vector<double> t = {0.05, 0.03};   // the 5 lowest |k+θ|: Γ's shell + θ
    const System sys = square_system(t);
    const std::span<const double> b = sys.cell.reciprocal();
    double theta[2] = {};
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t d = 0; d < 2; ++d) theta[d] += t[i] * b[i * 2 + d];
    // The occupation really is the shifted closed shell.
    const std::span<const double> q = sys.orbitals[0]->wavevectors();
    for (std::size_t m = 0; m < 5; ++m) {
        const double kx = q[m * 2] - theta[0], ky = q[m * 2 + 1] - theta[1];
        REQUIRE(kx * kx + ky * ky < 1.2 * b[0] * b[0]);
    }

    std::vector<std::vector<double>> configs = square_configs();
    configs.push_back({2.9, 0.2, 0.4, 3.3, 5.0, 1.7, 3.6, 5.9, 1.1, 2.4});
    Batch a(sys, configs), r(sys, configs);
    TrialWavefunction twisted_psi = trial_of(sys);
    twisted_psi.evaluate_batch(a.ws);
    std::vector<BackflowSlaterDeterminant::Block> blocks;
    blocks.push_back({0, 5, make_plane_wave_orbitals(sys.cell, 5)});
    std::vector<BackflowSlaterDeterminant::Channel> channels(1);
    channels[0].source_first = 0;
    channels[0].source_last = 5;
    channels[0].truncation = 3;
    for (double v : sys.free_params[0]) channels[0].params.push_back(Parameter{"", v, true});
    TrialWavefunction real_psi;
    real_psi.add_component(std::make_unique<BackflowSlaterDeterminant>(
        std::move(blocks), std::move(channels), sys.cell));
    real_psi.evaluate_batch(r.ws);

    const std::size_t nw = configs.size();
    const double shift = a.ws.log_psi()[0] - r.ws.log_psi()[0];
    for (std::size_t w = 1; w < nw; ++w)
        CHECK(a.ws.log_psi()[w] - r.ws.log_psi()[w] == Catch::Approx(shift).margin(1e-11));
    for (std::size_t k = 0; k < 5; ++k)
        for (std::size_t d = 0; d < 2; ++d)
            for (std::size_t w = 0; w < nw; ++w) {
                const std::size_t j = (k * 2 + d) * nw + w;
                CHECK(a.ws.gradients()[j] == Catch::Approx(r.ws.gradients()[j]).margin(1e-10));
                CHECK(a.ws.laplacians()[j] == Catch::Approx(r.ws.laplacians()[j]).margin(1e-9));
                CHECK(a.ws.phase_gradients()[j] == Catch::Approx(theta[d]).margin(1e-10));
                CHECK(a.ws.phase_laplacians()[j] == Catch::Approx(0.0).margin(1e-9));
            }
}

TEST_CASE("complex backflow at -theta is the complex conjugate of theta", "[backflow][twist]") {
    const System plus = square_system({0.2, 0.13}), minus = square_system({-0.2, -0.13});
    Batch a(plus, square_configs()), b(minus, square_configs());
    TrialWavefunction pa = trial_of(plus), pb = trial_of(minus);
    pa.evaluate_batch(a.ws);
    pb.evaluate_batch(b.ws);
    for (std::size_t w = 0; w < 2; ++w)
        CHECK(a.ws.log_psi()[w] == Catch::Approx(b.ws.log_psi()[w]).epsilon(1e-13));
    for (std::size_t k = 0; k < a.ws.gradients().size(); ++k) {
        CHECK(a.ws.gradients()[k] == Catch::Approx(b.ws.gradients()[k]).margin(1e-12));
        CHECK(a.ws.laplacians()[k] == Catch::Approx(b.ws.laplacians()[k]).margin(1e-11));
        CHECK(a.ws.phase_gradients()[k] == Catch::Approx(-b.ws.phase_gradients()[k]).margin(1e-12));
        CHECK(a.ws.phase_laplacians()[k] == Catch::Approx(-b.ws.phase_laplacians()[k]).margin(1e-11));
    }
}

TEST_CASE("complex backflow is unchanged by a lattice translation of one particle",
          "[backflow][twist]") {
    // ψ(r_j + a) = e^{iθ·a} ψ(r): the modulus and every derivative of ln ψ are
    // the same; the WalkerSet would wrap such a particle, so evaluate an
    // unwrapped configuration through the oracle against the component.
    const System sys = hexagonal_system();
    const std::vector<double> base = hexagonal_configs()[0];
    std::vector<double> shifted = base;
    shifted[2 * 2] += 6.0;   // particle 2 by the first lattice vector
    const cplx l0 = oracle(sys, base), l1 = oracle(sys, shifted);
    CHECK(l1.real() == Catch::Approx(l0.real()).epsilon(1e-13));
    std::vector<cplx> g0, g1, lap0, lap1;
    fd_derivatives(sys, base, g0, lap0);
    fd_derivatives(sys, shifted, g1, lap1);
    for (std::size_t q = 0; q < g0.size(); ++q) {
        CHECK(std::abs(g0[q] - g1[q]) < 1e-7);
        CHECK(std::abs(lap0[q] - lap1[q]) < 2e-5);
    }
    // The component sees the wrapped configuration: same as the base.
    Batch b(sys, {shifted});
    TrialWavefunction psi = trial_of(sys);
    psi.evaluate_batch(b.ws);
    CHECK(b.ws.log_psi()[0] == Catch::Approx(l0.real()).epsilon(1e-12));
}

// ── Parameters ───────────────────────────────────────────────────────────────

TEST_CASE("complex backflow names its parameters as the real one and rebuild moves psi",
          "[backflow][twist][optimize]") {
    const System sys = hexagonal_system();
    auto det = make_component(sys);
    const auto params = det->parameters();
    REQUIRE(params.size() == 7);   // [c0, c2, L] like-spin, [c0, c1, c2, L] from the holes
    CHECK(params[0].name == "bf0_c0");
    CHECK(params[1].name == "bf0_c2");
    CHECK(params[2].name == "bf0_L");
    CHECK(params[4].name == "bf1_c1");
    CHECK(params[6].name == "bf1_L");
    CHECK(det->is_nodal());
    CHECK(det->is_complex());
    CHECK(det->supports_parameter_gradient_derivatives());

    Batch a(sys, hexagonal_configs()), b(sys, hexagonal_configs());
    a.ws.enable_phase();
    b.ws.enable_phase();
    det->evaluate_batch(a.ws);
    auto changed = make_component(sys);
    changed->parameters()[0].value = 0.4;
    changed->rebuild();
    changed->evaluate_batch(b.ws);
    CHECK(b.ws.log_psi()[0] != Catch::Approx(a.ws.log_psi()[0]));
    System moved = sys;
    moved.channels[0].c = {0.4, 3.0 * 0.4 / 2.5, -0.03};
    CHECK(b.ws.log_psi()[0] == Catch::Approx(oracle(moved, hexagonal_configs()[0]).real()).epsilon(1e-12));

    // Fixed parameters: nothing to write.
    std::vector<double> none;
    CHECK_NOTHROW(det->compute_complex_parameter_derivatives(a.ws, none, none));
}

TEST_CASE("complex backflow parameter derivatives: both parts of d ln psi/dp and of the "
          "curvature match finite differences",
          "[backflow][twist][optimize]") {
    // Like-spin c_0 (which also moves the derived c_1) and L, a source
    // channel's free c_1, two spin blocks, a hexagonal cell.
    const auto cases = all_cases();
    for (std::size_t c = 0; c < cases.size(); ++c) {
        INFO("system " << c);
        check_parameter_derivatives(cases[c].first, cases[c].second);
    }
}

TEST_CASE("complex backflow parameter derivatives hold over random configurations and across "
          "the cutoff",
          "[backflow][twist][optimize]") {
    std::mt19937_64 rng(20261109);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const System sys = hexagonal_system();
    std::vector<std::vector<double>> configs;
    for (int k = 0; k < 6; ++k) {
        std::vector<double> pos;
        for (std::size_t p = 0; p < sys.np; ++p) {
            const auto [x, y] = hex_point(unit(rng), unit(rng));
            pos.push_back(x);
            pos.push_back(y);
        }
        configs.push_back(pos);
    }
    // Particle 1 just inside and just outside the like-spin cutoff of particle 0.
    const double L = sys.channels[0].L;
    for (double f : {1.0 - 1e-3, 1.0 + 1e-3}) {
        std::vector<double> pos = configs[0];
        pos[2] = pos[0] + f * L * 0.6;
        pos[3] = pos[1] + f * L * 0.8;
        configs.push_back(pos);
    }
    check_parameter_derivatives(sys, configs);
}

TEST_CASE("complex backflow parameter derivatives after moves across wraps equal a fresh "
          "evaluation",
          "[backflow][twist][optimize]") {
    // As VMC leaves the walkers before a measurement: deferred commits, a
    // particle wrapped out of the cell (its committed x and row in another
    // gauge than a fresh evaluation's), a refresh.  O and the curvature are
    // gauge-free, so they equal those of a fresh evaluation at the wrapped
    // positions.
    for (const System& sys : {square_system(), two_block_system()}) {
        Batch b(sys, square_configs());
        TrialWavefunction psi = free_trial_of(sys);
        psi.set_proposal_laplacian(false);
        psi.evaluate_batch(b.ws);
        const std::vector<std::pair<std::size_t, std::vector<double>>> moves = {
            {0, {-0.4, 6.3}}, {sys.np - 1, {4.1, 5.0}}, {2, {5.8, -0.2}}};
        for (std::size_t mv = 0; mv < moves.size(); ++mv) {
            const auto& [p, to] = moves[mv];
            psi.refresh_particle_derivatives(b.ws, p);
            propose(b.ws, p, to);
            psi.ratio_batch(b.ws, p);
            std::array<bool, 2> accept = {true, mv != 1};
            commit(psi, b.ws, p, accept, true);
        }
        psi.refresh_derivatives(b.ws);
        const ParameterDerivatives got = parameter_derivatives_of(psi, b.ws);
        const ParameterDerivatives ref =
            fresh_parameter_derivatives(sys, {config_of(b.ws, 0), config_of(b.ws, 1)});
        for (std::size_t e = 0; e < got.re.size(); ++e) {
            CHECK(got.re[e] == Catch::Approx(ref.re[e]).margin(1e-11));
            CHECK(got.im[e] == Catch::Approx(ref.im[e]).margin(1e-11));
        }
        for (std::size_t k = 0; k < 4; ++k)
            for (std::size_t e = 0; e < got.planes[k].size(); ++e)
                CHECK(got.planes[k][e] == Catch::Approx(ref.planes[k][e]).margin(1e-9));
    }
}

TEST_CASE("at a twist that shifts a closed shell, the backflow parameter derivatives are the "
          "real Gamma determinant's with no phase part",
          "[backflow][twist][optimize]") {
    // ln ψ_θ = ln ψ_Γ + iθ·Σ_j r_j + const (see the exact relation above), and
    // θ·Σ r does not depend on the parameters: O_θ = O_Γ (real), ∂_p ∇ ln ψ_θ
    // = ∂_p ∇ ln ψ_Γ.  The reference is the independent real backflow code.
    const System sys = square_system({0.05, 0.03});
    std::vector<std::vector<double>> configs = square_configs();
    configs.push_back({2.9, 0.2, 0.4, 3.3, 5.0, 1.7, 3.6, 5.9, 1.1, 2.4});
    const ParameterDerivatives twisted_d = fresh_parameter_derivatives(sys, configs);

    std::vector<BackflowSlaterDeterminant::Block> blocks;
    blocks.push_back({0, 5, make_plane_wave_orbitals(sys.cell, 5)});
    std::vector<BackflowSlaterDeterminant::Channel> channels(1);
    channels[0].source_first = 0;
    channels[0].source_last = 5;
    channels[0].truncation = 3;
    for (double v : sys.free_params[0]) channels[0].params.push_back(Parameter{"", v, false});
    TrialWavefunction real_psi;
    real_psi.add_component(std::make_unique<BackflowSlaterDeterminant>(
        std::move(blocks), std::move(channels), sys.cell));
    Batch r(sys, configs);
    real_psi.evaluate_batch(r.ws);
    const std::size_t n = real_psi.total_optimizable_params(), nw = configs.size();
    REQUIRE(n == n_parameters(sys));
    const std::size_t field = sys.np * sys.dim * nw;
    std::vector<double> o(n * nw), dg(n * field), dl(n * field);
    std::vector<char> supported(n, 0);
    real_psi.compute_parameter_derivatives(r.ws, o);
    REQUIRE(real_psi.compute_parameter_gradient_derivatives(r.ws, dg, dl, supported));

    for (std::size_t e = 0; e < o.size(); ++e) {
        CHECK(twisted_d.re[e] == Catch::Approx(o[e]).margin(1e-10));
        CHECK(twisted_d.im[e] == Catch::Approx(0.0).margin(1e-10));
    }
    for (std::size_t e = 0; e < dg.size(); ++e) {
        CHECK(twisted_d.planes[0][e] == Catch::Approx(dg[e]).margin(1e-9));
        CHECK(twisted_d.planes[1][e] == Catch::Approx(dl[e]).margin(1e-8));
        CHECK(twisted_d.planes[2][e] == Catch::Approx(0.0).margin(1e-9));
        CHECK(twisted_d.planes[3][e] == Catch::Approx(0.0).margin(1e-8));
    }
}

TEST_CASE("complex backflow at -theta has the conjugate parameter derivatives",
          "[backflow][twist][optimize]") {
    const ParameterDerivatives plus = fresh_parameter_derivatives(square_system({0.2, 0.13}),
                                                                  square_configs());
    const ParameterDerivatives minus = fresh_parameter_derivatives(square_system({-0.2, -0.13}),
                                                                   square_configs());
    for (std::size_t e = 0; e < plus.re.size(); ++e) {
        CHECK(plus.re[e] == Catch::Approx(minus.re[e]).margin(1e-11));
        CHECK(plus.im[e] == Catch::Approx(-minus.im[e]).margin(1e-11));
    }
    for (std::size_t e = 0; e < plus.planes[0].size(); ++e) {
        CHECK(plus.planes[0][e] == Catch::Approx(minus.planes[0][e]).margin(1e-9));
        CHECK(plus.planes[1][e] == Catch::Approx(minus.planes[1][e]).margin(1e-8));
        CHECK(plus.planes[2][e] == Catch::Approx(-minus.planes[2][e]).margin(1e-9));
        CHECK(plus.planes[3][e] == Catch::Approx(-minus.planes[3][e]).margin(1e-8));
    }
}

TEST_CASE("complex backflow reports the curvature term only when it can supply it",
          "[backflow][twist][optimize]") {
    // C = 2: eta'' steps at the cutoff, and the step moves with L.
    System c2 = two_block_system();
    c2.channels[0].C = 2;
    TrialWavefunction psi = free_trial_of(c2);
    CHECK_FALSE(psi.components()[0]->supports_parameter_gradient_derivatives());
    Batch b(c2, square_configs());
    psi.evaluate_batch(b.ws);
    const std::size_t n = psi.total_optimizable_params(), field = c2.np * c2.dim * 2;
    std::vector<double> dg(n * field), dl(n * field), dpg(n * field), dpl(n * field);
    std::vector<char> supported(n, 1);
    CHECK_FALSE(psi.compute_complex_parameter_gradient_derivatives(b.ws, dg, dl, dpg, dpl,
                                                                   supported));
    for (char s : supported) CHECK(s == 0);
    // The real-only entry point gives the modulus parts of the complex one.
    const System sys = hexagonal_system();
    Batch h(sys, hexagonal_configs());
    TrialWavefunction full = free_trial_of(sys);
    full.evaluate_batch(h.ws);
    const ParameterDerivatives both = parameter_derivatives_of(full, h.ws);
    const std::size_t m = full.total_optimizable_params(), hf = sys.np * sys.dim * 2;
    std::vector<double> rg(m * hf), rl(m * hf), ro(m * 2);
    std::vector<char> ok(m, 0);
    REQUIRE(full.compute_parameter_gradient_derivatives(h.ws, rg, rl, ok));
    full.compute_parameter_derivatives(h.ws, ro);
    CHECK(rg == both.planes[0]);
    CHECK(rl == both.planes[1]);
    CHECK(ro == both.re);
}

TEST_CASE("complex backflow refuses what it cannot represent", "[backflow][twist]") {
    // A same-group channel over a group with both spins.
    {
        System sys = two_block_system();
        sys.channels = {{0, 3, 3, 2.0, {0.1, 0.0}}};
        sys.free_params = {{0.1, 2.0}};
        CHECK_THROWS_WITH(make_component(sys), ContainsSubstring("one spin"));
    }
    // A cutoff beyond the Wigner–Seitz inradius.
    {
        System sys = square_system();
        sys.free_params[0].back() = 3.5;
        CHECK_THROWS_WITH(make_component(sys), ContainsSubstring("inradius"));
    }
    // Complex orbitals without a Hessian (Fock–Darwin).
    {
        std::vector<ComplexBackflowSlaterDeterminant::Block> blocks;
        blocks.push_back({0, 1, std::make_unique<FockDarwinOrbitalSet>(std::vector<int>{0}, 1.0)});
        std::vector<ComplexBackflowSlaterDeterminant::Channel> ch(1);
        ch[0].source_first = 1;
        ch[0].source_last = 2;
        ch[0].params = {Parameter{"", 0.1, true}, Parameter{"", 1.0, true}};
        CHECK_THROWS_WITH(ComplexBackflowSlaterDeterminant(std::move(blocks), std::move(ch)),
                          ContainsSubstring("Hessian"));
    }
}

