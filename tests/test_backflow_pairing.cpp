// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Backflow pairing determinant (wavefunction/backflow_pairing_determinant.h):
// the component contract battery against an independent oracle that forms the
// quasi-coordinates, φ by direct cosine sums, the unpaired entries and the
// determinants by its own loops, with Richardson-extrapolated finite
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

#include "population/particle_group.h"
#include "population/simulation_cell.h"
#include "population/walker.h"
#include "population/walker_set.h"
#include "wavefunction/backflow_pairing_determinant.h"
#include "wavefunction/orbital_set.h"
#include "wavefunction/pair_orbital.h"
#include "wavefunction/pairing_determinant.h"
#include "wavefunction/trial_wavefunction.h"

using namespace alea;
using Catch::Matchers::ContainsSubstring;

namespace {

constexpr double kPi = 3.14159265358979323846;

ParticleGroup group(const char* name, std::size_t first, std::size_t last) {
    ParticleGroup g;
    g.name = name;
    g.charge = 0.0;
    g.first = first;
    g.last = last;
    g.count = last - first;
    return g;
}

struct OracleChannel {
    std::size_t target_first, target_last, source_first, source_last;
    int C;
    double L;
    std::vector<double> c;        // every coefficient (c_1 already derived)
    std::vector<double> free;     // the component's [c_0, (c_1), c_2.., L]
};

struct OracleBlock {
    ParticleGroup rows, columns;
    std::shared_ptr<OrbitalSet> unpaired;   // or null
};

struct System {
    std::size_t dim = 2, np = 0;
    SimulationCell cell = SimulationCell(2);
    std::vector<OracleBlock> blocks;
    std::vector<OracleChannel> channels;
    std::size_t shells = 0;
    int poly = -1;
    std::vector<double> p, c;   // orbital p_s; [c_0, c_2..c_M]
    double L = 1.0;             // orbital cutoff
};

PairOrbital orbital_of(const System& sys) {
    PairOrbital o(sys.cell, sys.shells, sys.poly);
    o.set(sys.p, sys.c, sys.L);
    return o;
}

double oracle_eta(const OracleChannel& ch, double r) {
    if (r >= ch.L) return 0.0;
    double P = 0.0;
    for (std::size_t k = 0; k < ch.c.size(); ++k) P += ch.c[k] * std::pow(r, static_cast<double>(k));
    return std::pow(1.0 - r / ch.L, ch.C) * P;
}

// log|ψ| and its sign at pos ([np*dim], AoS).
std::pair<double, double> oracle(const System& sys, const std::vector<double>& pos) {
    const std::size_t dim = sys.dim, np = sys.np;
    std::vector<double> x(pos);
    for (std::size_t P = 0; P < np; ++P)
        for (const OracleChannel& ch : sys.channels) {
            if (P < ch.target_first || P >= ch.target_last) continue;
            for (std::size_t j = ch.source_first; j < ch.source_last; ++j) {
                if (j == P) continue;
                double del[3] = {};
                for (std::size_t d = 0; d < dim; ++d) del[d] = pos[P * dim + d] - pos[j * dim + d];
                sys.cell.min_image(del);
                double r = 0.0;
                for (std::size_t d = 0; d < dim; ++d) r += del[d] * del[d];
                const double eta = oracle_eta(ch, std::sqrt(r));
                for (std::size_t d = 0; d < dim; ++d) x[P * dim + d] += eta * del[d];
            }
        }
    const PairOrbital orb = orbital_of(sys);
    const auto phi = [&](const double* xa, const double* xc) {
        double del[3] = {};
        for (std::size_t d = 0; d < dim; ++d) del[d] = xa[d] - xc[d];
        double v = 0.0;
        for (std::size_t i = 0; i < orb.n_k(); ++i) {
            double ph = 0.0;
            for (std::size_t d = 0; d < dim; ++d) ph += orb.k(i)[d] * del[d];
            v += orb.weight(i) * std::cos(ph);
        }
        if (orb.has_polynomial()) {
            sys.cell.min_image(del);
            double r2 = 0.0;
            for (std::size_t d = 0; d < dim; ++d) r2 += del[d] * del[d];
            double g = 0.0, dg = 0.0, d2g = 0.0;
            orb.radial(std::sqrt(r2), g, dg, d2g);
            v += g;
        }
        return v;
    };
    double log_abs = 0.0, sign = 1.0;
    for (const OracleBlock& b : sys.blocks) {
        const std::size_t nr = b.rows.size(), nc = b.columns.size(), n = std::max(nr, nc);
        std::vector<double> m(n * n, 0.0);
        for (std::size_t a = 0; a < nr; ++a)
            for (std::size_t c = 0; c < nc; ++c)
                m[a * n + c] = phi(&x[(b.rows.first + a) * dim], &x[(b.columns.first + c) * dim]);
        if (b.unpaired) {
            const std::size_t nu = b.unpaired->size();
            std::vector<double> v(nu), g(nu * dim), l(nu * dim);
            const bool extra_columns = nr > nc;
            for (std::size_t j = 0; j < (extra_columns ? nr : nc); ++j) {
                const std::size_t q = extra_columns ? b.rows.first + j : b.columns.first + j;
                std::vector<double> xq(x.begin() + static_cast<long>(q * dim),
                                       x.begin() + static_cast<long>((q + 1) * dim));
                b.unpaired->evaluate(xq, dim, v, g, l, 0, 1);
                for (std::size_t o = 0; o < nu; ++o) {
                    const std::size_t a = extra_columns ? j : nr + o;
                    const std::size_t c = extra_columns ? nc + o : j;
                    m[a * n + c] = v[o];
                }
            }
        }
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

std::vector<Parameter> orbital_params(const System& sys) {
    std::vector<Parameter> out;
    for (std::size_t s = 0; s < sys.p.size(); ++s)
        out.push_back(Parameter{"p" + std::to_string(s), sys.p[s], s == 0});
    for (double v : sys.c) out.push_back(Parameter{"c", v, false});
    if (sys.poly >= 1) out.push_back(Parameter{"L", sys.L, false, true});
    return out;
}

std::unique_ptr<BackflowPairingDeterminant> make_component(const System& sys) {
    std::vector<BackflowPairingDeterminant::Block> blocks;
    for (const OracleBlock& b : sys.blocks)
        blocks.push_back({b.rows, b.columns, b.unpaired ? b.unpaired->clone() : nullptr});
    std::vector<BackflowTransformation::Channel> channels;
    for (const OracleChannel& ch : sys.channels) {
        BackflowTransformation::Channel c;
        c.target_first = ch.target_first;
        c.target_last = ch.target_last;
        c.source_first = ch.source_first;
        c.source_last = ch.source_last;
        c.truncation = ch.C;
        for (double v : ch.free) c.params.push_back(Parameter{"", v, false});
        channels.push_back(std::move(c));
    }
    return std::make_unique<BackflowPairingDeterminant>(orbital_of(sys), std::move(blocks),
                                                        orbital_params(sys), std::move(channels),
                                                        sys.cell);
}

// Periodic 6×6 cell, one unbalanced block: three electrons [0,3) in rows, two
// holes [3,5) in columns, one unpaired plane wave in the extra column.  The
// pair orbital has two stars and a quadratic polynomial; every channel is on:
// like-spin e←e and h←h (η'(0) = 0), e←h and h←e.
System periodic_system() {
    System s;
    s.np = 5;
    s.cell = SimulationCell(2, std::vector<double>{6.0, 0.0, 0.0, 6.0});
    const double b = 2.0 * kPi / 6.0;
    s.blocks.push_back({group("e", 0, 3), group("h", 3, 5),
                        std::make_shared<PlaneWaveOrbitalSet>(2, std::vector<double>{b, 0.0},
                                                              std::vector<char>{0})});
    s.shells = 2;
    s.poly = 2;
    s.p = {1.0, 0.35};
    s.c = {0.8, -0.05};
    s.L = 2.6;
    const auto like = [](std::size_t f, std::size_t l, double c0, double c2, double L) {
        return OracleChannel{f, l, f, l, 3, L, {c0, 3.0 * c0 / L, c2}, {c0, c2, L}};
    };
    s.channels.push_back(like(0, 3, 0.2, -0.03, 2.5));
    s.channels.push_back({0, 3, 3, 5, 3, 2.8, {-0.15, 0.1, 0.02}, {-0.15, 0.1, 0.02, 2.8}});
    s.channels.push_back({3, 5, 0, 3, 3, 2.2, {0.1, -0.12}, {0.1, -0.12, 2.2}});
    s.channels.push_back(like(3, 5, -0.1, 0.04, 2.9));
    return s;
}

// Open cell, two balanced blocks of two flavours: (e↑ [0,2), h↓ [2,4)) and
// (e↓ [4,5), h↑ [5,6)); polynomial pair orbital only.  Channels across blocks:
// e↑ ← h↓, h↓ ← e↑, and e↓ ← e↑ (an opposite-spin source in the other block).
System open_system() {
    System s;
    s.np = 6;
    s.blocks.push_back({group("eu", 0, 2), group("hd", 2, 4), nullptr});
    s.blocks.push_back({group("ed", 4, 5), group("hu", 5, 6), nullptr});
    s.poly = 3;
    s.c = {1.0, -0.08, 0.01};
    s.L = 3.5;
    s.channels.push_back({0, 2, 2, 4, 3, 2.5, {-0.2, 0.15, 0.03}, {-0.2, 0.15, 0.03, 2.5}});
    s.channels.push_back({2, 4, 0, 2, 3, 2.0, {0.12, -0.05}, {0.12, -0.05, 2.0}});
    s.channels.push_back({4, 5, 0, 2, 3, 3.0, {0.1, 0.05}, {0.1, 0.05, 3.0}});
    return s;
}

std::vector<std::vector<double>> periodic_configs() {
    return {{0.5, 0.7, 2.1, 1.0, 3.9, 4.2, 1.0, 1.1, 2.5, 1.3},
            {1.0, 1.9, 2.6, 0.3, 4.4, 3.1, 1.3, 2.2, 2.0, 0.9}};
}
std::vector<std::vector<double>> open_configs() {
    return {{-0.6, 0.2, 0.7, -0.3, -0.4, 0.5, 0.9, 0.1, 0.2, 0.9, 0.4, 1.1},
            {-0.3, -0.4, 0.9, 0.4, -0.7, -0.1, 0.5, 0.6, -0.2, 1.0, 0.1, 0.8}};
}

struct Batch {
    std::vector<Walker> walkers;
    WalkerSet ws;
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

// Richardson-extrapolated central differences of the oracle (error O(h⁴)),
// with |D(h) − D(h/2)| as an error estimate (large only next to a node).
void fd_derivatives(const System& sys, const std::vector<double>& pos, std::vector<double>& grad,
                    std::vector<double>& lap, std::vector<double>& ge, std::vector<double>& le) {
    const double f0 = oracle(sys, pos).first;
    grad.assign(pos.size(), 0.0);
    lap.assign(pos.size(), 0.0);
    ge.assign(pos.size(), 0.0);
    le.assign(pos.size(), 0.0);
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
        ge[q] = std::abs(g[0] - g[1]);
        le[q] = std::abs(l[0] - l[1]);
    }
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
                CHECK(ws.gradients()[(k * dim + d) * nw + w] == Catch::Approx(g[q]).margin(1e-7 + ge[q]));
                CHECK(ws.laplacians()[(k * dim + d) * nw + w] == Catch::Approx(l[q]).margin(2e-5 + le[q]));
            }
    }
}

TrialWavefunction trial_of(const System& sys) {
    TrialWavefunction psi;
    psi.add_component(make_component(sys));
    return psi;
}

void propose(WalkerSet& ws, std::size_t p, const std::vector<double>& to) {
    const std::size_t nw = ws.num_walkers(), dim = ws.dim();
    std::vector<double> soa(dim * nw);
    for (std::size_t d = 0; d < dim; ++d)
        for (std::size_t w = 0; w < nw; ++w) soa[d * nw + w] = to[d];
    ws.propose_move_batch(p, soa);
}

using Case = std::pair<System, std::vector<std::vector<double>>>;
std::vector<Case> cases() { return {{periodic_system(), periodic_configs()}, {open_system(), open_configs()}}; }

}  // namespace

TEST_CASE("backflow pairing: log psi and derivatives match the oracle and finite differences",
          "[backflow][pairing]") {
    for (std::size_t c = 0; c < 2; ++c) {
        INFO("system " << c);
        const auto [sys, configs] = cases()[c];
        Batch b(sys, configs);
        TrialWavefunction psi = trial_of(sys);
        psi.evaluate_batch(b.ws);
        check_against_oracle(sys, b.ws);
    }
}

TEST_CASE("backflow pairing derivatives hold over random configurations", "[backflow][pairing]") {
    std::mt19937_64 rng(20261108);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (std::size_t c = 0; c < 2; ++c) {
        INFO("system " << c);
        const System sys = cases()[c].first;
        std::vector<std::vector<double>> configs;
        for (int k = 0; k < 40; ++k) {
            std::vector<double> pos;
            for (std::size_t q = 0; q < sys.np * 2; ++q)
                pos.push_back(c == 0 ? 6.0 * unit(rng) : 2.4 * unit(rng) - 1.2);
            configs.push_back(pos);
        }
        Batch b(sys, configs);
        TrialWavefunction psi = trial_of(sys);
        psi.evaluate_batch(b.ws);
        check_against_oracle(sys, b.ws);
    }
}

TEST_CASE("backflow pairing ratio_batch agrees with the oracle at the proposal",
          "[backflow][pairing]") {
    for (std::size_t c = 0; c < 2; ++c) {
        const auto [sys, configs] = cases()[c];
        // A row particle, a column particle, and the last particle.
        for (std::size_t p : {std::size_t{1}, std::size_t{3}, sys.np - 1}) {
            INFO("system " << c << " moves particle " << p);
            Batch b(sys, configs);
            TrialWavefunction psi = trial_of(sys);
            psi.evaluate_batch(b.ws);
            const std::vector<double> to = c == 0 ? std::vector<double>{3.3, 1.2}
                                                  : std::vector<double>{0.35, -0.6};
            propose(b.ws, p, to);
            psi.ratio_batch(b.ws, p);
            const std::size_t nw = b.ws.num_walkers();
            for (std::size_t w = 0; w < nw; ++w) {
                std::vector<double> before = config_of(b.ws, w), after = before;
                for (std::size_t d = 0; d < 2; ++d) after[p * 2 + d] = to[d];
                const auto [l0, s0] = oracle(sys, before);
                const auto [l1, s1] = oracle(sys, after);
                CHECK(b.ws.log_ratios()[w] == Catch::Approx(l1 - l0).margin(1e-11));
                CHECK(b.ws.phase_diffs()[w] == Catch::Approx(s0 * s1 < 0 ? kPi : 0.0));
                std::vector<double> g, l, ge, le;
                fd_derivatives(sys, after, g, l, ge, le);
                for (std::size_t d = 0; d < 2; ++d) {
                    CHECK(b.ws.new_grad()[d * nw + w] == Catch::Approx(g[p * 2 + d]).margin(1e-7 + ge[p * 2 + d]));
                    CHECK(b.ws.new_lap()[d * nw + w] == Catch::Approx(l[p * 2 + d]).margin(2e-5 + le[p * 2 + d]));
                }
            }
        }
    }
}

TEST_CASE("backflow pairing commits match a fresh evaluation, eager and deferred",
          "[backflow][pairing]") {
    for (std::size_t c = 0; c < 2; ++c) {
        const auto [sys, configs] = cases()[c];
        for (int mode = 0; mode < 3; ++mode) {   // eager, deferred, deferred without new_lap
            INFO("system " << c << " mode " << mode);
            const bool deferred = mode > 0;
            Batch b(sys, configs);
            TrialWavefunction psi = trial_of(sys);
            if (mode == 2) psi.set_proposal_laplacian(false);
            psi.evaluate_batch(b.ws);
            const std::vector<std::pair<std::size_t, std::vector<double>>> moves = {
                {0, c == 0 ? std::vector<double>{0.9, 1.4} : std::vector<double>{-0.5, 0.3}},
                {3, c == 0 ? std::vector<double>{4.1, 5.0} : std::vector<double>{0.6, -0.2}},
                {sys.np - 1, c == 0 ? std::vector<double>{2.2, 3.3} : std::vector<double>{0.3, 0.7}}};
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

TEST_CASE("backflow pairing: crowd ranges, permutation and clone", "[backflow][pairing]") {
    const System sys = periodic_system();
    Batch whole(sys, periodic_configs()), split(sys, periodic_configs());
    TrialWavefunction a = trial_of(sys), b = trial_of(sys);
    a.evaluate_batch(whole.ws);
    b.evaluate_batch(split.ws);
    propose(whole.ws, 3, {1.7, 4.4});
    propose(split.ws, 3, {1.7, 4.4});
    a.ratio_batch(whole.ws, 3);
    b.ratio_batch(split.ws, 3, 0, 1);
    b.ratio_batch(split.ws, 3, 1, 2);
    for (std::size_t w = 0; w < 2; ++w) {
        CHECK(split.ws.log_ratios()[w] == whole.ws.log_ratios()[w]);
        for (std::size_t d = 0; d < 2; ++d)
            CHECK(split.ws.new_lap()[d * 2 + w] == whole.ws.new_lap()[d * 2 + w]);
    }

    Batch perm(sys, periodic_configs());
    TrialWavefunction psi = trial_of(sys);
    psi.evaluate_batch(perm.ws);
    const std::vector<std::size_t> src = {1, 1, 0};
    perm.ws.apply_permutation(src);
    psi.permute_caches(src);
    propose(perm.ws, 0, {2.2, 0.4});
    psi.ratio_batch(perm.ws, 0);
    for (std::size_t w = 0; w < src.size(); ++w) {
        std::vector<double> before = config_of(perm.ws, w), after = before;
        after[0] = 2.2;
        after[1] = 0.4;
        CHECK(perm.ws.log_ratios()[w] ==
              Catch::Approx(oracle(sys, after).first - oracle(sys, before).first).margin(1e-11));
    }

    Batch c1(sys, periodic_configs()), c2(sys, periodic_configs());
    TrialWavefunction orig = trial_of(sys);
    TrialWavefunction copy = orig.clone();
    orig.evaluate_batch(c1.ws);
    copy.evaluate_batch(c2.ws);
    for (std::size_t q = 0; q < c1.ws.laplacians().size(); ++q)
        CHECK(c1.ws.laplacians()[q] == c2.ws.laplacians()[q]);
}

TEST_CASE("backflow pairing parameter derivatives match finite differences",
          "[backflow][pairing][optimize]") {
    for (std::size_t c = 0; c < 2; ++c) {
        const auto [sys, configs] = cases()[c];
        auto det = make_component(sys);
        Batch b(sys, configs);
        det->evaluate_batch(b.ws);
        const auto params = det->parameters();
        std::size_t n_opt = 0;
        for (const Parameter& p : params) n_opt += p.is_fixed ? 0 : 1;
        std::vector<double> derivs(n_opt * 2);
        det->compute_parameter_derivatives(b.ws, derivs);
        std::size_t k = 0;
        for (std::size_t q = 0; q < params.size(); ++q) {
            if (params[q].is_fixed) continue;
            INFO("system " << c << " parameter " << q << " " << params[q].name);
            const double h = 1e-6;
            auto at = [&](double shift, std::size_t w) {
                auto copy = make_component(sys);
                copy->parameters()[q].value += shift;
                copy->rebuild();
                Batch fresh(sys, configs);
                copy->evaluate_batch(fresh.ws);
                return fresh.ws.log_psi()[w];
            };
            for (std::size_t w = 0; w < 2; ++w)
                CHECK(derivs[k * 2 + w] == Catch::Approx((at(h, w) - at(-h, w)) / (2 * h)).margin(1e-7));
            ++k;
        }
    }
}

TEST_CASE("backflow pairing with eta = 0 is the plain pairing determinant", "[backflow][pairing]") {
    for (std::size_t c = 0; c < 2; ++c) {
        INFO("system " << c);
        auto [sys, configs] = cases()[c];
        for (OracleChannel& ch : sys.channels)
            for (std::size_t k = 0; k + 1 < ch.free.size(); ++k) ch.free[k] = 0.0;
        Batch a(sys, configs), b(sys, configs);
        TrialWavefunction bf = trial_of(sys);
        bf.evaluate_batch(a.ws);
        std::vector<PairingDeterminant::Block> blocks;
        for (const OracleBlock& blk : sys.blocks)
            blocks.push_back({blk.rows, blk.columns, blk.unpaired ? blk.unpaired->clone() : nullptr});
        TrialWavefunction plain;
        plain.add_component(std::make_unique<PairingDeterminant>(orbital_of(sys), std::move(blocks),
                                                                 orbital_params(sys)));
        plain.evaluate_batch(b.ws);
        for (std::size_t w = 0; w < 2; ++w)
            CHECK(a.ws.log_psi()[w] == Catch::Approx(b.ws.log_psi()[w]).epsilon(1e-12));
        for (std::size_t q = 0; q < a.ws.gradients().size(); ++q) {
            CHECK(a.ws.gradients()[q] == Catch::Approx(b.ws.gradients()[q]).margin(1e-10));
            CHECK(a.ws.laplacians()[q] == Catch::Approx(b.ws.laplacians()[q]).margin(1e-9));
        }
    }
}

TEST_CASE("backflow pairing refuses channels it cannot represent", "[backflow][pairing]") {
    System sys = open_system();
    sys.channels.push_back({0, 1, 2, 4, 3, 2.0, {0.1, 0.0}, {0.1, 0.0, 2.0}});   // e↑ again, from h↓
    CHECK_THROWS_WITH(make_component(sys), ContainsSubstring("share source"));
    System outside = open_system();
    outside.channels[0].target_last = 7;    // beyond every block
    outside.np = 7;
    CHECK_THROWS(make_component(outside));
}

// ── Curvature term: parameter derivatives of the gradient and Laplacian ──────

namespace {

// Open 3-D cell, one unbalanced block: three electrons [0,3) in rows, two
// holes [3,5) in columns and one unpaired plane wave (explicit k); polynomial
// pair orbital; e←e (like-spin), e←h and h←e channels.
System open3d_system() {
    System s;
    s.dim = 3;
    s.np = 5;
    s.cell = SimulationCell(3);
    s.blocks.push_back({group("e", 0, 3), group("h", 3, 5),
                        std::make_shared<PlaneWaveOrbitalSet>(3, std::vector<double>{0.7, -0.3, 0.5},
                                                              std::vector<char>{1})});
    s.poly = 2;
    s.c = {0.9, -0.06};
    s.L = 3.0;
    const double c0 = 0.18, L = 2.4;
    s.channels.push_back({0, 3, 0, 3, 3, L, {c0, 3.0 * c0 / L, -0.03}, {c0, -0.03, L}});
    s.channels.push_back({0, 3, 3, 5, 3, 2.6, {-0.12, 0.08, 0.02}, {-0.12, 0.08, 0.02, 2.6}});
    s.channels.push_back({3, 5, 0, 3, 3, 2.2, {0.1, -0.1}, {0.1, -0.1, 2.2}});
    return s;
}

std::vector<std::vector<double>> open3d_configs() {
    return {{-0.6, 0.2, 0.1, 0.8, -0.2, -0.3, 0.1, -0.7, 0.6, 0.3, 0.4, -0.2, -0.4, -0.3, 0.5},
            {-0.3, -0.4, 0.5, 0.6, 0.5, -0.1, 0.1, -0.9, 0.2, -0.7, 0.5, 0.3, 0.6, 0.2, -0.4}};
}

void unfix_all(BackflowPairingDeterminant& det) {
    for (Parameter& p : det.parameters()) p.is_fixed = false;
}

std::pair<std::vector<double>, std::vector<double>> shifted_derivatives(
    const System& sys, const std::vector<std::vector<double>>& configs, std::size_t q,
    double shift) {
    auto det = make_component(sys);
    unfix_all(*det);
    det->parameters()[q].value += shift;
    det->rebuild();
    Batch b(sys, configs);
    det->evaluate_batch(b.ws);
    return {std::vector<double>(det->grad_c().begin(), det->grad_c().end()),
            std::vector<double>(det->lap_c().begin(), det->lap_c().end())};
}

void curvature_of(const System& sys, const std::vector<std::vector<double>>& configs,
                  std::vector<double>& dg, std::vector<double>& dl) {
    auto det = make_component(sys);
    unfix_all(*det);
    REQUIRE(det->supports_parameter_gradient_derivatives());
    Batch b(sys, configs);
    det->evaluate_batch(b.ws);
    const std::size_t field = sys.np * sys.dim * configs.size();
    dg.assign(det->parameters().size() * field, 0.0);
    dl.assign(det->parameters().size() * field, 0.0);
    REQUIRE(det->compute_parameter_gradient_derivatives(b.ws, dg, dl));
}

// Against Richardson-extrapolated central differences in every parameter
// (|D(h) − D(h/2)| joins the margin), as in the backflow Slater test.
void check_curvature(const System& sys, const std::vector<std::vector<double>>& configs) {
    std::vector<double> dg, dl;
    curvature_of(sys, configs, dg, dl);
    auto det = make_component(sys);
    const std::size_t field = sys.np * sys.dim * configs.size();
    for (std::size_t q = 0; q < det->parameters().size(); ++q) {
        const Parameter p = det->parameters()[q];
        INFO("parameter " << q << " " << p.name);
        const double h = 1e-4 * std::max(1.0, std::abs(p.value));
        std::vector<double> g[2], l[2];
        for (int level = 0; level < 2; ++level) {
            const double hl = level == 0 ? h : 0.5 * h;
            const auto [gp, lp] = shifted_derivatives(sys, configs, q, hl);
            const auto [gm, lm] = shifted_derivatives(sys, configs, q, -hl);
            g[level].resize(field);
            l[level].resize(field);
            for (std::size_t e = 0; e < field; ++e) {
                g[level][e] = (gp[e] - gm[e]) / (2.0 * hl);
                l[level][e] = (lp[e] - lm[e]) / (2.0 * hl);
            }
        }
        for (std::size_t e = 0; e < field; ++e) {
            const double gf = (4.0 * g[1][e] - g[0][e]) / 3.0;
            const double lf = (4.0 * l[1][e] - l[0][e]) / 3.0;
            INFO("entry " << e << "  fd error estimate " << std::abs(l[0][e] - l[1][e]));
            CHECK(dg[q * field + e] == Catch::Approx(gf).margin(1e-7 + std::abs(g[0][e] - g[1][e])));
            CHECK(dl[q * field + e] == Catch::Approx(lf).margin(1e-6 + std::abs(l[0][e] - l[1][e])));
        }
    }
}

}  // namespace

TEST_CASE("pair orbital radial third derivative matches finite differences",
          "[backflow][pairing][optimize]") {
    for (const System& sys : {periodic_system(), open_system(), open3d_system()}) {
        const PairOrbital orb = orbital_of(sys);
        const double h = 1e-5;
        for (double r : {0.05, 0.6, 1.4, 0.98 * sys.L}) {
            INFO("r " << r);
            double g = 0.0, dg = 0.0, d2p = 0.0, d2m = 0.0;
            orb.radial(r + h, g, dg, d2p);
            orb.radial(r - h, g, dg, d2m);
            CHECK(orb.radial_third(r) == Catch::Approx((d2p - d2m) / (2 * h)).margin(1e-7));
        }
        CHECK(orb.radial_third(sys.L) == 0.0);
    }
}

TEST_CASE("backflow pairing curvature: parameter derivatives of the gradient and Laplacian "
          "match finite differences",
          "[backflow][pairing][optimize]") {
    // Stars, polynomial and cutoff of the pair orbital, every channel kind,
    // an unpaired plane wave, two blocks, and 3-D.
    SECTION("periodic unbalanced block") { check_curvature(periodic_system(), periodic_configs()); }
    SECTION("open two blocks") { check_curvature(open_system(), open_configs()); }
    SECTION("3-D") { check_curvature(open3d_system(), open3d_configs()); }
}

TEST_CASE("backflow pairing curvature over random configurations", "[backflow][pairing][optimize]") {
    std::mt19937_64 rng(20261108);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const System sys = periodic_system();
    std::vector<std::vector<double>> configs;
    for (int k = 0; k < 10; ++k) {
        std::vector<double> pos;
        for (std::size_t p = 0; p < sys.np * sys.dim; ++p) pos.push_back(6.0 * unit(rng));
        configs.push_back(pos);
    }
    check_curvature(sys, configs);
}

TEST_CASE("backflow pairing curvature with eta = 0 matches the plain pairing determinant",
          "[backflow][pairing][optimize]") {
    // PairingDeterminant computes the orbital parameters' derivatives its own
    // way; with eta = 0 the quasi-coordinates are the positions.
    for (std::size_t c = 0; c < 2; ++c) {
        INFO("system " << c);
        auto [sys, configs] = cases()[c];
        for (OracleChannel& ch : sys.channels)
            for (std::size_t k = 0; k + 1 < ch.free.size(); ++k) ch.free[k] = 0.0;
        std::vector<double> dg, dl;
        curvature_of(sys, configs, dg, dl);

        std::vector<PairingDeterminant::Block> blocks;
        for (const OracleBlock& blk : sys.blocks)
            blocks.push_back({blk.rows, blk.columns, blk.unpaired ? blk.unpaired->clone() : nullptr});
        PairingDeterminant plain(orbital_of(sys), std::move(blocks), orbital_params(sys));
        for (Parameter& p : plain.parameters()) p.is_fixed = false;
        REQUIRE(plain.supports_parameter_gradient_derivatives());
        Batch b(sys, configs);
        plain.evaluate_batch(b.ws);
        const std::size_t n_orb = plain.parameters().size();
        const std::size_t field = sys.np * sys.dim * configs.size();
        std::vector<double> pg(n_orb * field), pl(n_orb * field);
        REQUIRE(plain.compute_parameter_gradient_derivatives(b.ws, pg, pl));
        for (std::size_t e = 0; e < n_orb * field; ++e) {
            CHECK(dg[e] == Catch::Approx(pg[e]).margin(1e-10));
            CHECK(dl[e] == Catch::Approx(pl[e]).margin(1e-9));
        }
    }
}

TEST_CASE("backflow pairing curvature after moves equals that of a fresh evaluation",
          "[backflow][pairing][optimize]") {
    const System sys = periodic_system();
    Batch b(sys, periodic_configs());
    auto det = make_component(sys);
    unfix_all(*det);
    TrialWavefunction psi;
    psi.add_component(std::move(det));
    psi.set_proposal_laplacian(false);
    psi.evaluate_batch(b.ws);
    const std::vector<std::pair<std::size_t, std::vector<double>>> moves = {
        {0, {0.9, 1.4}}, {4, {4.1, 5.0}}, {2, {2.2, 3.1}}};
    for (const auto& [p, to] : moves) {
        psi.refresh_particle_derivatives(b.ws, p);
        propose(b.ws, p, to);
        psi.ratio_batch(b.ws, p);
        std::array<bool, 2> accept = {true, p != 4};
        psi.apply_move_batch_deferred(b.ws, p, accept);
        b.ws.apply_move(p, b.ws.new_pos(), b.ws.new_grad(), b.ws.new_lap(), accept);
    }
    psi.refresh_derivatives(b.ws);
    const std::size_t nq = psi.total_optimizable_params();
    const std::size_t field = sys.np * sys.dim * 2;
    std::vector<double> dg(nq * field), dl(nq * field);
    std::vector<char> supported(nq, 0);
    REQUIRE(psi.compute_parameter_gradient_derivatives(b.ws, dg, dl, supported));
    for (char s : supported) CHECK(s == 1);
    std::vector<double> fg, fl;
    curvature_of(sys, {config_of(b.ws, 0), config_of(b.ws, 1)}, fg, fl);
    REQUIRE(fg.size() == dg.size());
    for (std::size_t e = 0; e < dg.size(); ++e) {
        CHECK(dg[e] == Catch::Approx(fg[e]).margin(1e-10));
        CHECK(dl[e] == Catch::Approx(fl[e]).margin(1e-9));
    }
}

TEST_CASE("backflow pairing reports the curvature term only when it can supply it",
          "[backflow][pairing][optimize]") {
    CHECK(make_component(periodic_system())->supports_parameter_gradient_derivatives());
    System c2 = open_system();
    c2.channels[0].C = 2;   // eta'' steps at the cutoff, and the step moves with L
    auto det = make_component(c2);
    CHECK_FALSE(det->supports_parameter_gradient_derivatives());
    Batch b(c2, open_configs());
    det->evaluate_batch(b.ws);
    const std::size_t field = c2.np * c2.dim * 2;
    std::vector<double> dg(det->parameters().size() * field), dl(dg.size());
    CHECK_FALSE(det->compute_parameter_gradient_derivatives(b.ws, dg, dl));
}
