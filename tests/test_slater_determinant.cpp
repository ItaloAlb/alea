// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <span>
#include <vector>

#include "population/particle_group.h"
#include "population/walker.h"
#include "population/walker_set.h"
#include "population/walker_ensemble.h"
#include "wavefunction/orbital_set.h"
#include "wavefunction/slater_determinant.h"
#include "wavefunction/trial_wavefunction.h"
#include "hamiltonian/hamiltonian.h"
#include "hamiltonian/kinetic.h"
#include "qmc/dmc.h"

using namespace alea;

namespace {

constexpr double kPi = 3.14159265358979323846;

ParticleGroup grp(std::size_t first, std::size_t last, double charge,
                  double inv_m) {
    ParticleGroup g;
    g.name = "g";
    g.charge = charge;
    g.first = first;
    g.last = last;
    g.inv_mass = {inv_m, inv_m};
    return g;
}

// ── Independent oracle: Gaussian orbital values + small-N determinants ───────
// Evaluate all orbital values of `orb` at a single point r ([dim]) → [n].
std::vector<double> orb_vals(const OrbitalSet& orb, const double* r,
                             std::size_t dim) {
    const std::size_t n = orb.size();
    std::vector<double> pos(dim), val(n), grad(n * dim), lap(n * dim);
    for (std::size_t d = 0; d < dim; ++d) pos[d] = r[d];  // nw = 1 → [d]
    orb.evaluate(pos, dim, val, grad, lap, 0, 1);
    return val;  // val[i*1 + 0]
}

struct BlockSpec {
    std::size_t                   first, last;
    std::shared_ptr<OrbitalSet>   orb;  // shared: test oracle + component both use it
};

// det of one block at the given (flat, AoS == SoA when nw=1) positions [np*dim].
double block_det(const BlockSpec& bs, const std::vector<double>& fp,
                 std::size_t dim) {
    const std::size_t N = bs.last - bs.first;
    auto row = [&](std::size_t kr) {
        return orb_vals(*bs.orb, fp.data() + (bs.first + kr) * dim, dim);
    };
    if (N == 1) return row(0)[0];
    if (N == 2) {
        auto r0 = row(0), r1 = row(1);
        return r0[0] * r1[1] - r0[1] * r1[0];
    }
    return 0.0;  // tests use N ≤ 2
}

double log_psi_ref(const std::vector<BlockSpec>& specs,
                   const std::vector<double>& fp, std::size_t dim) {
    double total = 0.0;
    for (const BlockSpec& bs : specs)
        total += std::log(std::abs(block_det(bs, fp, dim)));
    return total;
}

// Build the component blocks from the same specs (clones the orbital sets).
std::unique_ptr<SlaterDeterminant> make_det(const std::vector<BlockSpec>& specs) {
    std::vector<SlaterDeterminant::Block> blocks;
    for (const BlockSpec& bs : specs)
        blocks.push_back(SlaterDeterminant::Block{bs.first, bs.last, bs.orb->clone()});
    return std::make_unique<SlaterDeterminant>(std::move(blocks));
}

// Standard test geometry: 4 particles in 2D.  Up block [0,2) with 2 Gaussians,
// down block [2,3) with 1 Gaussian, particle 3 a spectator (in no block).
std::vector<BlockSpec> standard_specs() {
    auto up = std::make_shared<GaussianOrbitalSet>(
        2, std::vector<double>{-1.0, 0.0, 1.0, 0.0}, std::vector<double>{0.6, 0.6});
    auto dn = std::make_shared<GaussianOrbitalSet>(
        2, std::vector<double>{0.0, 0.0}, std::vector<double>{0.5});
    return {BlockSpec{0, 2, up}, BlockSpec{2, 3, dn}};
}

std::vector<double> standard_positions() {
    // [p*dim + d] (== SoA for nw = 1)
    return {-0.8, 0.1,   0.9, -0.2,   0.1, 0.3,   2.0, 2.0};
}

}  // namespace

// ── Test 1: Gaussian orbital value / gradient / laplacian ────────────────────
TEST_CASE("GaussianOrbitalSet matches closed form and finite differences",
          "[slater]") {
    const std::size_t dim = 2, nw = 1;
    GaussianOrbitalSet orb(dim, {0.0, 0.0, 1.0, -1.0}, {0.5, 1.0});
    std::vector<double> pos = {0.3, -0.2};  // [d*nw + w]
    std::vector<double> val(2 * nw), grad(2 * dim * nw), lap(2 * dim * nw);
    orb.evaluate(pos, dim, val, grad, lap, 0, nw);

    // Closed form for orbital 0 (center origin, α=0.5).
    const double phi0 = std::exp(-0.5 * (0.3 * 0.3 + 0.2 * 0.2));
    CHECK(val[0] == Catch::Approx(phi0));
    CHECK(grad[0] == Catch::Approx(phi0 * (-2.0 * 0.5 * 0.3)));        // ∂x
    CHECK(grad[1 * nw] == Catch::Approx(phi0 * (-2.0 * 0.5 * -0.2))); // ∂y

    // Finite-difference cross-check of grad and lap for every orbital/dim.
    const double h = 1e-4;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t d = 0; d < dim; ++d) {
            auto phi_at = [&](double shift) {
                std::vector<double> p = pos;
                p[d] += shift;
                std::vector<double> v(2), g(2 * dim), l(2 * dim);
                orb.evaluate(p, dim, v, g, l, 0, 1);
                return v[i];
            };
            const double fd_g = (phi_at(h) - phi_at(-h)) / (2 * h);
            const double fd_l = (phi_at(h) - 2 * phi_at(0) + phi_at(-h)) / (h * h);
            CHECK(grad[(i * dim + d) * nw] == Catch::Approx(fd_g).margin(1e-6));
            CHECK(lap[(i * dim + d) * nw] == Catch::Approx(fd_l).margin(1e-4));
        }
}

// ── Test 2: evaluate_batch — log_psi, gradients, laplacians ───────────────────
TEST_CASE("SlaterDeterminant evaluate_batch matches det product and FD derivs",
          "[slater]") {
    const std::size_t dim = 2, nw = 1, np = 4;
    auto specs = standard_specs();
    auto fp = standard_positions();

    WalkerSet ws(nw, np, dim);
    auto pos = ws.positions();
    for (std::size_t k = 0; k < fp.size(); ++k) pos[k] = fp[k];  // SoA == flat (nw=1)
    ws.dist_tables().build_all(ws.positions());

    TrialWavefunction psi;
    psi.add_component(make_det(specs));
    psi.evaluate_batch(ws);

    CHECK(ws.log_psi()[0] == Catch::Approx(log_psi_ref(specs, fp, dim)));

    const double h = 1e-4;
    for (std::size_t p = 0; p < np; ++p)
        for (std::size_t d = 0; d < dim; ++d) {
            auto ref_at = [&](double shift) {
                std::vector<double> q = fp;
                q[p * dim + d] += shift;
                return log_psi_ref(specs, q, dim);
            };
            const double fd_g = (ref_at(h) - ref_at(-h)) / (2 * h);
            const double fd_l =
                (ref_at(h) - 2 * ref_at(0) + ref_at(-h)) / (h * h);
            CHECK(ws.gradients()[(p * dim + d) * nw] ==
                  Catch::Approx(fd_g).margin(1e-5));
            CHECK(ws.laplacians()[(p * dim + d) * nw] ==
                  Catch::Approx(fd_l).margin(1e-3));
        }
}

// ── Test 3: ratio_batch — magnitude, phase, spectator ────────────────────────
TEST_CASE("SlaterDeterminant ratio_batch gives det ratio and node-crossing phase",
          "[slater]") {
    const std::size_t dim = 2, nw = 1, np = 4;
    auto specs = standard_specs();
    auto fp = standard_positions();

    auto build = [&]() {
        auto ws = std::make_shared<WalkerSet>(nw, np, dim);
        auto pos = ws->positions();
        for (std::size_t k = 0; k < fp.size(); ++k) pos[k] = fp[k];
        ws->dist_tables().build_all(ws->positions());
        return ws;
    };

    // Ratio of det_up for moving particle p to new position (flat [dim]).
    auto ratio_ref = [&](std::size_t p, const std::vector<double>& newr) {
        const double before = log_psi_ref(specs, fp, dim);
        std::vector<double> q = fp;
        for (std::size_t d = 0; d < dim; ++d) q[p * dim + d] = newr[d];
        const double after = log_psi_ref(specs, q, dim);
        // signed ratio: det_after / det_before (product over blocks; only the
        // moved block changes, so this equals that block's ratio).
        double sb = 1.0, sa = 1.0;
        for (const BlockSpec& bs : specs) {
            if (p >= bs.first && p < bs.last) {
                sb = block_det(bs, fp, dim);
                sa = block_det(bs, q, dim);
            }
        }
        (void)before; (void)after;
        return sa / sb;
    };

    SECTION("same-sign move of an up electron") {
        auto ws = build();
        std::vector<double> newr = {-0.6, 0.0};  // small move of particle 0
        std::vector<double> np_soa = {newr[0], newr[1]};  // [d*nw+w], nw=1
        ws->propose_move_batch(0, np_soa);
        TrialWavefunction psi; psi.add_component(make_det(specs));
        psi.evaluate_batch(*ws);
        // evaluate_batch rebuilds caches; re-propose (positions unchanged).
        ws->propose_move_batch(0, np_soa);
        psi.ratio_batch(*ws, 0);
        const double R = ratio_ref(0, newr);
        REQUIRE(R > 0.0);
        CHECK(std::exp(ws->log_ratios()[0]) == Catch::Approx(std::abs(R)));
        CHECK(ws->phase_diffs()[0] == Catch::Approx(0.0));
    }

    SECTION("sign-flipping move of an up electron → phase π") {
        auto ws = build();
        // Move particle 0 across particle 1 (swap-like) to flip det_up sign.
        std::vector<double> newr = {1.6, -0.2};
        std::vector<double> np_soa = {newr[0], newr[1]};
        TrialWavefunction psi; psi.add_component(make_det(specs));
        psi.evaluate_batch(*ws);
        ws->propose_move_batch(0, np_soa);
        psi.ratio_batch(*ws, 0);
        const double R = ratio_ref(0, newr);
        REQUIRE(R < 0.0);
        CHECK(std::exp(ws->log_ratios()[0]) == Catch::Approx(std::abs(R)));
        CHECK(ws->phase_diffs()[0] == Catch::Approx(kPi));
    }

    SECTION("spectator particle (in no block) → ratio 1, no phase") {
        auto ws = build();
        std::vector<double> np_soa = {1.0, 1.0};  // move particle 3 (spectator)
        TrialWavefunction psi; psi.add_component(make_det(specs));
        psi.evaluate_batch(*ws);
        ws->propose_move_batch(3, np_soa);
        psi.ratio_batch(*ws, 3);
        CHECK(ws->log_ratios()[0] == Catch::Approx(0.0));
        CHECK(ws->phase_diffs()[0] == Catch::Approx(0.0));
    }
}

// ── Test 4: full move cycle == fresh evaluate of the proposed config ─────────
TEST_CASE("SlaterDeterminant full move cycle matches a fresh evaluate",
          "[slater]") {
    const std::size_t dim = 2, nw = 1, np = 4;
    auto specs = standard_specs();
    auto fp = standard_positions();
    const std::size_t pmove = 1;                 // move an up electron
    std::vector<double> newr = {0.4, 0.5};

    // Incremental path: evaluate → propose → ratio → accept → commit.
    WalkerSet cur(nw, np, dim);
    { auto pos = cur.positions(); for (std::size_t k = 0; k < fp.size(); ++k) pos[k] = fp[k]; }
    cur.dist_tables().build_all(cur.positions());
    TrialWavefunction psi; psi.add_component(make_det(specs));
    psi.evaluate_batch(cur);
    std::vector<double> np_soa = {newr[0], newr[1]};
    cur.propose_move_batch(pmove, np_soa);
    psi.ratio_batch(cur, pmove);
    std::array<bool, 1> accept = {true};
    psi.apply_move_batch(cur, pmove, accept);
    cur.apply_move(pmove, np_soa, cur.new_grad(), cur.new_lap(), accept);

    // Fresh evaluate of the proposed configuration.
    std::vector<double> fp2 = fp;
    for (std::size_t d = 0; d < dim; ++d) fp2[pmove * dim + d] = newr[d];
    WalkerSet prop(nw, np, dim);
    { auto pos = prop.positions(); for (std::size_t k = 0; k < fp2.size(); ++k) pos[k] = fp2[k]; }
    prop.dist_tables().build_all(prop.positions());
    TrialWavefunction ref; ref.add_component(make_det(specs));
    ref.evaluate_batch(prop);

    CHECK(cur.log_psi()[0] == Catch::Approx(prop.log_psi()[0]));
    for (std::size_t k = 0; k < cur.gradients().size(); ++k)
        CHECK(cur.gradients()[k] == Catch::Approx(prop.gradients()[k]).margin(1e-12));
    for (std::size_t k = 0; k < cur.laplacians().size(); ++k)
        CHECK(cur.laplacians()[k] == Catch::Approx(prop.laplacians()[k]).margin(1e-12));
}

// ── Test 5: clone independence + permute_caches carries D⁻¹ ───────────────────
TEST_CASE("SlaterDeterminant clone is an independent deep copy", "[slater]") {
    const std::size_t dim = 2, nw = 1, np = 4;
    auto specs = standard_specs();
    auto fp = standard_positions();

    TrialWavefunction psi; psi.add_component(make_det(specs));
    auto eval_logpsi = [&](TrialWavefunction& wf) {
        WalkerSet ws(nw, np, dim);
        auto pos = ws.positions();
        for (std::size_t k = 0; k < fp.size(); ++k) pos[k] = fp[k];
        ws.dist_tables().build_all(ws.positions());
        wf.evaluate_batch(ws);
        return ws.log_psi()[0];
    };
    const double before = eval_logpsi(psi);
    TrialWavefunction copy = psi.clone();
    CHECK(eval_logpsi(copy) == Catch::Approx(before));
    CHECK(eval_logpsi(psi) == Catch::Approx(before));  // original still works
}

TEST_CASE("SlaterDeterminant permute_caches carries the inverse to new walkers",
          "[slater]") {
    const std::size_t dim = 2, nw = 2, np = 4;
    auto specs = standard_specs();

    // Two distinct walkers.  apply_permutation gathers the Walker* handles, so
    // these must be real walkers loaded into the SoA (not raw position writes).
    std::vector<std::vector<double>> wpos = {
        {-0.8, 0.1, 0.9, -0.2, 0.1, 0.3, 2.0, 2.0},
        {-0.5, -0.3, 0.7, 0.4, -0.2, 0.1, 1.5, 1.0}};

    std::vector<Walker> walkers;
    walkers.reserve(nw);
    for (std::size_t w = 0; w < nw; ++w) {
        Walker wk(np, dim);
        wk.set_positions(wpos[w]);  // AoS [p*dim + d]
        walkers.push_back(std::move(wk));
    }
    WalkerSet ws(nw, np, dim);
    for (std::size_t w = 0; w < nw; ++w) ws.add_walker(walkers[w]);
    ws.load_all();  // gathers Walker AoS → SoA and builds distance tables
    TrialWavefunction psi; psi.add_component(make_det(specs));
    psi.evaluate_batch(ws);

    // Swap the two walkers; the inverse must travel so ratios stay correct.
    std::vector<std::size_t> src = {1, 0};
    ws.apply_permutation(src);
    psi.permute_caches(src);

    // Propose moving up-electron 0 in each (now-swapped) walker.
    std::vector<double> newr = {-0.4, 0.0};
    std::vector<double> np_soa(dim * nw);
    for (std::size_t d = 0; d < dim; ++d)
        for (std::size_t w = 0; w < nw; ++w) np_soa[d * nw + w] = newr[d];
    ws.propose_move_batch(0, np_soa);
    psi.ratio_batch(ws, 0);

    // Oracle ratio per (permuted) walker uses the permuted positions.
    for (std::size_t w = 0; w < nw; ++w) {
        std::vector<double> fp = wpos[src[w]];          // permuted positions
        const BlockSpec& up = specs[0];
        const double before = block_det(up, fp, dim);
        std::vector<double> q = fp;
        for (std::size_t d = 0; d < dim; ++d) q[0 * dim + d] = newr[d];
        const double after = block_det(up, q, dim);
        CHECK(std::exp(ws.log_ratios()[w]) ==
              Catch::Approx(std::abs(after / before)));
    }
}

// ── Test 6: end-to-end DMC fixed-node gate with a real determinant ───────────
TEST_CASE("DMC fixed-node gate keeps a determinant walker in its nodal pocket",
          "[slater][dmc]") {
    const std::size_t dim = 2, nw = 32;
    // One up block of 2 same-spin electrons; H = Kinetic only (finite E_L).
    auto up = std::make_shared<GaussianOrbitalSet>(
        dim, std::vector<double>{-1.0, 0.0, 1.0, 0.0}, std::vector<double>{0.7, 0.7});
    std::vector<BlockSpec> specs = {BlockSpec{0, 2, up}};

    WalkerEnsemble ens(nw, {grp(0, 2, -1.0, 1.0)}, dim);
    std::vector<double> start = {-1.0, 0.0, 1.0, 0.0};  // e0 at c0, e1 at c1 → det>0
    for (auto& w : ens.walkers()) w.set_positions(start);

    TrialWavefunction psi; psi.add_component(make_det(specs));
    Hamiltonian ham;
    ham.add_component(std::make_unique<Kinetic>(
        std::vector<ParticleGroup>{grp(0, 2, -1.0, 1.0)}));

    DMCConfig cfg;
    cfg.tau = 0.05;
    cfg.n_equilibration_blocks = 0;
    cfg.n_accumulation_blocks = 15;
    cfg.n_steps = 8;
    cfg.n_substeps = 2;
    cfg.seed = 7;
    cfg.n_threads = 1;

    // Sign of det_up for one walker from its current positions.
    auto det_sign = [&](const WalkerSet& p, std::size_t w) {
        const std::span<const double> q = p.positions();
        const std::size_t n = p.num_walkers();
        double r0[2] = {q[(0 * dim + 0) * n + w], q[(0 * dim + 1) * n + w]};
        double r1[2] = {q[(1 * dim + 0) * n + w], q[(1 * dim + 1) * n + w]};
        auto v0 = orb_vals(*up, r0, dim);
        auto v1 = orb_vals(*up, r1, dim);
        return v0[0] * v1[1] - v0[1] * v1[0];
    };

    DMC dmc(cfg, ens, psi, ham);
    bool sign_flipped = false;
    int  measured = 0;
    dmc.set_measure_hook([&](const WalkerSet& p, const TrialWavefunction&) {
        ++measured;
        for (std::size_t w = 0; w < p.num_walkers(); ++w)
            if (det_sign(p, w) <= 0.0) sign_flipped = true;
    });
    dmc.run();

    CHECK(measured > 0);
    CHECK_FALSE(sign_flipped);          // gate confined every walker to det > 0
    CHECK(std::isfinite(dmc.energy())); // run stayed numerically sane
}

// ── Periodic orbital sets (Wigner-crystallisation milestone) ────────────────
//
// Two shapes of orbital are commensurate with a periodic cell, one per phase of
// the fluid/crystal crossing, and each has a geometric property that can be
// checked without running any Monte Carlo at all.

TEST_CASE("reciprocal shells close at the counts the lattice allows",
          "[orbitals][plane_wave]") {
    // Hexagonal cell: the reciprocal lattice is triangular too, so the shell
    // degeneracies are 1, 6, 6, 6, 12, 6, ... and the closed-shell particle
    // counts are the running sums 1, 7, 13, 19, 31, 37.  These are exactly the
    // "magic numbers" a 2-D HEG calculation is run at.
    const double a = 5.0;
    const std::array<double, 4> hex{a, 0.0, a * 0.5, a * std::sqrt(3.0) / 2.0};
    const alea::SimulationCell hcell(2, std::span<const double>(hex));
    const auto ht = alea::reciprocal_shell_table(hcell, 70);
    std::vector<std::size_t> hfirst(ht.closed_counts.begin(),
                                    ht.closed_counts.begin() + 9);
    // 61 is the production size, so it is pinned here; 55 closes a shell but is
    // NOT Loeschian, so it cannot host the crystal half of a phase comparison.
    CHECK(hfirst == std::vector<std::size_t>{1, 7, 13, 19, 31, 37, 43, 55, 61});

    // Square cell: degeneracies 1, 4, 4, 4, 8, ... → 1, 5, 9, 13, 21.
    const std::array<double, 4> sq{a, 0.0, 0.0, a};
    const alea::SimulationCell scell(2, std::span<const double>(sq));
    const auto st = alea::reciprocal_shell_table(scell, 25);
    std::vector<std::size_t> sfirst(st.closed_counts.begin(),
                                    st.closed_counts.begin() + 5);
    CHECK(sfirst == std::vector<std::size_t>{1, 5, 9, 13, 21});

    // An open shell is refused, not silently filled: the state would be one
    // arbitrary pick among degenerate k-points, so its energy is not a property
    // of the system.  The message must name the counts that do work.
    REQUIRE_THROWS_WITH(alea::make_plane_wave_orbitals(hcell, 4),
                        Catch::Matchers::ContainsSubstring("1, 7, 13"));
    REQUIRE_NOTHROW(alea::make_plane_wave_orbitals(hcell, 7));
}

TEST_CASE("plane-wave orbitals are periodic under a lattice translation",
          "[orbitals][plane_wave]") {
    // This is the property that the hand-placed Gaussians lacked and that got
    // `periodic + slater` refused: k·a_j = 2π·integer makes every orbital, and
    // hence the determinant, invariant when a walker is wrapped.
    const double a = 4.0;
    const std::array<double, 4> hex{a, 0.0, a * 0.5, a * std::sqrt(3.0) / 2.0};
    const alea::SimulationCell cell(2, std::span<const double>(hex));
    const auto pw = alea::make_plane_wave_orbitals(cell, 13);
    const std::size_t n = pw->size();

    const std::size_t nw = 2;
    std::vector<double> pos{0.37, 0.37 + hex[0] + 2.0 * hex[2],
                            0.91, 0.91 + hex[1] + 2.0 * hex[3]};
    std::vector<double> val(n * nw), grad(n * 2 * nw), lap(n * 2 * nw);
    pw->evaluate(pos, 2, val, grad, lap, 0, nw);

    for (std::size_t i = 0; i < n; ++i) {
        INFO("orbital " << i);
        CHECK(val[i * nw + 0] == Catch::Approx(val[i * nw + 1]).margin(1e-12));
        for (std::size_t d = 0; d < 2; ++d) {
            CHECK(grad[(i * 2 + d) * nw + 0] ==
                  Catch::Approx(grad[(i * 2 + d) * nw + 1]).margin(1e-12));
            CHECK(lap[(i * 2 + d) * nw + 0] ==
                  Catch::Approx(lap[(i * 2 + d) * nw + 1]).margin(1e-12));
        }
    }

    // Numerical check of the analytic derivatives on the same orbital set.
    const double h = 1e-5;
    for (std::size_t d = 0; d < 2; ++d) {
        std::vector<double> pp{pos[0], pos[1], pos[2], pos[3]};
        std::vector<double> pm = pp;
        pp[d * nw] += h;
        pm[d * nw] -= h;
        std::vector<double> vp(n * nw), vm(n * nw), g(n * 2 * nw), l(n * 2 * nw);
        pw->evaluate(pp, 2, vp, g, l, 0, nw);
        pw->evaluate(pm, 2, vm, g, l, 0, nw);
        for (std::size_t i = 0; i < n; ++i) {
            const double fd = (vp[i * nw] - vm[i * nw]) / (2.0 * h);
            const double fd2 =
                (vp[i * nw] - 2.0 * val[i * nw] + vm[i * nw]) / (h * h);
            INFO("orbital " << i << " dim " << d);
            CHECK(grad[(i * 2 + d) * nw] == Catch::Approx(fd).margin(1e-6));
            CHECK(lap[(i * 2 + d) * nw] == Catch::Approx(fd2).margin(1e-4));
        }
    }
}

TEST_CASE("commensurate triangular sites tile the cell exactly",
          "[orbitals][crystal]") {
    // N = 36 = 6² is the aligned case (q = 0); N = 37 = 4²+4·3+3² is a ROTATED
    // commensurate crystal, which is the interesting one — it is also a closed
    // reciprocal shell, so the same N serves the fluid and the crystal.
    for (const std::size_t N : {std::size_t{36}, std::size_t{37}}) {
        const double rs = 1.0;
        const double A  = static_cast<double>(N) * kPi * rs * rs;
        const double a  = std::sqrt(2.0 * A / std::sqrt(3.0));
        const std::array<double, 4> hex{a, 0.0, a * 0.5, a * std::sqrt(3.0) / 2.0};
        const alea::SimulationCell cell(2, std::span<const double>(hex));
        const std::vector<double> sites =
            alea::commensurate_triangular_sites(cell, N);
        REQUIRE(sites.size() == N * 2);

        // Every site has exactly 6 neighbours at the triangular spacing
        // d = sqrt(2A/(N√3)), and none closer — the signature of a perfect
        // triangular lattice, measured through the minimum image so the count
        // includes neighbours across the cell boundary.
        const double d = std::sqrt(2.0 * A / (static_cast<double>(N) * std::sqrt(3.0)));
        for (std::size_t i = 0; i < N; ++i) {
            int n_near = 0;
            double r_min = 1e300;
            for (std::size_t j = 0; j < N; ++j) {
                if (i == j) continue;
                double del[3] = {sites[i * 2] - sites[j * 2],
                                 sites[i * 2 + 1] - sites[j * 2 + 1], 0.0};
                const double r = std::sqrt(cell.min_image_dist2(del));
                r_min = std::min(r_min, r);
                if (std::abs(r - d) < 1e-9 * d) ++n_near;
            }
            INFO("N = " << N << ", site " << i << ", d = " << d
                        << ", r_min = " << r_min);
            CHECK(n_near == 6);
            CHECK(r_min == Catch::Approx(d).epsilon(1e-9));
        }
    }

    // A count that is not p²+pq+q² has no commensurate triangular crystal, and
    // pretending otherwise would put two particles on one site.
    const std::array<double, 4> hex{5.0, 0.0, 2.5, 5.0 * std::sqrt(3.0) / 2.0};
    const alea::SimulationCell cell(2, std::span<const double>(hex));
    REQUIRE_THROWS_WITH(alea::commensurate_triangular_sites(cell, 5),
                        Catch::Matchers::ContainsSubstring("p²+pq+q²"));
    // ... and a square cell cannot host one at all.
    const std::array<double, 4> sq{5.0, 0.0, 0.0, 5.0};
    const alea::SimulationCell scell(2, std::span<const double>(sq));
    REQUIRE_THROWS_WITH(alea::commensurate_triangular_sites(scell, 36),
                        Catch::Matchers::ContainsSubstring("60°"));
}

TEST_CASE("Gaussian crystal orbitals fold to the minimum image",
          "[orbitals][crystal]") {
    // Without the fold a site near the cell edge loses half its Gaussian to the
    // wrap; with it, a walker one lattice vector away is the same point.
    const double a = 10.0;
    const std::array<double, 4> hex{a, 0.0, a * 0.5, a * std::sqrt(3.0) / 2.0};
    const alea::SimulationCell cell(2, std::span<const double>(hex));
    const std::vector<double> centers{4.6, 0.2};   // deliberately near the edge
    const std::vector<double> alphas{0.35};
    const alea::GaussianOrbitalSet folded(2, centers, alphas, cell);
    const alea::GaussianOrbitalSet raw(2, centers, alphas);

    const std::size_t nw = 2;
    // Walker 1 is walker 0 shifted by a full cell vector: the same physical point.
    std::vector<double> pos{-4.9, -4.9 + hex[0], 0.3, 0.3 + hex[1]};
    std::vector<double> v(nw), g(2 * nw), l(2 * nw);
    folded.evaluate(pos, 2, v, g, l, 0, nw);
    CHECK(v[0] == Catch::Approx(v[1]).epsilon(1e-12));
    CHECK(g[0 * nw + 0] == Catch::Approx(g[0 * nw + 1]).epsilon(1e-12));

    // The unfolded set is the bug this guards against: the same two points give
    // wildly different amplitudes, which is a discontinuity in ψ.
    std::vector<double> vr(nw), gr(2 * nw), lr(2 * nw);
    raw.evaluate(pos, 2, vr, gr, lr, 0, nw);
    CHECK(vr[0] != Catch::Approx(vr[1]).epsilon(1e-6));

    // clone() must carry the cell, or a per-thread copy would silently drop the
    // fold and only the threaded run would be wrong.
    const auto cl = folded.clone();
    std::vector<double> vc(nw), gc(2 * nw), lc(2 * nw);
    cl->evaluate(pos, 2, vc, gc, lc, 0, nw);
    CHECK(vc[0] == Catch::Approx(vc[1]).epsilon(1e-12));
}

TEST_CASE("Gaussian crystal orbitals on a walker range match the scalar fold",
          "[orbitals][crystal]") {
    // evaluate() folds a whole walker range at once; each walker must get the
    // closed form at the scalar min_image of r − c, and walkers outside the
    // range must be left alone.  A skewed cell takes the general Bravais fold.
    const std::array<double, 4> skew{12.0, 3.0, -4.0, 9.0};
    const alea::SimulationCell cell(2, std::span<const double>(skew));
    const std::vector<double> centers{4.6, 0.2, -3.0, 5.5, 0.0, -4.4};
    const std::vector<double> alphas{0.35, 0.35, 0.35};
    const alea::GaussianOrbitalSet orb(2, centers, alphas, cell);

    constexpr std::size_t nw = 11, w0 = 2, w1 = 9, n = 3;
    std::vector<double> pos(2 * nw);   // [d * nw + w]
    for (std::size_t w = 0; w < nw; ++w) {
        pos[w]      = -14.0 + 2.9 * w;
        pos[nw + w] = 8.0 - 1.7 * w;
    }
    std::vector<double> v(n * nw, -7.0), g(n * 2 * nw, -7.0), l(n * 2 * nw, -7.0);
    orb.evaluate(pos, 2, v, g, l, w0, w1);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t w = 0; w < nw; ++w) {
            if (w < w0 || w >= w1) {
                CHECK(v[i * nw + w] == -7.0);
                CHECK(g[(i * 2 + 1) * nw + w] == -7.0);
                CHECK(l[(i * 2 + 0) * nw + w] == -7.0);
                continue;
            }
            double del[2] = {pos[w] - centers[2 * i], pos[nw + w] - centers[2 * i + 1]};
            cell.min_image(del);
            const double a = alphas[i];
            const double phi = std::exp(-a * (del[0] * del[0] + del[1] * del[1]));
            CHECK(v[i * nw + w] == Catch::Approx(phi).epsilon(1e-14));
            for (std::size_t d = 0; d < 2; ++d) {
                CHECK(g[(i * 2 + d) * nw + w] ==
                      Catch::Approx(-2.0 * a * del[d] * phi).epsilon(1e-14).margin(1e-300));
                CHECK(l[(i * 2 + d) * nw + w] ==
                      Catch::Approx((4.0 * a * a * del[d] * del[d] - 2.0 * a) * phi)
                          .epsilon(1e-14).margin(1e-300));
            }
        }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Nodal parameters: the Gaussian width α exposed to the linear optimiser
//
//  ψ = D·e^J with J real, so the nodes of ψ are exactly the nodes of D and no
//  Jastrow parameter can move the fixed-node energy.  For a Gaussian-orbital
//  crystal the ONLY parameter that can is α, so it has to reach the optimiser.
//  The derivative is analytic:
//      ∂ln|det D|/∂α = Tr(D⁻¹ ∂D/∂α),   ∂D_{c,a}/∂α = −|r_c − c_a|² φ_a(r_c)
//  and every test below pins one link of that chain against an independent
//  finite difference of the determinant itself.
// ═══════════════════════════════════════════════════════════════════════════

namespace {

// Same geometry as standard_specs(), with the two block widths dialled.
std::vector<BlockSpec> specs_with_alphas(double a_up, double a_dn) {
    auto up = std::make_shared<GaussianOrbitalSet>(
        2, std::vector<double>{-1.0, 0.0, 1.0, 0.0},
        std::vector<double>{a_up, a_up});
    auto dn = std::make_shared<GaussianOrbitalSet>(
        2, std::vector<double>{0.0, 0.0}, std::vector<double>{a_dn});
    return {BlockSpec{0, 2, up}, BlockSpec{2, 3, dn}};
}

// Three distinct walker configurations, flat [p*dim + d] each.
const std::vector<std::vector<double>>& walker_configs() {
    static const std::vector<std::vector<double>> c = {
        {-0.8,  0.1,  0.9, -0.2,  0.1,  0.3,  2.0, 2.0},
        {-0.5,  0.4,  1.2,  0.3, -0.2,  0.7,  1.0, 1.5},
        {-1.1, -0.3,  0.7,  0.5,  0.4, -0.6,  0.5, 2.5}};
    return c;
}

// Load walker_configs() into a WalkerSet's SoA positions.
void load_configs(WalkerSet& ws) {
    const std::size_t nw = ws.num_walkers(), dim = ws.dim();
    auto pos = ws.positions();
    for (std::size_t w = 0; w < nw; ++w)
        for (std::size_t p = 0; p < ws.n_particles(); ++p)
            for (std::size_t d = 0; d < dim; ++d)
                pos[(p * dim + d) * nw + w] =
                    walker_configs()[w % walker_configs().size()][p * dim + d];
    ws.dist_tables().build_all(ws.positions());
}

}  // namespace

// ── Test A: the orbital set owns the width ──────────────────────────────────
TEST_CASE("GaussianOrbitalSet exposes a uniform width as one parameter",
          "[orbitals][optimize]") {
    const GaussianOrbitalSet uniform(2, {-1.0, 0.0, 1.0, 0.0}, {0.6, 0.6});
    REQUIRE(uniform.n_parameters() == 1);
    const Parameter p = uniform.parameter(0);
    CHECK(p.name == "alpha");
    CHECK(p.value == Catch::Approx(0.6));
    // A width is positive by construction: optimise it as α = e^θ so the
    // nonlinear step can never drive the Gaussian through zero into a
    // non-normalisable e^{+αr²}.
    CHECK(p.sign_locked);

    // Widths that differ per orbital are not ONE width: there is no single
    // parameter to expose, and inventing a shared one would silently retie
    // orbitals the config deliberately separated.
    const GaussianOrbitalSet mixed(2, {-1.0, 0.0, 1.0, 0.0}, {0.6, 0.4});
    CHECK(mixed.n_parameters() == 0);

    // Plane waves carry no parameter at all — their k come from the cell.
    const PlaneWaveOrbitalSet pw(2, {0.0, 0.0}, {0});
    CHECK(pw.n_parameters() == 0);
}

// ── Test B: ∂φ_i/∂α against a finite difference of φ ────────────────────────
TEST_CASE("GaussianOrbitalSet width derivative matches finite differences",
          "[orbitals][optimize]") {
    const std::size_t dim = 2, nw = 2, n = 2;
    const double alpha = 0.6;
    const std::vector<double> centers{-1.0, 0.0, 1.0, 0.0};
    const GaussianOrbitalSet orb(dim, centers, {alpha, alpha});

    std::vector<double> pos{0.3, -0.7, -0.2, 0.9};  // [d*nw + w]
    std::vector<double> dval(1 * n * nw);
    orb.evaluate_parameter_derivatives(pos, dim, dval, {}, {}, 0, nw);

    const double h = 1e-6;
    auto val_at = [&](double a, std::size_t i, std::size_t w) {
        const GaussianOrbitalSet o(dim, centers, {a, a});
        std::vector<double> v(n * nw), g(n * dim * nw), l(n * dim * nw);
        o.evaluate(pos, dim, v, g, l, 0, nw);
        return v[i * nw + w];
    };
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t w = 0; w < nw; ++w) {
            const double fd =
                (val_at(alpha + h, i, w) - val_at(alpha - h, i, w)) / (2 * h);
            CHECK(dval[i * nw + w] == Catch::Approx(fd).margin(1e-7));
        }
}

// ── Test C: the determinant collects one parameter per Gaussian block ───────
TEST_CASE("SlaterDeterminant exposes one width per Gaussian block",
          "[slater][optimize]") {
    auto det = make_det(specs_with_alphas(0.6, 0.5));
    const auto params = det->parameters();
    REQUIRE(params.size() == 2);
    CHECK(params[0].value == Catch::Approx(0.6));
    CHECK(params[1].value == Catch::Approx(0.5));
    // Names carry the block so a two-block crystal's two sublattice widths are
    // distinguishable in the optimiser log.
    CHECK(params[0].name == "b0_alpha");
    CHECK(params[1].name == "b1_alpha");

    // Fixed by default: exposing the parameter must not silently start
    // optimising every existing config that has a determinant in it.
    CHECK(det->num_optimizable_params() == 0);
    for (Parameter& p : det->parameters()) p.is_fixed = false;
    CHECK(det->num_optimizable_params() == 2);
}

TEST_CASE("plane-wave determinant exposes no parameters", "[slater][optimize]") {
    const std::array<double, 4> hex{5.0, 0.0, 2.5, 5.0 * std::sqrt(3.0) / 2.0};
    const alea::SimulationCell cell(2, std::span<const double>(hex));
    std::vector<SlaterDeterminant::Block> blocks;
    blocks.push_back({0, 7, alea::make_plane_wave_orbitals(cell, 7)});
    SlaterDeterminant det(std::move(blocks));
    CHECK(det.parameters().empty());
    CHECK(det.num_optimizable_params() == 0);
}

// ── Test D: rebuild() pushes the new width back into the orbitals ───────────
TEST_CASE("SlaterDeterminant rebuild propagates the width into the orbitals",
          "[slater][optimize]") {
    const std::size_t dim = 2, nw = 3, np = 4;

    auto moved = make_det(specs_with_alphas(0.6, 0.5));
    SlaterDeterminant* det = moved.get();
    TrialWavefunction psi;
    psi.add_component(std::move(moved));

    det->parameters()[0].value = 0.9;
    det->parameters()[1].value = 0.4;
    det->rebuild();

    WalkerSet ws(nw, np, dim);
    load_configs(ws);
    psi.evaluate_batch(ws);

    // Identical to a determinant built from scratch at the new widths.
    TrialWavefunction ref;
    ref.add_component(make_det(specs_with_alphas(0.9, 0.4)));
    WalkerSet rws(nw, np, dim);
    load_configs(rws);
    ref.evaluate_batch(rws);

    for (std::size_t w = 0; w < nw; ++w)
        CHECK(ws.log_psi()[w] == Catch::Approx(rws.log_psi()[w]).epsilon(1e-12));
}

// ── Test E: ∂ln|ψ_det|/∂α against a finite difference of the determinant ────
TEST_CASE("SlaterDeterminant width derivatives match finite differences",
          "[slater][optimize]") {
    const std::size_t dim = 2, np = 4;
    const std::size_t nw = GENERATE(3, 64, 73);
    const double a_up = 0.6, a_dn = 0.5;

    auto moved = make_det(specs_with_alphas(a_up, a_dn));
    SlaterDeterminant* det = moved.get();
    for (Parameter& p : det->parameters()) p.is_fixed = false;
    TrialWavefunction psi;
    psi.add_component(std::move(moved));
    REQUIRE(psi.total_optimizable_params() == 2);

    WalkerSet ws(nw, np, dim);
    load_configs(ws);
    psi.evaluate_batch(ws);

    std::vector<double> derivs(2 * nw, 0.0);
    psi.compute_parameter_derivatives(ws, derivs);

    const double h = 1e-5;
    for (std::size_t w = 0; w < nw; ++w) {
        const std::vector<double>& fp = walker_configs()[w % walker_configs().size()];
        const double fd_up =
            (log_psi_ref(specs_with_alphas(a_up + h, a_dn), fp, dim) -
             log_psi_ref(specs_with_alphas(a_up - h, a_dn), fp, dim)) / (2 * h);
        const double fd_dn =
            (log_psi_ref(specs_with_alphas(a_up, a_dn + h), fp, dim) -
             log_psi_ref(specs_with_alphas(a_up, a_dn - h), fp, dim)) / (2 * h);
        CHECK(derivs[0 * nw + w] == Catch::Approx(fd_up).margin(1e-6));
        CHECK(derivs[1 * nw + w] == Catch::Approx(fd_dn).margin(1e-6));
    }
}

// A fixed width must not consume a slot: the optimiser's flat parameter vector
// is packed over non-fixed parameters only, so a stale slot shifts every
// column after it onto the wrong parameter.
TEST_CASE("SlaterDeterminant skips fixed widths in the derivative layout",
          "[slater][optimize]") {
    const std::size_t dim = 2, np = 4;
    const std::size_t nw = GENERATE(3, 64, 73);
    const double a_up = 0.6, a_dn = 0.5;

    auto moved = make_det(specs_with_alphas(a_up, a_dn));
    SlaterDeterminant* det = moved.get();
    det->parameters()[0].is_fixed = true;   // pin the up block
    det->parameters()[1].is_fixed = false;
    TrialWavefunction psi;
    psi.add_component(std::move(moved));
    REQUIRE(psi.total_optimizable_params() == 1);

    WalkerSet ws(nw, np, dim);
    load_configs(ws);
    psi.evaluate_batch(ws);

    std::vector<double> derivs(1 * nw, 0.0);
    psi.compute_parameter_derivatives(ws, derivs);

    const double h = 1e-5;
    for (std::size_t w = 0; w < nw; ++w) {
        const std::vector<double>& fp = walker_configs()[w % walker_configs().size()];
        const double fd_dn =
            (log_psi_ref(specs_with_alphas(a_up, a_dn + h), fp, dim) -
             log_psi_ref(specs_with_alphas(a_up, a_dn - h), fp, dim)) / (2 * h);
        CHECK(derivs[w] == Catch::Approx(fd_dn).margin(1e-6));
    }
}

// ── Test F: the fold belongs in the derivative too ──────────────────────────
TEST_CASE("periodic Gaussian width derivative uses the folded separation",
          "[slater][optimize][crystal]") {
    // |r − c|² is what multiplies φ in ∂φ/∂α, so a derivative computed from the
    // RAW separation would disagree with the value it differentiates the moment
    // a walker sits across the cell boundary from its site.
    const double a = 10.0;
    const std::array<double, 4> hex{a, 0.0, a * 0.5, a * std::sqrt(3.0) / 2.0};
    const alea::SimulationCell cell(2, std::span<const double>(hex));
    const std::vector<double> centers{4.6, 0.2};
    const double alpha = 0.05;
    const GaussianOrbitalSet orb(2, centers, {alpha}, cell);

    const std::size_t nw = 2, dim = 2;
    // Walker 1 is walker 0 translated by one cell vector: the same point.
    std::vector<double> pos{-4.9, -4.9 + hex[0], 0.3, 0.3 + hex[1]};
    std::vector<double> dval(1 * nw);
    orb.evaluate_parameter_derivatives(pos, dim, dval, {}, {}, 0, nw);
    CHECK(dval[0] == Catch::Approx(dval[1]).epsilon(1e-12));

    const double h = 1e-7;
    auto val_at = [&](double al, std::size_t w) {
        const GaussianOrbitalSet o(dim, centers, {al}, cell);
        std::vector<double> v(nw), g(dim * nw), l(dim * nw);
        o.evaluate(pos, dim, v, g, l, 0, nw);
        return v[w];
    };
    CHECK(dval[0] ==
          Catch::Approx((val_at(alpha + h, 0) - val_at(alpha - h, 0)) / (2 * h))
              .margin(1e-7));
}

// ── Test G: a per-thread clone must carry the widths AND stay independent ───
TEST_CASE("SlaterDeterminant clone carries the widths independently",
          "[slater][optimize]") {
    auto det = make_det(specs_with_alphas(0.6, 0.5));
    for (Parameter& p : det->parameters()) p.is_fixed = false;

    auto copy = det->clone();
    REQUIRE(copy->parameters().size() == 2);
    CHECK(copy->parameters()[0].value == Catch::Approx(0.6));
    CHECK(copy->num_optimizable_params() == 2);   // flags travel, not just values

    // Optimising through the clone must not reach back into the original's
    // orbitals — every OpenMP thread holds one of these.
    copy->parameters()[0].value = 1.3;
    copy->rebuild();
    CHECK(det->parameters()[0].value == Catch::Approx(0.6));

    const std::size_t dim = 2, nw = 3, np = 4;
    WalkerSet w1(nw, np, dim), w2(nw, np, dim);
    load_configs(w1);
    load_configs(w2);
    det->evaluate_batch(w1);
    copy->evaluate_batch(w2);
    CHECK(w1.log_psi()[0] != Catch::Approx(w2.log_psi()[0]).epsilon(1e-9));
}

// ── ∂α of the orbital gradient and laplacian ────────────────────────────────
// Needed for ∂E_L/∂α, which is the curvature term the linear method's Hbar
// wants.  Checked against finite differences of evaluate() itself.
TEST_CASE("GaussianOrbitalSet width derivatives of grad and lap match FD",
          "[orbitals][optimize]") {
    const std::size_t dim = 2, nw = 2, n = 2;
    const double alpha = 0.6;
    const std::vector<double> centers{-1.0, 0.0, 1.0, 0.0};
    const GaussianOrbitalSet orb(dim, centers, {alpha, alpha});

    std::vector<double> pos{0.3, -0.7, -0.2, 0.9};   // [d*nw + w]
    std::vector<double> dval(n * nw), dgrad(n * dim * nw), dlap(n * dim * nw);
    orb.evaluate_parameter_derivatives(pos, dim, dval, dgrad, dlap, 0, nw);

    const double h = 1e-6;
    auto at = [&](double a, std::vector<double>& g, std::vector<double>& l) {
        const GaussianOrbitalSet o(dim, centers, {a, a});
        std::vector<double> v(n * nw);
        g.assign(n * dim * nw, 0.0);
        l.assign(n * dim * nw, 0.0);
        o.evaluate(pos, dim, v, g, l, 0, nw);
    };
    std::vector<double> gp, lp, gm, lm;
    at(alpha + h, gp, lp);
    at(alpha - h, gm, lm);

    for (std::size_t k = 0; k < n * dim * nw; ++k) {
        CHECK(dgrad[k] == Catch::Approx((gp[k] - gm[k]) / (2 * h)).margin(1e-6));
        CHECK(dlap[k] == Catch::Approx((lp[k] - lm[k]) / (2 * h)).margin(1e-5));
    }
}

// ── ∂α of ∇lnψ_det and ∇²lnψ_det ───────────────────────────────────────────
// The determinant's inverse moves with α too (∂A = -A ∂D A), so this is not
// just the orbitals differentiated.  FD of the component's own committed caches.
TEST_CASE("SlaterDeterminant parameter derivatives of grad and lap match FD",
          "[slater][optimize]") {
    const std::size_t dim = 2, np = 4;
    const std::size_t nw = GENERATE(3, 64, 73);
    const double a_up = 0.6, a_dn = 0.5;

    auto build = [&](double au, double ad) {
        auto det = make_det(specs_with_alphas(au, ad));
        for (Parameter& p : det->parameters()) p.is_fixed = false;
        return det;
    };
    auto fields = [&](double au, double ad, std::vector<double>& g,
                      std::vector<double>& l) {
        auto det = build(au, ad);
        WalkerSet ws(nw, np, dim);
        load_configs(ws);
        det->evaluate_batch(ws);
        g.assign(det->grad_c().begin(), det->grad_c().end());
        l.assign(det->lap_c().begin(), det->lap_c().end());
    };

    auto det = build(a_up, a_dn);
    WalkerSet ws(nw, np, dim);
    load_configs(ws);
    det->evaluate_batch(ws);

    std::vector<double> dgrad(2 * np * dim * nw), dlap(2 * np * dim * nw);
    REQUIRE(det->compute_parameter_gradient_derivatives(ws, dgrad, dlap));

    const double h = 1e-6;
    for (std::size_t k = 0; k < 2; ++k) {
        std::vector<double> gp, lp, gm, lm;
        fields(k == 0 ? a_up + h : a_up, k == 1 ? a_dn + h : a_dn, gp, lp);
        fields(k == 0 ? a_up - h : a_up, k == 1 ? a_dn - h : a_dn, gm, lm);
        for (std::size_t q = 0; q < np * dim * nw; ++q) {
            const std::size_t idx = k * np * dim * nw + q;
            CHECK(dgrad[idx] ==
                  Catch::Approx((gp[q] - gm[q]) / (2 * h)).margin(1e-5));
            CHECK(dlap[idx] ==
                  Catch::Approx((lp[q] - lm[q]) / (2 * h)).margin(1e-4));
        }
    }
}
