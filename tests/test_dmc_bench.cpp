// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Manual benchmark harness (not a ctest case — run ./test_dmc_bench by hand).
//
// Two independent things are measured:
//
//   1. Thread scaling of DMC on a 2-particle exciton (the original benchmark).
//
//   2. A PERIODIC np=16 baseline, mirroring configs/periodic_2d_gas.yaml.  This
//      is the Milestone 2 reference point: the Ewald sum will be added to this
//      exact system, so the "before" number has to exist before any Ewald code
//      does.  Three variants are timed so the added cost can be attributed:
//        (a) kinetic + Jastrow only        — no pair potential at all
//        (b) + minimum-image Coulomb       — the cheapest possible pair term
//        (c) [later] + Ewald               — Milestone 2
//      (b) minus (a) is what a 120-pair potential costs today; that is the
//      yardstick for "Ewald's real-space half costs 2-3x a Coulomb pair".
//
// Reported per variant: wall time, and nanoseconds per Hamiltonian evaluation
// per walker, which is the size-independent number to compare against.
//
// ── BASELINE RECORDED 2026-09-22 (Milestone 1 tree, -O3 -march=native) ──────
// Periodic np=16 (8e+8h), hexagonal cell rs=1, 512 walkers, 4 threads:
//
//   Hamiltonian::evaluate_batch, kinetic only        :  3 139 ns/call
//   Hamiltonian::evaluate_batch, + 120-pair Coulomb  : 76 665 ns/call
//   cost of the 120-pair Coulomb term                :      1.20 ns/walker-pair
//
//   VMC step, no pair potential                      : 19 930 us
//   VMC step, with Coulomb                           : 19 960 us
//   => the Hamiltonian is 0.38 % of a step.
//
// That last number is the one that matters for Milestone 2: the wavefunction
// dominates so completely that even an Ewald term costing several times the
// Coulomb one barely moves the total.
//
// ── AFTER Ewald + the vectorization fix (same machine, same day) ───────────
//   Hamiltonian::evaluate_batch, + Ewald tol=1e-6      : 712 000 ns/call
//   => the Hamiltonian is ~3.4 % of a step.
//
// Getting there needed one non-obvious fix.  A blend mask written as
// `x < x_hi ? 1.0 : 0.0` reads to GCC as a conditional whose false branch makes
// the expensive polynomial dead, so if-conversion turns the branchless blend
// back INTO control flow and the loop drops to scalar.  Writing the mask with
// std::copysign instead — sign-bit arithmetic, with no conditional to
// recognise — keeps it vectorized.  KeldyshTerm had the same defect and got
// 3.9x faster from the same one-line change (649 us -> 167 us per call).
// Equally: std::fmin/std::fmax do NOT vectorize (libm calls with NaN
// semantics); a plain ternary clamp does, and becomes vminpd.

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <omp.h>
#include <span>
#include <vector>

#include "hamiltonian/coulomb.h"
#include "hamiltonian/hamiltonian.h"
#include "hamiltonian/ewald_term.h"
#include "hamiltonian/kinetic.h"
#include "population/particle_group.h"
#include "population/simulation_cell.h"
#include "population/walker.h"
#include "population/walker_ensemble.h"
#include "population/walker_set.h"
#include "qmc/dmc.h"
#include "qmc/vmc.h"
#include "wavefunction/pade_jastrow.h"
#include "wavefunction/trial_wavefunction.h"

using namespace alea;
using namespace std::chrono;

static ParticleGroup grp(std::size_t f, std::size_t l, double q, double im) {
    ParticleGroup g;
    g.name = "g";
    g.charge = q;
    g.first = f;
    g.last = l;
    g.inv_mass = {im, im};
    g.mass     = {1.0 / im, 1.0 / im};
    g.count    = l - f;
    return g;
}

// ── 1. Original 2-particle thread-scaling benchmark ─────────────────────────

static void bench_exciton(int nt, const char* label, int nw) {
    WalkerEnsemble ens(nw, {grp(0, 1, -1, 1), grp(1, 2, +1, 1)}, 2);
    numerics::Random rng(42);
    for (auto& w : ens.walkers()) {
        std::vector<double> p = {rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5),
                                 rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)};
        w.set_positions(p);
    }
    TrialWavefunction psi;
    psi.add_component(std::make_unique<PadeJastrow>(
        1, 1, grp(0, 1, -1, 1), grp(1, 2, +1, 1),
        std::vector<Parameter>{{"a", -1, true}, {"b", 0, true}}));
    Hamiltonian ham;
    ham.add_component(std::make_unique<Kinetic>(
        std::vector<ParticleGroup>{grp(0, 1, -1, 1), grp(1, 2, +1, 1)}));
    ham.add_component(
        std::make_unique<CoulombTerm>(grp(0, 1, -1, 1), grp(1, 2, +1, 1), 1));

    DMCConfig c;
    c.tau = 0.01;
    c.n_equilibration_blocks = 2;
    c.n_accumulation_blocks  = 4;
    c.n_steps = 5;
    c.n_substeps = 10;
    c.seed = 42;
    c.n_threads = nt;

    DMC dmc(c, ens, psi, ham);
    auto t0 = high_resolution_clock::now();
    dmc.run();
    auto t1 = high_resolution_clock::now();
    std::cout << "  " << label << ": " << duration_cast<milliseconds>(t1 - t0).count()
              << " ms   E=" << dmc.energy() << "  acc=" << dmc.acceptance_ratio()
              << "\n";
}

// ── 2. Periodic np=16: DIRECT Hamiltonian microbenchmark ───────────────────
//
// An end-to-end DMC comparison does NOT work here.  DMC branches, so two
// variants with different physics reach different populations and therefore do
// different amounts of work — the first attempt at this benchmark had the
// Coulomb variant coming out 10% FASTER than no-potential-at-all, which is
// impossible for purely added work.  The walker count, not the potential, was
// driving the difference.
//
// So time the thing actually in question: Hamiltonian::evaluate_batch on a
// FIXED WalkerSet.  No branching, no acceptance, no population drift — just the
// cost of the term, which is exactly the number Milestone 2 has to defend.

// Hexagonal cell at density parameter rs, mirroring the `rs:` YAML form:
// A = N pi rs^2 in 2D, and A = L^2 sqrt(3)/2 for a hexagonal cell.
static SimulationCell hex_cell_for(std::size_t np, double rs) {
    const double area = static_cast<double>(np) * std::numbers::pi_v<double> * rs * rs;
    const double L    = std::sqrt(2.0 * area / std::sqrt(3.0));
    const std::vector<double> a = {L, 0.0, 0.5 * L, L * std::sqrt(3.0) / 2.0};
    return SimulationCell(2, std::span<const double>(a));
}

struct HamBench {
    double ns_per_call;        // one evaluate_batch over all walkers
    double ns_per_walker;      // per walker, the size-independent figure
    double energy_checksum;    // keeps the optimizer from deleting the loop
};

enum class PairTerm { None, MinImageCoulomb, Ewald };

static HamBench bench_hamiltonian(const char* label, std::size_t nw,
                                  PairTerm which, int n_rep, double ewald_tol = 1e-6) {
    constexpr std::size_t n_e = 8, n_h = 8, np = n_e + n_h;
    const ParticleGroup  e    = grp(0, n_e, -1.0, 1.0);
    const ParticleGroup  h    = grp(n_e, np, +1.0, 1.0);
    const SimulationCell cell = hex_cell_for(np, 1.0);

    WalkerSet ws(nw, np, 2, cell);
    numerics::Random rng(2026);
    const auto a = cell.vectors();
    for (std::size_t i = 0; i < np; ++i)
        for (std::size_t w = 0; w < nw; ++w) {
            const double f0 = rng.uniform(-0.5, 0.5), f1 = rng.uniform(-0.5, 0.5);
            ws.positions()[(i * 2 + 0) * nw + w] = f0 * a[0] + f1 * a[2];
            ws.positions()[(i * 2 + 1) * nw + w] = f0 * a[1] + f1 * a[3];
        }
    ws.dist_tables().build_all(ws.positions());

    // Kinetic reads gradients/laplacians, so the wavefunction must run once.
    TrialWavefunction psi;
    psi.add_component(std::make_unique<PadeJastrow>(
        1, 1, e, h, std::vector<Parameter>{{"a", -0.5, true}, {"b", 0.3, true}}));
    psi.add_component(std::make_unique<PadeJastrow>(
        1, 1, e, e, std::vector<Parameter>{{"a", 0.25, true}, {"b", 0.4, true}}));
    psi.add_component(std::make_unique<PadeJastrow>(
        1, 1, h, h, std::vector<Parameter>{{"a", 0.25, true}, {"b", 0.4, true}}));
    psi.evaluate_batch(ws);

    Hamiltonian ham;
    ham.add_component(std::make_unique<Kinetic>(std::vector<ParticleGroup>{e, h}));
    if (which == PairTerm::MinImageCoulomb) {
        // Minimum-image truncation: NOT converged physics, but the right cost
        // shape — 120 pairs through the existing vectorized pair loop.
        ham.add_component(std::make_unique<CoulombTerm>(e, h, 1.0));
        ham.add_component(std::make_unique<CoulombTerm>(e, e, 1.0));
        ham.add_component(std::make_unique<CoulombTerm>(h, h, 1.0));
    } else if (which == PairTerm::Ewald) {
        // One term for the whole system: the reciprocal half couples every
        // charge through a single structure factor.
        ham.add_component(std::make_unique<EwaldTerm>(
            std::vector<ParticleGroup>{e, h}, 1.0, cell, ewald_tol));
    }

    for (int i = 0; i < 20; ++i) ham.evaluate_batch(ws);  // warm the caches

    double sink = 0.0;
    auto   t0   = high_resolution_clock::now();
    for (int rep = 0; rep < n_rep; ++rep) {
        ham.evaluate_batch(ws);
        sink += ws.local_energies()[0];
    }
    auto t1 = high_resolution_clock::now();

    const double ns =
        static_cast<double>(duration_cast<nanoseconds>(t1 - t0).count()) / n_rep;
    std::cout << "  " << label << std::setw(12) << ns << " ns/call  "
              << std::setw(10) << ns / static_cast<double>(nw) << " ns/walker\n";
    return {ns, ns / static_cast<double>(nw), sink};
}

// End-to-end VMC on the same periodic system.  VMC, not DMC, because VMC has a
// FIXED walker count — no branching, so two variants do identical amounts of
// work and the difference really is the potential.  This answers the question
// the microbenchmark cannot: what FRACTION of a real step is the Hamiltonian?
// That fraction is the headroom Milestone 2 has to spend.
static double bench_vmc_end_to_end(std::size_t nw, bool with_coulomb, int nt) {
    constexpr std::size_t n_e = 8, n_h = 8, np = n_e + n_h;
    const ParticleGroup  e    = grp(0, n_e, -1.0, 1.0);
    const ParticleGroup  h    = grp(n_e, np, +1.0, 1.0);
    const SimulationCell cell = hex_cell_for(np, 1.0);

    WalkerEnsemble ens(nw, {e, h}, 2, cell);
    numerics::Random rng(2026);
    const auto a = cell.vectors();
    std::vector<double> p(np * 2);
    for (auto& w : ens.walkers()) {
        for (std::size_t i = 0; i < np; ++i) {
            const double f0 = rng.uniform(-0.5, 0.5), f1 = rng.uniform(-0.5, 0.5);
            p[i * 2 + 0] = f0 * a[0] + f1 * a[2];
            p[i * 2 + 1] = f0 * a[1] + f1 * a[3];
        }
        w.set_positions(p);
    }

    TrialWavefunction psi;
    psi.add_component(std::make_unique<PadeJastrow>(
        1, 1, e, h, std::vector<Parameter>{{"a", -0.5, true}, {"b", 0.3, true}}));
    psi.add_component(std::make_unique<PadeJastrow>(
        1, 1, e, e, std::vector<Parameter>{{"a", 0.25, true}, {"b", 0.4, true}}));
    psi.add_component(std::make_unique<PadeJastrow>(
        1, 1, h, h, std::vector<Parameter>{{"a", 0.25, true}, {"b", 0.4, true}}));

    Hamiltonian ham;
    ham.add_component(std::make_unique<Kinetic>(std::vector<ParticleGroup>{e, h}));
    if (with_coulomb) {
        ham.add_component(std::make_unique<CoulombTerm>(e, h, 1.0));
        ham.add_component(std::make_unique<CoulombTerm>(e, e, 1.0));
        ham.add_component(std::make_unique<CoulombTerm>(h, h, 1.0));
    }

    VMCConfig c;
    c.tau = 0.02;
    c.n_equilibration_blocks = 4;
    c.n_accumulation_blocks  = 16;
    c.n_steps = 10;
    c.n_substeps = 3;
    c.seed = 2026;
    c.n_threads = nt;

    VMC vmc(c, ens, psi, ham);
    auto t0 = high_resolution_clock::now();
    vmc.run();
    auto t1 = high_resolution_clock::now();
    return static_cast<double>(duration_cast<milliseconds>(t1 - t0).count());
}

int main() {
    const int n_threads = omp_get_max_threads();
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "\n=== 1. Thread scaling: 2-particle exciton, 2048 walkers ===\n";
    bench_exciton(1, "1 thread ", 2048);
    bench_exciton(2, "2 threads", 2048);
    bench_exciton(n_threads, "all      ", 2048);

    std::cout << "\n=== 2. Periodic np=16 (8e+8h), hexagonal cell rs=1, 512 walkers ===\n";
    std::cout << "    Hamiltonian::evaluate_batch only — no DMC dynamics.\n";
    std::cout << "    (Milestone 2 baseline: Ewald will be added to this system)\n";
    constexpr std::size_t nw    = 512;
    constexpr int         n_rep = 3000;
    const HamBench base = bench_hamiltonian("(a) kinetic only                 : ",
                                            nw, PairTerm::None, n_rep);
    const HamBench coul = bench_hamiltonian("(b) + min-image Coulomb, 120 prs : ",
                                            nw, PairTerm::MinImageCoulomb, n_rep);
    const HamBench ew6  = bench_hamiltonian("(c) + Ewald, tol=1e-6            : ",
                                            nw, PairTerm::Ewald, n_rep, 1e-6);
    const HamBench ew8  = bench_hamiltonian("(d) + Ewald, tol=1e-8            : ",
                                            nw, PairTerm::Ewald, n_rep, 1e-8);

    const double pair_cost = coul.ns_per_call - base.ns_per_call;
    std::cout << "\n  a 120-pair Coulomb term costs " << pair_cost << " ns/call ("
              << 100.0 * pair_cost / base.ns_per_call << "% over kinetic alone)\n";
    std::cout << "  per walker-pair: " << pair_cost / (120.0 * nw) << " ns\n";
    std::cout << "  Ewald over kinetic-only : "
              << (ew6.ns_per_call - base.ns_per_call) << " ns/call (tol 1e-6), "
              << (ew8.ns_per_call - base.ns_per_call) << " ns/call (tol 1e-8)\n";
    std::cout << "  Ewald / min-image Coulomb ratio: "
              << ew6.ns_per_call / coul.ns_per_call << "x (tol 1e-6)\n";
    std::cout << "  checksum (ignore): "
              << base.energy_checksum + coul.energy_checksum + ew6.energy_checksum +
                     ew8.energy_checksum
              << "\n";

    std::cout << "\n=== 3. How much of a real step is the Hamiltonian? ===\n";
    std::cout << "    VMC, fixed population (no branching), same system.\n";
    const int    nt_v    = std::min(4, n_threads);
    const double vmc_a   = bench_vmc_end_to_end(nw, /*with_coulomb=*/false, nt_v);
    const double vmc_b   = bench_vmc_end_to_end(nw, /*with_coulomb=*/true, nt_v);

    // VMC evaluates the Hamiltonian once per STEP (vmc.cpp:143).
    const double n_steps_total   = (4.0 + 16.0) * 10.0;
    const double ham_share_ewald =
        100.0 * ew6.ns_per_call / (1e6 * vmc_b / n_steps_total);
    std::cout << "  (a) no pair potential : " << vmc_a << " ms\n";
    std::cout << "  (b) + Coulomb         : " << vmc_b << " ms\n";
    std::cout << "  per step: (a) " << 1e6 * vmc_a / 1e3 / n_steps_total
              << " us   (b) " << 1e6 * vmc_b / 1e3 / n_steps_total << " us\n";
    std::cout << "  Hamiltonian share of a step, Coulomb: "
              << 100.0 * coul.ns_per_call / (1e6 * vmc_b / n_steps_total) << " %\n";
    std::cout << "  Hamiltonian share of a step, Ewald  : " << ham_share_ewald
              << " %   <-- the Milestone 2 number\n";
    std::cout << "\n  HEADROOM: if the Hamiltonian is a small share of the step,\n"
              << "  an Ewald term costing several times the Coulomb one still\n"
              << "  moves the total only slightly.  That is the number that\n"
              << "  decides whether Milestone 2 is affordable.\n\n";
    return 0;
}
