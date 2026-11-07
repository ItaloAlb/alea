// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Fixed-node restriction (Ceperley, J. Stat. Phys. 63, 1237 (1991)): its
// contracts and two noninteracting fermions, for both link treatments (beads
// only, and the image action).
//
// Every check uses a reference that does not trust the code under test:
//  * determinants of the explicit Gaussian matrix by Eigen's full-pivot LU,
//    for every proposal, against the cached Sherman–Morrison ratios;
//  * the DMC SlaterDeterminant's node-crossing phase on the same configurations;
//  * incremental image-action changes (one bead, windows, the whole ring)
//    against differences of fresh total actions; the node distance against the
//    exact distance to a planar node and against finite differences of ln|det|;
//    the thermal derivative against a finite difference in tau;
//  * the exact finite-M energy of two fermions in a 2D trap with the node
//    x1 = x2 (independent quadrature: the validation battery's
//    v1_ref_fixed_node_finite_m.py, with and without --image), which the restricted primitive path integral must
//    reproduce at finite tau, through the driver and through YAML -> App;
//  * bit-identical chains where the nodes never act, and for two determinants
//    with different orbitals but the same nodal plane (beads only).
// Statistical comparisons use the driver's own error bar at a fixed seed with
// the same 4 sigma + 1e-3 tolerance as test_pimc_physics.cpp.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "builders/action_builders.h"
#include "hamiltonian/harmonic_trap.h"
#include "io/config_writer.h"
#include "io/parser.h"
#include "pimc/fixed_node_restriction.h"
#include "pimc/pimc.h"
#include "pimc/primitive_action.h"
#include "population/walker_set.h"
#include "qmc/app.h"
#include "wavefunction/orbital_set.h"
#include "wavefunction/slater_determinant.h"
#include "wavefunction/trial_wavefunction.h"

namespace {
using namespace alea;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;
using Crossing = FixedNodeRestriction::NodeCrossing;

constexpr double kInf = std::numeric_limits<double>::infinity();

ParticleGroup group(std::string name, std::size_t first, std::size_t last,
                    std::size_t dim) {
    ParticleGroup g;
    g.name = std::move(name);
    g.first = first;
    g.last = last;
    g.count = last - first;
    g.charge = 0.0;
    g.mass.assign(dim, 1.0);
    g.inv_mass.assign(dim, 1.0);
    return g;
}

double tolerance(double error) { return 4.0 * error + 1e-3; }

// Gaussian orbitals exp(-alpha |r - c_i|^2) at `centers` [n*dim], folded to the
// minimum image of `cell`, evaluated without OrbitalSet.
struct Orbitals {
    std::vector<double> centers;
    double alpha;
    SimulationCell cell;
    std::size_t dim;
    std::size_t size() const { return centers.size() / dim; }
    double value(std::size_t i, const double* r) const {
        double del[SimulationCell::kMaxDim] = {};
        for (std::size_t d = 0; d < dim; ++d) del[d] = r[d] - centers[i * dim + d];
        cell.min_image(del);
        double r2 = 0.0;
        for (std::size_t d = 0; d < dim; ++d) r2 += del[d] * del[d];
        return std::exp(-alpha * r2);
    }
    std::unique_ptr<OrbitalSet> set() const {
        return std::make_unique<GaussianOrbitalSet>(dim, centers,
                                                    std::vector<double>(size(), alpha), cell);
    }
};

// det[phi_i(r_j)] at slice k, particle `moved` (if any) at `r_new`.
double reference_det(const PathSet& path, std::size_t k, const Orbitals& orb,
                     std::size_t moved = static_cast<std::size_t>(-1),
                     const double* r_new = nullptr) {
    const std::size_t n = orb.size();
    Eigen::MatrixXd D(n, n);
    for (std::size_t j = 0; j < n; ++j) {
        double r[SimulationCell::kMaxDim];
        for (std::size_t d = 0; d < orb.dim; ++d)
            r[d] = j == moved ? r_new[d] : path.pos(j, d, k);
        for (std::size_t i = 0; i < n; ++i)
            D(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) = orb.value(i, r);
    }
    return D.fullPivLu().determinant();
}

// ln|det[phi_i(r_j)]| - sum_j ln max_i |phi_i(r_j)|: the row-normalized
// determinant whose gradient sets the image action's node distance.
double reference_log_scaled_det(const PathSet& path, std::size_t k, const Orbitals& orb,
                                std::size_t moved, const double* r_new) {
    double log_norms = 0.0;
    for (std::size_t j = 0; j < orb.size(); ++j) {
        double r[SimulationCell::kMaxDim];
        for (std::size_t d = 0; d < orb.dim; ++d)
            r[d] = j == moved ? r_new[d] : path.pos(j, d, k);
        double largest = 0.0;
        for (std::size_t i = 0; i < orb.size(); ++i) largest = std::max(largest, std::abs(orb.value(i, r)));
        log_norms += std::log(largest);
    }
    return std::log(std::abs(reference_det(path, k, orb, moved, r_new))) - log_norms;
}

FixedNodeRestriction one_block(const Orbitals& orb, Crossing crossing = Crossing::beads_only,
                               std::vector<double> inv_mass = {}) {
    if (inv_mass.empty()) inv_mass.assign(orb.size() * orb.dim, 1.0);
    std::vector<FixedNodeRestriction::Block> blocks;
    blocks.push_back({0, orb.size(), orb.set()});
    return FixedNodeRestriction(std::move(blocks), orb.size(), orb.dim, std::move(inv_mass),
                                crossing);
}

// Every bead at its orbital center plus a deterministic jitter.
void fill_near_centers(PathSet& path, const Orbitals& orb, double jitter, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(0.0, jitter);
    for (std::size_t p = 0; p < path.n_particles(); ++p)
        for (std::size_t k = 0; k < path.num_slices(); ++k)
            for (std::size_t d = 0; d < path.dim(); ++d)
                path.pos(p, d, k) = orb.centers[p * orb.dim + d] + gauss(rng);
    path.rebuild_tables();
}

// Random single-bead proposals: the restriction's decision must equal the sign
// of the fresh determinant ratio; accepted proposals are committed, and every
// `edit_every` steps several particles of one slice are moved behind the
// cache's back.  Returns {rejections, acceptances}.
std::pair<int, int> random_walk(PathSet& path, FixedNodeRestriction& restriction,
                                const Orbitals& orb, double step, int steps,
                                int edit_every, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(0.0, step);
    std::uniform_int_distribution<std::size_t> particle(0, path.n_particles() - 1);
    std::uniform_int_distribution<std::size_t> slice(0, path.num_slices() - 1);
    int rejected = 0, accepted = 0;
    for (int s = 0; s < steps; ++s) {
        const std::size_t p = particle(rng), k = slice(rng);
        double proposal[SimulationCell::kMaxDim];
        for (std::size_t d = 0; d < path.dim(); ++d) proposal[d] = path.pos(p, d, k) + gauss(rng);
        const double ratio = reference_det(path, k, orb, p, proposal) / reference_det(path, k, orb);
        const double delta = restriction.delta_potential_particle(
            path, p, k, std::span<const double>(proposal, path.dim()));
        INFO("step " << s << " ratio " << ratio);
        REQUIRE(delta == (ratio > 0.0 ? 0.0 : kInf));
        if (delta != 0.0) {
            ++rejected;
        } else if (rng() % 2 == 0) {
            ++accepted;
            for (std::size_t d = 0; d < path.dim(); ++d) path.pos(p, d, k) = proposal[d];
            path.update_particle_tables(p, k, k + 1);
        }
        if (edit_every > 0 && s % edit_every == edit_every - 1) {
            const std::size_t ke = slice(rng);
            for (std::size_t q = 0; q < std::min<std::size_t>(3, path.n_particles()); ++q)
                for (std::size_t d = 0; d < path.dim(); ++d) path.pos(q, d, ke) += 0.1 * gauss(rng);
            path.update_particle_tables(0, ke, ke + 1);
        }
    }
    return {rejected, accepted};
}

PimcConfig short_config(double beta, double tau, std::uint64_t seed) {
    PimcConfig cfg;
    cfg.threads = 2;
    cfg.samples = 32;
    cfg.beta = beta;
    cfg.time_step = tau;
    cfg.equilibration_blocks = 200;
    cfg.accumulation_blocks = 800;
    cfg.steps_per_block = 20;
    cfg.sub_steps = 2;
    cfg.seed = seed;
    return cfg;
}

void add_move(PimcConfig& cfg, const std::string& type, int levels, double step) {
    PimcConfig::MoveSpec move;
    move.type = type;
    move.levels = levels;
    move.step = step;
    cfg.moves.push_back(move);
}

// Primitive action of `ham`, optionally followed by a restriction.
Action restricted_action(std::span<const ParticleGroup> groups, const Hamiltonian& ham,
                         std::size_t dim, std::unique_ptr<ActionComponent> nodes) {
    Action action(groups, ham, dim, false, false);
    action.add_component(std::make_unique<PrimitiveKineticAction>(groups, dim));
    action.add_component(std::make_unique<PrimitivePotentialAction>(groups, ham));
    if (nodes) action.add_component(std::move(nodes));
    return action;
}

std::unique_ptr<FixedNodeRestriction> two_center_nodes(double s, double alpha,
                                                       Crossing crossing = Crossing::beads_only) {
    const Orbitals orb{{s, 0.0, -s, 0.0}, alpha, SimulationCell{}, 2};
    std::vector<FixedNodeRestriction::Block> blocks;
    blocks.push_back({0, 2, orb.set()});
    return std::make_unique<FixedNodeRestriction>(std::move(blocks), 2, 2,
                                                  std::vector<double>(4, 1.0), crossing);
}

// Exact finite-M energies of two m = 1 fermions in the 2D trap omega = 1 with
// the node x1 = x2, beta = 4, tau = 0.25 (M = 16), primitive action.  From
//   v1_ref_fixed_node_finite_m.py --beta 4 --slices 16 [--image]
// of the validation battery
// (quadrature converged to 1e-13; the continuum value is 3.0566).
constexpr double kBeadsOnlyEnergy = 2.6016594600186655;  // nodes checked at beads
constexpr double kImageEnergy = 3.0336417616;            // plus the image action
}  // namespace

TEST_CASE("Fixed-node decisions equal fresh determinant signs in an open cell",
          "[pimc][fixed-node]") {
    // Five broad overlapping orbitals: proposals cross nodes often.
    const Orbitals orb{{0.0, 0.0, 1.0, 0.2, -0.8, 0.9, 0.3, -1.1, -0.6, -0.5}, 0.7,
                       SimulationCell{}, 2};
    PathSet path(6, orb.size(), 2, 0.1);
    fill_near_centers(path, orb, 0.15, 11);
    auto restriction = one_block(orb);
    REQUIRE(restriction.total_action(path) == 0.0);
    const auto [rejected, accepted] = random_walk(path, restriction, orb, 0.6, 3000, 250, 5);
    CHECK(rejected > 100);
    CHECK(accepted > 100);
    CHECK(restriction.updates() > 100);   // Sherman–Morrison path exercised
    CHECK(restriction.rebuilds() > 6);    // and the rebuild path
}

TEST_CASE("Fixed-node decisions equal fresh determinant signs under periodic folding",
          "[pimc][fixed-node]") {
    // Seven commensurate triangular sites in a hexagonal cell; orbitals narrow
    // enough that the minimum-image fold passes the 1e-6 bound.
    const double rs = 1.0, area = 7.0 * std::numbers::pi * rs * rs;
    const double a = std::sqrt(2.0 * area / std::sqrt(3.0));
    const std::vector<double> vectors{a, 0.0, 0.5 * a, 0.5 * std::sqrt(3.0) * a};
    const SimulationCell cell(2, vectors);
    const double spacing = std::sqrt(2.0 * std::numbers::pi / std::sqrt(3.0)) * rs;
    const double alpha = 8.0 / (spacing * spacing);
    const Orbitals orb{commensurate_triangular_sites(cell, 7), alpha, cell, 2};
    REQUIRE(orb.set()->periodic_fold_bound() <= FixedNodeRestriction::kFoldTolerance);
    PathSet path(5, 7, 2, 0.1, cell);
    fill_near_centers(path, orb, 0.1 * spacing, 3);
    auto restriction = one_block(orb);
    REQUIRE(restriction.total_action(path) == 0.0);
    const auto [rejected, accepted] =
        random_walk(path, restriction, orb, 0.45 * spacing, 2000, 200, 9);
    CHECK(rejected > 20);
    CHECK(accepted > 100);
}

TEST_CASE("Fixed-node decisions equal the DMC determinant's node-crossing phase",
          "[pimc][fixed-node]") {
    const Orbitals orb{{0.0, 0.0, 1.0, 0.2, -0.8, 0.9, 0.3, -1.1}, 0.6, SimulationCell{}, 2};
    const std::size_t M = 8, np = orb.size(), dim = 2;
    PathSet path(M, np, dim, 0.1);
    fill_near_centers(path, orb, 0.3, 21);
    auto restriction = one_block(orb);

    // Slices as walkers: PathSet and WalkerSet share the SoA layout.
    WalkerSet walkers(M, np, dim);
    std::copy(path.positions().begin(), path.positions().end(), walkers.positions().begin());
    walkers.dist_tables().build_all(walkers.positions());
    std::vector<SlaterDeterminant::Block> blocks;
    blocks.push_back({0, np, orb.set()});
    TrialWavefunction psi;
    psi.add_component(std::make_unique<SlaterDeterminant>(std::move(blocks)));
    psi.evaluate_batch(walkers);

    std::mt19937_64 rng(4);
    std::normal_distribution<double> gauss(0.0, 0.7);
    int flips = 0;
    for (int s = 0; s < 400; ++s) {
        const std::size_t p = rng() % np, k = rng() % M;
        std::vector<double> soa(dim * M);  // [d*M + w]: only walker k moves
        for (std::size_t d = 0; d < dim; ++d)
            for (std::size_t w = 0; w < M; ++w) soa[d * M + w] = path.pos(p, d, w);
        double proposal[2];
        for (std::size_t d = 0; d < dim; ++d) soa[d * M + k] = proposal[d] = path.pos(p, d, k) + gauss(rng);
        walkers.propose_move_batch(p, soa);
        psi.ratio_batch(walkers, p);
        const bool dmc_flip = std::abs(walkers.phase_diffs()[k] - std::numbers::pi) < 1e-9;
        const double delta = restriction.delta_potential_particle(path, p, k, proposal);
        INFO("step " << s);
        REQUIRE((delta == kInf) == dmc_flip);
        flips += dmc_flip;
    }
    CHECK(flips > 20);
}

TEST_CASE("total_action admits one nodal pocket and refuses a slice that left it",
          "[pimc][fixed-node]") {
    const Orbitals orb{{0.0, 0.0, 1.5, 0.0, 0.0, 1.5, 1.5, 1.5}, 1.0, SimulationCell{}, 2};
    PathSet path(6, orb.size(), 2, 0.1);
    fill_near_centers(path, orb, 0.05, 2);
    for (const Crossing crossing : {Crossing::beads_only, Crossing::image}) {
        auto restriction = one_block(orb, crossing);
        const double inside = restriction.total_action(path);
        CHECK(std::isfinite(inside));
        CHECK(inside >= 0.0);
        if (crossing == Crossing::beads_only) CHECK(inside == 0.0);
        const int sign = restriction.block_sign(path, 0, 0);
        CHECK(std::abs(sign) == 1);

        // Swapping two particles on one slice flips that slice's sign.
        for (std::size_t d = 0; d < 2; ++d) std::swap(path.pos(0, d, 2), path.pos(1, d, 2));
        CHECK(restriction.block_sign(path, 0, 2) == -sign);
        CHECK(restriction.total_action(path) == kInf);
        for (std::size_t d = 0; d < 2; ++d) std::swap(path.pos(0, d, 2), path.pos(1, d, 2));
        CHECK(restriction.total_action(path) == inside);

        // Two particles on one point: a singular determinant sits on the node.
        const double saved[2] = {path.pos(0, 0, 3), path.pos(0, 1, 3)};
        for (std::size_t d = 0; d < 2; ++d) path.pos(0, d, 3) = path.pos(1, d, 3);
        CHECK(restriction.block_sign(path, 0, 3) == 0);
        CHECK(restriction.total_action(path) == kInf);
        for (std::size_t d = 0; d < 2; ++d) path.pos(0, d, 3) = saved[d];
    }
}

TEST_CASE("Image-action changes equal differences of fresh total actions",
          "[pimc][fixed-node][image]") {
    // Broad orbitals and a coarse tau: many links sit near a node.
    const Orbitals orb{{0.0, 0.0, 1.0, 0.2, -0.8, 0.9, 0.3, -1.1}, 0.7, SimulationCell{}, 2};
    const std::size_t M = 7, np = orb.size(), dim = 2;
    PathSet path(M, np, dim, 0.4);
    fill_near_centers(path, orb, 0.2, 31);
    // Anisotropic inverse masses exercise the mass-scaled distance.
    std::vector<double> inv_mass(np * dim);
    for (std::size_t i = 0; i < inv_mass.size(); ++i) inv_mass[i] = 0.5 + 0.25 * static_cast<double>(i % 3);
    auto restriction = one_block(orb, Crossing::image, inv_mass);
    auto fresh = [&]() { return one_block(orb, Crossing::image, inv_mass); };
    REQUIRE(std::isfinite(restriction.total_action(path)));

    std::mt19937_64 rng(77);
    std::normal_distribution<double> gauss(0.0, 0.35);
    int finite = 0, accepted = 0;
    std::vector<double> saved, proposal;
    for (int s = 0; s < 600; ++s) {
        const std::size_t p = rng() % np;
        const std::size_t first = rng() % M;
        // Single beads, windows and the whole ring.
        const std::size_t count = s % 3 == 0 ? 1 : (s % 3 == 1 ? 1 + rng() % (M - 1) : M);
        proposal.assign(count * dim, 0.0);
        saved.assign(count * dim, 0.0);
        for (std::size_t i = 0; i < count; ++i)
            for (std::size_t d = 0; d < dim; ++d) {
                saved[i * dim + d] = path.pos(p, d, (first + i) % M);
                proposal[i * dim + d] = saved[i * dim + d] + gauss(rng);
            }
        const double delta = count == 1 && s % 2 == 0
            ? restriction.delta_potential_particle(path, p, first,
                                                   std::span<const double>(proposal.data(), dim))
            : restriction.delta_potential_window(path, p, first, proposal);
        const double before = fresh().total_action(path);
        auto apply = [&](const std::vector<double>& values) {
            for (std::size_t i = 0; i < count; ++i)
                for (std::size_t d = 0; d < dim; ++d) path.pos(p, d, (first + i) % M) = values[i * dim + d];
        };
        apply(proposal);
        const double after = fresh().total_action(path);
        INFO("step " << s << " count " << count << " delta " << delta << " reference "
                     << after - before);
        if (std::isfinite(after)) {
            REQUIRE(std::isfinite(delta));
            // Sherman–Morrison inverses agree with fresh LU to rounding; a
            // missing or doubled link term would show at 1e-3 and above.
            CHECK(delta == Approx(after - before).margin(1e-8).epsilon(1e-7));
            ++finite;
        } else {
            CHECK(delta == kInf);
        }
        if (std::isfinite(after) && rng() % 2 == 0) {
            ++accepted;  // keep the move; the incremental cache must follow
            path.rebuild_tables();
        } else {
            apply(saved);
        }
    }
    CHECK(finite > 150);
    CHECK(accepted > 50);
    CHECK(restriction.gradient_evaluations() > 0);
}

TEST_CASE("The node distance is the distance to a planar node and matches ln|det| gradients",
          "[pimc][fixed-node][image]") {
    // Two nearly flat Gaussians at (+-1, 0): det ∝ sinh(2 alpha (x1 - x2)), whose
    // node x1 = x2 lies at mass-scaled distance |x1 - x2| / sqrt(1/m1 + 1/m2).
    const double alpha = 1e-4;
    const Orbitals two{{1.0, 0.0, -1.0, 0.0}, alpha, SimulationCell{}, 2};
    PathSet path(3, 2, 2, 0.1);
    path.pos(0, 0, 1) = 0.7;  path.pos(0, 1, 1) = 0.2;
    path.pos(1, 0, 1) = 0.25; path.pos(1, 1, 1) = -0.4;
    for (const auto& [m1, m2] : {std::pair{1.0, 1.0}, std::pair{2.0, 0.5}}) {
        auto restriction = one_block(two, Crossing::image, {1.0 / m1, 1.0 / m1, 1.0 / m2, 1.0 / m2});
        const double exact = 0.45 / std::sqrt(1.0 / m1 + 1.0 / m2);
        CHECK(restriction.node_distance(path, 0, 1) == Approx(exact).epsilon(1e-3));
    }

    // Four orbitals: 1/d^2 = sum_j |grad_j ln|f||^2 / m_j by central differences,
    // f = det D / prod_j max_i |phi_i(r_j)|, built here without the restriction's
    // algebra (no particle sits where two orbitals tie, so f is smooth here).
    const Orbitals orb{{0.0, 0.0, 1.0, 0.2, -0.8, 0.9, 0.3, -1.1}, 0.7, SimulationCell{}, 2};
    PathSet four(2, 4, 2, 0.1);
    fill_near_centers(four, orb, 0.3, 5);
    const std::vector<double> inv_mass{1.0, 1.0, 0.5, 0.5, 2.0, 2.0, 1.0, 0.25};
    auto restriction = one_block(orb, Crossing::image, inv_mass);
    double grad2 = 0.0;
    const double h = 1e-5;
    for (std::size_t p = 0; p < 4; ++p)
        for (std::size_t d = 0; d < 2; ++d) {
            double r[2] = {four.pos(p, 0, 0), four.pos(p, 1, 0)};
            r[d] += h;
            const double up = reference_log_scaled_det(four, 0, orb, p, r);
            r[d] -= 2 * h;
            const double down = reference_log_scaled_det(four, 0, orb, p, r);
            const double g = (up - down) / (2 * h);
            grad2 += g * g * inv_mass[p * 2 + d];
        }
    CHECK(restriction.node_distance(four, 0, 0) == Approx(1.0 / std::sqrt(grad2)).epsilon(1e-6));
}

TEST_CASE("Row normalization removes the orbital envelope from the node distance",
          "[pimc][fixed-node][image]") {
    // Two narrow Gaussians at (+-1, 0).  For any alpha the node of det is the
    // plane x1 = x2, so displacing the particles along y leaves the true
    // distance unchanged; it only moves them down the envelope exp(-alpha y^2),
    // a common factor of each row, which the row's largest orbital divides out
    // exactly.
    // Without it, |grad ln det| would gain 2 alpha |y_j| per particle (here
    // ~5x the node term) and the image action would see a node that is not
    // there.
    const double alpha = 2.0;
    const Orbitals two{{1.0, 0.0, -1.0, 0.0}, alpha, SimulationCell{}, 2};
    PathSet path(2, 2, 2, 0.1);
    path.pos(0, 0, 0) = 0.3;  path.pos(1, 0, 0) = -0.2;
    path.pos(0, 0, 1) = 0.3;  path.pos(1, 0, 1) = -0.2;
    path.pos(0, 1, 1) = 0.6;  path.pos(1, 1, 1) = -0.5;
    path.rebuild_tables();
    auto restriction = one_block(two, Crossing::image);
    const double on_axis = restriction.node_distance(path, 0, 0);
    CHECK(restriction.node_distance(path, 0, 1) == Approx(on_axis).epsilon(1e-12));

    // Closed form on the axis with x1 > 0 > x2 (each particle nearest its own
    // center, so m_j is its own orbital), D = x1 - x2:
    //   f = 1 - e^{-4 alpha D},   d = f / |grad f| = (e^{4 alpha D} - 1) / (4 alpha sqrt 2).
    const auto analytic = [alpha](double x1, double x2) {
        return std::expm1(4.0 * alpha * (x1 - x2)) / (4.0 * alpha * std::sqrt(2.0));
    };
    CHECK(on_axis == Approx(analytic(0.3, -0.2)).epsilon(1e-10));
    // A first-order estimate: f bends over within 1/(4 alpha) of this planar
    // node, so the estimate overshoots the true |x1 - x2|/sqrt(2) (13x at
    // 4 alpha D = 4; the header's narrow-orbital limitation) and converges to
    // it at the node.
    CHECK(on_axis > 13.0 * 0.5 / std::sqrt(2.0));
    path.pos(0, 0, 0) = 0.0025;  path.pos(1, 0, 0) = -0.0025;
    path.rebuild_tables();
    const double near = restriction.node_distance(path, 0, 0);
    CHECK(near == Approx(analytic(0.0025, -0.0025)).epsilon(1e-8));
    CHECK(near == Approx(0.005 / std::sqrt(2.0)).epsilon(0.025));  // 1 + 2 alpha D
}

TEST_CASE("The image action's thermal derivative is (1/M) dS/dtau",
          "[pimc][fixed-node][image]") {
    const Orbitals orb{{0.0, 0.0, 1.0, 0.2, -0.8, 0.9, 0.3, -1.1}, 0.7, SimulationCell{}, 2};
    const std::size_t M = 6;
    const double tau = 0.3, h = 1e-6;
    PathSet path(M, 4, 2, tau);
    fill_near_centers(path, orb, 0.25, 8);
    auto at_tau = [&](double t) {
        PathSet copy(M, 4, 2, t);
        std::copy(path.positions().begin(), path.positions().end(), copy.positions().begin());
        copy.rebuild_tables();
        return one_block(orb, Crossing::image).total_action(copy);
    };
    const double dS_dtau = (at_tau(tau + h) - at_tau(tau - h)) / (2 * h);
    REQUIRE(std::abs(dS_dtau) > 1e-3);  // links near the node: the term matters
    auto restriction = one_block(orb, Crossing::image);
    CHECK(restriction.beta_derivative(path) == Approx(dS_dtau / M).epsilon(1e-5));
    const auto thermo = restriction.measure_thermo(path);
    CHECK(thermo.kinetic == thermo.energy);
    CHECK(thermo.potential == 0.0);
}

TEST_CASE("A cloned restriction rebuilds its cache and decides identically",
          "[pimc][fixed-node]") {
    const Orbitals orb{{0.0, 0.0, 1.0, 0.2, -0.8, 0.9, 0.3, -1.1}, 0.7, SimulationCell{}, 2};
    PathSet path(4, orb.size(), 2, 0.1);
    fill_near_centers(path, orb, 0.2, 7);
    auto original = one_block(orb);
    (void)random_walk(path, original, orb, 0.5, 500, 0, 13);
    auto image = one_block(orb, Crossing::image);
    for (FixedNodeRestriction* restriction : {&original, &image}) {
        auto copy = restriction->clone();
        CHECK(copy->restricted_blocks() == restriction->restricted_blocks());
        std::mt19937_64 rng(17);
        std::normal_distribution<double> gauss(0.0, 0.6);
        for (int s = 0; s < 300; ++s) {
            const std::size_t p = rng() % orb.size(), k = rng() % path.num_slices();
            const double proposal[2] = {path.pos(p, 0, k) + gauss(rng), path.pos(p, 1, k) + gauss(rng)};
            CHECK(copy->delta_potential_particle(path, p, k, proposal) ==
                  restriction->delta_potential_particle(path, p, k, proposal));
        }
    }
}

TEST_CASE("Construction refuses inconsistent blocks and a coarse periodic fold",
          "[pimc][fixed-node]") {
    const Orbitals orb{{0.0, 0.0, 1.0, 0.0}, 1.0, SimulationCell{}, 2};
    auto blocks_of = [&](std::size_t first, std::size_t last) {
        std::vector<FixedNodeRestriction::Block> blocks;
        blocks.push_back({first, last, orb.set()});
        return blocks;
    };
    auto ones = [](std::size_t np) { return std::vector<double>(np * 2, 1.0); };
    CHECK_THROWS_WITH(FixedNodeRestriction(blocks_of(0, 3), 3, 2, ones(3), Crossing::image),
                      ContainsSubstring("one orbital per particle"));
    CHECK_THROWS_WITH(FixedNodeRestriction(blocks_of(1, 3), 2, 2, ones(2), Crossing::image),
                      ContainsSubstring("outside"));
    {
        auto blocks = blocks_of(0, 2);
        blocks.push_back({1, 3, orb.set()});
        CHECK_THROWS_WITH(FixedNodeRestriction(std::move(blocks), 3, 2, ones(3), Crossing::image),
                          ContainsSubstring("overlap"));
    }
    CHECK_THROWS_WITH(FixedNodeRestriction({}, 2, 2, ones(2), Crossing::image),
                      ContainsSubstring("no determinant block"));
    CHECK_THROWS_WITH(FixedNodeRestriction(blocks_of(0, 2), 2, 2, {1.0, 1.0, -1.0, 1.0},
                                           Crossing::image),
                      ContainsSubstring("inverse mass"));
    CHECK_THROWS_WITH(FixedNodeRestriction(blocks_of(0, 2), 2, 2, {1.0, 1.0}, Crossing::image),
                      ContainsSubstring("inverse mass"));

    // Broad orbitals in a 7-particle cell: images at the cell's inradius are
    // far above 1e-6 of the peak.
    const double area = 7.0 * std::numbers::pi;
    const double a = std::sqrt(2.0 * area / std::sqrt(3.0));
    const SimulationCell cell(2, std::vector<double>{a, 0.0, 0.5 * a, 0.5 * std::sqrt(3.0) * a});
    const Orbitals broad{commensurate_triangular_sites(cell, 7), 0.3, cell, 2};
    CHECK(broad.set()->periodic_fold_bound() ==
          std::exp(-0.3 * cell.ws_radius() * cell.ws_radius()));
    std::vector<FixedNodeRestriction::Block> blocks;
    blocks.push_back({0, 7, broad.set()});
    CHECK_THROWS_WITH(FixedNodeRestriction(std::move(blocks), 7, 2, ones(7), Crossing::image),
                      ContainsSubstring("periodic images"));
    CHECK(orb.set()->periodic_fold_bound() == 0.0);  // open cell: exact
}

TEST_CASE("Where the nodes never act, fixed_node reproduces the boltzmannon chain bit for bit",
          "[pimc][fixed-node]") {
    // One-particle blocks, beads only: a single Gaussian never changes sign.
    const std::vector groups{group("g", 0, 2, 2)};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(groups, 1.0));
    PimcConfig cfg = short_config(2.0, 0.25, 41);
    cfg.equilibration_blocks = 20;
    cfg.accumulation_blocks = 60;
    add_move(cfg, "bisection", 2, 0.0);
    add_move(cfg, "centroid", 0, 0.5);
    add_move(cfg, "single_slice", 0, 0.4);

    PimcConfig distinguishable = cfg;
    distinguishable.statistics = "boltzmannon";
    Action plain = restricted_action(groups, ham, 2, nullptr);
    Pimc reference(distinguishable, groups, ham, SimulationCell{}, 2, 2, nullptr, &plain);
    reference.run();

    std::vector<FixedNodeRestriction::Block> blocks;
    blocks.push_back({0, 1, Orbitals{{0.3, 0.0}, 1.0, SimulationCell{}, 2}.set()});
    blocks.push_back({1, 2, Orbitals{{-0.3, 0.0}, 1.0, SimulationCell{}, 2}.set()});
    PimcConfig fixed = cfg;
    fixed.statistics = "fixed_node";
    Action nodes = restricted_action(groups, ham, 2,
        std::make_unique<FixedNodeRestriction>(std::move(blocks), 2, 2,
                                               std::vector<double>(4, 1.0), Crossing::beads_only));
    Pimc restricted(fixed, groups, ham, SimulationCell{}, 2, 2, nullptr, &nodes);
    restricted.run();

    CHECK(restricted.fixed_node_run());
    CHECK_FALSE(restricted.quantum_statistics());
    REQUIRE(restricted.block_energies().size() == reference.block_energies().size());
    CHECK(std::equal(restricted.block_energies().begin(), restricted.block_energies().end(),
                     reference.block_energies().begin()));
}

TEST_CASE("Two trapped fermions with the node x1 = x2 match the exact finite-M energy",
          "[pimc][fixed-node][physics]") {
    const std::vector groups{group("g", 0, 2, 2)};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(groups, 1.0));
    PimcConfig cfg = short_config(4.0, 0.25, 8123);
    cfg.statistics = "fixed_node";
    cfg.exchange_classes = {ExchangeClass{"g", 0, "fermions", {0, 1}}};
    add_move(cfg, "bisection", 3, 0.0);
    add_move(cfg, "bisection", 2, 0.0);
    add_move(cfg, "centroid", 0, 0.6);

    SECTION("nodes checked at beads only") {
        // Two determinants with different orbitals but the same nodal plane:
        // the chains must be identical, not merely statistically equal.
        Action narrow = restricted_action(groups, ham, 2, two_center_nodes(0.7, 0.4));
        Pimc first(cfg, groups, ham, SimulationCell{}, 2, 2, nullptr, &narrow);
        first.run();
        Action wide = restricted_action(groups, ham, 2, two_center_nodes(1.3, 2.0));
        Pimc second(cfg, groups, ham, SimulationCell{}, 2, 2, nullptr, &wide);
        second.run();
        INFO(first.summary());
        INFO("E = " << first.energy() << " +/- " << first.energy_error() << " exact "
                    << kBeadsOnlyEnergy);
        CHECK(std::abs(first.energy() - kBeadsOnlyEnergy) <= tolerance(first.energy_error()));
        CHECK(first.energy_error() < kBeadsOnlyEnergy / 50.0);
        CHECK_THAT(first.summary(), ContainsSubstring("fixed_node"));
        CHECK(std::equal(first.block_energies().begin(), first.block_energies().end(),
                         second.block_energies().begin()));
    }
    SECTION("with the image action") {
        // Nearly flat orbitals make |psi|/|grad psi| the exact planar distance,
        // so the reference's exact image factor applies.
        Action image = restricted_action(groups, ham, 2, two_center_nodes(1.0, 1e-4, Crossing::image));
        Pimc pimc(cfg, groups, ham, SimulationCell{}, 2, 2, nullptr, &image);
        pimc.run();
        INFO(pimc.summary());
        INFO("E = " << pimc.energy() << " +/- " << pimc.energy_error() << " exact "
                    << kImageEnergy);
        CHECK(std::abs(pimc.energy() - kImageEnergy) <= tolerance(pimc.energy_error()));
        CHECK(pimc.energy_error() < kImageEnergy / 50.0);
        CHECK_THAT(pimc.summary(), ContainsSubstring("image action"));
    }
}

TEST_CASE("The driver refuses fixed_node runs it cannot sample", "[pimc][fixed-node]") {
    const std::vector groups{group("g", 0, 2, 2)};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(groups, 1.0));
    PimcConfig cfg = short_config(1.0, 0.25, 3);
    add_move(cfg, "bisection", 2, 0.0);
    cfg.exchange_classes = {ExchangeClass{"g", 0, "fermions", {0, 1}}};
    Action nodes = restricted_action(groups, ham, 2, two_center_nodes(0.5, 1.0, Crossing::image));
    Action plain = restricted_action(groups, ham, 2, nullptr);

    PimcConfig fixed = cfg;
    fixed.statistics = "fixed_node";
    CHECK_NOTHROW(Pimc(fixed, groups, ham, SimulationCell{}, 2, 2, nullptr, &nodes));
    // fixed_node without nodes, and nodes without fixed_node.
    CHECK_THROWS_WITH(Pimc(fixed, groups, ham, SimulationCell{}, 2, 2, nullptr, &plain),
                      ContainsSubstring("needs the fixed-node restriction"));
    PimcConfig distinguishable = cfg;
    distinguishable.statistics = "boltzmannon";
    CHECK_THROWS_WITH(Pimc(distinguishable, groups, ham, SimulationCell{}, 2, 2, nullptr, &nodes),
                      ContainsSubstring("only 'statistics: fixed_node'"));
    // Exchange moves cross the node.
    PimcConfig permuting = fixed;
    add_move(permuting, "permutation", 2, 0.0);
    CHECK_THROWS_WITH(Pimc(permuting, groups, ham, SimulationCell{}, 2, 2, nullptr, &nodes),
                      ContainsSubstring("identity permutation"));
    // Bosons have no nodes; a fermion class must be one determinant block.
    PimcConfig bosons = fixed;
    bosons.exchange_classes[0].statistics = "bosons";
    CHECK_THROWS_WITH(Pimc(bosons, groups, ham, SimulationCell{}, 2, 2, nullptr, &nodes),
                      ContainsSubstring("no nodes for the bosons"));
    {
        const std::vector three{group("g", 0, 3, 2)};
        Hamiltonian ham3;
        ham3.add_component(std::make_unique<HarmonicTrap>(three, 1.0));
        PimcConfig uncovered = fixed;
        uncovered.exchange_classes = {ExchangeClass{"g", 0, "fermions", {0, 1, 2}}};
        Action partial = restricted_action(three, ham3, 2,
            std::make_unique<FixedNodeRestriction>([] {
                std::vector<FixedNodeRestriction::Block> blocks;
                blocks.push_back({0, 2, Orbitals{{0.5, 0.0, -0.5, 0.0}, 1.0, SimulationCell{}, 2}.set()});
                return blocks;
            }(), 3, 2, std::vector<double>(6, 1.0), Crossing::image));
        CHECK_THROWS_WITH(Pimc(uncovered, three, ham3, SimulationCell{}, 2, 3, nullptr, &partial),
                          ContainsSubstring("not one determinant block"));
    }
    // Nodes on particles that are not one fermion class (here: no classes).
    PimcConfig unclassed = fixed;
    unclassed.exchange_classes.clear();
    CHECK_THROWS_WITH(Pimc(unclassed, groups, ham, SimulationCell{}, 2, 2, nullptr, &nodes),
                      ContainsSubstring("is not one fermion exchange class"));
    // The virials would need the singular nodal force.
    PimcConfig virial = fixed;
    virial.energy = "centroid_virial";
    CHECK_THROWS_WITH(Pimc(virial, groups, ham, SimulationCell{}, 2, 2, nullptr, &nodes),
                      ContainsSubstring("singular force"));
}

TEST_CASE("YAML nodes come from the wavefunction block and reproduce the exact energy",
          "[pimc][fixed-node][physics][app]") {
    // Nearly flat orbitals: the image action is exact for this planar node.
    const std::string yaml = R"(project: {name: fixed_node_dot, seed: 8123}
system:
  lattice: {type: open, dimension: 2, L: 1.0}
  particles:
    - {name: e, group: electrons, type: fermions, mass: 1.0, charge: 0, up: 2, down: 0}
wavefunction:
  - slater:
      group: electrons
      orbitals: {type: gaussian, centers: [[1.0, 0.0], [-1.0, 0.0]], alpha: 1.0e-4}
hamiltonian:
  terms:
    - {type: harmonic_trap, omega: 1.0}
action:
  - primitive: {}
  - nodes: wavefunction
qmc:
  - method: pimc
    threads: 2
    samples: 32
    beta: 4.0
    time_step: 0.25
    statistics: fixed_node
    equilibration_blocks: 200
    accumulation_blocks: 800
    steps_per_block: 20
    sub_steps: 2
    moves:
      - bisection: {levels: 3}
      - bisection: {levels: 2}
      - centroid: {step: 0.6}
)";
    const io::Config config = io::parse_config(yaml);
    REQUIRE(config.action.size() == 2);
    CHECK(config.action[1].component == "nodes");
    CHECK(config.action[1].form == "wavefunction");
    CHECK(config.pimc[0].statistics == "fixed_node");
    App app(config);
    auto pack = app.build_drivers();
    auto* pimc = dynamic_cast<Pimc*>(pack.drivers[0].get());
    REQUIRE(pimc != nullptr);
    pimc->run();
    INFO(pimc->summary());
    CHECK(pimc->fixed_node_run());
    CHECK_THAT(pimc->summary(), ContainsSubstring("image action"));
    CHECK(std::abs(pimc->energy() - kImageEnergy) <= tolerance(pimc->energy_error()));

    // The run's config snapshot replays the same nodes and statistics.
    {
        const auto directory = std::filesystem::temp_directory_path() / "alea_fixed_node_snapshot";
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory / "run");
        const auto source = directory / "source.yaml";
        std::ofstream(source) << yaml;
        io::write_config_snapshot(source.string(), {}, directory / "run", 8123);
        const io::Config replay = io::parse_config_file((directory / "run/source.yaml").string());
        REQUIRE(replay.action.size() == 2);
        CHECK(replay.action[1].component == "nodes");
        CHECK(replay.action[1].form == "wavefunction");
        CHECK(replay.pimc[0].statistics == "fixed_node");
        std::filesystem::remove_all(directory);
    }

    // Refusals at parse and build time.
    auto replaced = [&](const std::string& from, const std::string& to) {
        std::string text = yaml;
        text.replace(text.find(from), from.size(), to);
        return text;
    };
    CHECK_THROWS_WITH(io::parse_config(replaced("- nodes: wavefunction", "- nodes: free_particle")),
                      ContainsSubstring("unknown source"));
    CHECK_THROWS_WITH(io::parse_config(replaced("- nodes: wavefunction",
                                                "- nodes: wavefunction\n  - nodes: wavefunction")),
                      ContainsSubstring("nodes component given twice"));
    CHECK_THROWS_WITH(io::parse_config(replaced("statistics: fixed_node", "statistics: restricted")),
                      ContainsSubstring("unknown statistics"));
    {
        io::Config no_slater = io::parse_config(yaml);
        no_slater.wavefunction.clear();
        App bad(no_slater);
        CHECK_THROWS_WITH(bad.build_drivers(), ContainsSubstring("no slater item"));
    }
}

// Where the restriction's time goes.  Hidden: run explicitly on an idle machine,
//   test_pimc_physics "[fixed-node-bench]" -s
// Crystal at r_s = 40 with Drummond & Needs's exponent, 15-bead windows as in a
// level-4 bisection, one thread.  For each (N, M, crossing) it reports ns per
// proposed bead for proposals alone and for proposals that are accepted (the
// next call then syncs the slice).  The cache holds M n^2 (1 + dim) doubles; if
// the cost per bead grows with M at fixed N, the work is memory-bound (the
// slice's arrays come from DRAM), if it stays flat it is compute-bound.
TEST_CASE("Fixed-node cost per proposed bead versus cache size", "[.][fixed-node-bench]") {
    const double rs = 40.0, tau = std::pow(rs, 1.5) / 64.0;
    const double spacing = std::sqrt(2.0 * std::numbers::pi / std::sqrt(3.0)) * rs;
    const std::size_t window = 15;
    for (const std::size_t n : {std::size_t{37}, std::size_t{127}}) {
        const double a = std::sqrt(2.0 * static_cast<double>(n) * std::numbers::pi / std::sqrt(3.0)) * rs;
        const SimulationCell cell(2, std::vector<double>{a, 0.0, 0.5 * a, 0.5 * std::sqrt(3.0) * a});
        const Orbitals orb{commensurate_triangular_sites(cell, n), 0.071 * std::pow(rs, -1.5), cell, 2};
        for (const std::size_t M : {std::size_t{16}, std::size_t{128}, std::size_t{2048}}) {
            // Nearly collapsed rings: one displacement per particle on every
            // slice, so all slices start in one nodal pocket.
            PathSet path(M, n, 2, tau, cell);
            {
                std::mt19937_64 rng(101);
                std::normal_distribution<double> shift(0.0, 0.15 * spacing), noise(0.0, 0.01 * spacing);
                for (std::size_t p = 0; p < n; ++p)
                    for (std::size_t d = 0; d < 2; ++d) {
                        const double centre = orb.centers[p * 2 + d] + shift(rng);
                        for (std::size_t k = 0; k < M; ++k) path.pos(p, d, k) = centre + noise(rng);
                    }
                path.rebuild_tables();
            }
            for (const Crossing crossing : {Crossing::beads_only, Crossing::image}) {
                auto restriction = one_block(orb, crossing);
                REQUIRE(std::isfinite(restriction.total_action(path)));
                std::mt19937_64 rng(5);
                std::normal_distribution<double> gauss(0.0, 0.05 * spacing);
                std::vector<double> proposal(window * 2);
                double ns[2] = {0.0, 0.0};
                for (const bool accept : {false, true}) {
                    std::size_t windows = 0;
                    const auto start = std::chrono::steady_clock::now();
                    double elapsed = 0.0;
                    while (elapsed < 1.0 || windows < 200) {
                        const std::size_t p = rng() % n, first = rng() % M;
                        for (std::size_t i = 0; i < window; ++i)
                            for (std::size_t d = 0; d < 2; ++d)
                                proposal[i * 2 + d] = path.pos(p, d, (first + i) % M) + gauss(rng);
                        const double delta = restriction.delta_potential_window(path, p, first, proposal);
                        if (accept && std::isfinite(delta))
                            for (std::size_t i = 0; i < window; ++i)
                                for (std::size_t d = 0; d < 2; ++d)
                                    path.pos(p, d, (first + i) % M) = proposal[i * 2 + d];
                        ++windows;
                        elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                    }
                    ns[accept] = 1e9 * elapsed / static_cast<double>(windows * window);
                }
                const double cache_mb = static_cast<double>(M * n * n * (crossing == Crossing::image ? 3 : 1)) * 8.0 / 1e6;
                std::cout << "N=" << n << " M=" << M << " "
                          << (crossing == Crossing::image ? "image     " : "beads_only")
                          << " cache=" << cache_mb << " MB  ns/bead: proposals " << ns[0]
                          << "  accepted " << ns[1] << "  (rebuilds " << restriction.rebuilds()
                          << ", updates " << restriction.updates() << ")\n";
            }
        }
    }
}
