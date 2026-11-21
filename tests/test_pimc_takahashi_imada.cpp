// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Takahashi-Imada potential action (takahashi_imada_action.h), against oracles
// that do not use its force code: the slice action from central differences
// of the sampled V, deltas and bead actions from whole-slice evaluations, and
// the energy pieces from numerical derivatives of S_V in beta and in a
// coupling g of V (Hellmann-Feynman).
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>
#include <numbers>
#include <vector>

#include "hamiltonian/coulomb.h"
#include "hamiltonian/hamiltonian.h"
#include "hamiltonian/harmonic_trap.h"
#include "hamiltonian/inverse_power.h"
#include "hamiltonian/moire.h"
#include "pimc/action.h"
#include "pimc/path_set.h"
#include "pimc/primitive_action.h"
#include "pimc/takahashi_imada_action.h"
#include "population/simulation_cell.h"

namespace {
using namespace alea;
using Catch::Approx;
constexpr std::size_t none = HamiltonianComponent::npos;

ParticleGroup group(std::size_t first, std::size_t last, std::vector<double> mass,
                    double charge = 0.0) {
    ParticleGroup g;
    g.name = "bosons";
    g.first = first;
    g.last = last;
    g.count = last - first;
    g.charge = charge;
    g.mass = mass;
    for (double m : mass) g.inv_mass.push_back(1.0 / m);
    return g;
}

SimulationCell hexagonal_cell() {
    return SimulationCell(2, std::vector<double>{3.0, 0.0, 1.5, 1.5 * std::numbers::sqrt3});
}

// The study's terms (moire + dipolar r^-3) on rows [0, n), couplings scaled by g.
Hamiltonian dipolar_moire(std::size_t n, const SimulationCell& cell, double g = 1.0) {
    Hamiltonian ham;
    ham.add_component(std::make_unique<MoireTerm>(
        2, 0.3 * g, 4.296264364086565 * g, 0.2617993877991494,
        moire_reciprocal_vectors(1.0, 0.5 * std::numbers::pi, 2),
        std::vector<std::pair<std::size_t, std::size_t>>{{0, n}}));
    ham.add_component(std::make_unique<InversePowerTerm>(
        group(0, n, {1.0, 1.0}), group(0, n, {1.0, 1.0}), 5.439295845011993 * g, 3.0, cell));
    return ham;
}

// Trap + Coulomb, open cell, anisotropic mass (1, 2.5), couplings scaled by g.
const std::vector<double> kAnisotropic = {1.0, 2.5};
Hamiltonian trapped_coulomb(std::size_t n, double g = 1.0) {
    Hamiltonian ham;
    const auto a = group(0, n, kAnisotropic, 1.0);
    ham.add_component(std::make_unique<HarmonicTrap>(std::vector<ParticleGroup>{a}, 1.3 * g,
                                                     std::vector<double>{0.1, -0.2}));
    ham.add_component(std::make_unique<CoulombTerm>(a, a, 2.0 / g));
    return ham;
}

// Rows apart from each other, a little different on every slice.
void place(PathSet& path, double spacing) {
    for (std::size_t p = 0; p < path.capacity(); ++p)
        for (std::size_t k = 0; k < path.num_slices(); ++k) {
            double r[2] = {0.4 + spacing * static_cast<double>(p) + 0.03 * std::sin(1.0 + k + p),
                           0.3 + 0.6 * spacing * static_cast<double>(p) + 0.02 * std::cos(2.0 * k - p)};
            path.cell().wrap(r);
            path.pos(p, 0, k) = r[0];
            path.pos(p, 1, k) = r[1];
        }
    path.rebuild_tables();
}

// W(R_k) = (1/24) sum (1/m) (dV/dr)^2 with dV/dr from central differences of
// the sampled V (Richardson-extrapolated from steps h and h/2).
double numeric_correction(const Hamiltonian& ham, PathSet& path, std::size_t k,
                          const std::vector<double>& inv_mass) {
    std::vector<double> v(path.num_slices());
    auto V = [&]() {
        ham.evaluate_potential(path, v, k, k + 1);
        return v[k];
    };
    double sum = 0.0;
    for (std::size_t p = 0; p < path.n_particles(); ++p)
        for (std::size_t d = 0; d < path.dim(); ++d) {
            const double saved = path.pos(p, d, k);
            auto derivative = [&](double h) {
                path.pos(p, d, k) = saved + h;
                path.update_particle_tables(p, k, k + 1);
                const double plus = V();
                path.pos(p, d, k) = saved - h;
                path.update_particle_tables(p, k, k + 1);
                const double minus = V();
                path.pos(p, d, k) = saved;
                path.update_particle_tables(p, k, k + 1);
                return (plus - minus) / (2 * h);
            };
            const double h = 1e-4;
            const double g = (4 * derivative(h / 2) - derivative(h)) / 3;
            sum += inv_mass[d] * g * g;
        }
    return sum / 24.0;
}

struct System {
    const char* name;
    SimulationCell cell;
    std::vector<double> mass;
    Hamiltonian (*make)(std::size_t, const SimulationCell&, double);
    double spacing;
};

Hamiltonian make_dipolar(std::size_t n, const SimulationCell& cell, double g) {
    return dipolar_moire(n, cell, g);
}
Hamiltonian make_trapped(std::size_t n, const SimulationCell&, double g) {
    return trapped_coulomb(n, g);
}

std::vector<System> systems() {
    return {{"periodic moire + dipolar", hexagonal_cell(), {1.0, 1.0}, make_dipolar, 0.83},
            {"open trap + Coulomb, anisotropic mass", SimulationCell(2), kAnisotropic, make_trapped, 0.71}};
}
}  // namespace

TEST_CASE("The Takahashi-Imada slice action is tau V + (tau^3/24) sum |grad V|^2 / m",
          "[pimc][takahashi_imada]") {
    for (const auto& s : systems()) {
        DYNAMIC_SECTION(s.name) {
            const std::size_t n = 4, M = 6;
            const double tau = 0.07;
            PathSet path(M, n, 2, tau, s.cell);
            place(path, s.spacing);
            const Hamiltonian ham = s.make(n, s.cell, 1.0);
            const auto g = group(0, n, s.mass);
            TakahashiImadaPotentialAction ti(std::span<const ParticleGroup>(&g, 1), ham, 2);
            ti.prepare(M);
            std::vector<double> inv_mass = {1.0 / s.mass[0], 1.0 / s.mass[1]};
            std::vector<double> v(M);
            ham.evaluate_potential(path, v);
            double total = 0.0;
            for (std::size_t k = 0; k < M; ++k) {
                const double w = numeric_correction(ham, path, k, inv_mass);
                CAPTURE(k, w);
                CHECK(w > 0.0);
                CHECK(ti.slice_correction(path, k) == Approx(w).epsilon(1e-7));
                const double slice = tau * v[k] + tau * tau * tau * w;
                CHECK(ti.potential_action_slice(path, k) == Approx(slice).epsilon(1e-10));
                total += slice;
            }
            CHECK(ti.total_action(path) == Approx(total).epsilon(1e-10));
            CHECK(ti.total_potential(path) == Approx(v[0] + v[1] + v[2] + v[3] + v[4] + v[5]));
        }
    }
}

TEST_CASE("A Takahashi-Imada bead delta is the change of the total action and leaves the path",
          "[pimc][takahashi_imada]") {
    for (const auto& s : systems()) {
        DYNAMIC_SECTION(s.name) {
            const std::size_t n = 4, M = 5;
            PathSet path(M, n, 2, 0.05, s.cell);
            place(path, s.spacing);
            const Hamiltonian ham = s.make(n, s.cell, 1.0);
            const auto g = group(0, n, s.mass);
            Action action;
            action.add_component(std::make_unique<PrimitiveKineticAction>(
                std::span<const ParticleGroup>(&g, 1), 2));
            action.add_component(std::make_unique<TakahashiImadaPotentialAction>(
                std::span<const ParticleGroup>(&g, 1), ham, 2));
            REQUIRE(action.slice_local_deltas());
            REQUIRE(action.supports_free_bridge());
            const std::vector<double> before(path.positions().begin(), path.positions().end());
            for (std::size_t p = 0; p < n; ++p)
                for (std::size_t k = 0; k < M; k += 2) {
                    const std::vector<double> proposed = {path.pos(p, 0, k) + 0.13 - 0.05 * p,
                                                          path.pos(p, 1, k) - 0.09 + 0.02 * k};
                    CAPTURE(p, k);
                    // The composite's potential delta (the springs are the
                    // moves' business) against the whole TI action of the
                    // path with the bead moved, built from scratch.
                    const double delta = action.delta_potential_particle(path, p, k, proposed);
                    CHECK(std::vector<double>(path.positions().begin(), path.positions().end()) == before);
                    double wrapped[2] = {proposed[0], proposed[1]};
                    path.cell().wrap(wrapped);
                    PathSet moved(M, n, 2, 0.05, s.cell);
                    for (std::size_t i = 0; i < path.positions().size(); ++i)
                        moved.positions()[i] = path.positions()[i];
                    moved.pos(p, 0, k) = wrapped[0];
                    moved.pos(p, 1, k) = wrapped[1];
                    moved.rebuild_tables();
                    const double ti_old = action.components()[1]->total_action(path);
                    const double ti_new = action.components()[1]->total_action(moved);
                    CHECK(delta == Approx(ti_new - ti_old).epsilon(1e-9).margin(1e-12));
                }
        }
    }
}

TEST_CASE("A Takahashi-Imada bead action is the change of the slice action when the bead joins",
          "[pimc][takahashi_imada][grand_canonical]") {
    // A grand-canonical slice: capacity rows, the first n_rows in use, at most
    // one of them missing (skip).  The oracle builds the configurations R and
    // R + {r} as their own canonical paths and takes the slice actions.
    const SimulationCell cell = hexagonal_cell();
    const std::size_t capacity = 6, M = 4;
    const double tau = 0.06;
    PathSet path(M, capacity, 2, tau, cell);
    place(path, 0.83);
    const Hamiltonian worm_ham = dipolar_moire(capacity, cell);
    const auto g = group(0, capacity, {1.0, 1.0});
    const TakahashiImadaPotentialAction ti(std::span<const ParticleGroup>(&g, 1), worm_ham, 2);
    REQUIRE(ti.supports_bead_action());
    auto slice_action = [&](const std::vector<std::vector<double>>& rows) {
        const std::size_t n = rows.size();
        PathSet sub(2, n, 2, tau, cell);
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t k = 0; k < 2; ++k) {
                sub.pos(j, 0, k) = rows[j][0];
                sub.pos(j, 1, k) = rows[j][1];
            }
        sub.rebuild_tables();
        const auto gn = group(0, n, {1.0, 1.0});
        TakahashiImadaPotentialAction canonical(std::span<const ParticleGroup>(&gn, 1),
                                                dipolar_moire(n, cell), 2);
        canonical.prepare(2);
        return canonical.potential_action_slice(sub, 0);
    };
    struct Case { std::size_t p, n_rows, skip; };
    for (const Case c : {Case{2, 6, none}, Case{5, 5, none}, Case{0, 5, 3}, Case{4, 6, 1}}) {
        CAPTURE(c.p, c.n_rows, c.skip);
        const std::size_t k = 1;
        std::vector<double> bead = {1.37, 0.41};
        std::vector<std::vector<double>> rows;
        for (std::size_t j = 0; j < c.n_rows; ++j)
            if (j != c.p && j != c.skip) rows.push_back({path.pos(j, 0, k), path.pos(j, 1, k)});
        const double without = slice_action(rows);
        rows.push_back(bead);
        const double with = slice_action(rows);
        CHECK(ti.bead_action(path, k, c.p, bead, c.n_rows, c.skip) ==
              Approx(with - without).epsilon(1e-9));
    }
}

TEST_CASE("Takahashi-Imada energy pieces are the beta and coupling derivatives of S_V",
          "[pimc][takahashi_imada]") {
    for (const auto& s : systems()) {
        DYNAMIC_SECTION(s.name) {
            const std::size_t n = 3, M = 5;
            const double beta = 0.4, tau = beta / M;
            const auto g = group(0, n, s.mass);
            const std::span<const ParticleGroup> groups(&g, 1);
            auto S = [&](double b, double coupling) {
                PathSet path(M, n, 2, b / M, s.cell);
                place(path, s.spacing);
                TakahashiImadaPotentialAction ti(groups, s.make(n, s.cell, coupling), 2);
                ti.prepare(M);
                return ti.total_action(path);
            };
            PathSet path(M, n, 2, tau, s.cell);
            place(path, s.spacing);
            TakahashiImadaPotentialAction ti(groups, s.make(n, s.cell, 1.0), 2);
            ti.prepare(M);
            const auto m = ti.measure_thermo(path);
            const double h = 1e-4;
            // d_beta S_V at fixed M and coordinates: the whole energy piece.
            const double d_beta = (S(beta + h, 1.0) - S(beta - h, 1.0)) / (2 * h);
            CHECK(m.energy == Approx(d_beta).epsilon(1e-6));
            CHECK(ti.beta_derivative(path) == Approx(m.energy));
            // (1/beta) d_g S_V at g = 1: <V> (Hellmann-Feynman); K = E - V.
            const double d_g = (S(beta, 1.0 + h) - S(beta, 1.0 - h)) / (2 * h);
            CHECK(m.potential == Approx(d_g / beta).epsilon(1e-6));
            CHECK(m.kinetic + m.potential == Approx(m.energy).epsilon(1e-12));
            CHECK(m.kinetic > 0.0);
        }
    }
}

TEST_CASE("The Takahashi-Imada action refuses terms without forces, attractive poles and virial estimators",
          "[pimc][takahashi_imada]") {
    const auto g = group(0, 2, {1.0, 1.0});
    const std::span<const ParticleGroup> groups(&g, 1);
    struct NoForces final : HamiltonianComponent {
        std::string name() const override { return "no_forces"; }
        void evaluate_potential(const ParticleBatch&, std::span<double>, std::size_t,
                                std::size_t) const override {}
        std::unique_ptr<HamiltonianComponent> clone() const override {
            return std::make_unique<NoForces>(*this);
        }
    };
    Hamiltonian unsupported;
    unsupported.add_component(std::make_unique<NoForces>());
    CHECK_THROWS_AS(TakahashiImadaPotentialAction(groups, unsupported, 2), std::invalid_argument);

    // An electron and a hole in one plane: e^{+tau a/r} is not normalizable.
    const auto e = group(0, 1, {1.0, 1.0}, -1.0), hole = group(1, 2, {1.0, 1.0}, 1.0);
    const std::vector<ParticleGroup> pair{e, hole};
    Hamiltonian attractive;
    attractive.add_component(std::make_unique<CoulombTerm>(e, hole, 1.0));
    CHECK_THROWS_AS(TakahashiImadaPotentialAction(pair, attractive, 2), std::invalid_argument);

    const TakahashiImadaPotentialAction ti(groups, trapped_coulomb(2), 2);
    CHECK_THROWS_AS(ti.validate_open_centroid(2, 2), std::invalid_argument);
    CHECK_THROWS_AS(ti.validate_periodic_virial(hexagonal_cell(), 2, 2), std::invalid_argument);
    CHECK(ti.potential_approximation_notes().find("O(tau^2)") != std::string::npos);
}
