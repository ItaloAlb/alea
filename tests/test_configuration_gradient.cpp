// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Contracts of the per-configuration forces (HamiltonianComponent::
// configuration_gradient and bead_gradient) that the Takahashi-Imada action
// squares: the batched potential gradient as an oracle where it exists,
// central differences of the bead energy (the change of V when one row joins
// the configuration) for every row filter, and of the sampled V for the
// inverse-power term, which has no batched gradient.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "hamiltonian/coulomb.h"
#include "hamiltonian/hamiltonian.h"
#include "hamiltonian/harmonic_trap.h"
#include "hamiltonian/inverse_power.h"
#include "hamiltonian/moire.h"
#include "population/walker_set.h"

namespace {
using namespace alea;
using Catch::Approx;
constexpr std::size_t none = HamiltonianComponent::npos;

ParticleGroup make_group(std::size_t first, std::size_t last, double charge, std::size_t dim) {
    ParticleGroup g{};
    g.first = first; g.last = last; g.count = last - first; g.charge = charge;
    g.mass.assign(dim, 1.0); g.inv_mass.assign(dim, 1.0);
    return g;
}

// Generic positions, well apart and away from the Wigner-Seitz faces of the
// cells used below, so central differences do not straddle a minimum-image
// switch.
void place(ParticleBatch& batch) {
    const auto nw = batch.num_walkers(), dim = batch.dim();
    for (std::size_t p = 0; p < batch.n_particles(); ++p)
        for (std::size_t w = 0; w < nw; ++w) {
            double r[SimulationCell::kMaxDim];
            for (std::size_t d = 0; d < dim; ++d)
                r[d] = -1.3 + 0.71 * p + 0.19 * w + 0.37 * d + 0.11 * p * d + 0.05 * p * p * (d + 1);
            batch.cell().wrap(r);
            for (std::size_t d = 0; d < dim; ++d) batch.positions()[(p * dim + d) * nw + w] = r[d];
        }
    batch.dist_tables().build_all(batch.positions());
}

std::vector<double> row_position(const ParticleBatch& batch, std::size_t w, std::size_t j) {
    std::vector<double> r(batch.dim());
    for (std::size_t d = 0; d < batch.dim(); ++d)
        r[d] = batch.positions()[(j * batch.dim() + d) * batch.num_walkers() + w];
    return r;
}

// One species: moire + trap + intra-group pair terms (inverse power and a
// layered Coulomb), every term with a bead energy.
Hamiltonian one_species(std::size_t np, std::size_t dim, bool periodic) {
    const auto a = make_group(0, np, 1.0, dim);
    Hamiltonian ham;
    ham.add_component(std::make_unique<MoireTerm>(dim, 0.4, 0.9, 0.3,
        moire_reciprocal_vectors(2.0, 1.5707963267948966, dim), std::vector<std::pair<std::size_t, std::size_t>>{{0, np}}));
    if (!periodic)
        ham.add_component(std::make_unique<HarmonicTrap>(std::vector{a}, 0.6, std::vector<double>(dim, 0.1)));
    SimulationCell cell = periodic ? SimulationCell(2, std::vector<double>{4.0, 0.0, 2.0, 3.4641016151377544}) : SimulationCell(dim);
    ham.add_component(std::make_unique<InversePowerTerm>(a, a, 0.8, 3.0, cell));
    ham.add_component(std::make_unique<CoulombTerm>(a, a, 2.0, 0.4));
    return ham;
}

// d/dx of f along axis d by central differences at three steps (the
// smallest-error estimate must agree with the analytic value).
template<class F>
void check_derivative(F f, double analytic, double tolerance) {
    double best = INFINITY;
    for (double h : {1e-4, 3e-5, 1e-5}) {
        const double numeric = (f(+h) - f(-h)) / (2 * h);
        best = std::min(best, std::abs(numeric - analytic));
    }
    CAPTURE(analytic, best);
    CHECK(best <= tolerance * std::max(1.0, std::abs(analytic)));
}
}  // namespace

TEST_CASE("Configuration gradients of every row equal the batched potential gradient", "[configuration-gradient]") {
    for (std::size_t dim : {1, 2, 3}) {
        for (bool periodic : {false, true}) {
            if (periodic && dim == 3) continue;  // SimulationCell has no periodic 3-D cell
            DYNAMIC_SECTION("dimension " << dim << (periodic ? " periodic" : " open")) {
                std::vector<double> vectors = dim == 1 ? std::vector<double>{6.0}
                                                       : std::vector<double>{6.0, 0.0, 1.0, 5.0};
                SimulationCell cell = periodic ? SimulationCell(dim, vectors) : SimulationCell(dim);
                WalkerSet batch(3, 4, dim, cell);  // nw = 3, np = 4
                place(batch);
                const auto a = make_group(0, 3, -1.0, dim), b = make_group(3, 4, 2.0, dim);
                Hamiltonian ham;
                ham.add_component(std::make_unique<CoulombTerm>(a, a, 2.5));
                ham.add_component(std::make_unique<CoulombTerm>(a, b, 3.0, 0.7));
                ham.add_component(std::make_unique<MoireTerm>(dim, 1.2, 0.17, 0.4,
                    moire_reciprocal_vectors(5.0, 0.23, dim),
                    std::vector<std::pair<std::size_t, std::size_t>>{{0, 2}, {3, 4}}));
                if (!periodic)
                    ham.add_component(std::make_unique<HarmonicTrap>(std::vector{a, b}, 0.8,
                                                                     std::vector<double>(dim, 0.3)));
                REQUIRE(ham.supports_configuration_gradient());
                auto workspace = ham.make_potential_gradient_workspace(batch);
                std::vector<double> batched(batch.positions().size()); // [np*dim*nw]
                ham.evaluate_potential_gradient(batch, batched, workspace);
                std::vector<double> out(4 * dim, 9.0);
                for (std::size_t w = 0; w < 3; ++w) {
                    ham.configuration_gradient(batch, w, none, 4, none, out);
                    for (std::size_t j = 0; j < 4; ++j)
                        for (std::size_t d = 0; d < dim; ++d) {
                            CAPTURE(w, j, d);
                            CHECK(out[j * dim + d] == Approx(batched[(j * dim + d) * 3 + w]).margin(1e-12));
                        }
                }
            }
        }
    }
}

TEST_CASE("Inverse-power forces differentiate the sampled V, intra- and inter-group", "[configuration-gradient]") {
    for (double exponent : {3.0, 2.5}) {
        for (bool periodic : {false, true}) {
            DYNAMIC_SECTION("p = " << exponent << (periodic ? " periodic" : " open")) {
                SimulationCell cell = periodic ? SimulationCell(2, std::vector<double>{4.0, 0.0, 2.0, 3.4641016151377544})
                                               : SimulationCell(2);
                WalkerSet batch(2, 5, 2, cell);  // nw = 2, np = 5
                place(batch);
                const auto a = make_group(0, 3, 0.0, 2), b = make_group(3, 5, 0.0, 2);
                Hamiltonian ham;
                ham.add_component(std::make_unique<InversePowerTerm>(a, a, 1.3, exponent, cell));
                ham.add_component(std::make_unique<InversePowerTerm>(a, b, 0.7, exponent, cell));
                std::vector<double> out(10), values(2);
                for (std::size_t w = 0; w < 2; ++w) {
                    ham.configuration_gradient(batch, w, none, 5, none, out);
                    double total[2] = {0.0, 0.0};
                    for (std::size_t j = 0; j < 5; ++j)
                        for (std::size_t d = 0; d < 2; ++d) {
                            total[d] += out[j * 2 + d];
                            const std::size_t index = (j * 2 + d) * 2 + w;
                            const double saved = batch.positions()[index];
                            auto V = [&](double h) {
                                batch.positions()[index] = saved + h;
                                batch.refresh_particle_tables(j, w, w + 1);
                                ham.evaluate_potential(batch, values, w, w + 1);
                                batch.positions()[index] = saved;
                                batch.refresh_particle_tables(j, w, w + 1);
                                return values[w];
                            };
                            CAPTURE(w, j, d);
                            check_derivative(V, out[j * 2 + d], 1e-7);
                        }
                    // Pair forces only: Newton's third law.
                    CHECK(std::abs(total[0]) < 1e-12);
                    CHECK(std::abs(total[1]) < 1e-12);
                }
            }
        }
    }
}

TEST_CASE("Configuration gradients count only the first n_rows rows other than p and skip", "[configuration-gradient]") {
    for (bool periodic : {false, true}) {
        DYNAMIC_SECTION((periodic ? "periodic" : "open")) {
            const std::size_t np = 6, dim = 2, n_rows = 5, left_out = 2;
            SimulationCell cell = periodic ? SimulationCell(2, std::vector<double>{4.0, 0.0, 2.0, 3.4641016151377544})
                                           : SimulationCell(2);
            WalkerSet batch(2, np, dim, cell);  // nw = 2
            place(batch);
            const Hamiltonian ham = one_species(np, dim, periodic);
            std::vector<double> by_skip(np * dim, 9.0), by_p(np * dim, 9.0);
            for (std::size_t w = 0; w < 2; ++w) {
                ham.configuration_gradient(batch, w, none, n_rows, left_out, by_skip);
                ham.configuration_gradient(batch, w, left_out, n_rows, none, by_p);
                CHECK(by_skip == by_p);  // p and skip leave out a row the same way
                for (std::size_t j = 0; j < np; ++j) {
                    const bool counted = j < n_rows && j != left_out;
                    for (std::size_t d = 0; d < dim; ++d) {
                        CAPTURE(w, j, d);
                        if (!counted) {
                            CHECK(by_skip[j * dim + d] == 0.0);
                            continue;
                        }
                        // V(R) = V(R \ {j}) + e_j(r_j): only the bead energy of
                        // row j depends on r_j.
                        auto r = row_position(batch, w, j);
                        const double saved = r[d];
                        auto e = [&](double h) {
                            r[d] = saved + h;
                            const double value = ham.bead_energy(batch, w, j, r, n_rows, left_out);
                            r[d] = saved;
                            return value;
                        };
                        check_derivative(e, by_skip[j * dim + d], 1e-7);
                    }
                }
            }
        }
    }
}

TEST_CASE("A joining bead's gradient is its own force and the change of every other row's", "[configuration-gradient]") {
    for (bool periodic : {false, true}) {
        DYNAMIC_SECTION((periodic ? "periodic" : "open")) {
            const std::size_t np = 6, dim = 2;
            SimulationCell cell = periodic ? SimulationCell(2, std::vector<double>{4.0, 0.0, 2.0, 3.4641016151377544})
                                           : SimulationCell(2);
            WalkerSet batch(2, np, dim, cell);  // nw = 2
            place(batch);
            const Hamiltonian ham = one_species(np, dim, periodic);
            struct Case { std::size_t p, n_rows, skip; };
            // A row inside the configuration, one beyond n_rows (a new row of a
            // grand-canonical insert), and a slice with a missing bead.
            for (const Case c : {Case{1, 6, none}, Case{5, 5, none}, Case{0, 6, 3}}) {
                CAPTURE(c.p, c.n_rows, c.skip);
                std::vector<double> position = {0.37, -0.52};
                std::vector<double> self(dim, 9.0), rows(np * dim, 9.0);
                const std::size_t w = 1;
                ham.bead_gradient(batch, w, c.p, position, c.n_rows, c.skip, self, rows);
                for (std::size_t d = 0; d < dim; ++d) {
                    const double saved = position[d];
                    auto e = [&](double h) {
                        position[d] = saved + h;
                        const double value = ham.bead_energy(batch, w, c.p, position, c.n_rows, c.skip);
                        position[d] = saved;
                        return value;
                    };
                    check_derivative(e, self[d], 1e-7);
                }
                for (std::size_t j = 0; j < np; ++j) {
                    const bool counted = j < c.n_rows && j != c.p && j != c.skip;
                    for (std::size_t d = 0; d < dim; ++d) {
                        CAPTURE(j, d);
                        if (!counted) {
                            CHECK(rows[j * dim + d] == 0.0);
                            continue;
                        }
                        // V(R + r) - V(R) = e(r), so d e / d r_j is the change of
                        // row j's force.
                        const std::size_t index = (j * dim + d) * 2 + w;
                        const double saved = batch.positions()[index];
                        auto e = [&](double h) {
                            batch.positions()[index] = saved + h;
                            const double value = ham.bead_energy(batch, w, c.p, position, c.n_rows, c.skip);
                            batch.positions()[index] = saved;
                            return value;
                        };
                        check_derivative(e, rows[j * dim + d], 1e-7);
                    }
                }
            }
        }
    }
}

TEST_CASE("Configuration gradients refuse misuse and report unsupported terms", "[configuration-gradient]") {
    WalkerSet batch(2, 3, 2);  // nw = 2, np = 3
    place(batch);
    const Hamiltonian ham = one_species(3, 2, false);
    std::vector<double> small(5), self(2), rows(6);
    const std::vector<double> position = {0.1, 0.2};
    CHECK_THROWS_AS(ham.configuration_gradient(batch, 0, none, 3, none, small), std::invalid_argument);
    CHECK_THROWS_AS(ham.configuration_gradient(batch, 2, none, 3, none, rows), std::invalid_argument);
    CHECK_THROWS_AS(ham.configuration_gradient(batch, 0, none, 4, none, rows), std::invalid_argument);
    CHECK_THROWS_AS(ham.bead_gradient(batch, 0, 0, position, 3, none, self, small), std::invalid_argument);
    struct NoGradient final : HamiltonianComponent {
        std::string name() const override { return "no_gradient"; }
        void evaluate_potential(const ParticleBatch&, std::span<double>, std::size_t,
                                std::size_t) const override {}
        std::unique_ptr<HamiltonianComponent> clone() const override {
            return std::make_unique<NoGradient>(*this);
        }
    };
    Hamiltonian with_unsupported = ham.clone();
    with_unsupported.add_component(std::make_unique<NoGradient>());
    CHECK_FALSE(with_unsupported.supports_configuration_gradient());
    CHECK_THROWS_AS(with_unsupported.configuration_gradient(batch, 0, none, 3, none, rows), std::logic_error);
}
