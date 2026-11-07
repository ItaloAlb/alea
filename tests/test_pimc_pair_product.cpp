// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include "hamiltonian/coulomb.h"
#include "hamiltonian/harmonic_trap.h"
#include "hamiltonian/rytova_keldysh.h"
#include "hamiltonian/screened_coulomb.h"
#include "builders/action_builders.h"
#include "pimc/pair_product_action.h"
#include "pimc/pimc.h"
#include "qmc/app.h"
using namespace alea;
using Catch::Approx;
namespace {
ParticleGroup group(std::size_t p,double charge) {
    ParticleGroup g; g.first=p; g.last=p+1; g.count=1; g.charge=charge; g.mass={1.,1.}; g.inv_mass={1.,1.}; return g;
}
}
TEST_CASE("Pair action staged cyclic windows include coupled links and restore accepted state", "[pair-product][contract]") {
    const auto a=group(0,-1.),b=group(1,1.); const std::vector groups{a,b};
    Hamiltonian ham; ham.add_component(std::make_unique<CoulombTerm>(a,b,2.));
    ham.add_component(std::make_unique<HarmonicTrap>(groups,1.));
    io::Config config; config.action={{"kinetic","primitive"},{"pair_product","finite_grid",3.,9}};
    auto action=build_action(config,groups,ham,2); action.prepare_time_step(.125);
    PathSet path(4,2,2,.125);
    for (std::size_t i=0; i<path.positions().size(); ++i) path.positions()[i]=.2*std::sin(.7*i);
    action.prepare(path); path.rebuild_tables();
    const auto saved=path.all_positions();
    const double old=action.total_action(path);
    const std::array<double,6> proposed{.4,.1,.3,-.2,-.1,.25};
    const double delta=action.delta_potential_window(path,0,3,proposed);
    CHECK(path.all_positions()==saved);
    double spring_change=0;
    const double old_spring=action.spring_action_all(path,0,1)+action.spring_action_all(path,1,2)+
                            action.spring_action_all(path,2,3)+action.spring_action_all(path,3,0);
    for (std::size_t i=0; i<3; ++i) for (std::size_t d=0; d<2; ++d) path.pos(0,d,(3+i)%4)=proposed[2*i+d];
    const double fresh=action.total_action(path);
    spring_change=action.spring_action_all(path,0,1)+action.spring_action_all(path,1,2)+
                  action.spring_action_all(path,2,3)+action.spring_action_all(path,3,0)-old_spring;
    CHECK(delta==Approx(fresh-old-spring_change).margin(2e-13));
    std::copy(saved.begin(),saved.end(),path.positions().begin()); path.rebuild_tables();
    const std::array<double,2> outside{10.,0.};
    CHECK(std::isinf(action.delta_potential_particle(path,0,0,outside)));
    CHECK(path.all_positions()==saved);
    CHECK(action.clone().total_action(path)==action.total_action(path));
    CHECK(action.clone().measure_thermo(path).energy==action.measure_thermo(path).energy);
}
TEST_CASE("Numerical pair action uses radial capabilities for Coulomb RK and FSC", "[pair-product][potential]") {
    const auto a=group(0,-1.),b=group(1,1.); const std::vector groups{a,b};
    for (int kind=0; kind<3; ++kind) {
        Hamiltonian ham;
        if (kind==0) ham.add_component(std::make_unique<CoulombTerm>(a,b,2.));
        if (kind==1) ham.add_component(std::make_unique<KeldyshTerm>(a,b,2.,.7));
        if (kind==2) ham.add_component(std::make_unique<ScreenedCoulombTerm>(a,b,std::vector{2.},std::vector<double>{},1,0,0.,.01));
        PairProductAction action(groups,ham,2,3.,9); action.prepare_time_step(.125);
        PathSet path(2,2,2,.125); // Coincident pair: action finite even at the attractive pole.
        CHECK(std::isfinite(action.total_action(path)));
        CHECK(action.total_action(path)<0);
        path.pos(1,0,0)=.2; path.pos(1,0,1)=.3;
        CHECK(std::isfinite(action.measure_thermo(path).energy));
        CHECK_THROWS(action.validate_sampling(SimulationCell(2,std::vector{4.,0.,0.,4.}),.125,2));
    }
    auto anisotropic=groups; anisotropic[0].mass={1.,2.};
    CHECK_THROWS(PairProductAction(anisotropic,Hamiltonian{},2,3.,9));
}
TEST_CASE("Pair product YAML samples unsoftened attraction and keeps thread reproducibility", "[pair-product][driver]") {
    const std::string text=R"(project: {name: pair_contract, seed: 88381, output: false, log: false}
system:
  lattice: {type: open, dimension: 2, L: 0.5}
  particles:
    - {name: e, group: electrons, type: boltzmannon, mass: 1, charge: -1, up: 1, down: 0}
    - {name: h, group: holes, type: boltzmannon, mass: 2, charge: 1, up: 1, down: 0}
hamiltonian:
  terms:
    - {type: coulomb, pair: [electrons, holes], epsilon: 2}
    - {type: harmonic_trap, omega: 0.8}
action:
  - kinetic: primitive
  - pair_product: {form: finite_grid, extent: 3, grid_points: 9}
qmc:
  - {method: pimc, beta: 0.5, time_step: 0.125, samples: 4, threads: 1, equilibration_blocks: 16, accumulation_blocks: 32, steps_per_block: 8}
)";
    auto config=io::parse_config(text);
    App app(config); auto pack=app.build_drivers(); auto* driver=dynamic_cast<Pimc*>(pack.drivers[0].get());
    REQUIRE(driver); driver->run();
    CHECK(std::isfinite(driver->energy()));
    CHECK_THAT(driver->summary(),Catch::Matchers::ContainsSubstring("pair_product finite_grid"));
    CHECK_THAT(driver->summary(),Catch::Matchers::ContainsSubstring("identity connectivity (boltzmannon)"));
    const auto result=driver->run_result();
    const auto approximation=std::find_if(result.metadata.begin(),result.metadata.end(),
        [](const auto& item){return item.first=="action_approximation";});
    REQUIRE(approximation!=result.metadata.end());
    CHECK(approximation->second=="finite-tau composite; identity connectivity");
    config.pimc[0].threads=4;
    App threaded(config); auto threaded_pack=threaded.build_drivers(); auto* other=dynamic_cast<Pimc*>(threaded_pack.drivers[0].get());
    REQUIRE(other); other->run();
    CHECK(driver->export_paths().positions==other->export_paths().positions);
    CHECK(driver->energy()==other->energy());
    other->import_paths(driver->export_paths());
    CHECK(other->export_paths().positions==driver->export_paths().positions);
    auto bad=driver->export_paths(); bad.positions[0]=100.;
    CHECK_THROWS(other->import_paths(bad));
    CHECK(other->export_paths().positions==driver->export_paths().positions);
}

TEST_CASE("Pair product parser refuses missing domains typos forms and double counting", "[pair-product][parser]") {
    const std::string prefix="project: {seed: 2}\naction:\n  - kinetic: primitive\n  - pair_product: ";
    for (const std::string parameters : {"{}","{extent: 3, grid_ponts: 9}","{extent: 3, form: exact}","{extent: 3, grid_points: 8}","{extent: -1}"})
        CHECK_THROWS(io::parse_config(prefix+parameters+"\n"));
    CHECK_THROWS(io::parse_config(prefix+"{extent: 3}\n  - potential: primitive\n"));
    CHECK_THROWS(io::parse_config(prefix+"{extent: 3}\n  - pair_product: {extent: 3}\n"));
    const auto parsed=io::parse_config(prefix+"{extent: 3.25, grid_points: 13}\n");
    REQUIRE(parsed.action.size()==2);
    CHECK(parsed.action[1].extent==3.25); CHECK(parsed.action[1].grid_points==13);
}

TEST_CASE("Pair thermodynamic measurement differentiates an independent confined one bead partition", "[pair-product][physics]") {
    const auto a=group(0,-1.),b=group(1,1.); const std::vector groups{a,b};
    const double beta=.5,extent=3,h=1e-5;
    Hamiltonian ham; ham.add_component(std::make_unique<CoulombTerm>(a,b,2.));
    ham.add_component(std::make_unique<HarmonicTrap>(groups,1.));
    io::Config cfg; cfg.action={{"kinetic","primitive"},{"pair_product","finite_grid",extent,9}};
    auto action=build_action(cfg,groups,ham,2); action.prepare_time_step(beta);
    PathSet path(1,2,2,beta); action.prepare(path);
    auto radial=[](double r){return -.5/r;};
    numerics::FiniteGridPair minus(.5,beta-h,extent,9,radial,-.5),plus(.5,beta+h,extent,9,radial,-.5),base(.5,beta,extent,9,radial,-.5);
    double previous=0;
    for (int n : {48,96,192}) {
        double norm=0,energy=0,zminus=0,zplus=0;
        for (int i=0; i<n; ++i) for (int j=0; j<n; ++j) {
            const std::array<double,2> q{-extent+2*extent*(i+.5)/n,-extent+2*extent*(j+.5)/n};
            const double trap=.25*(q[0]*q[0]+q[1]*q[1]);
            const double weight=std::exp(-base.evaluate(q,q).action-beta*trap);
            for (std::size_t d=0; d<2; ++d) { path.pos(0,d,0)=q[d]/2; path.pos(1,d,0)=-q[d]/2; }
            norm+=weight; energy+=weight*action.measure_thermo(path).energy;
            // CM is a 2D primitive one-bead oscillator, E_CM=2/beta.
            // Relative normalization ~1/beta; only scalar u enters this
            // independent trace, never production thermal derivatives.
            zminus+=std::exp(-minus.evaluate(q,q).action-(beta-h)*trap)/(beta-h);
            zplus+=std::exp(-plus.evaluate(q,q).action-(beta+h)*trap)/(beta+h);
        }
        const double measured=energy/norm+1/beta,reference=2/beta-(std::log(zplus)-std::log(zminus))/(2*h);
        std::cout << "pair trace n=" << n << " E=" << measured << " derivative_error=" << measured-reference << '\n';
        CHECK(measured==Approx(reference).margin(2e-7));
        if (n==192) CHECK(std::abs(measured-previous)<.001);
        previous=measured;
    }
}

namespace {
ParticleGroup group_range(std::size_t first, std::size_t last, double charge, double mass) {
    ParticleGroup g;
    g.first = first; g.last = last; g.count = last - first; g.charge = charge;
    g.mass = {mass, mass}; g.inv_mass = {1. / mass, 1. / mass};
    return g;
}

// Test-only radial pair V(r) = mu omega^2 r^2 / 2 between groups a and b:
// for two particles the pair-product density matrix is then exact, so PIMC
// must reproduce the continuum energy at every bead count M.
class HarmonicPairTerm final : public HamiltonianComponent {
public:
    HarmonicPairTerm(ParticleGroup a, ParticleGroup b, double stiffness)
        : a_(std::move(a)), b_(std::move(b)), stiffness_(stiffness) {}
    std::string name() const override { return "harmonic_pair"; }
    PairActionRole pair_action_role() const noexcept override { return PairActionRole::radial_pairs; }
    std::vector<RadialPairPotential> radial_pair_potentials() const override {
        const double k = stiffness_;
        return {{a_, b_, [k](double r) { return 0.5 * k * r * r; }, 0.0, "test harmonic pair"}};
    }
    bool is_translation_invariant() const noexcept override { return true; }
    std::unique_ptr<HamiltonianComponent> clone() const override {
        return std::make_unique<HarmonicPairTerm>(*this);
    }
private:
    ParticleGroup a_, b_;
    double stiffness_;
};

// Exact 2D relative oscillator pair action u = -ln(rho_osc/rho_free) and its
// tau derivative (Mehler kernel; the same independent formula as
// test_finite_grid_pair.cpp).
std::array<double, 2> exact_oscillator(const std::array<double, 2>& a, const std::array<double, 2>& b,
                                       double mu, double omega, double tau) {
    double squares = 0, dot = 0, delta = 0;
    for (int d = 0; d < 2; ++d) {
        squares += a[d] * a[d] + b[d] * b[d];
        dot += a[d] * b[d];
        delta += (a[d] - b[d]) * (a[d] - b[d]);
    }
    const double x = omega * tau, s = std::sinh(x), c = std::cosh(x);
    return {std::log(s / x) + mu * omega * (squares * c - 2 * dot) / (2 * s) - mu * delta / (2 * tau),
            omega * c / s - 1 / tau + mu * omega * omega * (-squares + 2 * dot * c) / (2 * s * s) +
                mu * delta / (2 * tau * tau)};
}

double scalar(const RunResult& result, const std::string& name) {
    const auto it = std::find_if(result.scalars.begin(), result.scalars.end(),
                                 [&](const auto& item) { return item.first == name; });
    REQUIRE(it != result.scalars.end());
    return it->second;
}
} // namespace

TEST_CASE("Incremental pair-product window deltas equal full action differences", "[pair-product][contract]") {
    // Two electrons (same-group channel, i<j orientation) and one hole
    // (cross channel, both orientations), every window of every particle.
    const auto electrons = group_range(0, 2, -1., 1.), holes = group_range(2, 3, 1., 2.);
    const std::vector groups{electrons, holes};
    Hamiltonian ham;
    ham.add_component(std::make_unique<CoulombTerm>(electrons, holes, 2.));
    ham.add_component(std::make_unique<CoulombTerm>(electrons, electrons, 2.));
    const double tau = .125;
    PairProductAction action(groups, ham, 2, 4., 9);
    action.prepare_time_step(tau);
    for (std::size_t M : {1u, 2u, 3u, 5u}) {
        PathSet path(M, 3, 2, tau);
        for (std::size_t i = 0; i < path.positions().size(); ++i)
            path.positions()[i] = .6 * std::sin(.9 * static_cast<double>(i) + .3 * static_cast<double>(M));
        const auto saved = path.all_positions();
        const double old_action = action.total_action(path);
        for (std::size_t p = 0; p < 3; ++p)
            for (std::size_t first = 0; first < M; ++first)
                for (std::size_t count = 1; count <= M; ++count) {
                    CAPTURE(M, p, first, count);
                    std::vector<double> proposed(count * 2);
                    for (std::size_t i = 0; i < count; ++i)
                        for (std::size_t d = 0; d < 2; ++d)
                            proposed[i * 2 + d] = path.pos(p, d, (first + i) % M) +
                                .3 * std::cos(1.7 * static_cast<double>(i + d + p + count));
                    const double delta = action.delta_potential_window(path, p, first, proposed);
                    CHECK(path.all_positions() == saved);
                    PathSet moved(M, 3, 2, tau);
                    std::copy(saved.begin(), saved.end(), moved.positions().begin());
                    for (std::size_t i = 0; i < count; ++i)
                        for (std::size_t d = 0; d < 2; ++d)
                            moved.pos(p, d, (first + i) % M) = proposed[i * 2 + d];
                    const double reference = action.total_action(moved) - old_action;
                    REQUIRE(std::isfinite(reference));
                    CHECK(delta == Approx(reference).margin(1e-12));
                    if (count == 1)
                        CHECK(action.delta_potential_particle(path, p, first, proposed) == delta);
                }
        // A bead outside the relative domain is an infinite action change.
        const std::array<double, 2> outside{9., 0.};
        CHECK(action.delta_potential_window(path, 2, 0, outside) == std::numeric_limits<double>::infinity());
    }
}

TEST_CASE("Pair product reports proposals rejected at the finite domain wall", "[pair-product][driver]") {
    // e-h Bohr radius eps/mu = 3 against a relative domain of half-width 1:
    // proposals constantly leave the square.  The rejection is exact
    // (exp(-S) = 0 outside) but must be reported, never hidden in acc.
    const std::string text = R"(project: {name: pair_wall, seed: 5150, output: false, log: false}
system:
  lattice: {type: open, dimension: 2, L: 0.5}
  particles:
    - {name: e, group: electrons, type: boltzmannon, mass: 1, charge: -1, up: 1, down: 0}
    - {name: h, group: holes, type: boltzmannon, mass: 2, charge: 1, up: 1, down: 0}
hamiltonian:
  terms:
    - {type: coulomb, pair: [electrons, holes], epsilon: 2}
action:
  - kinetic: primitive
  - pair_product: {form: finite_grid, extent: 1, grid_points: 9}
qmc:
  - {method: pimc, beta: 0.5, time_step: 0.125, samples: 4, threads: 2, equilibration_blocks: 8, accumulation_blocks: 32, steps_per_block: 8}
)";
    auto config = io::parse_config(text);
    App app(config);
    auto pack = app.build_drivers();
    auto* driver = dynamic_cast<Pimc*>(pack.drivers[0].get());
    REQUIRE(driver);
    driver->run();
    const auto rejected = driver->infinite_action_rejections();
    CHECK(rejected > 0);
    const auto result = driver->run_result();
    CHECK(scalar(result, "infinite_action_rejections") == static_cast<double>(rejected));
    CHECK(scalar(result, "infinite_action_rejection_fraction") > 0);
    double per_move = 0;
    for (std::size_t i = 0; i < driver->move_specs().size(); ++i)
        per_move += scalar(result, "move_" + std::to_string(i) + "_infinite_action_rejections");
    CHECK(per_move == static_cast<double>(rejected));
    CHECK_THAT(driver->summary(), Catch::Matchers::ContainsSubstring("rejected for infinite action"));
}

TEST_CASE("Pair-product PIMC of two harmonically bound particles matches the exact energy at M = 1 and 2",
          "[pair-product][physics]") {
    // H = P^2/(2 M_tot) + p^2/(2 mu) + mu omega^2 r^2/2 in 2D, mu = 1.  For two
    // particles rho_free * exp(-u) is the exact density matrix, so the
    // internal energy E - d/(2 beta) (free open CM removed) must equal the 2D
    // oscillator omega coth(beta omega / 2) at every M, whereas the exact
    // primitive-action values here are 52% (M = 1) and 28% (M = 2) too low
    // (Gaussian determinant of the discretized ring).  Tolerance: 5 reblocked sigma
    // plus the finite-grid table error, measured below against the exact
    // Mehler kernel and weighted by the exact relative-coordinate density of
    // the M-bead path (independent of the PIMC samples).  <|d du/dtau|> enters
    // the estimator directly; M <|du|> is the first-order reweighting of an
    // O(1)-spread estimator.  Worst-case errors are larger but sit where the
    // box walls at |x| = extent bend the free table at tau = beta, which the
    // thermal density never visits.
    const double mass = 2., mu = 1., omega = 1., beta = 4., extent = 4.5;
    const auto a = group_range(0, 1, 0., mass), b = group_range(1, 2, 0., mass);
    const std::vector groups{a, b};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicPairTerm>(a, b, mu * omega * omega));
    const double exact_internal = omega / std::tanh(beta * omega / 2);
    for (std::size_t M : {1u, 2u}) {
        CAPTURE(M);
        const double tau = beta / static_cast<double>(M);
        Action action(groups, ham, 2, true, false);
        action.add_component(std::make_unique<PairProductAction>(groups, ham, 2, extent, 33));
        action.prepare_time_step(tau);  // Pimc's clones share these tables
        ActionComponent& pair = *action.components()[1];

        // Table error on a 13x13 grid of relative endpoints within ~4 sigma
        // per axis, weighted by the exact density of the relative ring:
        // M = 1: exp(-u(r,r)); M = 2: exp(-mu |r0-r1|^2 / tau - 2 u(r0,r1)).
        double weight_sum = 0, mean_du = 0, mean_dthermal = 0, max_du = 0;
        PathSet probe(M, 2, 2, tau);
        const auto set_relative = [&](std::size_t k, const std::array<double, 2>& r) {
            for (std::size_t d = 0; d < 2; ++d) { probe.pos(0, d, k) = r[d] / 2; probe.pos(1, d, k) = -r[d] / 2; }
        };
        const auto node = [](int i) {
            return std::array<double, 2>{-3. + .5 * (i / 13), -3. + .5 * (i % 13)};
        };
        for (int i = 0; i < 169; ++i)
            for (int j = 0; j < (M == 1 ? 1 : 169); ++j) {
                const auto r0 = node(i), r1 = M == 1 ? r0 : node(j);
                set_relative(0, r0);
                if (M == 2) set_relative(1, r1);
                const auto exact = exact_oscillator(r0, r1, mu, omega, tau);
                const double gap = (r0[0] - r1[0]) * (r0[0] - r1[0]) + (r0[1] - r1[1]) * (r0[1] - r1[1]);
                const double weight = M == 1 ? std::exp(-exact[0]) : std::exp(-mu * gap / tau - 2 * exact[0]);
                // M links per path: u(r0,r1) + u(r1,r0) = 2 u for M = 2.
                const double du = std::abs(pair.total_action(probe) / static_cast<double>(M) - exact[0]);
                weight_sum += weight;
                mean_du += weight * du;
                mean_dthermal += weight * std::abs(pair.beta_derivative(probe) - exact[1]);
                max_du = std::max(max_du, du);
            }
        mean_du /= weight_sum;
        mean_dthermal /= weight_sum;

        PimcConfig cfg;
        cfg.samples = 8; cfg.threads = 0; cfg.beta = beta; cfg.time_step = tau; cfg.seed = 20261005;
        cfg.equilibration_blocks = 100; cfg.accumulation_blocks = 400; cfg.steps_per_block = 25; cfg.sub_steps = 2;
        Pimc pimc(cfg, groups, ham, SimulationCell{}, 2, 2, nullptr, &action);
        pimc.run();
        const auto result = pimc.run_result();
        const double internal = scalar(result, "internal_energy"), sigma = scalar(result, "internal_energy_error");
        const double tolerance = 5 * sigma + mean_dthermal + static_cast<double>(M) * mean_du;
        std::cout << "harmonic pair M=" << M << " E_int=" << internal << " +/- " << sigma
                  << " exact=" << exact_internal << " table <|du|>=" << mean_du << " (max " << max_du
                  << ") <|d du/dtau|>=" << mean_dthermal << " tolerance=" << tolerance
                  << " rejected=" << pimc.infinite_action_rejections() << '\n';
        CHECK(scalar(result, "cm_energy_correction") == Approx(2 / (2 * beta)));
        CHECK(scalar(result, "energy_error_reliable") == 1);
        CHECK(std::abs(internal - exact_internal) < tolerance);
        CHECK(tolerance < .05 * exact_internal);  // the check stays discriminating
        CHECK(scalar(result, "infinite_action_rejection_fraction") < 1e-3);
    }
}
