// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Composite-action regression tests: admissibility is checked before sampling.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <limits>
#include <memory>
#include <vector>
#include "pimc/action.h"
#include "hamiltonian/coulomb.h"
#include "hamiltonian/rytova_keldysh.h"
#include "hamiltonian/harmonic_trap.h"
#include "hamiltonian/moire.h"
#include "hamiltonian/screened_coulomb.h"
#include "pimc/action_component.h"
#include "pimc/primitive_action.h"
#include "pimc/pair_product_action.h"
#include "pimc/move.h"
#include "builders/action_builders.h"
#include <atomic>
#include <array>
#include <cstdlib>
#include <new>

namespace {
thread_local bool count_allocations = false;
thread_local std::size_t allocation_count = 0;
}
#if defined(__GNUC__)
#define ALEA_NOINLINE __attribute__((noinline))
#else
#define ALEA_NOINLINE
#endif
ALEA_NOINLINE void* operator new(std::size_t size) {
    if (count_allocations) ++allocation_count;
    if (void* value = std::malloc(size ? size : 1)) return value;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
ALEA_NOINLINE void operator delete(void* value) noexcept { std::free(value); }
ALEA_NOINLINE void operator delete[](void* value) noexcept { std::free(value); }
ALEA_NOINLINE void operator delete(void* value, std::size_t) noexcept { std::free(value); }
ALEA_NOINLINE void operator delete[](void* value, std::size_t) noexcept { std::free(value); }

class ConstantAction final : public alea::ActionComponent {
public:
    explicit ConstantAction(double value) : value_(value) {}
    double total_action(const alea::PathSet&) override { return value_; }
    alea::ActionThermoMeasure measure_thermo(const alea::PathSet&) override {
        return {0.0, value_, value_};
    }
    std::unique_ptr<alea::ActionComponent> clone() const override {
        return std::make_unique<ConstantAction>(*this);
    }
private:
    double value_;
};

class ThrowingLinkAction final : public alea::ActionComponent {
public:
    double total_action(const alea::PathSet& path) override {
        if (path.pos(0,0,0)>10.) throw std::runtime_error("staged link failed");
        return path.pos(0,0,0)*path.pos(0,0,1);
    }
    std::unique_ptr<alea::ActionComponent> clone() const override {
        return std::make_unique<ThrowingLinkAction>(*this);
    }
};

TEST_CASE("Coupled action window restores all beads and tables after an exception", "[pimc-action][window]") {
    alea::ParticleGroup g;
    g.first=0; g.last=2; g.count=2; g.mass={1.,1.}; g.inv_mass={1.,1.};
    alea::Action action{std::vector{g},alea::Hamiltonian{},2,false,false};
    action.add_component(std::make_unique<ThrowingLinkAction>());
    alea::PathSet path(4,2,2,.125);
    path.pos(0,0,0)=.2; path.pos(0,0,1)=.4;
    path.pos(1,0,0)=1.; path.pos(1,1,3)=2.;
    path.rebuild_tables(); action.prepare(path);
    const auto saved=path.all_positions();
    const std::vector<double> tables(path.dist_tables().data().begin(),path.dist_tables().data().end());
    CHECK(action.delta_potential_particle(path,0,0,std::array{.6,.7})==Catch::Approx(.16));
    CHECK(path.all_positions()==saved);
    CHECK(std::vector<double>(path.dist_tables().data().begin(),path.dist_tables().data().end())==tables);
    // A cyclic window changes slice 3 first, then throws at slice 0.
    const std::array<double,4> proposed{3.,4.,20.,5.};
    CHECK_THROWS(action.delta_potential_window(path,0,3,proposed));
    CHECK(path.all_positions()==saved);
    CHECK(std::vector<double>(path.dist_tables().data().begin(),path.dist_tables().data().end())==tables);
}

TEST_CASE("Prepared pair action windows measurements and moves allocate nothing", "[pair-allocation]") {
    alea::ParticleGroup a,b;
    a.first=0; a.last=1; a.count=1; a.mass={1.}; a.inv_mass={1.}; a.charge=-1.;
    b=a; b.first=1; b.last=2; b.charge=1.;
    const std::vector groups{a,b};
    alea::Hamiltonian ham; ham.add_component(std::make_unique<alea::CoulombTerm>(a,b,2.));
    alea::io::Config cfg; cfg.action={{"kinetic","primitive"},{"pair_product","finite_grid",3.,9}};
    auto action=alea::build_action(cfg,groups,ham,2); action.prepare_time_step(.125);
    alea::PathSet path(4,2,2,.125);
    for (std::size_t i=0; i<path.positions().size(); ++i) path.positions()[i]=.2*std::sin(.7*i);
    path.rebuild_tables(); action.prepare(path);
    auto clone=action.clone(); clone.prepare(path);
    alea::SingleSliceMove single(.1); alea::BisectionMove bisection(2); alea::CentroidMove centroid(.1);
    single.prepare(path); bisection.prepare(path); centroid.prepare(path);
    alea::numerics::Random rng(7843);
    const std::array<double,6> proposed{.4,.1,.3,-.2,-.1,.25};
    allocation_count=0; count_allocations=true;
    double energy=0;
    try {
        energy+=clone.measure_thermo(path).energy;
        energy+=clone.delta_potential_window(path,0,3,proposed);
        for (int i=0; i<20; ++i) {
            single.attempt(path,clone,rng); bisection.attempt(path,clone,rng); centroid.attempt(path,clone,rng);
            energy+=clone.measure_thermo(path).energy;
        }
    } catch (...) { count_allocations=false; throw; }
    count_allocations=false;
    CHECK(allocation_count==0); CHECK(std::isfinite(energy));
}

class DerivativeAction final : public alea::ActionComponent {
public:
    double total_action(const alea::PathSet& path) override { return 2.0*path.beta(); }
    double normalization_energy(const alea::PathSet&) const override { return 7.0; }
    double beta_derivative(const alea::PathSet&) override { return 2.0; }
    std::unique_ptr<alea::ActionComponent> clone() const override {
        return std::make_unique<DerivativeAction>(*this);
    }
};

class FallbackPotential final : public alea::HamiltonianComponent {
public:
    explicit FallbackPotential(bool throws = false) : throws_(throws) {}
    std::string name() const override { return "test_fallback"; }
    void evaluate_potential(const alea::ParticleBatch& batch, std::span<double> out,
                            std::size_t begin, std::size_t end) const override {
        for (auto k = begin; k < end; ++k) {
            const double x = batch.positions()[k];
            if (throws_ && x > 10.0) throw std::runtime_error("proposed evaluation failed");
            out[k] += x*x + batch.dist_tables()(k, 0, 1);
        }
    }
    std::unique_ptr<alea::HamiltonianComponent> clone() const override {
        return std::make_unique<FallbackPotential>(*this);
    }
private:
    bool throws_;
};

namespace {
alea::ParticleGroup group(std::size_t first, std::size_t last, double charge = 0.0) {
    alea::ParticleGroup g;
    g.name = "g"; g.first = first; g.last = last; g.count = last - first;
    g.mass = {1.0, 1.0}; g.inv_mass = {1.0, 1.0}; g.charge = charge;
    return g;
}
}

TEST_CASE("Action is a genuine cloned component sum with bridge capabilities", "[pimc-action]") {
    alea::PathSet path(4, 1, 2, 0.25);
    const auto g = group(0, 1);
    alea::Hamiltonian ham;
    alea::Action action{std::vector{g}, ham, 2};
    CHECK(action.components().size() == 2);
    CHECK(action.supports_free_bridge());
    CHECK(action.measure_thermo(path).kinetic == Catch::Approx(4.0));
    action.add_component(std::make_unique<ConstantAction>(3.0));
    CHECK_FALSE(action.supports_free_bridge());
    auto copy = action.clone();
    copy.add_component(std::make_unique<ConstantAction>(2.0));
    CHECK(action.total_action(path) == Catch::Approx(3.0));
    CHECK(copy.total_action(path) == Catch::Approx(5.0));
    CHECK(copy.measure_thermo(path).energy == Catch::Approx(9.0));
    copy.add_component(std::make_unique<DerivativeAction>());
    CHECK(copy.measure_thermo(path).energy == Catch::Approx(18.0));
    CHECK(action.measure_thermo(path).energy == Catch::Approx(7.0));
}

TEST_CASE("Action builders install each YAML component independently", "[pimc-action]") {
    const auto g=group(0,1);
    alea::Hamiltonian ham;
    alea::io::Config cfg;
    cfg.action = {{"kinetic","primitive"},{"potential","primitive"}};
    auto action=alea::build_action(cfg,std::vector{g},ham,2);
    CHECK(action.components().size()==2);
    CHECK(action.supports_free_bridge());
    cfg.action.push_back({"kinetic","primitive"});
    auto duplicate=alea::build_action(cfg,std::vector{g},ham,2);
    CHECK(duplicate.components().size()==3);
    CHECK_FALSE(duplicate.supports_free_bridge());
    cfg.action={{"kinetic","unknown"}};
    CHECK_THROWS(alea::build_action(cfg,std::vector{g},ham,2));
}

TEST_CASE("Potential delta fallback restores coordinates and tables even on exceptions", "[pimc-action]") {
    const auto g = group(0, 2);
    alea::Hamiltonian ham;
    ham.add_component(std::make_unique<FallbackPotential>(true));
    alea::Action action{std::vector{g}, ham, 2};
    alea::PathSet path(4, 2, 2, 0.25);
    path.pos(1, 0, 1) = 4.0;
    path.rebuild_tables(); action.prepare(path);
    const auto saved = path.all_positions();
    const std::vector<double> saved_tables(path.dist_tables().data().begin(), path.dist_tables().data().end());
    CHECK(action.delta_potential_particle(path, 0, 1, std::vector{2.0, 0.0}) == Catch::Approx(0.5));
    CHECK(path.all_positions() == saved);
    CHECK(std::vector<double>(path.dist_tables().data().begin(), path.dist_tables().data().end()) == saved_tables);
    CHECK_THROWS(action.delta_potential_particle(path, 0, 1, std::vector{20.0, 0.0}));
    CHECK(path.all_positions() == saved);
    CHECK(std::vector<double>(path.dist_tables().data().begin(), path.dist_tables().data().end()) == saved_tables);
}

TEST_CASE("Optimized potential deltas equal full evaluation without mutating a bead", "[pimc-action]") {
    const auto g = group(0, 3);
    alea::Hamiltonian ham;
    SECTION("Coulomb") { ham.add_component(std::make_unique<alea::CoulombTerm>(g,g,2.0,0.2)); }
    SECTION("RK") { ham.add_component(std::make_unique<alea::KeldyshTerm>(g,g,2.0,3.0,0.2)); }
    SECTION("trap") { ham.add_component(std::make_unique<alea::HarmonicTrap>(std::vector{g},1.3)); }
    SECTION("moire") { ham.add_component(std::make_unique<alea::MoireTerm>(2,0.4,0.7,0.2,alea::moire_reciprocal_vectors(3.0,0.0,2),std::vector<std::pair<std::size_t,std::size_t>>{{0,3}})); }
    SECTION("screened") { ham.add_component(std::make_unique<alea::ScreenedCoulombTerm>(g,g,std::vector{1.0,2.0,1.0},std::vector{-5.0,5.0},2,0,0.0,0.01)); }
    SECTION("mixed") {
        ham.add_component(std::make_unique<alea::CoulombTerm>(g,g,2.0,0.2));
        ham.add_component(std::make_unique<alea::HarmonicTrap>(std::vector{g},1.3));
    }
    alea::SimulationCell cell(2,std::vector{8.0,0.0,0.0,8.0});
    alea::PathSet path(4,3,2,0.25,cell);
    for (std::size_t k=0;k<4;++k) {
        path.pos(0,0,k)=-3.5; path.pos(0,1,k)=0.2;
        path.pos(1,0,k)=3.3; path.pos(1,1,k)=0.6;
        path.pos(2,0,k)=1.0; path.pos(2,1,k)=-0.5;
    }
    path.rebuild_tables();
    alea::Action action{std::vector{g},ham,2}; action.prepare(path);
    const auto saved = path.all_positions();
    const std::vector<double> saved_tables(path.dist_tables().data().begin(),path.dist_tables().data().end());
    const double old=action.potential_action_slice(path,1);
    const double delta=action.delta_potential_particle(path,0,1,std::vector{4.2,1.1});
    CHECK(path.all_positions()==saved);
    CHECK(std::vector<double>(path.dist_tables().data().begin(),path.dist_tables().data().end())==saved_tables);
    path.pos(0,0,1)=4.2; path.pos(0,1,1)=1.1;
    path.wrap_particle(0,1,2); path.update_particle_tables(0,1,2);
    CHECK(delta == Catch::Approx(action.potential_action_slice(path,1)-old).margin(1e-11));
}

TEST_CASE("Radial deltas cover both sides of cross-group interactions", "[pimc-action]") {
    const auto a=group(0,2,-1.0), b=group(2,4,1.0);
    alea::Hamiltonian ham;
    ham.add_component(std::make_unique<alea::CoulombTerm>(a,b,2.0,0.7));
    ham.add_component(std::make_unique<alea::KeldyshTerm>(a,b,2.0,3.0,0.7));
    alea::PathSet path(4,4,2,0.25);
    for(std::size_t p=0;p<4;++p) for(std::size_t k=0;k<4;++k)
        path.pos(p,0,k)=1.0+static_cast<double>(p);
    path.rebuild_tables();
    alea::Action action{std::vector{a,b},ham,2}; action.prepare(path);
    for(std::size_t p=0;p<4;++p) {
        const auto old=action.potential_action_slice(path,2);
        const double proposal[2]={0.3+static_cast<double>(p),0.4};
        const auto delta=action.delta_potential_particle(path,p,2,proposal);
        path.pos(p,0,2)=proposal[0]; path.pos(p,1,2)=proposal[1];
        path.update_particle_tables(p,2,3);
        CHECK(delta==Catch::Approx(action.potential_action_slice(path,2)-old).margin(1e-12));
    }
}

TEST_CASE("Prepared primitive and generic action hot paths do not allocate", "[pimc-action]") {
    const auto g=group(0,2);
    alea::Hamiltonian ham;
    ham.add_component(std::make_unique<alea::CoulombTerm>(g,g,2.0,0.2));
    ham.add_component(std::make_unique<FallbackPotential>());
    alea::PathSet path(8,2,2,0.25);
    for(std::size_t k=0;k<8;++k) path.pos(1,0,k)=3.0;
    path.rebuild_tables();
    alea::Action action{std::vector{g},ham,2}; action.prepare(path);
    const double proposed[2]={1.0,0.1};
    allocation_count=0; count_allocations=true;
    double sink=0.0;
    try {
        for(std::size_t rep=0;rep<16;++rep) {
            sink+=action.potential_action_slice(path,rep%8);
            sink+=action.delta_potential_particle(path,0,rep%8,proposed);
            sink+=action.measure_thermo(path).energy;
        }
    } catch (...) { count_allocations=false; throw; }
    count_allocations=false;
    CHECK(allocation_count==0);
    CHECK(std::isfinite(sink));
}

TEST_CASE("Prepared moves with optimized and generic potentials do not allocate",
          "[pimc-action][incremental-moves]") {
    const auto g = group(0, 2, 1.0);
    alea::Hamiltonian ham;
    ham.add_component(std::make_unique<alea::CoulombTerm>(g, g, 2.0, 0.2));
    ham.add_component(std::make_unique<FallbackPotential>());
    alea::PathSet path(8, 2, 2, 0.0625);
    for (std::size_t k = 0; k < 8; ++k) path.pos(1, 0, k) = 3.0;
    path.rebuild_tables();
    alea::Action action{std::vector{g}, ham, 2};
    action.prepare(path);
    alea::SingleSliceMove single(0.25);
    alea::BisectionMove bisection(3);
    alea::CentroidMove centroid(0.4);
    const std::array<alea::Move*, 3> moves{&single, &bisection, &centroid};
    for (auto* move : moves) move->prepare(path);
    alea::numerics::Random rng(32819);
    allocation_count = 0;
    count_allocations = true;
    try {
        for (auto* move : moves)
            for (int attempt = 0; attempt < 16; ++attempt)
                move->attempt(path, action, rng);
    } catch (...) { count_allocations = false; throw; }
    count_allocations = false;
    CHECK(allocation_count == 0);
}

TEST_CASE("Action rejects malformed, nonpositive and nonfinite masses", "[pimc-action]") {
    alea::Hamiltonian ham;
    auto g = group(0, 1);
    SECTION("mass vector dimension mismatch") {
        g.mass = {1.0, 2.0, 3.0};
        CHECK_THROWS_AS(alea::Action(std::vector{g}, ham, 2), std::invalid_argument);
    }
    SECTION("nonpositive and nonfinite values") {
        for (double mass : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()}) {
            g.mass = {mass, 1.0};
            CHECK_THROWS_AS(alea::Action(std::vector{g}, ham, 2), std::invalid_argument);
        }
    }
}

TEST_CASE("Primitive kinetic action rejects ambiguous particle ownership",
          "[pimc-action][geometry]") {
    const alea::Hamiltonian free;
    const std::vector groups{group(0, 2), group(1, 3)};
    REQUIRE_THROWS_WITH(alea::Action(groups, free, 2),
        Catch::Matchers::ContainsSubstring("overlapping particle groups"));
}

TEST_CASE("Primitive kinetic mass storage rejects overflowing dimensions",
          "[pimc-action][geometry]") {
    const alea::Hamiltonian free;
    const std::vector groups{group(0, std::numeric_limits<std::size_t>::max())};
    REQUIRE_THROWS_WITH(alea::Action(groups, free, 2),
        Catch::Matchers::ContainsSubstring("mass storage dimensions overflow"));
}

TEST_CASE("Primitive Action refuses attractive Coulomb poles but accepts softened layers", "[pimc-action]") {
    auto e = group(0, 1, -1.0), h = group(1, 2, 1.0);
    const std::vector groups{e, h};
    alea::Hamiltonian singular;
    singular.add_component(std::make_unique<alea::CoulombTerm>(e, h, 2.0));
    CHECK_THROWS_AS(alea::Action(groups, singular, 2), std::invalid_argument);
    alea::Hamiltonian finite;
    finite.add_component(std::make_unique<alea::CoulombTerm>(e, h, 2.0, 0.5));
    CHECK_NOTHROW(alea::Action(groups, finite, 2));
}

TEST_CASE("Primitive Action refuses attractive screened intralayer poles", "[pimc-action]") {
    const auto a=group(0,1,-1.0), b=group(1,2,1.0);
    alea::Hamiltonian ham;
    ham.add_component(std::make_unique<alea::ScreenedCoulombTerm>(a,b,std::vector{1.0,2.0,1.0},std::vector{-5.0,5.0},2,0,0.0,0.01));
    CHECK_THROWS_AS(alea::Action(std::vector{a,b},ham,2),std::invalid_argument);
    alea::Hamiltonian weak;
    weak.add_component(std::make_unique<alea::ScreenedCoulombTerm>(
        a, b, std::vector{1e8}, std::vector<double>{}, 1, 0, 0.0, 0.1));
    CHECK_THROWS_AS(alea::Action(std::vector{a,b},weak,2),std::invalid_argument);
}

TEST_CASE("Primitive Action conservatively refuses attractive same-layer RK", "[pimc-action]") {
    auto e = group(0, 1, -1.0), h = group(1, 2, 1.0);
    const std::vector groups{e, h};
    alea::Hamiltonian singular;
    singular.add_component(std::make_unique<alea::KeldyshTerm>(e, h, 2.0, 3.0));
    CHECK_THROWS_AS(alea::Action(groups, singular, 2), std::invalid_argument);
    alea::Hamiltonian finite;
    finite.add_component(std::make_unique<alea::KeldyshTerm>(e, h, 2.0, 3.0, 0.5));
    CHECK_NOTHROW(alea::Action(groups, finite, 2));
}

TEST_CASE("Prepared open centroid measurements allocate nothing and require fresh clone workspaces", "[centroid-allocation]") {
    const std::vector groups{group(0,1,-1),group(1,2,1)};
    alea::Hamiltonian ham;
    ham.add_component(std::make_unique<alea::HarmonicTrap>(groups,1.0));
    ham.add_component(std::make_unique<alea::CoulombTerm>(groups[0],groups[1],1.7,.6));
    alea::PathSet path(16,2,2,.125);
    for (std::size_t i=0; i<path.positions().size(); ++i) path.positions()[i]=.1*std::sin(double(i));
    alea::Action action(groups,ham,2); action.prepare_open_centroid(path);
    auto clone=action.clone();
    REQUIRE_THROWS_AS(clone.measure_open_centroid(path),std::logic_error);
    clone.prepare_open_centroid(path);
    const double expected=action.measure_open_centroid(path).energy;
    const auto before=path.all_positions();
    allocation_count=0;
    count_allocations=true;
    double value=0;
    for (int i=0; i<100; ++i) {
        value+=action.measure_open_centroid(path).energy;
        value+=clone.measure_open_centroid(path).energy;
    }
    count_allocations=false;
    CHECK(allocation_count == 0);
    CHECK(value == Catch::Approx(200*expected));
    CHECK(path.all_positions() == before);
    clone.add_component(std::make_unique<ConstantAction>(0));
    REQUIRE_THROWS_AS(clone.measure_open_centroid(path),std::logic_error);
}

TEST_CASE("Prepared periodic virial measurements allocate nothing and invalidate on composition changes", "[periodic-allocation]") {
    const std::vector groups{group(0,1,-1),group(1,2,1)};
    alea::Hamiltonian ham;
    ham.add_component(std::make_unique<alea::CoulombTerm>(groups[0],groups[1],1.7,.6));
    const alea::SimulationCell cell(2,std::vector{4.,0.,1.,4.});
    alea::PathSet path(16,2,2,.125,cell);
    for (std::size_t i=0; i<path.positions().size(); ++i) path.positions()[i]=.3*std::sin(double(i));
    alea::Action action(groups,ham,2); action.prepare_periodic_virial(path);
    auto clone=action.clone();
    REQUIRE_THROWS_AS(clone.measure_periodic_virial(path),std::logic_error);
    clone.prepare_periodic_virial(path);
    const double expected=action.measure_periodic_virial(path).energy;
    const auto before=path.all_positions();
    allocation_count=0; count_allocations=true;
    double value=0;
    for (int i=0; i<100; ++i) {
        value+=action.measure_periodic_virial(path).energy;
        value+=clone.measure_periodic_virial(path).energy;
    }
    count_allocations=false;
    CHECK(allocation_count == 0); CHECK(value == Catch::Approx(200*expected));
    CHECK(path.all_positions() == before);
    clone.add_component(std::make_unique<ConstantAction>(0));
    REQUIRE_THROWS_AS(clone.measure_periodic_virial(path),std::logic_error);
}

TEST_CASE("Prepared RK virial measurements and clones allocate nothing on their first and repeated calls", "[rk-allocation]") {
    const std::vector groups{group(0,1,-1),group(1,2,1)};
    for (bool periodic : {false,true}) {
        alea::Hamiltonian ham;
        if (!periodic) ham.add_component(std::make_unique<alea::HarmonicTrap>(groups,1.));
        ham.add_component(std::make_unique<alea::KeldyshTerm>(groups[0],groups[1],2.,.7,.6));
        const auto cell=periodic ? alea::SimulationCell(2,std::vector{4.,0.,1.,4.}) : alea::SimulationCell{};
        alea::PathSet path(16,2,2,.125,cell);
        for (std::size_t i=0; i<path.positions().size(); ++i) path.positions()[i]=.3*std::sin(double(i));
        alea::Action action(groups,ham,2);
        if (periodic) action.prepare_periodic_virial(path); else action.prepare_open_centroid(path);
        auto clone=action.clone();
        auto measure=[&](alea::Action& prepared) {
            return periodic ? prepared.measure_periodic_virial(path).energy : prepared.measure_open_centroid(path).energy;
        };
        REQUIRE_THROWS_AS(measure(clone),std::logic_error);
        if (periodic) clone.prepare_periodic_virial(path); else clone.prepare_open_centroid(path);
        const auto before=path.all_positions();
        allocation_count=0; count_allocations=true;
        double value=0,expected=0;
        try {
            expected=measure(action);
            for (int i=0; i<100; ++i) { value+=measure(action); value+=measure(clone); }
        } catch (...) { count_allocations=false; throw; }
        count_allocations=false;
        CHECK(allocation_count==0);
        CHECK(value==Catch::Approx(200*expected));
        CHECK(path.all_positions()==before);
    }
}

// Speed gate G3: with exchange, the prepared
// permutation move, the cycle centroid move and the bisection/single-slice
// moves on permuted paths allocate nothing (scratch is sized by prepare()).
TEST_CASE("Prepared permutation and cycle moves allocate nothing",
          "[pair-allocation][permutation]") {
    alea::ParticleGroup e;
    e.first = 0; e.last = 3; e.count = 3; e.mass = {1., 1.}; e.inv_mass = {1., 1.}; e.charge = -1.;
    const std::vector groups{e};
    alea::Hamiltonian ham;
    ham.add_component(std::make_unique<alea::HarmonicTrap>(groups, 1.));
    ham.add_component(std::make_unique<alea::CoulombTerm>(e, e, 2.));
    alea::Action action(groups, ham, 2);
    alea::PathSet path(8, 3, 2, .125);
    path.set_exchange_classes(std::vector<std::size_t>{5, 5, 5}, std::vector<char>{1, 1, 1});
    for (std::size_t i = 0; i < path.positions().size(); ++i) path.positions()[i] = .4 * std::sin(.7 * i);
    path.rebuild_tables();
    path.set_permutation(std::vector<std::size_t>{1, 2, 0});
    action.prepare(path);
    alea::PermutationMove permutation(2, 3);
    alea::CentroidMove centroid(.2);
    alea::BisectionMove bisection(2);
    alea::SingleSliceMove single(.1);
    permutation.prepare(path); centroid.prepare(path); bisection.prepare(path); single.prepare(path);
    alea::numerics::Random rng(2718);
    std::size_t permutations = 0;
    allocation_count = 0; count_allocations = true;
    double energy = 0;
    try {
        for (int i = 0; i < 400; ++i) {
            permutations += permutation.attempt(path, action, rng);
            centroid.attempt(path, action, rng);
            bisection.attempt(path, action, rng);
            single.attempt(path, action, rng);
            energy += action.measure_thermo(path).energy;
        }
    } catch (...) { count_allocations = false; throw; }
    count_allocations = false;
    CHECK(allocation_count == 0);
    CHECK(permutations > 0);
    CHECK(std::isfinite(energy));
}

// G3 for the joint e+h move: prepared, it allocates nothing either.
TEST_CASE("Prepared joint permutation moves allocate nothing",
          "[pair-allocation][permutation]") {
    alea::ParticleGroup e, h;
    e.first = 0; e.last = 3; e.count = 3; e.mass = {1., 1.}; e.inv_mass = {1., 1.}; e.charge = -1.;
    h = e; h.first = 3; h.last = 6; h.mass = {2., 2.}; h.inv_mass = {.5, .5}; h.charge = 1.;
    const std::vector groups{e, h};
    alea::Hamiltonian ham;
    ham.add_component(std::make_unique<alea::HarmonicTrap>(groups, 1.));
    ham.add_component(std::make_unique<alea::CoulombTerm>(e, h, 2., 1.));
    alea::Action action(groups, ham, 2);
    alea::PathSet path(8, 6, 2, .125);
    path.set_exchange_classes(std::vector<std::size_t>{5, 5, 5, 6, 6, 6},
                              std::vector<char>{1, 1, 1, 1, 1, 1});
    for (std::size_t i = 0; i < path.positions().size(); ++i) path.positions()[i] = .4 * std::sin(.7 * i);
    path.rebuild_tables();
    action.prepare(path);
    alea::JointPermutationMove joint(2, 3, 1.0, {{{0, 1, 2}, {3, 4, 5}}});
    joint.prepare(path);
    alea::numerics::Random rng(1414);
    std::size_t accepted = 0;
    allocation_count = 0; count_allocations = true;
    try {
        for (int i = 0; i < 400; ++i) accepted += joint.attempt(path, action, rng);
    } catch (...) { count_allocations = false; throw; }
    count_allocations = false;
    CHECK(allocation_count == 0);
    CHECK(accepted > 0);
}

// G3 for the worm: prepared, open/close/swap/regrowth attempts and
// open-end samples allocate nothing (exchange_successors only swaps rows).
TEST_CASE("Prepared worm moves allocate nothing", "[pair-allocation][worm]") {
    alea::ParticleGroup e, h;
    e.first = 0; e.last = 3; e.count = 3; e.mass = {1., 1.}; e.inv_mass = {1., 1.}; e.charge = -1.;
    h = e; h.first = 3; h.last = 6; h.mass = {2., 2.}; h.inv_mass = {.5, .5}; h.charge = 1.;
    const std::vector groups{e, h};
    alea::Hamiltonian ham;
    ham.add_component(std::make_unique<alea::HarmonicTrap>(groups, 1.));
    ham.add_component(std::make_unique<alea::CoulombTerm>(e, h, 2., 1.));
    alea::Action action(groups, ham, 2);
    alea::PathSet path(16, 6, 2, .0625);
    path.set_exchange_classes(std::vector<std::size_t>{5, 5, 5, 6, 6, 6},
                              std::vector<char>{1, 1, 1, 1, 1, 1});
    for (std::size_t i = 0; i < path.positions().size(); ++i) path.positions()[i] = .3 * std::sin(.7 * i);
    path.rebuild_tables();
    action.prepare(path);
    alea::WormMove worm({{0, 1, 2}}, 4, 3, 0.05);
    worm.set_selection_probability(0.5);
    worm.prepare(path);
    alea::numerics::Random rng(1717);
    std::vector<double> displacement(4 * 2), weight(4);
    std::size_t accepted = 0, open = 0;
    allocation_count = 0; count_allocations = true;
    try {
        for (int i = 0; i < 2000; ++i) {
            accepted += worm.attempt(path, action, rng);
            if (path.is_open()) {
                ++open;
                worm.sample_open_ends(path, action, rng, displacement, weight);
            }
        }
    } catch (...) { count_allocations = false; throw; }
    count_allocations = false;
    CHECK(allocation_count == 0);
    CHECK(accepted > 0);
    CHECK(open > 0);
}
