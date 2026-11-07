// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Independent traces and torus integration checks for the sampled MIC action.
// Broad-link traces deliberately bypass the production narrow-link gate: they
// expose boundary terms that a full-image winding formula would miss.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <string_view>
#include <yaml-cpp/yaml.h>
#include "hamiltonian/coulomb.h"
#include "hamiltonian/harmonic_trap.h"
#include "hamiltonian/moire.h"
#include "hamiltonian/rytova_keldysh.h"
#include "io/config_writer.h"
#include "io/parser.h"
#include "pimc/pimc.h"
#include "qmc/app.h"

using namespace alea;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;
namespace {
ParticleGroup particle(std::size_t p,std::vector<double> mass,double charge=0) {
    ParticleGroup g; g.name="p"+std::to_string(p); g.first=p; g.last=p+1;
    g.count=1; g.mass=mass; g.charge=charge;
    for (double m : mass) g.inv_mass.push_back(1/m);
    return g;
}
double scalar(const RunResult& result,const std::string& key) {
    for (const auto& [name,value] : result.scalars) if (name == key) return value;
    throw std::runtime_error("missing scalar "+key);
}
std::string metadata(const RunResult& result,const std::string& key) {
    for (const auto& [name,value] : result.metadata) if (name == key) return value;
    throw std::runtime_error("missing metadata "+key);
}
const std::vector<double>& series(const RunResult& result,std::string_view key) {
    for (const auto& [name,value] : result.series) if (name == key) return value;
    throw std::runtime_error("missing series "+std::string(key));
}
double fold(double x,double L) { return x-L*std::floor(x/L+.5); }
double moire_value(double x) {
    double value=.13;
    for (int n=1; n<=3; ++n) value+=.16*std::cos(n*2*std::numbers::pi*x/4+.31);
    return value;
}
std::unique_ptr<MoireTerm> moire_1d() {
    const double g=2*std::numbers::pi/4;
    return std::make_unique<MoireTerm>(1,.13,.08,.31,
        std::vector<std::vector<double>>{{g},{2*g},{3*g}},
        std::vector<std::pair<std::size_t,std::size_t>>{{0,1}});
}
// A transfer-matrix trace, independently evaluated without production action,
// gradients, or estimator code. Normalization beta^(-M/2) must be differentiated.
double log_trace(int n,int M,double beta,bool potential) {
    const double tau=beta/M,h=4.0/n;
    double sum=0;
    for (int i=0; i<n; ++i) for (int j=0; j<n; ++j) {
        const double x=(i+.0)*h,y=j*h;
        if (M == 2) {
            const double delta=fold(y-x,4);
            const double v=potential ? moire_value(x)+moire_value(y) : 0;
            sum+=std::exp(-delta*delta/tau-tau*v);
        } else {
            const double a=fold(x,4),b=fold(y-x,4),c=fold(-y,4);
            sum+=std::exp(-(a*a+b*b+c*c)/(2*tau));
        }
    }
    return std::log(sum)-.5*M*std::log(tau); // beta-independent constants cancel.
}
class UnknownAction final : public ActionComponent {
public:
    double total_action(const PathSet&) override { return 0; }
    bool supports_free_bridge() const noexcept override { return true; }
    std::unique_ptr<ActionComponent> clone() const override { return std::make_unique<UnknownAction>(); }
};
class UnknownPotential final : public HamiltonianComponent {
public:
    std::string name() const override { return "unknown_periodic_fixture"; }
    bool supports_potential_gradient() const noexcept override { return true; }
    std::unique_ptr<HamiltonianComponent> clone() const override { return std::make_unique<UnknownPotential>(); }
};
class MissingGradient final : public HamiltonianComponent {
public:
    std::string name() const override { return "missing_periodic_gradient"; }
    bool supports_periodic_virial(const SimulationCell&,std::size_t) const noexcept override { return true; }
    std::unique_ptr<HamiltonianComponent> clone() const override { return std::make_unique<MissingGradient>(); }
};
}

TEST_CASE("Periodic virial integrates the two-bead truncated Gaussian including boundary correction", "[periodic-virial][physics]") {
    const std::vector groups{particle(0,{1.0})};
    Hamiltonian ham; const SimulationCell cell(1,std::vector{4.0});
    PathSet path(2,1,1,2.0,cell); Action action(groups,ham,1); action.prepare_periodic_virial(path);
    const double u=2/std::sqrt(2.0),beta=4;
    const double exact=1/(2*beta)+u*std::exp(-u*u)/(beta*std::sqrt(std::numbers::pi)*std::erf(u));
    CHECK(std::abs(exact-1/(2*beta))>.02); // W=0 still has a MIC boundary term.
    for (int n : {128,256,512}) {
        double z=0,thermal=0,virial=0;
        for (int j=0; j<=n; ++j) {
            const double x=-2+4.0*j/n;
            path.pos(0,0,0)=0; path.pos(0,0,1)=x;
            const double quadrature=(j==0 || j==n) ? 1 : (j%2 ? 4 : 2);
            const double weight=quadrature*std::exp(-x*x/2);
            z+=weight; thermal+=weight*action.measure_thermo(path).energy;
            virial+=weight*action.measure_periodic_virial(path).energy;
        }
        INFO("n=" << n << " thermal=" << thermal/z << " virial=" << virial/z << " exact=" << exact);
        CHECK(thermal/z == Approx(exact).margin(2e-9));
        CHECK(virial/z == Approx(exact).margin(2e-9));
    }
}

TEST_CASE("Periodic virial matches independent trace derivatives with winding and a periodic external potential", "[periodic-virial][physics]") {
    const std::vector groups{particle(0,{1.0})};
    const SimulationCell cell(1,std::vector{4.0});
    for (bool potential : {false,true}) {
        const int M=potential ? 2 : 3;
        const double beta=potential ? 2.4 : 6.0;
        Hamiltonian ham; if (potential) ham.add_component(moire_1d());
        PathSet path(M,1,1,beta/M,cell); Action action(groups,ham,1); action.prepare_periodic_virial(path);
        double previous_error=1;
        for (int n : {63,127,255}) {
            double z=0,mean=0,winding=0;
            for (int i=0; i<n; ++i) for (int j=0; j<n; ++j) {
                const double x=4.0*i/n,y=4.0*j/n;
                double spring=0,pot=0,W=0;
                if (M == 2) {
                    path.pos(0,0,0)=x; path.pos(0,0,1)=y;
                    const double delta=fold(y-x,4);
                    spring=delta*delta/path.tau(); pot=moire_value(x)+moire_value(y);
                } else {
                    path.pos(0,0,0)=0; path.pos(0,0,1)=x; path.pos(0,0,2)=y;
                    const double a=fold(x,4),b=fold(y-x,4),c=fold(-y,4);
                    spring=(a*a+b*b+c*c)/(2*path.tau()); W=(a+b+c)/4;
                }
                const double weight=std::exp(-spring-path.tau()*pot);
                z+=weight; mean+=weight*action.measure_periodic_virial(path).energy; winding+=weight*W*W;
            }
            double max_error=0;
            for (double step : {1e-3,3e-4,1e-4}) {
                const double reference=-(log_trace(n,M,beta+step,potential)-log_trace(n,M,beta-step,potential))/(2*step);
                max_error=std::max(max_error,std::abs(mean/z-reference));
                INFO("M=" << M << " n=" << n << " E=" << mean/z << " trace=" << reference);
            }
            CHECK(max_error<previous_error/3.0);
            previous_error=max_error;
            if (!potential) CHECK(winding/z>.03);
            if (n == 255) CHECK(max_error<3e-5);
            std::cout << "trace_1d M=" << M << " n=" << n << " mean=" << std::setprecision(17) << mean/z
                << " max_derivative_error=" << max_error << " winding2=" << winding/z << '\n';
        }
    }
}

TEST_CASE("Periodic field divergence and full action contraction agree with Cartesian finite differences", "[periodic-virial][derivatives]") {
    const std::array cells{SimulationCell(2,std::vector{4.,0.,0.,5.}),
        SimulationCell(2,std::vector{4.,0.,1.2,3.8})};
    for (const auto& cell : cells) {
        const std::vector groups{particle(0,cell.orthorhombic()?std::vector{.7,1.3}:std::vector{.7},-1),
            particle(1,cell.orthorhombic()?std::vector{2.1,.9}:std::vector{2.1},1)};
        Hamiltonian ham; ham.add_component(std::make_unique<CoulombTerm>(groups[0],groups[1],1.7,.6));
        const auto reciprocal=cell.reciprocal();
        std::vector<std::vector<double>> g(3,std::vector<double>(2));
        for (std::size_t d=0; d<2; ++d) {
            g[0][d]=reciprocal[d]; g[1][d]=reciprocal[2+d]; g[2][d]=g[0][d]+g[1][d];
        }
        ham.add_component(std::make_unique<MoireTerm>(2,.2,.11,.7,g,
            std::vector<std::pair<std::size_t,std::size_t>>{{0,2}}));
        PathSet path(7,2,2,.125,cell);
        for (std::size_t i=0; i<path.positions().size(); ++i) path.positions()[i]=.7*std::sin(.37*i)+.4*std::cos(.61*i);
        const auto saved=path.all_positions();
        Action action(groups,ham,2); action.prepare_periodic_virial(path);
        const auto pv=action.measure_periodic_virial(path),th=action.measure_thermo(path);
        PeriodicVirialField field; field.prepare(path); field.evaluate(path);
        const std::vector frozen(field.field().begin(),field.field().end());
        const double divergence=field.divergence();
        for (double step : {1e-4,3e-5,1e-5}) {
            double numeric_divergence=0;
            for (std::size_t i=0; i<saved.size(); ++i) {
                path.positions()[i]=saved[i]+step; field.evaluate(path); const double plus=field.field()[i];
                path.positions()[i]=saved[i]-step; field.evaluate(path); const double minus=field.field()[i];
                numeric_divergence+=(plus-minus)/(2*step); path.positions()[i]=saved[i];
            }
            CHECK(numeric_divergence == Approx(divergence).margin(2e-7));
            for (std::size_t i=0; i<saved.size(); ++i) path.positions()[i]=saved[i]+step*frozen[i];
            const double plus=action.total_action(path);
            for (std::size_t i=0; i<saved.size(); ++i) path.positions()[i]=saved[i]-step*frozen[i];
            const double minus=action.total_action(path);
            const double reference=th.energy+((plus-minus)/(2*step)-numeric_divergence)/(2*path.beta());
            CHECK(pv.energy == Approx(reference).margin(3e-7));
            std::copy(saved.begin(),saved.end(),path.positions().begin());
        }
        CHECK(pv.energy == Approx(pv.kinetic+pv.potential).margin(2e-14));
        CHECK(pv.potential == th.potential);
    }
}

TEST_CASE("Periodic virial integrates free rectangular anisotropic and skew isotropic two dimensional traces", "[periodic-virial][physics]") {
    const std::array cells{SimulationCell(2,std::vector{4.,0.,0.,5.}),
        SimulationCell(2,std::vector{4.,0.,1.123,3.717})};
    for (const auto& cell : cells) {
        const std::vector masses=cell.orthorhombic()?std::vector{.7,1.3}:std::vector{.7,.7};
        const std::vector groups{particle(0,masses)};
        const Hamiltonian ham; const double beta=4;
        PathSet path(2,1,2,beta/2,cell); Action action(groups,ham,2); action.prepare_periodic_virial(path);
        const auto vectors=cell.vectors();
        auto radius=[&](double x,double y) {
            // Independent brute-force lattice search, not cell.min_image.
            double best=std::numeric_limits<double>::infinity(),metric=0;
            for (int a=-3; a<=3; ++a) for (int b=-3; b<=3; ++b) {
                const double dx=x-a*vectors[0]-b*vectors[2],dy=y-a*vectors[1]-b*vectors[3];
                if (dx*dx+dy*dy<best) { best=dx*dx+dy*dy; metric=masses[0]*dx*dx+masses[1]*dy*dy; }
            }
            return metric;
        };
        for (int n : {63,127,255}) {
            double z=0,mean=0;
            std::array<double,3> partition{};
            const double step=1e-4;
            for (int i=0; i<n; ++i) for (int j=0; j<n; ++j) {
                const double f=double(i)/n-.5+.5/n,g=double(j)/n-.5+.5/n;
                const double x=f*vectors[0]+g*vectors[2],y=f*vectors[1]+g*vectors[3];
                const double metric=radius(x,y);
                const double weight=std::exp(-2*metric/beta);
                path.pos(0,0,0)=0; path.pos(0,1,0)=0; path.pos(0,0,1)=x; path.pos(0,1,1)=y;
                z+=weight; mean+=weight*action.measure_periodic_virial(path).energy;
                for (int k=0; k<3; ++k) partition[k]+=std::exp(-2*metric/(beta+(k-1)*step));
            }
            const double reference=-(std::log(partition[2]/std::pow((beta+step)/2,2))-
                std::log(partition[0]/std::pow((beta-step)/2,2)))/(2*step);
            const double error=std::abs(mean/z-reference);
            std::cout << "trace_2d ortho=" << cell.orthorhombic() << " n=" << n << " mean=" << std::setprecision(17)
                << mean/z << " reference=" << reference << " error=" << error << '\n';
            if (n == 255) CHECK(error<8e-5);
        }
    }
}

TEST_CASE("Periodic virial is invariant to bead images origin and cyclic labels and has regular limits", "[periodic-virial][geometry]") {
    const SimulationCell cell(2,std::vector{4.,0.,1.2,3.8});
    const std::vector groups{particle(0,{.7},-1),particle(1,{2.1},1)};
    Hamiltonian ham; ham.add_component(std::make_unique<CoulombTerm>(groups[0],groups[1],1.7,.6));
    PathSet path(7,2,2,.125,cell);
    for (std::size_t i=0; i<path.positions().size(); ++i) path.positions()[i]=1.4*std::sin(.37*i);
    Action action(groups,ham,2); action.prepare_periodic_virial(path);
    const auto saved=path.all_positions(); const auto expected=action.measure_periodic_virial(path);
    const auto vectors=cell.vectors();
    for (std::size_t p=0; p<2; ++p) for (std::size_t d=0; d<2; ++d) for (std::size_t k=0; k<7; ++k)
        path.pos(p,d,k)=saved[(p*2+d)*7+k]+double(int(k%3)-1)*vectors[d]+double(int((p+k)%3)-1)*vectors[2+d]+(d ? -.7 : 1.3);
    CHECK(action.measure_periodic_virial(path).energy == Approx(expected.energy).margin(2e-13));
    path.wrap_positions();
    CHECK(action.measure_periodic_virial(path).energy == Approx(expected.energy).margin(2e-13));
    for (std::size_t axis=0; axis<4; ++axis) for (std::size_t k=0; k<7; ++k)
        path.positions()[axis*7+k]=saved[axis*7+(k+3)%7];
    CHECK(action.measure_periodic_virial(path).energy == Approx(expected.energy).margin(2e-13));
    const std::vector one_group{particle(0,{1.})}; const Hamiltonian empty;
    for (std::size_t M : {1,4}) {
        PathSet ring(M,1,1,.125,SimulationCell(1,std::vector{4.}));
        for (std::size_t k=0; k<M; ++k) ring.pos(0,0,k)=double(k);
        PeriodicVirialField field; field.prepare(ring); field.evaluate(ring);
        for (double f : field.field()) CHECK(std::abs(f)<1e-15);
        CHECK(field.divergence() == Approx(M==1 ? 0 : -1).margin(1e-15));
        Action free(one_group,empty,1); free.prepare_periodic_virial(ring);
        CHECK(std::isfinite(free.measure_periodic_virial(ring).energy));
    }
    PathSet local(7,1,1,.125,SimulationCell(1,std::vector{1e5}));
    double centroid=0; for (std::size_t k=0; k<7; ++k) { local.pos(0,0,k)=std::sin(double(k)); centroid+=local.pos(0,0,k)/7; }
    PeriodicVirialField field; field.prepare(local); field.evaluate(local);
    for (std::size_t k=0; k<7; ++k) CHECK(field.field()[k] == Approx(local.pos(0,0,k)-centroid).margin(2e-9));
    CHECK(field.divergence() == Approx(6).margin(2e-8));
}

TEST_CASE("Periodic virial rejects unsupported components and discontinuous geometry before replicas", "[periodic-virial][admission]") {
    const std::vector groups{particle(0,{1.})}; const SimulationCell cell(1,std::vector{4.});
    Hamiltonian empty; PimcConfig cfg; cfg.beta=1; cfg.time_step=.125;
    cfg.energy="periodic_virial"; cfg.samples=std::numeric_limits<int>::max();
    CHECK_THROWS_WITH(Pimc(cfg,groups,empty,SimulationCell{},1,1),ContainsSubstring("requires a periodic cell"));
    Hamiltonian trap; trap.add_component(std::make_unique<HarmonicTrap>(groups,1.));
    CHECK_THROWS_WITH(Pimc(cfg,groups,trap,cell,1,1),ContainsSubstring("not admitted for harmonic_trap"));
    Hamiltonian unknown; unknown.add_component(std::make_unique<UnknownPotential>());
    CHECK_THROWS_WITH(Pimc(cfg,groups,unknown,cell,1,1),ContainsSubstring("not admitted for unknown_periodic_fixture"));
    Hamiltonian missing; missing.add_component(std::make_unique<MissingGradient>());
    CHECK_THROWS_WITH(Pimc(cfg,groups,missing,cell,1,1),ContainsSubstring("missing potential gradient in missing_periodic_gradient"));
    Hamiltonian moire; moire.add_component(moire_1d());
    CHECK_THROWS_WITH(Pimc(cfg,groups,moire,SimulationCell(1,std::vector{4.1}),1,1),ContainsSubstring("not admitted for moire"));
    Hamiltonian bad_range; bad_range.add_component(std::make_unique<MoireTerm>(1,0,.1,0,
        std::vector<std::vector<double>>{{0},{0},{0}},std::vector<std::pair<std::size_t,std::size_t>>{{0,2}}));
    CHECK_THROWS_WITH(Pimc(cfg,groups,bad_range,cell,1,1),ContainsSubstring("not admitted for moire"));
    const std::vector pair{particle(0,{1.},-1),particle(1,{2.},1)};
    Hamiltonian invalid_rk; invalid_rk.add_component(std::make_unique<KeldyshTerm>(pair[0],pair[1],2.,-3.,.5));
    CHECK_THROWS_WITH(Pimc(cfg,pair,invalid_rk,cell,1,2),ContainsSubstring("not admitted for rytova_keldysh"));
    Action composite(groups,empty,1); composite.add_component(std::make_unique<UnknownAction>());
    CHECK_THROWS_WITH(Pimc(cfg,groups,empty,cell,1,1,nullptr,&composite),ContainsSubstring("no periodic contribution"));
    const std::vector anisotropic{particle(0,{1.,2.})};
    Action skew(anisotropic,empty,2);
    CHECK_THROWS_WITH(skew.validate_periodic_virial(SimulationCell(2,std::vector{4.,0.,1.,4.}),1,2),ContainsSubstring("isotropic masses"));
    PathSet prepared(4,1,1,.125,cell); Action action(groups,moire,1); action.prepare_periodic_virial(prepared);
    PathSet other(4,1,1,.125,SimulationCell(1,std::vector{4.1}));
    CHECK_THROWS_WITH(action.measure_periodic_virial(other),ContainsSubstring("not prepared"));
    action.prepare(5); CHECK_THROWS_WITH(action.measure_periodic_virial(prepared),ContainsSubstring("not prepared"));
}

TEST_CASE("Periodic virial driver preserves sampling threads checkpoints and paired output", "[periodic-virial][driver]") {
    const std::vector groups{particle(0,{1.})}; const Hamiltonian ham;
    const SimulationCell cell(1,std::vector{4.});
    PimcConfig cfg; cfg.beta=16; cfg.time_step=.125; cfg.samples=4;
    cfg.equilibration_blocks=64; cfg.accumulation_blocks=512; cfg.steps_per_block=32; cfg.seed=20261041;
    cfg.moves={{"bisection",1.,7,.5},{"centroid",1.,3,.3}};
    Pimc thermal(cfg,groups,ham,cell,1,1); thermal.run();
    cfg.energy="periodic_virial"; cfg.compare_thermodynamic=true;
    Pimc pv(cfg,groups,ham,cell,1,1); pv.run();
    cfg.threads=4; Pimc threaded(cfg,groups,ham,cell,1,1); threaded.run();
    cfg.compare_thermodynamic=false; Pimc unpaired(cfg,groups,ham,cell,1,1); unpaired.run();
    CHECK(pv.export_paths().positions == thermal.export_paths().positions);
    CHECK(pv.export_paths().positions == threaded.export_paths().positions);
    CHECK(pv.export_paths().positions == unpaired.export_paths().positions);
    CHECK(pv.acceptance_ratio() == thermal.acceptance_ratio());
    CHECK(std::equal(pv.block_energies().begin(),pv.block_energies().end(),threaded.block_energies().begin()));
    CHECK(std::equal(pv.block_energies().begin(),pv.block_energies().end(),unpaired.block_energies().begin()));
    const auto result=pv.run_result(); const auto& differences=series(result,"acc_block_energy_difference");
    for (std::size_t b=0; b<differences.size(); ++b)
        CHECK(differences[b] == Approx(pv.block_energies()[b]-thermal.block_energies()[b]).margin(2e-14));
    const auto paired=numerics::reblock(differences);
    CHECK(scalar(result,"energy_difference_error") == paired.error);
    CHECK(std::abs(paired.mean)<4*paired.error);
    CHECK(std::abs(pv.energy()-6.6010084974e-9)<4*pv.energy_error());
    CHECK(std::abs(pv.winding2()-.999999788768)<4*pv.winding2_error());
    CHECK(pv.winding2()>.8);
    CHECK(metadata(result,"energy_estimator") == "periodic_virial");
    CHECK(metadata(result,"energy_comparison") == "paired periodic_virial-minus-thermodynamic");
    CHECK(metadata(result,"virial_geometry") == "smooth periodic fractional centroid field");
    // Checkpoint restart uses the same coordinates/RNG regardless of measurement.
    const auto checkpoint=pv.export_paths();
    cfg.energy="thermodynamic"; Pimc resume_th(cfg,groups,ham,cell,1,1); resume_th.import_paths(checkpoint); resume_th.run();
    cfg.energy="periodic_virial"; Pimc resume_pv(cfg,groups,ham,cell,1,1); resume_pv.import_paths(checkpoint); resume_pv.run();
    CHECK(resume_th.export_paths().positions == resume_pv.export_paths().positions);
    CHECK(resume_th.acceptance_ratio() == resume_pv.acceptance_ratio());
}

TEST_CASE("Periodic virial config round trips and runs six layered particles through App", "[periodic-virial][config][many-body]") {
    const std::string yaml=R"(project: {name: periodic_snapshot, seed: 11, output: false, log: false}
system:
  lattice: {type: periodic, dimension: 1, L: 4.0}
  particles:
    - {name: p, group: p, type: boltzmannon, mass: 1.0, charge: 0, up: 1, down: 0}
wavefunction: []
hamiltonian: {terms: []}
qmc:
  - {method: pimc, beta: 2.0, time_step: 0.125, energy: periodic_virial, compare_thermodynamic: true}
)";
    const auto config=io::parse_config(yaml); CHECK(config.pimc[0].energy == "periodic_virial");
    const auto directory=std::filesystem::temp_directory_path()/"alea_periodic_snapshot_20261004";
    std::filesystem::create_directories(directory/"run"); const auto source=directory/"source.yaml";
    std::ofstream(source)<<yaml; io::write_config_snapshot(source.string(),{},directory/"run",11);
    const auto replay=io::parse_config_file((directory/"run/source.yaml").string());
    CHECK(replay.pimc[0].energy == "periodic_virial"); CHECK(replay.pimc[0].compare_thermodynamic);
    std::filesystem::remove_all(directory);
    auto many=io::parse_config_file(std::string(ALEA_CONFIG_DIR)+"/pimc_many_body_eh.yaml");
    many.lattice.type="periodic"; many.lattice.L=8; many.lattice.truncation="minimum_image";
    many.traps.clear(); // Periodic interactions, without an open trap.
    App thermal_app(many); auto th_pack=thermal_app.build_drivers(); auto* th=dynamic_cast<Pimc*>(th_pack.drivers[0].get());
    REQUIRE(th != nullptr); th->run();
    many.pimc[0].energy="periodic_virial"; many.pimc[0].compare_thermodynamic=true;
    App pv_app(many); auto pv_pack=pv_app.build_drivers(); auto* pv=dynamic_cast<Pimc*>(pv_pack.drivers[0].get());
    REQUIRE(pv != nullptr); pv->run();
    CHECK(std::isfinite(pv->energy())); CHECK(pv->energy() == Approx(pv->kinetic()+pv->potential()).margin(2e-13));
    CHECK(pv->export_paths().positions == th->export_paths().positions);
    CHECK(pv->acceptance_ratio() == th->acceptance_ratio());
    CHECK(std::equal(pv->block_potential().begin(),pv->block_potential().end(),th->block_potential().begin()));
    // An infrastructure contract, not an interacting convergence reference.
}

TEST_CASE("Periodic virial utility is measured at several bead counts on identical free rings", "[periodic-utility]") {
    const std::vector groups{particle(0,{1.})}; const Hamiltonian ham;
    const SimulationCell cell(1,std::vector{4.});
    std::cout << "M,seed,thermo_seconds,pv_seconds,thermo_variance,pv_variance,thermo_sem,pv_sem,paired_mean,paired_sem,pv_energy,winding2,thermo_reliable,pv_reliable\n";
    for (int M : {16,64,256}) for (int repeat=0; repeat<3; ++repeat) {
        PimcConfig cfg; cfg.beta=2; cfg.time_step=2.0/M; cfg.samples=4;
        cfg.equilibration_blocks=64; cfg.accumulation_blocks=512; cfg.steps_per_block=32;
        cfg.seed=20264100+1000*M+100*repeat;
        cfg.moves={{"bisection",1.,int(std::log2(M)),.5},{"centroid",1.,3,.3}};
        auto started=std::chrono::steady_clock::now(); Pimc th(cfg,groups,ham,cell,1,1); th.run();
        const double th_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        cfg.energy="periodic_virial";
        started=std::chrono::steady_clock::now(); Pimc pv(cfg,groups,ham,cell,1,1); pv.run();
        const double pv_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        CHECK(th.export_paths().positions == pv.export_paths().positions);
        const auto result=pv.run_result();
        std::vector<double> differences(cfg.accumulation_blocks);
        for (std::size_t b=0; b<differences.size(); ++b) differences[b]=pv.block_energies()[b]-th.block_energies()[b];
        const auto paired=numerics::reblock(differences);
        CHECK(std::abs(paired.mean)<4*paired.error);
        CHECK(std::abs(pv.energy()-.1793245903520491)<4*pv.energy_error());
        auto variance=[](std::span<const double> blocks,double mean) {
            double sum=0; for (double x : blocks) sum+=(x-mean)*(x-mean);
            return sum/static_cast<double>(blocks.size()-1);
        };
        std::cout << std::setprecision(17) << M << ',' << cfg.seed << ',' << th_seconds << ',' << pv_seconds
            << ',' << variance(th.block_energies(),th.energy()) << ',' << variance(pv.block_energies(),pv.energy())
            << ',' << th.energy_error() << ',' << pv.energy_error() << ',' << paired.mean
            << ',' << paired.error << ',' << pv.energy() << ',' << pv.winding2()
            << ',' << scalar(th.run_result(),"energy_error_reliable") << ',' << scalar(result,"energy_error_reliable") << '\n';
        // Pilot timings are reported, never used as a pass/fail speed threshold.
    }
}
