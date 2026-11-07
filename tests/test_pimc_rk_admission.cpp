// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// RK estimator admission: physical-reference accuracy and independent finite-M
// traces are distinct checks. The retained core is a sampled approximation.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <string_view>
#include <vector>
#include "hamiltonian/harmonic_trap.h"
#include "hamiltonian/rytova_keldysh.h"
#include "hamiltonian/screened_coulomb.h"
#include "numerics/keldysh_kernel.h"
#include "pimc/pimc.h"
#include "qmc/app.h"
#include "io/parser.h"

using namespace alea;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;
namespace {
ParticleGroup particle(std::size_t first,std::size_t last,double charge,std::vector<double> mass={1.}) {
    ParticleGroup g; g.name="p"+std::to_string(first); g.first=first; g.last=last;
    g.count=last-first; g.charge=charge; g.mass=mass;
    for (double m : mass) g.inv_mass.push_back(1/m);
    return g;
}
// DLMF 11.5.3, K0(x)=(2/pi) integral exp(-x sinh t) dt.
// Return [K,x K']; scaled moments avoid large near-core derivatives.
std::array<long double,2> integral_reference(long double x,int panels) {
    const long double end=std::asinh(60/x),h=end/panels;
    std::array<long double,2> sum{};
    for (int i=0; i<=panels; ++i) {
        const long double s=x*std::sinh(i*h),weight=(i==0 || i==panels) ? 1 : (i%2 ? 4 : 2);
        const long double value=weight*std::exp(-s);
        sum[0]+=value; sum[1]-=s*value;
    }
    for (auto& value : sum) value*=h*2/(3*std::numbers::pi_v<long double>);
    return sum;
}
std::string metadata(const RunResult& result,const std::string& key) {
    for (const auto& [name,value] : result.metadata) if (name == key) return value;
    throw std::runtime_error("missing metadata "+key);
}
const std::vector<double>& series(const RunResult& result,std::string_view key) {
    for (const auto& [name,value] : result.series) if (name == key) return value;
    throw std::runtime_error("missing series "+std::string(key));
}
double pair_value(double separation,double sign) {
    return sign*std::numbers::pi/(2*4*.7)*numerics::keldysh_cheb().eval(std::hypot(separation,.6)/.7);
}
double fold(double x,double L) { return x-L*std::floor(x/L+.5); }
}

TEST_CASE("RK sampled value and force meet explicit integral reference budgets outside the frozen core", "[rk-admission][accuracy]") {
    const auto& kernel=numerics::keldysh_cheb();
    std::vector<double> points;
    for (int i=0; i<=512; ++i) points.push_back(std::pow(10.,-3.+9.*i/512));
    for (int i=0; i<=128; ++i) points.push_back(32.+8.*i/128);
    points.push_back(.001); points.push_back(.00101); points.push_back(.0011);
    double max_value=0,max_force=0,max_refinement=0,worst_x=0;
    for (double x : points) {
        const auto coarse=integral_reference(x,16384),fine=integral_reference(x,32768);
        const double refinement=std::abs(double((fine[1]-coarse[1])/fine[1]));
        const double value_error=std::abs(kernel.eval(x)-double(fine[0]));
        const double force_error=std::abs(x*kernel.d1(x)-double(fine[1]))/std::abs(double(fine[1]));
        INFO("x=" << x << " value error=" << value_error << " relative force error=" << force_error);
        CHECK(refinement<5e-12);
        CHECK(value_error<5e-9);
        CHECK(force_error<5e-6);
        max_value=std::max(max_value,value_error); max_refinement=std::max(max_refinement,refinement);
        if (force_error>max_force) { max_force=force_error; worst_x=x; }
    }
    std::cout << std::setprecision(17) << "RK reference max_value=" << max_value << " max_force_relative=" << max_force
        << " worst_x=" << worst_x << " max_refinement=" << max_refinement << '\n';
    for (double x : {0.,1e-8,1e-4,std::nextafter(.001,0.)}) {
        CHECK(kernel.eval(x) == kernel.eval(.001)); CHECK(kernel.d1(x) == 0.);
    }
    CHECK(kernel.d1(.001)<0); // Interior convention at a continuous slope kink.
}

TEST_CASE("RK physical force budgets scale correctly with charges dielectric screening and layer offset", "[rk-admission][accuracy]") {
    for (double r0 : {.02,1.7,20.}) for (double epsilon : {.6,3.2}) for (std::size_t dim : {1,2,3}) {
        const auto a=particle(0,1,-2.),b=particle(1,2,3.);
        KeldyshTerm pair(a,b,epsilon,r0,.01*r0);
        PathSet path(3,2,dim,.125);
        for (std::size_t k=0; k<3; ++k) path.pos(1,0,k)=r0*std::array{.03,1.,36.}[k];
        path.rebuild_tables(); auto workspace=pair.make_potential_gradient_workspace(path);
        std::vector<double> gradient(path.positions().size()); // [np*dim*M]
        pair.evaluate_potential_gradient(path,gradient,*workspace);
        for (std::size_t k=0; k<3; ++k) {
            const double radius=std::hypot(path.pos(1,0,k),.01*r0),x=radius/r0;
            const auto ref=integral_reference(x,32768);
            const double expected=(-6.)*std::numbers::pi/(2*epsilon*r0)*double(ref[1])/radius*(-path.pos(1,0,k)/radius);
            CHECK(gradient[k] == Approx(expected).epsilon(5e-6));
            CHECK(gradient[dim*3+k] == -gradient[k]);
            for (std::size_t d=1; d<dim; ++d) CHECK(gradient[d*3+k] == 0.);
        }
    }
}

TEST_CASE("RK admission preserves primitive attraction and confinement guards and rejects invalid scaling", "[rk-admission][admission]") {
    const auto a=particle(0,1,-1.),b=particle(1,2,1.); const std::vector groups{a,b};
    KeldyshTerm bounded(a,b,2.,.7,.6),repulsive(a,particle(1,2,-1.),2.,.7);
    CHECK(bounded.supports_open_centroid_virial()); CHECK(repulsive.supports_open_centroid_virial());
    const SimulationCell cell(1,std::vector{4.});
    CHECK(bounded.supports_periodic_virial(cell,2)); CHECK_FALSE(bounded.supports_periodic_virial(cell,1));
    CHECK_FALSE(bounded.supports_periodic_virial(SimulationCell{},2));
    for (double epsilon : {0.,-1.,std::numeric_limits<double>::max(),std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})
        CHECK_FALSE(KeldyshTerm(a,b,epsilon,.7,.6).supports_open_centroid_virial());
    for (double r0 : {0.,-1.,1e-310,std::numeric_limits<double>::infinity()})
        CHECK_FALSE(KeldyshTerm(a,b,2.,r0,.6).supports_open_centroid_virial());
    CHECK_FALSE(KeldyshTerm(a,b,2.,.7,std::numeric_limits<double>::infinity()).supports_open_centroid_virial());
    CHECK_FALSE(KeldyshTerm(particle(0,2,1.),particle(1,3,1.),2.,.7).supports_open_centroid_virial());
    CHECK(KeldyshTerm(particle(0,1,1e150),particle(1,2,1e150),1.,.01).supports_open_centroid_virial());
    CHECK_FALSE(KeldyshTerm(particle(0,1,1e152),particle(1,2,1e152),1.,.01).supports_open_centroid_virial());
    // K'~ -636 near the core: coefficient*K' overflows even when
    // coefficient*K'/r0 would be finite. Setup must reject before sampling.
    CHECK_FALSE(KeldyshTerm(particle(0,1,1e153),particle(1,2,1e153),.001,1000.).supports_open_centroid_virial());
    Hamiltonian singular; singular.add_component(std::make_unique<KeldyshTerm>(a,b,2.,.7));
    CHECK_THROWS_WITH(Action(groups,singular,1),ContainsSubstring("unsupported attractive singularity"));
    Hamiltonian ham; ham.add_component(bounded.clone());
    PimcConfig cfg; cfg.beta=1; cfg.time_step=.125; cfg.energy="centroid_virial";
    cfg.samples=std::numeric_limits<int>::max();
    CHECK_THROWS_WITH(Pimc(cfg,groups,ham,SimulationCell{},1,2),ContainsSubstring("absolute open confinement"));
    Hamiltonian screened; screened.add_component(std::make_unique<HarmonicTrap>(groups,1.));
    screened.add_component(std::make_unique<ScreenedCoulombTerm>(a,b,std::vector{1.5,1.5,1.5,1.5},
        std::vector{-1.,1.,3.},2,3,2.,.01));
    cfg.samples=1;
    CHECK_NOTHROW(Pimc(cfg,groups,screened,SimulationCell{},1,2));
}

TEST_CASE("Confined RK centroid integrates an independent two bead interacting partition derivative", "[rk-admission][physics]") {
    const double beta=2.,omega=1.,mu=.5,Q=12.;
    for (double sign : {-1.,1.}) {
        const auto a=particle(0,1,-1.),b=particle(1,2,-sign); const std::vector groups{a,b};
        Hamiltonian ham; ham.add_component(std::make_unique<HarmonicTrap>(groups,omega*omega));
        ham.add_component(std::make_unique<KeldyshTerm>(a,b,4.,.7,.6));
        PathSet path(2,2,1,beta/2); Action action(groups,ham,1); action.prepare_open_centroid(path);
        // Centre of mass and relative path separate. Integrate CM analytically
        // and q0,q1 by Simpson quadrature. The independent trace reads the
        // unchanged radial VALUE only, never Action or gradient/energy code.
        const double x=(beta/2)*omega/2;
        const double cm_energy=omega/(2*std::sqrt(1+x*x)*std::tanh(2*std::asinh(x)));
        for (int n : {128,256,512}) {
            std::vector<double> q(n+1),v(n+1),weights(n+1); // [relative grid point]
            for (int i=0; i<=n; ++i) {
                q[i]=-Q+2*Q*i/n; v[i]=.5*mu*omega*omega*q[i]*q[i]+pair_value(q[i],sign);
                weights[i]=(i==0 || i==n) ? 1 : (i%2 ? 4 : 2);
            }
            double z=0,mean=0;
            std::array<double,6> partition{}; // [beta minus/plus for three steps]
            const std::array steps{1e-3,3e-4,1e-4};
            for (int i=0; i<=n; ++i) for (int j=0; j<=n; ++j) {
                const double spring=mu*(q[j]-q[i])*(q[j]-q[i]),potential=v[i]+v[j];
                const double weight=weights[i]*weights[j]*std::exp(-spring/(beta/2)-(beta/2)*potential);
                path.pos(0,0,0)=q[i]/2; path.pos(1,0,0)=-q[i]/2;
                path.pos(0,0,1)=q[j]/2; path.pos(1,0,1)=-q[j]/2;
                z+=weight; mean+=weight*action.measure_open_centroid(path).energy;
                for (std::size_t s=0; s<steps.size(); ++s) for (int side=0; side<2; ++side) {
                    const double tau=(beta+(side ? 1 : -1)*steps[s])/2;
                    partition[2*s+side]+=weights[i]*weights[j]*std::exp(-spring/tau-tau*potential)/tau;
                }
            }
            const double integrated=mean/z+cm_energy-1/(2*beta);
            double max_error=0;
            for (std::size_t s=0; s<steps.size(); ++s) {
                const double reference=cm_energy-(std::log(partition[2*s+1])-std::log(partition[2*s]))/(2*steps[s]);
                max_error=std::max(max_error,std::abs(integrated-reference));
            }
            std::cout << std::setprecision(17) << "RK open sign=" << sign << " n=" << n << " E=" << integrated << " trace_error=" << max_error << '\n';
            if (n==512) CHECK(max_error<2e-7);
        }
    }
}

TEST_CASE("Periodic RK virial integrates an independent interacting torus trace", "[rk-admission][physics]") {
    const auto a=particle(0,1,-1.),b=particle(1,2,1.,{1.3}); const std::vector groups{a,b};
    const double beta=.25,L=4.; const SimulationCell cell(1,std::vector{L});
    Hamiltonian ham; ham.add_component(std::make_unique<KeldyshTerm>(a,b,4.,.7,.6));
    PathSet path(2,2,1,beta/2,cell); Action action(groups,ham,1); action.prepare_periodic_virial(path);
    for (int n : {31,63,127}) {
        // Global translation allows fixing bead p0,k0=0. Independent Gaussian
        // link/pair VALUE tables reduce reference cost without production action.
        std::vector<double> coordinate(n),delta2(n),pair(n); // [periodic grid difference]
        for (int i=0; i<n; ++i) {
            coordinate[i]=L*i/n; const double delta=fold(coordinate[i],L);
            delta2[i]=delta*delta; pair[i]=pair_value(delta,-1.);
        }
        double z=0,mean=0; std::array<double,2> partition{};
        const double step=1e-5;
        for (int i=0; i<n; ++i) for (int j=0; j<n; ++j) for (int k=0; k<n; ++k) {
            const double spring=delta2[i]+1.3*delta2[(k-j+n)%n],potential=pair[j]+pair[(k-i+n)%n];
            const double weight=std::exp(-spring/(beta/2)-(beta/2)*potential);
            path.pos(0,0,0)=0; path.pos(0,0,1)=coordinate[i]; path.pos(1,0,0)=coordinate[j]; path.pos(1,0,1)=coordinate[k];
            z+=weight; mean+=weight*action.measure_periodic_virial(path).energy;
            for (int side=0; side<2; ++side) {
                const double tau=(beta+(side ? 1 : -1)*step)/2;
                partition[side]+=std::exp(-spring/tau-tau*potential)/(tau*tau);
            }
        }
        const double reference=-(std::log(partition[1])-std::log(partition[0]))/(2*step);
        const double error=std::abs(mean/z-reference);
        std::cout << std::setprecision(17) << "RK periodic n=" << n << " E=" << mean/z << " trace=" << reference << " error=" << error << '\n';
        if (n==127) CHECK(error<5e-5);
    }
}

TEST_CASE("RK virial directions include the core kink blend tail and periodic image faces", "[rk-admission][derivatives]") {
    const auto a=particle(0,1,-1.),b=particle(1,2,-1.,{1.3}); const std::vector groups{a,b};
    for (double radius : {0.,.0005,.001,1.,31.9,32.,34.,39.9,40.,70.}) {
        Hamiltonian ham; ham.add_component(std::make_unique<HarmonicTrap>(groups,.2));
        ham.add_component(std::make_unique<KeldyshTerm>(a,b,2.,1.));
        PathSet path(2,2,1,.125);
        // Coincident/collapsed rings have zero fluctuation field, including
        // the low-clamp kink; derivatives are compared away from that kink.
        path.pos(1,0,0)=radius; path.pos(1,0,1)=radius==.001 ? radius : radius*1.001;
        Action action(groups,ham,1); action.prepare_open_centroid(path);
        const auto value=action.measure_open_centroid(path); const auto saved=path.all_positions();
        auto log_weight=[&](double ratio) {
            PathSet mapped(2,2,1,path.tau()*ratio);
            for (std::size_t p=0; p<2; ++p) {
                const double centroid=(saved[p*2]+saved[p*2+1])/2;
                for (std::size_t k=0; k<2; ++k) mapped.pos(p,0,k)=centroid+std::sqrt(ratio)*(saved[p*2+k]-centroid);
            }
            return -std::log(ratio)-action.total_action(mapped);
        };
        for (double step : {1e-4,3e-5,1e-5})
            CHECK(value.energy == Approx(-(log_weight(1+step)-log_weight(1-step))/(2*step*path.beta())).margin(3e-5));
        CHECK(path.all_positions()==saved);
    }
    const std::array cells{SimulationCell(2,std::vector{4.,0.,0.,5.}),SimulationCell(2,std::vector{4.,0.,1.2,3.8})};
    for (const auto& cell : cells) {
        const std::vector mixed{particle(0,1,-1.,cell.orthorhombic()?std::vector{.8,1.7}:std::vector{.8}),
            particle(1,2,1.,cell.orthorhombic()?std::vector{1.9,.6}:std::vector{1.9})};
        Hamiltonian ham; ham.add_component(std::make_unique<KeldyshTerm>(mixed[0],mixed[1],2.,.07,.03));
        PathSet path(7,2,2,.125,cell);
        for (std::size_t i=0; i<path.positions().size(); ++i) path.positions()[i]=1.7*std::sin(.37*i);
        Action action(mixed,ham,2); action.prepare_periodic_virial(path);
        PeriodicVirialField field; field.prepare(path); field.evaluate(path);
        const auto saved=path.all_positions(); const std::vector frozen(field.field().begin(),field.field().end());
        const double divergence=field.divergence(),th=action.measure_thermo(path).energy,pv=action.measure_periodic_virial(path).energy;
        for (double step : {1e-4,3e-5,1e-5}) {
            for (std::size_t i=0; i<saved.size(); ++i) path.positions()[i]=saved[i]+step*frozen[i];
            const double plus=action.total_action(path);
            for (std::size_t i=0; i<saved.size(); ++i) path.positions()[i]=saved[i]-step*frozen[i];
            const double minus=action.total_action(path);
            CHECK(pv == Approx(th+((plus-minus)/(2*step)-divergence)/(2*path.beta())).margin(3e-7));
            std::copy(saved.begin(),saved.end(),path.positions().begin());
        }
        const auto vectors=cell.vectors();
        for (std::size_t p=0; p<2; ++p) for (std::size_t d=0; d<2; ++d) for (std::size_t k=0; k<7; ++k)
            path.pos(p,d,k)+=vectors[d]*double(int(k%3)-1)+vectors[2+d]*double(int((p+k)%3)-1);
        CHECK(action.measure_periodic_virial(path).energy == Approx(pv).margin(3e-13));
    }
}

TEST_CASE("Six particle RK App runs preserve sampling and report the sampled approximation", "[rk-admission][driver]") {
    for (bool periodic : {false,true}) {
        auto config=io::parse_config_file(std::string(ALEA_CONFIG_DIR)+"/pimc_many_body_eh.yaml");
        config.coulombs.clear();
        config.keldysh={{ {"electrons","electrons"},2.,.7 },{{"holes","holes"},3.,.8},{{"electrons","holes"},4.,.9}};
        if (periodic) { config.lattice.type="periodic"; config.lattice.L=8.; config.lattice.truncation="minimum_image"; config.traps.clear(); }
        config.pimc[0].equilibration_blocks=64; config.pimc[0].accumulation_blocks=256;
        App th_app(config); auto th_pack=th_app.build_drivers(); auto* th=dynamic_cast<Pimc*>(th_pack.drivers[0].get());
        REQUIRE(th != nullptr); th->run();
        config.pimc[0].energy=periodic ? "periodic_virial" : "centroid_virial"; config.pimc[0].compare_thermodynamic=true;
        App pv_app(config); auto pv_pack=pv_app.build_drivers(); auto* pv=dynamic_cast<Pimc*>(pv_pack.drivers[0].get());
        REQUIRE(pv != nullptr); pv->run();
        config.pimc[0].threads=4;
        App threaded_app(config); auto threaded_pack=threaded_app.build_drivers(); auto* threaded=dynamic_cast<Pimc*>(threaded_pack.drivers[0].get());
        REQUIRE(threaded != nullptr); threaded->run();
        CHECK(pv->export_paths().positions==th->export_paths().positions);
        CHECK(pv->export_paths().positions==threaded->export_paths().positions);
        CHECK(pv->acceptance_ratio()==th->acceptance_ratio());
        CHECK(std::equal(pv->block_energies().begin(),pv->block_energies().end(),threaded->block_energies().begin()));
        CHECK(std::equal(pv->block_potential().begin(),pv->block_potential().end(),th->block_potential().begin()));
        CHECK(pv->energy()==Approx(pv->kinetic()+pv->potential()).margin(2e-13));
        const auto result=pv->run_result(); const auto paired=numerics::reblock(series(result,"acc_block_energy_difference"));
        std::cout << std::setprecision(17) << "RK driver periodic=" << periodic << " paired_mean=" << paired.mean
            << " paired_SEM=" << paired.error << '\n';
        CHECK(std::abs(paired.mean)<4*paired.error);
        CHECK_THAT(metadata(result,"potential_approximation"),ContainsSubstring("rho/r0<0.001 frozen"));
        CHECK_THAT(metadata(result,"potential_approximation"),ContainsSubstring("fit/core error excluded"));
        CHECK_THAT(th->summary(),ContainsSubstring("rho_core="));
    }
    const auto a=particle(0,1,-1.),b=particle(1,2,1.); const std::vector groups{a,b};
    Hamiltonian ham; ham.add_component(std::make_unique<KeldyshTerm>(a,b,2.,.7,.6));
    Action without_potential(groups,ham,1,true,false);
    CHECK(without_potential.potential_approximation_notes().empty());
    Action with_potential(groups,ham,1);
    CHECK(with_potential.clone().potential_approximation_notes()==with_potential.potential_approximation_notes());
}
