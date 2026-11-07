// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Represented finite-M identities and physical references are separate gates.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
#include "hamiltonian/ewald_term.h"
#include "hamiltonian/harmonic_trap.h"
#include "hamiltonian/screened_coulomb.h"
#include "pimc/pimc.h"
using namespace alea;
using Catch::Approx;
namespace {
ParticleGroup group(std::size_t p,double q) {
    ParticleGroup g; g.first=p; g.last=p+1; g.count=1;
    g.mass={1.}; g.inv_mass={1.}; g.charge=q; return g;
}
double value(HamiltonianComponent& term,PathSet& path) {
    path.rebuild_tables(); std::vector<double> out(path.num_slices()); // [M]
    term.evaluate_potential(path,out);
    return out[0];
}
// Independent erfc-split slab Ewald (Parry 1975), not the optimized spline.
double parry(double x,double y,double L,double epsilon,double offset,double alpha) {
    const double pi=std::numbers::pi,A=L*L;
    const std::array charges{-1.,1.},z{0.,offset};
    const std::array positions{0.,0.,x,y};
    double energy=0;
    for (int i=0; i<2; ++i) for (int j=0; j<2; ++j) {
        const double dx=positions[2*i]-positions[2*j],dy=positions[2*i+1]-positions[2*j+1],dz=z[i]-z[j],qq=charges[i]*charges[j];
        for (int a=-4; a<=4; ++a) for (int b=-4; b<=4; ++b) {
            if (i==j && a==0 && b==0) continue;
            const double r=std::sqrt(std::pow(dx+L*a,2)+std::pow(dy+L*b,2)+dz*dz);
            energy+=.5*qq*std::erfc(alpha*r)/r;
        }
        for (int a=-24; a<=24; ++a) for (int b=-24; b<=24; ++b) {
            if (a==0 && b==0) continue;
            const double kx=2*pi*a/L,ky=2*pi*b/L,k=std::hypot(kx,ky);
            const double factor=std::exp(k*dz)*std::erfc(k/(2*alpha)+alpha*dz)+
                                std::exp(-k*dz)*std::erfc(k/(2*alpha)-alpha*dz);
            energy+=pi/(2*A)*qq*std::cos(kx*dx+ky*dy)*factor/k;
        }
        energy-=pi/A*qq*(dz*std::erf(alpha*dz)+std::exp(-alpha*alpha*dz*dz)/(alpha*std::sqrt(pi)));
    }
    return (energy-2*alpha/std::sqrt(pi))/epsilon;
}
}

TEST_CASE("FSC centroid integrates a clamped uniform slab two bead trace", "[kernel-admission][physics]") {
    const auto a=group(0,-1.),b=group(1,1.); const std::vector groups{a,b};
    const double beta=2,mu=.5,omega=1,Q=12;
    Hamiltonian ham; ham.add_component(std::make_unique<HarmonicTrap>(groups,1.));
    ham.add_component(std::make_unique<ScreenedCoulombTerm>(a,b,std::vector{4.,4.,4.},std::vector{-1.,1.},2,3,2.,.01));
    PathSet path(2,2,1,beta/2); Action action(groups,ham,1); action.prepare_open_centroid(path);
    const double x=beta*omega/4,cm=omega/(2*std::sqrt(1+x*x)*std::tanh(2*std::asinh(x)));
    double previous=0;
    for (int n : {128,256,512}) {
        std::vector<double> q(n+1),v(n+1),w(n+1); // [relative coordinate]
        for (int i=0; i<=n; ++i) {
            q[i]=-Q+2*Q*i/n;
            v[i]=.5*mu*q[i]*q[i]-1/(4*std::hypot(std::max(std::abs(q[i]),.01),2.));
            w[i]=(i==0 || i==n)?1:(i%2?4:2);
        }
        double norm=0,energy=0; std::array<double,2> z{}; const double h=1e-4;
        for (int i=0; i<=n; ++i) for (int j=0; j<=n; ++j) {
            const double spring=mu*std::pow(q[i]-q[j],2),potential=v[i]+v[j];
            const double weight=w[i]*w[j]*std::exp(-spring/(beta/2)-(beta/2)*potential);
            path.pos(0,0,0)=q[i]/2; path.pos(1,0,0)=-q[i]/2;
            path.pos(0,0,1)=q[j]/2; path.pos(1,0,1)=-q[j]/2;
            norm+=weight; energy+=weight*action.measure_open_centroid(path).energy;
            for (int side=0; side<2; ++side) {
                const double tau=(beta+(side?1:-1)*h)/2;
                z[side]+=w[i]*w[j]*std::exp(-spring/tau-tau*potential)/tau;
            }
        }
        const double measured=energy/norm+cm-1/(2*beta),reference=cm-(std::log(z[1])-std::log(z[0]))/(2*h);
        std::cout << "FSC trace n=" << n << " E=" << measured << " trace_error=" << measured-reference << '\n';
        if (n==512) { CHECK(std::abs(measured-reference)<2e-7); CHECK(std::abs(measured-previous)<2e-6); }
        previous=measured;
    }
}

TEST_CASE("Ewald values and forces meet independent slab references under tolerance refinement", "[kernel-admission][accuracy]") {
    const auto a=group(0,-1.),b=group(1,1.); const std::vector groups{a,b};
    const double L=4,epsilon=1.5,offset=.7; const SimulationCell cell(2,std::vector{L,0.,0.,L});
    const auto direct=numerics::PotentialSpec::coulomb(epsilon),cross=numerics::PotentialSpec::coulomb(epsilon,offset);
    for (bool screened : {false,true}) for (double tol : {1e-6,1e-8}) {
        auto intra=direct,inter=cross;
        if (screened) {
            intra=build_screened_ewald_kernel({epsilon,epsilon,epsilon},{-1.,1.},2,0,0.,cell.ws_radius());
            inter=build_screened_ewald_kernel({epsilon,epsilon,epsilon},{-1.,1.},2,3,2.,cell.ws_radius());
        }
        const double separation=screened?2.:offset;
        EwaldTerm term(groups,{{0,0,intra},{0,1,inter},{1,1,intra}},cell,tol);
        PathSet path(1,2,2,.1,cell); auto work=term.make_potential_gradient_workspace(path);
        for (const auto& point : std::array{std::array{.01,.02},std::array{.8,1.1},std::array{1.999,.03},std::array{2.,1.}}) {
            path.pos(1,0,0)=point[0]; path.pos(1,1,0)=point[1];
            const double reference=parry(point[0],point[1],L,epsilon,separation,1.);
            CHECK(reference==Approx(parry(point[0],point[1],L,epsilon,separation,1.4)).margin(2e-10));
            CHECK(value(term,path)==Approx(reference).margin(tol==1e-6?2e-5:2e-7));
            std::vector<double> grad(4); term.evaluate_potential_gradient(path,grad,*work);
            for (int d=0; d<2; ++d) {
                const double h=1e-4;
                const double plus=parry(point[0]+(d==0?h:0),point[1]+(d==1?h:0),L,epsilon,separation,1.);
                const double minus=parry(point[0]-(d==0?h:0),point[1]-(d==1?h:0),L,epsilon,separation,1.);
                CHECK(grad[2+d]==Approx((plus-minus)/(2*h)).margin(tol==1e-6?2e-4:3e-6));
            }
        }
        CHECK(term.supports_periodic_virial(cell,2)); CHECK_FALSE(term.supports_periodic_virial(cell,3));
        CHECK_FALSE(term.supports_periodic_virial(SimulationCell(2,std::vector{5.,0.,0.,4.}),2));
        CHECK_FALSE(term.supports_open_centroid_virial());
        CHECK(term.clone()->potential_approximation_notes()==term.potential_approximation_notes());
    }
}

TEST_CASE("Admitted kernels obey directional virial identities and preserve sampling", "[kernel-admission][driver]") {
    const auto a=group(0,-1.),b=group(1,1.); const std::vector groups{a,b};
    const SimulationCell cell(2,std::vector{4.,0.,0.,4.});
    for (int kind=0; kind<3; ++kind) {
        Hamiltonian ham;
        if (kind==0) ham.add_component(std::make_unique<ScreenedCoulombTerm>(a,b,std::vector{1.,4.,2.},std::vector{-1.,1.},2,3,2.,.01));
        else {
            const auto intra=numerics::PotentialSpec::coulomb(1.5);
            const auto inter=kind==1?numerics::PotentialSpec::coulomb(1.5,.6):numerics::PotentialSpec::rytova_keldysh(1.5,.7,.6);
            ham.add_component(std::make_unique<EwaldTerm>(groups,std::vector<EwaldPair>{{0,0,intra},{0,1,inter},{1,1,intra}},cell,1e-7));
        }
        PathSet path(4,2,2,.125,cell); Action action(groups,ham,2); action.prepare_periodic_virial(path);
        for (std::size_t i=0; i<path.positions().size(); ++i) path.positions()[i]=.7*std::sin(.37*i);
        const auto saved=path.all_positions();
        PeriodicVirialField field; field.prepare(path); field.evaluate(path);
        const std::vector frozen(field.field().begin(),field.field().end());
        const double th=action.measure_thermo(path).energy,pv=action.measure_periodic_virial(path).energy;
        for (double h : {1e-4,3e-5,1e-5}) {
            for (std::size_t i=0; i<saved.size(); ++i) path.positions()[i]=saved[i]+h*frozen[i];
            const double plus=action.total_action(path);
            for (std::size_t i=0; i<saved.size(); ++i) path.positions()[i]=saved[i]-h*frozen[i];
            const double minus=action.total_action(path);
            CHECK(pv==Approx(th+((plus-minus)/(2*h)-field.divergence())/(2*path.beta())).margin(4e-7));
            std::copy(saved.begin(),saved.end(),path.positions().begin());
        }
        PimcConfig cfg; cfg.samples=2; cfg.beta=.5; cfg.time_step=.125;
        cfg.equilibration_blocks=8; cfg.accumulation_blocks=32; cfg.steps_per_block=8; cfg.seed=8751;
        Pimc thermo(cfg,groups,ham,cell,2,2); thermo.run();
        cfg.energy="periodic_virial"; cfg.compare_thermodynamic=true; cfg.threads=2;
        Pimc virial(cfg,groups,ham,cell,2,2); virial.run();
        CHECK(thermo.export_paths().positions==virial.export_paths().positions);
        CHECK_THAT(virial.summary(),Catch::Matchers::ContainsSubstring("excluded from statistical uncertainty"));
    }
}

TEST_CASE("Ewald periodic field has zero conditional flux under the scalar finite M weight", "[kernel-admission][physics]") {
    const auto a=group(0,-1.),b=group(1,1.); const std::vector groups{a,b};
    const double L=4,tau=.125; const SimulationCell cell(2,std::vector{L,0.,0.,L});
    for (int kind=0; kind<3; ++kind) {
        const auto intra=numerics::PotentialSpec::coulomb(1.5);
        auto cross=numerics::PotentialSpec::coulomb(1.5,.6);
        if (kind==1) cross=numerics::PotentialSpec::rytova_keldysh(1.5,.7,.6);
        if (kind==2) cross=build_screened_ewald_kernel({1.,4.,2.},{-1.,1.},2,3,2.,cell.ws_radius());
        // All pairs must share the same far dielectric; the FSC stack tail
        // is the mean of the exterior media, 1.5 here.
        EwaldTerm term(groups,{{0,0,intra},{0,1,cross},{1,1,intra}},cell,1e-8);
        PathSet path(2,2,2,tau,cell);
        for (std::size_t i=0; i<path.positions().size(); ++i) path.positions()[i]=.37*std::sin(.7*i);
        PeriodicVirialField field; field.prepare(path);
        auto workspace=term.make_potential_gradient_workspace(path);
        for (std::size_t axis=0; axis<8; ++axis) {
            const std::size_t p=axis/4,d=(axis/2)%2,k=axis%2;
            const double original=path.positions()[axis];
            double previous=0;
            for (int n : {256,512}) {
                double norm=0,flux=0;
                std::vector<double> potential(2),gradient(8); // [M], [np*dim*M]
                for (int i=0; i<n; ++i) {
                    path.positions()[axis]=-L/2+L*(i+.5)/n; path.rebuild_tables();
                    std::fill(potential.begin(),potential.end(),0.); term.evaluate_potential(path,potential);
                    double spring=0;
                    for (std::size_t pp=0; pp<2; ++pp) for (std::size_t dd=0; dd<2; ++dd) {
                        const double delta=path.pos(pp,dd,1)-path.pos(pp,dd,0);
                        const double folded=delta-L*std::floor(delta/L+.5);
                        spring+=folded*folded/tau;
                    }
                    const double weight=std::exp(-spring-tau*(potential[0]+potential[1]));
                    field.evaluate(path); const double F=field.field()[axis];
                    const double h=1e-5,current=path.positions()[axis];
                    path.positions()[axis]=current+h; field.evaluate(path); const double plus=field.field()[axis];
                    path.positions()[axis]=current-h; field.evaluate(path); const double minus=field.field()[axis];
                    path.positions()[axis]=current;
                    std::fill(gradient.begin(),gradient.end(),0.); term.evaluate_potential_gradient(path,gradient,*workspace);
                    const double delta=current-path.pos(p,d,1-k),folded=delta-L*std::floor(delta/L+.5);
                    const double derivative=2*folded/tau+tau*gradient[axis];
                    norm+=weight; flux+=weight*(F*derivative-(plus-minus)/(2*h));
                }
                const double residual=flux/norm;
                INFO("kernel=" << kind << " axis=" << axis << " n=" << n << " flux=" << residual);
                if (n==512) { CHECK(std::abs(residual)<2e-7); CHECK(std::abs(residual-previous)<2e-7); }
                previous=residual;
            }
            path.positions()[axis]=original;
        }
    }
}
