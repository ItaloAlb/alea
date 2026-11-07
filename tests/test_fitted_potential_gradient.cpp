// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Differentiate sampled scalar energies independently; test branch behavior
// separately from the integration-by-parts admission needed by virial.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <iomanip>
#include <iostream>
#include <vector>
#include "hamiltonian/hamiltonian.h"
#include "hamiltonian/ewald_term.h"
#include "hamiltonian/rytova_keldysh.h"
#include "hamiltonian/screened_coulomb.h"
#include "numerics/keldysh_kernel.h"
#include "pimc/path_set.h"

namespace {
using namespace alea;
using numerics::PotentialSpec;
using Catch::Approx;

ParticleGroup group(std::size_t first,std::size_t last,double charge) {
    ParticleGroup g{};
    g.first=first; g.last=last; g.count=last-first; g.charge=charge;
    g.mass={1.0,2.0}; g.inv_mass={1.0,0.5};
    return g;
}
SimulationCell square() { return SimulationCell(2,std::array{6.0,0.0,0.0,6.0}); }
const std::vector<ParticleGroup> groups{group(0,2,-1),group(2,4,1)};

void initialize(ParticleBatch& batch) {
    const std::array<double,8> positions{0.3,0.4,2.1,4.4,1.0,0.9,4.2,2.6}; // [p*2+d]
    const auto nw=batch.num_walkers();
    for (std::size_t p=0; p<4; ++p)
        for (std::size_t w=0; w<nw; ++w) {
            double r[2]{positions[2*p]+0.013*w*(p+1),positions[2*p+1]-0.017*w*(p+1)};
            batch.cell().wrap(r);
            for (std::size_t d=0; d<2; ++d) batch.positions()[(2*p+d)*nw+w]=r[d];
        }
    batch.dist_tables().build_all(batch.positions());
}

void finite_differences(const Hamiltonian& ham,WalkerSet& batch) {
    auto workspace=ham.make_potential_gradient_workspace(batch);
    std::vector<double> gradient(batch.positions().size()),values(batch.num_walkers());
    ham.evaluate_potential_gradient(batch,gradient,workspace);
    const auto nw=batch.num_walkers();
    for (double step : {1e-4,5e-5,2e-5})
        for (std::size_t p=0; p<batch.n_particles(); ++p)
            for (std::size_t d=0; d<batch.dim(); ++d)
                for (std::size_t w=0; w<nw; ++w) {
                    const auto index=(p*batch.dim()+d)*nw+w;
                    const double saved=batch.positions()[index];
                    batch.positions()[index]=saved+step;
                    batch.refresh_particle_tables(p,w,w+1);
                    ham.evaluate_potential(batch,values,w,w+1);
                    const double plus=values[w];
                    batch.positions()[index]=saved-step;
                    batch.refresh_particle_tables(p,w,w+1);
                    ham.evaluate_potential(batch,values,w,w+1);
                    const double minus=values[w];
                    batch.positions()[index]=saved;
                    batch.refresh_particle_tables(p,w,w+1);
                    CAPTURE(p,d,w,step,gradient[index]);
                    CHECK(std::abs(gradient[index]-(plus-minus)/(2*step))<2e-7);
                }
}

// Analytic derivative of Parry's independent erfc slab energy, with a full
// image/k sum (no optimized spline, Chebyshev fit or half-space weights).
// Same energy reference and conventions as test_ewald.cpp, Parry (1975).
std::vector<double> parry_gradient(double alpha,int nr,int nk) {
    const std::array<double,8> pos{0.3,0.4,2.1,4.4,1.0,0.9,4.2,2.6};
    const std::array<double,4> z{0,0,0.7,0.7},q{-1,-1,1,1};
    constexpr double L=6.0,eps=1.5;
    const double pi=std::acos(-1.0);
    std::vector<double> out(8,0.0); // [p*2+d]
    for (std::size_t i=0; i<4; ++i)
        for (std::size_t j=0; j<4; ++j) {
            const double dx=pos[2*i]-pos[2*j],dy=pos[2*i+1]-pos[2*j+1],dz=z[i]-z[j];
            const double qq=q[i]*q[j]/eps;
            for (int n1=-nr; n1<=nr; ++n1)
                for (int n2=-nr; n2<=nr; ++n2) {
                    if (i==j && n1==0 && n2==0) continue;
                    const double x=dx+L*n1,y=dy+L*n2,R=std::sqrt(x*x+y*y+dz*dz);
                    const double derivative=-qq*(std::erfc(alpha*R)/(R*R)+
                        2*alpha/std::sqrt(pi)*std::exp(-alpha*alpha*R*R)/R);
                    out[2*i]+=derivative*x/R; out[2*i+1]+=derivative*y/R;
                }
            for (int m1=-nk; m1<=nk; ++m1)
                for (int m2=-nk; m2<=nk; ++m2) {
                    if (m1==0 && m2==0) continue;
                    const double kx=2*pi*m1/L,ky=2*pi*m2/L,k=std::hypot(kx,ky);
                    const double f=std::exp(k*dz)*std::erfc(k/(2*alpha)+alpha*dz)+
                        std::exp(-k*dz)*std::erfc(k/(2*alpha)-alpha*dz);
                    const double strength=-pi/(L*L)*qq*std::sin(kx*dx+ky*dy)/k*f;
                    out[2*i]+=strength*kx; out[2*i+1]+=strength*ky;
                }
        }
    return out;
}
}

TEST_CASE("RK derivatives follow the sampled clamp fit and continuous tail transition", "[potential-gradient][fitted]") {
    const auto& kernel=numerics::keldysh_cheb();
    for (double x : {0.0004,0.002,0.03,0.5,3.0,15.0,31.0,32.0,35.0,39.0,40.0,41.0,1000.0})
        for (double scale : {1e-4,5e-5,2e-5}) {
            const double h=x*scale;
            const double fd=(kernel.eval(x+h)-kernel.eval(x-h))/(2*h);
            CAPTURE(x,h,fd);
            CHECK(kernel.d1(x)==Approx(fd).epsilon(2e-7).margin(1e-8));
        }
    for (double x : {kernel.kXjoin,kernel.x_hi}) {
        const double left=std::nextafter(x,0.0),right=std::nextafter(x,INFINITY);
        const double jump=kernel.eval(right)-kernel.eval(left);
        std::cout << std::setprecision(17) << "RK joined value residual at x=" << x << " = " << jump << '\n';
        CHECK(std::abs(jump)<1e-14);
        CHECK(std::abs(kernel.d1(right)-kernel.d1(left))<1e-14);
        CHECK(std::abs(kernel.d2(right)-kernel.d2(left))<1e-14);
    }
    CHECK(kernel.d1(0.0)==0.0);
}

TEST_CASE("RK spatial gradients differentiate scalar energies with softening and minimum images", "[potential-gradient][fitted]") {
    for (const auto& cell : {SimulationCell(2),square(),SimulationCell(2,std::array{6.0,0.0,2.0,5.0})}) {
        WalkerSet batch(3,4,2,cell); initialize(batch);
        Hamiltonian ham;
        ham.add_component(std::make_unique<KeldyshTerm>(groups[0],groups[0],1.5,0.04));
        ham.add_component(std::make_unique<KeldyshTerm>(groups[0],groups[1],1.5,0.7,0.6));
        finite_differences(ham,batch);
    }
    WalkerSet coincidence(2,2,2);
    KeldyshTerm term(group(0,1,-1),group(1,2,1),1.5,0.7);
    auto workspace=term.make_potential_gradient_workspace(coincidence);
    std::vector<double> output(8,0.0);
    term.evaluate_potential_gradient(coincidence,output,*workspace);
    CHECK(output==std::vector<double>(8,0.0));
}

TEST_CASE("Ewald gradients differentiate Coulomb RK and screened class-coupled scalar energies", "[potential-gradient][ewald]") {
    const auto cell=square();
    const auto coulomb=PotentialSpec::coulomb(1.5),across=PotentialSpec::coulomb(1.5,0.7);
    const auto rk=PotentialSpec::rytova_keldysh(1.5,0.04);
    const auto rk_across=PotentialSpec::rytova_keldysh(1.5,0.7,0.6);
    const auto screened=build_screened_ewald_kernel({1.5,1.5,1.5},{-1.0,1.0},2,0,0.0,cell.ws_radius());
    for (const auto& pairs : {std::vector<EwaldPair>{{0,0,coulomb},{1,1,coulomb},{0,1,coulomb}},
                              std::vector<EwaldPair>{{0,0,rk},{1,1,coulomb},{0,1,across}},
                              std::vector<EwaldPair>{{0,0,coulomb},{1,1,coulomb},{0,1,rk_across}},
                              std::vector<EwaldPair>{{0,0,screened},{1,1,screened},{0,1,across}}}) {
        WalkerSet batch(3,4,2,cell); initialize(batch);
        Hamiltonian ham;
        ham.add_component(std::make_unique<EwaldTerm>(groups,pairs,cell,1e-8));
        finite_differences(ham,batch);
    }
    const SimulationCell skew(2,std::array{6.0,0.0,2.0,5.0});
    WalkerSet skew_batch(2,4,2,skew); initialize(skew_batch);
    Hamiltonian skew_ham;
    skew_ham.add_component(std::make_unique<EwaldTerm>(groups,1.5,skew,1e-8));
    finite_differences(skew_ham,skew_batch);
}

TEST_CASE("Ewald Coulomb gradients agree with refined independent Parry forces", "[potential-gradient][ewald][physics]") {
    const auto reference=parry_gradient(1.0,4,16),refined=parry_gradient(1.4,6,24);
    for (std::size_t i=0; i<8; ++i) CHECK(std::abs(reference[i]-refined[i])<1e-10);
    const auto cell=square();
    WalkerSet batch(1,4,2,cell); initialize(batch);
    for (double tolerance : {1e-8,1e-10}) {
        const auto intra=PotentialSpec::coulomb(1.5),inter=PotentialSpec::coulomb(1.5,0.7);
        EwaldTerm term(groups,{{0,0,intra},{1,1,intra},{0,1,inter}},cell,tolerance);
        auto workspace=term.make_potential_gradient_workspace(batch);
        std::vector<double> output(8,0.0);
        term.evaluate_potential_gradient(batch,output,*workspace);
        double worst=0.0;
        for (std::size_t i=0; i<8; ++i) {
            worst=std::max(worst,std::abs(output[i]-refined[i]));
            CAPTURE(tolerance,i,output[i],refined[i]);
            CHECK(std::abs(output[i]-refined[i])<2e-6);
        }
        std::cout << std::setprecision(17) << "Parry force error at tol=" << tolerance << " = " << worst << '\n';
    }
}

TEST_CASE("Contrast screened tables and Ewald cutoffs expose separate continuity diagnostics", "[potential-gradient][ewald][continuity]") {
    const auto cell=square();
    const auto screened=build_screened_ewald_kernel({1.0,8.0,1.0},{-1.0,1.0},2,0,0.0,cell.ws_radius());
    const auto coulomb=PotentialSpec::coulomb(1.0,0.7);
    WalkerSet batch(2,4,2,cell); initialize(batch);
    Hamiltonian ham;
    ham.add_component(std::make_unique<EwaldTerm>(groups,
        std::vector<EwaldPair>{{0,0,screened},{1,1,screened},{0,1,coulomb}},cell,1e-8));
    finite_differences(ham,batch);
    double worst_join=0.0;
    const auto& table=*screened.table;
    for (int panel=1; panel<table.n_panels(); ++panel) {
        const double r=table.r_max()*panel/table.n_panels();
        worst_join=std::max(worst_join,std::abs(table.eval(std::nextafter(r,0.0))-
                                               table.eval(std::nextafter(r,INFINITY))));
    }
    REQUIRE(std::isfinite(worst_join));
    std::cout << std::setprecision(17) << "Screened panel max jump = " << worst_join << '\n';
    // RK's sampled-kernel correction matches the unchanged ideal spline at
    // the strict cutoff. Other representations retain roundoff diagnostics.
    const auto rk=PotentialSpec::rytova_keldysh(1.5,0.7);
    for (const auto& spec : {PotentialSpec::coulomb(1.5),rk,screened}) {
        EwaldTerm term(groups,spec,spec.epsilon,cell,1e-8);
        const auto& setup=term.setup();
        const double r=std::nextafter(setup.r_cut,0.0);
        const double value=spec.kind==PotentialSpec::Kind::Keldysh ?
            numerics::PI/(2*spec.epsilon*spec.r0)*numerics::keldysh_cheb().eval(r/spec.r0) :
            spec.kind==PotentialSpec::Kind::Screened ? spec.singular_coef/r+spec.table->eval(r) :
            1.0/(spec.epsilon*r);
        const double jump=-(value-setup.long_range.eval(r)-term.cutoff_correction().eval(r));
        REQUIRE(std::isfinite(jump));
        CHECK(std::abs(jump)<2e-13);
        std::cout << std::setprecision(17) << "Ewald cutoff jump kind=" << static_cast<int>(spec.kind) << " = " << jump << '\n';
    }
}

TEST_CASE("Ewald workspace binds geometry before writes and rejects untyped scratch", "[potential-gradient][ewald]") {
    const auto cell=square();
    EwaldTerm term(groups,1.5,cell,1e-8);
    WalkerSet batch(3,4,2,cell),other(3,4,2,SimulationCell(2,std::array{3.0,0.0,0.0,12.0}));
    initialize(batch); initialize(other);
    auto workspace=term.make_potential_gradient_workspace(batch);
    std::vector<double> output(24,9.0);
    CHECK_THROWS_AS(term.make_potential_gradient_workspace(other),std::invalid_argument);
    CHECK_THROWS_AS(term.evaluate_potential_gradient(other,output,*workspace),std::invalid_argument);
    auto untyped=term.HamiltonianComponent::make_potential_gradient_workspace(batch);
    CHECK_THROWS_AS(term.evaluate_potential_gradient(batch,output,*untyped),std::invalid_argument);
    CHECK(output==std::vector<double>(24,9.0));
    Hamiltonian ham; ham.add_component(term.clone());
    auto composite_workspace=ham.make_potential_gradient_workspace(batch);
    CHECK_THROWS_AS(ham.evaluate_potential_gradient(other,output,composite_workspace),std::invalid_argument);
    CHECK(output==std::vector<double>(24,9.0));
}

TEST_CASE("Ewald gradient tiles crowds clones paths and image shifts preserve state", "[potential-gradient][ewald]") {
    const SimulationCell cell(2,std::array{6.0,0.0,2.0,5.0});
    WalkerSet batch(67,4,2,cell); initialize(batch);
    Hamiltonian ham;
    ham.add_component(std::make_unique<EwaldTerm>(groups,1.5,cell,1e-8));
    auto workspace=ham.make_potential_gradient_workspace(batch);
    const std::vector<double> positions(batch.positions().begin(),batch.positions().end());
    const std::vector<double> tables(batch.dist_tables().data().begin(),batch.dist_tables().data().end());
    std::vector<double> full(536),split(536,9.0),other(536); // [4*2*67]
    ham.evaluate_potential_gradient(batch,full,workspace);
    auto first=ham.make_potential_gradient_workspace(batch),second=ham.make_potential_gradient_workspace(batch);
    #pragma omp parallel for num_threads(2)
    for (int c=0; c<2; ++c)
        ham.evaluate_potential_gradient(batch,split,c==0 ? first : second,c==0 ? 0 : 31,c==0 ? 31 : 67);
    CHECK(split==full);
    std::vector<HamiltonianGradientWorkspace> crowds;
    for (int c=0; c<4; ++c) crowds.push_back(ham.make_potential_gradient_workspace(batch));
    std::fill(split.begin(),split.end(),9.0);
    #pragma omp parallel for num_threads(4)
    for (int c=0; c<4; ++c)
        ham.evaluate_potential_gradient(batch,split,crowds[c],67*c/4,67*(c+1)/4);
    CHECK(split==full);
    CHECK(std::vector<double>(batch.positions().begin(),batch.positions().end())==positions);
    CHECK(std::vector<double>(batch.dist_tables().data().begin(),batch.dist_tables().data().end())==tables);
    for (std::size_t d=0; d<2; ++d)
        for (std::size_t w=0; w<67; ++w) {
            double total=0.0;
            for (std::size_t p=0; p<4; ++p) total+=full[(2*p+d)*67+w];
            CHECK(std::abs(total)<1e-12);
        }
    const auto clone=ham.clone(); auto clone_workspace=clone.make_potential_gradient_workspace(batch);
    clone.evaluate_potential_gradient(batch,other,clone_workspace); CHECK(other==full);
    PathSet path(67,4,2,0.1,cell);
    std::copy(positions.begin(),positions.end(),path.positions().begin()); path.rebuild_tables();
    auto path_workspace=ham.make_potential_gradient_workspace(path);
    ham.evaluate_potential_gradient(path,other,path_workspace); CHECK(other==full);
    for (std::size_t w=0; w<67; ++w) batch.positions()[w]+=6.0;
    batch.dist_tables().build_all(batch.positions());
    ham.evaluate_potential_gradient(batch,other,workspace);
    for (std::size_t i=0; i<536; ++i) CHECK(other[i]==Approx(full[i]).margin(1e-12));
}

TEST_CASE("Ewald bounded interlayer coincidence has finite scalar energy and zero gradient", "[potential-gradient][ewald][coincidence]") {
    const auto cell=square();
    const std::vector<ParticleGroup> pair_groups{group(0,1,-1),group(1,2,1)};
    const auto intra=PotentialSpec::coulomb(1.5),inter=PotentialSpec::coulomb(1.5,2.0);
    const auto screened=build_screened_ewald_kernel({1.5,1.5,1.5,1.5},{-1.0,1.0,3.0},2,3,2.0,cell.ws_radius());
    WalkerSet batch(2,2,2,cell);
    std::vector<double> reference(2),values(2);
    for (const auto& cross : {inter,screened}) {
        EwaldTerm term(pair_groups,{{0,0,intra},{1,1,intra},{0,1,cross}},cell,1e-8);
        std::fill(values.begin(),values.end(),0.0);
        term.evaluate_potential(batch,values);
        for (double value : values) CHECK(std::isfinite(value));
        auto workspace=term.make_potential_gradient_workspace(batch);
        std::vector<double> output(8,0.0);
        term.evaluate_potential_gradient(batch,output,*workspace);
        for (double gradient : output) CHECK(std::abs(gradient)<1e-13);
        if (cross.kind==PotentialSpec::Kind::Coulomb) reference=values;
        else for (std::size_t w=0; w<2; ++w) CHECK(values[w]==Approx(reference[w]).margin(1e-7));
    }
    EwaldTerm singular(pair_groups,1.5,cell,1e-8);
    auto workspace=singular.make_potential_gradient_workspace(batch);
    std::vector<double> output(8,0.0);
    CHECK_THROWS_AS(singular.evaluate_potential_gradient(batch,output,*workspace),std::domain_error);
}
