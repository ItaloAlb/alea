// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// G=grad V of the same clamped FSC fit sampled by the scalar potential.
// Unit/layer limits and scalar finite differences are independent references.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include "fast_screened_coulomb.h"
#include "hamiltonian/hamiltonian.h"
#include "hamiltonian/screened_coulomb.h"
#include "pimc/path_set.h"
#include "population/walker_set.h"

namespace {
using namespace alea;
using Catch::Approx;

ParticleGroup group(std::size_t first, std::size_t last, double charge) {
    ParticleGroup result{};
    result.first=first; result.last=last; result.count=last-first; result.charge=charge;
    return result;
}

ScreenedCoulombTerm term(ParticleGroup a, ParticleGroup b, int kind) {
    if (kind == 0) return ScreenedCoulombTerm(a,b,{1.5},{},1,0,0.0,0.01);
    return ScreenedCoulombTerm(a,b,{1.0,4.0,2.0,1.5},{-1.0,1.0,3.0},
                               2,kind == 1 ? 0 : 3,2.0,0.1);
}

void initialize(ParticleBatch& batch) {
    const auto nw=batch.num_walkers(), dim=batch.dim();
    for (std::size_t p=0; p<batch.n_particles(); ++p)
        for (std::size_t d=0; d<dim; ++d)
            for (std::size_t w=0; w<nw; ++w)
                batch.positions()[(p*dim+d)*nw+w]=0.29+0.61*p+0.07*d*p+0.03*w*(p+1);
    batch.dist_tables().build_all(batch.positions());
}

std::vector<double> snapshot(std::span<const double> values) {
    return {values.begin(),values.end()};
}

void check_differences(const Hamiltonian& ham, WalkerSet& batch) {
    auto workspace=ham.make_potential_gradient_workspace(batch);
    std::vector<double> gradient(batch.positions().size()),values(batch.num_walkers()); // [np*dim*nw], [nw]
    ham.evaluate_potential_gradient(batch,gradient,workspace);
    for (double step : {1e-4,5e-5,2e-5})
        for (std::size_t index=0; index<gradient.size(); ++index) {
            const auto nw=batch.num_walkers();
            const auto p=index/(nw*batch.dim()), w=index%nw;
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
            CAPTURE(index,step);
            CHECK(gradient[index] == Approx((plus-minus)/(2*step)).margin(3e-7).epsilon(2e-6));
        }
}
}

TEST_CASE("Screened gradients differentiate same and cross group sampled potentials in supported dimensions", "[potential-gradient][screened]") {
    for (int kind : {0,1,2})
        for (std::size_t dim : {1,2,3}) {
            DYNAMIC_SECTION("stack " << kind << " dimension " << dim) {
                WalkerSet batch(3,4,dim);
                initialize(batch);
                const auto a=group(0,2,-1.0), b=group(2,4,2.0);
                Hamiltonian ham;
                ham.add_component(std::make_unique<ScreenedCoulombTerm>(term(a,a,kind)));
                ham.add_component(std::make_unique<ScreenedCoulombTerm>(term(a,b,kind)));
                ham.add_component(std::make_unique<ScreenedCoulombTerm>(term(b,b,kind)));
                CHECK(ham.supports_potential_gradient());
                check_differences(ham,batch);
            }
        }
}

TEST_CASE("Screened gradients convert Rydberg once and do not double soften layer separation", "[potential-gradient][screened]") {
    WalkerSet batch(1,2,2);
    batch.positions()[2]=0.8; batch.positions()[3]=0.6;
    batch.dist_tables().build_all(batch.positions());
    const auto a=group(0,1,-2.0), b=group(1,2,3.0);
    for (bool interlayer : {false,true}) {
        const auto pair=interlayer ? ScreenedCoulombTerm(a,b,{1.5,1.5,1.5,1.5},
            {-1.0,1.0,3.0},2,3,2.0,0.01) : term(a,b,0);
        auto workspace=pair.make_potential_gradient_workspace(batch);
        std::vector<double> output(4,0.0); // [np*dim*nw]
        pair.evaluate_potential_gradient(batch,output,*workspace);
        // Exact uniform V_H=q_a*q_b/(epsilon*sqrt(rho^2+z^2)).
        // The controlled uniform-medium representation retains the analytic
        // kernel exactly; this physical check is independent of fit derivatives.
        const double radius=std::sqrt(interlayer ? 5.0 : 1.0);
        const double strength=6.0/(1.5*radius*radius*radius);
        const double tolerance=1e-12;
        CHECK(output[0] == Approx(-strength*0.8).epsilon(tolerance));
        CHECK(output[1] == Approx(-strength*0.6).epsilon(tolerance));
        CHECK(output[2] == -output[0]); CHECK(output[3] == -output[1]);
    }
}

TEST_CASE("Screened gradient validity clamps freeze the whole sampled potential including coincidence", "[potential-gradient][screened]") {
    const auto reference=build_screened_fit({1.5},{},1,0,0.0,0.01);
    const std::vector<double> radii{0.0,reference.rho_min/2,reference.rho_min,
        1.0,reference.rho_max,reference.rho_max*2}; // [nw]
    WalkerSet batch(radii.size(),2,1);
    std::copy(radii.begin(),radii.end(),batch.positions().begin()+radii.size());
    batch.dist_tables().build_all(batch.positions());
    const auto pair=term(group(0,1,-1),group(1,2,2),0);
    auto workspace=pair.make_potential_gradient_workspace(batch);
    std::vector<double> output(2*radii.size(),0.0),values(radii.size(),0.0); // [np*dim*nw], [nw]
    pair.evaluate_potential_gradient(batch,output,*workspace);
    pair.evaluate_potential(batch,values);
    for (std::size_t w=0; w<radii.size(); ++w) {
        const double radius=radii[w];
        if (radius < reference.rho_min || radius > reference.rho_max) {
            CHECK(output[w] == 0.0); CHECK(output[radii.size()+w] == 0.0);
        } else {
            CHECK(output[w] == Approx(-2.0/(1.5*radius*radius)).epsilon(1e-8));
            CHECK(output[radii.size()+w] == -output[w]);
        }
        CHECK(values[w] == Approx(-2.0/(1.5*std::clamp(radius,reference.rho_min,reference.rho_max))).epsilon(1e-9));
    }
    // A flat sampled low clamp must not erase the physical attractive cusp
    // used by primitive-action admission; this implementation changes no cusp.
    CHECK(pair.coulomb_cusp_residue(group(0,1,-1),group(1,2,2)) < 0.0);
}

TEST_CASE("Screened gradients preserve state splits clones paths and concurrent periodic crowds", "[potential-gradient][screened]") {
    for (const auto& cell : {SimulationCell(2),SimulationCell(2,std::vector<double>{4,0,0,5}),
                             SimulationCell(2,std::vector<double>{4,0,2,3.4641016151377544})}) {
        WalkerSet batch(11,4,2,cell);
        initialize(batch);
        Hamiltonian ham;
        ham.add_component(std::make_unique<ScreenedCoulombTerm>(term(group(0,2,-1),group(2,4,2),2)));
        check_differences(ham,batch);
        std::fill(batch.gradients().begin(),batch.gradients().end(),4.0);
        std::fill(batch.laplacians().begin(),batch.laplacians().end(),-3.0);
        std::fill(batch.local_energies().begin(),batch.local_energies().end(),12.0);
        std::vector<double> proposed(22,0.4); // [dim*nw]
        batch.propose_move_batch(0,proposed);
        const auto positions=snapshot(batch.positions()), distances=snapshot(batch.dist_tables().data());
        const auto pending=snapshot(batch.pending_dist_rows()), proposal=snapshot(batch.new_pos());
        auto workspace=ham.make_potential_gradient_workspace(batch);
        std::vector<double> full(88),other(88,7.0); // [np*dim*nw]
        ham.evaluate_potential_gradient(batch,full,workspace);
        ham.evaluate_potential_gradient(batch,other,workspace,2,9);
        for (std::size_t i=0; i<88; ++i) CHECK(other[i] == (i%11 < 2 || i%11 >= 9 ? 7.0 : full[i]));
        ham.evaluate_potential_gradient(batch,other,workspace,0,2);
        ham.evaluate_potential_gradient(batch,other,workspace,9,11);
        CHECK(other == full);
        for (std::size_t d=0; d<2; ++d)
            for (std::size_t w=0; w<11; ++w) {
                double sum=0.0;
                for (std::size_t p=0; p<4; ++p) sum += full[(p*2+d)*11+w];
                CHECK(std::abs(sum) < 1e-13);
            }
        auto first=ham.make_potential_gradient_workspace(batch), second=ham.make_potential_gradient_workspace(batch);
        #pragma omp parallel for num_threads(2)
        for (int crowd=0; crowd<2; ++crowd)
            ham.evaluate_potential_gradient(batch,other,crowd == 0 ? first : second,
                                            crowd == 0 ? 0 : 5,crowd == 0 ? 5 : 11);
        CHECK(other == full);
        const auto clone=ham.clone(); auto clone_workspace=clone.make_potential_gradient_workspace(batch);
        clone.evaluate_potential_gradient(batch,other,clone_workspace); CHECK(other == full);
        PathSet path(11,4,2,0.1,cell);
        std::copy(positions.begin(),positions.end(),path.positions().begin()); path.rebuild_tables();
        auto path_workspace=ham.make_potential_gradient_workspace(path);
        ham.evaluate_potential_gradient(path,other,path_workspace); CHECK(other == full);
        CHECK(snapshot(batch.positions()) == positions); CHECK(snapshot(batch.dist_tables().data()) == distances);
        CHECK(snapshot(batch.pending_dist_rows()) == pending); CHECK(snapshot(batch.new_pos()) == proposal);
        CHECK(snapshot(batch.gradients()) == std::vector<double>(88,4.0));
        CHECK(snapshot(batch.laplacians()) == std::vector<double>(88,-3.0));
        CHECK(snapshot(batch.local_energies()) == std::vector<double>(11,12.0));
        if (cell.is_periodic()) {
            for (std::size_t d=0; d<2; ++d)
                for (std::size_t w=0; w<11; ++w) batch.positions()[d*11+w] += cell.vectors()[d];
            batch.dist_tables().build_all(batch.positions());
            ham.evaluate_potential_gradient(batch,other,workspace);
            for (std::size_t i=0; i<88; ++i) CHECK(other[i] == Approx(full[i]).margin(1e-12));
        }
    }
}

TEST_CASE("Screened gradient preparation and evaluation reject invalid bindings and nonfinite data", "[potential-gradient][screened]") {
    WalkerSet batch(2,2,2),wrong_shape(3,2,2);
    initialize(batch); initialize(wrong_shape);
    const auto pair=term(group(0,1,-1),group(1,2,1),0);
    auto workspace=pair.make_potential_gradient_workspace(batch);
    std::vector<double> out(8,7.0),small(7,7.0); // [np*dim*nw], deliberately wrong size
    CHECK_THROWS_AS(pair.evaluate_potential_gradient(batch,small,*workspace),std::invalid_argument);
    CHECK_THROWS_AS(pair.evaluate_potential_gradient(wrong_shape,out,*workspace),std::invalid_argument);
    CHECK_THROWS_AS(pair.evaluate_potential_gradient(batch,out,*workspace,2,1),std::invalid_argument);
    CHECK_THROWS_AS(pair.evaluate_potential_gradient(batch,batch.positions(),*workspace),std::invalid_argument);
    const auto clone=pair.clone();
    CHECK_THROWS_AS(clone->evaluate_potential_gradient(batch,out,*workspace),std::invalid_argument);
    CHECK(out == std::vector<double>(8,7.0)); CHECK(small == std::vector<double>(7,7.0));
    const auto bad_group=term(group(0,1,1),group(1,3,1),0);
    CHECK_THROWS_AS(bad_group.make_potential_gradient_workspace(batch),std::invalid_argument);
    const auto bad_charge=term(group(0,1,std::numeric_limits<double>::infinity()),group(1,2,1),0);
    CHECK_THROWS_AS(bad_charge.make_potential_gradient_workspace(batch),std::invalid_argument);
    batch.positions()[0]=std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(pair.make_potential_gradient_workspace(batch),std::invalid_argument);
    CHECK_THROWS_AS(pair.evaluate_potential_gradient(batch,out,*workspace),std::domain_error);
}
