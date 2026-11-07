// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Independent analytic functions and image-charge sums check the corrected
// table, its sampled derivatives and screened Ewald cutoff. No virial admission.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include "hamiltonian/ewald_term.h"
#include "hamiltonian/screened_coulomb.h"
#include "numerics/radial_table.h"
#include "population/walker_set.h"

namespace {
using namespace alea;
using namespace alea::numerics;
using Catch::Approx;

// Expand the symmetric-slab Fourier solution as a geometric image series:
// g(r)=2/eps sum_(n>=1) eta^n/sqrt(r²+(n*thickness)²), eta=(eps-out)/(eps+out).
// Source at the slab centre, with the 1/(eps*r) pole kept separate. Derived
// from S. Ismail-Beigi, "A dielectric slab: potential from a point charge in
// the middle" (2013), pp. 1-2, https://volga.eng.yale.edu/sites/default/files/files/dielectric-slab.pdf.
std::array<double,3> images(double r,double eps,double outer,double thickness,int terms) {
    std::array<double,3> result{}; // [smooth value,d1,d2], Hartree units
    const double eta=(eps-outer)/(eps+outer);
    double weight=2.0/eps;
    for (int n=1; n<=terms; ++n) {
        weight*=eta;
        const double z=n*thickness, rho=std::hypot(r,z), inv=1.0/rho;
        result[0]+=weight*inv;
        result[1]-=weight*(r/rho)*inv*inv;
        result[2]+=weight*(2*(r/rho)*(r/rho)-(z/rho)*(z/rho))*inv*inv*inv;
    }
    return result;
}

void check_joins(const RadialTable& table,double value_budget,double slope_budget,double curvature_budget) {
    std::array<double,3> worst{}; // [value,d1,d2] one-sided gaps
    for (int p=1; p<table.n_panels(); ++p) {
        const double r=table.r_max()*(double(p)/table.n_panels());
        const double left=std::nextafter(r,0.0),right=std::nextafter(r,INFINITY);
        worst[0]=std::max(worst[0],std::abs(table.eval(left)-table.eval(right)));
        worst[1]=std::max(worst[1],std::abs(table.d1(left)-table.d1(right)));
        worst[2]=std::max(worst[2],std::abs(table.d2(left)-table.d2(right)));
    }
    std::cout << std::setprecision(17) << "Table panels=" << table.n_panels()
              << " join gaps=" << worst[0] << ',' << worst[1] << ',' << worst[2] << '\n';
    CHECK(worst[0]<value_budget); CHECK(worst[1]<slope_budget); CHECK(worst[2]<curvature_budget);
}

ParticleGroup group(std::size_t first,std::size_t last,double charge) {
    ParticleGroup result{};
    result.first=first; result.last=last; result.count=last-first; result.charge=charge;
    return result;
}
}

TEST_CASE("Corrected radial tables retain value accuracy and join two derivatives", "[screened-continuity][table]") {
    auto value=[](double r) { return std::exp(-r)*std::cos(23*r); };
    auto first=[](double r) { return std::exp(-r)*(-std::cos(23*r)-23*std::sin(23*r)); };
    auto second=[](double r) { return std::exp(-r)*(-528*std::cos(23*r)+46*std::sin(23*r)); };
    const auto table=RadialTable::build(value,5.0,1e-8);
    check_joins(table,2e-12,2e-10,2e-8);
    double error=0.0;
    for (int j=0; j<=1000; ++j) {
        const double r=5.0*j/1000;
        error=std::max(error,std::abs(table.eval(r)-value(r)));
        CHECK(table.d1(r) == Approx(first(r)).margin(3e-6));
        CHECK(table.d2(r) == Approx(second(r)).margin(2e-3));
    }
    CHECK(error<1e-8);
    for (int p=1; p<table.n_panels(); ++p) {
        const double r=table.r_max()*double(p)/table.n_panels();
        for (double scale : {1e-5,5e-6,2e-6}) {
            const double h=scale*table.r_max()/table.n_panels();
            CHECK(table.d1(r) == Approx((table.eval(r+h)-table.eval(r-h))/(2*h)).margin(3e-8));
            CHECK(table.d2(r) == Approx((table.d1(r+h)-table.d1(r-h))/(2*h)).margin(3e-6));
        }
    }
}

TEST_CASE("Radial tables validate setup and differentiate their constant clamps", "[screened-continuity][table]") {
    auto value=[](double r) { return 1+r+r*r; };
    const auto table=RadialTable::build(value,2,1e-12);
    for (double r : std::array<double,4>{-INFINITY,-1.0,3.0,INFINITY}) {
        CHECK(table.eval(r) == Approx(r < 0 ? 1.0 : 7.0));
        CHECK(table.d1(r) == 0.0); CHECK(table.d2(r) == 0.0);
    }
    CHECK(table.d1(0) == Approx(1.0)); CHECK(table.d1(2) == Approx(5.0));
    CHECK(table.d2(0) == Approx(2.0)); CHECK(table.d2(2) == Approx(2.0));
    CHECK(RadialTable{}.eval(1) == 0.0); CHECK(RadialTable{}.d1(0) == 0.0);
    for (double invalid : std::array<double,4>{0.0,-1.0,INFINITY,std::numeric_limits<double>::quiet_NaN()}) {
        CHECK_THROWS_AS(RadialTable::build(value,invalid,1e-12),std::invalid_argument);
        CHECK_THROWS_AS(RadialTable::build(value,2,invalid),std::invalid_argument);
    }
    CHECK_THROWS_AS(RadialTable::build({},2,1e-12),std::invalid_argument);
    CHECK_THROWS_AS(RadialTable::build(value,1e-310,1e-12),std::invalid_argument);
    CHECK_THROWS_AS(RadialTable::build([](double) { return INFINITY; },2,1e-12),std::runtime_error);
    CHECK(std::isnan(table.eval(NAN))); CHECK(std::isnan(table.d1(NAN))); CHECK(std::isnan(table.d2(NAN)));
    for (double radius : std::array<double,4>{INFINITY,NAN,0,std::numeric_limits<double>::max()})
        CHECK_THROWS_AS(build_screened_ewald_kernel({1.0},{},1,0,0,radius),std::invalid_argument);
    CHECK_THROWS_AS(build_screened_ewald_kernel({1.0},{},1,0,0,1e50),std::runtime_error);
}

TEST_CASE("Screened table forces agree with refined symmetric slab image charges", "[screened-continuity][screened][physics]") {
    for (const auto& parameters : {std::array{8.0,1.0,2.0},std::array{1.0,8.0,2.0},std::array{14.0,1.0,6.0}}) {
        const double eps=parameters[0],outer=parameters[1],thickness=parameters[2];
        const auto spec=build_screened_ewald_kernel({outer,eps,outer},{-thickness/2,thickness/2},2,0,0,8);
        check_joins(*spec.table,3e-13,2e-11,2e-9);
        std::array<double,3> worst{}; // maximum absolute smooth-part errors
        for (int j=0; j<=160; ++j) {
            const double r=spec.table->r_max()*j/160;
            const auto coarse=images(r,eps,outer,thickness,256),fine=images(r,eps,outer,thickness,512);
            const std::array actual{spec.table->eval(r),spec.table->d1(r),spec.table->d2(r)};
            for (int order=0; order<3; ++order) {
                CAPTURE(eps,outer,thickness,r,order);
                REQUIRE(std::abs(coarse[order]-fine[order])<2e-14);
                worst[order]=std::max(worst[order],std::abs(actual[order]-fine[order]));
                CHECK(actual[order] == Approx(fine[order]).margin(order == 0 ? 2e-12 : (order == 1 ? 2e-10 : 2e-8)));
            }
        }
        std::cout << std::setprecision(17) << "Image reference eps=" << eps << " outer=" << outer
                  << " thickness=" << thickness << " errors=" << worst[0] << ',' << worst[1] << ',' << worst[2] << '\n';
    }
    const auto uniform=build_screened_ewald_kernel({1.5,1.5,1.5,1.5},{-1,1,3},2,3,2,8);
    check_joins(*uniform.table,3e-13,2e-11,2e-9);
    for (double r : {0.0,0.01,0.5,1.0,2.0,4.0,8.0}) {
        const double rho=std::hypot(r,2.0),inv=1.0/rho;
        CHECK(uniform.table->eval(r) == Approx(inv/1.5).margin(2e-12));
        CHECK(uniform.table->d1(r) == Approx(-r*inv*inv*inv/1.5).margin(2e-10));
        CHECK(uniform.table->d2(r) == Approx((2*r*r-4)*std::pow(inv,5)/1.5).margin(2e-8));
    }
}

TEST_CASE("Screened Ewald cutoff joins the sampled table with value slope and curvature", "[screened-continuity][ewald]") {
    const SimulationCell cell(2,std::array{6.0,0.0,0.0,6.0});
    const auto spec=build_screened_ewald_kernel({1.0,8.0,1.0},{-1.0,1.0},2,0,0,cell.ws_radius());
    const auto a=group(0,1,-1),b=group(1,2,1);
    const EwaldTerm term({a,b},spec,1.0,cell,1e-8);
    const auto& setup=term.setup();
    const double r=std::nextafter(setup.r_cut,0.0);
    double value,slope,curvature;
    setup.long_range.eval(r,value,slope,curvature);
    CHECK(std::abs(spec.v_real(r)-value)<2e-13);
    CHECK(std::abs(spec.dv_real(r)-slope)<2e-11);
    CHECK(std::abs(spec.d2v_real(r)-curvature)<2e-9);
    // Cross a table knot and the Ewald cutoff in full scalar energies. Here
    // the cutoff is also a minimum-image boundary; both scalar routes agree.
    WalkerSet batch(1,2,2,cell);
    auto workspace=term.make_potential_gradient_workspace(batch);
    auto scalar=[&](double separation) {
        batch.positions()[2]=separation; batch.dist_tables().build_all(batch.positions());
        std::vector<double> result(1,0); term.evaluate_potential(batch,result);
        return result[0];
    };
    for (double separation : {spec.table->r_max()/spec.table->n_panels(),setup.r_cut}) {
        scalar(separation);
        std::vector<double> gradient(4,0); term.evaluate_potential_gradient(batch,gradient,*workspace);
        for (double h : {2e-5,1e-5,5e-6}) {
            const double plus=scalar(separation+h),minus=scalar(separation-h);
            CHECK(gradient[2] == Approx((plus-minus)/(2*h)).margin(2e-8));
        }
    }
}

TEST_CASE("Screened setup callables preserve bounded coincidence and direct Coulomb poles", "[screened-origin]") {
    const auto bounded=build_screened_ewald_kernel({1.5,1.5,1.5,1.5},{-1,1,3},2,3,2,8);
    // Uniform interlayer reference: V=1/(epsilon*sqrt(r^2+z^2)).
    for (double r : {0.0,1e-200,1e-100,0.01}) {
        const double distance=std::hypot(r,2.0), inverse=1.0/distance;
        CHECK(bounded.v_real(r) == Approx(inverse/1.5).margin(2e-12));
        CHECK(bounded.dv_real(r) == Approx(-r*std::pow(inverse,3)/1.5).margin(2e-10));
        CHECK(bounded.d2v_real(r) == Approx((2*r*r-4)*std::pow(inverse,5)/1.5).margin(2e-8));
    }
    const auto direct=build_screened_ewald_kernel({1.5},{},1,0,0,8);
    CHECK(direct.v_real(0.0) == INFINITY);
    CHECK(direct.dv_real(0.0) == -INFINITY);
    CHECK(direct.d2v_real(0.0) == INFINITY);
    for (double r : {0.01,1.0,9.0}) {
        CHECK(direct.v_real(r) == Approx(1/(1.5*r)));
        CHECK(direct.dv_real(r) == Approx(-1/(1.5*r*r)));
        CHECK(direct.d2v_real(r) == Approx(2/(1.5*r*r*r)));
    }
}
