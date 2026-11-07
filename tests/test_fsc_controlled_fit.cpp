// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Independent dielectric-slab image charges validate the FSC value and force
// moments and the new representation. The fitting algorithm is not a reference.
// V=2/eps sum_j q_j/hypot(rho,z_j), V'=-2*rho/eps sum_j q_j/(rho²+z_j²)^(3/2).
// Image reflections follow electrostatic interface continuity (Cavalcante et al.,
// Phys. Rev. B 97, 125427 (2018)); no BO or many-body sampling approximation.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "fast_screened_coulomb.h"

namespace {
using Catch::Approx;
using Context=std::unique_ptr<FSCContext,decltype(&fsc_free)>;
using Fit=std::unique_ptr<FSCControlledFit,decltype(&fsc_controlled_free)>;

std::array<double,2> images(double rho,double z,double lower_eps,double upper_eps) {
    constexpr double eps=4.;
    const double rd=(eps-lower_eps)/(eps+lower_eps),ru=(eps-upper_eps)/(eps+upper_eps);
    long double value=0.,derivative=0.,power=1.;
    // Source at zero, slab [-1,3], observer at z=0 or 2. Repeat images every
    // twice the slab thickness. |rd*ru|<=0.36 makes 160 terms more than enough.
    for (int n=0; n<160; ++n) {
        const std::array heights{z+8*n,6-z+8*n,z+2+8*n,8-z+8*n};
        const std::array weights{1.,ru,rd,rd*ru};
        for (int image=0; image<4; ++image) {
            const long double radius=std::hypot(static_cast<long double>(rho),heights[image]);
            value+=0.5L*power*weights[image]/radius;
            derivative-=0.5L*power*weights[image]*rho/(radius*radius*radius);
        }
        power*=rd*ru;
    }
    return {static_cast<double>(value),static_cast<double>(derivative)};
}

// Independent electrostatic boundary-value solve: use potentials at interfaces
// and at the source. The Dirichlet-to-Neumann matrix of a slab has diagonal
// eps*coth(k*t), off-diagonal -eps*csch(k*t). A source adds a jump 2 at z=0.
// This does not use FSC reflection coefficients, eps(k), integration or fits.
long double boundary_response(long double k,long double z) {
    if (k==0.) return 2.L/(1.L+1.5L);
    constexpr std::array<long double,4> positions{-1.,0.,1.,3.};
    constexpr std::array<long double,3> inner_eps{4.,4.,2.};
    Eigen::Matrix<long double,4,4> matrix=Eigen::Matrix<long double,4,4>::Zero();
    Eigen::Matrix<long double,4,1> rhs=Eigen::Matrix<long double,4,1>::Zero();
    matrix(0,0)=1.; matrix(3,3)=1.5L; rhs[1]=2.;
    for (int interval=0; interval<3; ++interval) {
        const long double phase=k*(positions[interval+1]-positions[interval]);
        const long double diagonal=inner_eps[interval]/std::tanh(phase);
        const long double off=-inner_eps[interval]/std::sinh(phase);
        matrix(interval,interval)+=diagonal; matrix(interval+1,interval+1)+=diagonal;
        matrix(interval,interval+1)=off; matrix(interval+1,interval)=off;
    }
    const Eigen::Matrix<long double,4,1> potential=matrix.ldlt().solve(rhs);
    if (z==0.) return potential[1];
    return (potential[2]*std::sinh(k*(3-z))+potential[3]*std::sinh(k*(z-1)))/std::sinh(2*k);
}

std::array<double,2> boundary_integral(double rho,double z,int panels) {
    constexpr long double cutoff=48.;
    const long double step=cutoff/panels;
    long double value=0.,derivative=0.;
    for (int node=0; node<=panels; ++node) {
        const long double k=node*step,phase=k*rho;
        const long double h=boundary_response(k,z)-std::exp(-k*z)/4.L;
        const int weight=node==0 || node==panels ? 1 : (node%2 ? 4 : 2);
        value+=weight*h*std::cyl_bessel_j(0.L,phase);
        derivative-=weight*k*h*std::cyl_bessel_j(1.L,phase);
    }
    const long double radius=std::hypot(static_cast<long double>(rho),z);
    return {static_cast<double>(0.5L/radius+2*step/3*value),
        static_cast<double>(-0.5L*rho/(radius*radius*radius)+2*step/3*derivative)};
}
}

TEST_CASE("Controlled FSC exactly recovers uniform direct and interlayer kernels across the full radial band", "[fsc-controlled]") {
    const double eps[4]={1.5,1.5,1.5,1.5},interfaces[3]={-1.,1.,3.};
    for (bool direct : {false,true}) {
        Context ctx(fsc_create(4,eps,interfaces,2,direct?0:3,direct?0.:2.),&fsc_free);
        FSCAccuracy accuracy{};
        Fit fit(fsc_fit_controlled(ctx.get(),.01,1333.,.01,.001,&accuracy),&fsc_controlled_free);
        REQUIRE(fit); CHECK(accuracy.status==0); CHECK(accuracy.segments==1);
        CHECK(accuracy.value_error_pct==0.); CHECK(accuracy.force_error_pct==0.);
        for (int point=0; point<=512; ++point) {
            const double rho=point==512 ? 1333. : .01*std::pow(1333./.01,point/512.);
            const double radius=std::hypot(rho,direct?0.:2.);
            CHECK(fsc_controlled_eval(fit.get(),rho)==Approx(2/(1.5*radius)).epsilon(2e-15));
            CHECK(fsc_controlled_d1(fit.get(),rho)==Approx(-2*rho/(1.5*radius*radius*radius)).epsilon(2e-15));
        }
    }
}

TEST_CASE("Controlled FSC values and forces meet independent slab image targets including the near core and far tail", "[fsc-controlled]") {
    for (double upper : {1.,2.}) for (bool direct : {false,true}) {
        const double eps[4]={1.,4.,4.,upper},interfaces[3]={-1.,1.,3.};
        Context ctx(fsc_create(4,eps,interfaces,2,direct?0:3,direct?0.:2.),&fsc_free);
        FSCAccuracy accuracy{};
        Fit fit(fsc_fit_controlled(ctx.get(),.01,1333.,.01,.01,&accuracy),&fsc_controlled_free);
        REQUIRE(fit); CHECK(accuracy.status==0);
        CHECK(accuracy.value_error_pct<=.005); CHECK(accuracy.force_error_pct<=.005);
        CHECK(accuracy.quadrature_value_pct<=.0001); CHECK(accuracy.quadrature_force_pct<=.0001);
        CHECK(accuracy.cutoff_value_pct<=.0001); CHECK(accuracy.cutoff_force_pct<=.0001);
        double max_value=0.,max_force=0.;
        for (int point=0; point<=1024; ++point) {
            const double rho=point==1024 ? 1333. : .01*std::pow(1333./.01,point/1024.);
            const auto reference=images(rho,direct?0.:2.,1.,upper);
            const double value=fsc_controlled_eval(fit.get(),rho),force=fsc_controlled_d1(fit.get(),rho);
            REQUIRE(std::isfinite(value)); REQUIRE(std::isfinite(force));
            max_value=std::max(max_value,std::abs(value/reference[0]-1));
            max_force=std::max(max_force,std::abs(force/reference[1]-1));
        }
        CAPTURE(upper,direct,max_value,max_force,accuracy.segments);
        CHECK(max_value<=1e-4); CHECK(max_force<=1e-4);
    }
}

TEST_CASE("Controlled FSC unequal interior dielectrics agree with an independent boundary solve and moment quadrature", "[fsc-controlled]") {
    const double eps[4]={1.,4.,2.,1.5},interfaces[3]={-1.,1.,3.};
    for (bool direct : {false,true}) {
        Context ctx(fsc_create(4,eps,interfaces,2,direct?0:3,direct?0.:2.),&fsc_free);
        FSCAccuracy accuracy{};
        Fit fit(fsc_fit_controlled(ctx.get(),.01,32.,.01,.01,&accuracy),&fsc_controlled_free);
        REQUIRE(fit);
        for (double rho : {.01,.0101,.1,.5,1.,3.,8.}) {
            const auto coarse=boundary_integral(rho,direct?0.:2.,16384);
            const auto fine=boundary_integral(rho,direct?0.:2.,32768);
            CAPTURE(direct,rho);
            // Independent reference must refine before it judges the production fit.
            CHECK(coarse[0]==Approx(fine[0]).epsilon(1e-8));
            CHECK(coarse[1]==Approx(fine[1]).epsilon(1e-8));
            CHECK(fsc_controlled_eval(fit.get(),rho)==Approx(fine[0]).epsilon(1e-4));
            CHECK(fsc_controlled_d1(fit.get(),rho)==Approx(fine[1]).epsilon(1e-4));
            double value,force; FSCAccuracy reference_accuracy{};
            REQUIRE(fsc_reference(ctx.get(),rho,.0001,&value,&force,&reference_accuracy));
            CHECK(value==Approx(fine[0]).epsilon(1e-8));
            CHECK(force==Approx(fine[1]).epsilon(1e-8));
        }
    }
}

TEST_CASE("Controlled FSC propagates to a different lower dielectric with independent transmitted images", "[fsc-controlled]") {
    const double eps[3]={1.,4.,2.},interfaces[2]={-1.,3.};
    Context ctx(fsc_create(3,eps,interfaces,2,1,-2.),&fsc_free);
    FSCAccuracy accuracy{};
    Fit fit(fsc_fit_controlled(ctx.get(),.01,1333.,.01,.01,&accuracy),&fsc_controlled_free);
    REQUIRE(fit);
    constexpr long double rd=.6L,ru=1.L/3.L;
    double max_value=0.,max_force=0.;
    for (int point=0; point<=1024; ++point) {
        const double rho=point==1024 ? 1333. : .01*std::pow(1333./.01,point/1024.);
        long double value=0.,derivative=0.,power=1.;
        // Transmission through the lower interface multiplies (1+rd).
        // The two image families repeat every twice the slab thickness.
        for (int n=0; n<160; ++n) {
            const std::array heights{2.+8*n,8.+8*n};
            const std::array weights{1.L,ru};
            for (int image=0; image<2; ++image) {
                const long double radius=std::hypot(static_cast<long double>(rho),heights[image]);
                value+=.5L*(1+rd)*power*weights[image]/radius;
                derivative-=.5L*(1+rd)*power*weights[image]*rho/(radius*radius*radius);
            }
            power*=rd*ru;
        }
        const double sampled=fsc_controlled_eval(fit.get(),rho),force=fsc_controlled_d1(fit.get(),rho);
        REQUIRE(std::isfinite(sampled)); REQUIRE(std::isfinite(force));
        max_value=std::max(max_value,std::abs(sampled/static_cast<double>(value)-1));
        max_force=std::max(max_force,std::abs(force/static_cast<double>(derivative)-1));
    }
    CAPTURE(max_value,max_force);
    CHECK(max_value<=1e-4); CHECK(max_force<=1e-4);
}

TEST_CASE("Controlled FSC refuses invalid inputs unresolved references and out of band evaluation", "[fsc-controlled]") {
    const double eps[3]={1.,4.,1.},interfaces[2]={-1.,1.};
    Context ctx(fsc_create(3,eps,interfaces,2,0,0.),&fsc_free);
    FSCAccuracy accuracy{};
    for (double invalid : {0.,-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        CHECK(fsc_fit_controlled(ctx.get(),.01,32.,invalid,.01,&accuracy)==nullptr);
        CHECK(accuracy.status==1);
        CHECK(fsc_fit_controlled(ctx.get(),.01,32.,.01,invalid,&accuracy)==nullptr);
    }
    CHECK(fsc_fit_controlled(ctx.get(),.01,1e12,.01,.01,&accuracy)==nullptr);
    CHECK(accuracy.status==2); // Work ceiling; no unconverged best-effort fit.
    CHECK(fsc_fit_controlled(nullptr,.01,32.,.01,.01,&accuracy)==nullptr);
    const double overflow_eps[3]={1e308,1e308,1e307};
    Context overflow(fsc_create(3,overflow_eps,interfaces,2,0,0.),&fsc_free);
    CHECK(fsc_fit_controlled(overflow.get(),.01,32.,.01,.01,&accuracy)==nullptr);
    CHECK(accuracy.status==1); // Finite inputs can overflow reflection arithmetic.
    CHECK(fsc_fit_controlled(ctx.get(),.01,32.,.01,.01,nullptr)==nullptr);
    CHECK(std::isnan(fsc_controlled_eval(nullptr,1.)));
    CHECK(std::isnan(fsc_controlled_d1(nullptr,1.)));
    fsc_controlled_free(nullptr);
    Fit fit(fsc_fit_controlled(ctx.get(),.01,4.,.01,.01,&accuracy),&fsc_controlled_free);
    REQUIRE(fit);
    for (double rho : {0.,.005,5.,std::numeric_limits<double>::infinity()}) {
        CHECK(std::isnan(fsc_controlled_eval(fit.get(),rho)));
        CHECK(std::isnan(fsc_controlled_d1(fit.get(),rho)));
    }
}

TEST_CASE("Controlled FSC derivatives follow the sampled scalar at every join and force targets refine independently", "[fsc-controlled]") {
    const double eps[4]={1.,4.,4.,2.},interfaces[3]={-1.,1.,3.};
    Context ctx(fsc_create(4,eps,interfaces,2,3,2.),&fsc_free);
    FSCAccuracy loose{},tight{};
    Fit first(fsc_fit_controlled(ctx.get(),.01,32.,.1,.1,&loose),&fsc_controlled_free);
    Fit fit(fsc_fit_controlled(ctx.get(),.01,32.,.1,.001,&tight),&fsc_controlled_free);
    REQUIRE(first); REQUIRE(fit);
    CHECK(tight.segments>loose.segments);
    CHECK(tight.value_error_pct<=.05); CHECK(tight.force_error_pct<=.0005);
    for (int join=1; join<tight.segments; ++join) {
        const double rho=fsc_controlled_boundary(fit.get(),join);
        const double left=std::nextafter(rho,0.),right=std::nextafter(rho,100.);
        const double value=fsc_controlled_eval(fit.get(),rho),force=fsc_controlled_d1(fit.get(),rho);
        CAPTURE(join,rho);
        CHECK(fsc_controlled_eval(fit.get(),left)==Approx(value).epsilon(2e-14));
        CHECK(fsc_controlled_eval(fit.get(),right)==Approx(value).epsilon(2e-14));
        CHECK(fsc_controlled_d1(fit.get(),left)==Approx(force).epsilon(2e-10));
        CHECK(fsc_controlled_d1(fit.get(),right)==Approx(force).epsilon(2e-10));
        // A symmetric derivative at a C1 join converges even with a C2 jump.
        const double step=1e-5*rho;
        const double difference=(fsc_controlled_eval(fit.get(),rho+step)-fsc_controlled_eval(fit.get(),rho-step))/(2*step);
        CHECK(difference==Approx(force).margin(2e-9).epsilon(2e-5));
        const auto reference=images(rho,2.,1.,2.);
        CHECK(force==Approx(reference[1]).epsilon(1e-5));
    }
    CHECK(std::isnan(fsc_controlled_boundary(fit.get(),-1)));
    CHECK(std::isnan(fsc_controlled_boundary(fit.get(),tight.segments+1)));
}
