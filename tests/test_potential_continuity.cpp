// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Independent value integrals/finite differences, plus endpoint and k=0
// contracts for the sampled RK approximation. No virial estimator is enabled.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>
#include "hamiltonian/ewald_term.h"
#include "hamiltonian/rytova_keldysh.h"
#include "numerics/radial_cutoff_correction.h"
#include "population/walker_set.h"

namespace {
using namespace alea;
using namespace alea::numerics;
using Catch::Approx;
ParticleGroup group(std::size_t first, std::size_t last, double charge) {
    ParticleGroup result{};
    result.first=first; result.last=last; result.count=last-first; result.charge=charge;
    return result;
}

// Composite Simpson integration of the positive Laplace representation
// K^(n)(x)=(2/pi)/x^(n+1) int_0^infty (-s)^n exp(-s)/sqrt(1+(s/x)^2) ds.
// Uses a different variable and rule from the production setup quadrature.
double reference(double x, int order, int intervals) {
    constexpr double maximum=50.0;
    const double h=maximum/intervals;
    double sum=0.0;
    for (int i=0; i<=intervals; ++i) {
        const double s=i*h;
        const double weight=i == 0 || i == intervals ? 1 : (i%2 ? 4 : 2);
        sum += weight*std::pow(-s,order)*std::exp(-s)/std::hypot(1.0,s/x);
    }
    return (2.0/PI)*sum*h/3.0/std::pow(x,order+1);
}
}

TEST_CASE("Joined RK values and derivatives agree with refined independent Laplace integrals", "[continuity][keldysh]") {
    const auto& kernel=keldysh_cheb();
    double worst=0.0,change=0.0;
    for (double x : {1.0,5.0,15.0,31.0,32.0,33.0,35.0,37.0,39.0,40.0,41.0,80.0,1000.0}) {
        for (int order : {0,1,2}) {
            const double value=reference(x,order,32768),refined=reference(x,order,65536);
            CAPTURE(x,order);
            REQUIRE(std::abs(value-refined)<(order == 0 ? 2e-10 : 3e-9));
            const double sampled=order == 0 ? kernel.eval(x) : (order == 1 ? kernel.d1(x) : kernel.d2(x));
            CHECK(sampled == Approx(refined).epsilon(3e-6).margin(order == 0 ? 1e-8 : 2e-7));
            if (order == 0) worst=std::max(worst,std::abs(sampled-refined));
        }
        if (x>=40.0) {
            const double inv=1.0/x;
            change=std::max(change,std::abs(kernel.eval(x)-(2.0/PI)*inv*(1.0-inv*inv)));
            CHECK(kernel.eval(x) == Approx(reference(x,0,65536)).margin(2e-13));
        }
    }
    std::cout << std::setprecision(17) << "RK independent value error = " << worst << ", tail value change = " << change << '\n';
    CHECK(kernel.eval(1e150)>0.0);
    CHECK(kernel.eval(1e150)*1e150 == Approx(2.0/PI).epsilon(1e-14));
    for (double x : {0.1,1.0,15.0,31.0,32.0,35.0,39.0,40.0,41.0,1000.0})
        for (double scale : {1e-4,5e-5,2e-5}) {
            const double h=x*scale;
            CHECK(kernel.d2(x) == Approx((kernel.d1(x+h)-kernel.d1(x-h))/(2*h)).epsilon(3e-6).margin(1e-8));
        }
}

TEST_CASE("Radial cutoff matching reproduces endpoint residuals derivatives and its exact mean", "[continuity][correction]") {
    const RadialCutoffCorrection correction(2.0,3.0,0.3,-0.2,0.4);
    CHECK(correction.eval(2.0) == 0.0); CHECK(correction.d1(2.0) == 0.0); CHECK(correction.d2(2.0) == 0.0);
    CHECK(correction.eval(3.0) == Approx(0.3)); CHECK(correction.d1(3.0) == Approx(-0.2));
    CHECK(correction.d2(3.0) == Approx(0.4));
    for (double r : {2.1,2.3,2.5,2.7,2.9})
        for (double h : {1e-4,5e-5,2e-5}) {
            CHECK(correction.d1(r) == Approx((correction.eval(r+h)-correction.eval(r-h))/(2*h)).margin(1e-7));
            CHECK(correction.d2(r) == Approx((correction.d1(r+h)-correction.d1(r-h))/(2*h)).margin(2e-7));
        }
    double integral=0.0;
    constexpr int intervals=1024;
    for (int i=0; i<=intervals; ++i) {
        const double r=2.0+double(i)/intervals;
        integral += (i == 0 || i == intervals ? 1 : (i%2 ? 4 : 2))*r*correction.eval(r);
    }
    CHECK(correction.radial_integral() == Approx(integral/(3*intervals)).margin(2e-12));
    CHECK(RadialCutoffCorrection{}.eval(1.0) == 0.0);
    CHECK_THROWS_AS(RadialCutoffCorrection(2,2,0,0,0),std::invalid_argument);
    CHECK_THROWS_AS(RadialCutoffCorrection(2,3,INFINITY,0,0),std::invalid_argument);
    // Avoid width² overflow and inverse-width² underflow when the curvature
    // itself is representable, even on a very wide synthetic interval.
    const RadialCutoffCorrection wide(0,1e200,0,0,1e-300);
    CHECK(wide.d2(1e200)/1e-300 == Approx(1.0).epsilon(1e-14));
}

TEST_CASE("RK Ewald sampled short range reaches zero with two derivatives without altering the spline", "[continuity][ewald]") {
    const SimulationCell cell(2,std::array{6.0,0.0,0.0,6.0});
    const auto a=group(0,1,-1),b=group(1,2,1);
    for (const auto& spec : {PotentialSpec::rytova_keldysh(1.5,0.7),
                             PotentialSpec::rytova_keldysh(1.5,0.075,0.6)}) {
        EwaldTerm term({a,b},spec,1.5,cell,1e-8);
        const auto& setup=term.setup(); const auto& correction=term.cutoff_correction();
        const auto index=setup.long_range.scaled.size()-3;
        const double h=1.0/setup.long_range.inv_delta, r=std::nextafter(setup.r_cut,0.0);
        const double rho=std::hypot(r,spec.d_layer), x=rho/spec.r0, drho=r/rho;
        const double pre=PI/(2*spec.epsilon*spec.r0), d2rho=(spec.d_layer/rho)*(spec.d_layer/rho)/rho;
        double value,slope,curvature;
        setup.long_range.eval(r,value,slope,curvature);
        const double residual=keldysh_cheb().eval(x)*pre-value-correction.eval(r);
        CHECK(std::abs(residual)<2e-13);
        CHECK(std::abs(pre/spec.r0*keldysh_cheb().d1(x)*drho-slope-correction.d1(r))<2e-12);
        CHECK(std::abs(pre/spec.r0*(keldysh_cheb().d2(x)/spec.r0*drho*drho+keldysh_cheb().d1(x)*d2rho)-curvature-correction.d2(r))<2e-10);
        // Endpoint coefficients still equal the original ideal transform-pair
        // kernel. Matching does not mutate that spline or its Fourier weights.
        CHECK(setup.long_range.scaled[index] == Approx(spec.v_real(setup.r_cut)).epsilon(2e-14));
        double maximum=0.0;
        for (int i=0; i<=100; ++i)
            maximum=std::max(maximum,std::abs(correction.eval(correction.begin()+h*i/100)));
        std::cout << std::setprecision(17) << "RK cutoff match r0=" << spec.r0 << " d=" << spec.d_layer << " residual=" << residual << " max correction=" << maximum << '\n';
        CHECK(maximum<2e-8);
        // Recover the actual short-range contribution from the full scalar
        // term by subtracting an independent structure-factor sum/constants.
        WalkerSet batch(1,2,2,cell);
        auto workspace=term.make_potential_gradient_workspace(batch);
        for (double separation : {correction.begin()+0.25*h,correction.begin()+0.7*h,r,setup.r_cut}) {
            batch.positions()[2]=separation; batch.dist_tables().build_all(batch.positions());
            std::vector<double> energy(1,0.0); term.evaluate_potential(batch,energy);
            double reciprocal=0.0;
            for (std::size_t k=0; k<setup.n_k(); ++k)
                reciprocal += 2*setup.weight[k]*(1.0-std::cos(setup.kx[k]*separation));
            const double observed=energy[0]-term.self_energy()-term.background_energy()-reciprocal;
            const double rr=std::hypot(separation,spec.d_layer);
            const double expected=separation < setup.r_cut ?
                -(pre*keldysh_cheb().eval(rr/spec.r0)-setup.long_range.eval(separation)-correction.eval(separation)) : 0.0;
            CHECK(observed == Approx(expected).margin(2e-13));
            if (separation < r) {
                std::vector<double> gradient(4,0.0);
                term.evaluate_potential_gradient(batch,gradient,*workspace);
                for (double scale : {1e-3,5e-4,2e-4}) {
                    const double step=h*scale;
                    auto scalar=[&](double position) {
                        batch.positions()[2]=position;
                        batch.dist_tables().build_all(batch.positions());
                        std::vector<double> result(1,0.0);
                        term.evaluate_potential(batch,result);
                        return result[0];
                    };
                    const double plus=scalar(separation+step),minus=scalar(separation-step);
                    CHECK(gradient[2] == Approx((plus-minus)/(2*step)).margin(2e-10));
                }
            }
        }
    }
    for (double r0 : {3000.0,6000.0})
        CHECK_THROWS_WITH(EwaldTerm({a,b},PotentialSpec::rytova_keldysh(1.5,r0),1.5,cell,1e-8),
            Catch::Matchers::ContainsSubstring("at or below the sampled low clamp"));
}

TEST_CASE("RK cutoff matching includes the correction mean in charged and class-coupled k zero energies", "[continuity][ewald]") {
    const SimulationCell cell(2,std::array{6.0,0.0,0.0,6.0});
    const auto spec=PotentialSpec::rytova_keldysh(1.5,0.7);
    const auto a=group(0,1,-1),b=group(1,2,1);
    for (bool several : {false,true}) {
        EwaldTerm term=several ? EwaldTerm({a,b},{{0,0,spec},{0,1,PotentialSpec::coulomb(1.5)},{1,1,PotentialSpec::coulomb(1.5)}},cell,1e-8) :
            EwaldTerm({a},spec,1.5,cell,1e-8);
        const auto& correction=term.cutoff_correction();
        double integral=0.0;
        constexpr int intervals=1024;
        const double width=correction.end()-correction.begin();
        for (int i=0; i<=intervals; ++i) {
            const double r=correction.begin()+width*i/intervals;
            integral += (i == 0 || i == intervals ? 1 : (i%2 ? 4 : 2))*r*correction.eval(r);
        }
        integral *= width/(3*intervals);
        const double shift=PI/cell.volume()*integral;
        const double coulomb_mean=several ? build_ewald(cell,1e-8,PotentialSpec::coulomb(1.5)).sr_k0 : 0.0;
        const double baseline=several ? spec.k0_offset/(2*cell.volume())-0.5*term.setup().sr_k0+0.5*coulomb_mean : -0.5*term.setup().sr_k0;
        CHECK(term.background_energy()-baseline == Approx(shift).margin(2e-14));
    }
}
