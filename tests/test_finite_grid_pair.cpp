// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Independent continuum oscillator density and thermal finite differences.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include "numerics/finite_grid_pair.h"
using alea::numerics::FiniteGridPair;
using Catch::Approx;
namespace {
std::array<double,2> oscillator(const std::array<double,2>& a,const std::array<double,2>& b,double mu,double omega,double tau) {
    double squares=0,dot=0,delta=0;
    for (int d=0; d<2; ++d) { squares+=a[d]*a[d]+b[d]*b[d]; dot+=a[d]*b[d]; delta+=std::pow(a[d]-b[d],2); }
    const double x=omega*tau,s=std::sinh(x),c=std::cosh(x);
    return {std::log(s/x)+mu*omega*(squares*c-2*dot)/(2*s)-mu*delta/(2*tau),
        omega*c/s-1/tau+mu*omega*omega*(-squares+2*dot*c)/(2*s*s)+mu*delta/(2*tau*tau)};
}
}
TEST_CASE("Finite grid pair preserves free and constant potential density ratios", "[pair-grid][physics]") {
    for (double constant : {0.,-.7,.3}) {
        FiniteGridPair grid(.7,.4,3.,9,[constant](double){return constant;},0.);
        for (const auto& point : std::array{std::array{0.,0.},std::array{.13,-.22},std::array{3.,-3.}}) {
            const auto v=grid.evaluate(point,{.3,.8});
            CHECK(v.action==Approx(.4*constant).margin(3e-13));
            CHECK(v.tau_derivative==Approx(constant).margin(2e-12));
        }
        CHECK(std::isinf(grid.evaluate({3.01,0.},{0.,0.}).action));
    }
}
TEST_CASE("Finite grid pair refines toward the independent continuum oscillator kernel", "[pair-grid][physics]") {
    const double mu=.7,omega=.8,tau=.5;
    const std::array<double,2> a{.375,-.1875},b{-.1875,.375};
    const auto exact=oscillator(a,b,mu,omega,tau);
    double previous=1e10;
    for (int points : {9,17,33}) {
        FiniteGridPair grid(mu,tau,3.,points,[=](double r){return .5*mu*omega*omega*r*r;},0.);
        const auto v=grid.evaluate(a,b);
        const double error=std::abs(v.action-exact[0])+std::abs(v.tau_derivative-exact[1]);
        std::cout << "pair harmonic n=" << points << " u=" << v.action << " du=" << v.tau_derivative << " combined_error=" << error << '\n';
        CHECK(error<previous); previous=error;
        if (points==33) { CHECK(std::abs(v.action-exact[0])<.004); CHECK(std::abs(v.tau_derivative-exact[1])<.008); }
    }
}
TEST_CASE("Pair thermal derivative differentiates the same interpolated density ratio", "[pair-grid][derivative]") {
    const double tau=.4,h=1e-5;
    auto radial=[](double r){return -1/r;};
    FiniteGridPair base(.5,tau,3.,9,radial,-1.),plus(.5,tau+h,3.,9,radial,-1.),minus(.5,tau-h,3.,9,radial,-1.);
    for (const auto& a : std::array{std::array{0.,0.},std::array{.1,.2},std::array{2.9,-2.8}}) {
        const std::array<double,2> b{-.2,.3};
        const auto v=base.evaluate(a,b),reverse=base.evaluate(b,a);
        CHECK(std::isfinite(v.action));
        CHECK(v.action==Approx(reverse.action).margin(2e-13));
        CHECK(v.tau_derivative==Approx((plus.evaluate(a,b).action-minus.evaluate(a,b).action)/(2*h)).margin(2e-8));
    }
    CHECK(base.evaluate({0.,0.},{0.,0.}).action<0);
}
TEST_CASE("Pair grid rejects invalid dimensions scaling and unresolved exponentials at setup", "[pair-grid][admission]") {
    auto radial=[](double r){return r*r;};
    for (int n : {2,4,6,34}) CHECK_THROWS(FiniteGridPair(.5,.1,3.,n,radial,0.));
    CHECK_THROWS(FiniteGridPair(0.,.1,3.,9,radial,0.));
    CHECK_THROWS(FiniteGridPair(.5,0.,3.,9,radial,0.));
    CHECK_THROWS(FiniteGridPair(.5,.1,0.,9,radial,0.));
    CHECK_THROWS(FiniteGridPair(.5,1e100,3.,9,radial,0.));
    CHECK_THROWS(FiniteGridPair(.5,.1,3.,9,[](double){return NAN;},0.));
}

TEST_CASE("Positive pair exponential agrees with independent SciPy Pade density references", "[pair-grid][reference]") {
    // The validation battery's v0_ref_pair_grid.py; scipy.linalg.expm (Pade/scaling-squaring),
    // refined 32-point cell averages, mu=.5,tau=.4,extent=3,n=9,V=-1/r.
    // This checks the discrete exponential, not its continuum grid accuracy.
    FiniteGridPair grid(.5,.4,3.,9,[](double r){return -1/r;},-1.);
    const std::array fixtures{
        std::array{40.,40.,-1.623968644130289,-3.200361567524694},
        std::array{40.,41.,-1.1681253305544412,-2.6203228040134214},
        std::array{31.,49.,-.9580460667520345,-2.1794817625072103},
        std::array{30.,50.,-.6675457533721678,-1.6761836760834097}};
    auto point=[](int index){return std::array<double,2>{-3.+.75*(index/9),-3.+.75*(index%9)};};
    for (const auto& row : fixtures) {
        const auto value=grid.evaluate(point(int(row[0])),point(int(row[1])));
        CHECK(value.action==Approx(row[2]).margin(1e-8));
        CHECK(value.tau_derivative==Approx(row[3]).margin(1e-8));
    }
}
