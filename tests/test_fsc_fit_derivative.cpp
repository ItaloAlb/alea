// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Differentiate scalar fit values independently, not the underlying integral.
// These contracts also cover representations Alea's automatic fit does not pick.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include "fast_screened_coulomb.h"

namespace {
using Catch::Approx;
using Fit = std::unique_ptr<FSCFit, decltype(&fsc_fit_free)>;
using Context = std::unique_ptr<FSCContext, decltype(&fsc_free)>;
constexpr double lower = 0.2, upper = 8.0;

Fit build_fit(const FSCContext* context, FSCMethod method) {
    FSCFit* fit = nullptr;
    switch (method) {
    case FSC_METHOD_PADE: fit = fsc_fit_pade(context,3,3,lower,upper,30.0,1e-9); break;
    case FSC_METHOD_CHEBYSHEV: fit = fsc_fit_chebyshev(context,15,lower,upper,30.0,1e-9); break;
    case FSC_METHOD_LEGENDRE: fit = fsc_fit_legendre(context,15,lower,upper,30.0,1e-9); break;
    case FSC_METHOD_IMAGE_CHARGES: fit = fsc_fit_image_chg(context,8,lower,upper,30.0,1e-9); break;
    case FSC_METHOD_BSPLINE: fit = fsc_fit_bspline(context,10,lower,upper,30.0,1e-9); break;
    default: break;
    }
    return Fit(fit,fsc_fit_free);
}

Context stack(int kind) {
    const double uniform[] = {1.5}, contrast[] = {1.0,4.0,2.0,1.5};
    const double interfaces[] = {-1.0,1.0,3.0};
    return Context(kind == 0 ? fsc_create(1,uniform,nullptr,1,0,0.0) :
        fsc_create(4,contrast,interfaces,2,kind == 1 ? 0 : 3,2.0),fsc_free);
}

void check_difference(const FSCFit* fit, double rho) {
    const double derivative = fsc_fit_d1(fit,rho);
    REQUIRE(std::isfinite(derivative));
    for (double relative_step : {1e-4,5e-5,2e-5}) {
        const double step = relative_step*rho;
        const double difference = (fsc_fit_eval(fit,rho+step)-fsc_fit_eval(fit,rho-step))/(2*step);
        CAPTURE(rho,relative_step,fsc_fit_label(fit));
        CHECK(derivative == Approx(difference).epsilon(3e-6).margin(2e-9));
    }
}
}

TEST_CASE("Every FSC fit derivative matches scalar differences on uniform and contrasting stacks", "[fsc][derivative]") {
    for (int kind : {0,1,2})
        for (int method = 0; method < FSC_METHOD_COUNT; ++method) {
            DYNAMIC_SECTION("stack " << kind << " method " << method) {
                const auto context = stack(kind);
                REQUIRE(context);
                const auto fit = build_fit(context.get(),static_cast<FSCMethod>(method));
                REQUIRE(fit);
                CHECK(fsc_fit_supports_d1(fit.get()));
                for (double rho : {0.3,0.7,1.2,3.1,6.3}) check_difference(fit.get(),rho);
                // Polynomial/rational/image evaluators extrapolate; only the
                // spline freezes its numerator, so do not invent a band clamp.
                for (double rho : {0.15,12.0}) check_difference(fit.get(),rho);
            }
        }
}

TEST_CASE("FSC polynomial derivatives retain the analytic uniform Coulomb limit", "[fsc][derivative]") {
    const auto context = stack(0);
    REQUIRE(context);
    for (auto method : {FSC_METHOD_CHEBYSHEV,FSC_METHOD_LEGENDRE}) {
        const auto fit = build_fit(context.get(),method);
        REQUIRE(fit);
        for (double rho : {lower,0.3,1.0,4.0,upper})
            CHECK(fsc_fit_d1(fit.get(),rho) == Approx(-2.0/(1.5*rho*rho)).epsilon(1e-8));
    }
}

TEST_CASE("FSC spline derivatives preserve endpoint branches and smooth interior knots", "[fsc][derivative]") {
    const auto context = stack(2);
    REQUIRE(context);
    const auto fit = build_fit(context.get(),FSC_METHOD_BSPLINE);
    REQUIRE(fit);
    for (double rho : {lower/2,lower,upper,upper*2})
        CHECK(fsc_fit_d1(fit.get(),rho) == Approx(-fsc_fit_eval(fit.get(),rho)/rho).epsilon(2e-14));
    // The builder uses log-spaced fitting samples and quantile knots. Check
    // each interior knot from both sides through the public scalar evaluator.
    constexpr int samples = 14*8;
    for (int j = 0; j < 10; ++j) {
        const int index = static_cast<int>((j+1.0)/11.0*(samples-1));
        const double rho = lower*std::pow(upper/lower,double(index)/(samples-1));
        check_difference(fit.get(),rho);
        CHECK(fsc_fit_d1(fit.get(),std::nextafter(rho,0.0)) ==
            Approx(fsc_fit_d1(fit.get(),std::nextafter(rho,upper))).margin(1e-12));
    }
}

TEST_CASE("FSC derivative API rejects invalid radii and preserves array and shared-fit contracts", "[fsc][derivative]") {
    CHECK_FALSE(fsc_fit_supports_d1(nullptr));
    CHECK(std::isnan(fsc_fit_d1(nullptr,1.0)));
    const auto context = stack(1);
    REQUIRE(context);
    for (int method = 0; method < FSC_METHOD_COUNT; ++method) {
        const auto fit = build_fit(context.get(),static_cast<FSCMethod>(method));
        REQUIRE(fit);
        for (double rho : {0.0,-1.0,std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()})
            CHECK(std::isnan(fsc_fit_d1(fit.get(),rho)));
        const std::array<double,5> rhos{0.3,0.7,1.2,3.1,6.3}; // [n]
        std::array<double,5> values{},derivatives{},array{},shared{}; // [n]
        for (std::size_t i = 0; i < rhos.size(); ++i) {
            values[i] = fsc_fit_eval(fit.get(),rhos[i]);
            derivatives[i] = fsc_fit_d1(fit.get(),rhos[i]);
        }
        const long count = fsc_get_eval_count();
        fsc_fit_d1_array(fit.get(),rhos.data(),5,array.data());
        CHECK(array == derivatives);
        array = rhos;
        fsc_fit_d1_array(fit.get(),array.data(),5,array.data());
        CHECK(array == derivatives);
        fsc_fit_d1_array(fit.get(),nullptr,0,nullptr);
        #pragma omp parallel for num_threads(2)
        for (int i = 0; i < 5; ++i) shared[i] = fsc_fit_d1(fit.get(),rhos[i]);
        CHECK(shared == derivatives);
        CHECK(fsc_get_eval_count() == count);
        for (std::size_t i = 0; i < rhos.size(); ++i)
            CHECK(fsc_fit_eval(fit.get(),rhos[i]) == values[i]);
    }
}
