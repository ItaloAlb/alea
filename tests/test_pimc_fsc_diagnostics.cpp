// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Reporting of the sampled open FSC fit, separate from estimator admission.
// Controlled fit targets do not replace the separate estimator admission gate.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "hamiltonian/harmonic_trap.h"
#include "hamiltonian/screened_coulomb.h"
#include "io/parser.h"
#include "pimc/pimc.h"
#include "qmc/app.h"

using namespace alea;
using Catch::Matchers::ContainsSubstring;
namespace {
ParticleGroup group(std::size_t first,double charge) {
    ParticleGroup result;
    result.first=first; result.last=first+1; result.count=1; result.charge=charge;
    result.mass={1.}; result.inv_mass={1.}; return result;
}
std::string notes(const RunResult& result) {
    for (const auto& [key,value] : result.metadata) if (key=="potential_approximation") return value;
    throw std::runtime_error("missing potential_approximation");
}
}

TEST_CASE("FSC metadata records controlled checks separately from represented virial admission", "[fsc-diagnostics]") {
    const auto electron=group(0,-1.),hole=group(1,1.);
    const std::vector groups{electron,hole};
    ScreenedCoulombTerm pair(electron,hole,{1.5,1.5,1.5,1.5},{-1.,1.,3.},2,3,2.,.01);
    const auto text=pair.potential_approximation_notes();
    CHECK_THAT(text,ContainsSubstring("requested_value_target_pct=0.01"));
    CHECK_THAT(text,ContainsSubstring("sampled_value_error_pct="));
    CHECK_THAT(text,ContainsSubstring("sampled_force_error_pct="));
    CHECK_THAT(text,ContainsSubstring("requested_force_target_pct=0.01"));
    CHECK_THAT(text,ContainsSubstring("force_target_met=yes"));
    CHECK_THAT(text,ContainsSubstring("value_target_met=yes"));
    CHECK_THAT(text,ContainsSubstring("sampled checks are not global force certification"));
    CHECK_THAT(text,ContainsSubstring("frozen"));
    CHECK(pair.clone()->potential_approximation_notes()==text);
    ScreenedCoulombTerm direct(electron,electron,{1.5},{},1,0,0.,.01);
    CHECK_THAT(direct.potential_approximation_notes(),ContainsSubstring("value_target_met=yes"));
    Hamiltonian ham; ham.add_component(pair.clone());
    ham.add_component(std::make_unique<HarmonicTrap>(groups,1.));
    PimcConfig cfg; cfg.beta=1.; cfg.time_step=.125; cfg.energy="centroid_virial";
    cfg.samples=1;
    CHECK_NOTHROW(Pimc(cfg,groups,ham,SimulationCell{},1,2));
    CHECK(pair.supports_periodic_virial(SimulationCell(2,std::vector{4.,0.,0.,4.}),2));
    Action without_potential(groups,ham,1,true,false);
    CHECK(without_potential.potential_approximation_notes().empty());
}

TEST_CASE("Six particle thermodynamic FSC PIMC carries actual fit diagnostics through App and clones", "[fsc-diagnostics]") {
    auto config=io::parse_config_file(std::string(ALEA_CONFIG_DIR)+"/pimc_many_body_eh.yaml");
    config.coulombs.pop_back(); // Replace the e-h term, preserving both intralayer Coulomb terms.
    io::ScreenedCoulombSpec screened;
    screened.pair={"electrons","holes"}; screened.epsilons={1.5,1.5,1.5,1.5};
    screened.interfaces={-1.,.5,1.5}; screened.source_layer=2; screened.target_layer=3;
    screened.fit_tol=.01;
    config.screened.push_back(screened);
    App app(config); auto pack=app.build_drivers(); auto* driver=dynamic_cast<Pimc*>(pack.drivers[0].get());
    REQUIRE(driver); driver->run();
    CHECK_THAT(notes(driver->run_result()),ContainsSubstring("screened_coulomb: sampled FSC controlled Hermite"));
    CHECK_THAT(notes(driver->run_result()),ContainsSubstring("sampled checks are not global force certification"));
    CHECK_THAT(driver->summary(),ContainsSubstring("sampled_value_error_pct="));
    const auto positions=driver->export_paths().positions;
    config.pimc[0].threads=4;
    App threaded(config); auto threaded_pack=threaded.build_drivers();
    auto* clone=dynamic_cast<Pimc*>(threaded_pack.drivers[0].get());
    REQUIRE(clone); clone->run();
    CHECK(clone->export_paths().positions==positions);
    CHECK(notes(clone->run_result())==notes(driver->run_result()));
}
