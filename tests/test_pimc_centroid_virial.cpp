// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Independent finite-M Gaussian cubature and temperature-scaling checks of
// the open primitive identity (Yamamoto, JCP 123, 104101 (2005), eqs. 6-10).
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <iostream>
#include <iomanip>
#include <string_view>
#include "hamiltonian/harmonic_trap.h"
#include "hamiltonian/coulomb.h"
#include "hamiltonian/moire.h"
#include "hamiltonian/screened_coulomb.h"
#include "io/config_writer.h"
#include "io/parser.h"
#include "pimc/pimc.h"
#include "pimc/primitive_action.h"
#include "qmc/app.h"

using namespace alea;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;

namespace {
ParticleGroup particle(std::size_t p, std::vector<double> mass) {
    ParticleGroup g;
    g.name="g"+std::to_string(p); g.first=p; g.last=p+1; g.count=1;
    g.mass=mass; for (double m : mass) g.inv_mass.push_back(1/m);
    return g;
}

// An admitted test potential with cross-particle AND cross-axis couplings.
// Positive-definite H confines absolute coordinates. It has no name dispatch.
class CoupledQuadratic final : public HamiltonianComponent {
public:
    Eigen::Matrix4d hessian;
    Eigen::Vector4d center;
    CoupledQuadratic() {
        Eigen::Matrix4d factor;
        factor << 1.2,.2,-.3,0, 0,.9,.1,.2, 0,0,1.4,-.1, 0,0,0,.7;
        hessian=factor.transpose()*factor;
        center << .7,-.4,1.3,.2;
    }
    std::string name() const override { return "coupled_quadratic_reference"; }
    bool supports_potential_gradient() const noexcept override { return true; }
    bool supports_open_centroid_virial() const noexcept override { return true; }
    bool confines_open_coordinate(std::size_t p,std::size_t d,std::size_t dim) const noexcept override {
        return dim == 2 && p < 2 && d < 2;
    }
    void evaluate_potential(const ParticleBatch& batch,std::span<double> out,
                            std::size_t begin,std::size_t end) const override {
        end=std::min(end,batch.num_walkers());
        for (auto k=begin; k<end; ++k) {
            Eigen::Vector4d x;
            for (int a=0; a<4; ++a) x[a]=batch.positions()[a*batch.num_walkers()+k]-center[a];
            out[k]+=.5*x.dot(hessian*x);
        }
    }
    void evaluate_potential_gradient(const ParticleBatch& batch,std::span<double> out,
        PotentialGradientWorkspace& workspace,std::size_t begin,std::size_t end) const override {
        end=workspace.checked_end(batch,*this,out,begin,end);
        for (auto k=begin; k<end; ++k) {
            Eigen::Vector4d x;
            for (int a=0; a<4; ++a) x[a]=batch.positions()[a*batch.num_walkers()+k]-center[a];
            const Eigen::Vector4d gradient=hessian*x;
            for (int a=0; a<4; ++a) out[a*batch.num_walkers()+k]+=gradient[a];
        }
    }
    std::unique_ptr<HamiltonianComponent> clone() const override {
        return std::make_unique<CoupledQuadratic>(*this);
    }
};
class MissingGradient final : public HamiltonianComponent {
public:
    std::string name() const override { return "missing_gradient_fixture"; }
    std::unique_ptr<HamiltonianComponent> clone() const override { return std::make_unique<MissingGradient>(); }
};
class UnsupportedAction final : public ActionComponent {
public:
    double total_action(const PathSet&) override { return 0; }
    bool supports_free_bridge() const noexcept override { return true; }
    std::unique_ptr<ActionComponent> clone() const override { return std::make_unique<UnsupportedAction>(); }
};
double scalar(const RunResult& result,const std::string& key) {
    for (const auto& [name,value] : result.scalars) if (name == key) return value;
    FAIL("missing scalar " << key); return 0;
}
std::string metadata(const RunResult& result,const std::string& key) {
    for (const auto& [name,value] : result.metadata) if (name == key) return value;
    FAIL("missing metadata " << key); return {};
}
const std::vector<double>& series(const RunResult& result,std::string_view key) {
    for (const auto& [name,value] : result.series) if (name == key) return value;
    throw std::runtime_error("missing series "+std::string(key));
}
}

TEST_CASE("Open centroid energy integrates exactly over coupled anisotropic finite bead Gaussians", "[centroid-virial][physics]") {
    const std::vector groups{particle(0,{.7,1.3}),particle(1,{2.1,.9})};
    CoupledQuadratic reference;
    Hamiltonian ham; ham.add_component(reference.clone());
    constexpr double beta=3.1;
    const Eigen::Vector4d masses(.7,1.3,2.1,.9);
    Eigen::MatrixXd normal_hessian(4,4);
    for (int a=0; a<4; ++a) for (int b=0; b<4; ++b)
        normal_hessian(a,b)=reference.hessian(a,b)/std::sqrt(masses[a]*masses[b]);
    // For this positive-definite symmetric matrix the singular values are
    // the squared physical normal frequencies, independently of path modes.
    Eigen::JacobiSVD<Eigen::MatrixXd,Eigen::NoQRPreconditioner> modes(normal_hessian);
    REQUIRE(modes.info() == Eigen::Success);
    for (std::size_t M : {1,2,7,16}) {
        CAPTURE(M);
        const double tau=beta/M;
        const int n=static_cast<int>(4*M);
        // Hessian of the full sampled action, not of either energy estimator.
        Eigen::MatrixXd precision=Eigen::MatrixXd::Zero(n,n);
        for (std::size_t k=0; k<M; ++k) {
            const auto next=(k+1)%M;
            for (int a=0; a<4; ++a) {
                for (int b=0; b<4; ++b) precision(a*M+k,b*M+k)+=tau*reference.hessian(a,b);
                precision(a*M+k,a*M+k)+=masses[a]/tau;
                precision(a*M+next,a*M+next)+=masses[a]/tau;
                precision(a*M+k,a*M+next)-=masses[a]/tau;
                precision(a*M+next,a*M+k)-=masses[a]/tau;
            }
        }
        Eigen::JacobiSVD<Eigen::MatrixXd,Eigen::NoQRPreconditioner> gaussian_modes(precision,Eigen::ComputeFullU);
        REQUIRE(gaussian_modes.info() == Eigen::Success);
        REQUIRE(gaussian_modes.singularValues().minCoeff() > 0.0);
        const Eigen::MatrixXd root=gaussian_modes.matrixU()*
            gaussian_modes.singularValues().cwiseSqrt().cwiseInverse().asDiagonal();
        PathSet path(M,2,2,tau);
        Action action(groups,ham,2); action.prepare_open_centroid(path);
        ActionThermoMeasure cv,thermo;
        // +/- sqrt(n) columns of L are exact cubature for every quadratic
        // expectation under this Gaussian: mean zero, covariance L L^T.
        for (int column=0; column<n; ++column) for (double sign : {-1.0,1.0}) {
            for (int a=0; a<4; ++a) for (std::size_t k=0; k<M; ++k)
                path.positions()[a*M+k]=reference.center[a]+sign*std::sqrt(n)*root(a*M+k,column);
            const auto snapshot=path.all_positions();
            const auto one=action.measure_open_centroid(path),two=action.measure_thermo(path);
            CHECK(path.all_positions() == snapshot);
            CHECK(one.energy == Approx(one.kinetic+one.potential).margin(2e-13));
            cv.energy+=one.energy/(2*n); cv.potential+=one.potential/(2*n); cv.kinetic+=one.kinetic/(2*n);
            thermo.energy+=two.energy/(2*n);
            if (M == 1) CHECK(one.kinetic == Approx(4/(2*beta)).margin(2e-14));
        }
        // Exact partition function in mass-normal normal modes:
        // Z_M=product_j [2 sinh(M asinh(beta omega_j/(2M)))]^-1.
        double exact=0;
        for (double eigenvalue : modes.singularValues()) {
            const double omega=std::sqrt(eigenvalue),x=tau*omega/2;
            exact+=omega/(2*std::sqrt(1+x*x)*std::tanh(M*std::asinh(x)));
        }
        CHECK(cv.energy == Approx(exact).margin(3e-12));
        CHECK(thermo.energy == Approx(exact).margin(3e-12));
        CHECK(cv.potential == Approx(exact/2).margin(3e-12));
        CHECK(cv.kinetic == Approx(exact/2).margin(3e-12));
        auto cloned=action.clone(); cloned.prepare_open_centroid(path);
        CHECK(cloned.measure_open_centroid(path).energy == action.measure_open_centroid(path).energy);
    }
}

TEST_CASE("Open centroid measurement equals an independent scaled temperature derivative", "[centroid-virial][physics]") {
    auto a=particle(0,{.8,1.7}),b=particle(1,{1.9,.6});
    a.charge=-1; b.charge=1;
    const std::vector groups{a,b};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(groups,1.4,std::vector{.3,-.2}));
    ham.add_component(std::make_unique<CoulombTerm>(a,b,1.8,.7));
    ham.add_component(std::make_unique<MoireTerm>(2,.13,.07,.4,
        std::vector<std::vector<double>>{{.8,.3},{-.2,.9}},
        std::vector<std::pair<std::size_t,std::size_t>>{{0,2}}));
    PathSet path(8,2,2,.2);
    for (std::size_t i=0; i<path.positions().size(); ++i) path.positions()[i]=.3*std::sin(.7*i)+.02*i;
    Action action(groups,ham,2); action.prepare_open_centroid(path);
    const auto value=action.measure_open_centroid(path);
    const auto positions=path.all_positions();
    std::array<double,4> centroids{};
    for (int axis=0; axis<4; ++axis) for (std::size_t k=0; k<8; ++k) centroids[axis]+=path.positions()[axis*8+k]/8;
    const double initial_action=action.total_action(path);
    auto log_weight=[&](double beta) {
        const double ratio=beta/path.beta();
        PathSet mapped(8,2,2,beta/8);
        for (int axis=0; axis<4; ++axis) for (std::size_t k=0; k<8; ++k)
            mapped.positions()[axis*8+k]=centroids[axis]+std::sqrt(ratio)*(positions[axis*8+k]-centroids[axis]);
        // Gaussian normalization and 28 fluctuation-coordinate Jacobian powers.
        return -16*std::log(ratio)+14*std::log(ratio)-action.total_action(mapped)+initial_action;
    };
    for (double h : {2e-4,1e-4,5e-5})
        CHECK(value.energy == Approx(-(log_weight(path.beta()+h)-log_weight(path.beta()-h))/(2*h)).margin(3e-8));
    CHECK(path.all_positions() == positions);
}

TEST_CASE("Centroid setup rejects unsupported domains and does not infer compatibility from bridges", "[centroid-virial][admission]") {
    const std::vector groups{particle(0,{1.0})};
    Hamiltonian ham; ham.add_component(std::make_unique<HarmonicTrap>(groups,1.0));
    PimcConfig cfg; cfg.energy="centroid_virial"; cfg.samples=std::numeric_limits<int>::max();
    cfg.beta=1; cfg.time_step=.125;
    const SimulationCell periodic(1,std::vector{20.0});
    REQUIRE_THROWS_WITH(Pimc(cfg,groups,ham,periodic,1,1),ContainsSubstring("periodic minimum-image"));
    const Hamiltonian empty;
    REQUIRE_THROWS_WITH(Pimc(cfg,groups,empty,SimulationCell{},1,1),ContainsSubstring("absolute open confinement"));
    Hamiltonian inverted; inverted.add_component(std::make_unique<HarmonicTrap>(groups,-1.0));
    REQUIRE_THROWS_WITH(Pimc(cfg,groups,inverted,SimulationCell{},1,1),ContainsSubstring("harmonic_trap"));
    ham.add_component(std::make_unique<MissingGradient>());
    REQUIRE_THROWS_WITH(Pimc(cfg,groups,ham,SimulationCell{},1,1),ContainsSubstring("missing_gradient_fixture"));
    Hamiltonian confined; confined.add_component(std::make_unique<HarmonicTrap>(groups,1.0));
    Action action(groups,confined,1); action.add_component(std::make_unique<UnsupportedAction>());
    REQUIRE(action.supports_free_bridge());
    REQUIRE_THROWS_WITH(Pimc(cfg,groups,confined,SimulationCell{},1,1,nullptr,&action),ContainsSubstring("no open centroid contribution"));
    const std::vector two_groups{particle(0,{1.0}),particle(1,{1.0})};
    Hamiltonian partial; partial.add_component(std::make_unique<HarmonicTrap>(groups,1.0));
    REQUIRE_THROWS_WITH(Pimc(cfg,two_groups,partial,SimulationCell{},1,2),ContainsSubstring("particle 1"));
    Hamiltonian fitted; fitted.add_component(std::make_unique<HarmonicTrap>(two_groups,1.0));
    fitted.add_component(std::make_unique<ScreenedCoulombTerm>(two_groups[0],two_groups[1],
        std::vector{1.5,1.5,1.5,1.5},std::vector{-1.0,1.0,3.0},2,3,2.0,.01));
    cfg.samples=1;
    REQUIRE_NOTHROW(Pimc(cfg,two_groups,fitted,SimulationCell{},1,2));
    cfg.energy="misspelled";
    REQUIRE_THROWS_WITH(Pimc(cfg,groups,confined,SimulationCell{},1,1),ContainsSubstring("misspelled"));
}

TEST_CASE("Centroid selection preserves paths RNG acceptance and paired block provenance", "[centroid-virial][driver]") {
    const std::vector groups{particle(0,{1.0,2.3})};
    Hamiltonian ham; ham.add_component(std::make_unique<HarmonicTrap>(groups,1.21,std::vector{.4,-.7}));
    PimcConfig cfg; cfg.samples=4; cfg.beta=2; cfg.time_step=.125;
    cfg.equilibration_blocks=64; cfg.accumulation_blocks=512; cfg.steps_per_block=32; cfg.seed=20261004;
    Pimc thermo(cfg,groups,ham,SimulationCell{},2,1); thermo.run();
    cfg.energy="centroid_virial"; cfg.compare_thermodynamic=true;
    Pimc cv(cfg,groups,ham,SimulationCell{},2,1); cv.run();
    cfg.threads=4;
    Pimc threaded(cfg,groups,ham,SimulationCell{},2,1); threaded.run();
    cfg.compare_thermodynamic=false;
    Pimc unpaired(cfg,groups,ham,SimulationCell{},2,1); unpaired.run();
    CHECK(cv.export_paths().positions == thermo.export_paths().positions);
    CHECK(cv.export_paths().positions == threaded.export_paths().positions);
    CHECK(cv.export_paths().positions == unpaired.export_paths().positions);
    CHECK(cv.acceptance_ratio() == thermo.acceptance_ratio());
    CHECK(std::equal(cv.block_energies().begin(),cv.block_energies().end(),threaded.block_energies().begin()));
    CHECK(std::equal(cv.block_energies().begin(),cv.block_energies().end(),unpaired.block_energies().begin()));
    CHECK(std::equal(cv.block_potential().begin(),cv.block_potential().end(),thermo.block_potential().begin()));
    const auto result=cv.run_result();
    CHECK(metadata(result,"energy_estimator") == "centroid_virial");
    CHECK(metadata(thermo.run_result(),"energy_estimator") == "thermodynamic");
    const auto& difference=series(result,"acc_block_energy_difference");
    for (std::size_t k=0; k<difference.size(); ++k)
        CHECK(difference[k] == Approx(cv.block_energies()[k]-thermo.block_energies()[k]).margin(2e-14));
    const auto paired=numerics::reblock(difference);
    CHECK(scalar(result,"energy_difference_error") == paired.error);
    CHECK(std::abs(paired.mean)<4*paired.error);
    const double x=cfg.time_step*1.1/2;
    const double exact=1.1/(std::sqrt(1+x*x)*std::tanh(16*std::asinh(x)));
    REQUIRE(cv.energy_error()<.025);
    CHECK(std::abs(cv.energy()-exact)<4*cv.energy_error());
}

TEST_CASE("Centroid config selection and paired diagnostic round trip through snapshots", "[centroid-virial][config]") {
    const std::string yaml=R"(project: {name: centroid_snapshot, seed: 11}
system:
  lattice: {type: open, dimension: 1}
  particles:
    - {name: e, group: e, type: boltzmannon, mass: 1.0, charge: 0, up: 1, down: 0}
wavefunction: []
hamiltonian:
  terms:
    - {type: harmonic_trap, omega: 1.0}
qmc:
  - {method: pimc, beta: 2.0, time_step: 0.125, energy: centroid_virial, compare_thermodynamic: true}
)";
    const auto config=io::parse_config(yaml);
    REQUIRE(config.pimc.size() == 1);
    const auto resolved=build_pimc_config(config.pimc[0],11);
    CHECK(resolved.energy == "centroid_virial"); CHECK(resolved.compare_thermodynamic);
    const auto directory=std::filesystem::temp_directory_path()/"alea_centroid_snapshot_20261004";
    std::filesystem::create_directories(directory/"run");
    const auto source=directory/"source.yaml";
    std::ofstream(source)<<yaml;
    io::write_config_snapshot(source.string(),{},directory/"run",11);
    const auto replay=io::parse_config_file((directory/"run/source.yaml").string());
    CHECK(replay.pimc[0].energy == config.pimc[0].energy);
    CHECK(replay.pimc[0].compare_thermodynamic);
    std::string invalid=yaml;
    invalid.replace(invalid.find("centroid_virial"),std::string("centroid_virial").size(),"misspelled");
    CHECK_THROWS_WITH(io::parse_config(invalid),ContainsSubstring("misspelled"));
    invalid=yaml; invalid.replace(invalid.find("centroid_virial"),std::string("centroid_virial").size(),"thermodynamic");
    CHECK_THROWS_WITH(io::parse_config(invalid),ContainsSubstring("compare_thermodynamic requires"));
    std::string default_yaml=yaml;
    default_yaml.erase(default_yaml.find(", energy: centroid_virial"),
        std::string(", energy: centroid_virial, compare_thermodynamic: true").size());
    std::ofstream(source)<<default_yaml;
    io::write_config_snapshot(source.string(),{},directory/"run",11);
    const auto defaults=YAML::LoadFile((directory/"run/source.yaml").string());
    CHECK(defaults["qmc"][0]["energy"].as<std::string>() == "thermodynamic");
    CHECK_FALSE(defaults["qmc"][0]["compare_thermodynamic"].as<bool>());
    std::filesystem::remove_all(directory);
}

TEST_CASE("Centroid measurement runs through App for six confined electrons and holes", "[centroid-virial][many-body]") {
    auto config=io::parse_config_file(std::string(ALEA_CONFIG_DIR)+"/pimc_many_body_eh.yaml");
    App thermal_app(config); auto thermal_pack=thermal_app.build_drivers();
    auto* thermal=dynamic_cast<Pimc*>(thermal_pack.drivers[0].get());
    REQUIRE(thermal != nullptr); thermal->run();
    config.pimc[0].energy="centroid_virial";
    config.pimc[0].compare_thermodynamic=true;
    App centroid_app(config); auto centroid_pack=centroid_app.build_drivers();
    auto* centroid=dynamic_cast<Pimc*>(centroid_pack.drivers[0].get());
    REQUIRE(centroid != nullptr); centroid->run();
    CHECK(std::isfinite(centroid->energy()));
    CHECK(centroid->energy() == Approx(centroid->kinetic()+centroid->potential()).margin(2e-13));
    CHECK(centroid->export_paths().positions == thermal->export_paths().positions);
    CHECK(centroid->acceptance_ratio() == thermal->acceptance_ratio());
    CHECK(std::equal(centroid->block_potential().begin(),centroid->block_potential().end(),thermal->block_potential().begin()));
    // This short infrastructure check certifies no equilibration or physical limit.
}

TEST_CASE("Centroid statistical utility is measured on identical oscillator paths at several bead counts", "[centroid-utility]") {
    const std::vector groups{particle(0,{1.0,2.3})};
    Hamiltonian ham; ham.add_component(std::make_unique<HarmonicTrap>(groups,1.21,std::vector{.4,-.7}));
    std::cout << "M,seed,thermo_seconds,cv_seconds,thermo_variance,cv_variance,thermo_sem,cv_sem,paired_mean,paired_sem,thermo_reliable,cv_reliable,cv_energy,exact_energy\n";
    for (std::size_t M : {4,16,64}) for (int repeat=0; repeat<3; ++repeat) {
        PimcConfig cfg; cfg.samples=4; cfg.beta=2; cfg.time_step=2.0/M;
        cfg.equilibration_blocks=64; cfg.accumulation_blocks=512; cfg.steps_per_block=32;
        cfg.seed=20262000+1000*M+100*repeat;
        auto started=std::chrono::steady_clock::now();
        Pimc thermo(cfg,groups,ham,SimulationCell{},2,1); thermo.run();
        const double thermal_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        cfg.energy="centroid_virial";
        started=std::chrono::steady_clock::now();
        Pimc cv(cfg,groups,ham,SimulationCell{},2,1); cv.run();
        const double cv_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        REQUIRE(cv.export_paths().positions == thermo.export_paths().positions);
        std::vector<double> differences(cfg.accumulation_blocks);
        for (std::size_t b=0; b<differences.size(); ++b) differences[b]=cv.block_energies()[b]-thermo.block_energies()[b];
        const auto paired=numerics::reblock(differences);
        const double x=cfg.time_step*1.1/2;
        const double exact=1.1/(std::sqrt(1+x*x)*std::tanh(M*std::asinh(x)));
        REQUIRE(cv.energy_error()<.025);
        CHECK(std::abs(cv.energy()-exact)<4*cv.energy_error());
        CHECK(std::abs(paired.mean)<4*paired.error);
        auto variance=[](std::span<const double> blocks,double mean) {
            double sum=0; for (double x : blocks) sum+=(x-mean)*(x-mean);
            return sum/static_cast<double>(blocks.size()-1);
        };
        std::cout << std::setprecision(17) << M << ',' << cfg.seed << ',' << thermal_seconds << ',' << cv_seconds
            << ',' << variance(thermo.block_energies(),thermo.energy()) << ',' << variance(cv.block_energies(),cv.energy())
            << ',' << thermo.energy_error() << ',' << cv.energy_error() << ',' << paired.mean << ',' << paired.error
            << ',' << scalar(thermo.run_result(),"energy_error_reliable") << ',' << scalar(cv.run_result(),"energy_error_reliable")
            << ',' << cv.energy() << ',' << exact << '\n';
        // A pilot measurement, not a timing threshold or equilibration proof.
    }
}
