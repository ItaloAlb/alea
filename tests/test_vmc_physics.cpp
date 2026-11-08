// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Physics regression tests for the VMC estimator.
//
// Each case is a self-contained YAML config (tests/configs/*.yaml) describing a
// 2D or 3D electron-hole exciton with a linear Padé-Jastrow u(r) = c·r over the
// single e-h pair. For that trial wavefunction the variational energy is known
// in closed form, so we run the full VMC and compare against the analytic value.
//
// Single e-h pair, equal masses m, charges ∓1, Coulomb ε=1:
//   E_L(r) = -(m_inv·c + 1)/r - m_inv·c²,   m_inv = 1/m
//   ⟨1/r⟩ = -2c (2D),  -c (3D)   [from P(r) ∝ r^{dim-1} e^{2cr}]
//   ⟨E_L⟩ = c²/m + 2c   (2D)
//   ⟨E_L⟩ = c²/m +  c   (3D)
// The Coulomb cusp is satisfied when m_inv·c + 1 = 0, i.e. c = -m. There E_L is
// constant → zero variance and the VMC mean is exact to machine precision.
// Off-cusp the mean is still exact but the variance is large (formally infinite
// in 2D, finite in 3D), so those cases use a looser tolerance.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "hamiltonian/coulomb.h"
#include "hamiltonian/hamiltonian.h"
#include "hamiltonian/kinetic.h"
#include "io/parser.h"
#include "population/particle_group.h"
#include "population/walker.h"
#include "population/walker.h"
#include "population/walker_ensemble.h"
#include "qmc/app.h"
#include "qmc/run_result.h"
#include "qmc/vmc.h"
#include "population/simulation_cell.h"
#include "wavefunction/complex_orbital_set.h"
#include "wavefunction/pade_jastrow.h"
#include "wavefunction/trial_wavefunction.h"
#include "wavefunction/wavefunction_component.h"

namespace {

struct PhysicsCase {
    const char* config;    // file name under tests/configs/
    double      expected;  // analytic ⟨E_L⟩
    double      tol;        // absolute tolerance on the VMC energy
};

double run_energy(const std::string& config_name) {
    const std::string path = std::string(ALEA_CONFIG_DIR) + "/" + config_name;
    const alea::io::Config cfg = alea::io::parse_config_file(path);
    alea::App app(cfg);
    auto pack = app.build_drivers();
    REQUIRE(pack.drivers.size() >= 1);
    auto* vmc = dynamic_cast<alea::VMC*>(pack.drivers[0].get());
    REQUIRE(vmc != nullptr);
    vmc->run();
    return vmc->energy();
}

}  // namespace

void check(const PhysicsCase& c) {
    INFO("config: " << c.config << "  expected: " << c.expected);
    CHECK(run_energy(c.config) == Catch::Approx(c.expected).margin(c.tol));
}

// Cusp satisfied (2D: c = m·q_e·q_h/ε, 3D: c = m·q_e·q_h/(2ε)).
// E_L is constant → VMC mean is exact to machine precision.
// Analytic energies:
//   2D: <E_L> = -m·(q_e·q_h/ε)²
//   3D: <E_L> = -m·(q_e·q_h)²/(4ε²)
TEST_CASE("VMC is exact for the cusp-satisfying exciton ground state",
          "[vmc][physics][cusp]") {
    // ── Original cases: charge = ∓1, ε = 1 ──────────────────────────────
    SECTION("2D, m=1, c=-1")   check({"exciton_2d_cusp_m1.yaml",  -1.0, 1e-6});
    SECTION("2D, m=0.5, c=-0.5") check({"exciton_2d_cusp_m0.5.yaml", -0.5, 1e-6});
    SECTION("3D, m=1, c=-0.5") check({"exciton_3d_cusp_m1.yaml", -0.25, 1e-6});
    SECTION("3D, m=2, c=-1")   check({"exciton_3d_cusp_m2.yaml",  -0.5, 1e-6});

    // ── Charge variation: q_e ≠ -1 ─────────────────────────────────────
    // 2D, q_e=-2, q_h=+1, ε=1 → c = 1·(-2)/1 = -2, E = -1·(-2)² = -4.0.
    SECTION("2D, m=1, q_e=-2, q_h=+1, c=-2") {
        check({"exciton_2d_cusp_qe2.yaml", -4.0, 1e-6});
    }
    // 2D, q_e=-1, q_h=+3, ε=1 → c = 1·(-3)/1 = -3, E = -1·(-3)² = -9.0.
    SECTION("2D, m=1, q_e=-1, q_h=+3, c=-3") {
        check({"exciton_2d_cusp_qh3.yaml", -9.0, 1e-6});
    }
    // 3D, q_e=-2, q_h=+1, ε=1 → c = 1·(-2)/2 = -1, E = -1·4/4 = -1.0.
    SECTION("3D, m=1, q_e=-2, q_h=+1, c=-1") {
        check({"exciton_3d_cusp_qe2.yaml", -1.0, 1e-6});
    }

    // ── Epsilon (dielectric) variation ──────────────────────────────────
    // 2D, q_e=-1, q_h=+1, ε=2 → c = 1·(-1)/2 = -0.5, E = -1·(-0.5)² = -0.25.
    SECTION("2D, m=1, ε=2, c=-0.5") {
        check({"exciton_2d_cusp_eps2.yaml", -0.25, 1e-6});
    }
    // 3D, q_e=-1, q_h=+1, ε=2 → c = 1·(-1)/(2·2) = -0.25, E = -1·1/(4·4) = -0.0625.
    SECTION("3D, m=1, ε=2, c=-0.25") {
        check({"exciton_3d_cusp_eps2.yaml", -0.0625, 1e-6});
    }
}

// Cusp violated (c ≠ c_cusp): E_L fluctuates, so the variance is large (formally
// infinite in 2D, finite in 3D). The mean still converges to the analytic
// variational energy, checked here with a tolerance scaled to the fluctuations.
// Analytic:
//   2D: <E_L> = c²/m - 2c·q_e·q_h/ε
//   3D: <E_L> = c²/m - c·q_e·q_h/ε
//
// A config can no longer express a wrong cusp (the builder derives a₁ from the
// Hamiltonian), so these trials are assembled directly from the components.
namespace {

struct OffCuspCase {
    std::size_t dim;
    double      q_e, q_h;  // charges, ε = 1, both masses 1
    double      c;         // u(r) = c·r, deliberately off the Kato value
    double      expected;
    double      tol;
};

double run_off_cusp_vmc(const OffCuspCase& k) {
    using namespace alea;
    auto grp = [&](std::size_t first, double charge) {
        ParticleGroup g;
        g.name = first == 0 ? "e" : "h";
        g.charge = charge;
        g.first = first;
        g.last = first + 1;
        g.inv_mass.assign(k.dim, 1.0);
        return g;
    };
    const ParticleGroup ge = grp(0, k.q_e), gh = grp(1, k.q_h);

    TrialWavefunction psi;
    std::vector<Parameter> params = {{"a1", k.c, true}};
    psi.add_component(std::make_unique<PadeJastrow>(1, 0, ge, gh, std::move(params)));

    Hamiltonian ham;
    ham.add_component(std::make_unique<Kinetic>(std::vector<ParticleGroup>{ge, gh}));
    ham.add_component(std::make_unique<CoulombTerm>(ge, gh, /*epsilon=*/1.0));

    WalkerEnsemble ens(/*n_walkers=*/256, {ge, gh}, k.dim);
    std::vector<double> pos(2 * k.dim, 0.0);
    pos[k.dim] = 1.0;   // hole one unit along x
    for (auto& w : ens.walkers()) w.set_positions(pos);

    // The statistics the YAML versions of these cases ran with.
    VMCConfig cfg;
    cfg.tau = 0.05;
    cfg.n_equilibration_blocks = 40;
    cfg.n_accumulation_blocks = 120;
    cfg.n_steps = 30;
    cfg.n_substeps = 5;
    cfg.seed = 42;

    VMC vmc(cfg, ens, psi, ham);
    vmc.run();
    return vmc.energy();
}

}  // namespace

TEST_CASE("VMC reproduces the analytic energy for off-cusp exciton trials",
          "[vmc][physics]") {
    // clang-format off
    const OffCuspCase cases[] = {
        // dim  q_e   q_h    c      expected  tol
        {  2,  -1.0, +1.0,  -0.5,  -0.75,    0.03 },   // weak
        {  2,  -1.0, +1.0,  -2.0,   0.0,     0.05 },   // strong
        {  3,  -1.0, +1.0,  -1.0,   0.0,     0.02 },
        {  3,  -1.0, +1.0,  -0.25, -0.1875,  0.01 },   // weak
        {  2,  -2.0, +1.0,  -1.0,  -3.0,     0.08 },   // 1 − 2(−1)(−2)
        {  2,  -1.0, +3.0,  -1.0,  -5.0,     0.15 },   // 1 − 2(−1)(−3)
        {  3,  -2.0, +1.0,  -0.5,  -0.75,    0.02 },   // 0.25 − (−0.5)(−2)
    };
    // clang-format on
    for (const auto& k : cases) {
        DYNAMIC_SECTION(k.dim << "D, q_e=" << k.q_e << ", q_h=" << k.q_h
                              << ", c=" << k.c) {
            CHECK(run_off_cusp_vmc(k) == Catch::Approx(k.expected).margin(k.tol));
        }
    }
}

// Anisotropic-mass physics test: deterministic (tau=0) VMC with per-dimension
// inverse masses must recover the analytic local energy at the initial config.
TEST_CASE("VMC tau=0 with anisotropic masses recovers analytic local energy",
          "[vmc][physics][anisotropy]") {
    using namespace alea;

    // Helper: anisotropic particle group.
    auto agrp = [](std::size_t first, std::size_t last, double charge,
                   double inv_mx, double inv_my) {
        ParticleGroup g;
        g.name = "g";
        g.charge = charge;
        g.first = first;
        g.last = last;
        g.inv_mass = {inv_mx, inv_my};
        return g;
    };

    // e⁻: inv_mass={1.0, 0.5}, h⁺: inv_mass={1.0, 0.5}.
    auto g1 = agrp(0, 1, -1.0, /*inv_mx=*/1.0, /*inv_my=*/0.5);
    auto g2 = agrp(1, 2, +1.0, /*inv_mx=*/1.0, /*inv_my=*/0.5);

    // ψ = exp(-r)  (a=-1, b=0).
    TrialWavefunction psi;
    std::vector<Parameter> params = {{"a", -1.0, true}, {"b", 0.0, true}};
    psi.add_component(std::make_unique<PadeJastrow>(1, 1, g1, g2, std::move(params)));

    // H = K + V  (ε=1).
    Hamiltonian ham;
    ham.add_component(
        std::make_unique<Kinetic>(std::vector<ParticleGroup>{g1, g2}));
    ham.add_component(std::make_unique<CoulombTerm>(g1, g2, /*epsilon=*/1.0));

    // p0=(0,0), p1=(0,1): r=1, θ=π/2 (y-axis).
    WalkerEnsemble ens(/*n_walkers=*/1, {g1, g2}, /*dim=*/2);
    std::vector<double> pos = {0.0, 0.0, 0.0, 1.0};
    ens.walkers()[0].set_positions(pos);

    // Analytic E_L at this config, ψ=exp(-r), inv_mass={1.0,0.5}:
    //   ∂_x = 0, ∂_y = 1  →  (∂_x)²=0, (∂_y)²=1
    //   ∂²_x = −1, ∂²_y = 0
    //   K_e = −½[1.0·(−1+0) + 0.5·(0+1)] = 0.5 − 0.25 = 0.25
    //   K_h = 0.25
    //   K = 0.5,  V = (−1)(+1)/1 = −1.0
    //   E_L = −0.5
    const double expected = -0.5;

    VMCConfig cfg;
    cfg.tau = 0.0;
    cfg.n_equilibration_blocks = 2;
    cfg.n_accumulation_blocks = 3;
    cfg.n_steps = 4;
    cfg.n_substeps = 1;
    cfg.seed = 42;

    VMC vmc(cfg, ens, psi, ham);
    vmc.run();

    CHECK(vmc.energy() == Catch::Approx(expected));
    CHECK(vmc.acceptance_ratio() == Catch::Approx(1.0));
    CHECK(vmc.energy_error() == Catch::Approx(0.0).margin(1e-12));
    CHECK(vmc.energy_variance() == Catch::Approx(0.0).margin(1e-12));
    CHECK(vmc.block_energies().size() == 3);
}

// With τ > 0 the VMC samples |ψ|² = e^{-2r} and converges to the variational
// mean.  For ψ = exp(-r) and μ = {1.0, 0.5}:
//   ⟨E_L⟩ = (μ_x + μ_y)/2 − 2 = −1.25
TEST_CASE("VMC tau>0 anisotropic masses converges to variational energy",
          "[vmc][physics][anisotropy]") {
    using namespace alea;

    auto agrp = [](std::size_t first, std::size_t last, double charge,
                   double inv_mx, double inv_my) {
        ParticleGroup g;
        g.name = "g";
        g.charge = charge;
        g.first = first;
        g.last = last;
        g.inv_mass = {inv_mx, inv_my};
        return g;
    };

    auto g1 = agrp(0, 1, -1.0, /*inv_mx=*/1.0, /*inv_my=*/0.5);
    auto g2 = agrp(1, 2, +1.0, /*inv_mx=*/1.0, /*inv_my=*/0.5);

    // ψ = exp(-r)  (a=-1, b=0).
    TrialWavefunction psi;
    std::vector<Parameter> params = {{"a", -1.0, true}, {"b", 0.0, true}};
    psi.add_component(std::make_unique<PadeJastrow>(1, 1, g1, g2, std::move(params)));

    Hamiltonian ham;
    ham.add_component(
        std::make_unique<Kinetic>(std::vector<ParticleGroup>{g1, g2}));
    ham.add_component(std::make_unique<CoulombTerm>(g1, g2, /*epsilon=*/1.0));

    // 128 walkers initialised at the same config; walkers decorrelate via diffusion.
    WalkerEnsemble ens(/*n_walkers=*/128, {g1, g2}, /*dim=*/2);
    std::vector<double> pos = {0.0, 0.0, 2.0, 0.0};
    for (auto& w : ens.walkers()) w.set_positions(pos);

    // ⟨E_L⟩ = (μ_x + μ_y)/2 − 2 = 0.75 − 2 = −1.25
    const double expected = -1.25;

    VMCConfig cfg;
    cfg.tau = 0.05;
    cfg.n_equilibration_blocks = 40;
    cfg.n_accumulation_blocks = 120;
    cfg.n_steps = 30;
    cfg.n_substeps = 5;
    cfg.seed = 42;

    VMC vmc(cfg, ens, psi, ham);
    vmc.run();

    CHECK(vmc.energy() == Catch::Approx(expected).margin(0.05));
    CHECK(vmc.acceptance_ratio() > 0.0);
    CHECK(vmc.acceptance_ratio() <= 1.0);
    CHECK(vmc.energy_variance() > 0.0);
    CHECK(vmc.energy_error() > 0.0);
}

// ── Parameterised anisotropic physics tests ─────────────────────────────
//
// For a 2D e-h pair with anisotropic inverse masses (inv_m_e_d, inv_m_h_d),
// the reduced mass in direction d is μ_d = 1/(inv_m_e_d + inv_m_h_d).
// The exact ground-state binding energy is approximated by:
//
//   E ≈ −2 √(μ_x · μ_y)
//
// The trial wavefunction uses c = −√(μ_x·μ_y) (geometric mean).

namespace {

struct AnisoCase {
    double inv_mx_e, inv_my_e, inv_mx_h, inv_my_h;
    double tol;
};

double run_aniso_vmc(const AnisoCase& c) {
    using namespace alea;

    auto agrp = [](std::size_t first, std::size_t last, double charge,
                   double inv_mx, double inv_my) {
        ParticleGroup g;
        g.name = "g";
        g.charge = charge;
        g.first = first;
        g.last = last;
        g.inv_mass = {inv_mx, inv_my};
        return g;
    };

    auto g1 = agrp(0, 1, -1.0, c.inv_mx_e, c.inv_my_e);
    auto g2 = agrp(1, 2, +1.0, c.inv_mx_h, c.inv_my_h);

    const double mu_x = 1.0 / (c.inv_mx_e + c.inv_mx_h);
    const double mu_y = 1.0 / (c.inv_my_e + c.inv_my_h);
    const double trial_c = -2.0 * std::sqrt(mu_x * mu_y);

    TrialWavefunction psi;
    std::vector<Parameter> params = {{"a", trial_c, true}, {"b", 0.0, true}};
    psi.add_component(std::make_unique<PadeJastrow>(1, 1, g1, g2, std::move(params)));

    Hamiltonian ham;
    ham.add_component(
        std::make_unique<Kinetic>(std::vector<ParticleGroup>{g1, g2}));
    ham.add_component(std::make_unique<CoulombTerm>(g1, g2, /*epsilon=*/1.0));

    WalkerEnsemble ens(/*n_walkers=*/128, {g1, g2}, /*dim=*/2);
    std::vector<double> pos = {0.0, 0.0, 2.0, 0.0};
    for (auto& w : ens.walkers()) w.set_positions(pos);

    VMCConfig cfg;
    cfg.tau = 0.05;
    cfg.n_equilibration_blocks = 40;
    cfg.n_accumulation_blocks = 120;
    cfg.n_steps = 30;
    cfg.n_substeps = 5;
    cfg.seed = 42;

    VMC vmc(cfg, ens, psi, ham);
    vmc.run();
    return vmc.energy();
}

double aniso_expected(double inv_mx_e, double inv_my_e,
                      double inv_mx_h, double inv_my_h) {
    const double mu_x = 1.0 / (inv_mx_e + inv_mx_h);
    const double mu_y = 1.0 / (inv_my_e + inv_my_h);
    return -2.0 * std::sqrt(mu_x * mu_y);
}

}  // namespace

TEST_CASE("VMC tau>0 anisotropic masses match -2 sqrt(mu_x * mu_y)",
          "[vmc][physics][anisotropy]") {
    // clang-format off
    const AnisoCase cases[] = {
        //  e_x   e_y   h_x   h_y     tol
        {   1.0,  1.0,  1.0,  1.0,   0.05 },
        {   1.0,  0.5,  1.0,  0.5,   0.10 },
        {   0.5,  1.0,  0.5,  1.0,   0.10 },
        {   2.0,  1.0,  1.0,  3.0,   0.10 },
        {   0.3,  1.5,  0.8,  0.6,   0.10 },
        {   1.7,  0.4,  2.1,  0.9,   0.15 },
        {   1.0,  2.0,  0.5,  1.0,   0.10 },
    };
    // clang-format on

    for (const auto& c : cases) {
        const double expected = aniso_expected(c.inv_mx_e, c.inv_my_e,
                                                c.inv_mx_h, c.inv_my_h);
        DYNAMIC_SECTION("e=(" << c.inv_mx_e << "," << c.inv_my_e << ") "
                       << "h=(" << c.inv_mx_h << "," << c.inv_my_h << ") "
                       << "expected=" << expected) {
            const double e = run_aniso_vmc(c);
            CHECK(e == Catch::Approx(expected).margin(c.tol));
        }
    }
}

// ── Helium atom: 2 electrons + heavy nucleus (Z=2) ─────────────────────────
//
// The trial wavefunction uses two Padé-Jastrow factors with Kato cusp
// parameters fixed:
//   e-n:  a₁ = -2  (cusp in 3D with μ≈1, q_e=-1, q_n=+2)
//   e-e:  a₁ = +0.5, b₁ = 0.5  (cusp + short-range saturation)
//
// For this two-parameter Jastrow the VMC variational energy is ≈ -2.90 a.u.
// The exact non-relativistic ground-state energy of helium is -2.904 a.u. [https://arxiv.org/pdf/quant-ph/0203150]
TEST_CASE("VMC estimates helium-3D ground-state energy near exact value",
          "[vmc][physics][helium]") {
    check({"helium_3d.yaml", -2.90, 0.05});
}

// ── Helium 2D: 2 electrons + heavy nucleus (Z=2) ───────────────────────────
//
// Kato cusp in 2D (d-1 = 1):
//   e-n:  a₁ = 2·μ·q_e·q_n / (ε·1) = 2·1·(-1)(+2) / 1 = -4
//   e-e:  a₁ = 2·μ·q_e²    / (ε·1) = 2·0.5·1     / 1 = +1.0
//
// For this Kato-satisfying Jastrow the VMC variational energy is ≈ -11.90 a.u.
// Exact non-relativistic ground state: -11.90 a.u. (quant-ph/0203150).
TEST_CASE("VMC estimates helium-2D ground-state energy near exact value",
          "[vmc][physics][helium]") {
    check({"helium_2d.yaml", -11.90, 0.05});
}

// ── g(r) pair-correlation physics ──────────────────────────────────────────
//
// Exact e-h ground state ψ = exp(-r): the pair correlation is analytic,
//   g_eh(r) ∝ e^{-2r}  → ln g_eh(r) is linear in r with slope -2,
//   P(r) ∝ g_eh(r)·2πr ∝ r e^{-2r}  → ⟨r⟩ = 1 (2D).
// Both VMC and the DMC mixed estimator (ψ exact → weights ≈ 1) reproduce this.
namespace {

alea::RunResult run_driver_result(const std::string& config_name) {
    const std::string path = std::string(ALEA_CONFIG_DIR) + "/" + config_name;
    const alea::io::Config cfg = alea::io::parse_config_file(path);
    alea::App app(cfg);
    auto pack = app.build_drivers();
    REQUIRE(pack.drivers.size() >= 1);
    pack.drivers[0]->run();
    return pack.drivers[0]->run_result();
}

const std::vector<double>& get_series(const alea::RunResult& r,
                                      const std::string& name) {
    for (const auto& [n, v] : r.series)
        if (n == name) return v;
    FAIL("series not found: " << name);
    static const std::vector<double> empty;
    return empty;
}

// Least-squares slope of ln(g(r)) vs r over [rmin, rmax], using positive bins.
double loglinear_slope(const std::vector<double>& r, const std::vector<double>& g,
                       double rmin, double rmax) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int n = 0;
    for (std::size_t i = 0; i < r.size(); ++i) {
        if (r[i] < rmin || r[i] > rmax || g[i] <= 0.0) continue;
        const double x = r[i], y = std::log(g[i]);
        sx += x; sy += y; sxx += x * x; sxy += x * y;
        ++n;
    }
    REQUIRE(n >= 3);
    return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

// ⟨r⟩ from g(r): P(bin) ∝ g(bin)·(2D annulus area).
double mean_separation_2d(const std::vector<double>& r,
                          const std::vector<double>& g) {
    constexpr double pi = 3.14159265358979323846;
    const double dr = r[1] - r[0];
    double num = 0.0, den = 0.0;
    for (std::size_t b = 0; b < r.size(); ++b) {
        const double r_lo  = r[b] - 0.5 * dr;
        const double r_hi  = r[b] + 0.5 * dr;
        const double shell = pi * (r_hi * r_hi - r_lo * r_lo);
        const double p     = g[b] * shell;
        num += r[b] * p;
        den += p;
    }
    return num / den;
}

void check_exciton_gofr(const std::string& config) {
    INFO("config: " << config);
    const alea::RunResult res = run_driver_result(config);
    const std::vector<double>& r = get_series(res, "gofr_r");
    const std::vector<double>& g = get_series(res, "gofr_electron_group_hole_group");
    REQUIRE(r.size() == g.size());

    // ln g_eh(r) ∝ -2r over the well-sampled region.
    CHECK(loglinear_slope(r, g, 0.3, 2.5) == Catch::Approx(-2.0).margin(0.2));
    // ⟨r⟩ = 1 in 2D.
    CHECK(mean_separation_2d(r, g) == Catch::Approx(1.0).margin(0.1));
}

}  // namespace

TEST_CASE("VMC g(r) reproduces the analytic exciton pair correlation",
          "[vmc][physics][gofr]") {
    check_exciton_gofr("exciton_2d_gofr.yaml");
}

TEST_CASE("DMC mixed g(r) reproduces the analytic exciton pair correlation",
          "[dmc][physics][gofr]") {
    check_exciton_gofr("exciton_2d_gofr_dmc.yaml");
}

// ── Periodic boundaries: the ideal gas ───────────────────────────────────────
// A non-interacting gas in a periodic cell (ψ = 1, kinetic only) samples the
// cell uniformly, so its pair correlation is flat: g(r) = 1 out to the
// Wigner–Seitz inradius.  That single number exercises the whole periodic
// stack at once:
//   • minimum-image distances — an unfolded r would starve the small-r bins and
//     pile counts up near the box diagonal
//   • position wrapping — walkers must stay in the cell
//   • the ρ = N/V normalisation — without the cell-volume factor g(r) would sit
//     at some arbitrary constant instead of 1
TEST_CASE("periodic ideal gas has a flat g(r) = 1", "[vmc][physics][gofr][cell]") {
    const alea::RunResult res = run_driver_result("periodic_2d_gas.yaml");
    const std::vector<double>& r = get_series(res, "gofr_r");
    const std::vector<double>& g =
        get_series(res, "gofr_electron_group_electron_group");
    REQUIRE(r.size() == g.size());
    REQUIRE(g.size() > 10);

    // Skip the first couple of bins: they hold very few pairs and their
    // relative error is large.
    double sum = 0.0, worst = 0.0;
    std::size_t n = 0;
    for (std::size_t i = 2; i < g.size(); ++i) {
        sum += g[i];
        worst = std::max(worst, std::abs(g[i] - 1.0));
        ++n;
    }
    INFO("mean g = " << sum / static_cast<double>(n) << ", worst |g-1| = " << worst);
    CHECK(sum / static_cast<double>(n) == Catch::Approx(1.0).margin(0.02));
    CHECK(worst < 0.15);
}

// The same run's one-body density is flat at rho = N/A.  With rs = 1 the area
// is A = N pi rs^2, so rho = 1/pi exactly, independent of N — a closed-form
// target for the periodic binning path (fractional coordinates over the cell)
// and its V_cell/n_bins^dim normalisation.
// ── Periodic boundaries: large-cell convergence to the open-boundary limit ───
// The same bound exciton is run in three cells that differ ONLY in the geometry:
// open, a large periodic cell, and a deliberately cramped one.
//
// Two separate things are being checked, and they are sensitive to different
// failure modes:
//
//   ENERGY. At the Coulomb cusp E_L is constant, so <E_L> = -1 exactly with zero
//   variance — in ANY cell, because the cancellation between the -1/r of the
//   potential and the +1/r of the kinetic term is r-independent. That makes the
//   energy useless as a convergence probe but excellent as a consistency probe:
//   the cancellation only survives if the folded r that Coulomb reads and the
//   folded direction that the Jastrow gradient reads are the same fold. A
//   mismatch destroys it pointwise and the variance stops being zero.
//
//   MEAN SEPARATION. <r> from g(r) cross-checks the periodic path against the
//   well-tested open one on a real bound state. <r> is a normalised ratio, so
//   the extra rho = N/V factor the periodic g(r) carries cancels and the three
//   runs stay comparable.
//
// Verified to have teeth by mutation: breaking the fold in the INCREMENTAL path
// (ratio_batch / apply_move_batch) sends both periodic runs to E ~ -300 with
// enormous variance while the open control stays exactly -1. Note it must be the
// incremental path — evaluate_batch runs once at startup and its gradients are
// immediately overwritten by accepted moves, so a bug confined to it washes out
// of a physics run entirely. That is why the direct-evaluation tests in
// test_periodic_wavefunction.cpp and these run-level tests are both needed:
// they cover different halves of the wavefunction protocol.
TEST_CASE("a large periodic cell reproduces the open-boundary exciton",
          "[vmc][physics][cell]") {
    auto scalar = [](const alea::RunResult& r, const std::string& name) {
        for (const auto& [n, v] : r.scalars)
            if (n == name) return v;
        FAIL("scalar not found: " << name);
        return 0.0;
    };
    auto run = [&](const char* cfg) {
        const alea::RunResult res = run_driver_result(cfg);
        const std::vector<double>& rr = get_series(res, "gofr_r");
        const std::vector<double>& g =
            get_series(res, "gofr_electron_group_hole_group");
        return std::tuple<double, double, double>{
            scalar(res, "energy"), scalar(res, "energy_variance"),
            mean_separation_2d(rr, g)};
    };

    const auto [e_open, v_open, r_open]    = run("exciton_2d_cell_open.yaml");
    const auto [e_large, v_large, r_large] = run("exciton_2d_cell_large.yaml");
    const auto [e_small, v_small, r_small] = run("exciton_2d_cell_small.yaml");

    INFO("open  E=" << e_open  << " var=" << v_open  << " <r>=" << r_open);
    INFO("large E=" << e_large << " var=" << v_large << " <r>=" << r_large);
    INFO("small E=" << e_small << " var=" << v_small << " <r>=" << r_small);

    // Cusp cancellation survives the fold: exact energy, zero variance.
    CHECK(e_open == Catch::Approx(-1.0).margin(1e-12));
    CHECK(e_large == Catch::Approx(-1.0).margin(1e-12));
    CHECK(e_small == Catch::Approx(-1.0).margin(1e-12));
    CHECK(std::abs(v_large) < 1e-12);
    CHECK(std::abs(v_small) < 1e-12);

    // Convergence: the periodic runs reproduce the open-boundary separation.
    //
    // The cramped cell agrees too, and that is not a weak tolerance — it is the
    // physics.  Inside the Wigner-Seitz cell psi and V are the SAME functions of
    // r as in the open system, so every observable restricted to r < inradius is
    // cell-independent under minimum-image truncation.  A single bound pair
    // therefore has no finite-size signal to find here.  Genuine E(L) convergence
    // needs the image interactions that Ewald restores, and belongs to that
    // milestone's tests, not this one.
    CHECK(r_large == Catch::Approx(r_open).margin(0.02));
    CHECK(r_small == Catch::Approx(r_open).margin(0.02));
}

// ── Periodic boundaries: the shared-scratch race detector ────────────────────
// DMC shares one TrialWavefunction and one WalkerSet across crowds, and the
// periodic fold keeps per-pair scratch in that shared state.  It is race-free
// only because concurrent crowds write disjoint [w_begin, w_end) slices — an
// argument, not a proof, and a violation would be intermittent and brutal to
// diagnose from a physics number.
//
// DMC's single OpenMP construct writes per-thread results into rs[t] and
// combines them sequentially, so there is no reduction-order freedom: two runs
// at the same seed and thread count must agree bit for bit.  The final walker
// positions are the sharp signal here — averaging into an energy could hide a
// small corruption that the raw population state cannot.
TEST_CASE("periodic DMC is bit-reproducible across runs (shared fold scratch)",
          "[dmc][cell][determinism]") {
    struct Snapshot {
        double              energy;
        std::size_t         nw;
        std::vector<double> positions;
    };

    auto snapshot = [](const std::string& config_name) {
        const std::string path = std::string(ALEA_CONFIG_DIR) + "/" + config_name;
        const alea::io::Config cfg = alea::io::parse_config_file(path);
        alea::App app(cfg);
        auto pack = app.build_drivers();
        REQUIRE(pack.drivers.size() >= 1);
        pack.drivers[0]->run();
        const alea::WalkerSet* fin = pack.drivers[0]->final_population();
        REQUIRE(fin != nullptr);
        Snapshot s;
        const alea::RunResult rr = pack.drivers[0]->run_result();
        s.energy = 0.0;
        bool found = false;
        for (const auto& [name, value] : rr.scalars)
            if (name == "energy") { s.energy = value; found = true; }
        REQUIRE(found);
        s.nw     = fin->num_walkers();
        const std::span<const double> p = fin->positions();
        s.positions.assign(p.begin(), p.end());
        return s;
    };

    // The minimum-image fold paths, and the Ewald paths (EwaldTerm ladders,
    // image-summed Jastrow ρ(k) caches and phase scratch).
    for (const char* config :
         {"periodic_dmc_determinism.yaml", "periodic_dmc_determinism_ewald.yaml"}) {
        INFO("config " << config);
        const Snapshot a = snapshot(config);
        const Snapshot b = snapshot(config);

        // A NaN energy would make every comparison below vacuously true.
        REQUIRE(std::isfinite(a.energy));

        CHECK(a.nw == b.nw);            // branching itself must be reproducible
        CHECK(a.energy == b.energy);    // exact equality, not Approx
        REQUIRE(a.positions.size() == b.positions.size());

        std::size_t differing = 0;
        for (std::size_t k = 0; k < a.positions.size(); ++k)
            if (a.positions[k] != b.positions[k]) ++differing;
        INFO("energy " << a.energy << " vs " << b.energy << "; " << differing << " of "
                       << a.positions.size() << " coordinates differ");
        CHECK(differing == 0);
    }
}

TEST_CASE("periodic ideal gas has S(k) = 1 and no Bragg peaks",
          "[vmc][physics][sofk][cell]") {
    // Uncorrelated particles: ⟨|ρ_k|²⟩ = N + N(N−1)|⟨e^{ik·r}⟩|², and the
    // uniform average of e^{ik·r} over the cell vanishes for every reciprocal
    // vector k ≠ 0, so S(k) = 1 on every shell and ⟨ρ_k⟩ = 0 on every vector.
    const alea::RunResult res = run_driver_result("periodic_2d_gas.yaml");
    const std::vector<double>& k   = get_series(res, "sofk_k");
    const std::vector<double>& s   = get_series(res, "sofk_electron_group_electron_group");
    const std::vector<double>& se  = get_series(res, "sofk_electron_group_electron_group_error");
    REQUIRE(k.size() >= 3);
    for (std::size_t i = 0; i < k.size(); ++i) {
        INFO("|k| = " << k[i] << "  S = " << s[i] << " +/- " << se[i]);
        CHECK(std::abs(s[i] - 1.0) < 4.0 * se[i] + 1e-3);
    }
    const std::vector<double>& re  = get_series(res, "rhok_electron_group_re");
    const std::vector<double>& ree = get_series(res, "rhok_electron_group_re_error");
    const std::vector<double>& im  = get_series(res, "rhok_electron_group_im");
    const std::vector<double>& ime = get_series(res, "rhok_electron_group_im_error");
    for (std::size_t q = 0; q < re.size(); ++q) {
        INFO("vector " << q);
        CHECK(std::abs(re[q]) < 4.0 * ree[q] + 1e-3);
        CHECK(std::abs(im[q]) < 4.0 * ime[q] + 1e-3);
    }
}

TEST_CASE("periodic ideal gas has a flat density rho = 1/pi at rs = 1",
          "[vmc][physics][density][cell]") {
    const alea::RunResult res = run_driver_result("periodic_2d_gas.yaml");
    const std::vector<double>& rho = get_series(res, "density_electron_group");
    REQUIRE(rho.size() == 12 * 12);  // n_bins^dim, binned over the cell

    // Fractional edges, not cartesian ones: the periodic grid IS the cell.
    const std::vector<double>& edges = get_series(res, "density_edges");
    REQUIRE(edges.size() == 13);
    CHECK(edges.front() == Catch::Approx(-0.5));
    CHECK(edges.back() == Catch::Approx(0.5));

    const double expected = 1.0 / std::acos(-1.0);
    double sum = 0.0, worst = 0.0;
    for (double v : rho) {
        sum += v;
        worst = std::max(worst, std::abs(v - expected));
    }
    const double mean = sum / static_cast<double>(rho.size());
    INFO("mean rho = " << mean << " (expected " << expected
                       << "), worst |rho - 1/pi| = " << worst);
    CHECK(mean == Catch::Approx(expected).margin(1e-6));
    CHECK(worst < 0.05 * expected);
}

// ── DMC: the Drummond (DTN) cusp Jastrow stabilizes DMC like the Padé form ────
//
// 2D Coulomb e-h exciton, m=1: the exact ground state is ψ = exp(-r), E = -1 Ha.
// The Drummond u-term's cusp Γ is auto-derived from the Coulomb residue (= -1),
// so the e–h Kato cusp is satisfied, the DMC local energy is finite at
// coalescence, and DMC projects the nodeless ground state to -1 — exactly as the
// cusp-correct Padé does.  The finite-range trial is not the exact eigenstate, so
// it keeps a small variance / time-step bias (margin), unlike Padé u = -r.
namespace {

double dmc_energy_ha(const std::string& config) {
    const alea::RunResult res = run_driver_result(config);
    for (const auto& [name, value] : res.scalars)
        if (name == "energy") return value;
    FAIL("energy scalar not found in run result");
    return 0.0;
}

}  // namespace

TEST_CASE("DMC projects a Drummond-cusp exciton trial to the exact ground state",
          "[dmc][physics][drummond]") {
    CHECK(dmc_energy_ha("exciton_2d_drummond_dmc.yaml") ==
          Catch::Approx(-1.0).margin(0.02));
}

TEST_CASE("DMC with a one-pair pairing determinant gives the exact bilayer exciton",
          "[dmc][physics][pairing]") {
    // psi = phi(r_e - r_h) e^{u}, nodeless, so DMC is exact up to time step
    // and statistics.  Exact: -0.5065190623 Ha* (tests/references/pair_bound_state.py,
    // d = 0.4, mu = 1/2), against López Ríos et al.'s VMC -0.5065191(2).
    CHECK(dmc_energy_ha("bilayer_exciton_pairing_dmc.yaml") ==
          Catch::Approx(-0.5065190623).margin(0.005));
}

// The same exciton with a backflow pairing determinant (e displaced by h, h
// by e): psi stays positive, so DMC stays exact.  Over 16 seeds (2026-11-08)
// this run gives DMC - exact = +1.02(50)e-3 with standard deviation 2.0e-3
// (printed ±~1.7e-3); the plain determinant of the same run length gives
// +0.71(42)e-3 and 1.7e-3, so the backflow adds no bias at the 6e-4 level and
// the common offset is the time step's.  Margin = |mean| + 4 deviations.
TEST_CASE("DMC with a backflow pairing determinant gives the exact bilayer exciton",
          "[dmc][physics][pairing][backflow]") {
    CHECK(dmc_energy_ha("bilayer_exciton_pairing_backflow_dmc.yaml") ==
          Catch::Approx(-0.5065190623).margin(9e-3));
}

// ── DMC: monolayer TMD exciton binding energies (Rytova–Keldysh) ─────────────
//
// Validates the whole physics chain (Hartree atomic units, the Rytova–Keldysh
// potential, DMC) against published DMC exciton binding energies for suspended
// TMD monolayers.  Two free particles (electron + hole, masses in m0) interact
// via Rytova–Keldysh with 2D screening length r0 (Å → Bohr, ε = 1).  The two-body
// ground-state energy E is the relative-motion bound state; the binding energy is
// −E (→ eV).  Reproduced to ~1% — the residual is DMC time-step bias (τ = 0.05)
// plus statistics.  Reference values supplied with the test data.
namespace {

double dmc_binding_eV(const std::string& config) {
    constexpr double kHartreeToEV = 27.211386;
    const alea::RunResult res = run_driver_result(config);
    for (const auto& [name, value] : res.scalars)
        if (name == "energy") return -value * kHartreeToEV;
    FAIL("energy scalar not found in run result");
    return 0.0;
}

// |observed − published| tolerance (eV): comfortably covers the ~1% DMC
// time-step/statistical residual, yet still catches a units/potential regression
// (which would miss by ≫ 0.1 eV).
constexpr double kBindingTol = 0.025;

}  // namespace

TEST_CASE("DMC reproduces the MoS2 exciton binding energy (Rytova-Keldysh)",
          "[dmc][keldysh][tmd]") {
    const double eb = dmc_binding_eV("keldysh_mos2_dmc.yaml");
    INFO("MoS2 binding = " << eb << " eV (published 0.5514)");
    CHECK(eb == Catch::Approx(0.5514).margin(kBindingTol));
}

TEST_CASE("DMC reproduces the MoSe2 exciton binding energy (Rytova-Keldysh)",
          "[dmc][keldysh][tmd]") {
    const double eb = dmc_binding_eV("keldysh_mose2_dmc.yaml");
    INFO("MoSe2 binding = " << eb << " eV (published 0.4778)");
    CHECK(eb == Catch::Approx(0.4778).margin(kBindingTol));
}

TEST_CASE("DMC reproduces the WS2 exciton binding energy (Rytova-Keldysh)",
          "[dmc][keldysh][tmd]") {
    const double eb = dmc_binding_eV("keldysh_ws2_dmc.yaml");
    INFO("WS2 binding = " << eb << " eV (published 0.5191)");
    CHECK(eb == Catch::Approx(0.5191).margin(kBindingTol));
}

TEST_CASE("DMC reproduces the WSe2 exciton binding energy (Rytova-Keldysh)",
          "[dmc][keldysh][tmd]") {
    const double eb = dmc_binding_eV("keldysh_wse2_dmc.yaml");
    INFO("WSe2 binding = " << eb << " eV (published 0.4667)");
    CHECK(eb == Catch::Approx(0.4667).margin(kBindingTol));
}

// ── DMC: monolayer TMD negative-trion (X⁻) binding energies ──────────────────
//
// The trion X⁻ = two electrons + one hole.  With the electron spins opposite
// (singlet) the spatial ground state is nodeless, so DMC is fixed-node-free and
// projects the exact ground state — just like the exciton.  The trion *binding*
// energy is the energy released when a neutral exciton captures a second
// electron,
//
//     E_b^trion = E(exciton) − E(trion)          (both total energies < 0),
//
// a small (~30 meV) splitting between two ~500 meV DMC energies.  Two things are
// essential to recover it reliably:
//   (1) Both energies use *optimized* (linear-method) Padé trials of comparable
//       quality.  The exciton-tuned exp(c·r) is a poor trion trial (E_var ≈ 0,
//       huge variance) AND is itself under-converged for the small-r₀ sulfides;
//       differencing trials of unequal quality leaves an uncancelled time-step
//       bias of 10+ meV — larger than the signal.  Each trion/exciton config
//       therefore carries parameters baked in from an offline optimization.
//   (2) Both run at the *same* τ and statistics, so the residual τ-bias and the
//       small model/implementation offset (a few meV on the absolute energies,
//       vs. the paper) cancel in the difference.
// The matched-quality difference reproduces the published values to ~1 meV.
// Reference (fixed-node-free DMC): Mayers, Berkelbach, Hybertsen & Reichman,
// PRB 92, 161404(R) (2015) [DOI: 10.1103/PhysRevB.92.161404].
namespace {

// E(exciton) − E(trion) in eV from two DMC runs with matched optimized trials.
// dmc_binding_eV returns −E·27.211386, so (−E_trion) − (−E_exciton) = E_X − E_T,
// which is exactly the (positive) trion binding energy.
double dmc_trion_binding_eV(const std::string& exciton_ref_cfg,
                            const std::string& trion_cfg) {
    return dmc_binding_eV(trion_cfg) - dmc_binding_eV(exciton_ref_cfg);
}

// The margin is set by the run-to-run noise of these fixed-length runs, not by
// the printed error. With 2048 accumulation blocks, over 256 seeds
// (2026-10-20) the binding's standard deviation is 1.35 / 1.30 / 1.29 /
// 1.33 meV (MoS2 / MoSe2 / WS2 / WSe2), the seed means are within 0.4 meV of
// the published values and no seed is more than 4.2 meV off, so 10 meV is
// about 7.5 standard deviations and any valid random stream passes. The
// earlier 512 blocks gave 2.4-2.8 meV with heavy tails: 1 of 1024 runs was
// 14.6 meV off, and a fixed seed failed when the random streams changed.
// The margin still firmly catches a regression: a missing e-e term, wrong r₀
// or a units error shifts the binding by ≫ 10 meV (and a sign error makes it
// negative). Lengthening or shortening the runs changes the noise.
constexpr double kTrionBindingTol = 0.010;  // eV

}  // namespace

TEST_CASE("DMC reproduces the MoS2 trion binding energy (Rytova-Keldysh)",
          "[dmc][keldysh][tmd][trion]") {
    const double eb = dmc_trion_binding_eV("keldysh_mos2_exciton_ref_dmc.yaml",
                                           "keldysh_mos2_trion_dmc.yaml");
    INFO("MoS2 trion binding = " << eb << " eV (published 0.0338)");
    CHECK(eb == Catch::Approx(0.0338).margin(kTrionBindingTol));
}

TEST_CASE("DMC reproduces the MoSe2 trion binding energy (Rytova-Keldysh)",
          "[dmc][keldysh][tmd][trion]") {
    const double eb = dmc_trion_binding_eV("keldysh_mose2_exciton_ref_dmc.yaml",
                                           "keldysh_mose2_trion_dmc.yaml");
    INFO("MoSe2 trion binding = " << eb << " eV (published 0.0284)");
    CHECK(eb == Catch::Approx(0.0284).margin(kTrionBindingTol));
}

TEST_CASE("DMC reproduces the WS2 trion binding energy (Rytova-Keldysh)",
          "[dmc][keldysh][tmd][trion]") {
    const double eb = dmc_trion_binding_eV("keldysh_ws2_exciton_ref_dmc.yaml",
                                           "keldysh_ws2_trion_dmc.yaml");
    INFO("WS2 trion binding = " << eb << " eV (published 0.0340)");
    CHECK(eb == Catch::Approx(0.0340).margin(kTrionBindingTol));
}

TEST_CASE("DMC reproduces the WSe2 trion binding energy (Rytova-Keldysh)",
          "[dmc][keldysh][tmd][trion]") {
    const double eb = dmc_trion_binding_eV("keldysh_wse2_exciton_ref_dmc.yaml",
                                           "keldysh_wse2_trion_dmc.yaml");
    INFO("WSe2 trion binding = " << eb << " eV (published 0.0295)");
    CHECK(eb == Catch::Approx(0.0295).margin(kTrionBindingTol));
}

// ── DMC: monolayer TMD biexciton (XX) binding energies ───────────────────────
//
// The biexciton XX = 2 electrons + 2 holes, all spins paired (e↑e↓, h↑h↓), so the
// spatial ground state is nodeless and DMC is fixed-node-free — like the exciton
// and the singlet trion.  The biexciton *binding* energy is the XX → X + X
// dissociation energy,
//
//     E_b(XX) = 2·E(exciton) − E(biexciton)          (both total energies < 0),
//
// a small (~20 meV) splitting BELOW the two-exciton threshold.  Because XX sits
// only ~20 meV under that threshold, in the open (unconfined) geometry a plain
// pairwise trial lets DMC dissociate into two free excitons.  Two ingredients
// recover the binding (thesis "Application of QMC to Excitonic and Electronic
// Systems", Eqs. 3.4–3.6, p.45-46):
//   (1) a LOCALIZING e-h trial — a [2/1] Padé with a NEGATIVE r² numerator term
//       (a2<0), so u_eh(r) → (a2/b1)·r → −∞ and ψ decays at large e-h separation,
//       holding the four particles together so DMC projects the bound state.  The
//       parameters are FIXED, not variance/energy-optimized: optimization drives
//       the trial to the dissociated two-exciton minimum (thesis p.46); the bound
//       ground state instead emerges under DMC ("fix and project").
//   (2) a large walker population (2048) — the population-control bias of the
//       four-body branching is severe at small populations.
// The matched exciton reference uses the SAME localizing e-h Padé at identical τ
// and statistics, so residual systematics cancel in the difference.
// Reference (fixed-node-free DMC): Mayers, Berkelbach, Hybertsen & Reichman,
// PRB 92, 161404(R) (2015).
namespace {

// 2·E(exciton) − E(biexciton) in eV.  dmc_binding_eV returns −E·27.211386, so
// dmc_binding_eV(xx) − 2·dmc_binding_eV(exciton) = 27.211386·(2·E_X − E_XX),
// which is exactly the (positive) biexciton binding energy.
double dmc_biexciton_binding_eV(const std::string& exciton_ref_cfg,
                                const std::string& biexciton_cfg) {
    return dmc_binding_eV(biexciton_cfg) - 2.0 * dmc_binding_eV(exciton_ref_cfg);
}

// Wider margin than the trion test: the four-body binding is a ~20 meV signal
// recovered with a fixed (non-optimized) localizing trial run at threads:0
// (multi-thread → not bit-reproducible), so the per-run scatter is a few meV.  7
// meV still firmly catches a regression — a non-localizing trial collapses toward
// the dissociation threshold (binding → 0), and a units / r0 / missing-term error
// misses by ≫ 7 meV.
constexpr double kBiexcitonBindingTol = 0.007;  // eV

}  // namespace

TEST_CASE("DMC reproduces the MoS2 biexciton binding energy (Rytova-Keldysh)",
          "[dmc][keldysh][tmd][biexciton]") {
    const double eb = dmc_biexciton_binding_eV(
        "keldysh_mos2_biexciton_excitonref_dmc.yaml",
        "keldysh_mos2_biexciton_dmc.yaml");
    INFO("MoS2 biexciton binding = " << eb << " eV (published 0.0227)");
    CHECK(eb == Catch::Approx(0.0227).margin(kBiexcitonBindingTol));
}

TEST_CASE("DMC reproduces the MoSe2 biexciton binding energy (Rytova-Keldysh)",
          "[dmc][keldysh][tmd][biexciton]") {
    const double eb = dmc_biexciton_binding_eV(
        "keldysh_mose2_biexciton_excitonref_dmc.yaml",
        "keldysh_mose2_biexciton_dmc.yaml");
    INFO("MoSe2 biexciton binding = " << eb << " eV (published 0.0177)");
    CHECK(eb == Catch::Approx(0.0177).margin(kBiexcitonBindingTol));
}

TEST_CASE("DMC reproduces the WS2 biexciton binding energy (Rytova-Keldysh)",
          "[dmc][keldysh][tmd][biexciton]") {
    const double eb = dmc_biexciton_binding_eV(
        "keldysh_ws2_biexciton_excitonref_dmc.yaml",
        "keldysh_ws2_biexciton_dmc.yaml");
    INFO("WS2 biexciton binding = " << eb << " eV (published 0.0233)");
    CHECK(eb == Catch::Approx(0.0233).margin(kBiexcitonBindingTol));
}

TEST_CASE("DMC reproduces the WSe2 biexciton binding energy (Rytova-Keldysh)",
          "[dmc][keldysh][tmd][biexciton]") {
    const double eb = dmc_biexciton_binding_eV(
        "keldysh_wse2_biexciton_excitonref_dmc.yaml",
        "keldysh_wse2_biexciton_dmc.yaml");
    INFO("WSe2 biexciton binding = " << eb << " eV (published 0.0200)");
    CHECK(eb == Catch::Approx(0.0200).margin(kBiexcitonBindingTol));
}

// ── Periodic fermions: the two phases of the Wigner-crystallisation crossing ──
//
// Both trial functions have an analytic non-interacting limit, which is what
// makes them checkable at all: the interacting energies they are built for have
// no closed form.  Together they cover the whole periodic-fermion stack.
TEST_CASE("VMC is exact for a closed-shell plane-wave determinant",
          "[vmc][physics][periodic][plane_wave]") {
    // ψ = det{cos/sin(k_i·r_j)} with no interaction is an EXACT eigenstate: each
    // orbital has ∇²φ = −|k|²φ, so E_L is the same constant everywhere and VMC
    // returns it with zero variance.  N = 7 fills k = 0 plus the first shell
    // (6 vectors of |k|² = 16π²/3a²), so with a² = 2Nπrs²/√3,
    //     E = 16π²/a² = 8π√3/(N rs²) = 6.2187407  for N = 7, rs = 1.
    const double kPi = 3.14159265358979323846;
    const double expected = 8.0 * kPi * std::sqrt(3.0) / 7.0;
    check({"periodic_2d_pw_free.yaml", expected, 1e-8});
}

TEST_CASE("VMC is exact for a non-overlapping crystal determinant",
          "[vmc][physics][periodic][crystal]") {
    // One narrow Gaussian per commensurate triangular site.  With α d² ≫ 1 the
    // determinant is diagonal and each particle is an independent 2-D Gaussian
    // with ⟨E_L⟩ = α, so E = N·α = 7·8 = 56.  The variance is not zero here
    // (Var = N α² = 448), hence the tolerance: it is 3.5 σ of the run's own mean.
    check({"periodic_2d_crystal_free.yaml", 56.0, 0.5});
}

// ── Twisted boundary conditions (fixed phase) ──────────────────────────────
namespace {
double scalar_of(const alea::RunResult& r, const std::string& name) {
    for (const auto& [n, v] : r.scalars)
        if (n == name) return v;
    FAIL("scalar not found: " << name);
    return 0.0;
}

alea::RunResult run_twisted(std::size_t driver, std::vector<double> twist,
                            const std::string& config = "heg2d_twisted.yaml") {
    const std::string path = std::string(ALEA_CONFIG_DIR) + "/" + config;
    alea::io::Config cfg = alea::io::parse_config_file(path);
    cfg.lattice.twist = std::move(twist);
    alea::App app(cfg);
    auto pack = app.build_drivers();
    REQUIRE(pack.drivers.size() > driver);
    REQUIRE(pack.psi->is_complex());
    pack.drivers[driver]->run();
    return pack.drivers[driver]->run_result();
}
}  // namespace

TEST_CASE("VMC at a twist reports <Im E_L> consistent with zero",
          "[vmc][physics][twist]") {
    const alea::RunResult r = run_twisted(0, {0.2, 0.1});
    const double im = scalar_of(r, "energy_imag");
    const double im_err = scalar_of(r, "energy_imag_error");
    INFO("Im E_L = " << im << " +/- " << im_err
                     << "   E = " << scalar_of(r, "energy"));
    REQUIRE(im_err > 0.0);
    CHECK(std::abs(im) < 4.0 * im_err);
}

TEST_CASE("Fixed-phase DMC is invariant under twist -> -twist (time reversal)",
          "[dmc][physics][twist]") {
    const alea::RunResult a = run_twisted(1, {0.2, 0.1});
    const alea::RunResult b = run_twisted(1, {-0.2, -0.1});
    bool fixed_phase = false;
    for (const auto& [k, v] : a.metadata)
        if (k == "node_constraint") fixed_phase = (v == "fixed_phase");
    CHECK(fixed_phase);
    const double ea = scalar_of(a, "energy"), eb = scalar_of(b, "energy");
    const double sa = scalar_of(a, "energy_error"), sb = scalar_of(b, "energy_error");
    INFO("E(+θ) = " << ea << " +/- " << sa << "   E(−θ) = " << eb << " +/- " << sb);
    REQUIRE(std::isfinite(ea));
    CHECK(std::abs(ea - eb) < 4.0 * std::hypot(sa, sb) + 1e-12);
}

// The same two checks with a like-spin backflow channel: the complex backflow
// determinant's phase planes reach Im E_L, and its moves, refreshes and
// branching run under fixed-phase DMC.
TEST_CASE("VMC with backflow at a twist reports <Im E_L> consistent with zero",
          "[vmc][physics][twist][backflow]") {
    const alea::RunResult r = run_twisted(0, {0.2, 0.1}, "heg2d_twisted_backflow.yaml");
    const double im = scalar_of(r, "energy_imag");
    const double im_err = scalar_of(r, "energy_imag_error");
    INFO("Im E_L = " << im << " +/- " << im_err
                     << "   E = " << scalar_of(r, "energy"));
    REQUIRE(im_err > 0.0);
    CHECK(std::abs(im) < 4.0 * im_err);
}

TEST_CASE("Fixed-phase DMC with backflow is invariant under twist -> -twist",
          "[dmc][physics][twist][backflow]") {
    const alea::RunResult a = run_twisted(1, {0.2, 0.1}, "heg2d_twisted_backflow.yaml");
    const alea::RunResult b = run_twisted(1, {-0.2, -0.1}, "heg2d_twisted_backflow.yaml");
    const double ea = scalar_of(a, "energy"), eb = scalar_of(b, "energy");
    const double sa = scalar_of(a, "energy_error"), sb = scalar_of(b, "energy_error");
    INFO("E(+θ) = " << ea << " +/- " << sa << "   E(−θ) = " << eb << " +/- " << sb);
    REQUIRE(std::isfinite(ea));
    CHECK(std::abs(ea - eb) < 4.0 * std::hypot(sa, sb) + 1e-12);
}

// Optimizing backflow at a twist.  At a twist that only shifts the Γ closed
// shell, ψ_θ = e^{iθ·Σr} ψ_Γ exactly (heg2d_backflow_opt_shifted_shell.yaml),
// so the complex optimizer (complex O, E_L, ∂E_L, the real parts of complex
// matrices) must retrace the real Γ optimizer: same Metropolis decisions, the
// same steps, every energy shifted by Nθ²/2.  An independent end-to-end check
// of the complex path against the real one, with real code on the other side.
namespace {
struct OptimizerRun {
    std::vector<double> params, energies;
    double final_energy = 0.0;
    int iterations = 0;
};
OptimizerRun run_shifted_shell_optimizer(std::vector<double> twist) {
    const std::string path =
        std::string(ALEA_CONFIG_DIR) + "/heg2d_backflow_opt_shifted_shell.yaml";
    alea::io::Config cfg = alea::io::parse_config_file(path);
    cfg.lattice.twist = std::move(twist);
    alea::App app(cfg);
    auto pack = app.build_drivers();
    REQUIRE(pack.drivers.size() == 1);
    pack.drivers[0]->run();
    const alea::RunResult r = pack.drivers[0]->run_result();
    OptimizerRun out;
    for (const auto& [name, values] : r.series) {
        if (name == "final_params") out.params = values;
        if (name == "energy_history") out.energies = values;
    }
    out.final_energy = scalar_of(r, "final_energy");
    out.iterations = static_cast<int>(scalar_of(r, "iterations"));
    return out;
}
}  // namespace

TEST_CASE("Optimizing backflow at a twist that shifts a closed shell retraces the Gamma "
          "optimization",
          "[optimize][physics][twist][backflow]") {
    const std::vector<double> twist = {0.05, 0.03};
    const OptimizerRun gamma = run_shifted_shell_optimizer({0.0, 0.0});
    const OptimizerRun twisted = run_shifted_shell_optimizer(twist);

    // N θ²/2 = Σ|k+θ|²/2 − Σ|k|²/2 over a closed shell (Σk = 0).
    const std::string path =
        std::string(ALEA_CONFIG_DIR) + "/heg2d_backflow_opt_shifted_shell.yaml";
    const alea::SimulationCell cell = alea::build_cell(alea::io::parse_config_file(path), 7);
    const double shift = alea::make_twisted_plane_wave_orbitals(cell, 7, twist)->kinetic_energy_sum() -
                         alea::make_twisted_plane_wave_orbitals(cell, 7, {0.0, 0.0})->kinetic_energy_sum();
    REQUIRE(shift > 0.0);

    REQUIRE(gamma.params.size() == 3);   // bf0_c0, bf0_c2, the Jastrow α0
    REQUIRE(twisted.params.size() == gamma.params.size());
    REQUIRE(gamma.iterations > 0);
    CHECK(twisted.iterations == gamma.iterations);
    // The optimizer moved the parameters (from 0.1, 0.0, 0.5) ...
    CHECK(std::abs(gamma.params[0] - 0.1) + std::abs(gamma.params[1]) > 1e-3);
    // ... and the twisted run moved them identically.
    for (std::size_t k = 0; k < gamma.params.size(); ++k) {
        INFO("parameter " << k << ": Gamma " << gamma.params[k] << ", twist " << twisted.params[k]);
        CHECK(twisted.params[k] == Catch::Approx(gamma.params[k]).epsilon(1e-7).margin(1e-9));
    }
    REQUIRE(twisted.energies.size() == gamma.energies.size());
    for (std::size_t k = 0; k < gamma.energies.size(); ++k) {
        INFO("iteration " << k);
        CHECK(twisted.energies[k] - gamma.energies[k] == Catch::Approx(shift).margin(1e-8));
    }
    CHECK(twisted.final_energy - gamma.final_energy == Catch::Approx(shift).margin(1e-8));
}

// At a generic twist Im O and Im E_L do not vanish, and they are tied by the
// hermiticity of H: under |ψ|², for any parameter p,
//   <∂_p E_L> = <E_L* O_p> − <E_L O_p> = −2i <Im E_L · O_p>,
// that is <Re ∂_p E_L> = 2 <Im E_L · Im O_p> and <Im ∂_p E_L> = −2 <Im E_L · Re O_p>.
// The two sides come from different code (the kinetic derivative of the
// backflow curvature term against the phase planes and Im O), on walkers left
// by VMC moves (deferred commits, wraps, refreshes).  The optimizer's energy
// gradient rests on exactly this.  Checked within 4 standard errors of the
// (thread, block) means, which carry the serial correlation.  The backflow is
// strong (c0 = 0.6, c2 = −0.1) so that 2<Im E_L Im O_p> stands 6σ above the
// residual's error for every parameter: the check fails if that term is
// dropped (it does not at the config's own c0 = 0.15, where it is 0.5σ).
TEST_CASE("At a twist, the backflow parameters obey <dE_L/dp> = -2i <Im E_L O_p>",
          "[vmc][physics][twist][backflow][optimize]") {
    const std::string path = std::string(ALEA_CONFIG_DIR) + "/heg2d_twisted_backflow.yaml";
    alea::io::Config cfg = alea::io::parse_config_file(path);
    bool overridden = false;
    for (alea::io::WavefunctionItem& item : cfg.wavefunction)
        if (auto* slater = std::get_if<alea::io::SlaterSpec>(&item)) {
            REQUIRE(slater->backflow.size() == 1);
            slater->backflow[0].params = {0.6, -0.1};
            overridden = true;
        }
    REQUIRE(overridden);
    alea::App app(cfg);
    auto pack = app.build_drivers();
    REQUIRE(pack.psi->is_complex());
    auto* vmc = dynamic_cast<alea::VMC*>(pack.drivers[0].get());
    REQUIRE(vmc != nullptr);
    const std::size_t n_opt = pack.psi->total_optimizable_params();
    REQUIRE(n_opt == 3);   // bf0_c0, bf0_c2, bf0_L (the Jastrow is fixed)

    // Per (thread, block): Σ of the residuals r1 = Re ∂E_L − 2 Im E_L Im O and
    // r2 = Im ∂E_L + 2 Im E_L Re O, and of 2 Im E_L Im O alone (the size of
    // what the identity balances), per parameter.
    constexpr int kThreads = 2, kSteps = 10;
    struct Acc {
        std::vector<std::vector<double>> r1, r2, scale;   // [block][parameter]
        std::vector<std::size_t> count;
    };
    std::vector<Acc> acc(kThreads);
    std::vector<std::size_t> calls(kThreads, 0);
    // Catch2 assertions are not thread-safe: the hook only records failures.
    std::vector<char> unsupported(kThreads, 0);
    bool thread_out_of_range = false;
    const alea::Hamiltonian& ham = *pack.hamiltonian;
    vmc->set_measure_hook([&](const alea::WalkerSet& ws, const alea::TrialWavefunction& psi) {
        const int tid = omp_get_thread_num();
        if (tid >= kThreads) {
#pragma omp atomic write
            thread_out_of_range = true;
            return;
        }
        Acc& a = acc[static_cast<std::size_t>(tid)];
        const std::size_t block = calls[static_cast<std::size_t>(tid)]++ / kSteps;
        if (a.r1.size() <= block) {
            a.r1.resize(block + 1, std::vector<double>(n_opt, 0.0));
            a.r2.resize(block + 1, std::vector<double>(n_opt, 0.0));
            a.scale.resize(block + 1, std::vector<double>(n_opt, 0.0));
            a.count.resize(block + 1, 0);
        }
        const std::size_t nw = ws.num_walkers(), field = ws.n_particles() * ws.dim() * nw;
        std::vector<double> o_re(n_opt * nw), o_im(n_opt * nw);
        std::vector<double> dg(n_opt * field), dl(n_opt * field), dpg(n_opt * field),
            dpl(n_opt * field), de_re(n_opt * nw), de_im(n_opt * nw);
        std::vector<char> supported(n_opt);
        psi.compute_complex_parameter_derivatives(ws, o_re, o_im);
        if (!psi.compute_complex_parameter_gradient_derivatives(ws, dg, dl, dpg, dpl, supported))
            unsupported[static_cast<std::size_t>(tid)] = 1;
        ham.complex_local_energy_parameter_derivatives(ws, dg, dl, dpg, dpl, n_opt, de_re, de_im);
        const std::span<const double> f = ws.local_energies_imag();
        for (std::size_t k = 0; k < n_opt; ++k)
            for (std::size_t w = 0; w < nw; ++w) {
                const std::size_t j = k * nw + w;
                a.r1[block][k] += de_re[j] - 2.0 * f[w] * o_im[j];
                a.r2[block][k] += de_im[j] + 2.0 * f[w] * o_re[j];
                a.scale[block][k] += 2.0 * f[w] * o_im[j];
            }
        a.count[block] += nw;
    });
    vmc->run();
    REQUIRE_FALSE(thread_out_of_range);
    for (char u : unsupported) REQUIRE(u == 0);

    for (std::size_t k = 0; k < n_opt; ++k) {
        double m1 = 0.0, q1 = 0.0, m2 = 0.0, q2 = 0.0, ms = 0.0, qs = 0.0;
        std::size_t groups = 0;
        for (const Acc& a : acc)
            for (std::size_t b = 0; b < a.count.size(); ++b) {
                if (a.count[b] == 0) continue;
                const double n = static_cast<double>(a.count[b]);
                const double g1 = a.r1[b][k] / n, g2 = a.r2[b][k] / n, gs = a.scale[b][k] / n;
                m1 += g1;
                q1 += g1 * g1;
                m2 += g2;
                q2 += g2 * g2;
                ms += gs;
                qs += gs * gs;
                ++groups;
            }
        REQUIRE(groups > 10);
        const double ng = static_cast<double>(groups);
        const auto mean_err = [ng](double m, double q) {
            const double mean = m / ng;
            return std::pair{mean, std::sqrt(std::max(q / ng - mean * mean, 0.0) / (ng - 1.0))};
        };
        const auto [r1, e1] = mean_err(m1, q1);
        const auto [r2, e2] = mean_err(m2, q2);
        const auto [s, es] = mean_err(ms, qs);
        INFO("parameter " << k << ": <Re dE_L> - 2<ImE ImO> = " << r1 << " +/- " << e1
                          << ",  <Im dE_L> + 2<ImE ReO> = " << r2 << " +/- " << e2
                          << ",  2<ImE ImO> = " << s << " +/- " << es);
        CHECK(std::abs(r1) < 4.0 * e1);
        CHECK(std::abs(r2) < 4.0 * e2);
        // The Im O · Im E_L term carries weight: without it r1 would sit 4σ out.
        CHECK(std::abs(s) > 4.0 * (e1 + es));
    }
}

// ── Magnetic field (fixed phase with a vector potential) ────────────────────
TEST_CASE("Fixed-phase DMC in a magnetic field projects to the exact Fock-Darwin level",
          "[dmc][physics][gauge]") {
    const double m = 0.7, wc = -1.3 / m;
    const double exact = 2.0 * std::sqrt(0.25 + 0.25 * wc * wc) + wc / 2.0;
    auto run = [](std::size_t driver) {
        const std::string path = std::string(ALEA_CONFIG_DIR) + "/fock_darwin_dot.yaml";
        alea::App app(alea::io::parse_config_file(path));
        auto pack = app.build_drivers();
        REQUIRE(pack.drivers.size() == 2);
        pack.drivers[driver]->run();
        return pack.drivers[driver]->run_result();
    };
    const alea::RunResult vmc = run(0);
    const alea::RunResult dmc = run(1);
    const double ev = scalar_of(vmc, "energy"), ed = scalar_of(dmc, "energy");
    const double sd = scalar_of(dmc, "energy_error");
    INFO("exact " << exact << "   VMC " << ev << "   DMC " << ed << " +/- " << sd);
    CHECK(ev > exact + 0.01);                       // wrong width: clearly above
    CHECK(std::abs(ed - exact) < 4.0 * sd + 2e-3);  // projected (τ = 0.01 bias)
}

// ── Two particles in a trap with an exact energy: VMC above, DMC on it ──────
// Runs stage 0 (VMC) and stage 1 (DMC) of `config`.  The DMC margin is set
// from the run-to-run spread of the config's fixed-length DMC over
// independent seeds, measured where each test is defined: the printed ± of
// one run is a floor, not that spread.  Changing the run length changes the
// noise.
namespace {
void check_two_body_exact(const std::string& config, double exact, double dmc_margin) {
    auto run = [&](std::size_t driver) {
        const std::string path = std::string(ALEA_CONFIG_DIR) + "/" + config;
        alea::App app(alea::io::parse_config_file(path));
        auto pack = app.build_drivers();
        REQUIRE(pack.drivers.size() == 2);
        pack.drivers[driver]->run();
        return pack.drivers[driver]->run_result();
    };
    const alea::RunResult vmc = run(0), dmc = run(1);
    const double ev = scalar_of(vmc, "energy"), ed = scalar_of(dmc, "energy");
    const double sv = scalar_of(vmc, "energy_error"), sd = scalar_of(dmc, "energy_error");
    INFO(config << ": exact " << exact << "   VMC " << ev << " +/- " << sv
                << "   DMC " << ed << " +/- " << sd);
    CHECK(ev - exact > 4.0 * sv);                         // VMC measurably above
    CHECK(std::abs(ed - exact) < dmc_margin);             // DMC exact within its noise
}
}  // namespace

// ── Literature check: two electrons in a quantum dot in a magnetic field ────
// Taut-type closed-form points of the relative problem (see the configs and
// tests/references/dot_two_electron_exact.py; physics: Wagner, Merkt & Chaplik, PRB 45,
// 1951 (1992)).  VMC is off (wrong Jastrow shape); DMC must be exact.
// Over independent seeds (2026-10-19) the DMC energy of these fixed-length
// runs scatters with a standard deviation of 1.4e-3 (triplet, 48 seeds;
// printed ±~4.8e-4) and 7.2e-3 (singlet, 24 seeds; printed ±~2.4e-3). The
// seed means, 1.13338 and 3.0012, show no τ bias at that level. The DMC
// margin is 4 of those deviations, still well inside the VMC offsets
// (+0.018, +0.13) that the trials' wrong Jastrow shape produces.
TEST_CASE("Two-electron dot in B: fixed-phase DMC gets the exact triplet energy",
          "[dmc][physics][gauge][literature]") {
    check_two_body_exact("dot2e_triplet.yaml", 17.0 / 15.0, 5.5e-3);
}

TEST_CASE("Two-electron dot in B: DMC gets the exact singlet energy",
          "[dmc][physics][gauge][literature]") {
    check_two_body_exact("dot2e_singlet.yaml", 3.0, 2.9e-2);
}

// ── Backflow keeps an exact node: two like-spin electrons, no field ─────────
// The real triplet f(r) cos φ_rel vanishes on x_1 = x_2, which the two-
// Gaussian determinant and its same-group backflow both reproduce (see the
// config), so fixed-node DMC with backflow is exact: E = 1.9139333019
// (tests/references/dot_two_electron_exact.py --w0 0.5 --B 0 --m 1).  Over 24
// seeds (2026-11-07) the DMC energy of this run length has mean exact
// + 0.04(32)e-3 and standard deviation 1.6e-3 (printed ±~1.2e-3; without the
// backflow block: + 0.03(21)e-3 and 1.05e-3).  Margin = 4 deviations, 6.5e-3,
// inside the VMC offset +0.0275 of the unoptimized backflow and Jastrow.
TEST_CASE("Two-electron dot: fixed-node DMC with backflow gets the exact triplet energy",
          "[dmc][physics][backflow][literature]") {
    check_two_body_exact("dot2e_triplet_backflow.yaml", 1.9139333019, 6.5e-3);
}

// ── Two dipoles (1/r³) in a trap at the coupling of the 2-D dipolar crystal ─
// ω = 2000 holds the pair at the lattice spacing of n r0² = 256 (see the
// config).  Exact E = 12044.7791588900 from tests/references/inverse_power_trap_exact.py,
// an independent finite-volume solution of the relative radial equation.
// The trial (zero_energy × Nosanow) is positive, so DMC is exact up to time
// step and population; VMC sits +26 above.  Over 72 seeds (2026-10-25) the
// DMC energy of this run length has mean exact + 0.33(8), the τ = 1e-5 bias
// (the branch cutoff cuts no walker here), and standard deviation 0.64
// (printed ±~0.68); the largest deviation from exact was 2.72 (one seed at
// 3.7 deviations from the mean; the rest within 2.2).  Margin = bias + 4
// deviations, rounded up: 3.0, i.e. 2.5e-4 of E and well inside the VMC
// offset of +26.
TEST_CASE("Two dipoles in a trap: DMC gets the exact energy at crystal coupling",
          "[dmc][physics][inverse_power][zero_energy][nosanow]") {
    check_two_body_exact("dipoles_trap_2body.yaml", 12044.7791588900, 3.0);
}
