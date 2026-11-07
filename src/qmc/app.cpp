// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "qmc/app.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <map>
#include <omp.h>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <type_traits>
#include <variant>
#include <vector>

#include "estimators/estimator.h"
#include "hamiltonian/moire.h"
#include "population/simulation_cell.h"
#include "io/logger.h"
#include "numerics/special_functions.h"
#include "io/walker_io.h"
#include "numerics/random.h"
#include "population/walker.h"
#include "population/walker_ensemble.h"
#include "population/walker_set.h"
#include "qmc/dmc.h"
#include "pimc/pimc.h"
#include "builders/hamiltonian_builders.h"
#include "builders/action_builders.h"
#include "builders/estimator_builders.h"
#include "builders/exchange_classes.h"
#include "builders/wavefunction_builders.h"

namespace alea {

namespace {

// Seed every walker's positions.  Particles are spread uniformly in
// [-L/2, L/2]^dim, EXCEPT those flagged in `at_origin` (size = n_particles, empty
// → none), which start exactly at the origin.  This pins a clamped nucleus
// (Born–Oppenheimer) at the fixed orbital centers used by a Slater determinant;
// a large nucleus mass (inv_mass ≈ 0) then keeps it there for the whole run.
// `anchors` (size np*dim when non-empty, NaN where absent) starts a particle on
// its own crystal site, scattered by `jitter`.  Uniform placement would put most
// particles far from every Gaussian center, so the determinant would start
// numerically singular; see crystal_site_anchors.  `partners` (see
// pairing_partners) starts a column particle of a pairing determinant next to
// its row partner, after every other particle is placed.
void initialize_positions(WalkerEnsemble& ensemble, double L,
                          std::uint64_t seed,
                          std::span<const char> at_origin = {},
                          const SimulationCell& cell = SimulationCell{},
                          std::span<const double> anchors = {},
                          double jitter = 0.0,
                          const PairPartners& partners = PairPartners{}) {
    numerics::Random rng(seed);
    const std::size_t np  = ensemble.n_particles();
    const std::size_t dim = ensemble.dim();
    const auto        a   = cell.vectors();
    std::vector<double> pos(np * dim);
    auto partner_of = [&](std::size_t p) -> std::ptrdiff_t {
        const bool fixed = p < at_origin.size() && at_origin[p] != 0;
        return !fixed && p < partners.partner.size() ? partners.partner[p] : -1;
    };
    for (Walker& w : ensemble.walkers()) {
        for (std::size_t p = 0; p < np; ++p) {
            if (partner_of(p) >= 0) continue;
            const bool fixed = p < at_origin.size() && at_origin[p] != 0;
            const bool anchored =
                !anchors.empty() && (p + 1) * dim <= anchors.size() &&
                !std::isnan(anchors[p * dim]);
            if (anchored && !fixed) {
                for (std::size_t d = 0; d < dim; ++d)
                    pos[p * dim + d] =
                        anchors[p * dim + d] + jitter * rng.gaussian();
            } else if (fixed) {
                for (std::size_t d = 0; d < dim; ++d) pos[p * dim + d] = 0.0;
            } else if (cell.is_periodic()) {
                // Uniform over the cell: draw fractional coordinates in
                // [-½, ½) and map them through the primitive vectors, so the
                // walkers fill the actual cell rather than a box that ignores
                // its shape.  `L` is unused on this path.
                double f[SimulationCell::kMaxDim];
                for (std::size_t i = 0; i < dim; ++i) f[i] = rng.uniform(-0.5, 0.5);
                for (std::size_t d = 0; d < dim; ++d) {
                    double s = 0.0;
                    for (std::size_t i = 0; i < dim; ++i) s += f[i] * a[i * dim + d];
                    pos[p * dim + d] = s;
                }
            } else {
                for (std::size_t d = 0; d < dim; ++d)
                    pos[p * dim + d] = rng.uniform(-0.5 * L, 0.5 * L);
            }
        }
        // A partner is never itself partnered (a group sits in one block).
        for (std::size_t p = 0; p < np; ++p) {
            const std::ptrdiff_t q = partner_of(p);
            if (q < 0) continue;
            for (std::size_t d = 0; d < dim; ++d)
                pos[p * dim + d] = pos[static_cast<std::size_t>(q) * dim + d] +
                                   partners.spread * rng.gaussian();
        }
        w.set_positions(pos);
    }
}

// Build the stage's `estimators:` list and attach it to its driver.  Absent →
// none (energy only).  Every rule about a kind — which samplers measure it,
// its cell, its result keys — lives in builders/estimator_builders.h.
template <class DriverT>
void attach_estimators(DriverT& driver, std::span<const io::EstimatorItem> items,
                       Sampler sampler, const std::vector<ParticleGroup>& groups,
                       std::size_t dim, const SimulationCell& cell) {
    auto ests = build_estimators(items, EstimatorContext{groups, cell, dim, sampler});
    if (!ests.empty()) driver.set_estimators(std::move(ests));
}

}  // namespace

// ── Builders ─────────────────────────────────────────────────────────────────

std::vector<ParticleGroup> build_groups(const io::Config& cfg) {
    std::vector<ParticleGroup> groups;
    groups.reserve(cfg.particles.size());
    const std::size_t dim = static_cast<std::size_t>(cfg.lattice.dimension);
    std::size_t offset = 0;
    for (const io::ParticleSpec& p : cfg.particles) {
        ParticleGroup g;
        g.name = p.group;
        g.charge = p.charge;
        g.first = offset;
        g.last = offset + static_cast<std::size_t>(p.count());
        offset = g.last;
        if (p.mass.size() == 1) {
            g.mass.assign(dim, p.mass[0]);
            g.inv_mass.assign(dim, 1.0 / p.mass[0]);
        } else {
            g.mass = p.mass;
            g.inv_mass.resize(p.mass.size());
            for (std::size_t d = 0; d < p.mass.size(); ++d)
                g.inv_mass[d] = 1.0 / p.mass[d];
        }
        groups.push_back(std::move(g));
    }
    return groups;
}

VMCConfig build_vmc_config(const io::QmcSpec& q, std::uint64_t seed) {
    VMCConfig vc;
    vc.n_equilibration_blocks = q.equilibration_blocks;
    vc.n_accumulation_blocks  = q.accumulation_blocks;
    vc.n_steps                = q.steps_per_block;
    vc.n_substeps             = q.sub_steps;
    vc.tau                    = q.time_step;
    vc.use_drift              = (q.random_walk == "langevin");
    vc.n_threads = (q.threads > 0) ? q.threads : omp_get_max_threads();
    vc.seed                   = seed;
    return vc;
}

CrystalAnchors crystal_site_anchors(const io::Config& cfg,
                                    const std::vector<ParticleGroup>& groups,
                                    const SimulationCell& cell) {
    const std::size_t dim = static_cast<std::size_t>(cfg.lattice.dimension);
    std::size_t np = 0;
    for (const ParticleGroup& g : groups) np = std::max(np, g.last);
    CrystalAnchors out;
    out.pos.assign(np * dim, std::numeric_limits<double>::quiet_NaN());
    for (const io::SlaterSpec* sp : io::items_of<io::SlaterSpec>(cfg.wavefunction)) {
        const io::SlaterSpec& s = *sp;
        if (!is_crystal_slater(s)) continue;
        const ParticleGroup& g = find_group(groups, s.group);
        const std::vector<double> sites = crystal_sites(s, g, cell, dim);
        for (std::size_t i = 0; i < g.size(); ++i)
            for (std::size_t d = 0; d < dim; ++d)
                out.pos[(g.first + i) * dim + d] = sites[i * dim + d];
        // sigma of exp(-alpha r^2): the physical spread of a particle about its
        // own site, and therefore the right scale for the initial jitter.
        if (s.alpha && *s.alpha > 0.0)
            out.jitter = std::max(out.jitter, 1.0 / std::sqrt(4.0 * *s.alpha));
    }
    // A Nosanow factor pins particle i to site i just as strongly: started
    // uniformly, every particle would sit far from its own site.
    for (const io::NosanowSpec* ns : io::items_of<io::NosanowSpec>(cfg.wavefunction)) {
        const ParticleGroup& g = find_group(groups, ns->group);
        const std::vector<double> sites =
            lattice_sites(ns->sites, ns->centers, g, cell, dim, "nosanow on group " + g.name);
        for (std::size_t i = 0; i < g.size(); ++i)
            for (std::size_t d = 0; d < dim; ++d)
                out.pos[(g.first + i) * dim + d] = sites[i * dim + d];
        if (ns->alpha > 0.0)
            out.jitter = std::max(out.jitter, 1.0 / std::sqrt(4.0 * ns->alpha));
    }
    return out;
}

PairPartners pairing_partners(const io::Config& cfg,
                              const std::vector<ParticleGroup>& groups,
                              const SimulationCell& cell) {
    std::size_t np = 0;
    for (const ParticleGroup& g : groups) np = std::max(np, g.last);
    PairPartners out;
    for (const io::PairingSpec* ps : io::items_of<io::PairingSpec>(cfg.wavefunction)) {
        if (out.partner.empty()) out.partner.assign(np, -1);
        for (const io::PairingBlockSpec& blk : ps->blocks) {
            const ParticleGroup& rows = find_group(groups, blk.rows);
            const ParticleGroup& cols = find_group(groups, blk.columns);
            for (std::size_t j = 0; j < std::min(rows.size(), cols.size()); ++j)
                out.partner[cols.first + j] = static_cast<std::ptrdiff_t>(rows.first + j);
        }
        // The pair separation of a Gaussian of width σ per axis exceeds L with
        // probability exp(−L²/2σ²); σ = L/8 makes that e^{−32}, so every pair
        // starts inside a compact φ.  Without a polynomial cutoff φ is a sum
        // of cosines, nonzero almost everywhere, and the inradius sets the scale.
        const double L = ps->poly_order >= 1 && ps->cutoff > 0.0 ? ps->cutoff
                                                                 : cell.ws_radius();
        out.spread = std::max(out.spread, L / 8.0);
    }
    return out;
}

TrialWavefunction build_wavefunction(const io::Config& cfg,
                                     const std::vector<ParticleGroup>& groups,
                                     const Hamiltonian* ham,
                                     const SimulationCell& cell) {
    const std::size_t dim = static_cast<std::size_t>(cfg.lattice.dimension);
    if (!cfg.lattice.twist.empty()) {
        if (cfg.lattice.type != "periodic")
            throw std::runtime_error(
                "lattice.twist needs a periodic lattice: a twist is a boundary "
                "condition ψ(r + a_i) = e^{2πi t_i} ψ(r) of the cell.");
        if (cfg.lattice.twist.size() != dim)
            throw std::runtime_error(
                "lattice.twist has " + std::to_string(cfg.lattice.twist.size()) +
                " components, the lattice is " + std::to_string(dim) + "-D");
        if (!is_gamma_twist(cfg.lattice.twist))
            for (const io::SlaterSpec* s : io::items_of<io::SlaterSpec>(cfg.wavefunction))
                if (s->orbital_type != "plane_wave")
                    throw std::runtime_error(
                        "lattice.twist away from Γ needs twisted orbitals; only "
                        "`plane_wave` slaters support a twist (Gaussian orbitals "
                        "are strictly periodic).");
    }

    // One item → one component, in document order: the optimiser's parameter
    // vector and the config snapshot both index the wavefunction this way.
    // Each kind's builder lives in src/builders/ (wavefunction_builders.h).
    const WavefunctionContext ctx{cfg, groups, ham, cell, dim};
    TrialWavefunction psi;
    for (const io::WavefunctionItem& item : cfg.wavefunction)
        psi.add_component(std::visit(
            [&](const auto& spec) { return build_component(spec, ctx); }, item));
    return psi;
}

// Build the simulation cell from `system.lattice`.
//
//   type: open      → no cell at all; `L` keeps its historical meaning as the
//                     half-width of the initial walker scatter box.
//   type: periodic  → a Bravais cell from exactly one of: explicit `vectors`,
//                     a density parameter `rs`, per-axis `L: [Lx, Ly]`, or a
//                     scalar `L` together with `shape`.
//
// The `rs` form is the thermodynamic-limit knob: it fixes ρ = N/V directly, so
// a sequence of runs at growing N and fixed rs walks toward N → ∞ at constant
// density.  In 2-D each particle owns π rs² of area, hence A = N π rs².
SimulationCell build_cell(const io::Config& cfg, std::size_t n_particles) {
    const std::size_t dim = static_cast<std::size_t>(cfg.lattice.dimension);
    if (cfg.lattice.type != "periodic") return SimulationCell(dim);

    const io::LatticeSpec& lat = cfg.lattice;
    if (!lat.vectors.empty()) {
        if (lat.vectors.size() != dim * dim)
            throw std::runtime_error(
                "lattice.vectors needs " + std::to_string(dim * dim) +
                " components for dimension " + std::to_string(dim) + ", got " +
                std::to_string(lat.vectors.size()));
        return SimulationCell(dim, lat.vectors);
    }

    const std::string shape = lat.shape.empty() ? "square" : lat.shape;

    if (!lat.lengths.empty()) {
        if (lat.lengths.size() != dim)
            throw std::runtime_error("lattice.L as a list needs one length per "
                                     "dimension (" + std::to_string(dim) + ")");
        std::vector<double> v(dim * dim, 0.0);
        for (std::size_t d = 0; d < dim; ++d) v[d * dim + d] = lat.lengths[d];
        return SimulationCell(dim, v);
    }

    double L = lat.L;
    if (lat.rs > 0.0) {
        if (dim != 2)
            throw std::runtime_error("lattice.rs is defined for dimension 2 only");
        if (n_particles == 0)
            throw std::runtime_error("lattice.rs needs a nonzero particle count");
        const double area =
            static_cast<double>(n_particles) * numerics::PI * lat.rs * lat.rs;
        // Invert the shape's area formula for the lattice constant.
        L = (shape == "hexagonal") ? std::sqrt(2.0 * area / std::sqrt(3.0))
                                   : std::sqrt(area);
    }
    if (L <= 0.0)
        throw std::runtime_error(
            "periodic lattice needs one of: lattice.vectors, lattice.rs, or "
            "lattice.L");

    std::vector<double> v(dim * dim, 0.0);
    if (dim == 1) {
        v[0] = L;
    } else if (shape == "hexagonal") {
        v[0] = L;        v[1] = 0.0;
        v[2] = 0.5 * L;  v[3] = L * std::sqrt(3.0) / 2.0;
    } else if (shape == "square" || shape == "rectangular") {
        for (std::size_t d = 0; d < dim; ++d) v[d * dim + d] = L;
    } else {
        throw std::runtime_error("unknown lattice shape: '" + shape +
                                 "' (expected square, rectangular or hexagonal)");
    }
    return SimulationCell(dim, v);
}

// Reject periodic configurations whose physics this milestone cannot represent,
// loudly rather than silently returning a wrong number.
void check_cell_support(const io::Config& cfg, const SimulationCell& cell) {
    if (cfg.lattice.type != "periodic") return;

    // A Slater determinant built from orbitals centred at fixed points is not
    // periodic: psi would jump across the cell boundary.  Periodic fermions
    // need plane-wave orbitals on the reciprocal lattice (a later milestone).
    // A `slater` determinant IS supported under PBC now, in two periodic forms:
    // plane waves on a closed reciprocal shell (the fluid) and Gaussians on
    // commensurate lattice sites folded to the minimum image (the crystal).  What
    // is still unsound is a shared ORBITAL POOL of Gaussians (per-orbital
    // `alphas`): both spin blocks draw from it, nothing ties it to one site per
    // particle, and the fold can pair two of its centers to the same image.
    // Explicit `centers` with ONE shared `alpha` are a crystal (one site per
    // particle, disjoint spin blocks) and are accepted: crystal_sites() refuses
    // two centers on the same periodic image.  That is how crystals whose sites
    // are not a plain triangular lattice of the cell (moiré minima, stripes,
    // shifted or floating crystals) are specified.
    for (const io::SlaterSpec* s : io::items_of<io::SlaterSpec>(cfg.wavefunction))
        if (s->orbital_type == "gaussian" && !is_crystal_slater(*s))
            throw std::runtime_error(
                "periodic lattice combined with a `slater` orbital pool of "
                "Gaussians (per-orbital `alphas`): under PBC use "
                "`orbitals: {type: plane_wave}` (uniform fluid), "
                "`orbitals: {sites: triangular, alpha: ...}`, or explicit "
                "`centers` (one per particle) with a single `alpha` (crystal).");

    // A periodic cell's estimator restrictions (collective_density) are their
    // builders' (builders/estimator_builders.h).

    // The moiré potential is a sum of cosines on its own superlattice.  It is
    // periodic under a simulation-cell translation a_j only if g_i·a_j is an
    // integer multiple of 2π for every moiré reciprocal vector g_i; otherwise
    // V(r) is discontinuous across the cell boundary and the walkers see a
    // potential that does not exist.  A moiré Jastrow is built from the same
    // g's (its higher shells are integer combinations of them), so it needs the
    // same test or ψ jumps when a walker is wrapped.
    std::vector<std::pair<double, double>> superlattices;  // (a_M, theta0)
    for (const io::MoireSpec& m : cfg.moires)
        superlattices.emplace_back(m.a_M, m.theta0);
    for (const io::MoireJastrowSpec* mj :
         io::items_of<io::MoireJastrowSpec>(cfg.wavefunction))
        superlattices.push_back(moire_jastrow_geometry(cfg, *mj));
    if (!superlattices.empty()) {
        const std::size_t dim = cell.dim();
        const auto        a   = cell.vectors();
        for (const auto& [a_M, theta0] : superlattices) {
            const std::vector<std::vector<double>> gs =
                moire_reciprocal_vectors(a_M, theta0, dim);
            // Moiré DIRECT lattice, the dual of those g's (g_i·A_j = 2π δ_ij).
            // Reported in the error below so the user can copy it straight into
            // `lattice.vectors` instead of guessing an orientation.
            std::vector<double> a1(dim, 0.0), a2(dim, 0.0);
            if (dim == 2 && gs.size() >= 2) {
                const double det = gs[0][0] * gs[1][1] - gs[0][1] * gs[1][0];
                if (std::abs(det) > 0.0) {
                    const double k = 2.0 * numerics::PI / det;
                    a1 = {k * gs[1][1], -k * gs[1][0]};
                    a2 = {-k * gs[0][1], k * gs[0][0]};
                }
            }
            for (const std::vector<double>& g : gs)
                for (std::size_t j = 0; j < dim; ++j) {
                    double dot = 0.0;
                    for (std::size_t d = 0; d < dim; ++d)
                        dot += g[d] * a[j * dim + d];
                    const double turns = dot / (2.0 * numerics::PI);
                    if (std::abs(turns - std::nearbyint(turns)) > 1e-8)
                        throw std::runtime_error(
                            "periodic lattice is not commensurate with the moiré "
                            "superlattice (a_M = " + std::to_string(a_M) +
                            ", theta0 = " + std::to_string(theta0) +
                            "): g·a/2π = " + std::to_string(turns) +
                            " is not an integer, so the moiré potential would be "
                            "discontinuous across the cell boundary.\n"
                            "  Every cell vector must be an integer combination of "
                            "the moiré lattice vectors, which for this term are\n"
                            "    A1 = (" + std::to_string(a1[0]) + ", " +
                            std::to_string(a1[1]) + ")\n"
                            "    A2 = (" + std::to_string(a2[0]) + ", " +
                            std::to_string(a2[1]) + ")\n"
                            "  Pass those (or integer multiples) as "
                            "`lattice.vectors`.  Note the `shape:` shortcut always "
                            "builds a cell aligned with the x axis, which matches "
                            "the moiré lattice only for particular theta0.");
                }
        }
    }
}

Hamiltonian build_hamiltonian(const io::Config& cfg,
                              const std::vector<ParticleGroup>& groups,
                              const SimulationCell& cell) {  // default in app.h
    // Each term's builder lives in src/builders/ (hamiltonian_builders.h).
    const HamiltonianContext ctx{cfg, groups, cell,
                                 static_cast<std::size_t>(cfg.lattice.dimension)};
    Hamiltonian h;
    h.add_component(build_kinetic(ctx));
    for (const io::HarmonicTrapSpec& t : cfg.traps) h.add_component(build_component(t, ctx));
    if (auto trap = build_centrifugal_trap(ctx)) h.add_component(std::move(trap));

    // Pair interactions: one cell-wide Ewald term under PBC, else one term per spec.
    if (auto ewald = build_ewald(ctx)) {
        h.add_component(std::move(ewald));
    } else {
        for (const io::CoulombSpec& c : cfg.coulombs) h.add_component(build_component(c, ctx));
        for (const io::KeldyshSpec& k : cfg.keldysh) h.add_component(build_component(k, ctx));
        for (const io::ScreenedCoulombSpec& s : cfg.screened)
            h.add_component(build_component(s, ctx));
    }
    // Absolutely convergent image sum: one term per spec, Ewald or not.
    for (const io::InversePowerSpec& s : cfg.inverse_powers)
        h.add_component(build_component(s, ctx));
    for (const io::MoireSpec& m : cfg.moires) h.add_component(build_component(m, ctx));
    return h;
}

DMCConfig build_dmc_config(const io::QmcSpec& q, std::uint64_t seed) {
    DMCConfig dc;
    dc.n_equilibration_blocks = q.equilibration_blocks;
    dc.n_accumulation_blocks  = q.accumulation_blocks;
    dc.n_steps                = q.steps_per_block;
    dc.n_substeps             = q.sub_steps;
    dc.tau                    = q.time_step;
    dc.n_threads = (q.threads > 0) ? q.threads : omp_get_max_threads();
    dc.seed                   = seed;
    dc.population_generations = q.population_generations;
    // The parser admitted only these three names.
    dc.branch_cutoff = q.branch_cutoff == "drv"  ? BranchCutoff::drv
                     : q.branch_cutoff == "none" ? BranchCutoff::none
                                                 : BranchCutoff::universal;
    if (q.branch_cutoff_alpha) dc.branch_cutoff_alpha = *q.branch_cutoff_alpha;
    // The parser admitted only bare, effective and suppressed; no correlation_time
    // (or `auto`) leaves 0, which the driver measures.
    dc.reweighting = q.reweighting == "suppressed" ? Reweighting::suppressed
                   : q.reweighting == "effective" ? Reweighting::effective
                                                   : Reweighting::bare;
    if (q.correlation_time) dc.correlation_time = *q.correlation_time;
    if (q.suppression_kappa) dc.suppression_kappa = *q.suppression_kappa;
    if (q.correlation_equilibration_sweeps)
        dc.correlation_equilibration_sweeps = *q.correlation_equilibration_sweeps;
    if (q.correlation_min_sweeps) dc.correlation_min_sweeps = *q.correlation_min_sweeps;
    if (q.correlation_max_sweeps) dc.correlation_max_sweeps = *q.correlation_max_sweeps;
    // The driver checks the lags against τ, sub_steps and the equilibration.
    if (q.forward_walking) {
        dc.forward_walking_lags     = q.forward_walking->lags;
        dc.forward_walking_interval = static_cast<std::size_t>(q.forward_walking->interval);
    }
    return dc;
}

PimcConfig build_pimc_config(const io::PimcSpec& p, std::uint64_t seed) {
    PimcConfig pc;
    pc.threads             = p.threads;
    pc.samples             = p.samples;
    pc.time_step           = p.time_step;
    pc.beta                = p.beta;
    pc.equilibration_blocks = p.equilibration_blocks;
    pc.accumulation_blocks  = p.accumulation_blocks;
    pc.steps_per_block      = p.steps_per_block;
    pc.sub_steps            = p.sub_steps;
    pc.seed                 = seed;
    pc.energy               = p.energy;
    pc.compare_thermodynamic = p.compare_thermodynamic;
    pc.statistics           = p.statistics;
    pc.sign_threshold       = p.sign_threshold;
    pc.moves                = p.moves;  // empty → io::default_pimc_moves(M)
    return pc;
}

OptimizeConfig build_optimize_config(const io::OptimizerSpec& spec,
                                     std::uint64_t seed) {
    OptimizeConfig oc;
    oc.n_walkers              = spec.samples;
    oc.n_equilibration_blocks = spec.equilibration_blocks;
    oc.n_accumulation_blocks  = spec.accumulation_blocks;
    oc.n_steps                = spec.steps_per_block;
    oc.n_substeps             = spec.sub_steps;
    oc.tau                    = spec.time_step;
    oc.use_drift              = (spec.random_walk == "langevin");
    oc.max_iterations         = spec.max_iterations;
    oc.psi_step_tol           = spec.psi_step_tol;
    oc.energy_tol             = spec.energy_tol;
    oc.convergence_window     = spec.convergence_window;
    oc.num_shifts             = spec.num_shifts;
    oc.max_param_change       = spec.max_param_change;
    oc.cost_increase_tol      = spec.cost_increase_tol;
    oc.parameters             = spec.parameters;
    oc.target_shift_i         = spec.target_shift_i;
    oc.max_failed_attempts    = spec.max_failed_attempts;
    oc.start_shift_i          = spec.shift_i;
    oc.start_shift_s          = spec.shift_s;
    oc.seed                   = seed;
    oc.n_threads = (spec.threads > 0) ? spec.threads : omp_get_max_threads();
    oc.lm_config.shift_i      = spec.shift_i;
    oc.lm_config.shift_s      = spec.shift_s;
    oc.lm_config.rescale       = spec.rescale;
    oc.lm_config.xi            = spec.xi;
    return oc;
}

// ── App ──────────────────────────────────────────────────────────────────────

App::App(io::Config config) : cfg_(std::move(config)) {}

DriverPack App::build_drivers(const std::filesystem::path& checkpoint_dir) {
    DriverPack pack;

    if (cfg_.qmc.empty() && cfg_.optimizers.empty() && cfg_.pimc.empty())
        return pack;

    // `flavours` is PIMC-only for now: Slater spin blocks, Jastrow cusps and
    // the k-space Jastrow seed read the {up, down} split and would misread an
    // N-flavour entry, so no ψ-based stage or wavefunction item may see one.
    const bool declares_flavours = std::any_of(
        cfg_.particles.begin(), cfg_.particles.end(),
        [](const io::ParticleSpec& p) { return !p.flavours.empty(); });
    if (declares_flavours &&
        (!cfg_.qmc.empty() || !cfg_.optimizers.empty() || !cfg_.wavefunction.empty()))
        throw std::runtime_error(
            "system.particles `flavours` is PIMC-only for now (determinants and "
            "Jastrow cusps read up/down): remove the vmc/dmc/optimizer stages and "
            "wavefunction items, or write the counts as up/down");

    // Exchange classes are (group, flavour) sets of fermions or bosons; the
    // driver refuses a class of more than one particle unless the run opts
    // into distinguishable statistics.
    const std::vector<ExchangeClass> exchange_classes = cfg_.pimc.empty()
        ? std::vector<ExchangeClass>{}
        : build_exchange_classes(cfg_.particles);
    const int exchange_class_max =
        static_cast<int>(largest_exchange_class(exchange_classes));

    // PIMC widths and springs require a finite positive mass on every axis.
    // Restrict this validation to PIMC so legacy VMC/DMC inputs are unchanged.
    if (!cfg_.pimc.empty()) {
        for (double twist : cfg_.lattice.twist)
            if (!std::isfinite(twist) || twist != 0.0)
                throw std::runtime_error("pimc: non-Gamma lattice twist requires a phase action; only zero twist is supported");
        const auto dim = static_cast<std::size_t>(cfg_.lattice.dimension);
        for (const auto& particle : cfg_.particles) {
            if (particle.mass.size() != 1 && particle.mass.size() != dim)
                throw std::runtime_error("pimc: mass must be scalar or dimension-sized for group '" + particle.group + "'");
            for (double mass : particle.mass)
                if (!std::isfinite(mass) || !(mass > 0.0))
                    throw std::runtime_error("pimc: mass must be finite and positive for group '" + particle.group + "'");
        }
    }

    // Build shared domain objects once, on the heap.  The Hamiltonian is built
    // first so the wavefunction can read each pair potential's 1/r residue to
    // set any Drummond jastrow cusp (Kinetic reads ψ only at evaluate time, so
    // this ordering is safe).
    const std::vector<ParticleGroup> groups = build_groups(cfg_);
    const std::size_t dim = static_cast<std::size_t>(cfg_.lattice.dimension);

    // The simulation cell: open (no boundaries) or a periodic Bravais cell.
    // Built once and carried by every ensemble, so every driver-side WalkerSet,
    // distance table and wavefunction component folds against the same geometry.
    // It must exist BEFORE the Hamiltonian, because a periodic run replaces the
    // per-pair Coulomb terms with a single cell-wide Ewald term.
    std::size_t n_particles = 0;
    for (const ParticleGroup& g : groups) n_particles += g.size();
    const SimulationCell cell = build_cell(cfg_, n_particles);
    check_cell_support(cfg_, cell);

    pack.hamiltonian = std::make_shared<Hamiltonian>(
        build_hamiltonian(cfg_, groups, cell));
    pack.psi = std::make_shared<TrialWavefunction>(
        build_wavefunction(cfg_, groups, pack.hamiltonian.get(), cell));

    // Particles whose group declares `initialization: fixed` start clamped at the
    // origin (a Born–Oppenheimer nucleus sitting at the Slater orbital centers).
    // Built from cfg_.particles using the same index mapping as build_groups.
    std::vector<char> at_origin;
    {
        std::size_t off = 0;
        for (const io::ParticleSpec& p : cfg_.particles) {
            const std::size_t cnt = static_cast<std::size_t>(p.count());
            const char fixed = (p.initialization == "fixed") ? 1 : 0;
            at_origin.resize(off + cnt, 0);
            for (std::size_t i = 0; i < cnt; ++i) at_origin[off + i] = fixed;
            off += cnt;
        }
    }

    // Crystal-site starting points for any crystal determinant or Nosanow
    // factor (empty otherwise).
    const CrystalAnchors anchors =
        !io::items_of<io::SlaterSpec>(cfg_.wavefunction).empty() ||
                !io::items_of<io::NosanowSpec>(cfg_.wavefunction).empty()
            ? crystal_site_anchors(cfg_, groups, cell)
            : CrystalAnchors{};
    // Column particles of any pairing determinant start next to their rows.
    const PairPartners partners = pairing_partners(cfg_, groups, cell);

    // ── Optimizer drivers ──────────────────────────────────────────────────
    for (const io::OptimizerSpec& spec : cfg_.optimizers) {
        const std::size_t nw = static_cast<std::size_t>(spec.samples);

        auto ensemble = std::make_shared<WalkerEnsemble>(nw, groups, dim, cell);
        initialize_positions(*ensemble, cfg_.lattice.L, cfg_.seed, at_origin, cell,
                             anchors.pos, anchors.jitter, partners);

        OptimizeConfig opt_cfg = build_optimize_config(spec, cfg_.seed);
        pack.drivers.push_back(std::make_unique<LinearOptimize>(
            std::move(opt_cfg), *ensemble, *pack.psi, *pack.hamiltonian));
        pack.ensembles.push_back(std::move(ensemble));
        pack.walkers_out.emplace_back();  // no checkpoint for optimiser runs
        pack.walkers_in_set.push_back(0);
    }

    // Programmatic legacy configs may not carry parser-provided order indexes.
    auto sampling_order = cfg_.pimc.empty()
        ? std::vector<io::SamplingStageIndex>{} : cfg_.sampling_order;
    if (sampling_order.empty()) {
        for (std::size_t i = 0; i < cfg_.qmc.size(); ++i)
            sampling_order.push_back(io::QmcStageIndex{i});
        for (std::size_t i = 0; i < cfg_.pimc.size(); ++i)
            sampling_order.push_back(io::PimcStageIndex{i});
    }
    std::vector<std::size_t> qmc_ordinals(cfg_.qmc.size());
    std::vector<std::size_t> pimc_ordinals(cfg_.pimc.size());
    std::vector<char> qmc_seen(cfg_.qmc.size()), pimc_seen(cfg_.pimc.size());
    if (sampling_order.size() != cfg_.qmc.size() + cfg_.pimc.size())
        throw std::runtime_error("sampling order must reference every sampling stage exactly once");
    for (std::size_t ordinal = 0; ordinal < sampling_order.size(); ++ordinal) {
        std::visit([&](const auto& stage) {
            using Stage = std::decay_t<decltype(stage)>;
            auto& ordinals = [&]() -> std::vector<std::size_t>& {
                if constexpr (std::is_same_v<Stage, io::QmcStageIndex>) return qmc_ordinals;
                else return pimc_ordinals;
            }();
            auto& seen = [&]() -> std::vector<char>& {
                if constexpr (std::is_same_v<Stage, io::QmcStageIndex>) return qmc_seen;
                else return pimc_seen;
            }();
            if (stage.index >= ordinals.size() || seen[stage.index])
                throw std::runtime_error("invalid or duplicate sampling order index");
            seen[stage.index] = 1;
            ordinals[stage.index] = ordinal;
        }, sampling_order[ordinal]);
    }
    // Mixed/PIMC initializers are indexed by sampling order, not optimizer count,
    // so removing optimizers from a snapshot cannot change a replay's streams.
    // Every sampling stage gets its own stage seed, and its samplers hash their
    // streams from it, so two VMC or DMC stages of one run (e.g. a time-step
    // series) never share random numbers.
    auto sampling_seed = [&](std::size_t ordinal) {
        return cfg_.seed + (cfg_.pimc.empty() ? pack.drivers.size() : ordinal) + 1;
    };

    // ── VMC / DMC drivers ─────────────────────────────────────────────────
    for (std::size_t qi = 0; qi < cfg_.qmc.size(); ++qi) {
        const io::QmcSpec& spec = cfg_.qmc[qi];
        const std::size_t nw = static_cast<std::size_t>(spec.samples);
        const std::uint64_t stage_seed = sampling_seed(qmc_ordinals[qi]);

        auto ensemble = std::make_shared<WalkerEnsemble>(nw, groups, dim, cell);
        initialize_positions(*ensemble, cfg_.lattice.L, stage_seed, at_origin, cell,
                             anchors.pos, anchors.jitter, partners);

        // Walker checkpoint in: overwrite the fresh uniform seed with a saved
        // population (adiabatic continuation / run resume).  Counts may differ —
        // cycle through the saved rows; duplicates decorrelate during
        // equilibration.  Geometry mismatch is fatal: silently falling back to
        // random positions would corrupt a continuation's physics.
        if (!spec.walkers_in.empty()) {
            const io::WalkerFile wf = io::read_walkers(spec.walkers_in);
            if (wf.np != ensemble->n_particles() || wf.dim != ensemble->dim())
                throw std::runtime_error(
                    "walkers_in: geometry mismatch: file has np="
                    + std::to_string(wf.np) + " dim=" + std::to_string(wf.dim)
                    + ", run needs np=" + std::to_string(ensemble->n_particles())
                    + " dim=" + std::to_string(ensemble->dim()));
            // A population sampled in a different cell was drawn at a different
            // density; resuming from it would silently mix two systems.
            // Open cells write no lattice attribute, so compare against an
            // empty span there and the check degenerates to "both absent".
            const std::span<const double> cellv =
                cell.is_periodic() ? cell.vectors() : std::span<const double>{};
            const bool cell_mismatch =
                (wf.lattice.size() != cellv.size()) ||
                [&] {
                    for (std::size_t k = 0; k < wf.lattice.size(); ++k)
                        if (std::abs(wf.lattice[k] - cellv[k]) >
                            1e-9 * (1.0 + std::abs(cellv[k])))
                            return true;
                    return false;
                }();
            if (cell_mismatch)
                throw std::runtime_error(
                    "walkers_in: simulation-cell mismatch between the checkpoint "
                    "and this run — the saved population was sampled at a "
                    "different density or cell shape.");
            seed_walker_rows(*ensemble, std::span<const double>(wf.positions)
                                            .subspan(0, wf.nw * wf.np * wf.dim));
            io::Logger::milestone("walkers: seeded "
                                  + std::to_string(ensemble->num_walkers())
                                  + " walkers from " + spec.walkers_in
                                  + " (" + std::to_string(wf.nw) + " stored)");
        }
        pack.walkers_in_set.push_back(spec.walkers_in.empty() ? 0 : 1);

        if (spec.method == "dmc") {
            DMCConfig dmc_cfg = build_dmc_config(spec, stage_seed);
            auto driver = std::make_unique<DMC>(dmc_cfg, *ensemble, *pack.psi,
                                                *pack.hamiltonian);
            // The mixed observables, and with forward walking their pure
            // copies, one set per lag, keyed apart from the mixed ones.
            auto mixed = build_estimators(
                spec.estimators, EstimatorContext{groups, cell, dim, Sampler::dmc, false});
            if (spec.forward_walking)
                driver->set_forward_walking_estimators(build_forward_walking_estimators(
                    spec.estimators,
                    EstimatorContext{groups, cell, dim, Sampler::dmc, /*forward_walking=*/true},
                    spec.forward_walking->lags.size(), mixed));
            if (!mixed.empty()) driver->set_estimators(std::move(mixed));
            // Milestone checkpoints (eq/a25/a50/a75) written next to the
            // final walkers_out path: "wk.h5" -> "wk.eq.h5" etc.  Crash
            // recovery + population-evolution diagnostics; the final
            // population is still written by main after run().
            //
            // `walkers_out` in the YAML is a bare filename (no directory) —
            // checkpoint_dir supplies the "checkpoints/<project>/<run>/" tree
            // it actually lands in, so the config never has to know the
            // timestamped run directory.
            std::string resolved_out = spec.walkers_out;
            if (!resolved_out.empty() && !checkpoint_dir.empty())
                resolved_out = (checkpoint_dir / spec.walkers_out).string();
            if (!resolved_out.empty()) {
                const std::filesystem::path base(resolved_out);
                driver->set_checkpoint_hook(
                    [base](const WalkerSet& ws, std::string_view tag,
                           int blocks_done) {
                        std::filesystem::path p(base);
                        p.replace_extension("." + std::string(tag)
                                            + p.extension().string());
                        io::write_walkers(p, ws.n_particles(), ws.dim(),
                                          ws.num_walkers(), ws.positions(),
                                          tag, blocks_done,
                                          ws.cell().is_periodic()
                                              ? ws.cell().vectors()
                                              : std::span<const double>{});
                    });
            }
            pack.drivers.push_back(std::move(driver));
            pack.walkers_out.push_back(resolved_out);
        } else {
            if (!spec.walkers_out.empty())
                io::Logger::warn("walkers_out ignored for method '" + spec.method
                                 + "' (only dmc saves its final population)");
            VMCConfig vmc_cfg = build_vmc_config(spec, stage_seed);
            auto driver = std::make_unique<VMC>(vmc_cfg, *ensemble, *pack.psi,
                                                *pack.hamiltonian);
            attach_estimators(*driver, spec.estimators, Sampler::vmc, groups, dim, cell);
            pack.drivers.push_back(std::move(driver));
            pack.walkers_out.emplace_back();
        }
        pack.ensembles.push_back(std::move(ensemble));
    }

    // ── PIMC drivers ────────────────────────────────────────────────────
    // Each replica starts as a ring collapsed onto a walker of a real
    // WalkerEnsemble: the ensemble also receives a
    // previous driver's carry-over, so VMC -> PIMC "just works".
    if (!cfg_.pimc.empty() &&
        (cfg_.magnetic_field || cfg_.rotation_omega))
        throw std::runtime_error(
            "pimc: magnetic_field / rotation put a vector potential in the "
            "kinetic term, which the primitive action cannot carry (it would "
            "need a phase action).  Not supported.");

    for (std::size_t pi = 0; pi < cfg_.pimc.size(); ++pi) {
        const io::PimcSpec& spec = cfg_.pimc[pi];
        const std::size_t nw = static_cast<std::size_t>(spec.samples);

        auto ensemble = std::make_shared<WalkerEnsemble>(nw, groups, dim, cell);
        // Own hashed stream: the stage seed itself also keys the replica
        // streams, and base + offset seeds would coincide with VMC/DMC's
        // base + thread streams.
        const std::uint64_t stage_seed = sampling_seed(pimc_ordinals[pi]);
        initialize_positions(*ensemble, cfg_.lattice.L,
                             numerics::stream_seed(stage_seed,
                                                   kPimcEnsembleInitStream, 0),
                             at_origin, cell, anchors.pos, anchors.jitter, partners);

        Action action = build_action(cfg_, groups, *pack.hamiltonian, dim, pack.psi.get());
        PimcConfig pc = build_pimc_config(spec, stage_seed);
        pc.exchange_class_max = exchange_class_max;
        pc.exchange_classes = exchange_classes;

        auto driver = std::make_unique<Pimc>(pc, groups, *pack.hamiltonian,
                                             cell, dim, n_particles,
                                             ensemble.get(), &action);
        attach_estimators(*driver, spec.estimators, Sampler::pimc, groups, dim, cell);
        pack.drivers.push_back(std::move(driver));
        pack.ensembles.push_back(std::move(ensemble));
        pack.walkers_out.emplace_back();   // pimc does not save a population
        pack.walkers_in_set.push_back(0);
    }

    pack.paths_in.resize(pack.drivers.size());
    pack.paths_out.resize(pack.drivers.size());
    for (std::size_t pi = 0; pi < cfg_.pimc.size(); ++pi) {
        const auto index = cfg_.optimizers.size() + cfg_.qmc.size() + pi;
        pack.paths_in[index] = cfg_.pimc[pi].paths_in;
        pack.paths_out[index] = cfg_.pimc[pi].paths_out;
        // Like walkers_out, a bare paths_out name lives in the paired run's
        // checkpoint directory. An explicit path remains under caller control.
        if (!pack.paths_out[index].empty() && !checkpoint_dir.empty() &&
            !std::filesystem::path(pack.paths_out[index]).has_parent_path())
            pack.paths_out[index] =
                (checkpoint_dir / pack.paths_out[index]).string();
        // Explicit full paths win over incoming VMC configurations just as an
        // explicit walkers_in checkpoint does for VMC/DMC.
        pack.walkers_in_set[index] = !cfg_.pimc[pi].paths_in.empty();
    }

    // Reorder all parallel metadata with the drivers. main's adjacent carry-over
    // then follows YAML sampling order, while every optimizer remains first.
    if (!cfg_.pimc.empty()) {
        std::vector<std::size_t> execution_order;
        for (std::size_t i = 0; i < cfg_.optimizers.size(); ++i)
            execution_order.push_back(i);
        for (const auto& stage : sampling_order)
            execution_order.push_back(std::visit([&](const auto& index) {
                using Stage = std::decay_t<decltype(index)>;
                if constexpr (std::is_same_v<Stage, io::QmcStageIndex>)
                    return cfg_.optimizers.size() + index.index;
                else
                    return cfg_.optimizers.size() + cfg_.qmc.size() + index.index;
            }, stage));
        auto reorder = [&](auto& items) {
            using Items = std::decay_t<decltype(items)>;
            Items ordered;
            ordered.reserve(items.size());
            for (auto index : execution_order) ordered.push_back(std::move(items[index]));
            items = std::move(ordered);
        };
        reorder(pack.drivers);
        reorder(pack.ensembles);
        reorder(pack.walkers_out);
        reorder(pack.walkers_in_set);
        reorder(pack.paths_in);
        reorder(pack.paths_out);
    }
    // A PIMC path checkpoint and a DMC walker checkpoint are distinct formats
    // even if both use .h5. DMC also writes eq/a25/a50/a75 beside its final
    // checkpoint; include those destinations in the preflight for either
    // execution order, before any stage can replace another stage's file.
    // Legacy-only DMC output admission and intentional same-stage in/out stay.
    if (!cfg_.pimc.empty()) {
        auto milestone_path = [](const std::filesystem::path& base,
                                 const char* tag) {
            std::filesystem::path path(base);
            path.replace_extension("." + std::string(tag) + path.extension().string());
            return path.lexically_normal();
        };
        for (std::size_t i = 0; i < pack.drivers.size(); ++i) {
            const std::string& target = pack.paths_out[i].empty()
                ? pack.walkers_out[i] : pack.paths_out[i];
            if (target.empty()) continue;
            const auto resolved = std::filesystem::absolute(target).lexically_normal();
            for (std::size_t j = 0; j < i; ++j) {
                const std::string& previous = pack.paths_out[j].empty()
                    ? pack.walkers_out[j] : pack.paths_out[j];
                if (previous.empty()) continue;
                const auto old = std::filesystem::absolute(previous).lexically_normal();
                bool collision = resolved == old;
                for (const char* tag : {"eq", "a25", "a50", "a75"}) {
                    if (!pack.walkers_out[i].empty())
                        collision |= milestone_path(resolved, tag) == old;
                    if (!pack.walkers_out[j].empty())
                        collision |= milestone_path(old, tag) == resolved;
                }
                if (collision)
                    throw std::runtime_error("pimc: duplicate checkpoint output path '" +
                                             target + "' across stages");
            }
        }
    }
    return pack;
}

void seed_walker_rows(WalkerEnsemble& ensemble, std::span<const double> rows) {
    const std::size_t row = ensemble.n_particles() * ensemble.dim();
    const std::size_t n_rows = row > 0 ? rows.size() / row : 0;
    if (n_rows == 0)
        throw std::runtime_error("seed_walker_rows: no configurations to seed from");
    auto walkers = ensemble.walkers();
    for (std::size_t w = 0; w < walkers.size(); ++w)
        walkers[w].set_positions(rows.subspan((w % n_rows) * row, row));
}

std::size_t carry_over_walkers(DriverPack& pack, std::size_t from, std::size_t to) {
    if (from >= pack.drivers.size() || to >= pack.drivers.size()) return 0;
    if (pack.walkers_in_set[to]) return 0;  // explicit walkers_in wins
    // Defensive: a driver without an ensemble has nothing to seed or receive
    // (PIMC now owns a real ensemble; its final_configurations() is empty by
    // design, so it hands nothing on — the rows check below returns 0).
    if (!pack.ensembles[from] || !pack.ensembles[to]) return 0;

    const std::vector<double> rows = pack.drivers[from]->final_configurations();
    WalkerEnsemble& target = *pack.ensembles[to];
    const std::size_t row = target.n_particles() * target.dim();
    if (rows.empty() || row == 0 || rows.size() % row != 0) return 0;

    seed_walker_rows(target, rows);
    pack.drivers[to]->reload_configurations();

    const std::size_t n_rows = rows.size() / row;
    io::Logger::milestone("walkers: carried " + std::to_string(n_rows)
                          + " configurations into the next driver ("
                          + std::to_string(target.num_walkers()) + " walkers)");
    return n_rows;
}

}  // namespace alea
