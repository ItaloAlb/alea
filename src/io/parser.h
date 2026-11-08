// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace alea::io {

// Simulation cell.
//
//   type: open      — no cell (every pre-existing config).  `L` keeps its
//                     historical meaning: the half-width of the initial walker
//                     scatter box, nothing more.
//   type: periodic  — a Bravais cell, supplied in exactly one of three ways:
//                       shape + L    square | rectangular | hexagonal
//                       vectors      explicit primitive vectors
//                       rs           density-driven; the cell area follows from
//                                    the particle count, A = N π rs² in 2-D, so
//                                    ρ = N/V is set directly.  This is the knob
//                                    held fixed while N is pushed toward the
//                                    thermodynamic limit.
struct LatticeSpec {
    std::string type = "open";   // "open" | "periodic"
    int         dimension = 0;
    double      L = 0.0;         // open: initial scatter half-width.
                                 // periodic: lattice constant (isotropic form).
    std::string shape;           // "square" | "rectangular" | "hexagonal"
    std::vector<double> lengths; // periodic + rectangular: per-axis lengths
    std::vector<double> vectors; // periodic: primitive vectors, row-major [i*dim+d]
    double      rs = 0.0;        // periodic: density parameter (>0 selects it)
    // Long-range terms (Coulomb, Rytova-Keldysh, screened Coulomb) are rejected
    // under `type: periodic` because their lattice sum is only conditionally
    // convergent and needs Ewald.  Setting this to "minimum_image" is an
    // explicit opt-in to truncating them at the nearest image instead — useful
    // for testing the geometry, NOT a physically converged interaction.
    std::string truncation;      // "" | "minimum_image"
    // Relative accuracy of the Ewald sum.  Mirrors screened_coulomb's fit_tol.
    // It lives on the lattice rather than on a term because a periodic run has
    // exactly ONE Ewald term: the reciprocal half couples every charge through
    // a single structure factor, so it cannot be split per group pair.
    double      ewald_tol = 1.0e-6;
    // periodic, several layers: the mean-field (k = 0) energy between layers.
    // "include" keeps the parallel-plate energy of the layers' net charges and
    // needs a neutral cell; "exclude" drops it, as if every layer were
    // neutralized in its own plane (CASINO's convention), which also defines a
    // charged multilayer cell.
    std::string capacitor = "include";
    // periodic: twist θ = Σ_i t_i b_i in FRACTIONAL reciprocal coordinates —
    // the twisted boundary condition ψ(r + a_i) = e^{2πi t_i} ψ(r).  Empty or
    // integer → Γ (real orbitals); otherwise plane-wave determinants become
    // complex and DMC runs fixed phase.  One twist per run; twist averaging
    // repeats the run over a grid of t and averages.
    std::vector<double> twist;
};

struct ParticleSpec {
    std::string         name;
    std::string         group;
    std::string         type;            // e.g. "fermions"
    std::vector<double> mass = {1.0};    // scalar → isotropic; [mx,my,…] → anisotropic
    double              charge = 0.0;
    int                 up = 0;
    int                 down = 0;
    std::string         initialization;  // e.g. "uniform"
    // Fixed out-of-plane (layer) coordinate.  The in-plane lattice.dimension is
    // the dynamical space; z places the particle in a parallel plane.  Physically
    // a per-group/layer property — particles of one group share it.  Default 0
    // (monolayer).  A pair's layer offset is z_a − z_b (e–h across layers →
    // √(r²+(z_e−z_h)²); same plane → r).
    double              z = 0.0;
    // Per-flavour counts (spin × valley × …) in flavour order; empty → the
    // two-flavour form {up, down}.  YAML gives one or the other.  A
    // flavour-blind Hamiltonian conserves each count, so polarization is an
    // input.  PIMC-only for now: determinants and Jastrow cusps read up/down.
    // Last member, so positional aggregate initializers stay valid.
    std::vector<int>    flavours = {};

    // Counts in flavour order: `flavours`, or {up, down}.  Particle indices of
    // the entry follow this order (flavour 0 first, so spin-up first).
    [[nodiscard]] std::vector<int> flavour_counts() const {
        return flavours.empty() ? std::vector<int>{up, down} : flavours;
    }
    // Number of particles the entry declares.
    [[nodiscard]] int count() const noexcept {
        if (flavours.empty()) return up + down;
        int total = 0;
        for (int n : flavours) total += n;
        return total;
    }
};

// A pair Jastrow.  Its Kato cusp is NEVER a YAML input: the builder derives it
// from the Hamiltonian, Γ = 2μR/(2ℓ+d−1), and pins it.  Every other coefficient
// is a variational parameter.
struct JastrowSpec {
    // "pade" | "padé" | "drummond" | "zero_energy".
    std::string                type;
    std::array<std::string, 2> pair;          // particle-group names
    // `pairs:` — one u(r) shared by several group pairs (drummond only), each
    // [group_a, group_b].  Non-empty replaces `pair`; read both through
    // jastrow_pairs().
    std::vector<std::array<std::string, 2>> pairs;

    // Padé u(r) = (a₁r + a₂r² + …)/(1 + b₁r + …), a₁ = Γ:
    //   numerator   = [a₂..a_N]   (`numerator:` in YAML)
    //   denominator = [b₁..b_M]   (`denominator:` in YAML)
    std::vector<double>        numerator;
    std::vector<double>        denominator;

    // Drummond: params = [α0, α2..αN] (α1 follows from Γ, α0 and L).
    std::vector<double>        params;

    // One flag per free coefficient: Padé numerator-then-denominator, Drummond
    // params order.
    std::vector<bool>          is_fixed;
    std::vector<bool>          can_flip_sign;

    // Drummond-only.  The cutoff length L is a separate (optionally optimized)
    // variational parameter.
    int                        truncation   = 3;      // C (≥2); 3 → smooth E_L at L
    double                     cutoff       = 0.0;    // initial cutoff length L
    bool                       cutoff_fixed = false;  // optimize L unless true

    // zero_energy-only.  K₀(κ r^{−s}) inside the matching radius R_m, a phonon
    // tail out to `cutoff` r_c (fixed; the Wigner–Seitz inradius by default
    // under periodic boundaries, required in an open cell).  κ and s come from
    // the Hamiltonian's C/r^p core on the pair, never from the config.
    double                     matching_radius       = 0.0;   // R_m
    bool                       matching_radius_fixed = true;  // optimize R_m if false
};

// The group pairs a jastrow item covers: `pairs` when given, else the single `pair`.
inline std::vector<std::array<std::string, 2>> jastrow_pairs(const JastrowSpec& j) {
    if (!j.pairs.empty()) return j.pairs;
    return {j.pair};
}

// "[a, b]" or "[a, b], [c, d]" for messages.
inline std::string jastrow_pairs_label(const JastrowSpec& j) {
    std::string label;
    for (const auto& p : jastrow_pairs(j))
        label += (label.empty() ? "[" : ", [") + p[0] + ", " + p[1] + "]";
    return label;
}

// One backflow channel of a slater item (wavefunction/backflow_slater_determinant.h):
// the particles of group `from` displace the determinant's quasi-coordinates by
// η(r) r, η(r) = (1 − r/L)^C Θ(L − r) Σ_k c_k r^k.  `params` lists the free
// coefficients: [c_0, c_2, ..] when `from` is the determinant's own group (a
// like-spin channel, c_1 = C c_0/L derived), [c_0, c_1, c_2, ..] otherwise.
// `cutoff` L: required in an open cell, the Wigner–Seitz inradius by default
// under periodic boundaries (and never beyond it).
struct BackflowSpec {
    // pairing items only: the row or column group the channel displaces (a
    // slater item displaces its own group and refuses the key).
    std::string           group;
    std::string           from;
    int                   truncation = 3;       // C (≥ 2; 3 keeps E_L continuous at L)
    std::optional<double> cutoff;
    bool                  cutoff_fixed = false;  // optimize L unless true
    std::vector<double>   params;
    std::vector<bool>     is_fixed;              // one per params entry; absent → all free
};

// Slater determinant trial factor (fermionic antisymmetry).  Declared under
// `wavefunction:` as `- slater: { group, orbitals }`.  ψ_det = det(D_up)·det(D_down)
// over the named group; the up/down block sizes come from that group's particle
// `up`/`down` counts (indices [first,first+up) are spin-up, the rest spin-down).
// Orbitals are a single shared (restricted) pool: the up/down blocks use the
// first `up`/`down` orbitals, so `orbitals` must hold at least max(up,down) of them.
struct SlaterSpec {
    std::string                      group;                  // particle-group name
    std::string                      orbital_type = "gaussian";  // "gaussian" | "plane_wave"
    std::vector<std::vector<double>> centers;                // [n_orb][dim]
    std::vector<double>              alphas;                 // [n_orb]
    // Generated orbital centers: "triangular" builds the N sites of a triangular
    // lattice commensurate with a hexagonal cell (the Wigner-crystal phase), so
    // the config does not have to spell out N centers.  Mutually exclusive with
    // `centers`.  `alpha` is then the single width shared by every site.
    std::string                      sites;
    std::optional<double>            alpha;
    // Hand the shared Gaussian width to the optimiser (`alpha_fixed: false`).
    // Defaults to fixed — α moves the nodes, so it is opted into deliberately.
    // Only a uniform-width Gaussian set has one; other cases are refused.
    bool                             alpha_fixed = true;
    // fock_darwin: angular momenta of the n = 0 orbitals (the pool the spin
    // blocks fill from the front, like `alphas`) and the Gaussian exponent β
    // (= mΩ for the exact one-body state).  See FockDarwinOrbitalSet.
    std::vector<int>                 ls;
    std::optional<double>            beta;
    // plane_wave: an explicit occupation at Γ, one orbital per entry: integer
    // reciprocal indices [n1, n2] of the cell and cos (false) or sin (true).
    // Empty → the closed shells of the cell.  Lets a group fill an open shell
    // with a chosen subset of its degenerate cos/sin.
    std::vector<std::vector<int>>    pw_k;
    std::vector<bool>                pw_sin;
    // `backflow:` channels (empty → the plain determinant).  gaussian and
    // plane_wave (at Γ or a twist); the builder refuses fock_darwin.
    std::vector<BackflowSpec>        backflow;
};

// One-body parabolic confinement V = Σ ½ m ω² |r − center|² on `groups`
// (empty → every particle), declared as `{type: harmonic_trap, omega, ...}`.
struct HarmonicTrapSpec {
    std::vector<std::string> groups;
    double                   omega = 0.0;
    std::vector<double>      center;    // empty → origin
};

// One-body moiré Jastrow (wavefunction/moire_jastrow.h), declared under
// `wavefunction:` as `- moire_jastrow: {group, shells, params, is_fixed}`.
// log ψ = Σ_i Σ_s Σ_j [a_s cos(k_sj·r_i) + b_s sin(k_sj·r_i)] over the first
// `shells` moiré reciprocal shells.  a_M / theta0 default to the Hamiltonian's
// `moire` term, so the factor sits on the same superlattice as the potential.
struct MoireJastrowSpec {
    std::string           group;
    int                   shells = 1;
    std::vector<double>   params;      // [a_1, b_1, a_2, b_2, ...]; empty → zeros
    std::vector<bool>     is_fixed;    // default: fixed, like every jastrow
    std::optional<double> a_M;
    std::optional<double> theta0;
};

// Nosanow one-body localization factor (wavefunction/nosanow_jastrow.h),
// declared as `- nosanow: { group, sites: triangular | centers, alpha,
// alpha_fixed }`.  log ψ = −α Σ_i |r_i − R_i|², particle i of the group on
// site i: generated (`sites: triangular`, commensurate with a hexagonal cell)
// or explicit `centers`, one per particle.
struct NosanowSpec {
    std::string                      group;
    std::string                      sites;        // "triangular", or empty
    std::vector<std::vector<double>> centers;      // [n][dim], or empty
    double                           alpha = 0.0;
    bool                             alpha_fixed = true;  // optimize α if false
};

// Long-range reciprocal-space Jastrow (wavefunction/kspace_jastrow.h), declared
// as `- kspace_jastrow: { pair: [a, b], shells: n, params: [p_1..p_n] }`.  One
// coefficient per |k| shell of the cell; `params` absent → seeded from the
// Gaskell RPA form with the Hamiltonian's own Ṽ(k) (same-group pairs; a two-
// group pair starts at zero).  Periodic cells only.
struct KSpaceJastrowSpec {
    std::array<std::string, 2> pair;
    int                        shells = 0;
    std::vector<double>        params;
    std::vector<bool>          is_fixed;        // default: fixed, like every jastrow
    std::vector<bool>          can_flip_sign;
};

// Galilean boost ψ → e^{iK·R_cm}ψ (wavefunction/center_of_mass_momentum.h),
// declared as `- center_of_mass_momentum: { K: [Kx, Ky] }`.  Makes the trial
// complex, so DMC runs fixed-phase.  At most one per config.
struct CenterOfMassMomentumSpec {
    std::vector<double> K;
};

// One entry of the `wavefunction:` list.  The list keeps document order, and
// build_wavefunction turns item i into component i: the optimiser's parameter
// vector and the config snapshot both rely on that one-to-one mapping.
// One determinant of a `pairing` item: rows from one group, columns from
// another, and the single-particle orbitals that complete an unbalanced block
// (wavefunction/pairing_determinant.h).
struct PairingBlockSpec {
    std::string                   rows;
    std::string                   columns;
    std::string                   unpaired_type;   // "plane_wave", or empty for none
    std::vector<std::vector<int>> unpaired_k;      // integer reciprocal indices, one per orbital
    std::vector<bool>             unpaired_sin;    // sin(k·r) instead of cos, per orbital
};

// Pairing determinant, declared as `- pairing: { blocks: [...], orbital: {...} }`.
// ψ = Π_b det φ(r_row − r_column) with one pair orbital φ shared by every
// block (wavefunction/pair_orbital.h):
//   φ = Σ_s p_s Σ_{k∈star s} cos(k·Δ) + (1 − r/L)³ Θ(L − r) Σ_m c_m r^m.
struct PairingSpec {
    std::vector<PairingBlockSpec> blocks;
    std::string         orbital_type = "cosine_polynomial";
    int                 shells       = 0;     // stars of the cosine part (k = 0 first)
    int                 poly_order   = -1;    // M ≥ 1, or −1 for no polynomial
    double              cutoff       = 0.0;   // L; 0 → the Wigner–Seitz inradius
    bool                cutoff_fixed = false; // optimize L unless true
    std::vector<double> params;               // [p_0..p_{S−1}, c_0, c_2..c_M]
    std::vector<bool>   is_fixed;             // same order; fixed unless false
    // `backflow:` channels {group, from, ...} (empty → no backflow); every
    // row and column quasi-coordinate may be displaced.
    std::vector<BackflowSpec> backflow;
};

using WavefunctionItem = std::variant<JastrowSpec, SlaterSpec, MoireJastrowSpec,
                                      CenterOfMassMomentumSpec, KSpaceJastrowSpec,
                                      NosanowSpec, PairingSpec>;

// The items of one kind, in document order (pointers into `items`).
template <class Spec>
[[nodiscard]] std::vector<const Spec*> items_of(
    const std::vector<WavefunctionItem>& items) {
    std::vector<const Spec*> out;
    for (const WavefunctionItem& item : items)
        if (const Spec* s = std::get_if<Spec>(&item)) out.push_back(s);
    return out;
}

struct CoulombSpec {
    std::array<std::string, 2> pair;
    double                     epsilon = 1.0;
};

struct KeldyshSpec {
    std::array<std::string, 2> pair;
    double                     epsilon = 1.0;
    double                     r0      = 1.0;  // 2D screening length
};

// `type: inverse_power` — V = strength / r^exponent between the pair's groups
// (exponent 3: dipoles polarized perpendicular to the plane).  Under PBC each
// pair is taken at its minimum image and a continuum tail outside the
// Wigner–Seitz cell is added (hamiltonian/inverse_power.h); it is never part
// of the Ewald sum.
struct InversePowerSpec {
    std::array<std::string, 2> pair;
    double                     strength = 0.0;  // C, energy × length^exponent
    double                     exponent = 0.0;  // p
};

// Screened Coulomb for an arbitrary multilayer dielectric stack, evaluated by the
// fast-screened-coulomb library (ETM method, Cavalcante PRB 97, 125427 (2018)).
// Each term self-contains the whole stack (per-layer ε + interface positions) and
// its source/observation layer.  Rytova–Keldysh is the single-layer special case.
struct ScreenedCoulombSpec {
    std::array<std::string, 2> pair;
    std::vector<double>        epsilons;           // per-layer ε, bottom → top (length N)
    std::vector<double>        interfaces;         // N-1 interface positions (Bohr)
    int                        source_layer = 1;   // c, charge layer (1-based)
    int                        target_layer = 0;   // t, observation layer (1-based); 0 → direct (t=c)
    double                     z_offset     = 0.0; // z_t, out-of-plane offset for indirect excitons
    bool                       has_z_offset = false;  // false → derive from particle z
    double                     fit_tol      = 0.1; // % sampled value and force targets; both required
};

// One-body moiré potential V(r) = V0 + 2 Ṽ Σ_{i=1}^3 cos(g_i·r + φ), applied to
// every particle of the listed groups.  The three reciprocal vectors g_i are the
// first shell of a triangular moiré superlattice of period a_M, oriented by
// theta0 (|g| = 4π/(√3 a_M), 120° apart).
struct MoireSpec {
    std::vector<std::string> groups;        // particle-group names this term acts on
    double                   V0     = 0.0;   // constant offset
    double                   Vtilde = 0.0;   // modulation amplitude Ṽ
    double                   phi    = 0.0;   // phase φ
    double                   a_M    = 1.0;   // moiré period
    double                   theta0 = 0.0;   // orientation of the g_i (optional)
};

// Opt-in observable estimators (besides the always-on energy), declared under a
// stage's `estimators:` list.  Absent → not computed, exactly like QMCpack's
// optional <estimator> elements.  One spec per kind, holding only that kind's
// keys; the kind's builder (builders/estimator_builders.h) owns its rules.
//
// Every kind takes an optional `name` that labels its result keys
// (gofr_<name>_r, ...), so one stage can hold several instances of a kind;
// an unnamed repeat is numbered instead (gofr_2_r).  collective_density's name
// is required and is its namespace already (cdensity_<name>).

// `type: gofr` — pair correlation g(r) of every group pair on [0, rmax).
struct GofrSpec {
    std::string name;
    double      rmax   = 0.0;
    int         n_bins = 0;
};

// `type: density` — one-body density per group on a grid of half-extent rmax
// (n_bins per axis).  per_particle: one histogram per particle
// (`<group>_p<i>`) instead of one per group — resolves spin channels inside a
// group (labels of same-spin particles are exchangeable, so use it via
// channel sums/differences).
struct DensitySpec {
    std::string name;
    double      rmax         = 0.0;
    int         n_bins       = 0;
    bool        per_particle = false;
};

// `type: sofk` — static structure factor S(k) on the cell's reciprocal
// shells with |k| <= kmax (inverse length).  Periodic cells only.
struct StructureFactorSpec {
    std::string name;
    double      kmax = 0.0;
};

// `type: collective_density` — histogram of the linear combination
// v = Σ coeff·Σ_{p∈group} r_p, given as (group, coeff) pairs (a missing coeff
// parses as 1.0), on a grid of half-extent rmax.  Open cells only.
struct CollectiveDensitySpec {
    std::string name;
    double      rmax   = 0.0;
    int         n_bins = 0;
    std::vector<std::pair<std::string, double>> terms;
};

// `type: electron_hole_proximity` — U_e, the fraction of electron
// configurations with no hole of `hole_group` inside r_cut (a geometric
// diagnostic: no assignment of bound pairs).
struct ElectronHoleProximitySpec {
    std::string name;
    std::string electron_group;
    std::string hole_group;
    double      r_cut = 0.0;
};

// `type: momentum_distribution` — one-body density matrix n(r) on [0, rmax)
// and momentum distribution n(k) up to kmax of `group`, sampled on the open
// paths of a PIMC worm (PIMC-only).  n_k: open-cell k grid size (0 under PBC);
// samples: extra-bead samples per open measurement.
struct MomentumDistributionSpec {
    std::string name;
    std::string group;
    double      rmax    = 0.0;
    int         n_bins  = 0;
    double      kmax    = 0.0;
    int         n_k     = 0;
    int         samples = 0;
};

// `type: bragg` — selected-star crystal order
// B = (1/n_G) Σ_G ⟨|Σ_{j∈group} e^{iG·r_j}|²⟩/N² for explicit Cartesian
// reciprocal vectors G of a periodic 2-D cell, flattened [direction, 2] (one
// member of each ±G pair).
struct BraggSpec {
    std::string name;
    std::string group;
    std::vector<double> wavevectors;
};

// `type: bond_order` — six-nearest sixfold bond order of `group` in a
// periodic 2-D cell (global and local |ψ₆|², and the bond-length distortion
// relative to reference_spacing, a length > 0).
struct BondOrderSpec {
    std::string name;
    std::string group;
    double      reference_spacing = 0.0;
};

// `type: density_matrix` — one-body density matrices of `groups`, and for each
// channel of `pairs` (two different groups) the two-body density matrix with
// one particle of each group displaced together and the pair condensate
// fraction, on [0, rmax) in n_bins, from ratios of ψ_T at a virtual
// displacement of every configuration, at every `every`-th measurement
// (VMC and DMC only).
struct DensityMatrixSpec {
    std::string name;
    std::vector<std::string> groups;
    std::vector<std::pair<std::string, std::string>> pairs;
    double      rmax   = 0.0;
    int         n_bins = 0;
    int         every  = 1;
};

// One entry of an `estimators:` list; build_estimators turns item i into
// estimator i.
using EstimatorItem =
    std::variant<GofrSpec, DensitySpec, StructureFactorSpec, CollectiveDensitySpec,
                 ElectronHoleProximitySpec, MomentumDistributionSpec, BraggSpec,
                 BondOrderSpec, DensityMatrixSpec>;

// Sampling keys shared by every Monte Carlo run under qmc:, with the SAME
// meaning in every method that reads them — one parser (parse_run_keys), so a
// key cannot drift in meaning between drivers.
//   samples    independent Markov chains (walkers; PIMC: replica paths)
//   sub_steps  sweeps between measurements
//   time_step  imaginary-time step τ (DMC propagator, PIMC slice e^{−τH});
//              the proposal scale for VMC Metropolis/Langevin moves
struct RunSpec {
    std::string device;        // "cpu"
    int         threads = 0;   // 0 → all cores
    int         samples = 0;
    int         equilibration_blocks = 0;
    int         accumulation_blocks = 0;
    int         steps_per_block = 0;
    int         sub_steps = 0;
    double      time_step = 0.0;
};

// Pure (forward-walking) estimators of a DMC stage, declared as
// `forward_walking: { lags: [T_1, T_2, ...], interval: n }` (qmc/forward_walking.h).
// The lags are imaginary times, strictly increasing, each a whole number of
// steps of time_step · sub_steps; interval is the number of steps between
// population snapshots and must divide steps_per_block.  Every estimator of
// the stage is then also measured purely at each lag, as is the potential energy.
struct ForwardWalkingSpec {
    std::vector<double> lags;
    int                 interval = 1;
};

struct QmcSpec : RunSpec {
    std::string method;        // "vmc" | "dmc"
    // vmc: "metropolis" (default, also when empty) | "langevin" (drift).
    // dmc: "drift_diffusion" | "langevin" — synonyms for the only DMC move.
    std::string random_walk;

    // DMC-specific (refused for VMC).
    double population_generations = 10.0;  // g: E_T feedback over g steps (gain 1/g)
    // Branch-weight cutoff (qmc/population_control.h): "universal" | "drv" |
    // "none"; alpha of "universal" (empty: the driver's default).
    std::string           branch_cutoff = "universal";
    std::optional<double> branch_cutoff_alpha;
    // Local energy in the branch weight (qmc/population_control.h,
    // Reweighting): "bare" | "effective" | "suppressed"; correlation_time θ of suppressed
    // (empty: measured, `auto`) and its kappa (empty: the driver's default).
    std::string           reweighting = "bare";
    std::optional<double> correlation_time;
    std::optional<double> suppression_kappa;
    std::optional<std::size_t> correlation_equilibration_sweeps;
    std::optional<std::size_t> correlation_min_sweeps;
    std::optional<std::size_t> correlation_max_sweeps;

    // Walker-population checkpoint (io/walker_io.h).  walkers_in seeds the
    // initial ensemble from a saved population (vmc + dmc); walkers_out saves
    // the final population after the run (dmc only — its branched population
    // is the physically meaningful one).  Empty → off.
    std::string walkers_in;
    std::string walkers_out;

    // Opt-in observable estimators (empty → energy only).
    std::vector<EstimatorItem> estimators;

    // DMC only: pure estimators by forward walking (absent → mixed only).
    std::optional<ForwardWalkingSpec> forward_walking;
};

// Parameters for the linear-method wavefunction optimiser.
// Appears under qmc: as {method: optimizer, type: linear, ...}.
struct OptimizerSpec : RunSpec {
    // The optimiser's VMC sampling defaults (the RunSpec keys).
    OptimizerSpec() {
        device               = "cpu";
        samples              = 128;   // number of walkers
        equilibration_blocks = 50;
        accumulation_blocks  = 50;
        steps_per_block      = 10;
        sub_steps            = 1;
        time_step            = 0.01;  // tau
    }

    // ── method identification ──────────────────────────────────────────────
    std::string type = "linear";   // "linear" (only supported type for now)

    // ── VMC sampling ───────────────────────────────────────────────────────
    std::string random_walk = "metropolis";  // "metropolis" | "langevin" (drift)

    // Which parameters this stage may move: "all" (default), "nodal" (only
    // components that set the nodes — determinants), or "jastrow" (everything
    // that does not).  Narrowing never UNfixes what the wavefunction pinned.
    // Two stages let you alternate, which also keeps Hbar from mixing blocks
    // with and without the curvature term.
    std::string parameters = "all";

    // ── optimisation loop ──────────────────────────────────────────────────
    int    max_iterations = 20;
    double psi_step_tol   = 1e-3;            // converged: ‖δψ‖/‖ψ‖ of the undamped step below this,
    double energy_tol     = 1e-6;            // ... or the slope of the last convergence_window
    int    convergence_window = 15;          //     fresh energies ≥ −max(energy_tol, 2σ) per iteration

    // ── adaptive shift (QMCPACK three-shift algorithm) ──────────────────────
    int    num_shifts        = 3;            // candidate shifts per iteration
    double max_param_change  = 0.3;          // max relative parameter change
    double cost_increase_tol = 0.0;          // tolerance for cost increase
    double target_shift_i    = -1.0;         // target shift (negative → disabled)
    int    max_failed_attempts = 5;          // max retries per iteration

    // ── linear-method stabilisation ────────────────────────────────────────
    double shift_i  = 0.1;                   // starting H̄ diagonal shift
    double shift_s  = 0.0;                   // starting overlap shift
    double rescale   = 1.0;                  // step-size factor λ (ξ damps large steps)
    double xi        = 0.5;                  // nonlinear normalisation ξ (Toulouse-Umrigar 2007)
};

// PIMC move specification (deserialized from the qmc: moves: list).
struct PimcMoveSpec {
    // "bisection" | "centroid" | "single_slice" | "permutation" | "joint_permutation" | "worm"
    std::string type;
    double      weight = 1.0;
    int         levels = 3;    // bisection / permutations: 2^levels slices per move
    double      step   = 0.5;  // centroid / single_slice step size
    int         max_cycle = 4; // permutations: longest proposed cycle
    // joint_permutation: the two groups exchanged together (A, B) and the
    // length scale lambda of the A-B pairing bias.
    std::array<std::string, 2> pair = {};
    double      pair_radius = 0.0;
    // worm: the group whose exchange classes it opens, the longest regrowth
    // (slices), the swap length (slices) and the worm constant C (NaN: tuned
    // during equilibration).
    std::string group = {};
    int         max_length = 0;
    int         swap_length = 0;
    double      worm_constant = std::numeric_limits<double>::quiet_NaN();
};

// The move set of a pimc run whose YAML omits `moves:`, for an M-slice ring:
// a bisection as deep as the ring allows (at most 3 levels, none when M = 1)
// plus a centroid translation of step 0.5, weight 1 each.  The single source
// for the driver and the config snapshot, which records the resolved set so
// that a replay never re-derives it from a possibly changed default.
inline std::vector<PimcMoveSpec> default_pimc_moves(std::size_t M) {
    std::vector<PimcMoveSpec> moves;
    int levels = 0;
    while (levels < 3 && (std::size_t{1} << (levels + 1)) <= M) ++levels;
    if (levels >= 1) {
        PimcMoveSpec bisection;
        bisection.type = "bisection";
        bisection.levels = levels;
        moves.push_back(bisection);
    }
    PimcMoveSpec centroid;
    centroid.type = "centroid";
    centroid.step = 0.5;
    moves.push_back(centroid);
    return moves;
}

// Path Integral Monte Carlo run specification.
// Appears under qmc: as {method: pimc, ...}.
struct PimcSpec : RunSpec {
    // The RunSpec keys default to "absent" (0); a pimc run gives them the
    // method's own sensible defaults, exactly like OptimizerSpec — absent
    // keys keep those, present keys win (parse_run_keys).
    PimcSpec() {
        device               = "cpu";
        samples              = 64;    // replica paths
        equilibration_blocks = 100;
        accumulation_blocks  = 500;
        steps_per_block      = 50;
        sub_steps            = 1;
        // time_step stays 0 (required): M = beta/time_step must be an
        // integer, and no default tau is meaningful without beta.
    }

    double      beta = 1.0;        // inverse temperature (M = beta/time_step)
    std::string energy = "thermodynamic";   // estimator type
    bool compare_thermodynamic = false; // paired diagnostic for either virial estimator
    // Statistics mode: empty keeps quantum statistics for every exchange
    // class of more than one particle (the move set must then change the
    // permutation, or the driver refuses); `boltzmannon` makes the particles
    // distinguishable; `fixed_node` samples one nodal pocket of the action's
    // `nodes` restriction.
    std::string statistics;
    // A signed run flags its results when |<s>| < sign_threshold * sigma_s.
    double sign_threshold = 5.0;
    std::vector<PimcMoveSpec>   moves;
    std::vector<EstimatorItem>  estimators;
    std::string paths_in;
    std::string paths_out;
};

// One entry of the top-level `action:` list (shared physics, alongside
// `hamiltonian:`): which piecewise component the Euclidean action includes.
//   kinetic: primitive     free-particle springs (required by any path integral)
//   potential: primitive   tau*V(R_k) from the Hamiltonian's potential terms
//   primitive: {}          shorthand for both (the default when `action:` is
//                          omitted; the snapshot records it explicitly)
struct ActionItemSpec {
    std::string component;   // "kinetic" | "potential" | "primitive" | "pair_product" | "nodes"
    std::string form = "primitive";  // "wavefunction" for nodes
    double extent = 0.0;     // explicit finite-grid relative endpoint half-width
    int grid_points = 17;
};

// Sampling-stage references preserve document order without interleaving the
// optimizers, which retain their existing optimizers-before-sampling contract.
struct QmcStageIndex { std::size_t index; };
struct PimcStageIndex { std::size_t index; };
using SamplingStageIndex = std::variant<QmcStageIndex, PimcStageIndex>;

struct Config {
    std::string               name;
    std::uint64_t             seed   = 0;
    bool                      log    = false;
    bool                      output = true;  // write an HDF5 results file per driver
    LatticeSpec               lattice;
    std::vector<ParticleSpec> particles;
    std::vector<WavefunctionItem> wavefunction;   // document order
    std::vector<CoulombSpec>  coulombs;
    std::vector<KeldyshSpec>  keldysh;
    std::vector<ScreenedCoulombSpec> screened;
    std::vector<InversePowerSpec> inverse_powers;
    std::vector<MoireSpec>    moires;
    std::vector<HarmonicTrapSpec> traps;
    // Uniform fields along z (2-D, open cells), via a vector potential in the
    // kinetic energy (hamiltonian/kinetic.h, GaugeField):
    //   `{type: magnetic_field, B}`   symmetric gauge about the origin
    //   `{type: rotation, omega}`     rotating frame; also adds −½mΩ²r²
    std::optional<double>     magnetic_field;
    std::optional<double>     rotation_omega;
    // Piecewise components of the PIMC Euclidean action (design: shared
    // physics next to hamiltonian:/wavefunction:).  Empty → default
    // (kinetic+potential primitive), recorded explicitly by the snapshot.
    std::vector<ActionItemSpec> action;
    std::vector<QmcSpec>      qmc;
    std::vector<OptimizerSpec> optimizers;
    std::vector<PimcSpec>     pimc;
    std::vector<SamplingStageIndex> sampling_order;
};

// Resolve a configured random seed.  A zero seed (the default when `project.seed`
// is absent or explicitly set to 0) means "pick a fresh seed at startup": it is
// replaced by a non-deterministic value drawn from std::random_device.  Any
// non-zero seed is returned unchanged so runs stay reproducible.  The result is
// always non-zero, so the chosen seed can be recorded in the run snapshot and
// replayed verbatim.
std::uint64_t resolve_seed(std::uint64_t seed);

// Parse a YAML document (text) into a Config. Throws on malformed YAML.
Config parse_config(const std::string& yaml_text);

// Convenience: read and parse a YAML file at `path`.
Config parse_config_file(const std::string& path);

}  // namespace alea::io
