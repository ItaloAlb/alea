// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hamiltonian/hamiltonian_component.h"
#include "numerics/ewald_breakup.h"
#include "numerics/keldysh_kernel.h"
#include "numerics/radial_cutoff_correction.h"
#include "population/particle_group.h"
#include "population/simulation_cell.h"

namespace alea {

// The interaction of one unordered pair of groups (indices into the `groups`
// vector handed to EwaldTerm; a == b is the intra-group pair).
struct EwaldPair {
    std::size_t             a = 0, b = 0;
    numerics::PotentialSpec kernel;
};

// Coulomb energy of a 2-D periodic cell, summed over all periodic images by the
// Ewald method.  See numerics/ewald_breakup.h for the splitting.
//
// ── Why this is ONE term for the whole system ───────────────────────────────
// CoulombTerm is per group-pair, because a pair sum decomposes that way.  An
// Ewald sum does not: the reciprocal half couples every charge at once through
// structure factors.  So one EwaldTerm replaces ALL the pair terms of a
// periodic run.
//
// ── One kernel per group pair ───────────────────────────────────────────────
// Each unordered group pair (g, h) carries its own kernel V_gh, and each
// DISTINCT kernel gets its own breakup.  The reciprocal half then needs one
// structure factor per group, ρ_g(k) = Σ_{i∈g} q_i exp(i k·r_i):
//
//   E_long = Σ_k [ Σ_g w_gg(k) |ρ_g|² + Σ_{g<h} 2 w_gh(k) Re(ρ_g ρ_h*) ]
//
// with w_gh = Ṽ_gh,long(k)/A.  Groups whose kernel rows are identical are
// merged into one structure-factor CLASS first, so the common single-kernel
// case is the plain |ρ(k)|² sum and costs what it always did.  The extra work
// otherwise is O(C²) per k per walker for C classes (C ≤ a handful), against
// the O(N) already spent building ρ — no new parallelism, so strong and weak
// scaling are unchanged.
//
// ── The k = 0 term ──────────────────────────────────────────────────────────
// Every kernel must share the Coulomb tail Ṽ → 2π/(ε k): only then do the 1/k
// divergences of the pairs cancel against overall neutrality.  What they may
// NOT share is the finite remainder at k = 0.  Poisson-summing the long-range
// halves leaves, from k = 0,
//
//   E_k0 = Σ_{g,h} Q_g Q_h [ c_gh/(2A) − ½ sr_k0_gh ]
//
// (Q_g the total charge of group g, c_gh = PotentialSpec::k0_offset, sr_k0 the
// short-range half's mean over the cell).  With a single kernel this is
// Q²(c/2A − sr_k0/2); a neutral cell has Q = 0, and a charged one gets the
// uniform background, which cancels ρ(0) and with it the c term, leaving the
// jellium constant −½Q² sr_k0.  With several kernels it is a real energy — for
// a layer offset it is the parallel-plate capacitor energy of the layers.
class EwaldTerm : public HamiltonianComponent {
public:
    // Throws if the cell is not a periodic 2-D cell, or if the tolerance is out
    // of range.  `groups` supplies the per-particle charges and the index
    // ranges; every particle in the cell must be covered.
    // Plain Coulomb: the dielectric is the constant `epsilon`.
    EwaldTerm(std::vector<ParticleGroup> groups, double epsilon,
              const SimulationCell& cell, double tol);

    // Any potential, supplied as its real-space value plus its dielectric
    // function eps(k).  `cusp_eps` is validated (> 0) and otherwise unused: the
    // cusp residue follows the kernel itself (see coulomb_cusp_residue).
    EwaldTerm(std::vector<ParticleGroup> groups,
              const numerics::PotentialSpec& pot, double cusp_eps,
              const SimulationCell& cell, double tol);

    // One kernel per unordered group pair.  exclude_capacitor drops every
    // kernel's k = 0 remainder c_gh (CASINO's convention: each layer
    // neutralized in its own plane), leaving E_k0 = −½ Σ_{g,h} Q_g Q_h sr_k0_gh;
    // that also defines a charged cell with several kernels, which is refused
    // otherwise.
    EwaldTerm(std::vector<ParticleGroup> groups, std::vector<EwaldPair> pairs,
              const SimulationCell& cell, double tol, bool exclude_capacitor = false);

    std::string name() const override { return "ewald"; }
    bool supports_potential_gradient() const noexcept override { return true; }
    bool supports_periodic_virial(const SimulationCell&, std::size_t) const noexcept override;
    std::string potential_approximation_notes() const override;
    std::unique_ptr<PotentialGradientWorkspace>
    make_potential_gradient_workspace(const ParticleBatch&) const override;
    // Real pairs: q_i*q_j*(V'-V_long'-C')*Delta/r within the strict cutoff.
    // RK's small sampled-kernel residual C matches the last panel to C² zero;
    // its integral also enters the k=0 term. The spline/Fourier weights stay exact.
    // Reciprocal: -2*q_i*sum_k k*Im(exp(ik.r_i)*sum_d w_cd*rho_d*).
    // Stored weights already include half-space folding. Fixed-cell self,
    // background/capacitor constants have zero in-plane derivative.
    void evaluate_potential_gradient(const ParticleBatch&, std::span<double>,
        PotentialGradientWorkspace&, std::size_t = 0, std::size_t = npos) const override;
    bool is_translation_invariant() const noexcept override { return true; }

    // r → 0 limit of r·V for a pair, unchanged by the splitting: V_short still
    // carries the potential's full singularity and the long-range half is
    // smooth there.  So the Jastrow cusp builder sees what it would see for the
    // open-boundary term of that pair's kernel: q_a q_b/ε for in-plane Coulomb;
    // 0 for Rytova-Keldysh (a log, no 1/r pole), for a pair in two different
    // planes (finite at r = 0), and for groups this term does not own.
    double coulomb_cusp_residue(const ParticleGroup& a,
                                const ParticleGroup& b) const override;

    bool primitive_pair_admissible(const ParticleGroup& a,
                                   const ParticleGroup& b) const noexcept override;

    // q_a q_b Ṽ_ab(k) of the pair's own kernel (Coulomb, Rytova–Keldysh,
    // screened, with any layer offset); 0 for groups this term does not own.
    double pair_fourier(const ParticleGroup& a, const ParticleGroup& b,
                        double k) const override;

    void evaluate_potential(const ParticleBatch& batch, std::span<double> v_out,
                            std::size_t w_begin = 0,
                            std::size_t w_end = npos) const override;

    // ΔV of moving particle p (class c) from r to r' in configuration k,
    // exactly evaluate(r') − evaluate(r) without touching the batch:
    //   real space   Σ_j q_p q_j [f_pj(|r'−r_j|) − f_pj(|r−r_j|)],
    //                f = (V − V_long) inside r_cut, 0 beyond   — O(N);
    //   reciprocal   Σ_k [ w_cc (2 Re(δ ρ_c*) + |δ|²) + Σ_{d≠c} 2 w_cd Re(δ ρ_d*) ],
    //                δ(k) = q_p (e^{ik·r'} − e^{ik·r})        — O(N_k);
    //   constants (self, background, capacitor) cancel.
    // ρ_d(k) of configuration k comes from a cache that validates itself: it
    // keeps the positions it was built from and, on every call, compares them
    // with the batch (O(N)); up to kDeltaIncrementalLimit changed particles are
    // folded in incrementally (O(N_k) each), more trigger a rebuild (O(N N_k)),
    // and every kDeltaRebuildEvery increments it is rebuilt to bound rounding
    // drift.  So no move has to notify it, and it cannot go stale.  The default
    // (two full evaluations, O(N² + N N_k)) cost PIMC ~1 ms per bead at N = 16
    // (measured on the charged-boson validation case).
    // Not thread-safe for one instance: PIMC gives each replica its own clone
    // (pimc.cpp), and VMC/DMC never call it.  A clone starts with an empty cache.
    double delta_potential_particle(ParticleBatch& configs, std::size_t p,
        std::size_t k, std::span<const double> proposed,
        std::span<double> scratch) const override;

    [[nodiscard]] std::unique_ptr<HamiltonianComponent> clone() const override {
        return std::make_unique<EwaldTerm>(*this);
    }

    // ── Introspection, for tests and diagnostics ────────────────────────────
    // The first kernel's splitting (the only one, for a single-kernel term).
    [[nodiscard]] const numerics::EwaldSetup& setup() const noexcept {
        return *kernels_.front().setup;
    }
    [[nodiscard]] double self_energy() const noexcept { return self_energy_; }
    [[nodiscard]] const numerics::RadialCutoffCorrection&
    cutoff_correction(std::size_t kernel=0) const { return kernels_.at(kernel).cutoff_correction; }
    // The k = 0 constant E_k0 (see above): the jellium background for a charged
    // single-kernel cell, exactly 0 for a neutral one.
    [[nodiscard]] double background_energy() const noexcept {
        return background_energy_;
    }
    [[nodiscard]] double net_charge() const noexcept { return net_charge_; }
    // Mean-field electrostatic energy of individually charged layers: each
    // group's charge smeared uniformly over its plane, interacting through the
    // pair kernels, (1/2A) Σ_{g,h} Q_g Q_h c_gh with c_gh the kernels' k = 0
    // remainders.  For Coulomb this is the parallel-plate capacitor,
    // −(π/(ε A)) Σ Q_g Q_h d_gh; for Rytova-Keldysh the r0 parts cancel
    // between the pairs and what is left is the capacitor of the actual
    // cross-plane kernel; for a screened stack it is the capacitor of that
    // dielectric environment.  Already INCLUDED in the local energy (inside the
    // k = 0 constant); reported apart because published bilayer energies are
    // quoted both with and without it.  Unlike background_energy() it depends
    // only on the kernels, not on the splitting.  0 for a single kernel, and 0
    // when the capacitor is excluded (then it is not in the local energy).
    [[nodiscard]] double capacitor_energy() const noexcept { return capacitor_energy_; }
    [[nodiscard]] bool capacitor_excluded() const noexcept { return exclude_capacitor_; }

private:
    struct GradientWorkspace;
    // A distinct pair kernel: the hot-loop parameters of V(r) plus its breakup.
    struct Kernel {
        numerics::PotentialSpec spec;
        double inv_eps           = 1.0;
        double keldysh_prefactor = 0.0;   // pi / (2 eps r0), Keldysh only
        double keldysh_inv_r0    = 0.0;
        double d2                = 0.0;   // d_layer², 0 for an in-plane pair
        numerics::RadialCutoffCorrection cutoff_correction;
        double sr_k0_correction = 0.0; // mean change to the sampled short range
        // Immutable splitting data, shared by clone() across OpenMP threads —
        // the same reasoning as ScreenedCoulombTerm's fit
        // (screened_coulomb.h:82-89): never mutated in the per-walker loop.
        std::shared_ptr<const numerics::EwaldSetup> setup;
    };

    // Real-space half for the particle pairs of groups (g, h), templated on the
    // potential so V(r) is evaluated inline with no indirection.  The breakup
    // supplies V_long as a spline, identical for every potential; V itself is
    // whatever that potential's own vectorized kernel already computes, so this
    // is the one place a potential shows up in a hot loop.  `Offset` selects
    // the cross-plane distance sqrt(r² + d²) at compile time, so in-plane pairs
    // pay nothing for it.
    template <numerics::PotentialSpec::Kind K, bool Offset>
    void real_space(const ParticleBatch& batch, double* __restrict le,
                    std::size_t w_begin, std::size_t w_end,
                    const Kernel& kern, std::size_t g, std::size_t h) const;

    [[nodiscard]] std::size_t kernel_of(std::size_t g, std::size_t h) const noexcept {
        return pair_kernel_[g * groups_.size() + h];
    }

    // Short-range half f(r) = V(r) − V_long(r) of one pair inside r_cut, 0
    // beyond: the scalar twin of real_space(), same formula per kernel kind.
    [[nodiscard]] double short_range_pair(const Kernel& kern, double r) const;

    // ── Structure-factor cache of delta_potential_particle ──────────────────
    static constexpr std::size_t   kDeltaIncrementalLimit = 4;
    static constexpr std::uint32_t kDeltaRebuildEvery     = 4096;
    struct DeltaCache {
        const ParticleBatch* batch = nullptr;
        std::size_t nw = 0, np = 0, dim = 0, n_k = 0;
        std::vector<double> rho_re, rho_im;     // [(k_conf * C + c) * n_k + kk]
        std::vector<double> snapshot;           // [(k_conf * np + i) * dim + d]
        std::vector<std::uint32_t> increments;  // per configuration, since rebuild
        std::vector<unsigned char> built;       // per configuration
        std::vector<double> e1_re, e1_im, e2_re, e2_im;  // e^{i n b·r}, [n]
        std::vector<double> z_re, z_im, y_re, y_im;      // e^{ik·r}, [kk]
        DeltaCache() = default;
        // Copies (clone) start empty: a cache belongs to the batch it saw.
        DeltaCache(const DeltaCache&) {}
        DeltaCache& operator=(const DeltaCache&) { *this = DeltaCache{}; return *this; }
        DeltaCache(DeltaCache&&) noexcept = default;
        DeltaCache& operator=(DeltaCache&&) noexcept = default;
        ~DeltaCache() = default;
    };
    // e^{ik·r} for every stored k of the 2-D position r into (re, im) [kk].
    void phases_of(const SimulationCell& cell, const double* r,
                   std::vector<double>& re, std::vector<double>& im) const;
    // Bring configuration k of the cache in line with the batch.
    void sync_delta_cache(const ParticleBatch& batch, std::size_t k) const;
    mutable DeltaCache delta_cache_;

    std::vector<ParticleGroup> groups_;
    SimulationCell cell_; // Geometry used to build every kernel and k-list.
    std::vector<double>        charges_;  // per particle index
    double                     self_energy_ = 0.0;  // walker-independent constant
    double                     background_energy_ = 0.0;  // E_k0
    double                     net_charge_  = 0.0;
    double                     capacitor_energy_ = 0.0;
    bool                       exclude_capacitor_ = false;

    std::vector<Kernel>      kernels_;      // distinct kernels
    std::vector<std::size_t> pair_kernel_;  // [g*G + h] -> kernels_ index
    // Structure-factor classes: groups with identical kernel rows share one ρ.
    std::size_t              n_classes_ = 0;
    std::vector<std::size_t> particle_class_;    // per particle
    std::vector<std::size_t> class_pair_kernel_; // [c*C + d] -> kernels_ index
    const numerics::KeldyshChebyshev* keldysh_cheb_ = nullptr;
    double requested_tolerance_ = 0.0;
    bool virial_admitted_ = false;
};

}  // namespace alea
