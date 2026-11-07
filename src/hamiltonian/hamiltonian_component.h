// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <cmath>
#include "population/particle_group.h"
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "population/walker_set.h"
#include "hamiltonian/potential_gradient_workspace.h"
#include "hamiltonian/radial_pair_potential.h"

namespace alea {

// O(N) changed-pair traversal shared by radial potential delta overrides.
// The evaluator receives the OLD/NEW in-plane distance and returns pair V(r).
template<class Radial>
double radial_particle_delta(const ParticleBatch& configs, std::size_t p,
    std::size_t k, std::span<const double> proposed, const ParticleGroup& a,
    const ParticleGroup& b, Radial radial) {
    const auto dim = configs.dim(), nw = configs.num_walkers();
    double candidate[SimulationCell::kMaxDim];
    for (std::size_t d = 0; d < dim; ++d) candidate[d] = proposed[d];
    configs.cell().wrap(candidate);
    double delta = 0.0;
    auto pair = [&](std::size_t partner) {
        if (partner == p) return;
        double displacement[SimulationCell::kMaxDim];
        for (std::size_t d = 0; d < dim; ++d)
            displacement[d] = candidate[d] - configs.positions()[partner * dim * nw + d * nw + k];
        const double distance = std::sqrt(configs.cell().min_image_dist2(displacement));
        delta += radial(distance) - radial(configs.dist_tables()(k, p, partner));
    };
    if (a == b) {
        if (a.contains(p)) for (auto j = a.first; j < a.last; ++j) pair(j);
    } else {
        if (a.contains(p)) for (auto j = b.first; j < b.last; ++j) pair(j);
        if (b.contains(p)) for (auto j = a.first; j < a.last; ++j) pair(j);
    }
    return delta;
}

// Leading short-range singularity V_ab(r) ≈ C / r^p of a pair potential that is
// stronger than the 1/r pole coulomb_cusp_residue reports (p > 1).
// coefficient == 0 means the pair has no such core.
struct PowerLawCore {
    double coefficient = 0.0;  // C
    double exponent    = 0.0;  // p
};

class HamiltonianComponent {
public:
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    virtual ~HamiltonianComponent() = default;

    virtual std::string name() const = 0;
    // Pair-product builders query physics, never potential names or RTTI.
    virtual PairActionRole pair_action_role() const noexcept { return PairActionRole::unsupported; }
    virtual std::vector<RadialPairPotential> radial_pair_potentials() const { return {}; }

    // Coefficient R of this term's short-range 1/r singularity for the group
    // pair (a,b):  R = lim_{r→0} r · V_ab(r).  It is what a Jastrow cusp must
    // cancel — the trial wavefunction sets Γ = 2 μ R / (d−1) so the divergence
    // in the local kinetic energy cancels the one in V.  Returns 0 when this
    // term does not couple (a,b) or has no 1/r pole (e.g. Rytova–Keldysh, whose
    // origin divergence is only logarithmic).  Keeping it potential-driven means
    // the Jastrow never has to switch on potential type.
    virtual double coulomb_cusp_residue(const ParticleGroup& a,
                                        const ParticleGroup& b) const {
        (void)a;
        (void)b;
        return 0.0;
    }

    // The pair (a,b)'s short-range core V ≈ C/r^p when it is more singular than
    // a 1/r pole.  No cusp can cancel it: for p > 2 the two-body wavefunction
    // must vanish at contact as exp(−κ r^{−(p−2)/2}), and the Jastrow derives κ
    // from (C, p) here, so it never has to switch on potential type.  {0, 0}
    // when this term does not couple (a,b) or has no such core.
    virtual PowerLawCore power_law_core(const ParticleGroup& a,
                                        const ParticleGroup& b) const {
        (void)a;
        (void)b;
        return {};
    }

    // Primitive exp(-tau V) must have a normalizable collision measure.
    // Components with weaker attractive singularities can conservatively refuse.
    virtual bool primitive_pair_admissible(const ParticleGroup&,
                                           const ParticleGroup&) const noexcept {
        return true;
    }
    virtual bool is_translation_invariant() const noexcept { return false; }

    // Potential change for one particle at configuration k; no committed change.
    // Default is exact evaluate/restore using caller-prepared [nw] scratch.
    // Requires current committed distance tables; positions/tables are restored
    // but ParticleBatch's disposable pending proposal buffers are overwritten.
    // Evaluation cost is component-dependent (e.g. O(N^2) pair V), plus O(N)
    // refresh at this ONE configuration; no global slice-table rebuild.
    virtual double delta_potential_particle(ParticleBatch& configs,
        std::size_t p, std::size_t k, std::span<const double> proposed,
        std::span<double> scratch) const {
        const auto dim = configs.dim(), nw = configs.num_walkers();
        if (p >= configs.n_particles() || k >= nw || proposed.size() != dim ||
            scratch.size() < nw || dim > SimulationCell::kMaxDim)
            throw std::invalid_argument("potential delta: invalid particle, slice or workspace");
        double saved[SimulationCell::kMaxDim], candidate[SimulationCell::kMaxDim];
        auto positions = configs.positions();
        for (std::size_t d = 0; d < dim; ++d) {
            saved[d] = positions[p * dim * nw + d * nw + k];
            candidate[d] = proposed[d];
        }
        configs.cell().wrap(candidate);
        scratch[k] = 0.0;
        evaluate_potential(configs, scratch, k, k + 1);
        const double old_value = scratch[k];
        struct Restore {
            ParticleBatch& configs; std::size_t p, k; const double* saved;
            ~Restore() {
                const auto dim = configs.dim(), nw = configs.num_walkers();
                for (std::size_t d = 0; d < dim; ++d)
                    configs.positions()[p * dim * nw + d * nw + k] = saved[d];
                configs.refresh_particle_tables(p, k, k + 1);
            }
        } restore{configs, p, k, saved};
        for (std::size_t d = 0; d < dim; ++d)
            positions[p * dim * nw + d * nw + k] = candidate[d];
        configs.refresh_particle_tables(p, k, k + 1);
        scratch[k] = 0.0;
        evaluate_potential(configs, scratch, k, k + 1);
        return scratch[k] - old_value;
    }

    // q_a q_b Ṽ(k): the 2-D Fourier transform of this term's interaction of the
    // pair (a, b), 0 for pairs it does not couple.  Read at setup to seed a
    // long-range Jastrow from the actual potential (Gaskell RPA).
    virtual double pair_fourier(const ParticleGroup& a, const ParticleGroup& b,
                                double k) const {
        (void)a;
        (void)b;
        (void)k;
        return 0.0;
    }

    // True when this term is a potential V(R): a function of the particle
    // positions alone, evaluated by evaluate_potential.  False for the kinetic
    // term, whose local value reads the trial wavefunction.  A driver that has
    // no ψ (PIMC) builds its weight from the potential terms only.
    [[nodiscard]] virtual bool is_potential() const noexcept { return true; }

    // Spatial derivative of THIS sampled potential, G=grad V (force=-G).
    // Unsupported potentials opt out, rather than silently supplying zero or
    // a numerical derivative with an undeclared step-size approximation.
    [[nodiscard]] virtual bool supports_potential_gradient() const noexcept { return false; }
    // Opt-in for the open primitive centroid identity: temperature-independent,
    // continuous sampled V with its gradient, bounded below apart from a
    // declared confining term. A gradient alone does not certify this domain.
    [[nodiscard]] virtual bool supports_open_centroid_virial() const noexcept { return false; }
    // Opt-in for torus integration by parts: the sampled V must be continuous,
    // periodic on this cell, temperature independent, and have its sampled
    // gradient. Finite primitive collision measure is checked separately.
    [[nodiscard]] virtual bool supports_periodic_virial(const SimulationCell&, std::size_t) const noexcept { return false; }
    // Describe sampled-potential approximations in PIMC output, independently
    // of estimator choice. Queried only for reporting, never in a hot loop.
    [[nodiscard]] virtual std::string potential_approximation_notes() const { return {}; }
    // Sufficient confinement of this absolute coordinate, not relative binding.
    [[nodiscard]] virtual bool confines_open_coordinate(std::size_t, std::size_t,
                                                        std::size_t) const noexcept { return false; }
    [[nodiscard]] virtual std::unique_ptr<PotentialGradientWorkspace>
    make_potential_gradient_workspace(const ParticleBatch& configs) const {
        if (!supports_potential_gradient())
            throw std::logic_error(name() + ": potential gradient is not supported");
        return std::unique_ptr<PotentialGradientWorkspace>(
            new PotentialGradientWorkspace(configs, *this));
    }
    // Accumulate over [w_begin,w_end); all other columns remain untouched.
    // Read-only on coordinates/tables, using caller-prepared scratch. Output
    // is [np*dim*nw], separate from WalkerSet's wavefunction derivatives.
    virtual void evaluate_potential_gradient(const ParticleBatch&,
        std::span<double>, PotentialGradientWorkspace&,
        std::size_t = 0, std::size_t = npos) const {
        throw std::logic_error(name() + ": potential gradient is not supported");
    }

    // Accumulate V(R_w) into v_out[w] for every configuration w of
    // [w_begin, w_end) (default = whole batch).  Stride is the global nw.
    // Reads only positions, distance tables and the cell, so the same code
    // serves walkers (VMC/DMC) and imaginary-time slices (PIMC).  Every
    // potential implements it; a term with is_potential() == false does not.
    virtual void evaluate_potential(const ParticleBatch& configs,
                                    std::span<double> v_out,
                                    std::size_t w_begin = 0,
                                    std::size_t w_end = npos) const {
        (void)configs; (void)v_out; (void)w_begin; (void)w_end;
        throw std::logic_error(name() + ": not a potential term (is_potential() "
                               "is false), evaluate_potential is undefined");
    }

    // Accumulate this component's local energy contribution into
    // batch.local_energies() [nw] over the contiguous walker range
    // [w_begin, w_end) (default = whole batch).  Stride is the global nw.
    // A potential's local energy is V(R) itself; ψ-dependent terms override.
    virtual void evaluate_batch(WalkerSet& batch,
                                std::size_t w_begin = 0,
                                std::size_t w_end = npos) {
        evaluate_potential(batch, batch.local_energies(), w_begin, w_end);
    }

    // Accumulate ∂E_L/∂p_k into `out` [n_opt * nw], given the wavefunction's
    // ∂(∂_d logψ)/∂p_k and ∂(∂²_d logψ)/∂p_k ([n_opt * np * dim * nw] each).
    // Only the kinetic term depends on the parameters — V(R) does not — so this
    // is a no-op for every potential.
    virtual void local_energy_parameter_derivatives(
        const WalkerSet& batch, std::span<const double> dgrad,
        std::span<const double> dlap, std::size_t n_opt,
        std::span<double> out) const {
        (void)batch; (void)dgrad; (void)dlap; (void)n_opt; (void)out;
    }

    // Accumulate the part of this term's local energy that depends on the
    // trial wavefunction into `out` [nw] (the kinetic term).  Potentials V(R)
    // leave it untouched.  Correlated sampling uses the split: a stored
    // configuration keeps V and recomputes only this part for new parameters.
    virtual void accumulate_psi_dependent_energy(const WalkerSet& batch,
                                                 std::span<double> out) const {
        (void)batch; (void)out;
    }

    [[nodiscard]] virtual std::unique_ptr<HamiltonianComponent> clone() const = 0;
};

}  // namespace alea
