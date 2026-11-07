// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "hamiltonian/ewald_term.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <iomanip>
#include <sstream>
#include <limits>

#include "numerics/special_functions.h"
#include "population/walker_set.h"

namespace alea {

namespace {

// Every unordered group pair, self-pairs included, on one kernel.
std::vector<EwaldPair> all_pairs(std::size_t n, const numerics::PotentialSpec& k) {
    std::vector<EwaldPair> out;
    for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = a; b < n; ++b) out.push_back({a, b, k});
    return out;
}

// Validates the cusp epsilon BEFORE the (costly) breakup is built.
const numerics::PotentialSpec& with_cusp_eps(const numerics::PotentialSpec& k,
                                             double cusp_eps) {
    if (!(cusp_eps > 0.0))
        throw std::runtime_error("EwaldTerm: epsilon must be positive");
    return k;
}

}  // namespace

EwaldTerm::EwaldTerm(std::vector<ParticleGroup> groups, double epsilon,
                     const SimulationCell& cell, double tol)
    : EwaldTerm(std::move(groups), numerics::PotentialSpec::coulomb(epsilon),
                epsilon, cell, tol) {}

EwaldTerm::EwaldTerm(std::vector<ParticleGroup> groups,
                     const numerics::PotentialSpec& pot, double cusp_eps,
                     const SimulationCell& cell, double tol)
    : EwaldTerm(groups, all_pairs(groups.size(), with_cusp_eps(pot, cusp_eps)),
                cell, tol) {}

EwaldTerm::EwaldTerm(std::vector<ParticleGroup> groups,
                     std::vector<EwaldPair> pairs, const SimulationCell& cell,
                     double tol, bool exclude_capacitor)
    : groups_(std::move(groups)), cell_(cell), requested_tolerance_(tol) {
    exclude_capacitor_ = exclude_capacitor;
    const std::size_t G = groups_.size();

    // ── Validation, before any (costly) breakup is built ────────────────────
    if (G == 0) throw std::runtime_error("EwaldTerm: no particle groups");
    std::vector<int> seen(G * G, 0);
    for (const EwaldPair& p : pairs) {
        if (p.a >= G || p.b >= G)
            throw std::runtime_error("EwaldTerm: pair (" + std::to_string(p.a) + ", " +
                                     std::to_string(p.b) + ") names a group index "
                                     "out of range (" + std::to_string(G) + " groups)");
        if (seen[p.a * G + p.b]++ || (p.a != p.b && seen[p.b * G + p.a]++))
            throw std::runtime_error("EwaldTerm: group pair (" + groups_[p.a].name +
                                     ", " + groups_[p.b].name + ") assigned twice");
        if (!(p.kernel.epsilon > 0.0))
            throw std::runtime_error("EwaldTerm: epsilon must be positive");
    }
    for (std::size_t a = 0; a < G; ++a)
        for (std::size_t b = a; b < G; ++b)
            if (!seen[a * G + b])
                throw std::runtime_error("EwaldTerm: group pair (" + groups_[a].name +
                                         ", " + groups_[b].name + ") has no kernel");
    // Every kernel here tends to 2π/(ε k) as k → 0.  The pairs' 1/k divergences
    // cancel against neutrality only if that ε is common to all of them.
    for (const EwaldPair& p : pairs)
        if (p.kernel.epsilon != pairs.front().kernel.epsilon)
            throw std::runtime_error(
                "EwaldTerm: pair kernels with different long-range epsilon (" +
                std::to_string(pairs.front().kernel.epsilon) + " vs " +
                std::to_string(p.kernel.epsilon) + ").  Every pair's potential must "
                "share the 1/(eps r) tail, or the periodic sum diverges even in a "
                "neutral cell.  Put layer-specific screening in the short-range "
                "parameters (r0, offset), not in eps.");
    bool several = false;
    for (const EwaldPair& p : pairs) several |= !p.kernel.same_as(pairs.front().kernel);
    double q_total = 0.0;
    for (const ParticleGroup& g : groups_)
        q_total += g.charge * static_cast<double>(g.last - g.first);
    if (several && std::abs(q_total) > 1e-12 && !exclude_capacitor_)
        throw std::runtime_error(
            "EwaldTerm: a charged cell (net charge " + std::to_string(q_total) +
            ") with more than one pair kernel.  The neutralising background couples "
            "to every charge through a single kernel, which is undefined here; "
            "`lattice.capacitor: exclude` defines it by neutralizing every layer "
            "in its own plane.");

    // ── Distinct kernels and the group-pair -> kernel map ───────────────────
    pair_kernel_.assign(G * G, 0);
    for (const EwaldPair& p : pairs) {
        std::size_t idx = 0;
        while (idx < kernels_.size() && !kernels_[idx].spec.same_as(p.kernel)) ++idx;
        if (idx == kernels_.size()) {
            Kernel k;
            k.spec    = p.kernel;
            k.inv_eps = 1.0 / p.kernel.epsilon;
            k.d2      = p.kernel.d_layer * p.kernel.d_layer;
            if (p.kernel.kind == numerics::PotentialSpec::Kind::Keldysh) {
                k.keldysh_prefactor = numerics::PI / (2.0 * p.kernel.epsilon * p.kernel.r0);
                k.keldysh_inv_r0    = 1.0 / p.kernel.r0;
                keldysh_cheb_       = &numerics::keldysh_cheb();
                // A cutoff in the frozen low clamp cannot be matched to the
                // ideal transform-pair endpoint by a small residual correction.
                if (cell.is_periodic() && cell.dim() == 2 &&
                    std::hypot(cell.ws_radius(),p.kernel.d_layer)*k.keldysh_inv_r0 <= keldysh_cheb_->x_lo)
                    throw std::runtime_error("ewald: RK cutoff is at or below the sampled low clamp; increase the cell or reduce r0");
            }
            k.setup = std::make_shared<const numerics::EwaldSetup>(
                numerics::build_ewald(cell, tol, p.kernel));
            if (p.kernel.kind == numerics::PotentialSpec::Kind::Keldysh) {
                const auto& spline=k.setup->long_range;
                const double r=k.setup->r_cut, h=1.0/spline.inv_delta;
                const double rho=std::hypot(r,p.kernel.d_layer), x=rho*k.keldysh_inv_r0;
                const double drho=r/rho, d2rho=(p.kernel.d_layer/rho)*(p.kernel.d_layer/rho)/rho;
                const double value=k.keldysh_prefactor*keldysh_cheb_->eval(x);
                const double slope=k.keldysh_prefactor*k.keldysh_inv_r0*keldysh_cheb_->d1(x)*drho;
                const double curvature=k.keldysh_prefactor*k.keldysh_inv_r0*
                    (k.keldysh_inv_r0*keldysh_cheb_->d2(x)*drho*drho+keldysh_cheb_->d1(x)*d2rho);
                const auto index=spline.scaled.size()-3;
                // Match the sampled kernel to the stored endpoint of the EXACT
                // spline. Changing that spline alone would corrupt its Fourier
                // identity. This last-panel correction changes only the small
                // real-space approximation residual and its k=0 mean.
                k.cutoff_correction=numerics::RadialCutoffCorrection(r-h,r,
                    value-spline.scaled[index],slope-spline.scaled[index+1]/h,
                    curvature-spline.scaled[index+2]/h/h);
                k.sr_k0_correction=-2.0*numerics::PI*(k.cutoff_correction.radial_integral()/k.setup->area);
                if (!std::isfinite(k.sr_k0_correction))
                    throw std::runtime_error("ewald: nonfinite RK cutoff correction mean");
            }
            kernels_.push_back(std::move(k));
        }
        pair_kernel_[p.a * G + p.b] = idx;
        pair_kernel_[p.b * G + p.a] = idx;
    }
    // The reciprocal loop walks ONE k-list for every kernel.  Same cell and
    // default k_cut make the lists identical; guard it rather than assume it.
    for (const Kernel& k : kernels_)
        if (k.setup->n1 != kernels_.front().setup->n1 ||
            k.setup->n2 != kernels_.front().setup->n2)
            throw std::logic_error("EwaldTerm: kernels disagree on the k-list");

    // ── Structure-factor classes ────────────────────────────────────────────
    // Two groups can share one ρ(k) exactly when their kernel ROWS agree: then
    // every coupling either of them has, to anything, is the same kernel.
    std::vector<std::size_t> group_class(G), class_rep;
    for (std::size_t g = 0; g < G; ++g) {
        std::size_t c = 0;
        for (; c < class_rep.size(); ++c) {
            bool same = true;
            for (std::size_t x = 0; x < G && same; ++x)
                same = kernel_of(g, x) == kernel_of(class_rep[c], x);
            if (same) break;
        }
        if (c == class_rep.size()) class_rep.push_back(g);
        group_class[g] = c;
    }
    n_classes_ = class_rep.size();
    class_pair_kernel_.resize(n_classes_ * n_classes_);
    for (std::size_t c = 0; c < n_classes_; ++c)
        for (std::size_t d = 0; d < n_classes_; ++d)
            class_pair_kernel_[c * n_classes_ + d] = kernel_of(class_rep[c], class_rep[d]);

    std::size_t np = 0;
    for (const ParticleGroup& g : groups_) np = std::max(np, g.last);
    charges_.assign(np, 0.0);
    particle_class_.assign(np, 0);
    for (std::size_t g = 0; g < G; ++g)
        for (std::size_t p = groups_[g].first; p < groups_[g].last; ++p) {
            charges_[p]        = groups_[g].charge;
            particle_class_[p] = group_class[g];
        }

    // Self-interaction: the reciprocal sum includes each charge acting on
    // itself (the i = j terms of |ρ_g(k)|²), which the real-space half
    // excludes.  Subtract V_long(0) of the group's own kernel, halved for
    // double counting.  For a constant eps this is 2a/(eps sqrt(pi)),
    // reproducing the textbook -a/sqrt(pi) * sum q^2.
    std::vector<double> Q(G, 0.0);
    for (std::size_t g = 0; g < G; ++g) {
        const ParticleGroup& gr = groups_[g];
        const double q2 = gr.charge * gr.charge * static_cast<double>(gr.last - gr.first);
        self_energy_ -= 0.5 * kernels_[kernel_of(g, g)].setup->v_long_zero * q2;
        Q[g] = gr.charge * static_cast<double>(gr.last - gr.first);
        net_charge_ += Q[g];
    }

    // ── Charged cell: the uniform neutralising background ───────────────────
    //
    // Under PBC the cell is not one cell but an infinite array of them.  If the
    // array carries net charge, the electrostatic potential at ANY point
    // diverges: a shell of cells at distance r holds ~2*pi*r/A of them, each
    // contributing ~Q/r, so every shell adds the same amount however far away
    // it is.  Summed out to radius R the potential grows like R -- doubling the
    // cutoff doubles the answer.  That is not a weakness of the Ewald sum; the
    // quantity does not exist.
    //
    // A uniform background of density -Q/A restores neutrality and makes it
    // exist.  It is not a numerical patch: in a real doped system the
    // compensating charge is physically there (donors, gates), merely outside
    // the layers one chooses to simulate.  What gets solved is jellium -- a
    // different system from "N bare charges", and a standard one.
    //
    // Its entire effect is one constant.  A uniform density has Fourier weight
    // only at k = 0, so no k != 0 term moves; at k = 0 it makes rho_total
    // vanish, which is exactly the term this sum omits.  What remains is the
    // short-range half, through which particles and background still interact:
    //
    //   charge-background      Q*(-Q/A)*int V_short d2r  =  -Q^2 * sr_k0
    //   background-background  (1/2)(Q/A)^2*A*int V_short = +Q^2 * sr_k0 / 2
    //   total                                            =  -Q^2 * sr_k0 / 2
    //
    // For a neutral cell Q = 0 and this vanishes identically, so nothing about
    // the neutral path changes.
    //
    // ── Several kernels ────────────────────────────────────────────────────
    // Then the pairs' k = 0 remainders no longer factor through (Σq)², and
    // E_k0 = Σ_{g,h} Q_g Q_h [c_gh/(2A) − ½ sr_k0_gh] (header) is a genuine,
    // finite energy of the neutral cell.
    //
    // ── Capacitor excluded ─────────────────────────────────────────────────
    // Dropping every c_gh leaves E_k0 = −½ Σ_{g,h} Q_g Q_h sr_k0_gh: every
    // layer neutralized in its own plane, so no mean-field energy between
    // layers, for a neutral or a charged cell.  With one kernel it is the
    // jellium constant above.  It is CASINO's convention: its total energy
    // adds −2πd N_e N_h/A to the bilayer Ewald sum (data of López Ríos et al.,
    // PRL 120, 177701 (2018)), which is −Σ Q_g Q_h c_gh/(2A) for Coulomb.
    // c_gh is a property of the kernel, not of the splitting, so the total
    // stays independent of the breakup.
    if (exclude_capacitor_ && kernels_.size() > 1) {
        for (std::size_t g = 0; g < G; ++g)
            for (std::size_t h = 0; h < G; ++h) {
                const Kernel& k = kernels_[kernel_of(g, h)];
                background_energy_ -= 0.5 * Q[g] * Q[h] * (k.setup->sr_k0 + k.sr_k0_correction);
            }
    } else if (kernels_.size() == 1) {
        background_energy_ =
            -0.5 * net_charge_ * net_charge_ *
            (kernels_.front().setup->sr_k0+kernels_.front().sr_k0_correction);
    } else {
        for (std::size_t g = 0; g < G; ++g)
            for (std::size_t h = 0; h < G; ++h) {
                const Kernel& k = kernels_[kernel_of(g, h)];
                background_energy_ += Q[g] * Q[h] *
                    (k.spec.k0_offset / (2.0 * k.setup->area) -
                     0.5 * (k.setup->sr_k0+k.sr_k0_correction));
                // The kernels' own k = 0 remainders alone: the mean-field
                // energy of each layer's charge smeared uniformly (header).
                capacitor_energy_ += Q[g] * Q[h] * k.spec.k0_offset / (2.0 * k.setup->area);
            }
    }
    virial_admitted_ = std::isfinite(self_energy_) && std::isfinite(background_energy_) &&
        std::isfinite(capacitor_energy_);
    for (const auto& k : kernels_) {
        const auto& spline=k.setup->long_range;
        const auto index=spline.scaled.size()-3;
        const double r=k.setup->r_cut,rho=std::hypot(r,k.spec.d_layer);
        double sampled=0;
        using Kind=numerics::PotentialSpec::Kind;
        if (k.spec.kind==Kind::Coulomb) sampled=k.inv_eps/rho;
        else if (k.spec.kind==Kind::Keldysh)
            sampled=k.keldysh_prefactor*keldysh_cheb_->eval(rho*k.keldysh_inv_r0)-k.cutoff_correction.eval(r);
        else sampled=k.spec.singular_coef/r+k.spec.table->eval(r);
        const double rounding=64*std::numeric_limits<double>::epsilon()*(1+std::abs(sampled)+std::abs(spline.scaled[index]));
        virial_admitted_ = virial_admitted_ && std::isfinite(sampled) &&
            std::abs(sampled-spline.scaled[index])<=rounding;
        for (double weight : k.setup->weight) virial_admitted_ = virial_admitted_ && std::isfinite(weight);
    }
    for (double q : charges_) virial_admitted_ = virial_admitted_ && std::isfinite(q);
}

bool EwaldTerm::supports_periodic_virial(const SimulationCell& cell, std::size_t np) const noexcept {
    if (!cell.is_periodic() || cell.dim()!=2 || np!=charges_.size() ||
        !std::equal(cell.vectors().begin(),cell.vectors().end(),cell_.vectors().begin())) return false;
    // The compact real part matches value/slope/curvature at r_cut. Its
    // support lies within the WS inradius. The reciprocal finite sum is
    // smooth and periodic, including self and k=0 constants. Accuracy of
    // that represented Hamiltonian is separate from its torus identity.
    return virial_admitted_;
}

std::string EwaldTerm::potential_approximation_notes() const {
    std::ostringstream out;
    out << std::setprecision(17) << "optimized Ewald; requested_tolerance=" << requested_tolerance_;
    for (std::size_t i=0; i<kernels_.size(); ++i) {
        const auto& k=kernels_[i];
        out << "; kernel=" << i << " r_cut=" << k.setup->r_cut
            << " k_cut=" << k.setup->k_cut << " reciprocal_vectors=" << k.setup->n_k()
            << " sampled_omitted_value_residual=" << k.setup->residual
            << " breakup_condition=" << k.setup->condition
            << " cutoff_correction_mean=" << k.sr_k0_correction;
    }
    out << "; sampled residual is not a global energy/force bound; kernel/breakup/cutoff errors excluded from statistical uncertainty";
    return out.str();
}

double EwaldTerm::coulomb_cusp_residue(const ParticleGroup& a,
                                       const ParticleGroup& b) const {
    const auto index_of = [&](const ParticleGroup& x) {
        std::size_t g = 0;
        while (g < groups_.size() && !(groups_[g] == x)) ++g;
        return g;
    };
    const std::size_t ga = index_of(a), gb = index_of(b);
    if (ga == groups_.size() || gb == groups_.size()) return 0.0;
    const Kernel& k = kernels_[kernel_of(ga, gb)];
    // Screened: the table is smooth, so the pole is the singular part alone
    // (1/eps_source within a layer, 0 across layers).
    if (k.spec.kind == numerics::PotentialSpec::Kind::Screened)
        return a.charge * b.charge * k.spec.singular_coef;
    if (k.spec.kind != numerics::PotentialSpec::Kind::Coulomb) return 0.0;
    // A layer offset makes V finite at r = 0: no pole, no cusp.
    if (k.d2 > 0.0) return 0.0;
    return a.charge * b.charge * k.inv_eps;
}

// Smooth Ewald subtraction does not remove a same-layer logarithmic pole.
// Refuse attractive RK unless softened, as for the open primitive component.
bool EwaldTerm::primitive_pair_admissible(const ParticleGroup& a,
                                         const ParticleGroup& b) const noexcept {
    const auto index_of = [&](const ParticleGroup& x) {
        std::size_t g = 0;
        while (g < groups_.size() && !(groups_[g] == x)) ++g;
        return g;
    };
    const auto ga = index_of(a), gb = index_of(b);
    if (ga == groups_.size() || gb == groups_.size()) return true;
    const auto& kernel = kernels_[kernel_of(ga, gb)];
    return kernel.spec.kind != numerics::PotentialSpec::Kind::Keldysh ||
           kernel.d2 > 0.0 || a.charge*b.charge >= 0.0;
}

double EwaldTerm::pair_fourier(const ParticleGroup& a, const ParticleGroup& b,
                               double k) const {
    const auto index_of = [&](const ParticleGroup& x) {
        std::size_t g = 0;
        while (g < groups_.size() && !(groups_[g] == x)) ++g;
        return g;
    };
    const std::size_t ga = index_of(a), gb = index_of(b);
    if (ga == groups_.size() || gb == groups_.size()) return 0.0;
    const Kernel& kern = kernels_[kernel_of(ga, gb)];
    if (!kern.spec.v_fourier) return 0.0;
    return a.charge * b.charge * kern.spec.v_fourier(k);
}

// V_short(r) = V(r) - V_long(r), over the nearest image of every pair.
//
// dt.row(i,j) is already the minimum-image distance, and `alpha` was chosen so
// every further image is below tolerance, so this single term IS the real-space
// lattice sum.
//
// V_long comes from the setup's Chebyshev table (smooth for any potential);
// V(r) is evaluated inline here, which is why this is templated.
template <numerics::PotentialSpec::Kind K, bool Offset>
void EwaldTerm::real_space(const ParticleBatch& batch, double* __restrict le,
                           std::size_t w_begin, std::size_t w_end,
                           const Kernel& kern, std::size_t g, std::size_t h) const {
    const BatchedDistanceTable& dt = batch.dist_tables();

    // The long-range half is the breakup's own quintic spline — exact, and
    // measured 5.7x faster than the 32-term Chebyshev Clenshaw it replaced,
    // because that recurrence is a dependency chain of length 32 while the six
    // quintics inside a panel are independent.  It does not vectorize (the
    // per-walker coefficient gather defeats GCC, verified with
    // -fopt-info-vec), and is still the faster of the two by a wide margin —
    // the vector instruction was never the thing that mattered here.
    const numerics::LongRangeSpline& sp = kern.setup->long_range;
    const double r_cut = kern.setup->r_cut;

    // Coulomb parameters.
    const double inv_eps = kern.inv_eps;
    // Keldysh parameters (unused, and folded away, for Coulomb).
    const double kpre = kern.keldysh_prefactor, kinv = kern.keldysh_inv_r0;
    [[maybe_unused]] const double d2 = kern.d2;
    // Screened parameters (unused otherwise).
    [[maybe_unused]] const double                    sing  = kern.spec.singular_coef;
    [[maybe_unused]] const numerics::RadialTable*    table = kern.spec.table.get();

    const ParticleGroup& ga = groups_[g];
    const ParticleGroup& gb = groups_[h];
    for (std::size_t i = ga.first; i < ga.last; ++i)
        for (std::size_t j = (g == h ? i + 1 : gb.first); j < gb.last; ++j) {
            // 1/eps already lives inside V(r) and V_long(r), so no epsilon here.
            const double coeff = charges_[i] * charges_[j];
            if (coeff == 0.0) continue;
            const double* __restrict r = dt.row(i, j).data();
#pragma omp simd
            for (std::size_t w = w_begin; w < w_end; ++w) {
                // The kernel sees the 3-D separation; the long-range spline
                // below is a function of the IN-PLANE r, which is what the
                // breakup was built on.
                double rr;
                if constexpr (Offset) rr = std::sqrt(r[w] * r[w] + d2);
                else                  rr = r[w];
                double v;
                if constexpr (K == numerics::PotentialSpec::Kind::Coulomb) {
                    v = inv_eps / rr;
                } else if constexpr (K == numerics::PotentialSpec::Kind::Screened) {
                    // Bulk core of the source layer (0 across layers) plus the
                    // smooth tabulated remainder; the stack's geometry, layer
                    // offset included, lives in the table.
                    // A cross-layer kernel has no pole. Avoid 0/0 at an
                    // in-plane coincidence; its finite value is the table.
                    v = (sing != 0.0 ? sing / rr : 0.0) + table->eval(rr);
                } else {
                    v = kpre * keldysh_cheb_->eval(rr*kinv)-kern.cutoff_correction.eval(r[w]);
                }
                // f_short = V - V_long strictly INSIDE r_cut, zero from there
                // on.  Two traps here, both caught by the Madelung test:
                //
                // The mask is not optional.  eval() returns 0 past r_cut, so an
                // unmasked (v - eval) would add the FULL potential for every
                // distant pair instead of nothing.
                //
                // It must also be STRICT.  eval() jumps from f(r_cut) to 0 at
                // the cutoff, so the boundary point is not harmless: a copysign
                // mask includes it, because copysign(1, +0.0) is +1, and in the
                // Madelung cell every nearest-neighbour pair sits at exactly
                // r_cut — that alone shifted the constant by 2.0.
                //
                // The project style prefers copysign over a comparison because a
                // comparison mask blocks vectorization.  It does not apply: the
                // per-walker coefficient gather in eval() already prevents this
                // loop from vectorizing (verified with -fopt-info-vec), and the
                // spline is still 5.7x faster than the Clenshaw it replaced.
                le[w] += r[w] < r_cut ? coeff * (v - sp.eval(r[w])) : 0.0;
            }
        }
}

void EwaldTerm::evaluate_potential(const ParticleBatch& batch,
                               std::span<double> v_out,
                               std::size_t w_begin, std::size_t w_end) const {
    const std::size_t nw = batch.num_walkers();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end) return;
    const std::size_t np  = charges_.size();
    const std::size_t dim = batch.dim();

    double* __restrict le = v_out.data();
    // Every kernel was built on the same cell with the same reciprocal cutoff,
    // so they share one k-list (checked at construction); take it from any.
    const numerics::EwaldSetup& s = *kernels_.front().setup;

    // ── Real space, per group pair, dispatched on its potential ────────────
    // The dispatch sits outside the walker loop, so the SIMD loop inside
    // real_space stays branch-free.
    const std::size_t G = groups_.size();
    for (std::size_t g = 0; g < G; ++g)
        for (std::size_t h = g; h < G; ++h) {
            using Kind = numerics::PotentialSpec::Kind;
            const Kernel& kern   = kernels_[kernel_of(g, h)];
            const bool    offset = kern.d2 > 0.0;
            if (kern.spec.kind == Kind::Coulomb) {
                if (offset) real_space<Kind::Coulomb, true>(batch, le, w_begin, w_end, kern, g, h);
                else        real_space<Kind::Coulomb, false>(batch, le, w_begin, w_end, kern, g, h);
            } else if (kern.spec.kind == Kind::Screened) {
                // Offset is baked into the table: always the in-plane distance.
                real_space<Kind::Screened, false>(batch, le, w_begin, w_end, kern, g, h);
            } else {
                if (offset) real_space<Kind::Keldysh, true>(batch, le, w_begin, w_end, kern, g, h);
                else        real_space<Kind::Keldysh, false>(batch, le, w_begin, w_end, kern, g, h);
            }
        }

    // ── Reciprocal space: (2π/A)·Σ_k erfc(k/2a)/k · |ρ(k)|² ────────────────
    //
    // BLOCKED OVER WALKERS.  The obvious layout — build the full exponential
    // ladders for all nw walkers, then loop over k — is memory bound, not flop
    // bound: at np=16, nw=512 the tables are ~2.6 MB, so every one of the n_k
    // passes re-reads them from L3.  Measured, that cost more than the entire
    // rest of the term.  Working a tile of walkers at a time keeps the ladders
    // and the accumulators inside L2 (and their hot rungs inside L1), which is
    // the difference between streaming ~19 MB per call and streaming almost
    // nothing.
    //
    // Scratch is thread_local and sized for one tile, so every OpenMP crowd has
    // its own.  That matters because DMC shares one Hamiltonian across crowds
    // (dmc.cpp:195) and because branching changes nw between calls, so a member
    // buffer could not be resized safely from inside the parallel region.
    constexpr std::size_t kTile = 64;
    thread_local std::vector<double> e1_re, e1_im, e2_re, e2_im, rho_re, rho_im;
    const std::size_t sz1 = static_cast<std::size_t>(s.n1_max + 1) * np * kTile;
    const std::size_t sz2 = static_cast<std::size_t>(s.n2_max + 1) * np * kTile;
    if (e1_re.size() < sz1) { e1_re.resize(sz1); e1_im.resize(sz1); }
    if (e2_re.size() < sz2) { e2_re.resize(sz2); e2_im.resize(sz2); }
    // One ρ per structure-factor class, [c * kTile + t].
    const std::size_t C = n_classes_;
    if (rho_re.size() < C * kTile) { rho_re.resize(C * kTile); rho_im.resize(C * kTile); }

    const auto   b   = batch.cell().reciprocal();
    const double b1x = b[0], b1y = b[1], b2x = b[2], b2y = b[3];
    const std::span<const double> pos = batch.positions();

    for (std::size_t w0 = w_begin; w0 < w_end; w0 += kTile) {
        const std::size_t nt = std::min(kTile, w_end - w0);

        // Build the exponential ladders by RECURRENCE:
        //   exp(i·n·b·r) = exp(i·(n-1)·b·r) · exp(i·b·r)
        // Two sin/cos calls per particle per walker, then pure multiplication.
        // Calling cos/sin inside the k loop instead would cost n_k times more
        // transcendentals and is what makes a naive Ewald unaffordable.
        auto build_ladder = [&](double bx, double by, int nmax,
                                std::vector<double>& er, std::vector<double>& ei) {
            for (std::size_t i = 0; i < np; ++i) {
                const double* __restrict px = pos.data() + (i * dim + 0) * nw;
                const double* __restrict py = pos.data() + (i * dim + 1) * nw;
                double* __restrict r0 = er.data() + i * kTile;
                double* __restrict i0 = ei.data() + i * kTile;
                for (std::size_t t = 0; t < nt; ++t) { r0[t] = 1.0; i0[t] = 0.0; }
                if (nmax < 1) continue;
                double* __restrict r1 = er.data() + (1 * np + i) * kTile;
                double* __restrict i1 = ei.data() + (1 * np + i) * kTile;
                for (std::size_t t = 0; t < nt; ++t) {
                    const double ph = bx * px[w0 + t] + by * py[w0 + t];
                    r1[t] = std::cos(ph);
                    i1[t] = std::sin(ph);
                }
                for (int n = 2; n <= nmax; ++n) {
                    const double* __restrict pr = er.data() + ((n - 1) * np + i) * kTile;
                    const double* __restrict pi = ei.data() + ((n - 1) * np + i) * kTile;
                    double* __restrict cr = er.data() + (n * np + i) * kTile;
                    double* __restrict ci = ei.data() + (n * np + i) * kTile;
#pragma omp simd
                    for (std::size_t t = 0; t < nt; ++t) {
                        cr[t] = pr[t] * r1[t] - pi[t] * i1[t];
                        ci[t] = pr[t] * i1[t] + pi[t] * r1[t];
                    }
                }
            }
        };
        build_ladder(b1x, b1y, s.n1_max, e1_re, e1_im);
        build_ladder(b2x, b2y, s.n2_max, e2_re, e2_im);

        for (std::size_t kk = 0; kk < s.n_k(); ++kk) {
            const int         n1 = s.n1[kk], n2 = s.n2[kk];
            const std::size_t a2 = static_cast<std::size_t>(std::abs(n2));
            // n2 < 0 uses the conjugate of the +|n2| rung rather than a second
            // ladder, which is where the half-k-space saving is actually banked.
            const double sgn = (n2 < 0) ? -1.0 : 1.0;

            // Tile rows are kTile apart; only the first nt of each are live.
            for (std::size_t c = 0; c < C; ++c)
                for (std::size_t t = 0; t < nt; ++t) {
                    rho_re[c * kTile + t] = 0.0;
                    rho_im[c * kTile + t] = 0.0;
                }

            for (std::size_t i = 0; i < np; ++i) {
                const double q = charges_[i];
                if (q == 0.0) continue;
                const std::size_t c = particle_class_[i];
                const double* __restrict ar =
                    e1_re.data() + (static_cast<std::size_t>(n1) * np + i) * kTile;
                const double* __restrict ai =
                    e1_im.data() + (static_cast<std::size_t>(n1) * np + i) * kTile;
                const double* __restrict br = e2_re.data() + (a2 * np + i) * kTile;
                const double* __restrict bi = e2_im.data() + (a2 * np + i) * kTile;
                double* __restrict rr = rho_re.data() + c * kTile;
                double* __restrict ri = rho_im.data() + c * kTile;
#pragma omp simd
                for (std::size_t t = 0; t < nt; ++t) {
                    const double bim = sgn * bi[t];
                    rr[t] += q * (ar[t] * br[t] - ai[t] * bim);
                    ri[t] += q * (ar[t] * bim + ai[t] * br[t]);
                }
            }

            // Σ_c w_cc |ρ_c|² + Σ_{c<d} 2 w_cd Re(ρ_c ρ_d*).  1/eps(k) is
            // already folded into the stored weights.  With one class this is
            // the plain w |ρ|² of a single-kernel cell, operation for operation.
            for (std::size_t c = 0; c < C; ++c) {
                const double* __restrict rc = rho_re.data() + c * kTile;
                const double* __restrict ic = rho_im.data() + c * kTile;
                const double wcc =
                    kernels_[class_pair_kernel_[c * C + c]].setup->weight[kk];
#pragma omp simd
                for (std::size_t t = 0; t < nt; ++t)
                    le[w0 + t] += wcc * (rc[t] * rc[t] + ic[t] * ic[t]);
                for (std::size_t d = c + 1; d < C; ++d) {
                    const double* __restrict rd = rho_re.data() + d * kTile;
                    const double* __restrict id = rho_im.data() + d * kTile;
                    const double w2 =
                        2.0 * kernels_[class_pair_kernel_[c * C + d]].setup->weight[kk];
#pragma omp simd
                    for (std::size_t t = 0; t < nt; ++t)
                        le[w0 + t] += w2 * (rc[t] * rd[t] + ic[t] * id[t]);
                }
            }
        }
    }

    // ── Constants: never inside the walker loop ────────────────────────────
    // Self-interaction, plus the neutralising background when the cell carries
    // net charge (zero otherwise).
    const double constants = self_energy_ + background_energy_;
    for (std::size_t w = w_begin; w < w_end; ++w) le[w] += constants;
}

// ── Single-particle ΔV (PIMC) ───────────────────────────────────────────────

double EwaldTerm::short_range_pair(const Kernel& kern, double r) const {
    // Same expression as real_space(), including its strict r < r_cut mask
    // (see the two traps documented there).
    if (!(r < kern.setup->r_cut)) return 0.0;
    using Kind = numerics::PotentialSpec::Kind;
    const bool offset = kern.d2 > 0.0 && kern.spec.kind != Kind::Screened;
    const double rr = offset ? std::sqrt(r * r + kern.d2) : r;
    double v;
    if (kern.spec.kind == Kind::Coulomb) {
        v = kern.inv_eps / rr;
    } else if (kern.spec.kind == Kind::Screened) {
        const double sing = kern.spec.singular_coef;
        v = (sing != 0.0 ? sing / rr : 0.0) + kern.spec.table->eval(rr);
    } else {
        v = kern.keldysh_prefactor * keldysh_cheb_->eval(rr * kern.keldysh_inv_r0) -
            kern.cutoff_correction.eval(r);
    }
    return v - kern.setup->long_range.eval(r);
}

void EwaldTerm::phases_of(const SimulationCell& cell, const double* r,
                          std::vector<double>& re, std::vector<double>& im) const {
    // The ladders of evaluate_potential() for one position: e^{i n b·r} by
    // recurrence, then e^{ik·r} = e^{i n1 b1·r} e^{i n2 b2·r}, n2 < 0 through
    // the conjugate rung.
    const numerics::EwaldSetup& s = *kernels_.front().setup;
    DeltaCache& c = delta_cache_;
    const auto b = cell.reciprocal();
    auto ladder = [&](double bx, double by, int nmax, std::vector<double>& er,
                      std::vector<double>& ei) {
        er.resize(static_cast<std::size_t>(nmax) + 1);
        ei.resize(static_cast<std::size_t>(nmax) + 1);
        er[0] = 1.0;
        ei[0] = 0.0;
        if (nmax < 1) return;
        const double ph = bx * r[0] + by * r[1];
        er[1] = std::cos(ph);
        ei[1] = std::sin(ph);
        for (std::size_t n = 2; n <= static_cast<std::size_t>(nmax); ++n) {
            er[n] = er[n - 1] * er[1] - ei[n - 1] * ei[1];
            ei[n] = er[n - 1] * ei[1] + ei[n - 1] * er[1];
        }
    };
    ladder(b[0], b[1], s.n1_max, c.e1_re, c.e1_im);
    ladder(b[2], b[3], s.n2_max, c.e2_re, c.e2_im);
    const std::size_t nk = s.n_k();
    re.resize(nk);
    im.resize(nk);
    for (std::size_t kk = 0; kk < nk; ++kk) {
        const std::size_t a1 = static_cast<std::size_t>(s.n1[kk]);
        const std::size_t a2 = static_cast<std::size_t>(std::abs(s.n2[kk]));
        const double bim = (s.n2[kk] < 0 ? -1.0 : 1.0) * c.e2_im[a2];
        re[kk] = c.e1_re[a1] * c.e2_re[a2] - c.e1_im[a1] * bim;
        im[kk] = c.e1_re[a1] * bim + c.e1_im[a1] * c.e2_re[a2];
    }
}

void EwaldTerm::sync_delta_cache(const ParticleBatch& batch, std::size_t k) const {
    DeltaCache& c = delta_cache_;
    const std::size_t nw = batch.num_walkers(), np = batch.n_particles(),
                      dim = batch.dim(), nk = kernels_.front().setup->n_k(),
                      C = n_classes_;
    if (c.batch != &batch || c.nw != nw || c.np != np || c.dim != dim || c.n_k != nk) {
        c.batch = &batch;
        c.nw = nw; c.np = np; c.dim = dim; c.n_k = nk;
        c.rho_re.assign(nw * C * nk, 0.0);
        c.rho_im.assign(nw * C * nk, 0.0);
        c.snapshot.assign(nw * np * dim, 0.0);
        c.increments.assign(nw, 0);
        c.built.assign(nw, 0);
    }
    const auto pos = batch.positions();
    auto at = [&](std::size_t i, std::size_t d) { return pos[i * dim * nw + d * nw + k]; };
    double* snap = c.snapshot.data() + k * np * dim;
    double* rho_re = c.rho_re.data() + k * C * nk;
    double* rho_im = c.rho_im.data() + k * C * nk;

    // Which particles moved since the cache last saw configuration k?
    std::size_t changed[kDeltaIncrementalLimit + 1];
    std::size_t n_changed = 0;
    if (c.built[k]) {
        for (std::size_t i = 0; i < np && n_changed <= kDeltaIncrementalLimit; ++i)
            for (std::size_t d = 0; d < dim; ++d)
                if (snap[i * dim + d] != at(i, d)) {
                    changed[n_changed++] = i;
                    break;
                }
        if (n_changed == 0) return;
    }
    const bool incremental = c.built[k] && n_changed <= kDeltaIncrementalLimit &&
                             c.increments[k] + n_changed < kDeltaRebuildEvery;
    if (incremental) {
        // ρ_c += q_i (e^{ik·r_i,new} − e^{ik·r_i,old}) for each moved particle.
        for (std::size_t m = 0; m < n_changed; ++m) {
            const std::size_t i = changed[m];
            const double q = charges_[i];
            double now[SimulationCell::kMaxDim]{};
            for (std::size_t d = 0; d < dim; ++d) now[d] = at(i, d);
            if (q != 0.0) {
                phases_of(batch.cell(), snap + i * dim, c.y_re, c.y_im);
                phases_of(batch.cell(), now, c.z_re, c.z_im);
                double* rr = rho_re + particle_class_[i] * nk;
                double* ri = rho_im + particle_class_[i] * nk;
                for (std::size_t kk = 0; kk < nk; ++kk) {
                    rr[kk] += q * (c.z_re[kk] - c.y_re[kk]);
                    ri[kk] += q * (c.z_im[kk] - c.y_im[kk]);
                }
            }
            for (std::size_t d = 0; d < dim; ++d) snap[i * dim + d] = now[d];
        }
        c.increments[k] += static_cast<std::uint32_t>(n_changed);
        return;
    }
    // Rebuild: ρ_c(k) = Σ_{i∈c} q_i e^{ik·r_i}, the reciprocal sum of
    // evaluate_potential() for this one configuration.
    std::fill(rho_re, rho_re + C * nk, 0.0);
    std::fill(rho_im, rho_im + C * nk, 0.0);
    for (std::size_t i = 0; i < np; ++i) {
        double now[SimulationCell::kMaxDim]{};
        for (std::size_t d = 0; d < dim; ++d) snap[i * dim + d] = now[d] = at(i, d);
        const double q = charges_[i];
        if (q == 0.0) continue;
        phases_of(batch.cell(), now, c.z_re, c.z_im);
        double* rr = rho_re + particle_class_[i] * nk;
        double* ri = rho_im + particle_class_[i] * nk;
        for (std::size_t kk = 0; kk < nk; ++kk) {
            rr[kk] += q * c.z_re[kk];
            ri[kk] += q * c.z_im[kk];
        }
    }
    c.increments[k] = 0;
    c.built[k] = 1;
}

double EwaldTerm::delta_potential_particle(ParticleBatch& configs, std::size_t p,
    std::size_t k, std::span<const double> proposed, std::span<double>) const {
    const std::size_t nw = configs.num_walkers(), np = configs.n_particles(),
                      dim = configs.dim();
    if (p >= np || k >= nw || proposed.size() != dim || dim != 2 || np != charges_.size())
        throw std::invalid_argument("ewald delta: invalid particle, configuration or dimension");
    const double q_p = charges_[p];
    if (q_p == 0.0) return 0.0;
    const SimulationCell& cell = configs.cell();
    const auto pos = configs.positions();
    double candidate[SimulationCell::kMaxDim]{}, old[SimulationCell::kMaxDim]{};
    for (std::size_t d = 0; d < dim; ++d) {
        candidate[d] = proposed[d];
        old[d] = pos[p * dim * nw + d * nw + k];
    }
    cell.wrap(candidate);

    // ── Real space: the moved particle's pairs only ────────────────────────
    std::size_t g_p = 0;
    while (g_p < groups_.size() && !groups_[g_p].contains(p)) ++g_p;
    double delta = 0.0;
    const BatchedDistanceTable& dt = configs.dist_tables();
    for (std::size_t g = 0; g < groups_.size(); ++g) {
        const Kernel& kern = kernels_[kernel_of(g_p, g)];
        for (std::size_t j = groups_[g].first; j < groups_[g].last; ++j) {
            if (j == p || charges_[j] == 0.0) continue;
            double del[SimulationCell::kMaxDim]{};
            for (std::size_t d = 0; d < dim; ++d)
                del[d] = candidate[d] - pos[j * dim * nw + d * nw + k];
            const double r_new = std::sqrt(cell.min_image_dist2(del));
            delta += q_p * charges_[j] *
                     (short_range_pair(kern, r_new) - short_range_pair(kern, dt(k, p, j)));
        }
    }

    // ── Reciprocal space: δ against the cached structure factors ───────────
    sync_delta_cache(configs, k);
    DeltaCache& c = delta_cache_;
    const std::size_t nk = c.n_k, C = n_classes_, cp = particle_class_[p];
    phases_of(cell, old, c.y_re, c.y_im);
    phases_of(cell, candidate, c.z_re, c.z_im);
    const double* rho_re = c.rho_re.data() + k * C * nk;
    const double* rho_im = c.rho_im.data() + k * C * nk;
    for (std::size_t d = 0; d < C; ++d) {
        const std::size_t lo = std::min(cp, d), hi = std::max(cp, d);
        const std::vector<double>& w = kernels_[class_pair_kernel_[lo * C + hi]].setup->weight;
        const double* rr = rho_re + d * nk;
        const double* ri = rho_im + d * nk;
        double sum = 0.0;
        if (d == cp) {
            for (std::size_t kk = 0; kk < nk; ++kk) {
                const double dr = q_p * (c.z_re[kk] - c.y_re[kk]);
                const double di = q_p * (c.z_im[kk] - c.y_im[kk]);
                sum += w[kk] * (2.0 * (dr * rr[kk] + di * ri[kk]) + dr * dr + di * di);
            }
        } else {
            for (std::size_t kk = 0; kk < nk; ++kk) {
                const double dr = q_p * (c.z_re[kk] - c.y_re[kk]);
                const double di = q_p * (c.z_im[kk] - c.y_im[kk]);
                sum += 2.0 * w[kk] * (dr * rr[kk] + di * ri[kk]);
            }
        }
        delta += sum;
    }
    return delta;
}

// One tile of caller-owned scratch; fixed during preparation. Layout mirrors
// the scalar energy ladders, but no thread_local resize occurs in measurement.
struct EwaldTerm::GradientWorkspace final : PotentialGradientWorkspace {
    static constexpr std::size_t tile = 64;
    inline static const char tag = 0;
    SimulationCell cell;
    std::vector<double> e1_re, e1_im, e2_re, e2_im; // [(n*np+p)*tile+t]
    std::vector<double> phase_re, phase_im; // [p*tile+t]
    std::vector<double> rho_re, rho_im; // [class*tile+t]

    GradientWorkspace(const ParticleBatch& batch, const EwaldTerm& term)
        : PotentialGradientWorkspace(batch,term), cell(term.cell_) {
        const auto np=batch.n_particles();
        const auto& s=*term.kernels_.front().setup;
        e1_re.resize(static_cast<std::size_t>(s.n1_max+1)*np*tile);
        e1_im.resize(e1_re.size());
        e2_re.resize(static_cast<std::size_t>(s.n2_max+1)*np*tile);
        e2_im.resize(e2_re.size());
        phase_re.resize(np*tile); phase_im.resize(np*tile);
        rho_re.resize(term.n_classes_*tile); rho_im.resize(rho_re.size());
    }
    bool matches(const ParticleBatch& batch,
                 const HamiltonianComponent& term) const noexcept override {
        const auto actual=batch.cell().vectors(), expected=cell.vectors();
        return PotentialGradientWorkspace::matches(batch,term) &&
            batch.dim()==2 && batch.cell().is_periodic() &&
            actual.size()==expected.size() &&
            std::equal(actual.begin(),actual.end(),expected.begin());
    }
    const void* implementation_tag() const noexcept override { return &tag; }
};

std::unique_ptr<PotentialGradientWorkspace>
EwaldTerm::make_potential_gradient_workspace(const ParticleBatch& batch) const {
    const auto actual=batch.cell().vectors(), expected=cell_.vectors();
    if (batch.n_particles()!=charges_.size() || batch.dim()!=2 ||
        !batch.cell().is_periodic() || actual.size()!=expected.size() ||
        !std::equal(actual.begin(),actual.end(),expected.begin()))
        throw std::invalid_argument("ewald: gradient batch must match the prepared cell and particles");
    for (double q : charges_)
        if (!std::isfinite(q)) throw std::invalid_argument("ewald: nonfinite gradient charge");
    return std::make_unique<GradientWorkspace>(batch,*this);
}

void EwaldTerm::evaluate_potential_gradient(const ParticleBatch& batch,
    std::span<double> out, PotentialGradientWorkspace& workspace,
    std::size_t w_begin, std::size_t w_end) const {
    w_end=workspace.checked_end(batch,*this,out,w_begin,w_end);
    if (workspace.implementation_tag()!=&GradientWorkspace::tag)
        throw std::invalid_argument("ewald: gradient requires its prepared specialized workspace");
    auto& scratch=static_cast<GradientWorkspace&>(workspace);
    if (w_begin==w_end) return;
    const auto nw=batch.num_walkers(), np=batch.n_particles();
    const auto pos=batch.positions(); // [np*2*nw]
    const auto& s=*kernels_.front().setup;

    // Differentiate exactly the real-space branch sampled by evaluate_potential.
    // Cutoff/image switches and fitted panel joins are NOT interface fluxes;
    // their continuity is a separate estimator-admission gate.
    const auto G=groups_.size();
    for (std::size_t g=0; g<G; ++g)
        for (std::size_t h=g; h<G; ++h) {
            const auto& kernel=kernels_[kernel_of(g,h)];
            const auto& sp=kernel.setup->long_range;
            using Kind=numerics::PotentialSpec::Kind;
            for (std::size_t i=groups_[g].first; i<groups_[g].last; ++i)
                for (std::size_t j=g==h ? i+1 : groups_[h].first; j<groups_[h].last; ++j) {
                    const double coefficient=charges_[i]*charges_[j];
                    if (coefficient==0.0) continue;
                    for (std::size_t w=w_begin; w<w_end; ++w) {
                        double delta[2]; // Cartesian in-plane displacement
                        for (std::size_t d=0; d<2; ++d)
                            delta[d]=pos[(2*i+d)*nw+w]-pos[(2*j+d)*nw+w];
                        const double r=std::sqrt(batch.cell().min_image_dist2(delta));
                        if (!std::isfinite(r))
                            throw std::domain_error("ewald: nonfinite gradient separation");
                        if (r>=kernel.setup->r_cut) continue;
                        const double rho=kernel.spec.kind==Kind::Screened ? r :
                            std::sqrt(r*r+kernel.d2);
                        if (r==0.0) {
                            const bool pole=kernel.spec.kind==Kind::Coulomb ? rho==0.0 :
                                kernel.spec.kind==Kind::Screened && kernel.spec.singular_coef!=0.0;
                            if (pole) throw std::domain_error("ewald: gradient at a singular collision");
                            // Smooth softened/clamped pair: rotational symmetry.
                            continue;
                        }
                        double f_long, df_long, d2f_long;
                        sp.eval(r,f_long,df_long,d2f_long);
                        double dv;
                        if (kernel.spec.kind==Kind::Coulomb) {
                            dv=-kernel.inv_eps/(rho*rho)*(r/rho);
                        } else if (kernel.spec.kind==Kind::Screened) {
                            // RadialTable's derivative is exact inside a panel;
                            // the sampled value clamps past r_max (slope zero).
                            dv=-kernel.spec.singular_coef/(r*r);
                            if (r<=kernel.spec.table->r_max()) dv+=kernel.spec.table->d1(r);
                        } else {
                            dv=kernel.keldysh_prefactor*keldysh_cheb_->d1(rho*kernel.keldysh_inv_r0)*
                                kernel.keldysh_inv_r0*(r/rho)-kernel.cutoff_correction.d1(r);
                        }
                        const double radial=coefficient*(dv-df_long);
                        for (std::size_t d=0; d<2; ++d) {
                            const double gradient=radial*(delta[d]/r);
                            if (!std::isfinite(gradient))
                                throw std::domain_error("ewald: nonfinite real-space gradient");
                            out[(2*i+d)*nw+w]+=gradient;
                            out[(2*j+d)*nw+w]-=gradient;
                        }
                    }
                }
        }

    constexpr auto tile=GradientWorkspace::tile;
    const auto C=n_classes_;
    const auto b=cell_.reciprocal();
    for (std::size_t w0=w_begin; w0<w_end; w0+=tile) {
        const auto nt=std::min(tile,w_end-w0);
        auto ladder=[&](double bx,double by,int nmax,std::vector<double>& re,std::vector<double>& im) {
            for (std::size_t i=0; i<np; ++i) {
                const auto row=i*tile;
                for (std::size_t t=0; t<nt; ++t) { re[row+t]=1.0; im[row+t]=0.0; }
                if (nmax<1) continue;
                for (std::size_t t=0; t<nt; ++t) {
                    const double phase=bx*pos[2*i*nw+w0+t]+by*pos[(2*i+1)*nw+w0+t];
                    re[np*tile+row+t]=std::cos(phase);
                    im[np*tile+row+t]=std::sin(phase);
                }
                for (int n=2; n<=nmax; ++n) {
                    const auto previous=(static_cast<std::size_t>(n-1)*np+i)*tile;
                    const auto current=(static_cast<std::size_t>(n)*np+i)*tile;
                    const auto first=(np+i)*tile;
#pragma omp simd
                    for (std::size_t t=0; t<nt; ++t) {
                        re[current+t]=re[previous+t]*re[first+t]-im[previous+t]*im[first+t];
                        im[current+t]=re[previous+t]*im[first+t]+im[previous+t]*re[first+t];
                    }
                }
            }
        };
        ladder(b[0],b[1],s.n1_max,scratch.e1_re,scratch.e1_im);
        ladder(b[2],b[3],s.n2_max,scratch.e2_re,scratch.e2_im);
        for (std::size_t k=0; k<s.n_k(); ++k) {
            for (std::size_t c=0; c<C; ++c)
                for (std::size_t t=0; t<nt; ++t)
                    scratch.rho_re[c*tile+t]=scratch.rho_im[c*tile+t]=0.0;
            const auto n1=static_cast<std::size_t>(s.n1[k]);
            const auto n2=static_cast<std::size_t>(std::abs(s.n2[k]));
            const double sign=s.n2[k]<0 ? -1.0 : 1.0;
            for (std::size_t i=0; i<np; ++i) {
                const auto a=(n1*np+i)*tile, b2=(n2*np+i)*tile, row=i*tile;
                const auto class_row=particle_class_[i]*tile;
#pragma omp simd
                for (std::size_t t=0; t<nt; ++t) {
                    const double real=scratch.e1_re[a+t]*scratch.e2_re[b2+t]-
                        scratch.e1_im[a+t]*sign*scratch.e2_im[b2+t];
                    const double imag=scratch.e1_re[a+t]*sign*scratch.e2_im[b2+t]+
                        scratch.e1_im[a+t]*scratch.e2_re[b2+t];
                    scratch.phase_re[row+t]=real; scratch.phase_im[row+t]=imag;
                    scratch.rho_re[class_row+t]+=charges_[i]*real;
                    scratch.rho_im[class_row+t]+=charges_[i]*imag;
                }
            }
            for (std::size_t i=0; i<np; ++i) {
                if (charges_[i]==0.0) continue;
                const auto c=particle_class_[i];
                for (std::size_t d=0; d<C; ++d) {
                    const double weight=kernels_[class_pair_kernel_[c*C+d]].setup->weight[k];
                    const double factor=-2.0*charges_[i]*weight;
#pragma omp simd
                    for (std::size_t t=0; t<nt; ++t) {
                        const double imag=scratch.phase_im[i*tile+t]*scratch.rho_re[d*tile+t]-
                            scratch.phase_re[i*tile+t]*scratch.rho_im[d*tile+t];
                        out[2*i*nw+w0+t]+=factor*s.kx[k]*imag;
                        out[(2*i+1)*nw+w0+t]+=factor*s.ky[k]*imag;
                    }
                }
            }
        }
    }
    // Constants have zero derivative, but remain in the scalar energy.
    for (std::size_t i=0; i<np; ++i)
        for (std::size_t d=0; d<2; ++d)
            for (std::size_t w=w_begin; w<w_end; ++w)
                if (!std::isfinite(out[(2*i+d)*nw+w]))
                    throw std::domain_error("ewald: nonfinite accumulated gradient");
}

}  // namespace alea
