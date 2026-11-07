// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "hamiltonian/screened_coulomb.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>

#include "fast_screened_coulomb.h"  // C library (self-guards extern "C")
#include "numerics/bessel_j0.h"
#include "numerics/special_functions.h"
#include "population/walker_set.h"

namespace alea {

namespace {

// Compact, locale-independent number formatting for diagnostic messages.
std::string num(double x) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.4g", x);
    return buf;
}

// g(r) = ∫_0^K J0(k r) h(k) dk by fixed Gauss-Legendre nodes, with h(k) sampled
// ONCE at the nodes: the r dependence lives only in J0, so a whole table of g
// costs one ε(k) sweep.  Panels are narrow enough for J0's phase at the largest
// r asked for (≤ 1 rad) and for h's own structure (≥ 200 panels over [0, K]).
struct HankelQuadrature {
    std::vector<double> k, wh;  // nodes and weight·h(node)

    [[nodiscard]] double g(double r) const {
        // Stable setup samples matter for endpoint curvature: summation noise
        // would become unresolved high-order coefficients in the table fit.
        long double s = 0.0L;
        for (std::size_t i = 0; i < k.size(); ++i)
            s += static_cast<long double>(wh[i]) * numerics::bessel_j0(k[i] * r);
        return static_cast<double>(s);
    }
    // g' = −∫ k J1(kr) h dk,  g'' = ∫ k² [J1(kr)/(kr) − J0(kr)] h dk.  Only used
    // past the table edge, which the breakup never asks for; std::cyl_bessel_j
    // is slow but this is not a loop anyone runs.
    [[nodiscard]] double g1(double r) const {
        double s = 0.0;
        for (std::size_t i = 0; i < k.size(); ++i)
            s -= wh[i] * k[i] * std::cyl_bessel_j(1.0, k[i] * r);
        return s;
    }
    [[nodiscard]] double g2(double r) const {
        double s = 0.0;
        for (std::size_t i = 0; i < k.size(); ++i) {
            const double x = k[i] * r;
            s += wh[i] * k[i] * k[i] *
                 (std::cyl_bessel_j(1.0, x) / x - std::cyl_bessel_j(0.0, x));
        }
        return s;
    }
};

constexpr double kGx[10] = {0.0765265211334973, 0.2277858511416451,
                            0.3737060887154195, 0.5108670019508271,
                            0.6360536807265150, 0.7463319064601508,
                            0.8391169718222188, 0.9122344282513259,
                            0.9639719272779138, 0.9931285991850949};
constexpr double kGw[10] = {0.1527533871307258, 0.1491729864726037,
                            0.1420961093183820, 0.1316886384491766,
                            0.1181945319615184, 0.1019301198172404,
                            0.0832767415767048, 0.0626720483341091,
                            0.0406014298003869, 0.0176140071391521};

}  // namespace

// ── The periodic kernel ─────────────────────────────────────────────────────
//
// In Hartree units the library's potential is V(r) = ∫_0^∞ J0(kr)/ε(k) dk, i.e.
// Ṽ(k) = 2π/(k ε(k)).  Writing 1/ε(k) = δ_tc/ε_c + h(k), with ε_c the source
// layer's own constant and δ_tc = 1 for a same-layer pair:
//
//   V(r) = δ_tc/(ε_c r) + g(r),     g(r) = ∫_0^∞ J0(kr) h(k) dk.
//
// h → 0 EXPONENTIALLY at large k -- as e^{−2kδ} (δ the distance to the nearest
// interface) for a same-layer pair, as e^{−kz} across layers -- so g is smooth
// on [0, r_max] and cheap to tabulate.  Past the k = K where |h| has fallen
// below double precision, h is set to exactly 0 in BOTH Ṽ and g, so the two
// stay an exact transform pair.  This is also what lets the kernel work at all:
// the library's transfer matrices call exp(k·d) unguarded and return NaN once
// k·d ≳ 700, far below the 6000/r_c the breakup samples.  The large-k value is
// not extrapolated, it is the analytic limit ε → ε_c (or 1/ε → 0), reached to
// double precision before the overflow; if the overflow comes first, we throw.
numerics::PotentialSpec build_screened_ewald_kernel(
    const std::vector<double>& epsilons, const std::vector<double>& interfaces,
    int source_layer, int target_layer, double z_offset, double r_max) {
    const std::string issue =
        screened_coulomb_stack_issue(epsilons, interfaces, source_layer);
    if (!issue.empty()) throw std::invalid_argument("screened_coulomb: " + issue);
    if (!std::isfinite(r_max) || !(r_max > 0.0))
        throw std::invalid_argument("screened_coulomb Ewald kernel: r_max must be finite and > 0");
    if (!std::isfinite(1.05*r_max))
        throw std::invalid_argument("screened_coulomb Ewald kernel: r_max exceeds the finite table range");

    // Library self-check.  Versions of fast-screened-coulomb before the
    // reflection-coefficient ε(k) lose all precision in interlayer ε(k) near
    // k·z ~ 20 WITHOUT returning NaN, which would silently corrupt the table
    // below.  A uniform stack has the exact answer ε e^{kz}; probe it where the
    // old code already failed (k = 13: it returned a NEGATIVE ε there).
    {
        const double u_eps[] = {2.0, 2.0, 2.0, 2.0}, u_d[] = {-1.0, 1.0, 3.0};
        FSCContext*  probe = fsc_create(4, u_eps, u_d, 2, 3, 2.0);
        const double got   = probe ? fsc_epsilon_k(probe, 13.0) : 0.0;
        if (probe) fsc_free(probe);
        const double want = 2.0 * std::exp(26.0);
        if (!(std::abs(got - want) < 1e-10 * want))
            throw std::runtime_error(
                "screened_coulomb Ewald kernel: this build of fast-screened-coulomb "
                "has the numerically unstable interlayer eps(k) (probe gave " +
                num(got) + ", exact " + num(want) + ").  Update the library to the "
                "reflection-coefficient eps(k) before using screened_coulomb in a "
                "periodic cell.");
    }

    const int     n_layers = static_cast<int>(epsilons.size());
    const double* d        = interfaces.empty() ? nullptr : interfaces.data();
    const std::shared_ptr<FSCContext> ctx(
        fsc_create(n_layers, epsilons.data(), d, source_layer, target_layer, z_offset),
        fsc_free);
    if (!ctx)
        throw std::invalid_argument(
            "screened_coulomb: fsc_create failed (check layer indices/parameters)");

    const bool   direct = target_layer == 0 || target_layer == source_layer;
    const double eps_c  = epsilons[static_cast<std::size_t>(source_layer - 1)];
    const double sing   = direct ? 1.0 / eps_c : 0.0;
    // Long range: a charge anywhere in the stack is seen, far away, through
    // the mean of the two semi-infinite outer media.  Computed from the stack
    // rather than read off ε(k→0), so pairs sharing the outer media compare
    // EQUAL -- EwaldTerm requires one common tail.  Verified against ε(k) below.
    const double eps_L = n_layers == 1 ? epsilons.front()
                                       : 0.5 * (epsilons.front() + epsilons.back());
    const auto inv_eps = [c = ctx.get()](double k) { return 1.0 / fsc_epsilon_k(c, k); };
    if (std::abs(inv_eps(1e-7) - 1.0 / eps_L) > 1e-5 / eps_L)
        throw std::runtime_error(
            "screened_coulomb Ewald kernel: eps(k -> 0) = " + num(1.0 / inv_eps(1e-7)) +
            " is not the outer-media mean " + num(eps_L));

    // ── K: where h has reached its limit to double precision ────────────────
    const double thr = 1e-15 / eps_L;
    const auto   h   = [&](double k) { return inv_eps(k) - sing; };
    const auto   gone = [&](double v) { return !std::isfinite(v) || std::abs(v) < thr; };
    double K = 0.0;
    for (double k = 1e-3; K == 0.0; k *= 1.05) {
        const double hk = h(k);
        if (!std::isfinite(hk) || k > 1e5)
            throw std::runtime_error(
                "screened_coulomb Ewald kernel: eps(k) left double range at k = " +
                num(k) + " before reaching its large-k limit.  The stack's interfaces "
                "are too far from the source for the library's transfer matrices; "
                "a thinner stack (or one with the outermost interfaces closer in) "
                "is needed for the periodic sum.");
        if (std::abs(hk) < thr && gone(h(1.25 * k)) && gone(h(1.6 * k))) K = k;
    }

    // ── g(r) and its table on [0, 1.05 r_max] ───────────────────────────────
    const double R = 1.05 * r_max;
    const double w = std::min(1.0 / R, K / 200.0);
    const double panels=std::ceil(K/w);
    if (!std::isfinite(panels) || panels>std::numeric_limits<int>::max())
        throw std::runtime_error("screened_coulomb Ewald kernel: quadrature panel count exceeds integer range; reduce r_max");
    const int    P = std::max(1, static_cast<int>(panels));
    auto quad = std::make_shared<HankelQuadrature>();
    quad->k.reserve(static_cast<std::size_t>(P) * 20);
    quad->wh.reserve(quad->k.capacity());
    for (int p = 0; p < P; ++p) {
        const double a = K * p / P, b = K * (p + 1) / P;
        const double c = 0.5 * (a + b), hw = 0.5 * (b - a);
        for (int i = 0; i < 10; ++i)
            for (double s : {-1.0, 1.0}) {
                const double k = c + s * hw * kGx[i];
                quad->k.push_back(k);
                quad->wh.push_back(hw * kGw[i] * h(k));
            }
    }
    const auto table = std::make_shared<const numerics::RadialTable>(
        numerics::RadialTable::build([&](double r) { return quad->g(r); }, R,
                                     1e-12 / eps_L));

    numerics::PotentialSpec p;
    p.kind          = numerics::PotentialSpec::Kind::Screened;
    p.epsilon       = eps_L;
    p.singular_coef = sing;
    p.table         = table;
    const std::shared_ptr<const HankelQuadrature> q = quad;
    // Interlayer kernels have no Coulomb pole: evaluate their analytic smooth
    // limit without zero/zero, including radii whose powers underflow. Direct
    // kernels retain the physical +/- infinity at r=0.
    p.v_real = [sing, table, q](double r) {
        return (sing == 0.0 ? 0.0 : sing / r) +
               (r <= table->r_max() ? table->eval(r) : q->g(r));
    };
    p.dv_real = [sing, table, q](double r) {
        return (sing == 0.0 ? 0.0 : -(sing / r) / r) +
               (r <= table->r_max() ? table->d1(r) : q->g1(r));
    };
    p.d2v_real = [sing, table, q](double r) {
        return (sing == 0.0 ? 0.0 : ((2.0 * sing / r) / r) / r) +
               (r <= table->r_max() ? table->d2(r) : q->g2(r));
    };
    p.v_fourier = [ctx, K, sing](double k) {
        return 2.0 * numerics::PI / k * (k < K ? 1.0 / fsc_epsilon_k(ctx.get(), k) : sing);
    };

    // ── k = 0 remainder: c = 2π lim (1/ε(k) − 1/ε_L)/k ──────────────────────
    // Two Richardson levels on a forward difference; the difference itself is
    // O(k) ~ 1e-3, so rounding costs ~1e-13 and truncation ends at O(k³).
    if (n_layers > 1) {
        const auto s  = [&](double k) { return (inv_eps(k) - 1.0 / eps_L) / k; };
        const double k0 = 4e-3;
        const double r1a = 2.0 * s(k0 / 2) - s(k0), r1b = 2.0 * s(k0 / 4) - s(k0 / 2);
        p.k0_offset = 2.0 * numerics::PI * (4.0 * r1b - r1a) / 3.0;
    }
    return p;
}

// Build a C1 smooth-remainder fit using independently refined Fourier moments.
// The additive FSC API checks value and force targets separately. fit_tol is a
// percent target for both observed discrepancies; it is no global error bound.
ScreenedFit build_screened_fit(const std::vector<double>& epsilons,
                               const std::vector<double>& interfaces,
                               int source_layer, int target_layer,
                               double z_offset, double fit_tol) {
    if (!std::isfinite(fit_tol) || fit_tol <= 0.0)
        throw std::invalid_argument("screened_coulomb: fit_tol must be finite and > 0 (percent)");
    const std::string issue=screened_coulomb_stack_issue(epsilons,interfaces,source_layer);
    if (!issue.empty()) throw std::invalid_argument("screened_coulomb: "+issue);
    if (!std::isfinite(z_offset) || target_layer<0 ||
        target_layer>static_cast<int>(epsilons.size()))
        throw std::invalid_argument("screened_coulomb: invalid target layer or layer offset");
    const auto ctx=std::unique_ptr<FSCContext,decltype(&fsc_free)>(
        fsc_create(static_cast<int>(epsilons.size()),epsilons.data(),
            interfaces.empty() ? nullptr : interfaces.data(),source_layer,target_layer,z_offset),&fsc_free);
    if (!ctx) throw std::invalid_argument("screened_coulomb: fsc_create failed");

    // Preserve the existing radial band and pathological-stack setup guard.
    // This legacy estimate chooses only the domain, never reference accuracy.
    double k_max=0.0,rho_max=0.0; int n_quad=0;
    fsc_optimal_params(ctx.get(),1e-4,1e-3,1.5,&k_max,&n_quad,&rho_max);
    if (!std::isfinite(k_max) || k_max>1e3 || !std::isfinite(rho_max) || rho_max<=0.0)
        throw std::runtime_error("screened_coulomb: dielectric response or radial range is unresolved; check the stack geometry");
    constexpr double rho_min=0.01;
    // The fixed 1e-3 Ry extent threshold can fall below rho_min for very weak
    // interactions. A reversed fit band is invalid; retain a positive interval
    // and the analytic pole rather than erase a weak primitive-action singularity.
    rho_max=std::max(rho_max,2*rho_min);
    // Keep the preexisting 5% safety ceiling even for looser requests. The
    // controlled route now applies it independently to both values and forces.
    const double target=std::min(fit_tol,5.0);
    FSCAccuracy accuracy{};
    auto fit=std::shared_ptr<FSCControlledFit>(
        fsc_fit_controlled(ctx.get(),rho_min,rho_max,target,target,&accuracy),&fsc_controlled_free);
    if (!fit || accuracy.status!=0)
        throw std::runtime_error("screened_coulomb: controlled value/force fit failed (status "+
            std::to_string(accuracy.status)+"); check geometry, accuracy targets and numerical resource limits");

    // The represented analytic core retains the exact bulk residue. Finite
    // interlayer separation has no pole; do not infer either from a fitted grid.
    const bool direct=target_layer==0 || target_layer==source_layer;
    const double cusp=direct ? 1.0/epsilons[static_cast<std::size_t>(source_layer-1)] : 0.0;
    std::ostringstream notes;
    notes << std::setprecision(17) << "sampled FSC controlled Hermite"
        << "; requested_value_target_pct=" << fit_tol
        << "; requested_force_target_pct=" << fit_tol
        << "; enforced_value_force_target_pct=" << target
        << "; sampled_value_error_pct=" << accuracy.value_error_pct
        << "; sampled_force_error_pct=" << accuracy.force_error_pct
        << "; quadrature_value_refinement_pct=" << accuracy.quadrature_value_pct
        << "; quadrature_force_refinement_pct=" << accuracy.quadrature_force_pct
        << "; cutoff_value_refinement_pct=" << accuracy.cutoff_value_pct
        << "; cutoff_force_refinement_pct=" << accuracy.cutoff_force_pct
        << "; value_target_met=yes; force_target_met=yes; segments=" << accuracy.segments
        << "; reference_points=" << accuracy.reference_points
        << "; rho<" << rho_min << " and rho>" << rho_max << " frozen"
        << "; sampled checks are not global force certification; fit/quadrature/cutoff/clamp errors excluded from statistical uncertainty";
    return ScreenedFit{std::move(fit),rho_min,rho_max,cusp,fit_tol,
        accuracy.value_error_pct,accuracy.force_error_pct,notes.str()};
}

// Pure geometric/validity diagnosis — see header.  The degeneracy modes were mapped
// empirically against the library: the ETM response ε(k) only converges to ε_c when
// the source charge sits inside a FINITE (interior) source layer with the origin
// strictly between its two interfaces; a semi-infinite (top/bottom) source layer, or
// the origin on/outside its layer, makes ε(k) miss ε_c at every k and the fit stall.
std::string screened_coulomb_stack_issue(const std::vector<double>& epsilons,
                                         const std::vector<double>& interfaces,
                                         int source_layer) {
    const int N = static_cast<int>(epsilons.size());
    if (N < 1)
        return "'epsilons' must list at least one layer";
    for (int i = 0; i < N; ++i)
        if (!std::isfinite(epsilons[i]) || !(epsilons[i] > 0.0))
            return "every layer dielectric constant must be positive (epsilons[" +
                   std::to_string(i) + "] = " + num(epsilons[i]) + ")";
    for (int i=1; i<N; ++i)
        if (epsilons[i]>std::numeric_limits<double>::max()-epsilons[i-1])
            return "adjacent dielectric sums must remain finite for interface reflection";
    if (static_cast<int>(interfaces.size()) != N - 1)
        return "'interfaces' must have exactly n_layers-1 = " + std::to_string(N - 1) +
               " entries (got " + std::to_string(interfaces.size()) + ")";
    if (source_layer < 1 || source_layer > N)
        return "'source_layer' " + std::to_string(source_layer) +
               " is out of range [1, " + std::to_string(N) + "]";

    if (N == 1)
        return "";  // homogeneous medium: no interfaces, always well-conditioned

    for (double interface : interfaces)
        if (!std::isfinite(interface)) return "every interface position must be finite";

    // Interfaces must increase monotonically in z (bottom → top).
    for (int i = 1; i < N - 1; ++i)
        if (!(interfaces[i] > interfaces[i - 1]))
            return "'interfaces' must be strictly increasing in z (interfaces[" +
                   std::to_string(i) + "] = " + num(interfaces[i]) + " <= interfaces[" +
                   std::to_string(i - 1) + "] = " + num(interfaces[i - 1]) + ")";

    // A semi-infinite (top/bottom) source layer makes ε(k) diverge from ε_c.
    if (source_layer == 1 || source_layer == N)
        return "'source_layer' " + std::to_string(source_layer) + " is the " +
               (source_layer == 1 ? std::string("bottom") : std::string("top")) +
               " (semi-infinite) layer; the source must be an INTERIOR layer "
               "(1 < source_layer < n_layers = " + std::to_string(N) + ") so it is a "
               "finite slab - a semi-infinite source makes the screened response eps(k) "
               "fail to converge and the fit stall.";

    // The source charge sits at the origin (z = 0); it must lie STRICTLY inside the
    // interior source layer, i.e. between its lower and upper interfaces.
    const double z_lo = interfaces[source_layer - 2];  // lower bound of source layer
    const double z_hi = interfaces[source_layer - 1];  // upper bound of source layer
    if (!(z_lo < 0.0 && 0.0 < z_hi))
        return "the source charge (origin, z = 0) must lie strictly inside source_layer " +
               std::to_string(source_layer) + ", which spans z = [" + num(z_lo) + ", " +
               num(z_hi) + "]. Shift the interface positions so the source layer "
               "straddles the origin (e.g. a slab spanning [-t/2, +t/2] for thickness t).";

    return "";
}

ScreenedCoulombTerm::ScreenedCoulombTerm(ParticleGroup group_a,
                                         ParticleGroup group_b,
                                         std::vector<double> epsilons,
                                         std::vector<double> interfaces,
                                         int source_layer, int target_layer,
                                         double z_offset, double fit_tol)
    : group_a_(std::move(group_a)),
      group_b_(std::move(group_b)),
      charge_product_(group_a_.charge * group_b_.charge),
      is_intragroup_(group_a_ == group_b_) {
    ScreenedFit r = build_screened_fit(epsilons, interfaces, source_layer,
                                       target_layer, z_offset, fit_tol);
    fit_          = std::move(r.fit);
    rho_min_      = r.rho_min;
    rho_max_      = r.rho_max;
    cusp_residue_ = r.cusp_residue;
    approximation_notes_ = std::move(r.approximation_notes);
    // The frozen extensions and C1 interior have continuous values. Slope
    // kinks have zero surface flux in integration by parts: the identity is
    // for THIS represented potential, with clamp bias reported separately.
    virial_admitted_ = std::isfinite(charge_product_) &&
        group_a_.first <= group_a_.last && group_b_.first <= group_b_.last &&
        (is_intragroup_ || std::max(group_a_.first,group_b_.first) >=
                          std::min(group_a_.last,group_b_.last)) &&
        !(charge_product_*cusp_residue_ < 0.0);
    for (int i=0; i<=1024 && virial_admitted_; ++i) {
        const double rho=rho_min_*std::exp(std::log(rho_max_/rho_min_)*i/1024.0);
        const double bounded=std::clamp(rho,rho_min_,rho_max_);
        virial_admitted_ = std::isfinite(.5*charge_product_*fsc_controlled_eval(fit_.get(),bounded)) &&
                          std::isfinite(.5*charge_product_*fsc_controlled_d1(fit_.get(),bounded));
    }
}

bool ScreenedCoulombTerm::supports_open_centroid_virial() const noexcept {
    return virial_admitted_;
}
bool ScreenedCoulombTerm::supports_periodic_virial(const SimulationCell& cell, std::size_t np) const noexcept {
    // Equal MIC radii imply equal values on Voronoi faces; force jumps do not
    // introduce a boundary flux for a continuous periodic weight.
    return cell.is_periodic() && virial_admitted_ && group_a_.last <= np && group_b_.last <= np;
}

std::string ScreenedCoulombTerm::potential_approximation_notes() const {
    return approximation_notes_;
}

std::vector<RadialPairPotential> ScreenedCoulombTerm::radial_pair_potentials() const {
    const double coefficient=.5*charge_product_,lo=rho_min_,hi=rho_max_;
    if (!std::isfinite(coefficient)) throw std::invalid_argument("FSC pair action: nonfinite charge coefficient");
    const auto fit=fit_;
    return {{group_a_,group_b_,[coefficient,lo,hi,fit](double r) {
        return coefficient*fsc_controlled_eval(fit.get(),std::clamp(r,lo,hi));
    },0.,approximation_notes_}}; // The represented clamp has no 1/r pole.
}

double ScreenedCoulombTerm::coulomb_cusp_residue(const ParticleGroup& a,
                                                 const ParticleGroup& b) const {
    // Match the requested pair against this term's groups in either order.
    const bool same = (a == group_a_ && b == group_b_) ||
                      (a == group_b_ && b == group_a_);
    if (!same) return 0.0;
    // R = q_a q_b · lim_{r→0} r·½·V_fit(r): nonzero only for an intralayer pair whose
    // bulk-Coulomb core diverges (finite interlayer separation gives exactly 0).
    return charge_product_ * cusp_residue_;
}

void ScreenedCoulombTerm::evaluate_potential(const ParticleBatch& batch,
                                         std::span<double> v_out,
                                         std::size_t w_begin, std::size_t w_end) const {
    const std::size_t nw = batch.num_walkers();  // global stride
    if (w_end > nw) w_end = nw;
    const BatchedDistanceTable& dt = batch.dist_tables();
    double* __restrict le = v_out.data();

    const FSCControlledFit* fit     = fit_.get();
    const double  rho_min = rho_min_;
    const double  rho_max = rho_max_;

    // Accumulate coeff·V_fit(ρ_ij) over the contiguous walker row of pair (i,j).
    // coeff already folds the ½ Rydberg→Hartree conversion into the charge
    // product.  ρ is the IN-PLANE distance — the stack screening ε(k) and the
    // layer separation are baked into the fit — clamped to the fit's validity band
    // [ρ_min, ρ_max] (the controlled fit is only defined there; mirrors the Keldysh
    // x_lo/x_hi clamp).  NOTE: fsc_controlled_eval is a C call, so this inner loop stays
    // scalar. Evaluation now includes log(rho) and segment lookup; throughput
    // needs measurement before any performance claim.
    auto add_pair = [&](double coeff, std::size_t i, std::size_t j) {
        const double* __restrict r = dt.row(i, j).data();
        for (std::size_t w = w_begin; w < w_end; ++w) {
            const double rho = r[w] < rho_min ? rho_min
                                              : (r[w] > rho_max ? rho_max : r[w]);
            le[w] += coeff * fsc_controlled_eval(fit, rho);
        }
    };

    const double coeff = 0.5 * charge_product_;  // ½ q_a q_b (Ry → Hartree)
    if (is_intragroup_) {
        // Intra-group: V = Σ_{i<j in A} ½ q² · V_fit(r_ij).
        for (std::size_t i = group_a_.first; i < group_a_.last; ++i)
            for (std::size_t j = i + 1; j < group_a_.last; ++j)
                add_pair(coeff, i, j);
    } else {
        // Inter-group: V = Σ_{i∈A, j∈B} ½ q_a q_b · V_fit(r_ij).
        for (std::size_t i = group_a_.first; i < group_a_.last; ++i)
            for (std::size_t j = group_b_.first; j < group_b_.last; ++j)
                add_pair(coeff, i, j);
    }
}

bool ScreenedCoulombTerm::supports_potential_gradient() const noexcept {
    return static_cast<bool>(fit_);
}

std::unique_ptr<PotentialGradientWorkspace>
ScreenedCoulombTerm::make_potential_gradient_workspace(const ParticleBatch& batch) const {
    if (group_a_.first > group_a_.last || group_b_.first > group_b_.last ||
        group_a_.last > batch.n_particles() || group_b_.last > batch.n_particles() ||
        !std::isfinite(charge_product_) || !std::isfinite(rho_min_) ||
        !std::isfinite(rho_max_) || rho_min_ <= 0.0 || rho_max_ <= rho_min_)
        throw std::invalid_argument("screened_coulomb: invalid gradient groups or parameters");
    return HamiltonianComponent::make_potential_gradient_workspace(batch);
}

void ScreenedCoulombTerm::evaluate_potential_gradient(const ParticleBatch& batch,
    std::span<double> out, PotentialGradientWorkspace& workspace,
    std::size_t w_begin, std::size_t w_end) const {
    w_end = workspace.checked_end(batch, *this, out, w_begin, w_end);
    const auto nw = batch.num_walkers(), dim = batch.dim();
    const auto pos = batch.positions(); // [np*dim*nw]
    const double coefficient = 0.5 * charge_product_; // Ry -> Hartree, once
    if (coefficient == 0.0) return;
    for (std::size_t i = group_a_.first; i < group_a_.last; ++i)
        for (std::size_t j = is_intragroup_ ? i+1 : group_b_.first; j < group_b_.last; ++j)
            for (std::size_t w = w_begin; w < w_end; ++w) {
                double delta[SimulationCell::kMaxDim]; // [dim]
                for (std::size_t d = 0; d < dim; ++d)
                    delta[d] = pos[(i*dim+d)*nw+w] - pos[(j*dim+d)*nw+w];
                const double rho = std::sqrt(batch.cell().min_image_dist2(delta));
                if (!std::isfinite(rho))
                    throw std::domain_error("screened_coulomb: nonfinite gradient separation");
                // The entire sampled potential is constant outside this band,
                // including coincidence. Layer separation is already in the fit.
                if (rho < rho_min_ || rho > rho_max_) continue;
                const double radial = coefficient * fsc_controlled_d1(fit_.get(), rho);
                if (!std::isfinite(radial))
                    throw std::domain_error("screened_coulomb: nonfinite radial gradient");
                for (std::size_t d = 0; d < dim; ++d) {
                    const double gradient = radial * (delta[d]/rho);
                    if (!std::isfinite(gradient))
                        throw std::domain_error("screened_coulomb: nonfinite spatial gradient");
                    out[(i*dim+d)*nw+w] += gradient;
                    out[(j*dim+d)*nw+w] -= gradient;
                }
            }
}

// Changed pairs only; retain the fitted kernel's existing validity-band clamp.
double ScreenedCoulombTerm::delta_potential_particle(ParticleBatch& configs,
    std::size_t p, std::size_t k, std::span<const double> proposed,
    std::span<double>) const {
    return radial_particle_delta(configs, p, k, proposed, group_a_, group_b_,
        [&](double r) {
            const double rho = r < rho_min_ ? rho_min_ : (r > rho_max_ ? rho_max_ : r);
            return 0.5 * charge_product_ * fsc_controlled_eval(fit_.get(), rho);
        });
}

}  // namespace alea
