// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// `- slater:` items → a real or complex Slater determinant over one group.
//
// ψ_det = det(D_up)·det(D_down) over a fermion group.  Block sizes come from
// that group's particle up/down counts (indices [first,first+up) spin-up, the
// rest spin-down); the orbitals are a shared restricted pool, so the up/down
// blocks use the first up/down orbitals — except a crystal, whose spin blocks
// take disjoint sites.
#include "builders/wavefunction_builders.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "wavefunction/backflow_slater_determinant.h"
#include "wavefunction/complex_orbital_set.h"
#include "wavefunction/complex_slater_determinant.h"
#include "wavefunction/orbital_set.h"
#include "wavefunction/slater_determinant.h"

namespace alea {

bool is_crystal_slater(const io::SlaterSpec& s) {
    return !s.sites.empty() ||
           (!s.centers.empty() && s.alpha.has_value() && s.alphas.empty());
}

std::vector<double> lattice_sites(const std::string& sites,
                                  const std::vector<std::vector<double>>& centers,
                                  const ParticleGroup& g, const SimulationCell& cell,
                                  std::size_t dim, const std::string& label) {
    if (!sites.empty()) {
        if (sites != "triangular")
            throw std::runtime_error(label + ": unknown site lattice '" + sites +
                                     "' (only 'triangular')");
        return commensurate_triangular_sites(cell, g.size());
    }
    if (centers.size() != g.size())
        throw std::runtime_error(
            label + ": `centers` must list one site per particle of group " + g.name +
            ": got " + std::to_string(centers.size()) + " for " + std::to_string(g.size()));
    std::vector<double> out;
    out.reserve(g.size() * dim);
    for (const std::vector<double>& c : centers) {
        if (c.size() != dim)
            throw std::runtime_error(label + ": center dimension mismatch for group " +
                                     g.name);
        out.insert(out.end(), c.begin(), c.end());
    }
    return out;
}

// Explicit centers must also be pairwise distinct modulo the cell: two centers
// on the same periodic image make two identical rows and a singular
// determinant, which is exactly what an unchecked hand-placed list under PBC
// can produce.
std::vector<double> crystal_sites(const io::SlaterSpec& s, const ParticleGroup& g,
                                  const SimulationCell& cell, std::size_t dim) {
    const std::vector<double> out =
        lattice_sites(s.sites, s.centers, g, cell, dim, "slater crystal");
    if (s.sites.empty() && cell.is_periodic()) {
        const double tol2 = 1e-16 * cell.volume();
        for (std::size_t i = 0; i < g.size(); ++i)
            for (std::size_t j = i + 1; j < g.size(); ++j) {
                double del[SimulationCell::kMaxDim] = {};
                for (std::size_t d = 0; d < dim; ++d)
                    del[d] = out[i * dim + d] - out[j * dim + d];
                if (cell.min_image_dist2(del) < tol2)
                    throw std::runtime_error(
                        "slater: crystal centers " + std::to_string(i) + " and " +
                        std::to_string(j) + " of group " + s.group +
                        " are the same site modulo the cell; the determinant "
                        "would be singular.");
            }
    }
    return out;
}

namespace {

// The spin blocks of group g: up on [first, first+n_up), down after it.  `make`
// builds the orbital set of a block from (first orbital, orbital count).
template <class Block, class MakeOrbitals>
std::vector<Block> spin_blocks(const ParticleGroup& g, std::size_t n_up,
                               std::size_t n_dn, MakeOrbitals&& make) {
    std::vector<Block> blocks;
    if (n_up > 0) blocks.push_back({g.first, g.first + n_up, make(true, n_up)});
    if (n_dn > 0)
        blocks.push_back({g.first + n_up, g.first + n_up + n_dn, make(false, n_dn)});
    return blocks;
}

// `alpha_fixed: false` unpins the determinant's orbital widths, its first
// n_orbital parameters (a backflow determinant lists its channels after them).
// One that exposes none is refused rather than silently optimising nothing.
void unpin_widths(const io::SlaterSpec& s, WavefunctionComponent& det,
                  std::size_t n_orbital) {
    if (s.alpha_fixed) return;
    const std::span<Parameter> params = det.parameters().first(n_orbital);
    if (params.empty())
        throw std::runtime_error(
            "slater: `alpha_fixed: false` on group " + s.group +
            " but these orbitals expose no width to optimise — plane waves "
            "take their k from the cell, and per-orbital `alphas` are not "
            "one shared width.  Use `sites`/`alpha`, or a uniform `alphas` "
            "list, if the width is meant to be variational.");
    for (Parameter& p : params) p.is_fixed = false;
}

// The `backflow:` channels of a slater item: the source group's particle range,
// the free coefficients and the cutoff L (its own parameter, last).  Whether c_1
// is derived (a like-spin channel) is the component's rule, not the config's.
std::vector<BackflowSlaterDeterminant::Channel> backflow_channels(
    const io::SlaterSpec& s, const WavefunctionContext& ctx) {
    std::vector<BackflowSlaterDeterminant::Channel> channels;
    for (const io::BackflowSpec& b : s.backflow) {
        const ParticleGroup& src = find_group(ctx.groups, b.from);
        BackflowSlaterDeterminant::Channel ch;
        ch.source_first = src.first;
        ch.source_last = src.last;
        ch.truncation = b.truncation;
        for (std::size_t k = 0; k < b.params.size(); ++k)
            ch.params.push_back(
                Parameter{"", b.params[k], b.is_fixed.empty() ? false : b.is_fixed[k]});
        double cutoff = 0.0;
        if (b.cutoff) {
            cutoff = *b.cutoff;
        } else if (ctx.cell.is_periodic()) {
            cutoff = ctx.cell.ws_radius();
        } else {
            throw std::runtime_error(
                "slater on group " + s.group + ": the backflow channel from " + b.from +
                " needs a `cutoff` in an open cell (under periodic boundaries it "
                "defaults to the Wigner-Seitz inradius)");
        }
        ch.params.push_back(Parameter{"", cutoff, b.cutoff_fixed});
        channels.push_back(std::move(ch));
    }
    return channels;
}

// The real determinant of a group's spin blocks: plain, or with backflow
// when the item lists `backflow:` channels.
template <class MakeOrbitals>
std::unique_ptr<WavefunctionComponent> real_determinant(
    const io::SlaterSpec& s, const ParticleGroup& g, std::size_t n_up, std::size_t n_dn,
    const WavefunctionContext& ctx, MakeOrbitals&& make) {
    if (s.backflow.empty()) {
        auto det = std::make_unique<SlaterDeterminant>(
            spin_blocks<SlaterDeterminant::Block>(g, n_up, n_dn, make));
        unpin_widths(s, *det, det->parameters().size());
        return det;
    }
    try {
        auto det = std::make_unique<BackflowSlaterDeterminant>(
            spin_blocks<BackflowSlaterDeterminant::Block>(g, n_up, n_dn, make),
            backflow_channels(s, ctx), ctx.cell);
        unpin_widths(s, *det, det->n_orbital_parameters());
        return det;
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error("slater on group " + s.group + ": " + e.what());
    }
}

// Backflow is implemented for real determinants only; a complex one refuses it
// rather than silently dropping the channels.
void refuse_complex_backflow(const io::SlaterSpec& s, const std::string& why) {
    if (!s.backflow.empty())
        throw std::runtime_error(
            "slater on group " + s.group + ": backflow is implemented for real "
            "determinants (plane_wave at the Gamma twist, gaussian), not for " + why);
}

// Fock–Darwin: complex angular-momentum orbitals (field/rotation).  One pool of
// l values; each spin block takes it from the front, like a shared Gaussian
// pool.  Complex → DMC runs fixed phase.
std::unique_ptr<WavefunctionComponent> build_fock_darwin(
    const io::SlaterSpec& s, const ParticleGroup& g, std::size_t n_up,
    std::size_t n_dn, const WavefunctionContext& ctx) {
    refuse_complex_backflow(s, "complex fock_darwin orbitals");
    if (ctx.dim != 2)
        throw std::runtime_error("slater: fock_darwin orbitals are 2-D");
    if (!s.beta || *s.beta <= 0.0)
        throw std::runtime_error(
            "slater: fock_darwin orbitals need a positive `beta` "
            "(the exact one-body value is m·sqrt(ω₀² + ω_c²/4))");
    if (s.ls.size() < std::max(n_up, n_dn))
        throw std::runtime_error(
            "slater: group " + s.group + " needs at least " +
            std::to_string(std::max(n_up, n_dn)) + " values in `l`, got " +
            std::to_string(s.ls.size()));
    return std::make_unique<ComplexSlaterDeterminant>(
        spin_blocks<ComplexSlaterDeterminant::Block>(
            g, n_up, n_dn, [&](bool, std::size_t k) {
                return std::make_unique<FockDarwinOrbitalSet>(
                    std::vector<int>(s.ls.begin(), s.ls.begin() + k), *s.beta);
            }));
}

// Plane waves: the uniform-fluid phase.  Each spin block fills complete shells
// of the cell's reciprocal lattice; make_plane_wave_orbitals refuses an open
// shell and names the valid particle counts, because a partly filled shell
// makes the energy depend on an arbitrary choice among degenerate k-points.
std::unique_ptr<WavefunctionComponent> build_plane_wave(
    const io::SlaterSpec& s, const ParticleGroup& g, std::size_t n_up,
    std::size_t n_dn, const WavefunctionContext& ctx) {
    if (!s.centers.empty() || !s.alphas.empty() || !s.sites.empty())
        throw std::runtime_error(
            "slater: plane-wave orbitals take no centers/alphas/sites — "
            "the k-points come from the cell's reciprocal lattice.");
    const std::vector<double>& twist = ctx.cfg.lattice.twist;
    // Twisted boundary conditions → complex determinant.  Away from Γ the
    // occupied k+θ are no longer closed under k → −k, so no real cos/sin basis
    // exists: build e^{i(k+θ)·r} orbitals and a complex determinant (DMC then
    // runs fixed phase).  An integer twist IS Γ and keeps the cheaper real path.
    if (!is_gamma_twist(twist)) {
        refuse_complex_backflow(s, "twisted (complex) plane waves");
        return std::make_unique<ComplexSlaterDeterminant>(
            spin_blocks<ComplexSlaterDeterminant::Block>(
                g, n_up, n_dn, [&](bool, std::size_t k) {
                    return make_twisted_plane_wave_orbitals(ctx.cell, k, twist);
                }));
    }
    return real_determinant(s, g, n_up, n_dn, ctx, [&](bool, std::size_t k) {
        return make_plane_wave_orbitals(ctx.cell, k);
    });
}

// Gaussians: explicit centers (a molecule's shared orbital pool), or a crystal
// with one site per particle.
std::unique_ptr<WavefunctionComponent> build_gaussian(
    const io::SlaterSpec& s, const ParticleGroup& g, std::size_t n_up,
    std::size_t n_dn, const WavefunctionContext& ctx) {
    const std::size_t dim = ctx.dim;
    std::vector<double> pool;      // [n_orb * dim] centers
    std::vector<double> pool_a;    // [n_orb] widths
    bool disjoint_blocks = false;  // one electron per site, not a shared pool
    if (!s.sites.empty() && (!s.centers.empty() || !s.alphas.empty()))
        throw std::runtime_error(
            "slater: `sites` generates the orbital centers, so it cannot "
            "be combined with explicit centers/alphas.");
    if (!s.centers.empty() && s.alpha && !s.alphas.empty())
        throw std::runtime_error(
            "slater: give either one shared `alpha` (crystal: one site per "
            "particle) or per-orbital `alphas` (orbital pool), not both.");
    if (is_crystal_slater(s)) {
        if (!s.alpha || *s.alpha <= 0.0)
            throw std::runtime_error(
                "slater: a crystal determinant needs a positive `alpha` (the "
                "Gaussian width shared by every site).");
        pool = crystal_sites(s, g, ctx.cell, dim);
        pool_a.assign(g.size(), *s.alpha);
        // A Wigner-crystal site holds ONE electron, so the spin blocks take
        // DISJOINT sites (up on the first n_up, down on the rest) rather than
        // the shared restricted pool that a molecule's orbitals use.
        disjoint_blocks = true;
        if (n_up + n_dn != g.size())
            throw std::runtime_error(
                "slater: a crystal determinant has one orbital per particle "
                "of group " + s.group + ", so up+down must equal the group size.");
    } else {
        const std::size_t n_orb = s.alphas.size();
        if (s.centers.size() != n_orb)
            throw std::runtime_error("slater: centers/alphas length mismatch "
                                     "for group " + s.group);
        if (n_orb < std::max(n_up, n_dn))
            throw std::runtime_error("slater: group " + s.group +
                " needs at least " + std::to_string(std::max(n_up, n_dn)) +
                " orbitals, got " + std::to_string(n_orb));
        pool.reserve(n_orb * dim);
        for (std::size_t i = 0; i < n_orb; ++i) {
            if (s.centers[i].size() != dim)
                throw std::runtime_error("slater: orbital center dimension "
                    "mismatch for group " + s.group);
            pool.insert(pool.end(), s.centers[i].begin(), s.centers[i].end());
        }
        pool_a = s.alphas;
    }

    // A GaussianOrbitalSet over k orbitals: the up block starts at orbital 0;
    // the down block too for a shared pool, or after the up sites for a crystal.
    auto make_orbitals = [&](bool up, std::size_t k) {
        const std::size_t off = (up || !disjoint_blocks) ? 0 : n_up;
        std::vector<double> centers(pool.begin() + off * dim,
                                    pool.begin() + (off + k) * dim);
        std::vector<double> alphas(pool_a.begin() + off, pool_a.begin() + off + k);
        return std::make_unique<GaussianOrbitalSet>(dim, std::move(centers),
                                                    std::move(alphas), ctx.cell);
    };
    return real_determinant(s, g, n_up, n_dn, ctx, make_orbitals);
}

}  // namespace

std::unique_ptr<WavefunctionComponent> build_component(
    const io::SlaterSpec& s, const WavefunctionContext& ctx) {
    const ParticleGroup& g = find_group(ctx.groups, s.group);
    const io::ParticleSpec* ps = nullptr;
    for (const io::ParticleSpec& p : ctx.cfg.particles)
        if (p.group == s.group) { ps = &p; break; }
    if (!ps)
        throw std::runtime_error("slater: no particle declares group " + s.group);
    // VMC/DMC take the statistics from ψ: a determinant makes the group
    // antisymmetric whatever `type` says.  For bosons that is the wrong
    // symmetry; their trial is a positive Jastrow (times a `nosanow` factor
    // for a crystal).
    if (ps->type == "bosons")
        throw std::runtime_error(
            "slater on group " + s.group + ", whose particles are bosons: a "
            "determinant is antisymmetric.  A bosonic trial is a product of "
            "positive factors (jastrow, nosanow).");
    if (g.size() == 0)
        throw std::runtime_error("slater: group " + s.group +
                                 " has no particles, so there is no determinant");
    const std::size_t n_up = static_cast<std::size_t>(ps->up);
    const std::size_t n_dn = static_cast<std::size_t>(ps->down);

    if (s.orbital_type == "fock_darwin") return build_fock_darwin(s, g, n_up, n_dn, ctx);
    if (s.orbital_type == "plane_wave")  return build_plane_wave(s, g, n_up, n_dn, ctx);
    if (s.orbital_type == "gaussian")    return build_gaussian(s, g, n_up, n_dn, ctx);
    throw std::runtime_error("slater: unsupported orbital type '" + s.orbital_type +
                             "' (only 'gaussian', 'plane_wave' or 'fock_darwin')");
}

}  // namespace alea
