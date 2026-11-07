// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// `- moire_jastrow:` items → a one-body MoireJastrow on one group.
//
// log ψ = Σ_i Σ_s Σ_j [a_s cos(k_sj·r_i) + b_s sin(k_sj·r_i)] over the first
// `shells` moiré reciprocal shells (see wavefunction/moire_jastrow.h).
#include "builders/wavefunction_builders.h"

#include <stdexcept>
#include <utility>

#include "hamiltonian/moire.h"
#include "wavefunction/moire_jastrow.h"

namespace alea {

std::pair<double, double> moire_jastrow_geometry(const io::Config& cfg,
                                                 const io::MoireJastrowSpec& m) {
    if (m.a_M && m.theta0) return {*m.a_M, *m.theta0};
    if (cfg.moires.empty() && !m.a_M)
        throw std::runtime_error(
            "moire_jastrow on group " + m.group + " has no `a_M` and there is no "
            "`moire` Hamiltonian term to inherit it from.");
    double a_M = 0.0, theta0 = 0.0;
    if (!cfg.moires.empty()) {
        a_M    = cfg.moires.front().a_M;
        theta0 = cfg.moires.front().theta0;
        for (const io::MoireSpec& t : cfg.moires)
            if (t.a_M != a_M || t.theta0 != theta0)
                throw std::runtime_error(
                    "moire_jastrow on group " + m.group + " inherits its "
                    "superlattice, but the `moire` terms disagree on a_M/theta0; "
                    "set `a_M` and `theta0` on the jastrow explicitly.");
    }
    return {m.a_M.value_or(a_M), m.theta0.value_or(theta0)};
}

std::unique_ptr<WavefunctionComponent> build_component(
    const io::MoireJastrowSpec& m, const WavefunctionContext& ctx) {
    const ParticleGroup& g = find_group(ctx.groups, m.group);
    if (m.shells < 1 || m.shells > 3)
        throw std::runtime_error("moire_jastrow: shells must be 1, 2 or 3 (got " +
                                 std::to_string(m.shells) + ")");
    const std::size_t n_par = 2 * static_cast<std::size_t>(m.shells);
    if (!m.params.empty() && m.params.size() != n_par)
        throw std::runtime_error(
            "moire_jastrow on group " + m.group + ": " +
            std::to_string(m.shells) + " shell(s) need " + std::to_string(n_par) +
            " params [a_1, b_1, ...], got " + std::to_string(m.params.size()));
    const auto [a_M, theta0] = moire_jastrow_geometry(ctx.cfg, m);

    // Named a_1, b_1, a_2, ...; empty params → all zero (ψ starts flat).
    std::vector<Parameter> params;
    params.reserve(n_par);
    for (std::size_t i = 0; i < n_par; ++i) {
        const double value = m.params.empty() ? 0.0 : m.params[i];
        const bool   fixed = i < m.is_fixed.size() ? m.is_fixed[i] : true;
        const std::string nm =
            std::string(i % 2 == 0 ? "a" : "b") + std::to_string(i / 2 + 1);
        params.push_back(Parameter{nm, value, fixed, /*sign_locked=*/false});
    }
    return std::make_unique<MoireJastrow>(
        ctx.dim,
        moire_jastrow_shell_vectors(moire_reciprocal_vectors(a_M, theta0, ctx.dim),
                                    static_cast<std::size_t>(m.shells), ctx.dim),
        std::vector<std::pair<std::size_t, std::size_t>>{{g.first, g.last}},
        std::move(params));
}

}  // namespace alea
