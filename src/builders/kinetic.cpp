// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Kinetic energy (with the uniform-field gauge), harmonic traps, and the
// centrifugal trap of a rotating frame.
#include "builders/hamiltonian_builders.h"

#include <stdexcept>

#include "hamiltonian/harmonic_trap.h"
#include "hamiltonian/kinetic.h"

namespace alea {

namespace {

// Uniform B / rotation along z enter as a vector potential in the kinetic term.
// Symmetric gauge about the origin, so only open cells: on a torus a uniform B
// needs flux quantization and magnetic translations instead.
GaugeField gauge_field(const HamiltonianContext& ctx) {
    GaugeField gauge;
    if (ctx.cfg.magnetic_field) gauge.B = *ctx.cfg.magnetic_field;
    if (ctx.cfg.rotation_omega) gauge.omega = *ctx.cfg.rotation_omega;
    if (gauge.active()) {
        if (ctx.cfg.lattice.type != "open")
            throw std::runtime_error(
                "magnetic_field / rotation need an open lattice: the symmetric "
                "gauge A = ½B ẑ×r is not periodic (a torus needs flux "
                "quantization and magnetic translations, not implemented).");
        if (ctx.cfg.lattice.dimension != 2)
            throw std::runtime_error(
                "magnetic_field / rotation are implemented in 2-D (field along z)");
    }
    return gauge;
}

}  // namespace

std::unique_ptr<HamiltonianComponent> build_kinetic(const HamiltonianContext& ctx) {
    return std::make_unique<Kinetic>(ctx.groups, gauge_field(ctx));
}

std::unique_ptr<HamiltonianComponent> build_centrifugal_trap(
    const HamiltonianContext& ctx) {
    const double omega = ctx.cfg.rotation_omega.value_or(0.0);
    if (omega == 0.0) return nullptr;
    return std::make_unique<HarmonicTrap>(ctx.groups, -omega * omega);
}

// V = Σ ½ m ω² |r − center|² on the listed groups (empty → every particle).
std::unique_ptr<HamiltonianComponent> build_component(
    const io::HarmonicTrapSpec& t, const HamiltonianContext& ctx) {
    std::vector<ParticleGroup> trapped;
    if (t.groups.empty()) trapped = ctx.groups;
    for (const std::string& gname : t.groups)
        trapped.push_back(find_group(ctx.groups, gname));
    if (!t.center.empty() && t.center.size() != ctx.dim)
        throw std::runtime_error("harmonic_trap: center has the wrong dimension");
    return std::make_unique<HarmonicTrap>(trapped, t.omega * t.omega, t.center);
}

}  // namespace alea
