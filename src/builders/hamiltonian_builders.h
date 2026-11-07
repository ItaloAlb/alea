// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Config → Hamiltonian components.  build_hamiltonian (qmc/app.cpp) only
// orchestrates:
//
//   kinetic (with the uniform-field gauge) + traps + centrifugal trap
//   + pair interactions: ONE Ewald term for a periodic run, else one open-
//     boundary term per spec
//   + inverse-power pairs (C/r^p): one term per spec in any cell
//   + one-body moiré potentials
//
// A pair-potential family (Coulomb, Rytova–Keldysh, screened Coulomb) lives in
// one file and provides both of its forms: build_component(spec) for open
// boundaries, and ewald_kernel(specs, a, b) — its kernel for one group pair —
// which build_ewald assembles into the single cell-wide Ewald term.  Adding a
// family means one such file plus one line in build_ewald's family dispatch.

#include <memory>
#include <span>
#include <vector>

#include "../hamiltonian/hamiltonian_component.h"
#include "../io/parser.h"
#include "../numerics/ewald_breakup.h"
#include "../population/particle_group.h"
#include "../population/simulation_cell.h"
#include "common.h"

namespace alea {

// Everything a Hamiltonian builder may read besides its own spec.
struct HamiltonianContext {
    const io::Config&                 cfg;
    const std::vector<ParticleGroup>& groups;
    const SimulationCell&             cell;
    std::size_t                       dim;
};

// ── Kinetic energy, confinement, fields ───────────────────────────────────

// −½Σ∇²/m, carrying the vector potential of `magnetic_field` / `rotation` when
// either is present (symmetric gauge: open 2-D cells only).
[[nodiscard]] std::unique_ptr<HamiltonianComponent> build_kinetic(
    const HamiltonianContext& ctx);

// The −½mΩ²r² a rotating frame adds; nullptr when there is no `rotation`.
[[nodiscard]] std::unique_ptr<HamiltonianComponent> build_centrifugal_trap(
    const HamiltonianContext& ctx);

[[nodiscard]] std::unique_ptr<HamiltonianComponent> build_component(
    const io::HarmonicTrapSpec& spec, const HamiltonianContext& ctx);
[[nodiscard]] std::unique_ptr<HamiltonianComponent> build_component(
    const io::MoireSpec& spec, const HamiltonianContext& ctx);

// ── Pair interactions ─────────────────────────────────────────────────────

// Open-boundary pair terms, one per spec.
[[nodiscard]] std::unique_ptr<HamiltonianComponent> build_component(
    const io::CoulombSpec& spec, const HamiltonianContext& ctx);
[[nodiscard]] std::unique_ptr<HamiltonianComponent> build_component(
    const io::KeldyshSpec& spec, const HamiltonianContext& ctx);
[[nodiscard]] std::unique_ptr<HamiltonianComponent> build_component(
    const io::ScreenedCoulombSpec& spec, const HamiltonianContext& ctx);

// V = C/r^p (p > d under PBC).  Built per spec in open and periodic cells
// alike: its image sum converges absolutely, so it is not an Ewald family.
[[nodiscard]] std::unique_ptr<HamiltonianComponent> build_component(
    const io::InversePowerSpec& spec, const HamiltonianContext& ctx);

// A family's Ewald kernel for the group pair (a, b).  Under PBC every pair of
// groups interacts, listed in a spec or not, so each family decides what an
// unlisted pair gets.  `specs` is non-empty.
[[nodiscard]] numerics::PotentialSpec ewald_kernel(
    std::span<const io::CoulombSpec> specs, const ParticleGroup& a,
    const ParticleGroup& b, const HamiltonianContext& ctx);
[[nodiscard]] numerics::PotentialSpec ewald_kernel(
    std::span<const io::KeldyshSpec> specs, const ParticleGroup& a,
    const ParticleGroup& b, const HamiltonianContext& ctx);
[[nodiscard]] numerics::PotentialSpec ewald_kernel(
    std::span<const io::ScreenedCoulombSpec> specs, const ParticleGroup& a,
    const ParticleGroup& b, const HamiltonianContext& ctx);

// The single Ewald term replacing every pair spec of a periodic run, or
// nullptr when the pair terms stay per-spec (open cell, `minimum_image`
// truncation, or no pair specs at all).  Refuses mixed families and — for
// Coulomb — differing epsilon: one structure factor, one dielectric constant.
[[nodiscard]] std::unique_ptr<HamiltonianComponent> build_ewald(
    const HamiltonianContext& ctx);

}  // namespace alea
