// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Exchange classes for PIMC: the sets of mutually indistinguishable particles.
//
// Two particles can exchange only if they share a group (same mass, charge
// and layer), a quantum statistics (fermions or bosons) and a flavour
// (spin × valley × …).  A flavour-blind Hamiltonian conserves every class
// count, so the classes are fixed for a run; the permutation of a closed path
// factorizes as P = Π_c P_c over classes, and the fermion sign is
// Π_c (−1)^{P_c} over the fermionic ones (Ceperley, Rev. Mod. Phys. 67, 279
// (1995), Sec. V).
//
// Particle indices follow build_groups: entries in YAML order, and inside an
// entry flavour 0's particles first, then flavour 1's, … (the {up, down} form
// puts spin-up first, as the Slater builder assumes).  An entry repeated for
// the same group adds to that group's classes by flavour index, so a class
// need not be contiguous; its indices are stored explicitly.
//
// Boltzmannon entries are distinguishable and form no class.  An entry
// without `type` counts as fermions, so a missing type can never bypass the
// PIMC statistics gate.

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "../io/parser.h"
#include "../population/exchange_class.h"

namespace alea {

// Classes ordered by their first particle index; classes with no particles
// (a zero flavour count) are omitted.  Throws std::runtime_error for an
// unknown particle type or for different types within one group.
[[nodiscard]] std::vector<ExchangeClass> build_exchange_classes(
    std::span<const io::ParticleSpec> particles);

// Size of the largest class; 1 when there is none (nothing can exchange).
[[nodiscard]] std::size_t largest_exchange_class(
    std::span<const ExchangeClass> classes) noexcept;

}  // namespace alea
