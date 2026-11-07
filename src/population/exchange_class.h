// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// A set of mutually indistinguishable particles: one group (same mass, charge
// and layer), one quantum statistics and one flavour (spin × valley × …).
// Only particles of one class may be connected by a path permutation; the
// fermion sign of a closed path is Π_c (−1)^{P_c} over its fermionic classes.
// Built from the configuration by build_exchange_classes (builders/).

#include <cstddef>
#include <string>
#include <vector>

namespace alea {

struct ExchangeClass {
    std::string              group;
    std::size_t              flavour = 0;
    std::string              statistics;  // "fermions" | "bosons"
    std::vector<std::size_t> particles;   // global particle indices, increasing
};

}  // namespace alea
