// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>
#include <cstddef>
#include <vector>

namespace alea {

/** Represents a group of identical particles sharing
 *  the same physical properties (mass, charge).
 *
 *  Particles of this species occupy indices [first, last)
 *  in the configuration state.
 *
 *  mass and inv_mass are per-dimension to support anisotropic effective mass.
 */
struct ParticleGroup {
    std::string         name;
    std::vector<double> mass;
    std::vector<double> inv_mass;
    double              charge;
    std::size_t         first;
    std::size_t         last;
    std::size_t         count;

    std::size_t size() const noexcept { return last - first; }

    bool contains(std::size_t i) const noexcept { return i >= first && i < last; }

    bool operator==(const ParticleGroup& other) const noexcept {
        return first == other.first && last == other.last;
    }
};

} // namespace alea