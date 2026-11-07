// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Helpers shared by the builders (see common.h and wavefunction_builders.h).
#include "builders/wavefunction_builders.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "io/logger.h"

namespace alea {

const ParticleGroup& find_group(std::span<const ParticleGroup> groups,
                                const std::string& name) {
    for (const ParticleGroup& g : groups)
        if (g.name == name) return g;
    throw std::runtime_error("unknown particle group: " + name);
}

double layer_separation(const io::Config& cfg, const std::string& group_a,
                        const std::string& group_b) {
    auto group_z = [&](const std::string& name) {
        for (const io::ParticleSpec& p : cfg.particles)
            if (p.group == name) return p.z;
        return 0.0;
    };
    return std::abs(group_z(group_a) - group_z(group_b));
}

std::vector<Parameter> make_parameters(const std::vector<double>& values,
                                       const std::vector<bool>& is_fixed,
                                       const std::vector<bool>& can_flip_sign,
                                       const std::string& label) {
    std::vector<Parameter> params;
    params.reserve(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const bool fixed    = i < is_fixed.size() ? is_fixed[i] : true;
        const bool can_flip = i < can_flip_sign.size() ? can_flip_sign[i] : true;
        bool sign_locked = !can_flip;
        if (sign_locked && values[i] == 0.0) {
            io::Logger::warn(label + " param p" + std::to_string(i) +
                " has can_flip_sign=false but starts at 0; sign is undefined,"
                " leaving it unconstrained.");
            sign_locked = false;
        }
        params.push_back(
            Parameter{"p" + std::to_string(i), values[i], fixed, sign_locked});
    }
    return params;
}

bool is_gamma_twist(const std::vector<double>& t) {
    return std::all_of(t.begin(), t.end(), [](double ti) {
        return std::abs(ti - std::nearbyint(ti)) < 1e-12;
    });
}

}  // namespace alea
