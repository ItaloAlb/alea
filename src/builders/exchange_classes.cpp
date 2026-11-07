// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "exchange_classes.h"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace alea {

std::vector<ExchangeClass> build_exchange_classes(
    std::span<const io::ParticleSpec> particles) {
    std::map<std::string, std::string> group_type;  // validation only
    std::vector<ExchangeClass> classes;
    std::size_t offset = 0;
    for (const io::ParticleSpec& particle : particles) {
        const std::string type = particle.type.empty() ? "fermions" : particle.type;
        if (type != "fermions" && type != "bosons" && type != "boltzmannon")
            throw std::runtime_error("pimc: unknown particle type '" + type +
                "' for group '" + particle.group +
                "' (expected fermions, bosons or boltzmannon)");
        const auto [entry, inserted] = group_type.try_emplace(particle.group, type);
        if (!inserted && entry->second != type)
            throw std::runtime_error("pimc: incompatible particle types '" +
                entry->second + "' and '" + type + "' in group '" +
                particle.group + "'");

        const std::vector<int> counts = particle.flavour_counts();
        for (std::size_t flavour = 0; flavour < counts.size(); ++flavour) {
            // `up`/`down` are not range-checked by the parser; a negative count
            // cast to size_t would request ~2^64 indices.
            if (counts[flavour] < 0)
                throw std::runtime_error("pimc: negative particle count in group '" +
                                         particle.group + "'");
            const auto n = static_cast<std::size_t>(counts[flavour]);
            if (type != "boltzmannon" && n > 0) {
                auto it = std::find_if(classes.begin(), classes.end(),
                    [&](const ExchangeClass& c) {
                        return c.group == particle.group && c.flavour == flavour;
                    });
                if (it == classes.end()) {
                    classes.push_back({particle.group, flavour, type, {}});
                    it = std::prev(classes.end());
                }
                for (std::size_t i = 0; i < n; ++i) it->particles.push_back(offset + i);
            }
            offset += n;
        }
    }
    // Indices were appended in increasing order, so each class is sorted and
    // the first element orders the classes deterministically.
    std::sort(classes.begin(), classes.end(),
              [](const ExchangeClass& a, const ExchangeClass& b) {
                  return a.particles.front() < b.particles.front();
              });
    return classes;
}

std::size_t largest_exchange_class(std::span<const ExchangeClass> classes) noexcept {
    std::size_t largest = 1;
    for (const ExchangeClass& c : classes)
        largest = std::max(largest, c.particles.size());
    return largest;
}

}  // namespace alea
