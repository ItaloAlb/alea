// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>
#include <cstddef>
#include <span>

namespace alea {

class ConfigurationState;

class Walker {
public:
    explicit Walker(std::size_t n_particles, std::size_t dim);
    Walker(const Walker&) = default;
    Walker& operator=(const Walker&) = default;

    std::span<const double> positions()  const noexcept { return positions_; }
    double                  local_energy() const noexcept { return local_energy_; }

    double          weight()       const noexcept { return weight_; }
    double          multiplicity() const noexcept { return multiplicity_; }
    std::size_t     age()          const noexcept { return age_; }
    std::size_t     id()           const noexcept { return id_; }
    std::size_t     parent_id()    const noexcept { return parent_id_; }

    void set_positions(std::span<const double> pos) {positions_.assign(pos.begin(), pos.end());}
    void set_local_energy(double e)         { local_energy_ = e; }
    
    void set_weight(double w)               { weight_ = w; }
    void set_multiplicity(double m)         { multiplicity_ = m; }

    void increment_age()                    { age_++; }
    void reset_age()                        { age_ = 0; }
    
    void set_id(std::size_t id)             { id_ = id; }
    void set_parent_id(std::size_t pid)     { parent_id_ = pid; }

    // Branching
    void mark_for_death() { weight_ = 0.0; multiplicity_ = 0.0; }
    bool is_dead() const noexcept { return multiplicity_ == 0.0; }

    std::size_t  n_particles() const noexcept { return n_particles_; }
    std::size_t  dim()         const noexcept { return dim_; }

private:
    std::size_t n_particles_;
    std::size_t dim_;

    std::vector<double> positions_;
    double              local_energy_ = 0.0;

    double          weight_       = 1.0;
    double          multiplicity_ = 1.0;
    std::size_t     age_          = 0;
    std::size_t     id_           = 0;
    std::size_t     parent_id_    = 0;
};

}  // namespace alea