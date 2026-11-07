// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "io/parser.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <string_view>
#include <random>
#include <sstream>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace alea::io {

namespace {

// The fixed-R / spring-restraint route (relative coordinates, Ψ_eh pairing,
// umbrella guide, spring restraint) was removed from main.  Its keywords are
// refused by name rather than ignored: dropping them would silently build a
// different trial wavefunction or Hamiltonian than the config describes.
[[noreturn]] void removed_fixed_r(const std::string& what) {
    throw std::runtime_error(
        what + " belonged to the fixed-R / spring-restraint route, which was "
        "removed from alea; check out the git tag archive/fixed-r to run it.");
}

// Refuse unsupported configuration rather than silently solving another model.
void require_keys(const YAML::Node& node, const std::string& context,
                  std::initializer_list<std::string_view> allowed) {
    if (!node.IsMap()) throw std::runtime_error(context + ": expected a map");
    for (const auto& item : node) {
        const std::string key = item.first.as<std::string>();
        if (std::find(allowed.begin(), allowed.end(), key) == allowed.end())
            throw std::runtime_error(context + ": unknown key '" + key + "'");
    }
}

// The sampling keys every qmc: stage shares (RunSpec), plus `method`.  A plain
// array: GCC 11 rejects a constexpr std::initializer_list at namespace scope
// ("modification of <temporary> is not a constant expression").
constexpr std::string_view kRunKeys[] = {
    "method", "device", "threads", "samples", "equilibration_blocks",
    "accumulation_blocks", "steps_per_block", "sub_steps", "time_step"};

// require_keys for a qmc: stage: the shared sampling keys plus the method's own.
void require_stage_keys(const YAML::Node& node, const std::string& context,
                        std::initializer_list<std::string_view> own) {
    if (!node.IsMap()) throw std::runtime_error(context + ": expected a map");
    for (const auto& item : node) {
        const std::string key = item.first.as<std::string>();
        if (std::find(std::begin(kRunKeys), std::end(kRunKeys), key) ==
                std::end(kRunKeys) &&
            std::find(own.begin(), own.end(), key) == own.end())
            throw std::runtime_error(context + ": unknown key '" + key + "'");
    }
}

// The sampling keys every qmc: method shares (RunSpec).  Absent keys keep the
// method's own defaults.
void parse_run_keys(const YAML::Node& qn, RunSpec& r) {
    if (qn["device"]) {
        r.device = qn["device"].as<std::string>();
        if (r.device != "cpu")
            throw std::runtime_error("qmc: device '" + r.device +
                                     "' is not supported (only 'cpu')");
    }
    if (qn["threads"])       r.threads = qn["threads"].as<int>();
    if (qn["samples"])       r.samples = qn["samples"].as<int>();
    if (qn["equilibration_blocks"])
        r.equilibration_blocks = qn["equilibration_blocks"].as<int>();
    if (qn["accumulation_blocks"])
        r.accumulation_blocks = qn["accumulation_blocks"].as<int>();
    if (qn["steps_per_block"])
        r.steps_per_block = qn["steps_per_block"].as<int>();
    if (qn["sub_steps"])     r.sub_steps = qn["sub_steps"].as<int>();
    if (qn["time_step"])     r.time_step = qn["time_step"].as<double>();
}

// One `estimators:` entry → the spec of its kind.  Every stage (`stage` is
// "qmc" or "pimc", for messages) reads its list through here, so a key means
// the same in vmc, dmc and pimc, and each kind admits only its own keys.
// Which samplers may measure a kind is decided by its builder, not here.
EstimatorItem parse_estimator(const YAML::Node& en, const std::string& stage) {
    if (!en.IsMap()) throw std::runtime_error(stage + " estimator: expected a map");
    if (!en["type"]) throw std::runtime_error(stage + " estimator: `type` is required");
    const std::string type = en["type"].as<std::string>();
    const std::string where = stage + " " + type;
    // A name labels HDF5 result keys, where '/' would open a group.
    const auto name_of = [&]() -> std::string {
        if (!en["name"]) return {};
        std::string name = en["name"].as<std::string>();
        if (name.empty() || name.find('/') != std::string::npos)
            throw std::runtime_error(where + ": name must be nonempty and contain no '/'");
        return name;
    };
    if (type == "gofr") {
        require_keys(en, where, {"type", "name", "rmax", "n_bins"});
        GofrSpec s;
        s.name = name_of();
        if (en["rmax"])   s.rmax   = en["rmax"].as<double>();
        if (en["n_bins"]) s.n_bins = en["n_bins"].as<int>();
        return s;
    }
    if (type == "density") {
        require_keys(en, where, {"type", "name", "rmax", "n_bins", "per_particle"});
        DensitySpec s;
        s.name = name_of();
        if (en["rmax"])         s.rmax         = en["rmax"].as<double>();
        if (en["n_bins"])       s.n_bins       = en["n_bins"].as<int>();
        if (en["per_particle"]) s.per_particle = en["per_particle"].as<bool>();
        return s;
    }
    if (type == "sofk") {
        require_keys(en, where, {"type", "name", "kmax"});
        StructureFactorSpec s;
        s.name = name_of();
        if (en["kmax"]) s.kmax = en["kmax"].as<double>();
        return s;
    }
    if (type == "collective_density") {
        require_keys(en, where, {"type", "name", "rmax", "n_bins", "terms"});
        CollectiveDensitySpec s;
        s.name = name_of();  // required: the estimator refuses an empty one
        if (en["rmax"])   s.rmax   = en["rmax"].as<double>();
        if (en["n_bins"]) s.n_bins = en["n_bins"].as<int>();
        if (en["terms"]) {
            if (!en["terms"].IsSequence())
                throw std::runtime_error(where + ": terms must be a list");
            for (const YAML::Node& tn : en["terms"]) {
                require_keys(tn, where + " term", {"group", "coeff"});
                s.terms.emplace_back(tn["group"] ? tn["group"].as<std::string>() : "",
                                     tn["coeff"] ? tn["coeff"].as<double>() : 1.0);
            }
        }
        return s;
    }
    if (type == "electron_hole_proximity") {
        require_keys(en, where, {"type", "name", "electron_group", "hole_group", "r_cut"});
        if (!en["electron_group"] || !en["hole_group"] || !en["r_cut"])
            throw std::runtime_error(where + ": electron_group, hole_group and r_cut are required");
        ElectronHoleProximitySpec s;
        s.name = name_of();
        s.electron_group = en["electron_group"].as<std::string>();
        s.hole_group = en["hole_group"].as<std::string>();
        s.r_cut = en["r_cut"].as<double>();
        if (!std::isfinite(s.r_cut) || s.r_cut <= 0.0)
            throw std::runtime_error(where + ": r_cut must be finite and positive");
        return s;
    }
    if (type == "momentum_distribution") {
        require_keys(en, where,
                     {"type", "name", "group", "rmax", "n_bins", "kmax", "n_k", "samples"});
        if (!en["group"] || !en["rmax"] || !en["n_bins"] || !en["kmax"] || !en["samples"])
            throw std::runtime_error(where + ": group, rmax, n_bins, kmax and samples are required");
        MomentumDistributionSpec s;
        s.name = name_of();
        s.group = en["group"].as<std::string>();
        s.rmax = en["rmax"].as<double>();
        s.n_bins = en["n_bins"].as<int>();
        s.kmax = en["kmax"].as<double>();
        s.samples = en["samples"].as<int>();
        if (en["n_k"]) s.n_k = en["n_k"].as<int>();
        return s;
    }
    if (type == "bragg") {
        require_keys(en, where, {"type", "name", "group", "wavevectors"});
        if (!en["group"] || !en["wavevectors"] || !en["wavevectors"].IsSequence() ||
            en["wavevectors"].size() == 0)
            throw std::runtime_error(where + ": group and nonempty wavevectors are required");
        BraggSpec s;
        s.name = name_of();
        s.group = en["group"].as<std::string>();
        for (const YAML::Node& vector : en["wavevectors"]) {
            if (!vector.IsSequence() || vector.size() != 2)
                throw std::runtime_error(where + ": wavevectors need Cartesian [kx, ky] pairs");
            for (const YAML::Node& coordinate : vector) {
                const double value = coordinate.as<double>();
                if (!std::isfinite(value))
                    throw std::runtime_error(where + ": wavevectors must be finite");
                s.wavevectors.push_back(value);
            }
        }
        return s;
    }
    if (type == "bond_order") {
        require_keys(en, where, {"type", "name", "group", "reference_spacing"});
        if (!en["group"] || !en["reference_spacing"])
            throw std::runtime_error(where + ": group and reference_spacing are required");
        BondOrderSpec s;
        s.name = name_of();
        s.group = en["group"].as<std::string>();
        s.reference_spacing = en["reference_spacing"].as<double>();
        if (!std::isfinite(s.reference_spacing) || s.reference_spacing <= 0.0)
            throw std::runtime_error(where + ": reference_spacing must be finite and positive");
        return s;
    }
    if (type == "density_matrix") {
        require_keys(en, where, {"type", "name", "groups", "pairs", "rmax", "n_bins", "every"});
        if (!en["rmax"] || !en["n_bins"])
            throw std::runtime_error(where + ": rmax and n_bins are required");
        if (!en["groups"] && !en["pairs"])
            throw std::runtime_error(where + ": groups, pairs or both are required");
        DensityMatrixSpec s;
        s.name = name_of();
        s.rmax = en["rmax"].as<double>();
        s.n_bins = en["n_bins"].as<int>();
        if (en["every"]) s.every = en["every"].as<int>();
        if (en["groups"]) {
            if (!en["groups"].IsSequence())
                throw std::runtime_error(where + ": groups must be a list of group names");
            for (const YAML::Node& g : en["groups"]) s.groups.push_back(g.as<std::string>());
        }
        if (en["pairs"]) {
            if (!en["pairs"].IsSequence())
                throw std::runtime_error(where + ": pairs must be a list of [group, group]");
            for (const YAML::Node& pair : en["pairs"]) {
                if (!pair.IsSequence() || pair.size() != 2)
                    throw std::runtime_error(where + ": each pair must be [group, group]");
                s.pairs.emplace_back(pair[0].as<std::string>(), pair[1].as<std::string>());
            }
        }
        return s;
    }
    throw std::runtime_error(stage + " estimator: unknown type '" + type +
                             "' (expected gofr, density, sofk, collective_density, "
                             "electron_hole_proximity, momentum_distribution, bragg, "
                             "bond_order or density_matrix)");
}

std::vector<EstimatorItem> parse_estimators(const YAML::Node& list, const std::string& stage) {
    if (!list.IsSequence())
        throw std::runtime_error(stage + ": estimators must be a list");
    std::vector<EstimatorItem> items;
    for (const YAML::Node& en : list) items.push_back(parse_estimator(en, stage));
    return items;
}

}  // namespace

std::uint64_t resolve_seed(std::uint64_t seed) {
    if (seed != 0) return seed;

    // seed == 0 → "auto": draw a fresh seed from the OS entropy source.
    // std::random_device yields 32 bits per call, so compose two draws into the
    // full 64-bit seed space.
    std::random_device rd;
    std::uint64_t fresh =
        (static_cast<std::uint64_t>(rd()) << 32) | static_cast<std::uint64_t>(rd());
    // Keep the contract that a resolved seed is non-zero so it can be recorded
    // and replayed; the all-zero draw is astronomically rare but cheap to guard.
    return fresh != 0 ? fresh : 0x9E3779B97F4A7C15ULL;
}

Config parse_config(const std::string& yaml_text) {
    const YAML::Node root = YAML::Load(yaml_text);
    Config c;

    // Every block below refuses keys it does not read: a misspelt key (or a
    // whole misspelt section) would otherwise leave its default in place while
    // the file says something else.
    if (root.IsMap())
        require_keys(root, "config", {"project", "system", "wavefunction",
                                      "hamiltonian", "action", "qmc"});

    if (const YAML::Node project = root["project"]) {
        require_keys(project, "project", {"name", "seed", "log", "output"});
        if (project["name"])   c.name   = project["name"].as<std::string>();
        if (project["seed"])   c.seed   = project["seed"].as<std::uint64_t>();
        if (project["log"])    c.log    = project["log"].as<bool>();
        if (project["output"]) c.output = project["output"].as<bool>();
    }

    if (const YAML::Node system = root["system"]) {
        if (system["coordinates"]) removed_fixed_r("system.coordinates");
        require_keys(system, "system", {"lattice", "particles"});

        if (const YAML::Node lat = system["lattice"]) {
            require_keys(lat, "lattice", {"type", "dimension", "L", "shape", "rs",
                                          "truncation", "ewald_tol", "capacitor", "twist", "vectors"});
            if (lat["type"]) c.lattice.type = lat["type"].as<std::string>();
            if (lat["dimension"]) c.lattice.dimension = lat["dimension"].as<int>();
            // `L` is either a scalar lattice constant / scatter half-width or a
            // per-axis list for a rectangular cell.
            if (lat["L"]) {
                if (lat["L"].IsSequence()) c.lattice.lengths = lat["L"].as<std::vector<double>>();
                else                       c.lattice.L      = lat["L"].as<double>();
            }
            if (lat["shape"]) c.lattice.shape = lat["shape"].as<std::string>();
            if (lat["rs"])    c.lattice.rs    = lat["rs"].as<double>();
            if (lat["truncation"])
                c.lattice.truncation = lat["truncation"].as<std::string>();
            if (lat["ewald_tol"]) c.lattice.ewald_tol = lat["ewald_tol"].as<double>();
            if (lat["capacitor"]) {
                c.lattice.capacitor = lat["capacitor"].as<std::string>();
                if (c.lattice.capacitor != "include" && c.lattice.capacitor != "exclude")
                    throw std::runtime_error("lattice.capacitor must be include or exclude, got '" +
                                             c.lattice.capacitor + "'");
            }
            if (lat["twist"]) c.lattice.twist = lat["twist"].as<std::vector<double>>();
            if (const YAML::Node v = lat["vectors"]) {
                // Accept either [[a1x,a1y],[a2x,a2y]] or a flat row-major list.
                for (const YAML::Node& row : v) {
                    if (row.IsSequence())
                        for (const YAML::Node& e : row)
                            c.lattice.vectors.push_back(e.as<double>());
                    else
                        c.lattice.vectors.push_back(row.as<double>());
                }
            }
            // `type` was previously parsed and never read by anything, so any
            // string silently "worked".  Validate it now that it is load-bearing.
            if (!c.lattice.truncation.empty() &&
                c.lattice.truncation != "minimum_image")
                throw std::runtime_error("unknown lattice truncation: '" +
                                         c.lattice.truncation +
                                         "' (expected 'minimum_image')");
            if (c.lattice.type != "open" && c.lattice.type != "periodic")
                throw std::runtime_error("unknown lattice type: '" + c.lattice.type +
                                         "' (expected 'open' or 'periodic')");
        }

        for (const YAML::Node& pn : system["particles"]) {
            ParticleSpec p;
            if (pn["name"]) p.name = pn["name"].as<std::string>();
            if (pn["group"]) p.group = pn["group"].as<std::string>();
            if (pn["type"]) p.type = pn["type"].as<std::string>();
            if (const YAML::Node& m = pn["mass"]) {
                if (m.IsScalar())
                    p.mass = {m.as<double>()};
                else if (m.IsSequence())
                    p.mass = m.as<std::vector<double>>();
            }
            if (pn["charge"]) p.charge = pn["charge"].as<double>();
            if (pn["z"]) p.z = pn["z"].as<double>();
            if (pn["r0"] || pn["epsilon"])
                removed_fixed_r("per-particle `r0`/`epsilon` screening");
            // A misspelt key (e.g. `flavors`) would otherwise leave the entry
            // with zero particles and no warning.
            const std::string context = "particle '" + p.name + "'";
            require_keys(pn, context, {"name", "group", "type", "mass", "charge", "z",
                                       "up", "down", "flavours", "initialization"});
            if (pn["up"]) p.up = pn["up"].as<int>();
            if (pn["down"]) p.down = pn["down"].as<int>();
            if (const YAML::Node& fl = pn["flavours"]) {
                if (pn["up"] || pn["down"])
                    throw std::runtime_error(context + ": `flavours` replaces "
                                             "`up`/`down`; give one or the other");
                if (!fl.IsSequence() || fl.size() == 0)
                    throw std::runtime_error(context + ": `flavours` must be a "
                                             "non-empty list of per-flavour counts");
                for (const YAML::Node& n : fl) {
                    int c = -1;
                    try {
                        c = n.as<int>();
                    } catch (const YAML::Exception&) {
                        throw std::runtime_error(context + ": `flavours` entries "
                                                 "must be integers");
                    }
                    if (c < 0)
                        throw std::runtime_error(context + ": `flavours` counts "
                                                 "must be non-negative");
                    p.flavours.push_back(c);
                }
                if (p.count() == 0)
                    throw std::runtime_error(context + ": `flavours` declares "
                                             "no particles");
            }
            if (pn["initialization"])
                p.initialization = pn["initialization"].as<std::string>();
            c.particles.push_back(std::move(p));
        }
    }

    for (const YAML::Node& item : root["wavefunction"]) {
        // One kind per item: a second key in the same item (an indentation
        // slip that merges two factors) would otherwise be dropped.
        if (!item.IsMap() || item.size() != 1)
            throw std::runtime_error(
                "wavefunction: each item must be a single-key map naming its "
                "kind, e.g. '- jastrow: {...}'");
        if (const YAML::Node jn = item["jastrow"]) {
            JastrowSpec j;
            if (jn["type"]) j.type = jn["type"].as<std::string>();
            if (j.type == "umbrella") removed_fixed_r("the `umbrella` jastrow");
            if (jn["pairing"]) removed_fixed_r("the `pairing` jastrow (Ψ_eh)");
            if (jn["role"]) removed_fixed_r("the jastrow `role` key");
            if (jn["pair"]) {
                j.pair[0] = jn["pair"][0].as<std::string>();
                j.pair[1] = jn["pair"][1].as<std::string>();
            }
            if (const YAML::Node pn = jn["pairs"]) {
                if (jn["pair"])
                    throw std::runtime_error(
                        "jastrow: give either `pair:` (one group pair) or `pairs:` "
                        "(one u(r) shared by several), not both");
                for (const YAML::Node& p : pn) {
                    if (!p.IsSequence() || p.size() != 2)
                        throw std::runtime_error(
                            "jastrow `pairs:` entries must be [group_a, group_b]");
                    j.pairs.push_back({p[0].as<std::string>(), p[1].as<std::string>()});
                }
                if (j.pairs.empty())
                    throw std::runtime_error("jastrow `pairs:` is empty");
            }
            const std::string pairs_label = jastrow_pairs_label(j);
            // The cusp is derived from the Hamiltonian, never configured.
            if (jn["cusp"])
                throw std::runtime_error(
                    "jastrow on " + pairs_label + ": `cusp:` "
                    "is not a config input — the Kato cusp is derived from the "
                    "Hamiltonian and fixed.  Remove the key.");
            const bool is_pade = (j.type == "pade" || j.type == "padé");
            if (!is_pade && j.type != "drummond" && j.type != "zero_energy")
                throw std::runtime_error("unknown jastrow type '" + j.type +
                                         "' (expected padé, drummond or zero_energy)");
            if (!j.pairs.empty() && j.type != "drummond")
                throw std::runtime_error(
                    j.type + " jastrow on " + pairs_label + ": `pairs:` (one u(r) "
                    "shared by several group pairs) is supported for drummond only; "
                    "give this type one `pair:` per item.");
            if (j.type == "zero_energy") {
                const std::string label = "zero_energy jastrow on " + pairs_label;
                require_keys(jn, label,
                             {"type", "pair", "matching_radius", "matching_radius_fixed",
                              "cutoff"});
                if (!jn["matching_radius"])
                    throw std::runtime_error(label + " needs `matching_radius` (R_m, "
                                             "where K0 is joined to the tail)");
                j.matching_radius = jn["matching_radius"].as<double>();
                if (jn["matching_radius_fixed"])
                    j.matching_radius_fixed = jn["matching_radius_fixed"].as<bool>();
            } else if (is_pade) {
                // The old layout put the cusp a₁ in params[0]; read under the
                // new meaning it would silently shift every coefficient.
                if (jn["num_orders"] || jn["params"])
                    throw std::runtime_error(
                        "padé jastrow on " + pairs_label +
                        ": `num_orders`/`params` are no longer read.  The cusp "
                        "a₁ is derived from the Hamiltonian; write the free "
                        "coefficients as `numerator: [a₂..a_N]` and "
                        "`denominator: [b₁..b_M]`.");
                require_keys(jn, "padé jastrow on " + pairs_label,
                             {"type", "pair", "numerator", "denominator", "is_fixed",
                              "can_flip_sign"});
                if (jn["numerator"])
                    j.numerator = jn["numerator"].as<std::vector<double>>();
                if (jn["denominator"])
                    j.denominator = jn["denominator"].as<std::vector<double>>();
            } else {
                require_keys(jn, "drummond jastrow on " + pairs_label,
                             {"type", "pair", "pairs", "params", "is_fixed", "can_flip_sign",
                              "truncation", "cutoff", "cutoff_fixed"});
                if (jn["params"]) j.params = jn["params"].as<std::vector<double>>();
            }
            if (jn["is_fixed"]) j.is_fixed = jn["is_fixed"].as<std::vector<bool>>();
            if (jn["can_flip_sign"])
                j.can_flip_sign = jn["can_flip_sign"].as<std::vector<bool>>();
            if (jn["truncation"]) j.truncation = jn["truncation"].as<int>();
            if (jn["cutoff"]) j.cutoff = jn["cutoff"].as<double>();
            if (jn["cutoff_fixed"]) j.cutoff_fixed = jn["cutoff_fixed"].as<bool>();
            c.wavefunction.emplace_back(std::move(j));
        } else if (const YAML::Node sn = item["slater"]) {
            require_keys(sn, "slater", {"group", "orbitals", "backflow"});
            SlaterSpec s;
            if (sn["group"]) s.group = sn["group"].as<std::string>();
            if (const YAML::Node bf = sn["backflow"]) {
                const std::string where = "slater backflow of group " + s.group;
                if (!bf.IsSequence() || bf.size() == 0)
                    throw std::runtime_error(where + ": expected a nonempty list of channels "
                                             "{from, params, ...}");
                for (const YAML::Node& cn : bf) {
                    if (!cn.IsMap())
                        throw std::runtime_error(where + ": each channel is a map");
                    require_keys(cn, where, {"from", "truncation", "cutoff", "cutoff_fixed",
                                             "params", "is_fixed"});
                    BackflowSpec b;
                    if (!cn["from"])
                        throw std::runtime_error(where + ": a channel needs `from:` (the "
                                                 "group whose particles displace the rows)");
                    b.from = cn["from"].as<std::string>();
                    if (!cn["params"] || !cn["params"].IsSequence() || cn["params"].size() == 0)
                        throw std::runtime_error(
                            where + " from " + b.from + ": needs `params`, the free "
                            "coefficients [c_0, ...] (c_1 is derived for the group itself)");
                    b.params = cn["params"].as<std::vector<double>>();
                    if (cn["is_fixed"]) b.is_fixed = cn["is_fixed"].as<std::vector<bool>>();
                    if (!b.is_fixed.empty() && b.is_fixed.size() != b.params.size())
                        throw std::runtime_error(where + " from " + b.from +
                                                 ": `is_fixed` needs one flag per params entry");
                    if (cn["truncation"]) b.truncation = cn["truncation"].as<int>();
                    if (cn["cutoff"]) b.cutoff = cn["cutoff"].as<double>();
                    if (cn["cutoff_fixed"]) b.cutoff_fixed = cn["cutoff_fixed"].as<bool>();
                    s.backflow.push_back(std::move(b));
                }
            }
            if (const YAML::Node orb = sn["orbitals"]) {
                require_keys(orb, "slater orbitals of group " + s.group,
                             {"type", "centers", "alphas", "sites", "alpha",
                              "alpha_fixed", "l", "beta"});
                if (orb["type"]) s.orbital_type = orb["type"].as<std::string>();
                if (orb["centers"])
                    for (const YAML::Node& cn : orb["centers"])
                        s.centers.push_back(cn.as<std::vector<double>>());
                if (orb["alphas"])
                    s.alphas = orb["alphas"].as<std::vector<double>>();
                if (orb["sites"]) s.sites = orb["sites"].as<std::string>();
                if (orb["alpha"]) s.alpha = orb["alpha"].as<double>();
                if (orb["alpha_fixed"])
                    s.alpha_fixed = orb["alpha_fixed"].as<bool>();
                if (orb["l"])    s.ls   = orb["l"].as<std::vector<int>>();
                if (orb["beta"]) s.beta = orb["beta"].as<double>();
            }
            c.wavefunction.emplace_back(std::move(s));
        } else if (const YAML::Node mn = item["moire_jastrow"]) {
            require_keys(mn, "moire_jastrow",
                         {"group", "shells", "params", "is_fixed", "a_M", "theta0"});
            MoireJastrowSpec m;
            if (mn["group"])    m.group    = mn["group"].as<std::string>();
            if (mn["shells"])   m.shells   = mn["shells"].as<int>();
            if (mn["params"])   m.params   = mn["params"].as<std::vector<double>>();
            if (mn["is_fixed"]) m.is_fixed = mn["is_fixed"].as<std::vector<bool>>();
            if (mn["a_M"])      m.a_M      = mn["a_M"].as<double>();
            if (mn["theta0"])   m.theta0   = mn["theta0"].as<double>();
            c.wavefunction.emplace_back(std::move(m));
        } else if (const YAML::Node kn = item["kspace_jastrow"]) {
            require_keys(kn, "kspace_jastrow",
                         {"pair", "shells", "params", "is_fixed", "can_flip_sign"});
            KSpaceJastrowSpec k;
            if (kn["pair"]) {
                k.pair[0] = kn["pair"][0].as<std::string>();
                k.pair[1] = kn["pair"][1].as<std::string>();
            }
            if (!kn["shells"])
                throw std::runtime_error("kspace_jastrow needs `shells:` (number of |k| shells)");
            k.shells = kn["shells"].as<int>();
            if (kn["params"]) k.params = kn["params"].as<std::vector<double>>();
            if (kn["is_fixed"]) k.is_fixed = kn["is_fixed"].as<std::vector<bool>>();
            if (kn["can_flip_sign"])
                k.can_flip_sign = kn["can_flip_sign"].as<std::vector<bool>>();
            c.wavefunction.emplace_back(std::move(k));
        } else if (const YAML::Node nn = item["nosanow"]) {
            require_keys(nn, "nosanow", {"group", "sites", "centers", "alpha", "alpha_fixed"});
            NosanowSpec n;
            if (!nn["group"]) throw std::runtime_error("nosanow needs `group:`");
            n.group = nn["group"].as<std::string>();
            const std::string label = "nosanow on group " + n.group;
            if (!nn["alpha"])
                throw std::runtime_error(label + " needs `alpha:` (the width of "
                                         "exp(-alpha |r - R|^2) about each site)");
            n.alpha = nn["alpha"].as<double>();
            if (nn["alpha_fixed"]) n.alpha_fixed = nn["alpha_fixed"].as<bool>();
            if (nn["sites"]) n.sites = nn["sites"].as<std::string>();
            if (nn["centers"])
                for (const YAML::Node& cn : nn["centers"])
                    n.centers.push_back(cn.as<std::vector<double>>());
            if (n.sites.empty() == n.centers.empty())
                throw std::runtime_error(label + " needs exactly one of `sites: "
                                         "triangular` or `centers: [[x, y], ...]`");
            c.wavefunction.emplace_back(std::move(n));
        } else if (const YAML::Node bn = item["center_of_mass_momentum"]) {
            require_keys(bn, "center_of_mass_momentum", {"K"});
            if (!items_of<CenterOfMassMomentumSpec>(c.wavefunction).empty())
                throw std::runtime_error(
                    "center_of_mass_momentum given twice: the boost is a single "
                    "total momentum K");
            if (!bn["K"])
                throw std::runtime_error("center_of_mass_momentum needs K: [Kx, ...]");
            c.wavefunction.emplace_back(
                CenterOfMassMomentumSpec{bn["K"].as<std::vector<double>>()});
        } else if (const YAML::Node pn = item["pairing"]) {
            require_keys(pn, "pairing", {"blocks", "orbital"});
            PairingSpec ps;
            if (!pn["blocks"] || !pn["blocks"].IsSequence() || pn["blocks"].size() == 0)
                throw std::runtime_error("pairing needs `blocks:`, a list of "
                                         "{rows: group, columns: group}");
            for (const YAML::Node& bn2 : pn["blocks"]) {
                require_keys(bn2, "pairing block", {"rows", "columns", "unpaired"});
                if (!bn2["rows"] || !bn2["columns"])
                    throw std::runtime_error("pairing block needs `rows:` and `columns:`");
                PairingBlockSpec blk;
                blk.rows = bn2["rows"].as<std::string>();
                blk.columns = bn2["columns"].as<std::string>();
                if (const YAML::Node un = bn2["unpaired"]) {
                    const std::string label =
                        "pairing block (" + blk.rows + ", " + blk.columns + ") unpaired";
                    require_keys(un, label, {"type", "k", "sin"});
                    blk.unpaired_type = un["type"] ? un["type"].as<std::string>() : "";
                    if (blk.unpaired_type != "plane_wave")
                        throw std::runtime_error(label + ": unknown type '" + blk.unpaired_type +
                                                 "' (expected plane_wave)");
                    if (!un["k"] || un["k"].size() == 0)
                        throw std::runtime_error(label + " needs `k:`, a list of integer "
                                                 "reciprocal indices [n1, n2]");
                    for (const YAML::Node& kn2 : un["k"])
                        blk.unpaired_k.push_back(kn2.as<std::vector<int>>());
                    if (un["sin"]) blk.unpaired_sin = un["sin"].as<std::vector<bool>>();
                    if (!blk.unpaired_sin.empty() && blk.unpaired_sin.size() != blk.unpaired_k.size())
                        throw std::runtime_error(label + ": `sin:` needs one entry per k");
                }
                ps.blocks.push_back(std::move(blk));
            }
            const YAML::Node on = pn["orbital"];
            if (!on) throw std::runtime_error("pairing needs `orbital:`");
            require_keys(on, "pairing orbital", {"type", "shells", "poly_order", "cutoff",
                                                 "cutoff_fixed", "params", "is_fixed"});
            if (on["type"]) ps.orbital_type = on["type"].as<std::string>();
            if (ps.orbital_type != "cosine_polynomial")
                throw std::runtime_error("pairing orbital: unknown type '" + ps.orbital_type +
                                         "' (expected cosine_polynomial)");
            if (on["shells"]) ps.shells = on["shells"].as<int>();
            if (on["poly_order"]) ps.poly_order = on["poly_order"].as<int>();
            if (ps.shells < 0) throw std::runtime_error("pairing orbital: shells must be >= 0");
            if (on["cutoff"]) ps.cutoff = on["cutoff"].as<double>();
            if (on["cutoff_fixed"]) ps.cutoff_fixed = on["cutoff_fixed"].as<bool>();
            if (on["params"]) ps.params = on["params"].as<std::vector<double>>();
            if (on["is_fixed"]) ps.is_fixed = on["is_fixed"].as<std::vector<bool>>();
            c.wavefunction.emplace_back(std::move(ps));
        } else {
            // Reject rather than ignore, as for hamiltonian terms: a dropped
            // factor builds a different trial than the config describes.
            std::string keys;
            if (item.IsMap())
                for (const auto& kv : item)
                    keys += (keys.empty() ? "" : ", ") + kv.first.as<std::string>();
            throw std::runtime_error(
                "unknown wavefunction item '" + keys + "' (expected jastrow, "
                "slater, pairing, moire_jastrow, kspace_jastrow, nosanow or "
                "center_of_mass_momentum)");
        }
    }

    if (const YAML::Node ham = root["hamiltonian"]) {
        require_keys(ham, "hamiltonian", {"terms"});
        for (const YAML::Node& term : ham["terms"]) {
            const std::string type =
                term["type"] ? term["type"].as<std::string>() : "";
            auto term_keys = [&](std::initializer_list<std::string_view> keys) {
                require_keys(term, "hamiltonian term '" + type + "'", keys);
            };
            if (type == "coulomb") {
                term_keys({"type", "pair", "epsilon"});
                CoulombSpec cs;
                if (term["pair"]) {
                    cs.pair[0] = term["pair"][0].as<std::string>();
                    cs.pair[1] = term["pair"][1].as<std::string>();
                }
                if (term["epsilon"]) cs.epsilon = term["epsilon"].as<double>();
                c.coulombs.push_back(std::move(cs));
            } else if (type == "rytova_keldysh") {
                term_keys({"type", "pair", "epsilon", "r0"});
                KeldyshSpec ks;
                if (term["pair"]) {
                    ks.pair[0] = term["pair"][0].as<std::string>();
                    ks.pair[1] = term["pair"][1].as<std::string>();
                }
                if (term["epsilon"]) ks.epsilon = term["epsilon"].as<double>();
                if (term["r0"])      ks.r0      = term["r0"].as<double>();
                c.keldysh.push_back(std::move(ks));
            } else if (type == "inverse_power") {
                term_keys({"type", "pair", "strength", "exponent"});
                // No defaults: a forgotten exponent or strength must not quietly
                // become some other interaction.
                for (const char* key : {"pair", "strength", "exponent"})
                    if (!term[key])
                        throw std::runtime_error(
                            std::string("hamiltonian term 'inverse_power' needs `") +
                            key + "`");
                InversePowerSpec ip;
                ip.pair[0]  = term["pair"][0].as<std::string>();
                ip.pair[1]  = term["pair"][1].as<std::string>();
                ip.strength = term["strength"].as<double>();
                ip.exponent = term["exponent"].as<double>();
                c.inverse_powers.push_back(std::move(ip));
            } else if (type == "screened_coulomb") {
                term_keys({"type", "pair", "epsilons", "interfaces", "source_layer",
                           "target_layer", "z_offset", "fit_tol"});
                ScreenedCoulombSpec scs;
                if (term["pair"]) {
                    scs.pair[0] = term["pair"][0].as<std::string>();
                    scs.pair[1] = term["pair"][1].as<std::string>();
                }
                if (term["epsilons"])
                    scs.epsilons = term["epsilons"].as<std::vector<double>>();
                if (term["interfaces"])
                    scs.interfaces = term["interfaces"].as<std::vector<double>>();
                if (term["source_layer"])
                    scs.source_layer = term["source_layer"].as<int>();
                if (term["target_layer"])
                    scs.target_layer = term["target_layer"].as<int>();
                if (term["z_offset"]) {
                    scs.z_offset     = term["z_offset"].as<double>();
                    scs.has_z_offset = true;
                }
                if (term["fit_tol"]) scs.fit_tol = term["fit_tol"].as<double>();
                c.screened.push_back(std::move(scs));
            } else if (type == "moire" || type == "moiré") {
                term_keys({"type", "group", "V0", "Vtilde", "phi", "a_M", "theta0"});
                MoireSpec ms;
                if (term["group"])
                    ms.groups = term["group"].as<std::vector<std::string>>();
                if (term["V0"])     ms.V0     = term["V0"].as<double>();
                if (term["Vtilde"]) ms.Vtilde = term["Vtilde"].as<double>();
                if (term["phi"])    ms.phi    = term["phi"].as<double>();
                if (term["a_M"])    ms.a_M    = term["a_M"].as<double>();
                if (term["theta0"]) ms.theta0 = term["theta0"].as<double>();
                c.moires.push_back(std::move(ms));
            } else if (type == "spring") {
                removed_fixed_r("the `spring` restraint term");
            } else if (type == "harmonic_trap") {
                term_keys({"type", "groups", "omega", "center"});
                HarmonicTrapSpec hs;
                if (term["groups"])
                    hs.groups = term["groups"].as<std::vector<std::string>>();
                if (!term["omega"])
                    throw std::runtime_error("harmonic_trap needs omega");
                hs.omega = term["omega"].as<double>();
                if (term["center"]) hs.center = term["center"].as<std::vector<double>>();
                c.traps.push_back(std::move(hs));
            } else if (type == "magnetic_field") {
                term_keys({"type", "B"});
                if (c.magnetic_field)
                    throw std::runtime_error(
                        "magnetic_field given twice: it is one uniform field B");
                if (!term["B"]) throw std::runtime_error("magnetic_field needs B");
                c.magnetic_field = term["B"].as<double>();
            } else if (type == "rotation") {
                term_keys({"type", "omega"});
                if (c.rotation_omega)
                    throw std::runtime_error(
                        "rotation given twice: it is one angular velocity omega");
                if (!term["omega"]) throw std::runtime_error("rotation needs omega");
                c.rotation_omega = term["omega"].as<double>();
            } else {
                // Reject rather than ignore: a silently dropped term looks like a
                // converged physical result, which is the worst failure mode here
                // (a missing spring restraint would just read as "unbound").
                throw std::runtime_error("unknown hamiltonian term type: '" +
                                         type + "'");
            }
        }
    }

    // Top-level `action:` — the piecewise components of the PIMC Euclidean
    // action (shared physics, declared next to hamiltonian:).  Each entry is
    // a single-key map: kinetic / potential, `primitive` as shorthand for both,
    // or pair_product replacing the primitive pair potential. Absent → empty
    // (= both primitive components, the default recorded by the snapshot).
    // Unknown components are rejected, like unknown hamiltonian terms: a
    // silently dropped piece would sample a different action than the config
    // describes.
    if (const YAML::Node act = root["action"]) {
        if (!act.IsSequence())
            throw std::runtime_error(
                "action: must be a list of components (kinetic, potential, pair_product, nodes)");
        for (const YAML::Node& item : act) {
            if (!item.IsMap() || item.size() != 1)
                throw std::runtime_error(
                    "action: each entry must be a single-key map, e.g. "
                    "'- kinetic: primitive'");
            const auto kv = item.begin();
            const std::string comp = kv->first.as<std::string>();
            ActionItemSpec as;
            if (comp == "kinetic" || comp == "potential" || comp == "primitive") {
                as.component = comp;
                const YAML::Node form = kv->second;
                if (form.IsMap()) {
                    require_keys(form, "action " + comp, {"form"});
                    if (form["form"]) as.form = form["form"].as<std::string>();
                } else if (form.IsScalar()) {
                    as.form = form.as<std::string>();
                } else if (!form.IsNull()) {
                    throw std::runtime_error("action " + comp + ": expected a form or map");
                }
                if (as.form != "primitive")
                    throw std::runtime_error(
                        "action: unknown form '" + as.form + "' for " +
                        comp + " (expected 'primitive')");
            } else if (comp == "pair_product") {
                as.component=comp; as.form="finite_grid";
                const YAML::Node parameters=kv->second;
                if (!parameters.IsMap()) throw std::runtime_error("action pair_product: expected a map with explicit extent");
                require_keys(parameters,"action pair_product",{"form","extent","grid_points"});
                if (parameters["form"]) as.form=parameters["form"].as<std::string>();
                if (as.form!="finite_grid") throw std::runtime_error("action pair_product: unsupported form '"+as.form+"'");
                if (!parameters["extent"]) throw std::runtime_error("action pair_product: explicit finite-domain extent is required");
                as.extent=parameters["extent"].as<double>();
                if (parameters["grid_points"]) as.grid_points=parameters["grid_points"].as<int>();
                if (!std::isfinite(as.extent) || as.extent<=0 || as.grid_points<5 || as.grid_points>33 || as.grid_points%2==0)
                    throw std::runtime_error("action pair_product: positive extent and odd grid_points in [5,33] required");
            } else if (comp == "nodes") {
                // Fixed-node restriction (Ceperley, J. Stat. Phys. 63, 1237 (1991)): the
                // only source of nodes is the wavefunction: block.
                as.component = comp;
                as.form = "wavefunction";
                const YAML::Node form = kv->second;
                if (form.IsMap()) {
                    require_keys(form, "action nodes", {"form"});
                    if (form["form"]) as.form = form["form"].as<std::string>();
                } else if (form.IsScalar()) {
                    as.form = form.as<std::string>();
                } else if (!form.IsNull()) {
                    throw std::runtime_error("action nodes: expected 'wavefunction' or a map");
                }
                if (as.form != "wavefunction")
                    throw std::runtime_error(
                        "action nodes: unknown source '" + as.form +
                        "' (expected 'wavefunction': the determinants of the "
                        "wavefunction: items)");
            } else {
                throw std::runtime_error(
                    "unknown action component: '" + comp +
                    "' (expected kinetic, potential, primitive, pair_product or nodes)");
            }
            c.action.push_back(std::move(as));
        }
        // At most one entry per component (primitive counts as both).
        bool has_kin = false, has_pot = false, has_nodes = false;
        for (const ActionItemSpec& a : c.action) {
            if (a.component == "nodes") {
                if (has_nodes)
                    throw std::runtime_error("action: nodes component given twice");
                has_nodes = true;
            }
            if (a.component == "primitive" || a.component == "kinetic") {
                if (has_kin)
                    throw std::runtime_error(
                        "action: kinetic component given twice");
                has_kin = true;
            }
            if (a.component == "primitive" || a.component == "potential" || a.component == "pair_product") {
                if (has_pot)
                    throw std::runtime_error(
                        "action: potential component given twice");
                has_pot = true;
            }
        }
    } else {
        // Default: primitive (kinetic springs + tau*V).  Materialized in the
        // parsed Config so a snapshot round-trip (which writes it explicitly)
        // compares equal to the original document.
        c.action.push_back(ActionItemSpec{"primitive", "primitive"});
    }

    for (const YAML::Node& qn : root["qmc"]) {
        const std::string method =
            qn["method"] ? qn["method"].as<std::string>() : "";

        if (method == "pimc") {
            // ── PIMC run ────────────────────────────────────────────────
            require_keys(qn, "pimc", {"method", "device", "threads", "samples",
                "equilibration_blocks", "accumulation_blocks", "steps_per_block",
                "sub_steps", "time_step", "beta", "energy", "compare_thermodynamic", "statistics", "sign_threshold", "moves", "estimators", "paths_in", "paths_out"});
            PimcSpec p;
            parse_run_keys(qn, p);
            if (qn["paths_in"]) p.paths_in = qn["paths_in"].as<std::string>();
            if (qn["paths_out"]) p.paths_out = qn["paths_out"].as<std::string>();
            if (qn["beta"]) p.beta = qn["beta"].as<double>();
            if (qn["energy"]) p.energy = qn["energy"].as<std::string>();
            if (p.energy != "thermodynamic" && p.energy != "centroid_virial" && p.energy != "periodic_virial")
                throw std::runtime_error(
                    "pimc: energy estimator '" + p.energy +
                    "' not supported (choose 'thermodynamic', 'centroid_virial' or 'periodic_virial')");
            if (qn["compare_thermodynamic"])
                p.compare_thermodynamic=qn["compare_thermodynamic"].as<bool>();
            if (p.compare_thermodynamic && p.energy == "thermodynamic")
                throw std::runtime_error("pimc: compare_thermodynamic requires energy: centroid_virial or periodic_virial");
            if (!qn["beta"])
                throw std::runtime_error("pimc: `beta:` is required");
            if (qn["statistics"])
                p.statistics = qn["statistics"].as<std::string>();
            if (p.statistics != "" && p.statistics != "boltzmannon" &&
                p.statistics != "fixed_node")
                throw std::runtime_error(
                    "pimc: unknown statistics '" + p.statistics +
                    "' (omit it for the particles' own fermion/boson "
                    "statistics, or write 'boltzmannon' or 'fixed_node')");
            if (qn["sign_threshold"]) {
                p.sign_threshold = qn["sign_threshold"].as<double>();
                if (!std::isfinite(p.sign_threshold) || !(p.sign_threshold > 0.0))
                    throw std::runtime_error("pimc: `sign_threshold:` must be finite and > 0");
            }
            if (!std::isfinite(p.beta) || !(p.beta > 0.0))
                throw std::runtime_error("pimc: `beta:` must be finite and > 0");
            if (!std::isfinite(p.time_step) || !(p.time_step > 0.0))
                throw std::runtime_error("pimc: `time_step:` must be finite and > 0");
            if (p.samples <= 0)
                throw std::runtime_error("pimc: `samples:` must be > 0");
            if (p.equilibration_blocks < 0 || p.accumulation_blocks <= 0 ||
                p.steps_per_block <= 0 || p.sub_steps <= 0)
                throw std::runtime_error(
                    "pimc: need equilibration_blocks >= 0 and "
                    "accumulation_blocks/steps_per_block/sub_steps > 0");
            if (qn["moves"]) {
                if (!qn["moves"].IsSequence())
                    throw std::runtime_error("pimc: moves must be a list");
                for (const YAML::Node& mn : qn["moves"]) {
                    require_keys(mn, "pimc move", {"bisection", "centroid", "single_slice", "permutation", "joint_permutation", "worm"});
                    if (mn.size() != 1)
                        throw std::runtime_error("pimc move: expected exactly one move type");
                    const auto payload = mn.begin()->second;
                    if (mn["bisection"])
                        require_keys(payload, "pimc bisection move", {"levels", "weight"});
                    else if (mn["permutation"])
                        require_keys(payload, "pimc permutation move", {"levels", "max_cycle", "weight"});
                    else if (mn["joint_permutation"])
                        require_keys(payload, "pimc joint_permutation move",
                                     {"pair", "levels", "max_cycle", "pair_radius", "weight"});
                    else if (mn["worm"])
                        require_keys(payload, "pimc worm move",
                                     {"group", "max_length", "swap_length", "C", "weight"});
                    else
                        require_keys(payload, "pimc move", {"step", "weight"});
                    PimcMoveSpec ms;
                    if (mn["worm"]) {
                        // Explicit lengths (no default can follow M); C is
                        // optional and tuned in equilibration when absent.
                        ms.type = "worm";
                        if (!payload["group"] || !payload["max_length"] || !payload["swap_length"])
                            throw std::runtime_error(
                                "pimc worm move: `group`, `max_length` and `swap_length` are required");
                        ms.group = payload["group"].as<std::string>();
                        ms.max_length = payload["max_length"].as<int>();
                        ms.swap_length = payload["swap_length"].as<int>();
                        if (payload["C"]) ms.worm_constant = payload["C"].as<double>();
                        if (payload["weight"]) ms.weight = payload["weight"].as<double>();
                    } else if (mn["joint_permutation"]) {
                        ms.type = "joint_permutation";
                        if (!payload["pair"] || !payload["levels"] || !payload["max_cycle"] ||
                            !payload["pair_radius"])
                            throw std::runtime_error(
                                "pimc joint_permutation move: `pair`, `levels`, `max_cycle` "
                                "and `pair_radius` are required");
                        if (!payload["pair"].IsSequence() || payload["pair"].size() != 2)
                            throw std::runtime_error(
                                "pimc joint_permutation move: `pair` must list two groups");
                        ms.pair[0] = payload["pair"][0].as<std::string>();
                        ms.pair[1] = payload["pair"][1].as<std::string>();
                        ms.levels = payload["levels"].as<int>();
                        ms.max_cycle = payload["max_cycle"].as<int>();
                        ms.pair_radius = payload["pair_radius"].as<double>();
                        if (payload["weight"]) ms.weight = payload["weight"].as<double>();
                    } else if (mn["permutation"]) {
                        // Explicit window and cycle length: no default can
                        // follow M, and the snapshot replays them as written.
                        ms.type = "permutation";
                        if (!payload["levels"] || !payload["max_cycle"])
                            throw std::runtime_error(
                                "pimc permutation move: `levels` and `max_cycle` are required");
                        ms.levels = payload["levels"].as<int>();
                        ms.max_cycle = payload["max_cycle"].as<int>();
                        if (payload["weight"]) ms.weight = payload["weight"].as<double>();
                    } else if (mn["bisection"]) {
                        ms.type = "bisection";
                        if (mn["bisection"]["levels"])
                            ms.levels = mn["bisection"]["levels"].as<int>();
                        if (mn["bisection"]["weight"])
                            ms.weight = mn["bisection"]["weight"].as<double>();
                    } else if (mn["centroid"]) {
                        ms.type = "centroid";
                        if (mn["centroid"]["step"])
                            ms.step = mn["centroid"]["step"].as<double>();
                        if (mn["centroid"]["weight"])
                            ms.weight = mn["centroid"]["weight"].as<double>();
                    } else if (mn["single_slice"]) {
                        ms.type = "single_slice";
                        if (mn["single_slice"]["step"])
                            ms.step = mn["single_slice"]["step"].as<double>();
                        if (mn["single_slice"]["weight"])
                            ms.weight = mn["single_slice"]["weight"].as<double>();
                    } else {
                        throw std::runtime_error(
                            "pimc: unknown move type (expected bisection, "
                            "centroid, single_slice, permutation, joint_permutation or worm)");
                    }
                    p.moves.push_back(std::move(ms));
                }
            }
            if (qn["estimators"])
                p.estimators = parse_estimators(qn["estimators"], "pimc");
            c.sampling_order.push_back(PimcStageIndex{c.pimc.size()});
            c.pimc.push_back(std::move(p));
        } else if (method == "optimizer") {
            // `param_tol` (a bound on the parameter change) was replaced by
            // `psi_step_tol`, a different criterion (the relative change of ψ),
            // and was then silently ignored.  Its value does not carry over.
            if (qn["param_tol"])
                throw std::runtime_error(
                    "optimizer: 'param_tol' was removed; convergence is now "
                    "'psi_step_tol' (relative wavefunction change of the undamped "
                    "step, default 1e-3), a different criterion.  Delete the key, "
                    "or set psi_step_tol.");
            require_stage_keys(qn, "optimizer",
                {"type", "random_walk", "parameters", "max_iterations", "psi_step_tol",
                 "energy_tol", "convergence_window", "num_shifts", "max_param_change",
                 "cost_increase_tol",
                 "target_shift_i", "max_failed_attempts", "shift_i", "shift_s",
                 "rescale", "xi"});
            OptimizerSpec opt;
            parse_run_keys(qn, opt);
            if (qn["type"])          opt.type = qn["type"].as<std::string>();
            if (opt.type != "linear")
                throw std::runtime_error("optimizer: type '" + opt.type +
                                         "' is not supported (only 'linear')");
            if (qn["random_walk"])   opt.random_walk = qn["random_walk"].as<std::string>();
            if (opt.random_walk != "metropolis" && opt.random_walk != "langevin")
                throw std::runtime_error("optimizer: random_walk must be 'metropolis' "
                                         "or 'langevin', got '" + opt.random_walk + "'");
            if (qn["parameters"])    opt.parameters = qn["parameters"].as<std::string>();
            if (qn["max_iterations"])  opt.max_iterations = qn["max_iterations"].as<int>();
            if (qn["psi_step_tol"])   opt.psi_step_tol = qn["psi_step_tol"].as<double>();
            if (qn["energy_tol"])     opt.energy_tol = qn["energy_tol"].as<double>();
            if (qn["convergence_window"])
                opt.convergence_window = qn["convergence_window"].as<int>();
            if (opt.convergence_window < 3)
                throw std::runtime_error("optimizer: convergence_window must be at least 3 "
                                         "(fresh energies the convergence trend is fitted "
                                         "to), got " +
                                         std::to_string(opt.convergence_window));
            if (qn["num_shifts"])     opt.num_shifts = qn["num_shifts"].as<int>();
            if (qn["max_param_change"]) opt.max_param_change = qn["max_param_change"].as<double>();
            if (qn["cost_increase_tol"]) opt.cost_increase_tol = qn["cost_increase_tol"].as<double>();
            if (qn["target_shift_i"]) opt.target_shift_i = qn["target_shift_i"].as<double>();
            if (qn["max_failed_attempts"]) opt.max_failed_attempts = qn["max_failed_attempts"].as<int>();
            if (qn["shift_i"])        opt.shift_i = qn["shift_i"].as<double>();
            if (qn["shift_s"])        opt.shift_s = qn["shift_s"].as<double>();
            if (qn["rescale"])        opt.rescale = qn["rescale"].as<double>();
            if (qn["xi"])             opt.xi = qn["xi"].as<double>();
            c.optimizers.push_back(std::move(opt));
        } else {
            // A misspelled or absent method previously fell through to VMC,
            // silently changing the physical algorithm (including DMC runs).
            if (method != "vmc" && method != "dmc")
                throw std::runtime_error("qmc: unknown or missing method '" + method +
                                         "' (expected vmc, dmc, pimc, optimizer)");
            // `feedback` was removed: it divided ln(W/N) by a bare constant, so
            // its real loop gain was τ·sub_steps/feedback — unstable at large τ.
            // Refuse it rather than silently reinterpret an old config.
            if (qn["feedback"])
                throw std::runtime_error(
                    "qmc: 'feedback' was removed; use 'population_generations: g' "
                    "(dimensionless, E_T = E_ref - ln(W/N)/(g*time_step*sub_steps), "
                    "default 10)");
            // Parsed for years, never read by the DMC driver: its branching has no
            // age cut, no branch interval and no reconfiguration switch.
            for (const char* key : {"max_age", "branch_interval", "reconfiguration"})
                if (qn[key])
                    throw std::runtime_error(
                        std::string("qmc: '") + key + "' was removed: the DMC driver "
                        "never used it (population control is set by "
                        "population_generations).  Delete the key.");
            if (method == "dmc")
                require_stage_keys(qn, "dmc", {"random_walk", "population_generations",
                                               "branch_cutoff", "reweighting",
                                               "walkers_in", "walkers_out", "estimators",
                                               "forward_walking"});
            else
                require_stage_keys(qn, "vmc", {"random_walk", "walkers_in", "walkers_out",
                                               "estimators"});
            QmcSpec q;
            q.method = method;
            parse_run_keys(qn, q);
            if (qn["random_walk"])   q.random_walk = qn["random_walk"].as<std::string>();
            // random_walk names the move the method makes, so a value it cannot
            // honour is refused instead of silently replaced.  DMC always moves
            // by importance-sampled drift-diffusion ('langevin' is a synonym).
            if (method == "vmc" && !q.random_walk.empty() &&
                q.random_walk != "metropolis" && q.random_walk != "langevin")
                throw std::runtime_error("vmc: random_walk must be 'metropolis' or "
                                         "'langevin', got '" + q.random_walk + "'");
            if (method == "dmc" && !q.random_walk.empty() &&
                q.random_walk != "drift_diffusion" && q.random_walk != "langevin")
                throw std::runtime_error(
                    "dmc: random_walk must be 'drift_diffusion' (or its synonym "
                    "'langevin'): DMC always moves walkers by importance-sampled "
                    "drift-diffusion; got '" + q.random_walk + "'");
            if (qn["population_generations"]) {
                q.population_generations = qn["population_generations"].as<double>();
                // Loop gain 1/g: g ≤ ½ oscillates without bound, g < 1 overshoots.
                if (q.population_generations < 1.0)
                    throw std::runtime_error(
                        "qmc: population_generations must be >= 1 (loop gain 1/g; "
                        "g < 1 overshoots, g <= 0.5 is unstable)");
            }
            // branch_cutoff: a scheme name, or {scheme, alpha} (alpha only for
            // the universal scheme, whose scale it sets).
            if (const YAML::Node bc = qn["branch_cutoff"]) {
                if (bc.IsScalar()) {
                    q.branch_cutoff = bc.as<std::string>();
                } else if (bc.IsMap()) {
                    require_keys(bc, "dmc: branch_cutoff", {"scheme", "alpha"});
                    if (!bc["scheme"])
                        throw std::runtime_error("dmc: branch_cutoff needs `scheme`");
                    q.branch_cutoff = bc["scheme"].as<std::string>();
                    if (bc["alpha"]) q.branch_cutoff_alpha = bc["alpha"].as<double>();
                } else {
                    throw std::runtime_error(
                        "dmc: branch_cutoff must be a scheme name or {scheme, alpha}");
                }
                if (q.branch_cutoff != "universal" && q.branch_cutoff != "drv" &&
                    q.branch_cutoff != "none")
                    throw std::runtime_error("dmc: branch_cutoff scheme must be "
                                             "'universal', 'drv' or 'none', got '" +
                                             q.branch_cutoff + "'");
                if (q.branch_cutoff_alpha && q.branch_cutoff != "universal")
                    throw std::runtime_error("dmc: branch_cutoff alpha belongs to the "
                                             "universal scheme only");
                if (q.branch_cutoff_alpha &&
                    !(std::isfinite(*q.branch_cutoff_alpha) && *q.branch_cutoff_alpha > 0.0))
                    throw std::runtime_error(
                        "dmc: branch_cutoff alpha must be finite and positive");
            }
            // The suppression's calibration keys belong only to suppressed.
            // effective isolates Eq. 10's time step with f exactly one.
            if (const YAML::Node rw = qn["reweighting"]) {
                if (rw.IsScalar()) {
                    q.reweighting = rw.as<std::string>();
                } else if (rw.IsMap()) {
                    require_keys(rw, "dmc: reweighting",
                                 {"type", "correlation_time", "kappa", "correlation_sampling"});
                    if (!rw["type"])
                        throw std::runtime_error("dmc: reweighting needs `type`");
                    q.reweighting = rw["type"].as<std::string>();
                    if (const YAML::Node ct = rw["correlation_time"]) {
                        if (ct.as<std::string>() != "auto") {
                            q.correlation_time = ct.as<double>();
                            if (!(std::isfinite(*q.correlation_time) && *q.correlation_time > 0.0))
                                throw std::runtime_error(
                                    "dmc: reweighting correlation_time must be 'auto' or a "
                                    "finite positive imaginary time");
                        }
                    }
                    if (rw["kappa"]) {
                        q.suppression_kappa = rw["kappa"].as<double>();
                        if (!(std::isfinite(*q.suppression_kappa) && *q.suppression_kappa > 0.0))
                            throw std::runtime_error(
                                "dmc: reweighting kappa must be finite and positive");
                    }
                    if (const YAML::Node sampling = rw["correlation_sampling"]) {
                        require_keys(sampling, "dmc: correlation_sampling",
                                     {"equilibration_sweeps", "min_sweeps", "max_sweeps"});
                        auto count = [&](const char* key, std::optional<std::size_t>& out,
                                         int minimum) {
                            if (!sampling[key]) return;
                            const int value = sampling[key].as<int>();
                            if (value < minimum)
                                throw std::runtime_error(std::string("dmc: correlation_sampling ") +
                                    key + " must be >= " + std::to_string(minimum));
                            out = static_cast<std::size_t>(value);
                        };
                        count("equilibration_sweeps", q.correlation_equilibration_sweeps, 0);
                        count("min_sweeps", q.correlation_min_sweeps, 2);
                        count("max_sweeps", q.correlation_max_sweeps, 2);
                        if (q.correlation_min_sweeps.value_or(256) >
                            q.correlation_max_sweeps.value_or(8192))
                            throw std::runtime_error(
                                "dmc: correlation_sampling needs min_sweeps <= max_sweeps");
                        if (q.correlation_time)
                            throw std::runtime_error(
                                "dmc: correlation_sampling is unused with a fixed correlation_time");
                    }
                } else {
                    throw std::runtime_error(
                        "dmc: reweighting must be a type name or {type, correlation_time, "
                        "kappa, correlation_sampling}");
                }
                if (q.reweighting != "bare" && q.reweighting != "effective" &&
                    q.reweighting != "suppressed")
                    throw std::runtime_error("dmc: reweighting type must be 'bare', 'effective' or "
                                             "'suppressed', got '" + q.reweighting + "'");
                if (q.reweighting != "suppressed" &&
                    (rw.IsMap() && (rw["correlation_time"] || rw["kappa"] ||
                                    rw["correlation_sampling"])))
                    throw std::runtime_error("dmc: reweighting correlation_time, sampling and kappa "
                                             "belong to the suppressed type only");
            }
            // Walker-population checkpoint: seed this run's walkers from a saved
            // population / save the final population (see io/walker_io.h).
            if (qn["walkers_in"])       q.walkers_in = qn["walkers_in"].as<std::string>();
            if (qn["walkers_out"])      q.walkers_out = qn["walkers_out"].as<std::string>();

            // Opt-in observable estimators (absent → none).
            if (qn["estimators"])
                q.estimators = parse_estimators(qn["estimators"], "qmc");

            // Pure estimators (DMC only; require_stage_keys refused it in VMC).
            // The cross-key rules (whole steps, interval | steps_per_block,
            // lags within the equilibration) are the DMC driver's.
            if (const YAML::Node fw = qn["forward_walking"]) {
                if (!fw.IsMap())
                    throw std::runtime_error(
                        "dmc: forward_walking must be a map {lags: [...], interval: n}");
                require_keys(fw, "dmc: forward_walking", {"lags", "interval"});
                const YAML::Node lags = fw["lags"];
                if (!lags || !lags.IsSequence() || lags.size() == 0)
                    throw std::runtime_error(
                        "dmc: forward_walking needs `lags`, a nonempty list of imaginary times");
                ForwardWalkingSpec spec;
                for (const YAML::Node& lag : lags) {
                    const double t = lag.as<double>();
                    if (!(std::isfinite(t) && t >= 0.0))
                        throw std::runtime_error(
                            "dmc: forward_walking lags must be finite imaginary times >= 0");
                    if (!spec.lags.empty() && !(t > spec.lags.back()))
                        throw std::runtime_error(
                            "dmc: forward_walking lags must be strictly increasing");
                    spec.lags.push_back(t);
                }
                if (fw["interval"]) {
                    spec.interval = fw["interval"].as<int>();
                    if (spec.interval < 1)
                        throw std::runtime_error(
                            "dmc: forward_walking interval must be at least 1 step");
                }
                q.forward_walking = std::move(spec);
            }
            c.sampling_order.push_back(QmcStageIndex{c.qmc.size()});
            c.qmc.push_back(std::move(q));
        }
    }

    // A zero seed (explicit or defaulted) means "pick one for me": resolve it to
    // a concrete non-zero value now so every driver and the run snapshot share
    // the exact seed that was used.
    c.seed = resolve_seed(c.seed);

    return c;
}

Config parse_config_file(const std::string& path) {
    std::ifstream file(path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return parse_config(buffer.str());
}

}  // namespace alea::io
