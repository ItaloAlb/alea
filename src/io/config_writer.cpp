// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "io/config_writer.h"

#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>

#include <yaml-cpp/yaml.h>

#include "io/logger.h"
#include "io/parser.h"

namespace fs = std::filesystem;

namespace alea::io {

namespace {

// A YAML sequence written inline (`params: [a, b]`), at full precision.
template <class It>
YAML::Node flow_sequence(It first, It last) {
    YAML::Node seq(YAML::NodeType::Sequence);
    for (; first != last; ++first) seq.push_back(*first);
    seq.SetStyle(YAML::EmitterStyle::Flow);
    return seq;
}

}  // namespace

void write_config_snapshot(const std::string& source_yaml_path,
                           const std::vector<std::vector<double>>& final_params,
                           const fs::path& run_dir, std::uint64_t seed) {
    if (run_dir.empty()) return;

    try {
        YAML::Node root = YAML::LoadFile(source_yaml_path);

        // (0) Record the concrete seed the run used.  If the user asked for an
        // auto seed (0), this is the random value that was drawn — replaying the
        // snapshot reproduces the same chain instead of picking a new seed.
        root["project"]["seed"] = seed;

        // (1) Write the final parameter values back into the wavefunction
        // items.  build_wavefunction turns item i into component i, so
        // final_params[i] belongs to the i-th item, whatever its kind.  An item
        // with no entry (or an empty one) keeps what the document says.
        if (YAML::Node wf = root["wavefunction"]) {
            for (std::size_t i = 0; i < wf.size() && i < final_params.size(); ++i) {
                const std::vector<double>& vals = final_params[i];
                if (vals.empty()) continue;
                YAML::Node item = wf[i];

                if (YAML::Node jastrow = item["jastrow"]) {
                    const std::string type =
                        jastrow["type"] ? jastrow["type"].as<std::string>() : "";
                    if (type == "drummond") {
                        // [α0, α2..αN, L]: the final value is the cutoff
                        // length, which lives in its own `cutoff` field.
                        jastrow["params"] = flow_sequence(vals.begin(), vals.end() - 1);
                        jastrow["cutoff"] = vals.back();
                    } else if (type == "zero_energy") {
                        // [R_m]; the core and the cutoff are not parameters.
                        jastrow["matching_radius"] = vals.front();
                    } else {
                        // Padé [a₁, a₂..a_N, b₁..b_M]: a₁ is the derived cusp
                        // and never goes back into the config.  The numerator
                        // length is the document's own.
                        const std::size_t n_num =
                            jastrow["numerator"] ? jastrow["numerator"].size() : 0;
                        const auto num_begin = vals.begin() + 1;
                        const auto num_end   = num_begin + static_cast<std::ptrdiff_t>(n_num);
                        if (n_num > 0)
                            jastrow["numerator"] = flow_sequence(num_begin, num_end);
                        if (num_end != vals.end())
                            jastrow["denominator"] = flow_sequence(num_end, vals.end());
                    }
                } else if (YAML::Node slater = item["slater"]) {
                    // The determinant's optimised width.  It sets the NODES, so
                    // a snapshot that dropped it would replay the STARTING nodal
                    // surface.  Written into whichever form the document uses,
                    // `alpha` or a uniform `alphas`.
                    YAML::Node orb = slater["orbitals"];
                    if (!orb) continue;
                    const double width = vals.front();
                    if (orb["alpha"]) {
                        orb["alpha"] = width;
                    } else if (orb["alphas"]) {
                        const std::vector<double> widths(orb["alphas"].size(), width);
                        orb["alphas"] = flow_sequence(widths.begin(), widths.end());
                    }
                } else if (YAML::Node ns = item["nosanow"]) {
                    // [α], the localization width.
                    ns["alpha"] = vals.front();
                } else if (YAML::Node mj = item["moire_jastrow"]) {
                    // The (a_s, b_s) coefficients, all of them.
                    mj["params"] = flow_sequence(vals.begin(), vals.end());
                } else if (YAML::Node pr = item["pairing"]) {
                    // [p.., c_0, c_2..c_M] then L when there is a polynomial;
                    // these set the NODES, so the snapshot must replay them.
                    YAML::Node orb = pr["orbital"];
                    const bool poly =
                        orb["poly_order"] && orb["poly_order"].as<int>() >= 1;
                    const auto end = poly ? vals.end() - 1 : vals.end();
                    orb["params"] = flow_sequence(vals.begin(), end);
                    if (poly) orb["cutoff"] = vals.back();
                } else if (YAML::Node kj = item["kspace_jastrow"]) {
                    // One coefficient per shell (a Gaskell seed becomes explicit).
                    kj["params"] = flow_sequence(vals.begin(), vals.end());
                }
            }
        }

        // (2) Drop optimizer entries from the qmc list — they are not a
        // sampling method and their result is already folded into the params
        // above, so replaying must not re-optimise. Keep the mixed sampling
        // document order: PIMC's sampling-ordinal seeds do not include the
        // optimizer offset, so this removal preserves its replay stream.
        // A PIMC entry stays, and triggers (3).
        bool has_pimc = false;
        if (YAML::Node qmc = root["qmc"]) {
            YAML::Node kept(YAML::NodeType::Sequence);
            for (const YAML::Node& entry : qmc) {
                const std::string method =
                    entry["method"] ? entry["method"].as<std::string>() : "";
                if (method == "pimc") {
                    has_pimc = true;
                    // Record the resolved measurement even when YAML used defaults.
                    YAML::Node measurement=entry;
                    if (!measurement["energy"]) measurement["energy"]="thermodynamic";
                    if (!measurement["compare_thermodynamic"]) measurement["compare_thermodynamic"]=false;
                    if (!measurement["sign_threshold"])
                        measurement["sign_threshold"] = PimcSpec{}.sign_threshold;
                    // Likewise the resolved default move set (it depends on
                    // M = beta/time_step), so a replay never re-derives it
                    // from a possibly changed default.
                    if (!measurement["moves"] && measurement["beta"] &&
                        measurement["time_step"]) {
                        const double slices =
                            measurement["beta"].as<double>() /
                            measurement["time_step"].as<double>();
                        if (std::isfinite(slices) && slices >= 1.0 &&
                            slices < 1e15) {
                            YAML::Node moves(YAML::NodeType::Sequence);
                            for (const PimcMoveSpec& ms : default_pimc_moves(
                                     static_cast<std::size_t>(
                                         std::llround(slices)))) {
                                YAML::Node params(YAML::NodeType::Map);
                                if (ms.type == "bisection")
                                    params["levels"] = ms.levels;
                                else
                                    params["step"] = ms.step;
                                params["weight"] = ms.weight;
                                params.SetStyle(YAML::EmitterStyle::Flow);
                                YAML::Node item;
                                item[ms.type] = params;
                                moves.push_back(item);
                            }
                            measurement["moves"] = moves;
                        }
                    }
                }
                if (method != "optimizer") kept.push_back(entry);
            }
            root["qmc"] = kept;
        }

        // (3) A PIMC run records the default action explicitly: the action
        // is shared physics (declared next to hamiltonian:), and a snapshot
        // that omitted it would make the reader re-derive the default instead
        // of replaying the document.  VMC/DMC-only snapshots are untouched.
        if (has_pimc && !root["action"]) {
            YAML::Node act(YAML::NodeType::Sequence);
            YAML::Node item;
            item["primitive"] = YAML::Node(YAML::NodeType::Map);
            act.push_back(item);
            root["action"] = act;
        }
        if (has_pimc && root["action"] && root["action"].IsSequence()) {
            // Record resolved numerical-backend defaults as physics inputs;
            // a future default must not change a replayed pair action.
            for (YAML::Node item : root["action"]) {
                if (item.IsMap() && item.size()==1 && item.begin()->first.as<std::string>()=="pair_product") {
                    YAML::Node pair=item["pair_product"];
                    if (!pair["form"]) pair["form"]="finite_grid";
                    if (!pair["grid_points"]) pair["grid_points"]=17;
                }
            }
        }

        const fs::path out = run_dir / fs::path(source_yaml_path).filename();
        std::ofstream file(out);
        if (!file) {
            Logger::warn("could not open config snapshot for writing: " +
                         out.string());
            return;
        }
        file << "# Snapshot of the configuration used for this run.\n"
                "# Wavefunction params hold the final (post-optimisation) "
                "values; the\n# optimizer has been removed from the qmc list.\n";
        file << root << '\n';
    } catch (const std::exception& e) {
        Logger::warn(std::string("failed to write config snapshot: ") +
                     e.what());
    }
}

}  // namespace alea::io
