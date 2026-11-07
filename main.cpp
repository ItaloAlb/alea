// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/config_writer.h"
#include "io/logger.h"
#include "io/parser.h"
#include "io/path_io.h"
#include "io/results_writer.h"
#include "io/walker_io.h"
#include "population/walker_set.h"
#include "pimc/pimc.h"
#include "qmc/app.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <config.yaml>\n";
        return 1;
    }

    try {
        const alea::io::Config cfg = alea::io::parse_config_file(argv[1]);

        // Decide where results (and the log) live.  When logging is on we reuse
        // its run directory so the .h5 files sit next to alea.log; otherwise we
        // create the run directory only if output is requested.
        std::filesystem::path run_dir;
        const bool want_output = cfg.output;
        if (cfg.log) {
            alea::io::Logger::init("results", cfg.name);
            run_dir = alea::io::Logger::run_dir();
        } else if (want_output) {
            run_dir = alea::io::make_run_dir("results", cfg.name);
        }

        // Walker checkpoints (any qmc entry's `walkers_out`) live in their own
        // "checkpoints/<project>/<run>/" tree, sharing this run's timestamp
        // with "results/<project>/<run>/" so the two are trivially paired —
        // without either the YAML or the qmc layer ever naming a directory.
        std::filesystem::path checkpoint_dir;
        bool needs_checkpoint_dir = false;
        for (const alea::io::QmcSpec& spec : cfg.qmc)
            needs_checkpoint_dir |= !spec.walkers_out.empty();
        for (const alea::io::PimcSpec& spec : cfg.pimc)
            needs_checkpoint_dir |= !spec.paths_out.empty();
        if (needs_checkpoint_dir) {
            checkpoint_dir = run_dir.empty()
                ? alea::io::make_run_dir("checkpoints", cfg.name)
                : std::filesystem::path("checkpoints") / cfg.name
                      / run_dir.filename();
            std::filesystem::create_directories(checkpoint_dir);
        }

        alea::App app(cfg);
        auto pack = app.build_drivers(checkpoint_dir);

        std::size_t lost_results = 0;  // stages whose results file was not written

        for (std::size_t i = 0; i < pack.drivers.size(); ++i) {
            auto& d = pack.drivers[i];
            // Start from where the previous driver ended (QMCPACK's carry-over):
            // VMC → DMC begins in |ψ_T|², DMC τ₁ → DMC τ₂ in the projected state.
            if (i > 0) alea::carry_over_walkers(pack, i - 1, i);
            // A path checkpoint is a full M×N×d configuration, never a
            // walker/bead-0 projection. Validate geometry and grid in the
            // driver before allowing any path to change.
            auto* pimc = dynamic_cast<alea::Pimc*>(d.get());
            if (!pack.paths_in[i].empty()) {
                if (!pimc)
                    throw std::runtime_error("paths_in is only valid for PIMC");
                pimc->import_paths(alea::io::read_paths(pack.paths_in[i]));
            }
            d->run();
            if (!pack.paths_out[i].empty()) {
                if (!pimc)
                    throw std::runtime_error("paths_out is only valid for PIMC");
                alea::io::write_paths(pack.paths_out[i], pimc->export_paths());
            }
            std::cout << d->summary() << '\n';
            // A stage whose results file cannot be produced is reported at
            // once and fails the run at the end, but the remaining stages
            // still run: their results (and checkpoints) may well be
            // writable, and a finished stage is expensive to repeat.
            if (want_output) {
                try {
                    alea::io::write_run_result(d->run_result(), run_dir, i);
                } catch (const std::exception& e) {
                    std::cerr << "error: " << e.what() << '\n';
                    alea::io::Logger::error(e.what());  // alea.log, when logging
                    ++lost_results;
                }
            }
            // Walker checkpoint out (dmc only): serialise the final branched
            // population so a later run can resume / adiabatically continue it.
            if (const alea::WalkerSet* fin = d->final_population();
                fin && !pack.walkers_out[i].empty())
                alea::io::write_walkers(pack.walkers_out[i],
                                        fin->n_particles(), fin->dim(),
                                        fin->num_walkers(), fin->positions(), {}, -1,
                                        fin->cell().is_periodic()
                                            ? fin->cell().vectors()
                                            : std::span<const double>{});
        }

        // Drop a snapshot of the config used next to the log/results.  The
        // wavefunction now holds its final parameters, so if an optimiser ran
        // the snapshot carries the optimised values (and no optimizer entry).
        if (!run_dir.empty() && pack.psi) {
            std::vector<std::vector<double>> final_params;
            final_params.reserve(pack.psi->components().size());
            for (const auto& component : pack.psi->components()) {
                std::vector<double> values;
                const auto params = component->parameters();
                values.reserve(params.size());
                for (const alea::Parameter& p : params)
                    values.push_back(p.value);
                final_params.push_back(std::move(values));
            }
            alea::io::write_config_snapshot(argv[1], final_params, run_dir,
                                            cfg.seed);
        }

        if (cfg.log)
            alea::io::Logger::shutdown();
        if (lost_results > 0) {
            std::cerr << "error: " << lost_results
                      << " stage(s) finished but their results file was not written "
                         "(see above)\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
