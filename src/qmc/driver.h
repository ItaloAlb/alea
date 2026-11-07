// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>
#include <vector>

#include "run_result.h"

namespace alea {

class WalkerSet;

// Abstract base for all QMC methods (VMC, DMC, linear optimiser, etc.).
// Each concrete driver owns its own wavefunction, Hamiltonian, and ensemble
// clones so that methods run serially with shared initial state without
// interference.
class Driver {
public:
    virtual ~Driver() = default;

    // Execute the method.  May modify the wavefunction parameters in place
    // (e.g. the optimiser updates them).  Subsequent drivers see the new
    // values because they share the same TrialWavefunction reference.
    virtual void run() = 0;

    // One-line summary of the result, suitable for printing.
    [[nodiscard]] virtual std::string summary() const = 0;

    // Method-agnostic snapshot of the run for persistence (see RunResult).
    // Built from in-memory accumulators after run() — performs no I/O itself.
    [[nodiscard]] virtual RunResult run_result() const = 0;

    // Final walker population after run(), for checkpointing (walkers_out).
    // Null for methods whose final population is not physically meaningful
    // (VMC, optimiser); DMC returns its branched global population.  The
    // caller (main) serialises it via io::write_walkers — no I/O here.
    [[nodiscard]] virtual const WalkerSet* final_population() const {
        return nullptr;
    }

    // ── Driver-to-driver walker carry-over (QMCPACK: QMCDriverNew::finalize
    //    saves the population, MCPopulation::createWalkers seeds the next
    //    driver from it) ──────────────────────────────────────────────────
    //
    // Final walker configurations after run(), as AoS rows [w][p][d].  Empty
    // for a driver with nothing worth handing on (the default; e.g. the
    // optimiser), in which case the next driver keeps its own initial state.
    [[nodiscard]] virtual std::vector<double> final_configurations() const {
        return {};
    }

    // Re-read the initial configurations from the ensemble the driver was
    // built on — call after overwriting the ensemble's walker positions and
    // BEFORE run().  No-op for drivers that read the ensemble lazily.
    virtual void reload_configurations() {}
};

}  // namespace alea
