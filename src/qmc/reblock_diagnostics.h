// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <string>

#include "../numerics/statistics.h"
#include "run_result.h"

namespace alea {

// Reliability and output of a reblocked error bar (numerics::reblock), shared
// by the VMC and DMC results; the keys are the ones PIMC writes.
//
// effective blocks  n_eff = n / (1 + 2 tau_int)   (n = samples in the series)
// reliable          a reblocking plateau was found, the row that supplied the
//                   error has >= 16 blocks, and n_eff >= kMinEffectiveBlocks.
//
// Why 50 and not PIMC's 16: a short run underestimates its own tau_int, so its
// n_eff looks larger than it is and its error comes out too small.  In the WS2
// exciton DMC test (tau_int ~ 20 blocks), 14 runs of 512 blocks estimated
// n_eff = 19-49 and gave errors ~30 % below the scatter of independent runs;
// the same 14 seeds run for 4096 blocks estimated n_eff = 57-151 and gave the
// right error.  At n_eff = 50 the error bar is itself known to ~1/sqrt(100) = 10 %.
inline constexpr std::size_t kMinEffectiveBlocks = 50;

[[nodiscard]] inline double effective_blocks(const numerics::ReblockResult& stats,
                                             std::size_t n_samples) noexcept {
    return static_cast<double>(n_samples) / (1.0 + 2.0 * stats.tau_int);
}

[[nodiscard]] inline bool error_reliable(const numerics::ReblockResult& stats,
                                         std::size_t n_samples) noexcept {
    return stats.plateau_found && stats.selected_n_blocks >= 16 &&
           effective_blocks(stats, n_samples) >= static_cast<double>(kMinEffectiveBlocks);
}

inline void add_reblock_diagnostics(RunResult& r, const std::string& name,
                                    const numerics::ReblockResult& stats,
                                    std::size_t n_samples) {
    r.add_scalar(name + "_effective_blocks", effective_blocks(stats, n_samples));
    r.add_scalar(name + "_tau_int", stats.tau_int);
    r.add_scalar(name + "_error_naive", stats.naive_error);
    r.add_scalar(name + "_reblock_plateau", stats.plateau_found ? 1 : 0);
    r.add_scalar(name + "_error_reliable", error_reliable(stats, n_samples) ? 1 : 0);
    r.add_scalar(name + "_reblock_level", static_cast<double>(stats.selected_level));
    r.add_scalar(name + "_reblock_n_blocks", static_cast<double>(stats.selected_n_blocks));
    r.add_scalar(name + "_reblock_onset_level", static_cast<double>(stats.level));
    r.add_scalar(name + "_error_on_error", stats.error_on_error);
}

}  // namespace alea
