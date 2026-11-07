// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace alea::io {

// Write a snapshot of the configuration actually used for a run into `run_dir`,
// next to alea.log and the HDF5 results.
//
// The snapshot is the original YAML re-emitted with two changes so that a later
// re-run reproduces the *final* state of this run rather than its starting
// point:
//
//   1. Each wavefunction item's parameters are overwritten with the values the
//      wavefunction held at the end of the run (`final_params`).  When an
//      optimiser ran, these are the optimised parameters, so a subsequent
//      VMC/DMC launched from the snapshot starts from the improved
//      wavefunction instead of the original guess.  Per kind: a jastrow's
//      `params` (and a Drummond's `cutoff`), a moire_jastrow's `params`, and a
//      `slater` block's Gaussian width (`orbitals.alpha`, or every entry of a
//      uniform `orbitals.alphas`).  The width matters most: it sits inside the
//      determinant, so it sets the NODES, and a snapshot that dropped it would
//      send a replayed DMC back to the starting nodal surface while claiming to
//      carry the optimised guide.  Parameterless items are left untouched.
//
//   2. Optimizer entries (`method: optimizer`) are dropped from the `qmc`
//      list: the optimiser is not a sampling method, and its parameters are
//      already baked into (1), so it must not re-run on replay.
//
//   3. `project.seed` is overwritten with `seed` — the concrete value actually
//      used by the run.  When the user requested an auto seed (`seed: 0`), the
//      snapshot therefore records the random seed that was drawn, so a replay
//      reproduces the same Markov chain instead of drawing a fresh one.
//
//   4. A `method: pimc` entry records its resolved defaults — energy
//      estimator, the move set (io::default_pimc_moves(M) when `moves:` was
//      omitted), the default `action:` and pair-action backend — so a replay
//      reads them instead of re-deriving them from possibly changed defaults.
//
// `final_params[i]` belongs to the i-th item of the `wavefunction` list, in
// document order — build_wavefunction turns item i into component i.
// A no-op if `run_dir` is empty.  Any filesystem/YAML error is logged as a
// warning and swallowed — writing the snapshot must never take down a finished
// run.
void write_config_snapshot(const std::string& source_yaml_path,
                           const std::vector<std::vector<double>>& final_params,
                           const std::filesystem::path& run_dir,
                           std::uint64_t seed);

}  // namespace alea::io
