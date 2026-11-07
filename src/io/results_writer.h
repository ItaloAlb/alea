// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <filesystem>

#include "../qmc/run_result.h"

namespace alea::io {

// Serialise a finished run to  <run_dir>/<driver_index>_<method>.h5.
//
// Each named series in `result` becomes a 1-D HDF5 dataset; each scalar and each
// metadata entry becomes a root-level attribute.  The driver_index prefix keeps
// two drivers of the same method (e.g. two DMC runs) from clobbering each other.
//
// All or nothing: the file is built as `<name>.h5.partial` and renamed when
// complete, so a results file that exists holds everything the run produced.
// Any HDF5/filesystem error removes the partial file and throws
// std::runtime_error naming the file and the cause.  The caller decides what a
// lost file means for the run — main reports it and keeps running the
// remaining stages, then exits non-zero.  A no-op if `run_dir` is empty.
void write_run_result(const RunResult&             result,
                      const std::filesystem::path& run_dir,
                      std::size_t                  driver_index);

}  // namespace alea::io
