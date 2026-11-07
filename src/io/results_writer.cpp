// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "io/results_writer.h"

#include <exception>
#include <stdexcept>
#include <string>
#include <system_error>

#include <highfive/H5File.hpp>

#include "io/logger.h"

namespace alea::io {

void write_run_result(const RunResult&             result,
                      const std::filesystem::path& run_dir,
                      std::size_t                  driver_index) {
    if (run_dir.empty()) return;

    const std::filesystem::path path =
        run_dir / (std::to_string(driver_index) + "_" + result.method + ".h5");
    // Built under a temporary name and renamed only once complete.  HDF5 has
    // no transaction: a dataset that fails mid-file leaves the datasets before
    // it on disk, and that partial file opens cleanly in any reader.
    std::filesystem::path partial = path;
    partial += ".partial";

    try {
        {
            HighFive::File file(partial.string(), HighFive::File::Overwrite);

            // Series → 1-D datasets.  Skip empty series: a zero-length dataset
            // carries no information and trips up some HighFive versions.
            for (const auto& [name, values] : result.series)
                if (!values.empty())
                    file.createDataSet(name, values);

            // Scalars and metadata → root attributes.
            for (const auto& [name, value] : result.scalars)
                file.createAttribute(name, value);
            for (const auto& [name, value] : result.metadata)
                file.createAttribute(name, value);
        }  // closed (flushed, and unlocked on Windows) before the rename
        std::filesystem::rename(partial, path);
    } catch (const std::exception& e) {
        std::error_code ignored;  // best effort: the write error is the one to report
        std::filesystem::remove(partial, ignored);
        throw std::runtime_error("results: failed to write " + path.string() + ": " +
                                 e.what());
    }
    Logger::milestone("results: wrote " + path.string());
}

}  // namespace alea::io
