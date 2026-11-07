// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace alea {

// Method-agnostic snapshot of a finished QMC run, produced by Driver::run_result().
//
// This is pure data — *no* I/O lives here, in keeping with the project rule that
// /src/qmc/ never touches the filesystem.  An io-layer writer (io::write_run_result)
// consumes this struct and serialises it (currently to HDF5: series → datasets,
// scalars/metadata → attributes).
//
// It is built once, at the end of run(), so the convenience of
// std::vector<std::pair<...>> (ordered, no hashing) costs nothing — this is never
// a hot path.  Series may have different lengths (e.g. per-(thread,block) energies
// vs per-block diagnostics); each maps to its own 1-D dataset.
struct RunResult {
    std::string method;  // "vmc" | "dmc" | "optimizer"

    // Scalar results and numeric metadata → HDF5 attributes (double).
    std::vector<std::pair<std::string, double>> scalars;

    // String metadata (seed, device, random_walk, …) → HDF5 attributes (string).
    std::vector<std::pair<std::string, std::string>> metadata;

    // Named 1-D time series → HDF5 datasets. Lengths may differ between entries.
    std::vector<std::pair<std::string, std::vector<double>>> series;

    void add_scalar(std::string name, double value) {
        scalars.emplace_back(std::move(name), value);
    }
    void add_metadata(std::string name, std::string value) {
        metadata.emplace_back(std::move(name), std::move(value));
    }
    void add_series(std::string name, std::vector<double> values) {
        series.emplace_back(std::move(name), std::move(values));
    }

    // Every series name once, and every attribute name (scalars and metadata
    // share the HDF5 root attributes) once.  The writer cannot store two
    // datasets or attributes under one name, and a failed write loses the rest
    // of the file, so each driver calls this at the end of run_result(): an
    // ambiguous result fails loudly before it is written.  Different estimator
    // instances can alias keys even with distinct namespaces (x's "_error" and
    // a separate x_error), which only this full check catches.
    void check_unique_keys() const {
        std::unordered_set<std::string> series_keys;
        for (const auto& [name, values] : series)
            if (!series_keys.insert(name).second)
                throw std::runtime_error(method + ": duplicate result series key '" +
                                         name + "'");
        std::unordered_set<std::string> attribute_keys;
        for (const auto& [name, value] : scalars)
            if (!attribute_keys.insert(name).second)
                throw std::runtime_error(method + ": duplicate result attribute key '" +
                                         name + "'");
        for (const auto& [name, value] : metadata)
            if (!attribute_keys.insert(name).second)
                throw std::runtime_error(method + ": duplicate result attribute key '" +
                                         name + "'");
    }
};

}  // namespace alea
