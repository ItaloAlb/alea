// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Strict, versioned HDF5 serialization of phase-1 PIMC bead configurations.
// Binary64 SoA data is copied without changing the physical path or cell.
#include "io/path_io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

#include <highfive/H5File.hpp>

#include "population/simulation_cell.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace alea::io {
namespace {

constexpr std::uint64_t kVersion = 1;
constexpr const char* kFormat = "alea_pimc_paths";
constexpr const char* kLayout = "replica_particle_dimension_slice_soa";
constexpr double kGridRoundoff = 8.0 * std::numeric_limits<double>::epsilon();

// Fail consistently so caller can distinguish checkpoint errors from no input.
[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error("path checkpoint: " + message);
}

// Multiply shape dimensions without allowing an overflow to imitate valid data.
std::size_t product(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
        fail("shape product overflows size_t");
    return a * b;
}

// Validate metadata before reading or allocating payload arrays.
void validate_metadata(const PathCheckpoint& cp) {
    if (!cp.replicas || !cp.m || !cp.np || !cp.dim ||
        cp.dim > SimulationCell::kMaxDim)
        fail("replicas, slices, particles and supported dimension must be nonzero");
    product(product(product(cp.replicas, cp.np), cp.dim), cp.m);
    if (!std::isfinite(cp.beta) || cp.beta <= 0.0 ||
        !std::isfinite(cp.tau) || cp.tau <= 0.0)
        fail("beta and tau must be finite and positive");
    const double expected_beta = static_cast<double>(cp.m) * cp.tau;
    if (!std::isfinite(expected_beta) ||
        std::abs(cp.beta - expected_beta) >
            kGridRoundoff * std::max(cp.beta, expected_beta))
        fail("beta does not equal M*tau within floating-point roundoff");
    if (cp.lattice.empty()) return;
    if (cp.lattice.size() != cp.dim * cp.dim)
        fail("lattice shape must be dim*dim");
    for (double value : cp.lattice)
        if (!std::isfinite(value)) fail("lattice contains nonfinite values");
    const SimulationCell cell(cp.dim, cp.lattice);
    if (!std::isfinite(cell.volume()) || cell.volume() <= 0.0 ||
        !std::isfinite(cell.ws_radius()) || cell.ws_radius() <= 0.0)
        fail("lattice has invalid volume or minimum-image geometry");
    for (double value : cell.reciprocal())
        if (!std::isfinite(value)) fail("lattice reciprocal contains nonfinite values");
}

// Require native-width numeric storage rather than HDF5's silent conversions.
void require_type(const HighFive::DataType& actual,
                  const HighFive::DataType& expected, const std::string& name) {
    if (actual != expected) fail("unsupported serialized type for " + name);
}

// Numeric attributes are scalar, not singleton arrays with ambiguous shape.
template <class T>
T read_scalar(const HighFive::File& file, const std::string& name) {
    const auto attribute = file.getAttribute(name);
    if (!attribute.getSpace().getDimensions().empty())
        fail("attribute " + name + " must be scalar");
    require_type(attribute.getDataType(), HighFive::AtomicType<T>{}, name);
    T value{};
    attribute.read(value);
    return value;
}

// Read integral metadata losslessly into the platform's indexing type.
std::size_t read_size(const HighFive::File& file, const std::string& name) {
    const auto value = read_scalar<std::uint64_t>(file, name);
    if (value > std::numeric_limits<std::size_t>::max())
        fail("attribute " + name + " exceeds size_t");
    return static_cast<std::size_t>(value);
}

// Read a mandatory scalar string marker, rejecting array reinterpretation.
std::string read_marker(const HighFive::File& file, const std::string& name) {
    const auto attribute = file.getAttribute(name);
    if (!attribute.getSpace().getDimensions().empty() ||
        attribute.getDataType().getClass() != HighFive::DataTypeClass::String)
        fail("attribute " + name + " must be a scalar string");
    std::string result;
    attribute.read(result);
    return result;
}

// Validate rank/type/extent before allowing HighFive to allocate or read beads.
template <class T>
std::vector<T> read_flat(const HighFive::File& file, const std::string& name,
                         std::size_t count) {
    const auto dataset = file.getDataSet(name);
    if (dataset.getDimensions() != std::vector<std::size_t>{count})
        fail("dataset " + name + " has wrong rank or length");
    require_type(dataset.getDataType(), HighFive::AtomicType<T>{}, name);
    std::vector<T> result;
    dataset.read(result);
    return result;
}

// Exclusive staging directory avoids collisions without introducing a physics
// RNG. Its contents are owned by this call, never an existing checkpoint.
std::filesystem::path make_stage(const std::filesystem::path& destination) {
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    for (unsigned attempt = 0; attempt < 100; ++attempt) {
        const auto stage = destination.parent_path() /
                           (".alea_path_stage_" + std::to_string(tick) + "_" +
                            std::to_string(attempt));
        if (std::filesystem::create_directory(stage)) return stage;
    }
    fail("unable to reserve a staging directory");
}

// Replace only a fully closed stage; never delete the old checkpoint first.
void replace_checkpoint(const std::filesystem::path& stage,
                         const std::filesystem::path& destination) {
#ifdef _WIN32
    if (!MoveFileExW(stage.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING))
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
                                "replacing checkpoint");
#else
    std::filesystem::rename(stage, destination);
#endif
}

} // namespace

void validate_path_checkpoint(const PathCheckpoint& cp) {
    validate_metadata(cp);
    const auto count = product(product(product(cp.replicas, cp.np), cp.dim), cp.m);
    if (cp.positions.size() != count) fail("positions shape does not match metadata");
    if (cp.permutation.size() != product(cp.replicas, cp.np))
        fail("permutation shape does not match metadata");
    for (double bead : cp.positions)
        if (!std::isfinite(bead)) fail("positions contain nonfinite beads");
    // Each replica's permutation must be a bijection of the particles; which
    // exchanges are admissible (identity only, or within exchange classes)
    // is a property of the run, checked by the driver on import.
    std::vector<char> seen(cp.np);
    for (std::size_t r = 0; r < cp.replicas; ++r) {
        std::fill(seen.begin(), seen.end(), 0);
        for (std::size_t p = 0; p < cp.np; ++p) {
            const auto target = cp.permutation[r * cp.np + p];
            if (target >= cp.np || seen[target])
                fail("permutation is not a bijection in every replica");
            seen[target] = 1;
        }
    }
}

PathCheckpoint read_paths(const std::filesystem::path& path) {
    try {
        HighFive::File file(path.string(), HighFive::File::ReadOnly);
        if (read_marker(file, "format") != kFormat ||
            read_scalar<std::uint64_t>(file, "version") != kVersion)
            fail("unsupported format or version");
        if (read_marker(file, "layout") != kLayout)
            fail("unsupported positions layout");
        PathCheckpoint cp;
        cp.replicas = read_size(file, "n_replicas");
        cp.m = read_size(file, "n_slices");
        cp.np = read_size(file, "n_particles");
        cp.dim = read_size(file, "dim");
        cp.beta = read_scalar<double>(file, "beta");
        cp.tau = read_scalar<double>(file, "tau");
        const auto boundary = read_marker(file, "boundary");
        if (boundary == "periodic") {
            const auto lattice = file.getAttribute("lattice_vectors");
            if (!cp.dim || cp.dim > SimulationCell::kMaxDim ||
                lattice.getSpace().getDimensions() != std::vector<std::size_t>{cp.dim * cp.dim})
                fail("lattice attribute has wrong rank or length");
            require_type(lattice.getDataType(), HighFive::AtomicType<double>{}, "lattice_vectors");
            lattice.read(cp.lattice);
        } else if (boundary != "open" || file.hasAttribute("lattice_vectors")) {
            fail("unsupported or contradictory boundary metadata");
        }
        validate_metadata(cp);
        const auto count = product(product(product(cp.replicas, cp.np), cp.dim), cp.m);
        cp.positions = read_flat<double>(file, "positions", count);
        const auto permutation = read_flat<std::uint64_t>(file, "permutation", product(cp.replicas, cp.np));
        cp.permutation.reserve(permutation.size());
        for (auto value : permutation) {
            if (value > std::numeric_limits<std::size_t>::max())
                fail("permutation index exceeds size_t");
            cp.permutation.push_back(static_cast<std::size_t>(value));
        }
        validate_path_checkpoint(cp);
        return cp;
    } catch (const std::exception& error) {
        throw std::runtime_error("paths_in: failed to read " + path.string() + ": " + error.what());
    }
}

void write_paths(const std::filesystem::path& path, const PathCheckpoint& cp) {
    validate_path_checkpoint(cp);
    if (path.empty()) fail("output path is empty");
    std::filesystem::path stage;
    try {
        const auto destination = std::filesystem::absolute(path);
        stage = make_stage(destination);
        const auto staged_file = stage / "paths.h5";
        {
            HighFive::File file(staged_file.string(), HighFive::File::Excl);
            file.createAttribute("format", std::string(kFormat));
            file.createAttribute("version", kVersion);
            file.createAttribute("layout", std::string(kLayout));
            file.createAttribute("n_replicas", static_cast<std::uint64_t>(cp.replicas));
            file.createAttribute("n_slices", static_cast<std::uint64_t>(cp.m));
            file.createAttribute("n_particles", static_cast<std::uint64_t>(cp.np));
            file.createAttribute("dim", static_cast<std::uint64_t>(cp.dim));
            file.createAttribute("beta", cp.beta);
            file.createAttribute("tau", cp.tau);
            file.createAttribute("boundary", std::string(cp.lattice.empty() ? "open" : "periodic"));
            if (!cp.lattice.empty()) file.createAttribute("lattice_vectors", cp.lattice);
            file.createDataSet("positions", cp.positions);
            file.createDataSet("permutation", std::vector<std::uint64_t>(cp.permutation.begin(), cp.permutation.end()));
            file.flush();
            // HighFive's destructor logs close failures instead of throwing.
            // Explicit checked close prevents publishing a failed checkpoint;
            // the invalidated handle is ignored by the subsequent destructor.
            if (H5Fclose(file.getId()) < 0)
                fail("failed to close staged HDF5 checkpoint");
        }
        replace_checkpoint(staged_file, destination);
        std::error_code ec;
        std::filesystem::remove(stage, ec);
    } catch (const std::exception& error) {
        if (!stage.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(stage, ec);
        }
        throw std::runtime_error("paths_out: failed to write " + path.string() + ": " + error.what());
    }
}

void require_path_compatibility(const PathCheckpoint& cp, std::size_t replicas,
                                std::size_t m, std::size_t np, std::size_t dim,
                                double beta, double tau, const SimulationCell& cell) {
    validate_path_checkpoint(cp);
    if (cp.replicas != replicas || cp.m != m || cp.np != np || cp.dim != dim)
        fail("checkpoint replica/bead/particle/dimension shape differs from current run");
    if (cp.beta != beta || cp.tau != tau)
        fail("checkpoint beta or tau differs from current run");
    // Default-constructed open cells deliberately carry dimension zero: the
    // path shape, not an absent lattice, supplies the spatial dimension.
    const bool compatible_cell_dim = cell.dim() == dim ||
                                     (!cell.is_periodic() && cell.dim() == 0);
    if (!compatible_cell_dim || cell.is_periodic() != !cp.lattice.empty())
        fail("checkpoint boundary differs from current run");
    if (cell.is_periodic() &&
        (cell.vectors().size() != cp.lattice.size() ||
         !std::equal(cp.lattice.begin(), cp.lattice.end(), cell.vectors().begin())))
        fail("checkpoint lattice vectors differ from current run");
}

} // namespace alea::io
