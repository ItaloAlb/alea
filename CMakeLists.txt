# SPDX-FileCopyrightText: 2026 The Alea Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.20)
# C is enabled because HDF5 (fetched below) is a C library and needs a C compiler.
project(alea VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generates compile_commands.json for IDE IntelliSense (clangd, VS Code)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)

include(FetchContent)
# Eigen is header-only — use FetchContent_Populate to avoid its CMake project()
# which requires a working C compiler (problematic in MSYS2/Windows).
FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
)
FetchContent_GetProperties(Eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(Eigen)
endif()
set(eigen_INCLUDE_DIR ${eigen_SOURCE_DIR})

FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        0.8.0
)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_MakeAvailable(yaml-cpp)

# ── HDF5 (built from source) + HighFive (header-only C++ wrapper) ────────────
# Both come via FetchContent so there is no system dependency.  HDF5 is built
# lean: static only, no tools/examples/tests/HL, and no zlib/szip (compression
# off → no external deps).  The first configure/build is slow; later builds are
# incremental.
FetchContent_Declare(
    HDF5
    GIT_REPOSITORY https://github.com/HDFGroup/hdf5.git
    GIT_TAG        hdf5_1.14.5
    GIT_SHALLOW    TRUE
)
# GCC 15 on MSYS2/MinGW: _Float16 type is detected but FLT16_MAX and related
# macros are not available in float.h, causing compile errors in HDF5.
set(HDF5_ENABLE_NONSTANDARD_FEATURE_FLOAT16 OFF CACHE BOOL "" FORCE)
# GCC 15 is stricter about implicit function declarations (e.g., vasprintf).
# Downgrade from error to warning so HDF5 can compile.
# CMake reconfiguration must not append this flag repeatedly: that changes
# every HDF5 compile command and rebuilds hundreds of unchanged C sources.
if(NOT CMAKE_C_FLAGS MATCHES "(^|[ \t])-Wno-error=implicit-function-declaration($|[ \t])")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error=implicit-function-declaration" CACHE STRING "" FORCE)
endif()
set(HDF5_EXTERNALLY_CONFIGURED 1)
set(BUILD_TESTING            OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_TOOLS         OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_EXAMPLES      OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_UTILS         OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_HL_LIB        OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_CPP_LIB       OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_FORTRAN       OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_JAVA          OFF CACHE BOOL "" FORCE)
set(HDF5_ENABLE_Z_LIB_SUPPORT  OFF CACHE BOOL "" FORCE)
set(HDF5_ENABLE_SZIP_SUPPORT   OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS        OFF CACHE BOOL "" FORCE)
set(ONLY_SHARED_LIBS         OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(HDF5)

# HighFive is header-only.  Populate it (like Eigen above) WITHOUT running its
# CMakeLists — its unconditional install/export rules clash with a
# FetchContent-built HDF5.  We only need its include directory; the HDF5 headers
# and library come from linking hdf5-static below.
FetchContent_Declare(
    HighFive
    GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
    GIT_TAG        v2.10.0
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(HighFive)
if(NOT highfive_POPULATED)
    FetchContent_Populate(HighFive)
endif()
set(highfive_INCLUDE_DIR ${highfive_SOURCE_DIR}/include)

# ── fast-screened-coulomb (ItaloAlb): multilayer screened-Coulomb via the ETM
# method (Cavalcante PRB 97, 125427 (2018)).  Ships a Makefile + a single C99
# source `src/fast_screened_coulomb.c` (+ header), no CMake and no deps beyond
# libm.  Populate it (do NOT MakeAvailable — there is no CMakeLists) and compile
# the one .c straight into alea, exactly like the Eigen/HighFive populates above.
FetchContent_Declare(
    fsc
    GIT_REPOSITORY https://github.com/ItaloAlb/fast-screened-coulomb.git
    GIT_TAG        108390b2f10d96baf94840720daea1a026246075
)
FetchContent_GetProperties(fsc)
if(NOT fsc_POPULATED)
    FetchContent_Populate(fsc)
endif()
include(cmake/fsc_fit_derivative.cmake)
alea_require_fsc_fit_derivative("${fsc_SOURCE_DIR}" "${FETCHCONTENT_SOURCE_DIR_FSC}")
include(cmake/fsc_controlled_fit.cmake)
alea_require_fsc_controlled_fit("${fsc_SOURCE_DIR}" "${FETCHCONTENT_SOURCE_DIR_FSC}")
set(fsc_INCLUDE_DIR ${fsc_SOURCE_DIR}/src)
set(fsc_SOURCE ${fsc_SOURCE_DIR}/src/fast_screened_coulomb.c)

set(SOURCES
    ${fsc_SOURCE}
    main.cpp
    src/builders/center_of_mass_momentum.cpp
    src/builders/common.cpp
    src/builders/coulomb.cpp
    src/builders/inverse_power.cpp
    src/builders/estimators.cpp
    src/builders/ewald.cpp
    src/builders/exchange_classes.cpp
    src/builders/kinetic.cpp
    src/builders/moire_potential.cpp
    src/builders/rytova_keldysh.cpp
    src/builders/screened_coulomb.cpp
    src/builders/jastrow.cpp
    src/builders/kspace_jastrow.cpp
    src/wavefunction/kspace_jastrow.cpp
    src/builders/moire_jastrow.cpp
    src/builders/nosanow.cpp
    src/builders/pairing.cpp
    src/builders/slater.cpp
    src/estimators/collective_density.cpp
    src/estimators/density.cpp
    src/estimators/density_matrix.cpp
    src/estimators/electron_hole_proximity.cpp
    src/estimators/momentum_distribution.cpp
    src/estimators/pair_correlation.cpp
    src/estimators/structure_factor.cpp
    src/estimators/bragg_order.cpp
    src/estimators/bond_order.cpp
    src/hamiltonian/coulomb.cpp
    src/hamiltonian/ewald_term.cpp
    src/hamiltonian/hamiltonian.cpp
    src/hamiltonian/kinetic.cpp
    src/hamiltonian/moire.cpp
    src/hamiltonian/rytova_keldysh.cpp
    src/hamiltonian/screened_coulomb.cpp
    src/io/config_writer.cpp
    src/io/logger.cpp
    src/io/parser.cpp
    src/io/path_io.cpp
    src/io/results_writer.cpp
    src/io/walker_io.cpp
    src/numerics/ewald_breakup.cpp
    src/numerics/long_range_breakup.cpp
    src/numerics/pair_radial.cpp
    src/numerics/statistics.cpp
    src/population/batched_distance_table.cpp
    src/population/simulation_cell.cpp
    src/population/walker.cpp
    src/population/walker_ensemble.cpp
    src/population/particle_batch.cpp
    src/population/walker_set.cpp
    src/qmc/app.cpp
    src/qmc/cost_function.cpp
    src/qmc/linear_method.cpp
    src/qmc/linear_optimize.cpp
    src/qmc/vmc.cpp
    src/wavefunction/drummond_jastrow.cpp
    src/wavefunction/zero_energy_jastrow.cpp
    src/wavefunction/nosanow_jastrow.cpp
    src/wavefunction/moire_jastrow.cpp
    src/wavefunction/center_of_mass_momentum.cpp
    src/hamiltonian/harmonic_trap.cpp
    src/hamiltonian/inverse_power.cpp
    src/wavefunction/complex_slater_determinant.cpp
    src/wavefunction/pade_jastrow.cpp
    src/wavefunction/periodic_pair_sum.cpp
    src/wavefunction/kspace_pair_sum.cpp
    src/wavefunction/slater_determinant.cpp
    src/wavefunction/backflow_slater_determinant.cpp
    src/wavefunction/pair_orbital.cpp
    src/wavefunction/pairing_determinant.cpp
    src/wavefunction/trial_wavefunction.cpp
    src/wavefunction/wavefunction_component.cpp
    # src/qmc/optimizer.cpp  # not yet implemented
    src/qmc/dmc.cpp
    src/qmc/forward_walking.cpp
    src/qmc/population_control.cpp
    src/pimc/path_set.cpp
    src/pimc/action.cpp
    src/pimc/fixed_node_restriction.cpp
    src/pimc/pair_product_action.cpp
    src/numerics/finite_grid_pair.cpp
    src/pimc/move.cpp
    src/pimc/pimc.cpp
)

add_executable(alea ${SOURCES})

target_include_directories(alea PRIVATE
    src/ ${eigen_INCLUDE_DIR} ${highfive_INCLUDE_DIR} ${fsc_INCLUDE_DIR})

target_link_libraries(alea PRIVATE
    OpenMP::OpenMP_CXX
    yaml-cpp::yaml-cpp
    hdf5-static)

target_compile_options(alea PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -fopenmp-simd
    -fno-math-errno
    $<$<CONFIG:Release>:-O3 -march=native>
    $<$<CONFIG:Debug>:-O0 -g>
)
