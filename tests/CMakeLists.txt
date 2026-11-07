# SPDX-FileCopyrightText: 2026 The Alea Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.20)
# C is enabled (besides CXX) because the fast-screened-coulomb dependency is a C99
# library; targets that use ScreenedCoulombTerm compile its single .c source.
project(alea_tests C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Lets clangd/VS Code pick up the right flags (-std=c++20, include dirs) for
# the sources under test while the root build is still in flux.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.4
)
FetchContent_MakeAvailable(Catch2)

FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        0.8.0
)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_MakeAvailable(yaml-cpp)

find_package(OpenMP REQUIRED)

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
# Eigen headers are in the source dir.
set(eigen_INCLUDE_DIR ${eigen_SOURCE_DIR})

# fast-screened-coulomb: single C99 source consumed by ScreenedCoulombTerm and
# its tests.  Populate (no CMakeLists upstream) and compile the .c into the
# targets that need it.
FetchContent_Declare(
    fsc
    GIT_REPOSITORY https://github.com/ItaloAlb/fast-screened-coulomb.git
    GIT_TAG        108390b2f10d96baf94840720daea1a026246075
)
FetchContent_GetProperties(fsc)
if(NOT fsc_POPULATED)
    FetchContent_Populate(fsc)
endif()
include(../cmake/fsc_fit_derivative.cmake)
alea_require_fsc_fit_derivative("${fsc_SOURCE_DIR}" "${FETCHCONTENT_SOURCE_DIR_FSC}")
include(../cmake/fsc_controlled_fit.cmake)
alea_require_fsc_controlled_fit("${fsc_SOURCE_DIR}" "${FETCHCONTENT_SOURCE_DIR_FSC}")
set(fsc_INCLUDE_DIR ${fsc_SOURCE_DIR}/src)
set(fsc_SOURCE ${fsc_SOURCE_DIR}/src/fast_screened_coulomb.c)

# ── HDF5 (built from source) + HighFive (header-only C++ wrapper) ────────────
# Same lean setup as the root CMakeLists: static only, no tools/HL/compression.
# Needed since walker_io.cpp (walker checkpoints) is linked into app.cpp tests.
FetchContent_Declare(
    HDF5
    GIT_REPOSITORY https://github.com/HDFGroup/hdf5.git
    GIT_TAG        hdf5_1.14.5
    GIT_SHALLOW    TRUE
)
set(HDF5_ENABLE_NONSTANDARD_FEATURE_FLOAT16 OFF CACHE BOOL "" FORCE)
# Keep this configuration idempotent: otherwise a harmless target addition
# changes all HDF5 compile commands on every Ninja CMake regeneration.
if(NOT CMAKE_C_FLAGS MATCHES "(^|[ \t])-Wno-error=implicit-function-declaration($|[ \t])")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error=implicit-function-declaration" CACHE STRING "" FORCE)
endif()
set(HDF5_EXTERNALLY_CONFIGURED 1)
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

# HighFive is header-only; populate WITHOUT running its CMakeLists (its install
# rules clash with a FetchContent-built HDF5).  Targets add its include/ and
# link hdf5-static.
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

# -fno-math-errno lets std::sqrt vectorize (vsqrtpd) instead of carrying the
# errno domain-check branch. Inert for correctness here (distances are ≥ 0).
add_compile_options(-fno-math-errno)

enable_testing()

# Sources under test live in ../src. Each test target compiles only the
# implementation files it actually needs, so the (currently in-flux) main
# build does not have to compile.
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)

add_executable(test_periodic_wavefunction
    test_periodic_wavefunction.cpp
    ${SRC_DIR}/hamiltonian/coulomb.cpp
    ${SRC_DIR}/hamiltonian/hamiltonian.cpp
    ${SRC_DIR}/hamiltonian/kinetic.cpp
    ${SRC_DIR}/hamiltonian/moire.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/wavefunction/drummond_jastrow.cpp
    ${SRC_DIR}/wavefunction/pade_jastrow.cpp
    ${SRC_DIR}/wavefunction/periodic_pair_sum.cpp
    ${SRC_DIR}/wavefunction/kspace_pair_sum.cpp
    ${SRC_DIR}/numerics/pair_radial.cpp
    ${SRC_DIR}/numerics/long_range_breakup.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
)
target_include_directories(test_periodic_wavefunction PRIVATE ${SRC_DIR})
target_link_libraries(test_periodic_wavefunction PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_periodic_wavefunction PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_ewald
    test_ewald.cpp
    ${SRC_DIR}/hamiltonian/ewald_term.cpp
    ${SRC_DIR}/numerics/ewald_breakup.cpp
    ${SRC_DIR}/numerics/long_range_breakup.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
)
target_include_directories(test_ewald PRIVATE ${SRC_DIR})
target_link_libraries(test_ewald PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_ewald PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd -fno-math-errno)

add_executable(test_ewald_screened
    test_ewald_screened.cpp
    ${SRC_DIR}/hamiltonian/ewald_term.cpp
    ${SRC_DIR}/hamiltonian/screened_coulomb.cpp
    ${SRC_DIR}/numerics/ewald_breakup.cpp
    ${SRC_DIR}/numerics/long_range_breakup.cpp
    ${fsc_SOURCE}
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
)
target_include_directories(test_ewald_screened PRIVATE ${SRC_DIR} ${fsc_INCLUDE_DIR})
target_link_libraries(test_ewald_screened PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_ewald_screened PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd -fno-math-errno)

add_executable(test_simulation_cell
    test_simulation_cell.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_simulation_cell PRIVATE ${SRC_DIR})
target_link_libraries(test_simulation_cell PRIVATE Catch2::Catch2WithMain)
# Production SIMD/FMA arithmetic, so the vectorized batch fold is compared
# with the scalar one under the instructions it really runs on.
target_compile_options(test_simulation_cell PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd
    $<$<CONFIG:Release>:-march=native> $<$<CONFIG:Release>:-fno-math-errno>)

add_executable(test_batched_distance_table
    test_batched_distance_table.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_batched_distance_table PRIVATE ${SRC_DIR})
target_link_libraries(test_batched_distance_table PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_batched_distance_table PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_walker
    test_walker.cpp
    ${SRC_DIR}/population/walker.cpp
)
target_include_directories(test_walker PRIVATE ${SRC_DIR})
target_link_libraries(test_walker PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_walker PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_walker_set
    test_walker_set.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_walker_set PRIVATE ${SRC_DIR})
target_link_libraries(test_walker_set PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_walker_set PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_pade_jastrow
    test_pade_jastrow.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/wavefunction/pade_jastrow.cpp
    ${SRC_DIR}/wavefunction/periodic_pair_sum.cpp
    ${SRC_DIR}/wavefunction/kspace_pair_sum.cpp
    ${SRC_DIR}/numerics/pair_radial.cpp
    ${SRC_DIR}/numerics/long_range_breakup.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_pade_jastrow PRIVATE ${SRC_DIR})
target_link_libraries(test_pade_jastrow PRIVATE Catch2::Catch2WithMain OpenMP::OpenMP_CXX)
target_compile_options(test_pade_jastrow PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_drummond_jastrow
    test_drummond_jastrow.cpp
    ${SRC_DIR}/wavefunction/drummond_jastrow.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_drummond_jastrow PRIVATE ${SRC_DIR})
target_link_libraries(test_drummond_jastrow PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_drummond_jastrow PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_zero_energy_jastrow
    test_zero_energy_jastrow.cpp
    ${SRC_DIR}/wavefunction/zero_energy_jastrow.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_zero_energy_jastrow PRIVATE ${SRC_DIR})
target_link_libraries(test_zero_energy_jastrow PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_zero_energy_jastrow PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_nosanow_jastrow
    test_nosanow_jastrow.cpp
    ${SRC_DIR}/wavefunction/nosanow_jastrow.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_nosanow_jastrow PRIVATE ${SRC_DIR})
target_link_libraries(test_nosanow_jastrow PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_nosanow_jastrow PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_coulomb
    test_coulomb.cpp
    ${SRC_DIR}/hamiltonian/coulomb.cpp
    ${SRC_DIR}/hamiltonian/rytova_keldysh.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_coulomb PRIVATE ${SRC_DIR})
target_link_libraries(test_coulomb PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_coulomb PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_inverse_power
    test_inverse_power.cpp
    ${SRC_DIR}/hamiltonian/inverse_power.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_inverse_power PRIVATE ${SRC_DIR})
target_link_libraries(test_inverse_power PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_inverse_power PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_long_range_breakup
    test_long_range_breakup.cpp
    ${SRC_DIR}/numerics/long_range_breakup.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_long_range_breakup PRIVATE ${SRC_DIR})
target_link_libraries(test_long_range_breakup PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_long_range_breakup PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_kspace_jastrow
    test_kspace_jastrow.cpp
    ${SRC_DIR}/wavefunction/kspace_jastrow.cpp
    ${SRC_DIR}/wavefunction/kspace_pair_sum.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_kspace_jastrow PRIVATE ${SRC_DIR})
target_link_libraries(test_kspace_jastrow PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_kspace_jastrow PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd -O2)

add_executable(test_structure_factor
    test_structure_factor.cpp
    test_bragg_order.cpp
    test_bond_order.cpp
    ${SRC_DIR}/estimators/structure_factor.cpp
    ${SRC_DIR}/estimators/pair_correlation.cpp
    ${SRC_DIR}/estimators/collective_density.cpp
    ${SRC_DIR}/estimators/density.cpp
    ${SRC_DIR}/estimators/electron_hole_proximity.cpp
    ${SRC_DIR}/estimators/bragg_order.cpp
    ${SRC_DIR}/estimators/bond_order.cpp
    ${SRC_DIR}/estimators/momentum_distribution.cpp
    ${SRC_DIR}/numerics/statistics.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_structure_factor PRIVATE ${SRC_DIR})
target_link_libraries(test_structure_factor PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_structure_factor PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_pair_radial
    test_pair_radial.cpp
    ${SRC_DIR}/numerics/pair_radial.cpp
)
target_include_directories(test_pair_radial PRIVATE ${SRC_DIR})
target_link_libraries(test_pair_radial PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_pair_radial PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd -O2)

add_executable(test_image_sum_jastrow
    test_image_sum_jastrow.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/wavefunction/pade_jastrow.cpp
    ${SRC_DIR}/wavefunction/drummond_jastrow.cpp
    ${SRC_DIR}/wavefunction/periodic_pair_sum.cpp
    ${SRC_DIR}/wavefunction/kspace_pair_sum.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
    ${SRC_DIR}/numerics/pair_radial.cpp
    ${SRC_DIR}/numerics/long_range_breakup.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_image_sum_jastrow PRIVATE ${SRC_DIR})
target_link_libraries(test_image_sum_jastrow PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_image_sum_jastrow PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd -O2)

add_executable(test_lpqhi_basis
    test_lpqhi_basis.cpp
)
target_include_directories(test_lpqhi_basis PRIVATE ${SRC_DIR})
target_link_libraries(test_lpqhi_basis PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_lpqhi_basis PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_bessel_j0
    test_bessel_j0.cpp
)
target_include_directories(test_bessel_j0 PRIVATE ${SRC_DIR})
target_link_libraries(test_bessel_j0 PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_bessel_j0 PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_rytova_keldysh
    test_rytova_keldysh.cpp
    ${SRC_DIR}/hamiltonian/rytova_keldysh.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_rytova_keldysh PRIVATE ${SRC_DIR})
target_link_libraries(test_rytova_keldysh PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_rytova_keldysh PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_screened_coulomb
    test_screened_coulomb.cpp
    ${SRC_DIR}/hamiltonian/screened_coulomb.cpp
    ${SRC_DIR}/hamiltonian/coulomb.cpp
    ${SRC_DIR}/hamiltonian/rytova_keldysh.cpp
    ${fsc_SOURCE}
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_screened_coulomb PRIVATE ${SRC_DIR} ${fsc_INCLUDE_DIR})
target_link_libraries(test_screened_coulomb PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_screened_coulomb PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_fsc_controlled_fit test_fsc_controlled_fit.cpp ${fsc_SOURCE})
target_include_directories(test_fsc_controlled_fit PRIVATE ${fsc_INCLUDE_DIR} ${eigen_INCLUDE_DIR})
target_link_libraries(test_fsc_controlled_fit PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_fsc_controlled_fit PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_kinetic
    test_kinetic.cpp
    ${SRC_DIR}/hamiltonian/kinetic.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_kinetic PRIVATE ${SRC_DIR})
target_link_libraries(test_kinetic PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_kinetic PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_moire
    test_moire.cpp
    ${SRC_DIR}/hamiltonian/moire.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_moire PRIVATE ${SRC_DIR})
target_link_libraries(test_moire PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_moire PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_moire_jastrow
    test_moire_jastrow.cpp
    ${SRC_DIR}/wavefunction/moire_jastrow.cpp
    ${SRC_DIR}/hamiltonian/moire.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_moire_jastrow PRIVATE ${SRC_DIR})
target_link_libraries(test_moire_jastrow PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_moire_jastrow PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_hamiltonian
    test_hamiltonian.cpp
    ${SRC_DIR}/hamiltonian/hamiltonian.cpp
    ${SRC_DIR}/hamiltonian/kinetic.cpp
    ${SRC_DIR}/hamiltonian/coulomb.cpp
    ${SRC_DIR}/hamiltonian/rytova_keldysh.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_hamiltonian PRIVATE ${SRC_DIR})
target_link_libraries(test_hamiltonian PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_hamiltonian PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_potential_gradient
    test_potential_gradient.cpp
    test_fitted_potential_gradient.cpp
    test_screened_potential_gradient.cpp
    test_fsc_fit_derivative.cpp
    test_potential_continuity.cpp
    test_screened_table_continuity.cpp
    ${SRC_DIR}/hamiltonian/hamiltonian.cpp
    ${SRC_DIR}/hamiltonian/kinetic.cpp
    ${SRC_DIR}/hamiltonian/coulomb.cpp
    ${SRC_DIR}/hamiltonian/harmonic_trap.cpp
    ${SRC_DIR}/hamiltonian/moire.cpp
    ${SRC_DIR}/hamiltonian/rytova_keldysh.cpp
    ${SRC_DIR}/hamiltonian/ewald_term.cpp
    ${SRC_DIR}/hamiltonian/screened_coulomb.cpp
    ${SRC_DIR}/numerics/ewald_breakup.cpp
    ${SRC_DIR}/numerics/long_range_breakup.cpp
    ${fsc_SOURCE}
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
    ${SRC_DIR}/pimc/path_set.cpp
)
target_include_directories(test_potential_gradient PRIVATE ${SRC_DIR} ${fsc_INCLUDE_DIR})
target_link_libraries(test_potential_gradient PRIVATE Catch2::Catch2WithMain OpenMP::OpenMP_CXX)
target_compile_options(test_potential_gradient PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_trial_wavefunction
    test_trial_wavefunction.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/wavefunction/pade_jastrow.cpp
    ${SRC_DIR}/wavefunction/periodic_pair_sum.cpp
    ${SRC_DIR}/wavefunction/kspace_pair_sum.cpp
    ${SRC_DIR}/numerics/pair_radial.cpp
    ${SRC_DIR}/numerics/long_range_breakup.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_trial_wavefunction PRIVATE ${SRC_DIR})
target_link_libraries(test_trial_wavefunction PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_trial_wavefunction PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_walker_ensemble
    test_walker_ensemble.cpp
    ${SRC_DIR}/population/walker_ensemble.cpp
    ${SRC_DIR}/population/walker.cpp
)
target_include_directories(test_walker_ensemble PRIVATE ${SRC_DIR})
target_link_libraries(test_walker_ensemble PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_walker_ensemble PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_population_control
    test_population_control.cpp
    ${SRC_DIR}/qmc/population_control.cpp
)
target_include_directories(test_population_control PRIVATE ${SRC_DIR})
target_link_libraries(test_population_control PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_population_control PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_statistics
    test_statistics.cpp
    ${SRC_DIR}/numerics/statistics.cpp
)
target_include_directories(test_statistics PRIVATE ${SRC_DIR})
target_link_libraries(test_statistics PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_statistics PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_xoshiro_random
    test_xoshiro_random.cpp
)
target_include_directories(test_xoshiro_random PRIVATE ${SRC_DIR})
target_link_libraries(test_xoshiro_random PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_xoshiro_random PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_drift
    test_drift.cpp
)
target_include_directories(test_drift PRIVATE ${SRC_DIR})
target_link_libraries(test_drift PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_drift PRIVATE -Wall -Wextra -Wpedantic)

# EhPairing is header-only but pulls in Eigen for the determinant branch.

add_executable(test_vmc
    test_vmc.cpp
    ${SRC_DIR}/qmc/vmc.cpp
    ${SRC_DIR}/numerics/statistics.cpp
    ${SRC_DIR}/io/logger.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/wavefunction/pade_jastrow.cpp
    ${SRC_DIR}/wavefunction/periodic_pair_sum.cpp
    ${SRC_DIR}/wavefunction/kspace_pair_sum.cpp
    ${SRC_DIR}/numerics/pair_radial.cpp
    ${SRC_DIR}/numerics/long_range_breakup.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
    ${SRC_DIR}/hamiltonian/hamiltonian.cpp
    ${SRC_DIR}/hamiltonian/kinetic.cpp
    ${SRC_DIR}/hamiltonian/coulomb.cpp
    ${SRC_DIR}/hamiltonian/rytova_keldysh.cpp
    ${SRC_DIR}/population/walker_ensemble.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_vmc PRIVATE ${SRC_DIR})
target_link_libraries(test_vmc PRIVATE Catch2::Catch2WithMain OpenMP::OpenMP_CXX)
target_compile_options(test_vmc PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_parser
    test_parser.cpp
    ${SRC_DIR}/io/parser.cpp
)
target_include_directories(test_parser PRIVATE ${SRC_DIR})
target_link_libraries(test_parser PRIVATE Catch2::Catch2WithMain yaml-cpp::yaml-cpp)
target_compile_options(test_parser PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(test_parser PRIVATE
    ALEA_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../configs")

add_executable(test_logger
    test_logger.cpp
    ${SRC_DIR}/io/logger.cpp
)
target_include_directories(test_logger PRIVATE ${SRC_DIR})
target_link_libraries(test_logger PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_logger PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_walker_io
    test_walker_io.cpp
    ${SRC_DIR}/io/walker_io.cpp
    ${SRC_DIR}/io/logger.cpp
)
target_include_directories(test_walker_io PRIVATE ${SRC_DIR} ${highfive_INCLUDE_DIR})
target_link_libraries(test_walker_io PRIVATE Catch2::Catch2WithMain hdf5-static)
target_compile_options(test_walker_io PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_results_writer
    test_results_writer.cpp
    ${SRC_DIR}/io/results_writer.cpp
    ${SRC_DIR}/io/logger.cpp
)
target_include_directories(test_results_writer PRIVATE ${SRC_DIR} ${highfive_INCLUDE_DIR})
target_link_libraries(test_results_writer PRIVATE Catch2::Catch2WithMain hdf5-static)
target_compile_options(test_results_writer PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_config_writer
    test_config_writer.cpp
    ${SRC_DIR}/io/config_writer.cpp
    ${SRC_DIR}/io/parser.cpp
    ${SRC_DIR}/io/logger.cpp
)
target_include_directories(test_config_writer PRIVATE ${SRC_DIR})
target_link_libraries(test_config_writer PRIVATE Catch2::Catch2WithMain yaml-cpp::yaml-cpp)
target_compile_options(test_config_writer PRIVATE -Wall -Wextra -Wpedantic)

# Everything App::build_drivers reaches: shared by every target that links
# app.cpp, so a new builder source is added in one place.
set(APP_SOURCES
    ${SRC_DIR}/builders/center_of_mass_momentum.cpp
    ${SRC_DIR}/builders/common.cpp
    ${SRC_DIR}/builders/coulomb.cpp
    ${SRC_DIR}/builders/inverse_power.cpp
    ${SRC_DIR}/builders/ewald.cpp
    ${SRC_DIR}/builders/exchange_classes.cpp
    ${SRC_DIR}/builders/kinetic.cpp
    ${SRC_DIR}/builders/moire_potential.cpp
    ${SRC_DIR}/builders/rytova_keldysh.cpp
    ${SRC_DIR}/builders/screened_coulomb.cpp
    ${SRC_DIR}/builders/jastrow.cpp
    ${SRC_DIR}/builders/kspace_jastrow.cpp
    ${SRC_DIR}/wavefunction/kspace_jastrow.cpp
    ${SRC_DIR}/builders/moire_jastrow.cpp
    ${SRC_DIR}/builders/nosanow.cpp
    ${SRC_DIR}/builders/pairing.cpp
    ${SRC_DIR}/builders/slater.cpp
    ${SRC_DIR}/wavefunction/pair_orbital.cpp
    ${SRC_DIR}/wavefunction/pairing_determinant.cpp
    ${SRC_DIR}/qmc/app.cpp
    ${SRC_DIR}/wavefunction/moire_jastrow.cpp
    ${SRC_DIR}/wavefunction/center_of_mass_momentum.cpp
    ${SRC_DIR}/hamiltonian/harmonic_trap.cpp
    ${SRC_DIR}/hamiltonian/inverse_power.cpp
    ${SRC_DIR}/wavefunction/complex_slater_determinant.cpp
    ${SRC_DIR}/hamiltonian/ewald_term.cpp
    ${SRC_DIR}/numerics/ewald_breakup.cpp
    ${SRC_DIR}/numerics/long_range_breakup.cpp
    ${SRC_DIR}/wavefunction/slater_determinant.cpp
    ${SRC_DIR}/hamiltonian/screened_coulomb.cpp
    ${fsc_SOURCE}
    ${SRC_DIR}/hamiltonian/moire.cpp
    ${SRC_DIR}/wavefunction/drummond_jastrow.cpp
    ${SRC_DIR}/wavefunction/zero_energy_jastrow.cpp
    ${SRC_DIR}/wavefunction/nosanow_jastrow.cpp
    ${SRC_DIR}/estimators/pair_correlation.cpp
    ${SRC_DIR}/estimators/collective_density.cpp
    ${SRC_DIR}/estimators/density.cpp
    ${SRC_DIR}/estimators/electron_hole_proximity.cpp
    ${SRC_DIR}/estimators/momentum_distribution.cpp
    ${SRC_DIR}/builders/estimators.cpp
    ${SRC_DIR}/estimators/density_matrix.cpp
    ${SRC_DIR}/estimators/structure_factor.cpp
    ${SRC_DIR}/estimators/bragg_order.cpp
    ${SRC_DIR}/qmc/cost_function.cpp
    ${SRC_DIR}/estimators/bond_order.cpp
    ${SRC_DIR}/qmc/dmc.cpp
    ${SRC_DIR}/qmc/population_control.cpp
    ${SRC_DIR}/pimc/path_set.cpp
    ${SRC_DIR}/pimc/action.cpp
    ${SRC_DIR}/pimc/fixed_node_restriction.cpp
    ${SRC_DIR}/pimc/pair_product_action.cpp
    ${SRC_DIR}/numerics/finite_grid_pair.cpp
    ${SRC_DIR}/pimc/move.cpp
    ${SRC_DIR}/pimc/pimc.cpp
    ${SRC_DIR}/qmc/linear_method.cpp
    ${SRC_DIR}/qmc/linear_optimize.cpp
    ${SRC_DIR}/qmc/vmc.cpp
    ${SRC_DIR}/numerics/statistics.cpp
    ${SRC_DIR}/io/parser.cpp
    ${SRC_DIR}/io/logger.cpp
    ${SRC_DIR}/io/path_io.cpp
    ${SRC_DIR}/io/walker_io.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/wavefunction/pade_jastrow.cpp
    ${SRC_DIR}/wavefunction/periodic_pair_sum.cpp
    ${SRC_DIR}/wavefunction/kspace_pair_sum.cpp
    ${SRC_DIR}/numerics/pair_radial.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
    ${SRC_DIR}/hamiltonian/hamiltonian.cpp
    ${SRC_DIR}/hamiltonian/kinetic.cpp
    ${SRC_DIR}/hamiltonian/coulomb.cpp
    ${SRC_DIR}/hamiltonian/rytova_keldysh.cpp
    ${SRC_DIR}/population/walker_ensemble.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)

add_executable(test_app
    test_app.cpp
    ${APP_SOURCES}
)
target_include_directories(test_app PRIVATE ${SRC_DIR} ${eigen_INCLUDE_DIR} ${fsc_INCLUDE_DIR} ${highfive_INCLUDE_DIR})
target_link_libraries(test_app PRIVATE Catch2::Catch2WithMain yaml-cpp::yaml-cpp OpenMP::OpenMP_CXX hdf5-static)
target_compile_options(test_app PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)
target_compile_definitions(test_app PRIVATE
    ALEA_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../configs")

add_executable(test_build_characterization
    test_build_characterization.cpp
    ${APP_SOURCES}
)
target_include_directories(test_build_characterization PRIVATE ${SRC_DIR} ${eigen_INCLUDE_DIR} ${fsc_INCLUDE_DIR} ${highfive_INCLUDE_DIR})
target_link_libraries(test_build_characterization PRIVATE Catch2::Catch2WithMain yaml-cpp::yaml-cpp OpenMP::OpenMP_CXX hdf5-static)
target_compile_options(test_build_characterization PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd -O2)
target_compile_definitions(test_build_characterization PRIVATE
    ALEA_REPO_DIR="${CMAKE_CURRENT_SOURCE_DIR}/..")

add_executable(test_vmc_physics
    test_vmc_physics.cpp
    ${APP_SOURCES}
)
target_include_directories(test_vmc_physics PRIVATE ${SRC_DIR} ${eigen_INCLUDE_DIR} ${fsc_INCLUDE_DIR} ${highfive_INCLUDE_DIR})
target_link_libraries(test_vmc_physics PRIVATE Catch2::Catch2WithMain yaml-cpp::yaml-cpp OpenMP::OpenMP_CXX hdf5-static)
# This test runs full VMC simulations (millions of sweeps); -O2 keeps the suite
# fast (seconds, not minutes) since the test build is otherwise unoptimised.
target_compile_options(test_vmc_physics PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd -O2)
target_compile_definitions(test_vmc_physics PRIVATE
    ALEA_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/configs")

add_executable(test_linear_optimize
    test_linear_optimize.cpp
    ${SRC_DIR}/qmc/linear_optimize.cpp
    ${SRC_DIR}/qmc/cost_function.cpp
    ${SRC_DIR}/qmc/linear_method.cpp
    ${SRC_DIR}/qmc/vmc.cpp
    ${SRC_DIR}/numerics/statistics.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/wavefunction/slater_determinant.cpp
    ${SRC_DIR}/wavefunction/pade_jastrow.cpp
    ${SRC_DIR}/wavefunction/periodic_pair_sum.cpp
    ${SRC_DIR}/wavefunction/kspace_pair_sum.cpp
    ${SRC_DIR}/numerics/pair_radial.cpp
    ${SRC_DIR}/numerics/long_range_breakup.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
    ${SRC_DIR}/hamiltonian/hamiltonian.cpp
    ${SRC_DIR}/hamiltonian/kinetic.cpp
    ${SRC_DIR}/hamiltonian/coulomb.cpp
    ${SRC_DIR}/hamiltonian/rytova_keldysh.cpp
    ${SRC_DIR}/population/walker_ensemble.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
    ${SRC_DIR}/io/logger.cpp
)
target_include_directories(test_linear_optimize PRIVATE ${SRC_DIR} ${eigen_INCLUDE_DIR})
target_link_libraries(test_linear_optimize PRIVATE Catch2::Catch2WithMain OpenMP::OpenMP_CXX)
target_compile_options(test_linear_optimize PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_dmc
    test_dmc.cpp
    ${SRC_DIR}/qmc/dmc.cpp
    ${SRC_DIR}/qmc/population_control.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/wavefunction/pade_jastrow.cpp
    ${SRC_DIR}/wavefunction/periodic_pair_sum.cpp
    ${SRC_DIR}/wavefunction/kspace_pair_sum.cpp
    ${SRC_DIR}/numerics/pair_radial.cpp
    ${SRC_DIR}/numerics/long_range_breakup.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
    ${SRC_DIR}/hamiltonian/hamiltonian.cpp
    ${SRC_DIR}/hamiltonian/kinetic.cpp
    ${SRC_DIR}/hamiltonian/coulomb.cpp
    ${SRC_DIR}/hamiltonian/rytova_keldysh.cpp
    ${SRC_DIR}/population/walker_ensemble.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
    ${SRC_DIR}/numerics/statistics.cpp
    ${SRC_DIR}/io/logger.cpp
)
target_include_directories(test_dmc PRIVATE ${SRC_DIR})
target_link_libraries(test_dmc PRIVATE Catch2::Catch2WithMain OpenMP::OpenMP_CXX)
target_compile_options(test_dmc PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_fixed_node
    test_fixed_node.cpp
    ${SRC_DIR}/qmc/dmc.cpp
    ${SRC_DIR}/qmc/population_control.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/wavefunction/pade_jastrow.cpp
    ${SRC_DIR}/wavefunction/periodic_pair_sum.cpp
    ${SRC_DIR}/wavefunction/kspace_pair_sum.cpp
    ${SRC_DIR}/numerics/pair_radial.cpp
    ${SRC_DIR}/numerics/long_range_breakup.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
    ${SRC_DIR}/hamiltonian/hamiltonian.cpp
    ${SRC_DIR}/hamiltonian/kinetic.cpp
    ${SRC_DIR}/hamiltonian/coulomb.cpp
    ${SRC_DIR}/hamiltonian/rytova_keldysh.cpp
    ${SRC_DIR}/population/walker_ensemble.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
    ${SRC_DIR}/numerics/statistics.cpp
    ${SRC_DIR}/io/logger.cpp
)
target_include_directories(test_fixed_node PRIVATE ${SRC_DIR})
target_link_libraries(test_fixed_node PRIVATE Catch2::Catch2WithMain OpenMP::OpenMP_CXX)
target_compile_options(test_fixed_node PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_pairing_determinant
    test_pairing_determinant.cpp
    ${SRC_DIR}/wavefunction/pair_orbital.cpp
    ${SRC_DIR}/wavefunction/pairing_determinant.cpp
    ${SRC_DIR}/wavefunction/slater_determinant.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_pairing_determinant PRIVATE ${SRC_DIR} ${eigen_INCLUDE_DIR})
target_link_libraries(test_pairing_determinant PRIVATE Catch2::Catch2WithMain OpenMP::OpenMP_CXX)
target_compile_options(test_pairing_determinant PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_slater_determinant
    test_slater_determinant.cpp
    test_slater_sequence.cpp
    ${SRC_DIR}/wavefunction/slater_determinant.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
    ${SRC_DIR}/qmc/dmc.cpp
    ${SRC_DIR}/qmc/population_control.cpp
    ${SRC_DIR}/hamiltonian/hamiltonian.cpp
    ${SRC_DIR}/hamiltonian/kinetic.cpp
    ${SRC_DIR}/population/walker_ensemble.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
    ${SRC_DIR}/numerics/statistics.cpp
    ${SRC_DIR}/io/logger.cpp
)
target_include_directories(test_slater_determinant PRIVATE ${SRC_DIR} ${eigen_INCLUDE_DIR})
target_link_libraries(test_slater_determinant PRIVATE Catch2::Catch2WithMain OpenMP::OpenMP_CXX)
# Cover the same SIMD/FMA arithmetic as production on the Release host.
target_compile_options(test_slater_determinant PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd
    $<$<CONFIG:Release>:-march=native> $<$<CONFIG:Release>:-fno-math-errno>)

add_executable(test_sweep_blocking
    test_sweep_blocking.cpp
    ${SRC_DIR}/qmc/vmc.cpp
    ${SRC_DIR}/qmc/dmc.cpp
    ${SRC_DIR}/qmc/population_control.cpp
    ${SRC_DIR}/wavefunction/slater_determinant.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/wavefunction/pade_jastrow.cpp
    ${SRC_DIR}/wavefunction/periodic_pair_sum.cpp
    ${SRC_DIR}/wavefunction/kspace_pair_sum.cpp
    ${SRC_DIR}/numerics/pair_radial.cpp
    ${SRC_DIR}/numerics/long_range_breakup.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
    ${SRC_DIR}/hamiltonian/hamiltonian.cpp
    ${SRC_DIR}/hamiltonian/kinetic.cpp
    ${SRC_DIR}/population/walker_ensemble.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
    ${SRC_DIR}/numerics/statistics.cpp
    ${SRC_DIR}/io/logger.cpp
)
target_include_directories(test_sweep_blocking PRIVATE ${SRC_DIR} ${eigen_INCLUDE_DIR})
target_link_libraries(test_sweep_blocking PRIVATE Catch2::Catch2WithMain OpenMP::OpenMP_CXX)
# Production SIMD/FMA arithmetic: block boundaries move walkers between vector
# bodies and scalar remainders, which is where a bitwise difference could hide.
target_compile_options(test_sweep_blocking PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd
    $<$<CONFIG:Release>:-march=native> $<$<CONFIG:Release>:-fno-math-errno>)

add_executable(test_driver
    test_driver.cpp
    ${APP_SOURCES}
)
target_include_directories(test_driver PRIVATE ${SRC_DIR} ${eigen_INCLUDE_DIR} ${fsc_INCLUDE_DIR} ${highfive_INCLUDE_DIR})
target_link_libraries(test_driver PRIVATE Catch2::Catch2WithMain yaml-cpp::yaml-cpp OpenMP::OpenMP_CXX hdf5-static)
target_compile_options(test_driver PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_dmc_bench
    test_dmc_bench.cpp
    ${SRC_DIR}/hamiltonian/ewald_term.cpp
    ${SRC_DIR}/numerics/ewald_breakup.cpp
    ${SRC_DIR}/numerics/long_range_breakup.cpp
    ${SRC_DIR}/qmc/vmc.cpp
    ${SRC_DIR}/qmc/dmc.cpp
    ${SRC_DIR}/qmc/population_control.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/wavefunction/pade_jastrow.cpp
    ${SRC_DIR}/wavefunction/periodic_pair_sum.cpp
    ${SRC_DIR}/wavefunction/kspace_pair_sum.cpp
    ${SRC_DIR}/numerics/pair_radial.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
    ${SRC_DIR}/hamiltonian/hamiltonian.cpp
    ${SRC_DIR}/hamiltonian/kinetic.cpp
    ${SRC_DIR}/hamiltonian/coulomb.cpp
    ${SRC_DIR}/hamiltonian/rytova_keldysh.cpp
    ${SRC_DIR}/population/walker_ensemble.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
    ${SRC_DIR}/numerics/statistics.cpp
    ${SRC_DIR}/io/logger.cpp
)
target_include_directories(test_dmc_bench PRIVATE ${SRC_DIR})
target_link_libraries(test_dmc_bench PRIVATE OpenMP::OpenMP_CXX)
# Production flags (CMakeLists.txt:158 uses -O3 -march=native for Release).
# A benchmark compiled at -O2 measures a binary nobody runs.
target_compile_options(test_dmc_bench PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd -fno-math-errno -O3 -march=native)

add_executable(test_cost_function
    test_cost_function.cpp
    ${SRC_DIR}/qmc/cost_function.cpp
    ${SRC_DIR}/qmc/linear_method.cpp
)
target_include_directories(test_cost_function PRIVATE ${SRC_DIR} ${eigen_INCLUDE_DIR})
target_link_libraries(test_cost_function PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_cost_function PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_linear_method
    test_linear_method.cpp
    ${SRC_DIR}/qmc/linear_method.cpp
)
target_include_directories(test_linear_method PRIVATE ${SRC_DIR} ${eigen_INCLUDE_DIR})
target_link_libraries(test_linear_method PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_linear_method PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_pair_correlation
    test_pair_correlation.cpp
    ${SRC_DIR}/estimators/pair_correlation.cpp
    ${SRC_DIR}/estimators/collective_density.cpp
    ${SRC_DIR}/estimators/density.cpp
    ${SRC_DIR}/estimators/electron_hole_proximity.cpp
    ${SRC_DIR}/estimators/momentum_distribution.cpp
    ${SRC_DIR}/estimators/structure_factor.cpp
    ${SRC_DIR}/numerics/statistics.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_pair_correlation PRIVATE ${SRC_DIR})
target_link_libraries(test_pair_correlation PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_pair_correlation PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_density
    test_density.cpp
    ${SRC_DIR}/estimators/pair_correlation.cpp
    ${SRC_DIR}/estimators/collective_density.cpp
    ${SRC_DIR}/estimators/density.cpp
    ${SRC_DIR}/estimators/electron_hole_proximity.cpp
    ${SRC_DIR}/estimators/momentum_distribution.cpp
    ${SRC_DIR}/estimators/structure_factor.cpp
    ${SRC_DIR}/numerics/statistics.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_density PRIVATE ${SRC_DIR})
target_link_libraries(test_density PRIVATE Catch2::Catch2WithMain)
# Constant-density roundoff depends on the production FMA contraction path.
target_compile_options(test_density PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd
    $<$<CONFIG:Release>:-march=native> $<$<CONFIG:Release>:-fno-math-errno>)

add_executable(test_estimator_builders
    test_estimator_builders.cpp
    ${SRC_DIR}/builders/estimators.cpp
    ${SRC_DIR}/estimators/density_matrix.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
    ${SRC_DIR}/estimators/bragg_order.cpp
    ${SRC_DIR}/estimators/bond_order.cpp
    ${SRC_DIR}/estimators/pair_correlation.cpp
    ${SRC_DIR}/estimators/collective_density.cpp
    ${SRC_DIR}/estimators/density.cpp
    ${SRC_DIR}/estimators/electron_hole_proximity.cpp
    ${SRC_DIR}/estimators/momentum_distribution.cpp
    ${SRC_DIR}/estimators/structure_factor.cpp
    ${SRC_DIR}/io/logger.cpp
    ${SRC_DIR}/numerics/statistics.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_estimator_builders PRIVATE ${SRC_DIR})
target_link_libraries(test_estimator_builders PRIVATE Catch2::Catch2WithMain OpenMP::OpenMP_CXX)
target_compile_options(test_estimator_builders PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_density_matrix
    test_density_matrix.cpp
    ${SRC_DIR}/estimators/density_matrix.cpp
    ${SRC_DIR}/wavefunction/drummond_jastrow.cpp
    ${SRC_DIR}/wavefunction/pair_orbital.cpp
    ${SRC_DIR}/wavefunction/pairing_determinant.cpp
    ${SRC_DIR}/wavefunction/slater_determinant.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
    ${SRC_DIR}/numerics/statistics.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_density_matrix PRIVATE ${SRC_DIR} ${eigen_INCLUDE_DIR})
target_link_libraries(test_density_matrix PRIVATE Catch2::Catch2WithMain OpenMP::OpenMP_CXX)
target_compile_options(test_density_matrix PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_collective_density
    test_collective_density.cpp
    ${SRC_DIR}/estimators/pair_correlation.cpp
    ${SRC_DIR}/estimators/collective_density.cpp
    ${SRC_DIR}/estimators/density.cpp
    ${SRC_DIR}/estimators/electron_hole_proximity.cpp
    ${SRC_DIR}/estimators/momentum_distribution.cpp
    ${SRC_DIR}/estimators/structure_factor.cpp
    ${SRC_DIR}/numerics/statistics.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_collective_density PRIVATE ${SRC_DIR})
target_link_libraries(test_collective_density PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_collective_density PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_fixed_phase
    test_fixed_phase.cpp
    ${SRC_DIR}/wavefunction/center_of_mass_momentum.cpp
    ${SRC_DIR}/hamiltonian/harmonic_trap.cpp
    ${SRC_DIR}/wavefunction/complex_slater_determinant.cpp
    ${SRC_DIR}/wavefunction/pade_jastrow.cpp
    ${SRC_DIR}/wavefunction/periodic_pair_sum.cpp
    ${SRC_DIR}/wavefunction/kspace_pair_sum.cpp
    ${SRC_DIR}/numerics/pair_radial.cpp
    ${SRC_DIR}/numerics/long_range_breakup.cpp
    ${SRC_DIR}/wavefunction/slater_determinant.cpp
    ${SRC_DIR}/hamiltonian/hamiltonian.cpp
    ${SRC_DIR}/hamiltonian/kinetic.cpp
    ${SRC_DIR}/hamiltonian/coulomb.cpp
    ${SRC_DIR}/hamiltonian/rytova_keldysh.cpp
    ${SRC_DIR}/qmc/vmc.cpp
    ${SRC_DIR}/qmc/dmc.cpp
    ${SRC_DIR}/qmc/population_control.cpp
    ${SRC_DIR}/numerics/statistics.cpp
    ${SRC_DIR}/io/logger.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
    ${SRC_DIR}/population/walker_ensemble.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_fixed_phase PRIVATE ${SRC_DIR} ${eigen_INCLUDE_DIR})
target_link_libraries(test_fixed_phase PRIVATE Catch2::Catch2WithMain OpenMP::OpenMP_CXX)
target_compile_options(test_fixed_phase PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_twisted_boundary
    test_twisted_boundary.cpp
    ${SRC_DIR}/wavefunction/complex_slater_determinant.cpp
    ${SRC_DIR}/hamiltonian/hamiltonian.cpp
    ${SRC_DIR}/hamiltonian/kinetic.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_twisted_boundary PRIVATE ${SRC_DIR} ${eigen_INCLUDE_DIR})
target_link_libraries(test_twisted_boundary PRIVATE Catch2::Catch2WithMain OpenMP::OpenMP_CXX)
target_compile_options(test_twisted_boundary PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

add_executable(test_gauge_field
    test_gauge_field.cpp
    ${SRC_DIR}/hamiltonian/hamiltonian.cpp
    ${SRC_DIR}/hamiltonian/kinetic.cpp
    ${SRC_DIR}/hamiltonian/harmonic_trap.cpp
    ${SRC_DIR}/wavefunction/complex_slater_determinant.cpp
    ${SRC_DIR}/wavefunction/slater_determinant.cpp
    ${SRC_DIR}/wavefunction/trial_wavefunction.cpp
    ${SRC_DIR}/wavefunction/wavefunction_component.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
)
target_include_directories(test_gauge_field PRIVATE ${SRC_DIR} ${eigen_INCLUDE_DIR})
target_link_libraries(test_gauge_field PRIVATE Catch2::Catch2WithMain OpenMP::OpenMP_CXX)
target_compile_options(test_gauge_field PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

# ── PIMC (path-integral Monte Carlo) ─────────────────────────────────────────
# Unit tests for PathSet / Action / Moves + the driver gates and YAML parsing.
add_executable(test_pimc
    test_pimc.cpp
    ${SRC_DIR}/pimc/path_set.cpp
    ${SRC_DIR}/pimc/action.cpp
    ${SRC_DIR}/pimc/fixed_node_restriction.cpp
    ${SRC_DIR}/pimc/pair_product_action.cpp
    ${SRC_DIR}/numerics/finite_grid_pair.cpp
    ${SRC_DIR}/pimc/move.cpp
    ${SRC_DIR}/pimc/pimc.cpp
    ${SRC_DIR}/hamiltonian/hamiltonian.cpp
    ${SRC_DIR}/hamiltonian/harmonic_trap.cpp
    ${SRC_DIR}/hamiltonian/coulomb.cpp
    ${SRC_DIR}/population/particle_batch.cpp
    ${SRC_DIR}/population/walker_set.cpp
    ${SRC_DIR}/population/walker_ensemble.cpp
    ${SRC_DIR}/population/walker.cpp
    ${SRC_DIR}/population/batched_distance_table.cpp
    ${SRC_DIR}/population/simulation_cell.cpp
    ${SRC_DIR}/numerics/statistics.cpp
    ${SRC_DIR}/io/parser.cpp
    ${SRC_DIR}/io/logger.cpp
)
target_sources(test_pimc PRIVATE ${SRC_DIR}/io/path_io.cpp)
target_include_directories(test_pimc PRIVATE ${SRC_DIR} ${eigen_INCLUDE_DIR} ${highfive_INCLUDE_DIR})
target_link_libraries(test_pimc PRIVATE Catch2::Catch2WithMain yaml-cpp::yaml-cpp OpenMP::OpenMP_CXX hdf5-static)
target_compile_options(test_pimc PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

# Independent contracts share the unit source set, not the production App.
get_target_property(PIMC_UNIT_SOURCES test_pimc SOURCES)
list(REMOVE_ITEM PIMC_UNIT_SOURCES test_pimc.cpp)
foreach(contract action contract incremental_moves exchange permutation joint worm validation_fast)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_pimc_${contract}.cpp")
        continue()
    endif()
    add_executable(test_pimc_${contract} test_pimc_${contract}.cpp ${PIMC_UNIT_SOURCES})
    target_include_directories(test_pimc_${contract} PRIVATE ${SRC_DIR} ${eigen_INCLUDE_DIR} ${highfive_INCLUDE_DIR})
    target_link_libraries(test_pimc_${contract} PRIVATE Catch2::Catch2WithMain yaml-cpp::yaml-cpp OpenMP::OpenMP_CXX hdf5-static)
    target_compile_options(test_pimc_${contract} PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)
endforeach()
# The exchange estimator tests measure a signed one-body density.
if(TARGET test_pimc_worm)
    target_sources(test_pimc_worm PRIVATE ${SRC_DIR}/estimators/momentum_distribution.cpp)
endif()
if(TARGET test_pimc_permutation)
    target_sources(test_pimc_permutation PRIVATE ${SRC_DIR}/estimators/density.cpp)
endif()
if(TARGET test_pimc_action)
    target_sources(test_pimc_action PRIVATE
        ${SRC_DIR}/hamiltonian/rytova_keldysh.cpp
        ${SRC_DIR}/hamiltonian/moire.cpp
        ${SRC_DIR}/hamiltonian/screened_coulomb.cpp ${fsc_SOURCE})
    target_sources(test_pimc_action PRIVATE ${SRC_DIR}/io/path_io.cpp)
    target_include_directories(test_pimc_action PRIVATE ${fsc_INCLUDE_DIR} ${highfive_INCLUDE_DIR})
    target_link_libraries(test_pimc_action PRIVATE hdf5-static)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_pimc_estimators.cpp")
add_executable(test_pimc_estimators test_pimc_estimators.cpp ${APP_SOURCES})
target_include_directories(test_pimc_estimators PRIVATE ${SRC_DIR} ${eigen_INCLUDE_DIR} ${fsc_INCLUDE_DIR} ${highfive_INCLUDE_DIR})
target_link_libraries(test_pimc_estimators PRIVATE Catch2::Catch2WithMain yaml-cpp::yaml-cpp OpenMP::OpenMP_CXX hdf5-static)
target_compile_options(test_pimc_estimators PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd -O2)
endif()

# PIMC physics: full YAML -> App -> Pimc run against analytic results.
add_executable(test_pimc_physics
    test_pimc_physics.cpp
    test_pimc_many_body.cpp
    test_pimc_centroid_virial.cpp
    test_pimc_periodic_virial.cpp
    test_pimc_rk_admission.cpp
    test_pimc_fsc_diagnostics.cpp
    test_pimc_kernel_admission.cpp
    test_pimc_pair_product.cpp
    test_pimc_fixed_node.cpp
    ${SRC_DIR}/io/config_writer.cpp
    ${APP_SOURCES}
)
target_include_directories(test_pimc_physics PRIVATE ${SRC_DIR} ${eigen_INCLUDE_DIR} ${fsc_INCLUDE_DIR} ${highfive_INCLUDE_DIR})
target_link_libraries(test_pimc_physics PRIVATE Catch2::Catch2WithMain yaml-cpp::yaml-cpp OpenMP::OpenMP_CXX hdf5-static)
target_compile_options(test_pimc_physics PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd -O2)
target_compile_definitions(test_pimc_physics PRIVATE
    ALEA_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/configs")

add_executable(test_pimc_checkpoint test_pimc_checkpoint.cpp ${PIMC_UNIT_SOURCES} ${SRC_DIR}/io/path_io.cpp)
target_include_directories(test_pimc_checkpoint PRIVATE ${SRC_DIR} ${eigen_INCLUDE_DIR} ${highfive_INCLUDE_DIR})
target_link_libraries(test_pimc_checkpoint PRIVATE Catch2::Catch2WithMain yaml-cpp::yaml-cpp OpenMP::OpenMP_CXX hdf5-static)
target_compile_options(test_pimc_checkpoint PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_path_io.cpp")
    add_executable(test_path_io test_path_io.cpp ${SRC_DIR}/population/simulation_cell.cpp)
    if(EXISTS "${SRC_DIR}/io/path_io.cpp")
        target_sources(test_path_io PRIVATE ${SRC_DIR}/io/path_io.cpp)
    endif()
    target_include_directories(test_path_io PRIVATE ${SRC_DIR} ${highfive_INCLUDE_DIR})
    target_link_libraries(test_path_io PRIVATE Catch2::Catch2WithMain hdf5-static)
    target_compile_options(test_path_io PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Small standalone contract; the production translation unit is absent during
# the initial runtime RED and included automatically when implemented.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_electron_hole_proximity.cpp")
    add_executable(test_electron_hole_proximity
        test_electron_hole_proximity.cpp
        ${SRC_DIR}/numerics/statistics.cpp
        ${SRC_DIR}/pimc/path_set.cpp
        ${SRC_DIR}/population/particle_batch.cpp
        ${SRC_DIR}/population/batched_distance_table.cpp
        ${SRC_DIR}/population/simulation_cell.cpp)
    if(EXISTS "${SRC_DIR}/estimators/electron_hole_proximity.cpp")
        target_sources(test_electron_hole_proximity PRIVATE
            ${SRC_DIR}/estimators/electron_hole_proximity.cpp)
    endif()
    target_include_directories(test_electron_hole_proximity PRIVATE ${SRC_DIR})
    target_link_libraries(test_electron_hole_proximity PRIVATE Catch2::Catch2WithMain OpenMP::OpenMP_CXX)
    target_compile_options(test_electron_hole_proximity PRIVATE -Wall -Wextra -Wpedantic -fopenmp-simd)
endif()

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
add_executable(test_finite_grid_pair test_finite_grid_pair.cpp ${SRC_DIR}/numerics/finite_grid_pair.cpp)
target_include_directories(test_finite_grid_pair PRIVATE ${SRC_DIR} ${eigen_INCLUDE_DIR})
target_link_libraries(test_finite_grid_pair PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_finite_grid_pair PRIVATE -Wall -Wextra -Wpedantic -O2)
catch_discover_tests(test_finite_grid_pair)
catch_discover_tests(test_inverse_power)
catch_discover_tests(test_zero_energy_jastrow)
catch_discover_tests(test_nosanow_jastrow)
catch_discover_tests(test_long_range_breakup)
# test_ewald was built but never registered, so ctest skipped its 23 cases
# (the Ewald breakup, Madelung, bilayer and single-particle delta contracts).
catch_discover_tests(test_ewald)
catch_discover_tests(test_pair_radial)
catch_discover_tests(test_structure_factor)
catch_discover_tests(test_estimator_builders)
catch_discover_tests(test_density_matrix)
catch_discover_tests(test_kspace_jastrow)
catch_discover_tests(test_image_sum_jastrow)
catch_discover_tests(test_lpqhi_basis)
catch_discover_tests(test_bessel_j0)
catch_discover_tests(test_driver)
catch_discover_tests(test_linear_optimize)
catch_discover_tests(test_cost_function)
catch_discover_tests(test_linear_method)
catch_discover_tests(test_batched_distance_table)
catch_discover_tests(test_walker)
catch_discover_tests(test_walker_set)
catch_discover_tests(test_pade_jastrow)
catch_discover_tests(test_drummond_jastrow)
catch_discover_tests(test_coulomb)
catch_discover_tests(test_rytova_keldysh)
catch_discover_tests(test_screened_coulomb)
catch_discover_tests(test_fsc_controlled_fit)
catch_discover_tests(test_kinetic)
catch_discover_tests(test_moire)
catch_discover_tests(test_moire_jastrow)
catch_discover_tests(test_hamiltonian)
catch_discover_tests(test_potential_gradient)
catch_discover_tests(test_trial_wavefunction)
catch_discover_tests(test_walker_ensemble)
catch_discover_tests(test_population_control)
catch_discover_tests(test_statistics)
catch_discover_tests(test_xoshiro_random)
catch_discover_tests(test_drift)
catch_discover_tests(test_vmc)
catch_discover_tests(test_dmc)
catch_discover_tests(test_fixed_node)
catch_discover_tests(test_pairing_determinant)
catch_discover_tests(test_slater_determinant)
catch_discover_tests(test_sweep_blocking)
catch_discover_tests(test_parser)
catch_discover_tests(test_logger)
catch_discover_tests(test_walker_io)
catch_discover_tests(test_results_writer)
catch_discover_tests(test_config_writer)
catch_discover_tests(test_app)
catch_discover_tests(test_build_characterization)
catch_discover_tests(test_vmc_physics)
catch_discover_tests(test_pair_correlation)
catch_discover_tests(test_density)
catch_discover_tests(test_collective_density)
catch_discover_tests(test_fixed_phase)
catch_discover_tests(test_twisted_boundary)
catch_discover_tests(test_gauge_field)
catch_discover_tests(test_pimc)
catch_discover_tests(test_pimc_physics)
catch_discover_tests(test_pimc_contract)
catch_discover_tests(test_pimc_incremental_moves)
catch_discover_tests(test_pimc_exchange)
catch_discover_tests(test_pimc_permutation)
catch_discover_tests(test_pimc_joint)
catch_discover_tests(test_pimc_worm)
catch_discover_tests(test_pimc_validation_fast)
catch_discover_tests(test_pimc_checkpoint)
if(TARGET test_electron_hole_proximity)
    catch_discover_tests(test_electron_hole_proximity)
endif()
if(TARGET test_path_io)
    catch_discover_tests(test_path_io)
endif()
if(TARGET test_pimc_action)
    catch_discover_tests(test_pimc_action)
endif()
if(TARGET test_pimc_estimators)
    catch_discover_tests(test_pimc_estimators)
endif()
