# SPDX-FileCopyrightText: 2026 The Alea Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Pin + additive patch until the fitted-derivative API is released upstream.
# Explicit source overrides are owned by their caller and must provide API v1.
function(alea_require_fsc_fit_derivative source_dir source_override)
    file(READ "${source_dir}/src/fast_screened_coulomb.h" fsc_header)
    if(fsc_header MATCHES "#define FSC_FIT_DERIVATIVE_API 1([\r\n]|$)")
        return()
    endif()
    if(source_override)
        message(FATAL_ERROR
            "FSC source override lacks FSC_FIT_DERIVATIVE_API 1. Update the library "
            "or apply cmake/patches/fsc_fit_derivative.patch to the pinned baseline.")
    endif()
    find_package(Git REQUIRED)
    set(fsc_patch "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patches/fsc_fit_derivative.patch")
    execute_process(COMMAND "${GIT_EXECUTABLE}" apply --check "${fsc_patch}"
        WORKING_DIRECTORY "${source_dir}" RESULT_VARIABLE checked
        ERROR_VARIABLE check_error)
    if(NOT checked EQUAL 0)
        message(FATAL_ERROR "FSC derivative patch is incompatible: ${check_error}")
    endif()
    execute_process(COMMAND "${GIT_EXECUTABLE}" apply "${fsc_patch}"
        WORKING_DIRECTORY "${source_dir}" RESULT_VARIABLE applied
        ERROR_VARIABLE apply_error)
    if(NOT applied EQUAL 0)
        message(FATAL_ERROR "FSC derivative patch failed: ${apply_error}")
    endif()
endfunction()
