# SPDX-FileCopyrightText: 2026 The Alea Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Additive controlled value/force fitting after the pinned derivative patch.
# Source overrides belong to their caller and must explicitly provide API v1.
function(alea_require_fsc_controlled_fit source_dir source_override)
    file(READ "${source_dir}/src/fast_screened_coulomb.h" fsc_header)
    if(fsc_header MATCHES "#define FSC_CONTROLLED_FIT_API 1([\r\n]|$)")
        return()
    endif()
    if(source_override)
        message(FATAL_ERROR
            "FSC source override lacks FSC_CONTROLLED_FIT_API 1. Update the library "
            "or apply cmake/patches/fsc_controlled_fit.patch after the derivative patch.")
    endif()
    find_package(Git REQUIRED)
    set(fsc_patch "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patches/fsc_controlled_fit.patch")
    execute_process(COMMAND "${GIT_EXECUTABLE}" apply --check "${fsc_patch}"
        WORKING_DIRECTORY "${source_dir}" RESULT_VARIABLE checked ERROR_VARIABLE check_error)
    if(NOT checked EQUAL 0)
        message(FATAL_ERROR "FSC controlled-fit patch is incompatible: ${check_error}")
    endif()
    execute_process(COMMAND "${GIT_EXECUTABLE}" apply "${fsc_patch}"
        WORKING_DIRECTORY "${source_dir}" RESULT_VARIABLE applied ERROR_VARIABLE apply_error)
    if(NOT applied EQUAL 0)
        message(FATAL_ERROR "FSC controlled-fit patch failed: ${apply_error}")
    endif()
endfunction()
