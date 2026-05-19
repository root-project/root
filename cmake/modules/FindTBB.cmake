##===-- CMakeLists.txt ----------------------------------------------------===##
#
# Copyright (C) Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# This file incorporates work covered by the following copyright and permission
# notice:
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
#
##===----------------------------------------------------------------------===##

include(FindPackageHandleStandardArgs)

find_package(TBB QUIET CONFIG)
if (TBB_FOUND)
    find_package_handle_standard_args(TBB CONFIG_MODE)
    return()
endif()

if (NOT TBB_FIND_COMPONENTS)
    set(TBB_FIND_COMPONENTS tbb tbbmalloc)
    foreach (_tbb_component ${TBB_FIND_COMPONENTS})
        set(TBB_FIND_REQUIRED_${_tbb_component} 1)
    endforeach()
endif()

if (WIN32)
    list(APPEND ADDITIONAL_LIB_DIRS ENV PATH ENV LIB)
    list(APPEND ADDITIONAL_INCLUDE_DIRS ENV INCLUDE ENV CPATH)
else()
    list(APPEND ADDITIONAL_LIB_DIRS ENV LIBRARY_PATH ENV LD_LIBRARY_PATH ENV DYLD_LIBRARY_PATH)
    list(APPEND ADDITIONAL_INCLUDE_DIRS ENV CPATH ENV C_INCLUDE_PATH ENV CPLUS_INCLUDE_PATH ENV INCLUDE_PATH)
endif()

find_path(_tbb_include_dir NAMES oneapi/tbb.h PATHS ${ADDITIONAL_INCLUDE_DIRS})
if (_tbb_include_dir)
    file(READ "${_tbb_include_dir}/oneapi/tbb/version.h" _tbb_version_info LIMIT 2048)
    string(REGEX REPLACE ".*#define TBB_VERSION_MAJOR ([0-9]+).*" "\\1" _tbb_ver_major "${_tbb_version_info}")
    string(REGEX REPLACE ".*#define TBB_VERSION_MINOR ([0-9]+).*" "\\1" _tbb_ver_minor "${_tbb_version_info}")
    string(REGEX REPLACE ".*#define TBB_VERSION_PATCH ([0-9]+).*" "\\1" _tbb_ver_patch "${_tbb_version_info}")

    set(TBB_VERSION "${_tbb_ver_major}.${_tbb_ver_minor}.${_tbb_ver_patch}")
    unset(_tbb_version_info)
    unset(_tbb_ver_major)
    unset(_tbb_ver_minor)
    unset(_tbb_ver_patch)

    set(_TBB_BUILD_MODES RELEASE DEBUG)
    set(_TBB_DEBUG_SUFFIX _debug)

    foreach (_tbb_component ${TBB_FIND_COMPONENTS})
        if (NOT TARGET TBB::${_tbb_component})
            add_library(TBB::${_tbb_component} SHARED IMPORTED)
            set_property(TARGET TBB::${_tbb_component} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${_tbb_include_dir})

            set_target_properties(TBB::${_tbb_component} PROPERTIES
                                  INTERFACE_COMPILE_DEFINITIONS "__TBB_NO_IMPLICIT_LINKAGE=1")

            foreach(_TBB_BUILD_MODE ${_TBB_BUILD_MODES})
                set(_tbb_component_lib_name ${_tbb_component}${_TBB_${_TBB_BUILD_MODE}_SUFFIX})
                set(_tbb_component_filename ${_tbb_component_lib_name})

                if (WIN32 AND _tbb_component STREQUAL tbb)
                    set(_tbb_component_filename ${_tbb_component}12${_TBB_${_TBB_BUILD_MODE}_SUFFIX})
                endif()

                if (WIN32)
                    find_library(${_tbb_component_lib_name}_lib ${_tbb_component_filename} PATHS ${ADDITIONAL_LIB_DIRS})
                    find_file(${_tbb_component_lib_name}_dll ${_tbb_component_filename}.dll PATHS ${ADDITIONAL_LIB_DIRS})

                    set_target_properties(TBB::${_tbb_component} PROPERTIES
                                          IMPORTED_LOCATION_${_TBB_BUILD_MODE} "${${_tbb_component_lib_name}_dll}"
                                          IMPORTED_IMPLIB_${_TBB_BUILD_MODE}   "${${_tbb_component_lib_name}_lib}"
                                          )
                else()
                    find_library(${_tbb_component_lib_name}_so ${_tbb_component_filename} PATHS ${ADDITIONAL_LIB_DIRS})

                    set_target_properties(TBB::${_tbb_component} PROPERTIES
                                          IMPORTED_LOCATION_${_TBB_BUILD_MODE} "${${_tbb_component_lib_name}_so}"
                                          )
                endif()
                if (${_tbb_component_lib_name}_lib AND ${_tbb_component_lib_name}_dll OR ${_tbb_component_lib_name}_so)
                    set_property(TARGET TBB::${_tbb_component} APPEND PROPERTY IMPORTED_CONFIGURATIONS ${_TBB_BUILD_MODE})
                    list(APPEND TBB_IMPORTED_TARGETS TBB::${_tbb_component})
                    set(TBB_${_tbb_component}_FOUND 1)
                endif()
                unset(${_tbb_component_lib_name}_lib CACHE)
                unset(${_tbb_component_lib_name}_dll CACHE)
                unset(${_tbb_component_lib_name}_so CACHE)
                unset(_tbb_component_lib_name)
            endforeach()
        endif()
    endforeach()
    unset(_TBB_BUILD_MODES)
    unset(_TBB_DEBUG_SUFFIX)
endif()
unset(_tbb_include_dir CACHE)

list(REMOVE_DUPLICATES TBB_IMPORTED_TARGETS)

find_package_handle_standard_args(TBB
                                  REQUIRED_VARS TBB_IMPORTED_TARGETS
                                  HANDLE_COMPONENTS)
