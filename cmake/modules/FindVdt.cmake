# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

#.rst:
# FindVdt
# -------
#
# Find the Vdt library header and define variables.
#
# Imported Targets
# ^^^^^^^^^^^^^^^^
#
# VDT::VDT if Vdt has been found.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
#   VDT_FOUND          - True if Vdt is found.
#   VDT_INCLUDE_DIRS   - Where to find vdt/vdtMath.h
#
# ::
#
#   VDT_VERSION        - The version of Vdt found (x.y.z)
#   VDT_VERSION_MAJOR  - The major version of Vdt
#   VDT_VERSION_MINOR  - The minor version of Vdt
#   VDT_VERSION_PATCH  - The patch version of Vdt
#

if(NOT VDT_INCLUDE_DIR)
  find_path(VDT_INCLUDE_DIR NAME vdt/vdtMath.h PATH_SUFFIXES include)
endif()

if(NOT VDT_LIBRARY)
  find_library(VDT_LIBRARY NAMES vdt)
endif()

if(VDT_INCLUDE_DIR)
  file(STRINGS "${VDT_INCLUDE_DIR}/vdt/vdtMath.h" VDT_H REGEX "^#define VDT_VERSION_[A-Z]+[ ]+[0-9]+.*$")
  string(REGEX REPLACE ".+VDT_VERSION_MAJOR[ ]+([0-9]+).*$" "\\1" VDT_VERSION_MAJOR "${VDT_H}")
  string(REGEX REPLACE ".+VDT_VERSION_MINOR[ ]+([0-9]+).*$" "\\1" VDT_VERSION_MINOR "${VDT_H}")
  string(REGEX REPLACE ".+VDT_VERSION_PATCH[ ]+([0-9]+).*$" "\\1" VDT_VERSION_PATCH "${VDT_H}")
  set(VDT_VERSION "${VDT_VERSION_MAJOR}.${VDT_VERSION_MINOR}.${VDT_VERSION_PATCH}")
  if("${VDT_VERSION}" STREQUAL "..")
    if(EXISTS "${VDT_INCLUDE_DIR}/vdt/tanh.h")
      set(VDT_VERSION "0.4")
    else()
      set(VDT_VERSION "0.3")
    endif()
  endif()
endif()

# Don't show in GUI
mark_as_advanced(VDT_FOUND VDT_VERSION VDT_INCLUDE_DIR VDT_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Vdt
  REQUIRED_VARS VDT_INCLUDE_DIR VDT_LIBRARY
  VERSION_VAR VDT_VERSION)


if(VDT_FOUND)
  set(VDT_INCLUDE_DIRS ${VDT_INCLUDE_DIR})

  if(NOT VDT_LIBRARIES)
    set(VDT_LIBRARIES ${VDT_LIBRARY})
  endif()

  if(NOT TARGET VDT::VDT)
    add_library(VDT::VDT SHARED IMPORTED)
    target_include_directories(VDT::VDT SYSTEM INTERFACE ${VDT_INCLUDE_DIRS})

    set_target_properties(VDT::VDT
      PROPERTIES
        IMPORTED_LOCATION "${VDT_LIBRARY}"
    )
  endif()
endif()
