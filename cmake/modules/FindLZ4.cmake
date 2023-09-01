# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

#.rst:
# FindLZ4
# -------
#
# Find the LZ4 library header and define variables.
#
# Imported Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines :prop_tgt:`IMPORTED` target ``LZ4::LZ4``,
# if LZ4 has been found
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
#   LZ4_FOUND          - True if LZ4 is found.
#   LZ4_INCLUDE_DIRS   - Where to find lz4.h
#
# ::
#
#   LZ4_VERSION        - The version of LZ4 found (x.y.z)
#   LZ4_VERSION_MAJOR  - The major version of LZ4
#   LZ4_VERSION_MINOR  - The minor version of LZ4
#   LZ4_VERSION_PATCH  - The patch version of LZ4

find_path(LZ4_INCLUDE_DIR NAME lz4.h PATH_SUFFIXES include)

if(NOT LZ4_LIBRARY)
  find_library(LZ4_LIBRARY NAMES lz4 PATH_SUFFIXES lib)
endif()

mark_as_advanced(LZ4_INCLUDE_DIR)

if(LZ4_INCLUDE_DIR AND EXISTS "${LZ4_INCLUDE_DIR}/lz4.h")
  file(STRINGS "${LZ4_INCLUDE_DIR}/lz4.h" LZ4_H REGEX "^#define LZ4_VERSION_[A-Z]+[ ]+[0-9]+.*$")
  string(REGEX REPLACE ".+LZ4_VERSION_MAJOR[ ]+([0-9]+).*$"   "\\1" LZ4_VERSION_MAJOR "${LZ4_H}")
  string(REGEX REPLACE ".+LZ4_VERSION_MINOR[ ]+([0-9]+).*$"   "\\1" LZ4_VERSION_MINOR "${LZ4_H}")
  string(REGEX REPLACE ".+LZ4_VERSION_RELEASE[ ]+([0-9]+).*$" "\\1" LZ4_VERSION_PATCH "${LZ4_H}")
  set(LZ4_VERSION "${LZ4_VERSION_MAJOR}.${LZ4_VERSION_MINOR}.${LZ4_VERSION_PATCH}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LZ4
  REQUIRED_VARS LZ4_LIBRARY LZ4_INCLUDE_DIR VERSION_VAR LZ4_VERSION)

if(LZ4_FOUND)
  set(LZ4_INCLUDE_DIRS "${LZ4_INCLUDE_DIR}")

  if(NOT LZ4_LIBRARIES)
    set(LZ4_LIBRARIES ${LZ4_LIBRARY})
  endif()

  if(NOT TARGET LZ4::LZ4)
    add_library(LZ4::LZ4 UNKNOWN IMPORTED)
    set_target_properties(LZ4::LZ4 PROPERTIES
      IMPORTED_LOCATION "${LZ4_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${LZ4_INCLUDE_DIRS}")
  endif()
endif()
