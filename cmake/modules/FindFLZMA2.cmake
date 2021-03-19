# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

#.rst:
# FindFLMA2
# -------
#
# Find the FLMA2 library header and define variables.
#
# Imported Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines :prop_tgt:`IMPORTED` target ``FLMA2::FLMA2``,
# if FLMA2 has been found
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
#   FLMA2_FOUND          - True if FLMA2 is found.
#   FLMA2_INCLUDE_DIRS   - Where to find fast-lzma2.h
#
# ::
#
#   FLMA2_VERSION        - The version of FLMA2 found (x.y.z)
#   FLMA2_VERSION_MAJOR  - The major version of FLMA2
#   FLMA2_VERSION_MINOR  - The minor version of FLMA2
#   FLMA2_VERSION_PATCH  - The patch version of FLMA2

find_path(FLMA2_INCLUDE_DIR NAME fast-lzma2.h PATH_SUFFIXES include)

if(NOT FLMA2_LIBRARY)
  find_library(FLMA2_LIBRARY NAMES fast-lzma2 PATH_SUFFIXES lib)
endif()

mark_as_advanced(FLMA2_INCLUDE_DIR)

if(FLMA2_INCLUDE_DIR AND EXISTS "${FLMA2_INCLUDE_DIR}/fast-lzma2.h")
  file(STRINGS "fast-lzma2.h" FLMA2_H REGEX "^#define FL2_VERSION_[A-Z]+[ ]+[0-9]+.*$")
  string(REGEX REPLACE ".+FLMA2_VERSION_MAJOR[ ]+([0-9]+).*$"   "\\1" FL2_VERSION_MAJOR "${FLMA2_H}")
  string(REGEX REPLACE ".+FLMA2_VERSION_MINOR[ ]+([0-9]+).*$"   "\\1" FL2_VERSION_MINOR "${FLMA2_H}")
  string(REGEX REPLACE ".+FLMA2_VERSION_RELEASE[ ]+([0-9]+).*$" "\\1" FL2_VERSION_PATCH "${FLMA2_H}")
  set(FLMA2_VERSION_STRING "${FLMA2_VERSION_MAJOR}.${FLMA2_VERSION_MINOR}.${FLMA2_VERSION_PATCH}")
  unset(FLMA2_H)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FLMA2
  REQUIRED_VARS FLMA2_LIBRARY FLMA2_INCLUDE_DIR VERSION_VAR FLMA2_VERSION)

if(FLMA2_FOUND)
  set(FLMA2_INCLUDE_DIRS "${FLMA2_INCLUDE_DIR}")

  if(NOT FLMA2_LIBRARIES)
    set(FLMA2_LIBRARIES ${FLMA2_LIBRARY})
  endif()

  if(NOT TARGET FLMA2::FLMA2)
    add_library(FLMA2::FLMA2 UNKNOWN IMPORTED)
    set_target_properties(FLMA2::FLMA2 PROPERTIES
      IMPORTED_LOCATION "${FLMA2_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${FLMA2_INCLUDE_DIRS}")
  endif()
endif()
