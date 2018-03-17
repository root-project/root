#.rst:
# FindLZMA
# -------
#
# Find the LZMA library header and define variables.
#
# Imported Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines :prop_tgt:`IMPORTED` target ``LZMA::LZMA``,
# if LZMA has been found
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
#   LZMA_FOUND          - True if LZMA is found.
#   LZMA_INCLUDE_DIRS   - Where to find lzma.h
#
# ::
#
#   LZMA_VERSION        - The version of LZMA found (x.y.z)
#   LZMA_VERSION_MAJOR  - The major version of LZMA
#   LZMA_VERSION_MINOR  - The minor version of LZMA
#   LZMA_VERSION_PATCH  - The patch version of LZMA

find_path(LZMA_INCLUDE_DIR NAME lzma.h PATH_SUFFIXES include)
find_library(LZMA_LIBRARY NAMES lzma PATH_SUFFIXES lib)

if(LZMA_INCLUDE_DIR AND EXISTS "${LZMA_INCLUDE_DIR}/lzma/version.h")
  file(STRINGS "${LZMA_INCLUDE_DIR}/lzma/version.h" LZMA_H REGEX "^#define LZMA_VERSION_[A-Z]+[ ]+[0-9]+.*$")
  string(REGEX REPLACE ".+LZMA_VERSION_MAJOR[ ]+([0-9]+).*$" "\\1" LZMA_VERSION_MAJOR "${LZMA_H}")
  string(REGEX REPLACE ".+LZMA_VERSION_MINOR[ ]+([0-9]+).*$" "\\1" LZMA_VERSION_MINOR "${LZMA_H}")
  string(REGEX REPLACE ".+LZMA_VERSION_PATCH[ ]+([0-9]+).*$" "\\1" LZMA_VERSION_PATCH "${LZMA_H}")
  set(LZMA_VERSION "${LZMA_VERSION_MAJOR}.${LZMA_VERSION_MINOR}.${LZMA_VERSION_PATCH}")
  unset(LZMA_H)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LZMA
  REQUIRED_VARS LZMA_INCLUDE_DIR LZMA_LIBRARY VERSION_VAR LZMA_VERSION)

if(LZMA_FOUND)
  set(LZMA_INCLUDE_DIRS "${LZMA_INCLUDE_DIR}")
  set(LZMA_LIBRARIES ${LZMA_LIBRARY})

  if(NOT TARGET LZMA::LZMA)
    add_library(LZMA::LZMA UNKNOWN IMPORTED)
    set_target_properties(LZMA::LZMA PROPERTIES
      IMPORTED_LOCATION "${LZMA_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${LZMA_INCLUDE_DIRS}")
  endif()
endif()

mark_as_advanced(LZMA_INCLUDE_DIR LZMA_LIBRARY)
