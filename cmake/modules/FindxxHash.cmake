# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

#.rst:
# FindxxHash
# -----------
#
# Find the xxHash library header and define variables.
#
# Imported Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines :prop_tgt:`IMPORTED` target ``xxHash::xxHash``,
# if xxHash has been found
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
#   xxHash_FOUND          - True if xxHash is found.
#   xxHash_INCLUDE_DIRS   - Where to find xxhash.h
#
# ::
#
#   xxHash_VERSION        - The version of xxHash found (x.y.z)
#   xxHash_VERSION_MAJOR  - The major version of xxHash
#   xxHash_VERSION_MINOR  - The minor version of xxHash
#   xxHash_VERSION_PATCH  - The patch version of xxHash

find_path(xxHash_INCLUDE_DIR NAME xxhash.h PATH_SUFFIXES include)
# Each library name given to the NAMES option is first considered as is, if it contains a library suffix, 
# and then considered with platform-specific prefixes (e.g. lib) and suffixes (e.g. .so), as defined by 
# the variables CMAKE_FIND_LIBRARY_PREFIXES and CMAKE_FIND_LIBRARY_SUFFIXES. Therefore one may specify 
# library file names such as libfoo.a directly. This can be used to locate static libraries on UNIX-like systems.
find_library(xxHash_LIBRARY NAMES libxxhash.${CMAKE_STATIC_LIBRARY_SUFFIX} libxxHash.${CMAKE_STATIC_LIBRARY_SUFFIX} xxhash xxHash PATH_SUFFIXES lib)

mark_as_advanced(xxHash_INCLUDE_DIR)

if(xxHash_INCLUDE_DIR AND EXISTS "${xxHash_INCLUDE_DIR}/xxhash.h")
  file(STRINGS "${xxHash_INCLUDE_DIR}/xxhash.h" XXHASH_H REGEX "^#define XXH_VERSION_[A-Z]+[ ]+[0-9]+$")
  string(REGEX REPLACE ".+XXH_VERSION_MAJOR[ ]+([0-9]+).*$"   "\\1" xxHash_VERSION_MAJOR "${XXHASH_H}")
  string(REGEX REPLACE ".+XXH_VERSION_MINOR[ ]+([0-9]+).*$"   "\\1" xxHash_VERSION_MINOR "${XXHASH_H}")
  string(REGEX REPLACE ".+XXH_VERSION_RELEASE[ ]+([0-9]+).*$" "\\1" xxHash_VERSION_PATCH "${XXHASH_H}")
  set(xxHash_VERSION "${xxHash_VERSION_MAJOR}.${xxHash_VERSION_MINOR}.${xxHash_VERSION_PATCH}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(xxHash
  REQUIRED_VARS xxHash_LIBRARY xxHash_INCLUDE_DIR VERSION_VAR xxHash_VERSION)

if(xxHash_FOUND)
  set(xxHash_INCLUDE_DIRS "${xxHash_INCLUDE_DIR}")

  if(NOT xxHash_LIBRARIES)
    set(xxHash_LIBRARIES ${xxHash_LIBRARY})
  endif()

  if(NOT TARGET xxHash::xxHash)
    add_library(xxHash::xxHash UNKNOWN IMPORTED)
    set_target_properties(xxHash::xxHash PROPERTIES
      IMPORTED_LOCATION "${xxHash_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${xxHash_INCLUDE_DIRS}")
  endif()
endif()
