# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

#.rst:
# FindPCRE2
# --------
#
# Find PCRE2 library
#
# Imported Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines :prop_tgt:`IMPORTED` target:
#
# ``PCRE2::PCRE2``
#   The pcre2 library, if found.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
# This module will set the following variables in your project:
#
# ``PCRE2_FOUND``
#   True if PCRE2 has been found.
# ``PCRE2_INCLUDE_DIRS``
#   Where to find pcre2.h
# ``PCRE2_LIBRARIES``
#   The libraries to link against to use PCRE2.
# ``PCRE2_VERSION``
#   The version of the PCRE2 found (e.g. 10.42)
#
# Obsolete variables
# ^^^^^^^^^^^^^^^^^^
#
# The following variables may also be set, for backwards compatibility:
#
# ``PCRE2_PCRE2_LIBRARY``
#   where to find the PCRE2_PCRE2 library.
# ``PCRE2_INCLUDE_DIR``
#   where to find the pcre2.h header (same as PCRE2_INCLUDE_DIRS)
#

foreach(var PCRE2_FOUND PCRE2_INCLUDE_DIR PCRE2_PCRE2_LIBRARY PCRE2_LIBRARIES)
  unset(${var} CACHE)
endforeach()

find_path(PCRE2_INCLUDE_DIR NAMES pcre2.h PATH_SUFFIXES include)
mark_as_advanced(PCRE2_INCLUDE_DIR)

if (PCRE2_INCLUDE_DIR AND EXISTS "${PCRE2_INCLUDE_DIR}/pcre2.h")
  file(STRINGS "${PCRE2_INCLUDE_DIR}/pcre2.h" PCRE2_H REGEX "^#define PCRE2_(MAJOR|MINOR).*$")
  string(REGEX REPLACE "^.*PCRE2_MAJOR[ ]+([0-9]+).*$" "\\1" PCRE2_VERSION_MAJOR "${PCRE2_H}")
  string(REGEX REPLACE "^.*PCRE2_MINOR[ ]+([0-9]+).*$" "\\1" PCRE2_VERSION_MINOR "${PCRE2_H}")
  set(PCRE2_VERSION "${PCRE2_VERSION_MAJOR}.${PCRE2_VERSION_MINOR}")
endif()

if(NOT PCRE2_PCRE2_LIBRARY)
  find_library(PCRE2_PCRE2_LIBRARY_RELEASE NAMES pcre2-8)
  find_library(PCRE2_PCRE2_LIBRARY_DEBUG NAMES pcre2-8${CMAKE_DEBUG_POSTFIX} pcre2-8d)
  include(SelectLibraryConfigurations)
  select_library_configurations(PCRE2_PCRE2)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PCRE2
  REQUIRED_VARS
    PCRE2_INCLUDE_DIR
    PCRE2_PCRE2_LIBRARY
  VERSION_VAR
    PCRE2_VERSION
)

if(PCRE2_FOUND)
  set(PCRE2_INCLUDE_DIRS "${PCRE2_INCLUDE_DIR}")

  if (NOT PCRE2_LIBRARIES)
    set(PCRE2_LIBRARIES "${PCRE2_PCRE2_LIBRARY}")
  endif()

  if(NOT TARGET PCRE2::PCRE2)
    add_library(PCRE2::PCRE2 UNKNOWN IMPORTED)
    set_target_properties(PCRE2::PCRE2 PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${PCRE2_INCLUDE_DIRS}")

    if(PCRE2_PCRE2_LIBRARY_DEBUG)
      set_property(TARGET PCRE2::PCRE2 APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(PCRE2::PCRE2 PROPERTIES
        IMPORTED_LOCATION_DEBUG "${PCRE2_PCRE2_LIBRARY_DEBUG}")
    endif()

    if(PCRE2_PCRE2_LIBRARY_RELEASE)
      set_property(TARGET PCRE2::PCRE2 APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(PCRE2::PCRE2 PROPERTIES
        IMPORTED_LOCATION_RELEASE "${PCRE2_PCRE2_LIBRARY_RELEASE}")
    endif()

    if(NOT PCRE2_PCRE2_LIBRARY_DEBUG AND NOT PCRE2_PCRE2_LIBRARY_RELEASE)
      set_property(TARGET PCRE2::PCRE2 APPEND PROPERTY
        IMPORTED_LOCATION "${PCRE2_PCRE2_LIBRARY}")
    endif()
  endif()
endif()
