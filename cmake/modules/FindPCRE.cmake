# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

#.rst:
# FindPCRE
# --------
#
# Find PCRE library
#
# Imported Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines :prop_tgt:`IMPORTED` target:
#
# ``PCRE::PCRE``
#   The pcre library, if found.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
# This module will set the following variables in your project:
#
# ``PCRE_FOUND``
#   True if PCRE has been found.
# ``PCRE_INCLUDE_DIRS``
#   Where to find pcre.h
# ``PCRE_LIBRARIES``
#   The libraries to link against to use PCRE.
# ``PCRE_VERSION``
#   The version of the PCRE found (e.g. 8.42)
#
# Obsolete variables
# ^^^^^^^^^^^^^^^^^^
#
# The following variables may also be set, for backwards compatibility:
#
# ``PCRE_PCRE_LIBRARY``
#   where to find the PCRE_PCRE library.
# ``PCRE_INCLUDE_DIR``
#   where to find the pcre.h header (same as PCRE_INCLUDE_DIRS)
#

foreach(var PCRE_FOUND PCRE_INCLUDE_DIR PCRE_PCRE_LIBRARY PCRE_LIBRARIES)
  unset(${var} CACHE)
endforeach()

find_path(PCRE_INCLUDE_DIR NAMES pcre.h PATH_SUFFIXES include)
mark_as_advanced(PCRE_INCLUDE_DIR)

if (PCRE_INCLUDE_DIR AND EXISTS "${PCRE_INCLUDE_DIR}/pcre.h")
  file(STRINGS "${PCRE_INCLUDE_DIR}/pcre.h" PCRE_H REGEX "^#define PCRE_(MAJOR|MINOR).*$")
  string(REGEX REPLACE "^.*PCRE_MAJOR[ ]+([0-9]+).*$" "\\1" PCRE_VERSION_MAJOR "${PCRE_H}")
  string(REGEX REPLACE "^.*PCRE_MINOR[ ]+([0-9]+).*$" "\\1" PCRE_VERSION_MINOR "${PCRE_H}")
  set(PCRE_VERSION "${PCRE_VERSION_MAJOR}.${PCRE_VERSION_MINOR}")
endif()

if(NOT PCRE_PCRE_LIBRARY)
  find_library(PCRE_PCRE_LIBRARY_RELEASE NAMES pcre)
  find_library(PCRE_PCRE_LIBRARY_DEBUG NAMES pcre${CMAKE_DEBUG_POSTFIX} pcred)
  include(SelectLibraryConfigurations)
  select_library_configurations(PCRE_PCRE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PCRE
  REQUIRED_VARS
    PCRE_INCLUDE_DIR
    PCRE_PCRE_LIBRARY
  VERSION_VAR
    PCRE_VERSION
)

if(PCRE_FOUND)
  set(PCRE_INCLUDE_DIRS "${PCRE_INCLUDE_DIR}")

  if (NOT PCRE_LIBRARIES)
    set(PCRE_LIBRARIES "${PCRE_PCRE_LIBRARY}")
  endif()

  if(NOT TARGET PCRE::PCRE)
    add_library(PCRE::PCRE UNKNOWN IMPORTED)
    set_target_properties(PCRE::PCRE PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${PCRE_INCLUDE_DIRS}")

    if(PCRE_PCRE_LIBRARY_DEBUG)
      set_property(TARGET PCRE::PCRE APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(PCRE::PCRE PROPERTIES
        IMPORTED_LOCATION_DEBUG "${PCRE_PCRE_LIBRARY_DEBUG}")
    endif()

    if(PCRE_PCRE_LIBRARY_RELEASE)
      set_property(TARGET PCRE::PCRE APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(PCRE::PCRE PROPERTIES
        IMPORTED_LOCATION_RELEASE "${PCRE_PCRE_LIBRARY_RELEASE}")
    endif()

    if(NOT PCRE_PCRE_LIBRARY_DEBUG AND NOT PCRE_PCRE_LIBRARY_RELEASE)
      set_property(TARGET PCRE::PCRE APPEND PROPERTY
        IMPORTED_LOCATION "${PCRE_PCRE_LIBRARY}")
    endif()
  endif()
endif()
