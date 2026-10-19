# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

#.rst:
# FindLHC4CODEC
# -------------
#
# Find the lhc4codec library header and define variables.
#
# Imported Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines :prop_tgt:`IMPORTED` target ``LHC4CODEC::LHC4CODEC``,
# if lhc4codec has been found.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
#   LHC4CODEC_FOUND          - True if lhc4codec is found.
#   LHC4CODEC_INCLUDE_DIRS   - Where to find lhc4codec/lhc4codec_c.h
#   LHC4CODEC_LIBRARIES      - The lhc4codec library path
#   LHC4CODEC_VERSION        - The lhc4codec version string, if available
#   LHC4CODEC_INSTALL_HELP   - How to install a missing package (always set)

# Prebuilt packages: https://gitlab.cern.ch/apeters/lhc4codec-bin
set(LHC4CODEC_INSTALL_HELP [=[
lhc4codec was not found (need lhc4codec/lhc4codec_c.h and liblhc4codec).

Install a prebuilt package from https://gitlab.cern.ch/apeters/lhc4codec-bin :

  # AlmaLinux 9 / 10 (x86_64)
  sudo curl -fsSL -o /etc/yum.repos.d/lhc4codec.repo \
    https://gitlab.cern.ch/apeters/lhc4codec-bin/-/raw/master/lhc4codec-el9.repo   # or -el10
  sudo dnf install lhc4codec lhc4codec-devel

  # macOS, Apple silicon (macOS 13+)
  brew tap apeters/lhc4codec https://gitlab.cern.ch/apeters/lhc4codec-bin.git
  brew install apeters/lhc4codec/lhc4codec

Or point CMake at an existing prefix:  -DLHC4CODEC_ROOT=/path/to/prefix
Or let ROOT clone and build it:        -Dbuiltin_lhc4codec=ON
  (fetches https://gitlab.cern.ch/apeters/lhc4codec.git)
]=])

set(_lhc4codec_hints)
if(LHC4CODEC_ROOT)
  list(APPEND _lhc4codec_hints "${LHC4CODEC_ROOT}")
endif()
if(DEFINED ENV{LHC4CODEC_ROOT})
  list(APPEND _lhc4codec_hints "$ENV{LHC4CODEC_ROOT}")
endif()

if(APPLE)
  # Homebrew: arm64 default is /opt/homebrew, Intel is /usr/local.
  list(APPEND _lhc4codec_hints /opt/homebrew /usr/local)
  find_program(_lhc4codec_brew brew)
  if(_lhc4codec_brew)
    execute_process(COMMAND "${_lhc4codec_brew}" --prefix lhc4codec
                    OUTPUT_VARIABLE _lhc4codec_brew_formula
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    if(_lhc4codec_brew_formula)
      list(APPEND _lhc4codec_hints "${_lhc4codec_brew_formula}")
    endif()
    execute_process(COMMAND "${_lhc4codec_brew}" --prefix
                    OUTPUT_VARIABLE _lhc4codec_brew_prefix
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    if(_lhc4codec_brew_prefix)
      list(APPEND _lhc4codec_hints "${_lhc4codec_brew_prefix}")
    endif()
  endif()
  unset(_lhc4codec_brew CACHE)
endif()

find_path(LHC4CODEC_INCLUDE_DIR lhc4codec/lhc4codec_c.h
  HINTS ${_lhc4codec_hints}
  PATH_SUFFIXES include
)

find_library(LHC4CODEC_LIBRARY
  NAMES lhc4codec lhc4codec_static
  HINTS ${_lhc4codec_hints}
  PATH_SUFFIXES lib lib64
)

if(LHC4CODEC_INCLUDE_DIR AND EXISTS "${LHC4CODEC_INCLUDE_DIR}/lhc4codec/lhc4codec_c.h")
  file(READ "${LHC4CODEC_INCLUDE_DIR}/lhc4codec/lhc4codec_c.h" _lhc4codec_header)
  string(REGEX MATCH "#define LHC4CODEC_VERSION \"([^\"]+)\"" _lhc4codec_version_match "${_lhc4codec_header}")
  if(CMAKE_MATCH_1)
    set(LHC4CODEC_VERSION "${CMAKE_MATCH_1}")
  endif()
  unset(_lhc4codec_header)
  unset(_lhc4codec_version_match)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LHC4CODEC
  REQUIRED_VARS LHC4CODEC_LIBRARY LHC4CODEC_INCLUDE_DIR
  VERSION_VAR LHC4CODEC_VERSION
)

if(LHC4CODEC_FOUND)
  set(LHC4CODEC_INCLUDE_DIRS "${LHC4CODEC_INCLUDE_DIR}")
  set(LHC4CODEC_LIBRARIES "${LHC4CODEC_LIBRARY}")

  if(NOT TARGET LHC4CODEC::LHC4CODEC)
    add_library(LHC4CODEC::LHC4CODEC UNKNOWN IMPORTED)
    set_target_properties(LHC4CODEC::LHC4CODEC PROPERTIES
      IMPORTED_LOCATION "${LHC4CODEC_LIBRARIES}"
      INTERFACE_INCLUDE_DIRECTORIES "${LHC4CODEC_INCLUDE_DIRS}")
  endif()
endif()

unset(_lhc4codec_hints)
unset(_lhc4codec_brew_formula)
unset(_lhc4codec_brew_prefix)
