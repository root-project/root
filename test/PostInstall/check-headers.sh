#!/bin/bash

# Adapted from the XRootD project with friendly permission from G. Amadio.
#
# This script checks that each installed ROOT header can be included individually
# without errors. The intention is to identify which headers may have missing
# includes, missing forward declarations, or missing header dependencies, that is,
# headers from ROOT which it includes, but were not installed by the install target.

# We need to split CXXFLAGS
# shellcheck disable=SC2086

: "${CXX:=c++}"
: "${CXXFLAGS:=-std=c++17 -Wall -Wextra -Wno-unused-parameter -Wno-unused-const-variable}"
: "${INCLUDE_DIR:=${1}}"
: "${NCPU:=$(getconf _NPROCESSORS_ONLN)}"

if ! command -v "${CXX}" >/dev/null; then
	echo "Please set CXX to a valid compiler"
  exit 2
fi
if [ ! -d "${INCLUDE_DIR}" ]; then
  echo "Usage: ${0} <ROOT include directory>"
  echo "Alternatively, set INCLUDE_DIR in the environment"
  exit 2
fi


# Check all installed headers for include errors.
HEADERS=$(find "${INCLUDE_DIR}" -type f -name '*.h*')

xargs -P ${NCPU:-1} -n 1 "${CXX}" -fsyntax-only -x c++ ${CXXFLAGS} -I"${INCLUDE_DIR}" <<< "${HEADERS}" || exit 1

