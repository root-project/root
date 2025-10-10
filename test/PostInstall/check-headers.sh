#!/bin/bash

# Adapted from the XRootD project with friendly permission from G. Amadio.
#
# This script checks that each installed ROOT header can be included individually
# without errors. The intention is to identify which headers may have missing
# includes, missing forward declarations, or missing header dependencies, that is,
# headers from ROOT which it includes, but were not installed by the install target.

# We need to split CXXFLAGS
# shellcheck disable=SC2086

: "${INCLUDE_DIR:=${1}}"
: "${CXX:=$(${INCLUDE_DIR}/../bin/root-config --cxx || echo c++)}"
: "${CXXFLAGS:=-Wall -Wextra -Wno-unused-parameter -Wno-unused-const-variable}"
: "${NCPU:=$(getconf _NPROCESSORS_ONLN)}"
: "${CXXSTANDARD:=$(${INCLUDE_DIR}/../bin/root-config --cxxstandard || echo 17)}"

if ! command -v "${CXX}" >/dev/null; then
	echo "Please set CXX to a valid compiler"
  exit 2
fi
if [ ! -d "${INCLUDE_DIR}" ]; then
  echo "Usage: ${0} <ROOT include directory>"
  echo "Alternatively, set INCLUDE_DIR in the environment"
  exit 2
fi


# Check all installed headers for include errors. Some headers cannot be used standalone:
suppressions="TMVA\|vdt"							# External
suppressions+="\|RField[A-Z]\|RtypesImp.h\|TAtomicCount[A-Z]\|CladDerivator.h\|TBranchProxyTemplate"	# Not to be used standalone
suppressions+="\|TWin32"							# Why are these installed in Linux?
suppressions+="\|xRooHypoSpace.h\|xRooFit"					# Uses macros to declare namespaces
suppressions+="\|RDaos.h"							# Might not be installed
suppressions+="\|RIoUring.hxx"							# Might not be installed
suppressions+="\|CPyCppyy/DispatchPtr.h\|CPyCppyy/API.h"			# Would need to include Python.h
suppressions+="\|bvh/"								# Includes a non-functioning std::span in c++17
suppressions+="\|cfortran.h"							# Seems unable to run with modern compilers
suppressions+="\|hipSYCL.h\|GenVectorX"						# Unconditionally installed on Fedora/Ubuntu even if broken
suppressions+="\|TR[A-Z].*__ctors.h"						# R interface without any includes, so cannot be parsed as C++
suppressions+="\|RTaskArena.hxx\|TThreadExecutor.hxx\|TTreeProcessorMT.hxx"	# Will raise errors if imt=Off

HEADERS=$(find "${INCLUDE_DIR}" -type f -name '*.h*' | grep -v "${suppressions}")

xargs -P ${NCPU:-1} -n 1 "${CXX}" -fsyntax-only -x c++ -std=c++${CXXSTANDARD} ${CXXFLAGS} -I"${INCLUDE_DIR}" <<< "${HEADERS}" || exit 2

