# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/vecops package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTVecOps
  HEADERS
    ROOT/RVec.hxx
  SOURCES
    src/RVec.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Core
)

if(builtin_vdt OR vdt)
  target_link_libraries(ROOTVecOps PUBLIC VDT)
endif()

include(CheckCXXSymbolExists)
check_symbol_exists(m __sqrt_finite HAVE_FINITE_MATH)
if(NOT HAVE_FINITE_MATH AND NOT MSVC)
  target_compile_options(ROOTVecOps PRIVATE -fno-finite-math-only)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
