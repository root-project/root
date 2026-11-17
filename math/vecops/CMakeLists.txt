# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/vecops package
############################################################################

set(VECOPS_HEADERS
  ROOT/RVec.hxx
)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTVecOps
  HEADERS
    ${VECOPS_HEADERS}
  SOURCES
    src/RVec.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Core
)

set(RELATIVE_INC_HEADERS ${VECOPS_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(ROOTVecOps
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

if(builtin_vdt OR vdt)
  target_link_libraries(ROOTVecOps PRIVATE VDT::VDT)
endif()

include(CheckCXXSymbolExists)
check_symbol_exists(m __sqrt_finite HAVE_FINITE_MATH)
if(NOT HAVE_FINITE_MATH AND NOT MSVC)
  target_compile_options(ROOTVecOps PRIVATE -fno-finite-math-only)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
