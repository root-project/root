# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/foundation package
############################################################################

set(Foundation_dict_headers
  ESTLType.h
  RStringView.h
  TClassEdit.h
  ROOT/RIntegerSequence.hxx
  ROOT/RMakeUnique.hxx
  ROOT/RNotFn.hxx
  ROOT/RSpan.hxx
  ROOT/RStringView.hxx
  ROOT/span.hxx
  ROOT/TypeTraits.hxx
  PARENT_SCOPE
)

set(sources
  src/FoundationUtils.cxx
  src/RConversionRuleParser.cxx
  src/TClassEdit.cxx
)

ROOT_OBJECT_LIBRARY(Foundation ${sources})
target_include_directories(Foundation PRIVATE
   ${CMAKE_SOURCE_DIR}/core/foundation/res
   ${CMAKE_SOURCE_DIR}/core/base/inc
   ${CMAKE_BINARY_DIR}/ginclude
)

ROOT_OBJECT_LIBRARY(Foundation_Stage1 ${sources}) # used by rootcling_stage1
target_include_directories(Foundation_Stage1 PRIVATE
   ${CMAKE_SOURCE_DIR}/core/foundation/res
   ${CMAKE_SOURCE_DIR}/core/base/inc
   ${CMAKE_BINARY_DIR}/ginclude
)

if(MSVC)
   target_include_directories(Foundation PRIVATE ${CMAKE_SOURCE_DIR}/core/winnt/inc)
   target_include_directories(Foundation_Stage1 PRIVATE ${CMAKE_SOURCE_DIR}/core/winnt/inc)
endif()

set_target_properties(Foundation_Stage1 PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS} ${CLING_CXXFLAGS}")

ROOT_INSTALL_HEADERS()
ROOT_ADD_TEST_SUBDIRECTORY(test)
