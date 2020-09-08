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
  TError.h
  ROOT/RIntegerSequence.hxx
  ROOT/RMakeUnique.hxx
  ROOT/RNotFn.hxx
  ROOT/RSpan.hxx
  ROOT/RStringView.hxx
  ROOT/span.hxx
  ROOT/TypeTraits.hxx
  PARENT_SCOPE
)

set(FOUNDATION_SOURCES
  src/FoundationUtils.cxx
  src/RConversionRuleParser.cxx
  src/TClassEdit.cxx
  src/TError.cxx
)

if(CMAKE_CXX_STANDARD GREATER 11)
  list(APPEND Foundation_dict_headers ROOT/RLogger.hxx)
  list(APPEND FOUNDATION_SOURCES v7/src/RLogger.cxx)
  set(FOUNDATION_V7_INC ${CMAKE_SOURCE_DIR}/core/foundation/v7/inc)
  # TLogger.hxx may be used without root7 flag, but is placed in v7/inc/,
  # so we need to tell ROOT_INSTALL_HEADERS() where to find it
  set(FOUNDATION_HEADER_DIRS inc/ v7/inc/)
endif()

if(root7)
  list(APPEND Foundation_dict_headers
    ROOT/RError.hxx)
  list(APPEND FOUNDATION_SOURCES
    v7/src/RError.cxx)
  ROOT_ADD_TEST_SUBDIRECTORY(v7/test)
endif()

ROOT_OBJECT_LIBRARY(Foundation ${FOUNDATION_SOURCES})
target_include_directories(Foundation PRIVATE
   ${FOUNDATION_V7_INC}
   ${CMAKE_SOURCE_DIR}/core/foundation/res
   ${CMAKE_BINARY_DIR}/ginclude
)

ROOT_OBJECT_LIBRARY(Foundation_Stage1 ${FOUNDATION_SOURCES}) # used by rootcling_stage1
target_include_directories(Foundation_Stage1 PRIVATE
   ${FOUNDATION_V7_INC}
   ${CMAKE_SOURCE_DIR}/core/foundation/res
   ${CMAKE_BINARY_DIR}/ginclude
)

if(MSVC)
   target_include_directories(Foundation PRIVATE ${CMAKE_SOURCE_DIR}/core/winnt/inc)
   target_include_directories(Foundation_Stage1 PRIVATE ${CMAKE_SOURCE_DIR}/core/winnt/inc)
endif()

set_target_properties(Foundation_Stage1 PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS} ${CLING_CXXFLAGS}")

ROOT_INSTALL_HEADERS(${FOUNDATION_HEADER_DIRS})
ROOT_ADD_TEST_SUBDIRECTORY(test)
