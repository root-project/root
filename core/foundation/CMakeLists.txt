# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/foundation package
############################################################################

set_property(TARGET Core APPEND PROPERTY DICT_HEADERS
  ESTLType.h
  RStringView.h
  TClassEdit.h
  TError.h
  ThreadLocalStorage.h
  ROOT/RLogger.hxx
  ROOT/RNotFn.hxx
  ROOT/RRangeCast.hxx
  ROOT/RSpan.hxx
  ROOT/RStringView.hxx
  ROOT/StringUtils.hxx
  ROOT/span.hxx
  ROOT/TypeTraits.hxx
)

set(FOUNDATION_SOURCES
  src/FoundationUtils.cxx
  src/RConversionRuleParser.cxx
  src/RLogger.cxx
  src/StringUtils.cxx
  src/TClassEdit.cxx
  src/TError.cxx
)

set(FOUNDATION_HEADER_DIRS inc/)

if(root7)
  set_property(TARGET Core APPEND PROPERTY DICT_HEADERS ROOT/RError.hxx)
  list(APPEND FOUNDATION_SOURCES
    v7/src/RError.cxx)
  list(APPEND FOUNDATION_HEADER_DIRS v7/inc/)
  ROOT_ADD_TEST_SUBDIRECTORY(v7/test)
endif()

target_sources(Core PRIVATE ${FOUNDATION_SOURCES})

target_include_directories(Core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/v7/inc>
  PRIVATE
    res
)

ROOT_OBJECT_LIBRARY(Foundation_Stage1 ${FOUNDATION_SOURCES}) # used by rootcling_stage1

target_include_directories(Foundation_Stage1
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
  PRIVATE
    $<$<BOOL:${MSVC}>:${CMAKE_SOURCE_DIR}/core/winnt/inc>
    ${CMAKE_BINARY_DIR}/ginclude res
)

set_target_properties(Foundation_Stage1 PROPERTIES
  COMPILE_FLAGS "${COMPILE_FLAGS} ${CLING_CXXFLAGS}"
  VISIBILITY_INLINES_HIDDEN "ON"
)

ROOT_INSTALL_HEADERS(${FOUNDATION_HEADER_DIRS})
ROOT_ADD_TEST_SUBDIRECTORY(test)
