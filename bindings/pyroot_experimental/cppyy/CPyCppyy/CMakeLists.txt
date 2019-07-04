# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.


#include_directories(BEFORE include)

ROOT_STANDARD_LIBRARY_PACKAGE(cppyy
  HEADERS
    CPyCppyy/CommonDefs.h
    CPyCppyy/TPyArg.h
    CPyCppyy/TPyException.h
    CPyCppyy/TPyReturn.h
    CPyCppyy/API.h
  SOURCES
    src/CallContext.cxx
    src/Converters.cxx
    src/CPPClassMethod.cxx
    src/CPPConstructor.cxx
    src/CPPDataMember.cxx
    src/CPPFunction.cxx
    src/CPPInstance.cxx
    src/CPPMethod.cxx
    src/CPPOverload.cxx
    src/CPPScope.cxx
    src/CPPGetSetItem.cxx
    src/CPyCppyyModule.cxx
    src/CustomPyTypes.cxx
    src/Dispatcher.cxx
    src/Executors.cxx
    src/LowLevelViews.cxx
    src/MemoryRegulator.cxx
    src/ProxyWrappers.cxx
    src/PyStrings.cxx
    src/Pythonize.cxx
    src/TemplateProxy.cxx
    src/TPyArg.cxx
    src/TPyClassGenerator.cxx
    src/TPyException.cxx
    src/TPyReturn.cxx
    src/API.cxx
    src/TupleOfInstances.cxx
    src/TypeManip.cxx
    src/Utility.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
)

if(NOT MSVC)
  target_compile_options(cppyy PRIVATE
    -Wno-shadow -Wno-strict-aliasing -Wno-unused-but-set-parameter)
endif()

# Disables warnings coming from PyCFunction casts
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL 8)
  target_compile_options(cppyy PRIVATE -Wno-cast-function-type)
endif()

# Disables warnings originating from deprecated register keyword in Python
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_STANDARD GREATER_EQUAL 11)
    target_compile_options(cppyy PRIVATE -Wno-register)
endif()
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang" AND CMAKE_CXX_STANDARD GREATER_EQUAL 11)
    target_compile_options(cppyy PRIVATE -Wno-deprecated-register)
endif()

target_include_directories(cppyy PRIVATE ${CMAKE_BINARY_DIR}/include) # needed for string_view backport

target_include_directories(cppyy PUBLIC ${PYTHON_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_link_libraries(cppyy cppyy_backend ${PYTHON_LIBRARIES})
