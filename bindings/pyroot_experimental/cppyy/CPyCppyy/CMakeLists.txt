# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

set(headers
    inc/CPyCppyy/CommonDefs.h
    inc/CPyCppyy/TPyException.h
    inc/CPyCppyy/DispatchPtr.h
    inc/CPyCppyy/PyObjectPtr.h
)

set(sources
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
    src/DispatchPtr.cxx
    src/Executors.cxx
    src/LowLevelViews.cxx
    src/MemoryRegulator.cxx
    src/ProxyWrappers.cxx
    src/PyStrings.cxx
    src/Pythonize.cxx
    src/TemplateProxy.cxx
    src/TPyException.cxx
    src/TupleOfInstances.cxx
    src/TypeManip.cxx
    src/Utility.cxx
)

add_library(cppyy SHARED ${headers} ${sources})

if(NOT MSVC)
  target_compile_options(cppyy PRIVATE
    -Wno-shadow -Wno-strict-aliasing)
endif()
if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(cppyy PRIVATE
    -Wno-unused-but-set-parameter)
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

# Disables warnings due to new field tp_vectorcall in Python 3.8
if(NOT MSVC AND ${PYTHON_VERSION_STRING} VERSION_GREATER_EQUAL "3.8")
  target_compile_options(cppyy PRIVATE -Wno-missing-field-initializers)
endif()

target_include_directories(cppyy PRIVATE ${CMAKE_BINARY_DIR}/include) # needed for string_view backport

target_include_directories(cppyy PUBLIC ${PYTHON_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_link_libraries(cppyy cppyy_backend ${PYTHON_LIBRARIES})

set_property(GLOBAL APPEND PROPERTY ROOT_EXPORTED_TARGETS cppyy)
install(TARGETS cppyy EXPORT ${CMAKE_PROJECT_NAME}Exports DESTINATION ${CMAKE_INSTALL_PYROOTDIR})

file(COPY ${headers} DESTINATION ${CMAKE_BINARY_DIR}/include/CPyCppyy)
install(FILES ${headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/CPyCppyy)

