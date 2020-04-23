# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

set(headers
    inc/CPyCppyy/API.h
    inc/CPyCppyy/PyResult.h
    inc/CPyCppyy/TPyArg.h
    inc/CPyCppyy/CommonDefs.h
    inc/CPyCppyy/PyException.h
    inc/CPyCppyy/DispatchPtr.h
)

set(sources
    src/API.cxx
    src/CallContext.cxx
    src/Converters.cxx
    src/CPPClassMethod.cxx
    src/CPPConstructor.cxx
    src/CPPDataMember.cxx
    src/CPPExcInstance.cxx
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
    src/PyException.cxx
    src/PyResult.cxx
    src/TPyArg.cxx
    src/TupleOfInstances.cxx
    src/TypeManip.cxx
    src/Utility.cxx
)

foreach(val RANGE ${how_many_pythons})
  list(GET python_include_dirs ${val} python_include_dir)
  list(GET python_under_version_strings ${val} python_under_version_string)
  list(GET python_version_strings ${val} python_version_string)
  list(GET python_libraries ${val} python_library)

  set(libname cppyy${python_under_version_string})

  add_library(${libname} SHARED ${headers} ${sources})

  if(NOT MSVC)
    target_compile_options(${libname} PRIVATE
      -Wno-shadow -Wno-strict-aliasing)
  endif()
  if(NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    target_compile_options(${libname} PRIVATE
      -Wno-unused-but-set-parameter)
  endif()

  # Disables warnings coming from PyCFunction casts
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL 8)
    target_compile_options(${libname} PRIVATE -Wno-cast-function-type)
  endif()

  # Disables warnings originating from deprecated register keyword in Python
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_STANDARD GREATER_EQUAL 11)
    target_compile_options(${libname} PRIVATE -Wno-register)
  endif()
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND CMAKE_CXX_STANDARD GREATER_EQUAL 11)
    target_compile_options(${libname} PRIVATE -Wno-register)
    target_compile_options(${libname} PRIVATE -Wno-deprecated-register)
  endif()

  # Disables warnings due to new field tp_vectorcall in Python 3.8
  if(NOT MSVC AND ${python_version_string} VERSION_GREATER_EQUAL "3.8")
    target_compile_options(${libname} PRIVATE -Wno-missing-field-initializers)
  endif()

  target_include_directories(${libname}
     PRIVATE
        ${CMAKE_SOURCE_DIR}/core/foundation/inc   # needed for string_view backport
     PUBLIC
        ${python_include_dir}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  )

  target_link_libraries(${libname} cppyy_backend${python_under_version_string} ${python_library})

  set_property(GLOBAL APPEND PROPERTY ROOT_EXPORTED_TARGETS ${libname})

  # Install library
  install(TARGETS ${libname} EXPORT ${CMAKE_PROJECT_NAME}Exports DESTINATION ${runtimedir})
endforeach()

file(COPY ${headers} DESTINATION ${CMAKE_BINARY_DIR}/include/CPyCppyy)
install(FILES ${headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/CPyCppyy)
