# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

set(headers
    include/CPyCppyy/API.h
    include/CPyCppyy/Reflex.h
    include/CPyCppyy/PyResult.h
    include/CPyCppyy/CommonDefs.h
    include/CPyCppyy/PyException.h
    include/CPyCppyy/DispatchPtr.h
)

set(sources
    src/API.cxx
    src/CPPClassMethod.cxx
    src/CPPConstructor.cxx
    src/CPPDataMember.cxx
    src/CPPEnum.cxx
    src/CPPExcInstance.cxx
    src/CPPFunction.cxx
    src/CPPGetSetItem.cxx
    src/CPPInstance.cxx
    src/CPPMethod.cxx
    src/CPPOperator.cxx
    src/CPPOverload.cxx
    src/CPPScope.cxx
    src/CPyCppyyModule.cxx
    src/CallContext.cxx
    src/Converters.cxx
    src/CustomPyTypes.cxx
    src/DispatchPtr.cxx
    src/Dispatcher.cxx
    src/Executors.cxx
    src/LowLevelViews.cxx
    src/MemoryRegulator.cxx
    src/ProxyWrappers.cxx
    src/PyException.cxx
    src/PyResult.cxx
    src/PyStrings.cxx
    src/Pythonize.cxx
    src/TemplateProxy.cxx
    src/TupleOfInstances.cxx
    src/TypeManip.cxx
    src/Utility.cxx
)


add_library(CPyCppyy SHARED ${headers} ${sources})
# Set the suffix to '.so' and the prefix to 'lib'
set_target_properties(CPyCppyy PROPERTIES  ${ROOT_LIBRARY_PROPERTIES})
target_link_libraries(CPyCppyy PRIVATE Core)
if(MSVC)
  target_link_libraries(CPyCppyy PRIVATE cppyy_backend)
  target_link_libraries(CPyCppyy PUBLIC Python3::Python)
  set_target_properties(CPyCppyy PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
elseif(APPLE)
  target_link_libraries(CPyCppyy PRIVATE -Wl,-bind_at_load -Wl,-w -Wl,-undefined -Wl,dynamic_lookup cppyy_backend)
else()
  target_link_libraries(CPyCppyy PRIVATE -Wl,--unresolved-symbols=ignore-all cppyy_backend)
endif()

if(NOT MSVC)
  target_compile_options(CPyCppyy PRIVATE -Wno-strict-aliasing)
endif()
if(NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND NOT MSVC)
  target_compile_options(CPyCppyy PRIVATE
    -Wno-unused-but-set-parameter)
endif()


add_library(cppyy SHARED src/CPyCppyyPyModule.cxx)

# Set the suffix to '.so' and the prefix to 'lib'
set_target_properties(cppyy PROPERTIES  ${ROOT_LIBRARY_PROPERTIES})
if(MSVC)
  target_link_libraries(cppyy PRIVATE CPyCppyy)
  set_target_properties(cppyy PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
  set_target_properties(cppyy PROPERTIES PREFIX "lib")
  set_target_properties(cppyy PROPERTIES SUFFIX ".pyd")
elseif(APPLE)
  target_link_libraries(cppyy PRIVATE -Wl,-bind_at_load -Wl,-w -Wl,-undefined -Wl,dynamic_lookup CPyCppyy)
else()
  target_link_libraries(cppyy PRIVATE -Wl,--unresolved-symbols=ignore-all CPyCppyy)
endif()

# Avoid warnings due to invalid function casts from C++ functions in CPyCppyy
# to CPython API function typedefs (e.g. PyCFunction). This is a common pattern
# in CPython extension implementations, explicitly encouraged by the official
# CPython docs for C/C++ extensions. see
# https://docs.python.org/3/extending/extending.html#keyword-parameters-for-extension-functions
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(CPyCppyy PRIVATE -Wno-cast-function-type)
endif()

# Disables warnings in Python 3.8 caused by the temporary extra filed for tp_print compatibility
# (see https://github.com/python/cpython/blob/3.8/Include/cpython/object.h#L260).
# Note that Python 3.8 is the lowers Python version that is still supported by
# ROOT, so this compile option can be completely removed soon.
if(NOT MSVC AND Python3_VERSION VERSION_LESS 3.9)
  target_compile_options(CPyCppyy PRIVATE -Wno-missing-field-initializers)
endif()

target_compile_definitions(CPyCppyy PRIVATE NO_CPPYY_LEGACY_NAMESPACE)

target_include_directories(CPyCppyy SYSTEM PUBLIC ${Python3_INCLUDE_DIRS})

target_include_directories(CPyCppyy
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(NOT MSVC)
  # Make sure that relative RUNPATH to main ROOT libraries is always correct.

  file(RELATIVE_PATH pymoduledir_to_libdir_build ${localruntimedir} "${localruntimedir}")
  file(RELATIVE_PATH pymoduledir_to_libdir_install ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_PYTHONDIR} "${CMAKE_INSTALL_FULL_LIBDIR}")

  if(APPLE)
    set_target_properties(cppyy PROPERTIES
        BUILD_RPATH "@loader_path/${pymoduledir_to_libdir_build}"
        INSTALL_RPATH "@loader_path/${pymoduledir_to_libdir_install}"
    )
  else()
    set_target_properties(cppyy PROPERTIES
        BUILD_RPATH "$ORIGIN/${pymoduledir_to_libdir_build}"
        INSTALL_RPATH "$ORIGIN/${pymoduledir_to_libdir_install}"
    )
  endif()

endif()

set_property(GLOBAL APPEND PROPERTY ROOT_EXPORTED_TARGETS CPyCppyy)
set_property(GLOBAL APPEND PROPERTY ROOT_EXPORTED_TARGETS cppyy)

# Install library
install(TARGETS CPyCppyy EXPORT ${CMAKE_PROJECT_NAME}Exports
                            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT libraries
                            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries
                            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries)

install(TARGETS cppyy EXPORT ${CMAKE_PROJECT_NAME}Exports
                            RUNTIME DESTINATION ${CMAKE_INSTALL_PYTHONDIR} COMPONENT libraries
                            LIBRARY DESTINATION ${CMAKE_INSTALL_PYTHONDIR} COMPONENT libraries
                            ARCHIVE DESTINATION ${CMAKE_INSTALL_PYTHONDIR} COMPONENT libraries)

file(COPY ${headers} DESTINATION ${CMAKE_BINARY_DIR}/include/CPyCppyy)

install(FILES ${headers}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/CPyCppyy
        COMPONENT headers)
