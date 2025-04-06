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
    src/CallContext.cxx
    src/Converters.cxx
    src/CPPClassMethod.cxx
    src/CPPConstructor.cxx
    src/CPPDataMember.cxx
    src/CPPEnum.cxx
    src/CPPExcInstance.cxx
    src/CPPFunction.cxx
    src/CPPInstance.cxx
    src/CPPMethod.cxx
    src/CPPOperator.cxx
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
    src/TupleOfInstances.cxx
    src/TypeManip.cxx
    src/Utility.cxx
)

file(RELATIVE_PATH PYTHONDIR_TO_LIBDIR "${CMAKE_INSTALL_FULL_PYTHONDIR}" "${CMAKE_INSTALL_FULL_LIBDIR}")

set(libname cppyy)

add_library(${libname} SHARED ${headers} ${sources})
# Set the suffix to '.so' and the prefix to 'lib'
set_target_properties(${libname} PROPERTIES  ${ROOT_LIBRARY_PROPERTIES})
if(MSVC)
  target_link_libraries(${libname} PUBLIC cppyy_backend Python3::Python)
  set_target_properties(${libname} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
  set_target_properties(${libname} PROPERTIES PREFIX "lib")
  set_target_properties(${libname} PROPERTIES SUFFIX ".pyd")
elseif(APPLE)
  target_link_libraries(${libname} PUBLIC -Wl,-bind_at_load -Wl,-w -Wl,-undefined -Wl,dynamic_lookup cppyy_backend)
else()
  target_link_libraries(${libname} PUBLIC -Wl,--unresolved-symbols=ignore-all cppyy_backend)
endif()

if(NOT MSVC)
  target_compile_options(${libname} PRIVATE -Wno-strict-aliasing)
endif()
if(NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND NOT MSVC)
  target_compile_options(${libname} PRIVATE
    -Wno-unused-but-set-parameter)
endif()

# Avoid warnings due to invalid function casts from C++ functions in CPyCppyy
# to CPython API function typedefs (e.g. PyCFunction). This is a common pattern
# in CPython extension implementations, explicitly encouraged by the official
# CPython docs for C/C++ extensions. see
# https://docs.python.org/3/extending/extending.html#keyword-parameters-for-extension-functions
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(${libname} PRIVATE -Wno-cast-function-type)
endif()

# Disables warnings in Python 3.8 caused by the temporary extra filed for tp_print compatibility
# (see https://github.com/python/cpython/blob/3.8/Include/cpython/object.h#L260).
# Note that Python 3.8 is the lowers Python version that is still supported by
# ROOT, so this compile option can be completely removed soon.
if(NOT MSVC AND Python3_VERSION VERSION_LESS 3.9)
  target_compile_options(${libname} PRIVATE -Wno-missing-field-initializers)
endif()

target_compile_definitions(${libname} PRIVATE NO_CPPYY_LEGACY_NAMESPACE)

target_include_directories(${libname}
    SYSTEM PUBLIC ${Python3_INCLUDE_DIRS})

target_include_directories(${libname}
    PRIVATE
      ${CMAKE_BINARY_DIR}/ginclude
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

set_property(GLOBAL APPEND PROPERTY ROOT_EXPORTED_TARGETS ${libname})

# Install library
install(TARGETS ${libname} EXPORT ${CMAKE_PROJECT_NAME}Exports
                            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT libraries
                            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries
                            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries)
if (NOT MSVC AND NOT CMAKE_INSTALL_LIBDIR STREQUAL CMAKE_INSTALL_PYTHONDIR)
  # add a symlink to ${libname} in CMAKE_INSTALL_PYTHONDIR
  set(LIB_FILE_NAME ${CMAKE_SHARED_LIBRARY_PREFIX}${libname}.so)
  install(CODE "file(CREATE_LINK ${PYTHONDIR_TO_LIBDIR}/${LIB_FILE_NAME}
    \$ENV{DESTDIR}${CMAKE_INSTALL_FULL_PYTHONDIR}/${LIB_FILE_NAME} SYMBOLIC)")
endif()

file(COPY ${headers} DESTINATION ${CMAKE_BINARY_DIR}/include/CPyCppyy)
install(FILES ${headers}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/CPyCppyy
        COMPONENT headers)
