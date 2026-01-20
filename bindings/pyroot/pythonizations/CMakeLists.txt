# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

################################################################
# CMakeLists.txt file for building ROOT pythonizations libraries
################################################################

list(APPEND PYROOT_EXTRA_HEADERS
     inc/TPyDispatcher.h)

set(cpp_sources
    src/PyROOTModule.cxx
    src/RPyROOTApplication.cxx
    src/GenericPyz.cxx
    src/TClassPyz.cxx
    src/TTreePyz.cxx
    src/CPPInstancePyz.cxx
    src/TPyDispatcher.cxx
    inc/TPyDispatcher.h
)

set(ROOT_headers_dir inc)

# Copy headers inside build_dir/include/ROOT
file(COPY ${ROOT_headers_dir}/ DESTINATION ${CMAKE_BINARY_DIR}/include/ROOT)

set(libname ROOTPythonizations)

add_library(${libname} SHARED ${cpp_sources})

# Set the suffix to '.so' and the prefix to 'lib'
set_target_properties(${libname} PROPERTIES ${ROOT_LIBRARY_PROPERTIES_NO_VERSION})
target_link_libraries(${libname} PUBLIC Core Tree CPyCppyy)
if(MSVC)
  set_target_properties(${libname} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
  set_target_properties(${libname} PROPERTIES SUFFIX ".pyd")
elseif(APPLE)
  target_link_libraries(${libname} PUBLIC -Wl,-bind_at_load -Wl,-w -Wl,-undefined -Wl,dynamic_lookup)
else()
  target_link_libraries(${libname} PUBLIC -Wl,--unresolved-symbols=ignore-all)
endif()

target_include_directories(${libname}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>)

# Disables warnings caused by Py_RETURN_TRUE/Py_RETURN_FALSE
if(NOT MSVC)
  target_compile_options(${libname} PRIVATE -Wno-strict-aliasing)
endif()


# Create meta-target PyROOT3 (INTERFACE library)
# Export of targets are not supported for custom targets(add_custom_targets())
add_library(PyROOT INTERFACE)
target_link_libraries(PyROOT INTERFACE cppyy_backend cppyy ROOTPythonizations)

# Define library output directories for build and install trees
set(pymoduledir_build "${localruntimedir}/ROOT")
set(pymoduledir_install "${CMAKE_INSTALL_PYTHONDIR}/ROOT")

# To make sure that the library also ends up in the right subdirectory in the
# build directory tree.
if(MSVC)
  set_target_properties(${libname}
      PROPERTIES
          RUNTIME_OUTPUT_DIRECTORY_DEBUG ${pymoduledir_build}
          RUNTIME_OUTPUT_DIRECTORY_RELEASE ${pymoduledir_build}
          RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${pymoduledir_build}
          RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${pymoduledir_build})
else()
  set_target_properties(${libname} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${pymoduledir_build})
endif()

if(NOT MSVC)
  # Make sure that relative RUNPATH to main ROOT libraries is always correct.
  ROOT_APPEND_LIBDIR_TO_INSTALL_RPATH(${libname} ${pymoduledir_install})
endif()

# Install library
install(TARGETS ${libname} EXPORT ${CMAKE_PROJECT_NAME}Exports
                            RUNTIME DESTINATION ${pymoduledir_install} COMPONENT libraries
                            LIBRARY DESTINATION ${pymoduledir_install}  COMPONENT libraries
                            ARCHIVE DESTINATION ${pymoduledir_install} COMPONENT libraries)

# Install headers required by pythonizations
install(FILES ${PYROOT_EXTRA_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ROOT
        COMPONENT headers)

# For the ROOT Python package
add_subdirectory(python)

ROOT_ADD_TEST_SUBDIRECTORY(test)
