# SOFIE generates the C++ code to evaluate ONNX models. ROOT only uses it at
# runtime: libSOFIE is loaded when it is needed, and no ROOT target links
# against it or includes its headers. This builtin builds the library and
# installs it next to the ROOT libraries, where ROOT finds it at runtime.
#
# The sources are currently part of the ROOT repository, in the sofie
# subdirectory. Once SOFIE is released on its own, they will be downloaded
# like the sources of the other builtins:
#   set(ROOT_SOFIE_VERSION ...)
#   set(ROOT_SOFIE_HASH ...)
#   URL ${lcgpackages}/sofie-${ROOT_SOFIE_VERSION}.tar.gz
#   URL_HASH SHA256=${ROOT_SOFIE_HASH}

if(WIN32)
  # The DLL goes to bin, like the ones of the ROOT libraries
  set(ROOT_SOFIE_LIBRARY ${CMAKE_BINARY_DIR}/bin/SOFIE.dll)
  install(FILES ${ROOT_SOFIE_LIBRARY} DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT libraries)
else()
  set(ROOT_SOFIE_LIBRARY ${CMAKE_BINARY_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}SOFIE${CMAKE_SHARED_LIBRARY_SUFFIX})
  install(FILES ${ROOT_SOFIE_LIBRARY} DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries)
endif()

if(MSVC AND NOT CMAKE_GENERATOR MATCHES Ninja)
  if(winrtdebug)
    set(ROOT_SOFIE_BUILD_COMMAND_FLAGS --config Debug)
  else()
    set(ROOT_SOFIE_BUILD_COMMAND_FLAGS --config $<IF:$<CONFIG:Debug,RelWithDebInfo>,RelWithDebInfo,Release>)
  endif()
endif()

# The sources are in the repository, so SOFIE has to be built again whenever
# they change. The build step depends on them, instead of running on every build
# (BUILD_ALWAYS), so that building ROOT again when nothing changed does nothing.
# The CMakeLists.txt of SOFIE lists all its sources, so a new source changes it:
# CMake then runs again, and finds the new source.
file(GLOB_RECURSE ROOT_SOFIE_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/sofie/inc/*
  ${CMAKE_CURRENT_SOURCE_DIR}/sofie/src/*
)
list(APPEND ROOT_SOFIE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/sofie/CMakeLists.txt)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/sofie/CMakeLists.txt)

ExternalProject_Add(
  BUILTIN_SOFIE
  PREFIX ${CMAKE_BINARY_DIR}/builtins/SOFIE-prefix
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sofie
  CMAKE_ARGS -G ${CMAKE_GENERATOR}
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_CXX_FLAGS=${ROOT_EXTERNAL_CXX_FLAGS}
             -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
             -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
             -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}
             -DCMAKE_INSTALL_LIBDIR=lib
             -DCMAKE_INSTALL_BINDIR=bin
  BUILD_COMMAND ${CMAKE_COMMAND} --build . ${ROOT_SOFIE_BUILD_COMMAND_FLAGS}
  INSTALL_COMMAND ${CMAKE_COMMAND} --install . ${ROOT_SOFIE_BUILD_COMMAND_FLAGS}
  LOG_CONFIGURE 1
  LOG_BUILD 1
  LOG_INSTALL 1
  LOG_OUTPUT_ON_FAILURE 1
  BUILD_BYPRODUCTS ${ROOT_SOFIE_LIBRARY}
)
ExternalProject_Add_StepDependencies(BUILTIN_SOFIE build ${ROOT_SOFIE_SOURCES})
