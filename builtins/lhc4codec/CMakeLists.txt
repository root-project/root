# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# Build lhc4codec as part of ROOT. Upstream:
#   https://gitlab.cern.ch/apeters/lhc4codec.git
#
# If LHC4CODEC_SOURCE_DIR is set, that checkout is built via add_subdirectory
# (offline / development). Otherwise the source is fetched with ExternalProject
# (clad-style git clone; requires network and GitLab credentials).

include(ExternalProject)

# **PLEASE UPDATE ALSO THE FOLLOWING LINE WHEN UPDATING THE VERSION**
# 16 Sep 2026, https://gitlab.cern.ch/apeters/lhc4codec.git @ 922d272 (packaging/CI, bzip3 pin)
set(ROOT_LHC4CODEC_VERSION 2.3.0)
set(ROOT_LHC4CODEC_GIT_REPOSITORY "https://gitlab.cern.ch/apeters/lhc4codec.git")
set(ROOT_LHC4CODEC_GIT_TAG "922d2729da39cb393c4f109d393ffbaca3424a57")

set(_lhc4codec_build_args
  -DLHC4CODEC_BUILD_CLI=OFF
  -DLHC4CODEC_BUILD_TESTS=OFF
  -DLHC4CODEC_METAL=OFF
  -DLHC4CODEC_HIP=OFF
  -DLHC4CODEC_NATIVE=OFF
  -DLHC4CODEC_LTO=OFF
  -DLHC4CODEC_ZSTD=AUTO
  -DLHC4CODEC_LZMA=AUTO
  -DLHC4CODEC_BZIP3=BUNDLED
  -DCMAKE_CXX_STANDARD=20
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
)

if(LHC4CODEC_SOURCE_DIR)
  if(NOT EXISTS "${LHC4CODEC_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "LHC4CODEC_SOURCE_DIR=\"${LHC4CODEC_SOURCE_DIR}\" does not contain a CMake project")
  endif()

  message(STATUS "Building builtin lhc4codec from ${LHC4CODEC_SOURCE_DIR}")

  set(LHC4CODEC_BUILD_CLI OFF CACHE BOOL "" FORCE)
  set(LHC4CODEC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(LHC4CODEC_METAL OFF CACHE BOOL "" FORCE)
  set(LHC4CODEC_HIP OFF CACHE BOOL "" FORCE)
  set(LHC4CODEC_NATIVE OFF CACHE BOOL "" FORCE)
  set(LHC4CODEC_LTO OFF CACHE BOOL "" FORCE)
  set(LHC4CODEC_ZSTD AUTO CACHE STRING "" FORCE)
  set(LHC4CODEC_LZMA AUTO CACHE STRING "" FORCE)
  set(LHC4CODEC_BZIP3 BUNDLED CACHE STRING "" FORCE)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  add_subdirectory(${LHC4CODEC_SOURCE_DIR} ${CMAKE_BINARY_DIR}/builtins/lhc4codec-build EXCLUDE_FROM_ALL)

  set(LHC4CODEC_INCLUDE_DIRS ${LHC4CODEC_SOURCE_DIR}/include)
  set(LHC4CODEC_LIBRARIES lhc4codec_static)

  if(NOT TARGET LHC4CODEC::LHC4CODEC)
    add_library(LHC4CODEC::LHC4CODEC ALIAS lhc4codec_static)
  endif()
else()
  message(STATUS "Downloading and building lhc4codec ${ROOT_LHC4CODEC_VERSION} from "
                  "${ROOT_LHC4CODEC_GIT_REPOSITORY} (${ROOT_LHC4CODEC_GIT_TAG})")

  set(ROOT_LHC4CODEC_PREFIX ${CMAKE_BINARY_DIR}/builtins/lhc4codec-prefix)
  set(ROOT_LHC4CODEC_SOURCE_DIR ${ROOT_LHC4CODEC_PREFIX}/src/BUILTIN_LHC4CODEC)
  set(ROOT_LHC4CODEC_BUILD_DIR ${ROOT_LHC4CODEC_PREFIX}/src/BUILTIN_LHC4CODEC-build)
  # Staged headers live outside the git tree so the path exists at configure time
  # (CMake validates INTERFACE_INCLUDE_DIRECTORIES on imported targets).
  set(ROOT_LHC4CODEC_INCLUDE_DIR ${ROOT_LHC4CODEC_PREFIX}/include)
  set(ROOT_LHC4CODEC_LIBRARY
      ${ROOT_LHC4CODEC_BUILD_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}lhc4codec${CMAKE_STATIC_LIBRARY_SUFFIX})
  set(ROOT_LHC4CODEC_BZIP3_LIBRARY
      ${ROOT_LHC4CODEC_BUILD_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}lhc4_bzip3_bundled${CMAKE_STATIC_LIBRARY_SUFFIX})
  file(MAKE_DIRECTORY ${ROOT_LHC4CODEC_INCLUDE_DIR})

  if(MSVC)
    set(_lhc4codec_build_type Release)
    if(winrtdebug)
      set(_lhc4codec_build_type Debug)
    endif()
    if(NOT CMAKE_GENERATOR MATCHES Ninja)
      set(_lhc4codec_build_flags --config ${_lhc4codec_build_type})
    endif()
    set(_lhc4codec_extra_cmake_args
      -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
      -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
      -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG})
  else()
    set(_lhc4codec_build_type Release)
    set(_lhc4codec_build_flags "")
    set(_lhc4codec_extra_cmake_args "")
  endif()

  ExternalProject_Add(
    BUILTIN_LHC4CODEC
    GIT_REPOSITORY ${ROOT_LHC4CODEC_GIT_REPOSITORY}
    GIT_TAG ${ROOT_LHC4CODEC_GIT_TAG}
    GIT_SUBMODULES third_party/bzip3
    PREFIX ${ROOT_LHC4CODEC_PREFIX}
    UPDATE_COMMAND ""
    CMAKE_ARGS -G ${CMAKE_GENERATOR}
      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      -DCMAKE_BUILD_TYPE=${_lhc4codec_build_type}
      ${_lhc4codec_build_args}
      ${_lhc4codec_extra_cmake_args}
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> ${_lhc4codec_build_flags}
                  --target lhc4codec_static lhc4_bzip3_bundled
    INSTALL_COMMAND ""
    LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_OUTPUT_ON_FAILURE 1
    BUILD_BYPRODUCTS ${ROOT_LHC4CODEC_LIBRARY} ${ROOT_LHC4CODEC_BZIP3_LIBRARY}
    TIMEOUT 600
  )

  # Copy public headers after the git clone, before the nested configure/build.
  ExternalProject_Add_Step(BUILTIN_LHC4CODEC stage-headers
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${ROOT_LHC4CODEC_INCLUDE_DIR}/lhc4codec
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            <SOURCE_DIR>/include/lhc4codec
            ${ROOT_LHC4CODEC_INCLUDE_DIR}/lhc4codec
    DEPENDEES download
    DEPENDERS configure
  )

  add_library(LHC4CODEC::LHC4CODEC IMPORTED STATIC GLOBAL)
  add_dependencies(LHC4CODEC::LHC4CODEC BUILTIN_LHC4CODEC)
  set_target_properties(LHC4CODEC::LHC4CODEC PROPERTIES
    IMPORTED_LOCATION ${ROOT_LHC4CODEC_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${ROOT_LHC4CODEC_INCLUDE_DIR}
    INTERFACE_LINK_LIBRARIES ${ROOT_LHC4CODEC_BZIP3_LIBRARY}
  )

  set(LHC4CODEC_INCLUDE_DIRS ${ROOT_LHC4CODEC_INCLUDE_DIR})
  set(LHC4CODEC_LIBRARIES ${ROOT_LHC4CODEC_LIBRARY})
endif()

set(LHC4CODEC_FOUND TRUE)
set(LHC4CODEC_VERSION ${ROOT_LHC4CODEC_VERSION})

set(LHC4CODEC_INCLUDE_DIRS ${LHC4CODEC_INCLUDE_DIRS} PARENT_SCOPE)
set(LHC4CODEC_LIBRARIES ${LHC4CODEC_LIBRARIES} PARENT_SCOPE)
set(LHC4CODEC_FOUND ${LHC4CODEC_FOUND} PARENT_SCOPE)
set(LHC4CODEC_VERSION ${LHC4CODEC_VERSION} PARENT_SCOPE)
