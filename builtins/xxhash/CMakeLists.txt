# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# **PLEASE UPDATE ALSO THE FOLLOWING LINE WHEN UPDATING THE VERSION**
# 30 Dec 2024, https://github.com/Cyan4973/xxHash/releases/tag/v0.8.3
set(ROOT_XXHASH_VERSION 0.8.3)
set(ROOT_XXHASH_HASH "aae608dfe8213dfd05d909a57718ef82f30722c392344583d3f39050c7f29a80")
# The official sources come w/o CMakeList.txt as of version 0.8.3. We add ours for the moment.
# FIXME: This patch should be removed as soon as 0.8.4 is released and use instead the native one
# https://github.com/Cyan4973/xxHash/blob/dev/build/cmake/CMakeLists.txt
set(ROOT_XXHASH_PATCH_FILE ${CMAKE_CURRENT_SOURCE_DIR}/xxhash_add_cmakelists.patch)

set(ROOT_XXHASH_PREFIX ${CMAKE_BINARY_DIR}/builtins/XXHASH-prefix)
# check if the patch has already been applied
if(NOT EXISTS "${ROOT_XXHASH_PREFIX}/src/BUILTIN_XXHASH/CMakeLists.txt")
  set(XXHASH_PATCH_COMMAND PATCH_COMMAND ${GIT_EXECUTABLE} apply --ignore-space-change --ignore-whitespace ${ROOT_XXHASH_PATCH_FILE})
endif()

set(ROOT_XXHASH_LIBRARY ${ROOT_XXHASH_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}xxhash${CMAKE_STATIC_LIBRARY_SUFFIX}
)
if (NOT DEFINED GIT_EXECUTABLE)
  set(GIT_EXECUTABLE "git")
endif()

if(WIN32 AND NOT CMAKE_GENERATOR MATCHES Ninja)
  if(winrtdebug)
    set(ROOT_XXHASH_BUILD_COMMAND_FLAGS "--config Debug")
  else()
    set(ROOT_XXHASH_BUILD_COMMAND_FLAGS "--config $<IF:$<CONFIG:Debug,RelWithDebInfo>,RelWithDebInfo,Release>")
  endif()
endif()

ExternalProject_Add(
  BUILTIN_XXHASH
  PREFIX ${ROOT_XXHASH_PREFIX}
  URL ${lcgpackages}/xxHash-${ROOT_XXHASH_VERSION}.tar.gz
  URL_HASH SHA256=${ROOT_XXHASH_HASH}
  ${XXHASH_PATCH_COMMAND}
  CMAKE_ARGS -G ${CMAKE_GENERATOR}
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DCMAKE_C_VISIBILITY_PRESET=hidden
  BUILD_COMMAND ${CMAKE_COMMAND} --build . ${ROOT_XXHASH_BUILD_COMMAND_FLAGS}
  INSTALL_COMMAND ${CMAKE_COMMAND} --build . ${ROOT_XXHASH_BUILD_COMMAND_FLAGS} --target install
  LOG_CONFIGURE 1
  LOG_BUILD 1
  LOG_INSTALL 1
  LOG_OUTPUT_ON_FAILURE 1
  BUILD_BYPRODUCTS ${ROOT_XXHASH_LIBRARY}
  TIMEOUT 600)

file(MAKE_DIRECTORY ${ROOT_XXHASH_PREFIX}/include)
add_library(xxHash::xxHash IMPORTED STATIC GLOBAL)
add_dependencies(xxHash::xxHash BUILTIN_XXHASH)
set_target_properties(xxHash::xxHash PROPERTIES
    IMPORTED_LOCATION ${ROOT_XXHASH_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${ROOT_XXHASH_PREFIX}/include)

# Set the canonical output of find_package according to
# https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#standard-variable-names
set(xxHash_INCLUDE_DIRS ${ROOT_XXHASH_PREFIX}/include PARENT_SCOPE)
set(xxHash_LIBRARIES ${ROOT_XXHASH_LIBRARY} PARENT_SCOPE)
set(xxHash_FOUND TRUE PARENT_SCOPE)
set(xxHash_VERSION ${ROOT_XXHASH_VERSION} PARENT_SCOPE)
set_property(GLOBAL APPEND PROPERTY ROOT_BUILTIN_TARGETS BUILTIN_XXHASH)
