# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# **PLEASE UPDATE ALSO THE FOLLOWING LINE WHEN UPDATING THE VERSION**
# 20 Apr 2023, https://github.com/unuran/unuran/releases/tag/unuran-1.11.0 but using preprocessed configure and make ROOTdist option within this repo and then uploading to LCG
set(ROOT_UNURAN_VERSION 1.11.0)
set(ROOT_UNURAN_URL "${lcgpackages}/unuran-${ROOT_UNURAN_VERSION}.tar.gz")
set(ROOT_UNURAN_HASH "098793854c590b4c2c7e98bc48a45408875f48c5ad47650b5fabbd3e94dd8049")
set(ROOT_UNURAN_PREFIX ${CMAKE_BINARY_DIR}/builtins/UNURAN-prefix)
set(ROOT_UNURAN_LIBRARY ${ROOT_UNURAN_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}unuran${CMAKE_STATIC_LIBRARY_SUFFIX})
set(ROOT_UNURAN_PATCH_FILE_1 "${CMAKE_CURRENT_SOURCE_DIR}/0001-allow-compilation-from-CMake.patch")
if (NOT DEFINED GIT_EXECUTABLE)
  set(GIT_EXECUTABLE "git")
endif()
# check if the patch has already been applied
if(NOT EXISTS "${ROOT_UNURAN_PREFIX}/src/BUILTIN_UNURAN/src/CMakeLists.txt")
  set(UNURAN_PATCH_COMMAND PATCH_COMMAND ${GIT_EXECUTABLE} apply --ignore-space-change --ignore-whitespace ${ROOT_UNURAN_PATCH_FILE_1})
endif()


include(ExternalProject)

# Clear cache variables set by find_package(UNURAN)
# to ensure that we use the builtin version
foreach(var UNURAN_LIBRARIES UNURAN_LIBRARY UNURAN_LIBRARY_DEBUG UNURAN_LIBRARY_RELEASE UNURAN_FOUND UNURAN_VERSION UNURAN_INCLUDE_DIR UNURAN_LIBRARY UNURAN_LIBRARIES)
  unset(${var})
  unset(${var} CACHE)
endforeach()

if(WIN32 AND NOT CMAKE_GENERATOR MATCHES Ninja)
  if(winrtdebug)
    set(ROOT_UNURAN_BUILD_COMMAND_FLAGS "--config Debug")
  else()
    set(ROOT_UNURAN_BUILD_COMMAND_FLAGS "--config $<IF:$<CONFIG:Debug,RelWithDebInfo>,RelWithDebInfo,Release>")
  endif()
endif()

ExternalProject_Add(
  BUILTIN_UNURAN
  PREFIX ${ROOT_UNURAN_PREFIX}
  URL ${ROOT_UNURAN_URL}
  URL_HASH SHA256=${ROOT_UNURAN_HASH}
  ${UNURAN_PATCH_COMMAND}
  SOURCE_SUBDIR src
  LOG_DOWNLOAD  TRUE
# The following three are disabled to avoid CI failure in Ubuntu22 that has outdated CMake version
# See https://github.com/root-project/root/issues/16733
#  LOG_CONFIGURE TRUE
#  LOG_BUILD     TRUE
#  LOG_INSTALL   TRUE
  LOG_OUTPUT_ON_FAILURE TRUE
  CMAKE_ARGS -G ${CMAKE_GENERATOR}
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
  -DCMAKE_INSTALL_LIBDIR=<INSTALL_DIR>/lib
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DCMAKE_C_VISIBILITY_PRESET=hidden
  # -DGSL_LIBRARIES=${GSL_LIBRARIES}
  -DROOT_ARCHITECTURE=${ROOT_ARCHITECTURE}
  -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
  BUILD_COMMAND ${CMAKE_COMMAND} --build . ${ROOT_UNURAN_BUILD_COMMAND_FLAGS}
  INSTALL_COMMAND ${CMAKE_COMMAND} --build . ${ROOT_UNURAN_BUILD_COMMAND_FLAGS} --target install
  BUILD_BYPRODUCTS
    ${ROOT_UNURAN_LIBRARY}
  TIMEOUT 600
)

set(ROOT_UNURAN_INCLUDE_DIR ${ROOT_UNURAN_PREFIX}/include)
file(MAKE_DIRECTORY ${ROOT_UNURAN_INCLUDE_DIR})
add_library(Unuran::unuran IMPORTED STATIC GLOBAL)
add_dependencies(Unuran::unuran BUILTIN_UNURAN)
set_target_properties(Unuran::unuran PROPERTIES
    IMPORTED_LOCATION ${ROOT_UNURAN_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${ROOT_UNURAN_INCLUDE_DIR})
# target_compile_options(Unuran::unuran INTERFACE 
# Set the canonical output of find_package according to
# https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#standard-variable-names
set(Unuran_INCLUDE_DIRS ${ROOT_UNURAN_INCLUDE_DIR} PARENT_SCOPE)
set(Unuran_LIBRARIES ${ROOT_UNURAN_LIBRARY} PARENT_SCOPE)
set(Unuran_FOUND TRUE PARENT_SCOPE)
set(Unuran_VERSION ${ROOT_UNURAN_VERSION} PARENT_SCOPE)

# For compatibility with non-standard case FindUnuran.cmake
set(UNURAN_INCLUDE_DIRS ${ROOT_UNURAN_INCLUDE_DIR} PARENT_SCOPE)
set(UNURAN_LIBRARIES ${ROOT_UNURAN_LIBRARY} PARENT_SCOPE)
set(UNURAN_FOUND TRUE PARENT_SCOPE)
set(UNURAN_VERSION ${ROOT_UNURAN_VERSION} PARENT_SCOPE)

##---We need to disable some warnings-------------------------------------------------------------------
#string(REPLACE -Wall "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
#if(${CMAKE_CXX_COMPILER_ID} MATCHES Clang)
#  ROOT_ADD_C_FLAG(CMAKE_C_FLAGS -Wno-parentheses-equality)
#  ROOT_ADD_C_FLAG(CMAKE_C_FLAGS -Wno-deprecated-non-prototype)
#endif()
#if(${CMAKE_CXX_COMPILER_ID} MATCHES GNU)
#  ROOT_ADD_C_FLAG(CMAKE_C_FLAGS -Wno-maybe-uninitialized)
#  ROOT_ADD_C_FLAG(CMAKE_C_FLAGS -Wno-alloc-size-larger-than)
#
#endif()
