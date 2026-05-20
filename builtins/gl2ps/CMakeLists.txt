# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# **PLEASE UPDATE ALSO THE FOLLOWING LINE WHEN UPDATING THE VERSION**
# 30 Apr 2020, https://gitlab.onelab.info/gl2ps/gl2ps/-/releases/gl2ps_1_4_2
set(ROOT_GL2PS_VERSION 1.4.2)
set(ROOT_GL2PS_HASH "8d1c00c1018f96b4b97655482e57dcb0ce42ae2f1d349cd6d4191e7848d9ffe9")
# Backport fix for static Win32 having being overwritten by shared lib: https://gitlab.onelab.info/gl2ps/gl2ps/-/work_items/30
# We use all seven commits between 1.4.2 and before bump to 1.4.3
set(ROOT_GL2PS_PATCH_FILE ${CMAKE_CURRENT_SOURCE_DIR}/v142_post_tag_fixes.patch)

set(ROOT_GL2PS_PREFIX ${CMAKE_BINARY_DIR}/builtins/GL2PS-prefix)

set(ROOT_GL2PS_LIBRARY ${ROOT_GL2PS_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gl2ps${CMAKE_STATIC_LIBRARY_SUFFIX})

if(WIN32 AND NOT CMAKE_GENERATOR MATCHES Ninja)
  if(winrtdebug)
    set(ROOT_GL2PS_BUILD_COMMAND_FLAGS "--config Debug")
  else()
    set(ROOT_GL2PS_BUILD_COMMAND_FLAGS "--config $<IF:$<CONFIG:Debug,RelWithDebInfo>,RelWithDebInfo,Release>")
  endif()
endif()

# flag required only on Windows, on old alma paltforms it make troubles
if(WIN32)
  set(patch_args --ignore-whitespace)
endif()

ExternalProject_Add(
  BUILTIN_GL2PS
  PREFIX ${ROOT_GL2PS_PREFIX}
  URL ${lcgpackages}/gl2ps-${ROOT_GL2PS_VERSION}.tgz
  URL_HASH SHA256=${ROOT_GL2PS_HASH}
  PATCH_COMMAND git apply ${patch_args} ${ROOT_GL2PS_PATCH_FILE}
  CMAKE_ARGS -G ${CMAKE_GENERATOR}
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DCMAKE_C_VISIBILITY_PRESET=hidden
             -DCMAKE_POLICY_VERSION_MINIMUM=3.5
             -DENABLE_PNG=OFF
             -DENABLE_ZLIB=OFF
             -DBUILD_SHARED_LIBS=OFF
  BUILD_COMMAND ${CMAKE_COMMAND} --build . ${ROOT_GL2PS_BUILD_COMMAND_FLAGS}
  INSTALL_COMMAND ${CMAKE_COMMAND} --build . ${ROOT_GL2PS_BUILD_COMMAND_FLAGS} --target install
  LOG_DOWNLOAD 1
  LOG_CONFIGURE 1
  LOG_BUILD 1
  LOG_INSTALL 1
  LOG_OUTPUT_ON_FAILURE 1
  BUILD_BYPRODUCTS ${ROOT_GL2PS_LIBRARY}
  TIMEOUT 600)

file(MAKE_DIRECTORY ${ROOT_GL2PS_PREFIX}/include)
add_library(gl2ps::gl2ps IMPORTED STATIC GLOBAL)
add_dependencies(gl2ps::gl2ps BUILTIN_GL2PS)
set_target_properties(gl2ps::gl2ps PROPERTIES
    IMPORTED_LOCATION ${ROOT_GL2PS_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${ROOT_GL2PS_PREFIX}/include)

# Set the canonical output of find_package according to
# https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#standard-variable-names
set(gl2ps_INCLUDE_DIRS ${ROOT_GL2PS_PREFIX}/include PARENT_SCOPE)
set(gl2ps_LIBRARIES ${ROOT_GL2PS_LIBRARY} PARENT_SCOPE)
set(gl2ps_FOUND TRUE PARENT_SCOPE)
set(gl2ps_VERSION ${ROOT_GL2PS_VERSION} PARENT_SCOPE)
