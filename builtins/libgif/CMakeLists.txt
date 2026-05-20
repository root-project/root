# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# **PLEASE UPDATE ALSO THE FOLLOWING LINE WHEN UPDATING THE VERSION**
# 19 Feb 2024, https://sourceforge.net/projects/giflib/files/giflib-5.x/
set(ROOT_LIBGIF_VERSION 5.2.2)
set(ROOT_LIBGIF_HASH "302b2758bbdb8b5c63eafad21b15a53bb7aef7e5d4d278fce170c160364082b5")
# The sources come w/o CMakeList.txt as of version 5.2.2. We add ours, it's very simple.
set(ROOT_LIBGIF_PATCH_FILE ${CMAKE_CURRENT_SOURCE_DIR}/libgif_add_cmakelists.patch)

set(ROOT_LIBGIF_PREFIX ${CMAKE_BINARY_DIR}/builtins/LIBGIF-prefix)

set(ROOT_LIBGIF_LIBRARY ${ROOT_LIBGIF_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gif-static${CMAKE_STATIC_LIBRARY_SUFFIX})

if(WIN32 AND NOT CMAKE_GENERATOR MATCHES Ninja)
    if(winrtdebug)
        set(ROOT_LIBGIF_BUILD_COMMAND_FLAGS "--config Debug")
    else()
        set(ROOT_LIBGIF_BUILD_COMMAND_FLAGS "--config $<IF:$<CONFIG:Debug,RelWithDebInfo>,RelWithDebInfo,Release>")
    endif()
endif()

# flag required only on Windows, on old alma paltforms it make troubles
if(WIN32)
  set(patch_args --ignore-whitespace)
endif()

ExternalProject_Add(
    BUILTIN_LIBGIF
    PREFIX ${ROOT_LIBGIF_PREFIX}
    URL ${lcgpackages}/giflib-${ROOT_LIBGIF_VERSION}.tar.gz
    URL_HASH SHA256=${ROOT_LIBGIF_HASH}
    PATCH_COMMAND git apply ${patch_args} ${ROOT_LIBGIF_PATCH_FILE}
    CMAKE_ARGS -G ${CMAKE_GENERATOR}
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    -DCMAKE_C_VISIBILITY_PRESET=hidden
    BUILD_COMMAND ${CMAKE_COMMAND} --build . ${ROOT_LIBGIF_BUILD_COMMAND_FLAGS}
    INSTALL_COMMAND ${CMAKE_COMMAND} --build . ${ROOT_LIBGIF_BUILD_COMMAND_FLAGS} --target install
    LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 LOG_OUTPUT_ON_FAILURE 1
    BUILD_BYPRODUCTS ${ROOT_LIBGIF_LIBRARY}
    TIMEOUT 600
)

file(MAKE_DIRECTORY ${ROOT_LIBGIF_PREFIX}/include)
add_library(GIF::GIF IMPORTED STATIC GLOBAL)
set_target_properties(GIF::GIF PROPERTIES
    IMPORTED_LOCATION ${ROOT_LIBGIF_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${ROOT_LIBGIF_PREFIX}/include
)
add_dependencies(GIF::GIF BUILTIN_LIBGIF)

# Set the canonical output of find_package according to
# https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#standard-variable-names
set(GIF_INCLUDE_DIRS ${ROOT_LIBGIF_PREFIX}/include PARENT_SCOPE)
set(GIF_LIBRARIES ${ROOT_LIBGIF_LIBRARY} PARENT_SCOPE)
set(GIF_FOUND TRUE PARENT_SCOPE)
set(GIF_VERSION ${ROOT_LIBGIF_VERSION} PARENT_SCOPE)
