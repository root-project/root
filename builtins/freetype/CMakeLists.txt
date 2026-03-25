# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# **PLEASE UPDATE ALSO THE FOLLOWING LINE WHEN UPDATING THE VERSION**
# 22 Mar 2026, https://github.com/freetype/freetype/releases/tag/VER-2-14-3
set(FREETYPE_VERSION 2.14.3)
set(FREETYPE_HASH "e61b31ab26358b946e767ed7eb7f4bb2e507da1cfefeb7a8861ace7fd5c899a1")

set(FREETYPE_PREFIX ${CMAKE_BINARY_DIR}/builtins/FREETYPE-prefix)

# The CMake of FreeType will call the static library differently if the CMAKE_BUILD_TYPE is Debug, bu adding a 
# "d" to the name of the file. This is unusual if compared to the behaviour of other ROOT builtins, where the name
# changes only between Linux/macOS and Windows.
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(FREETYPE_LIBRARY ${FREETYPE_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}freetyped${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
    set(FREETYPE_LIBRARY ${FREETYPE_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}freetype${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

ExternalProject_Add(
    BUILTIN_FREETYPE
    URL https://lcgpackages.web.cern.ch/tarFiles/sources/freetype-${FREETYPE_VERSION}.tar.gz
    URL_HASH SHA256=${FREETYPE_HASH}
    PREFIX ${FREETYPE_PREFIX}
    CMAKE_ARGS -G ${CMAKE_GENERATOR}
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=${FREETYPE_PREFIX}
    -DFT_DISABLE_BZIP2=TRUE
    -DFT_DISABLE_BROTLI=TRUE
    -DFT_DISABLE_PNG=TRUE
    -DFT_DISABLE_HARFBUZZ=TRUE
    -DBUILD_SHARED_LIBS=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    -DCMAKE_C_VISIBILITY_PRESET=hidden
    -DZLIB_LIBRARY=${ZLIB_LIBRARIES}
    -DZLIB_INCLUDE_DIR=${ZLIB_INCLUDE_DIRS}
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config $<CONFIG>
    INSTALL_COMMAND ${CMAKE_COMMAND} --install . --config $<CONFIG>
    LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 LOG_OUTPUT_ON_FAILURE 1
    BUILD_IN_SOURCE 0
    BUILD_BYPRODUCTS ${FREETYPE_LIBRARY}
    TIMEOUT 600
)

add_library(Freetype::Freetype IMPORTED STATIC GLOBAL)

# Set the canonical output of find_package according to
# https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#standard-variable-names
set(FREETYPE_INCLUDE_DIRS ${FREETYPE_PREFIX}/include/freetype2 PARENT_SCOPE)
set(FREETYPE_LIBRARIES ${FREETYPE_LIBRARY} PARENT_SCOPE)
set(Freetype_FOUND TRUE PARENT_SCOPE)
set(Freetype_VERSION ${FREETYPE_VERSION} PARENT_SCOPE)

if(builtin_zlib)
    add_dependencies(BUILTIN_FREETYPE BUILTIN_ZLIB)
endif()
