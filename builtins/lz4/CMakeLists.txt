# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# **PLEASE UPDATE ALSO THE FOLLOWING LINE WHEN UPDATING THE VERSION**
# 22 July 2024, https://github.com/lz4/lz4/releases/tag/v1.10.0
set(ROOT_LZ4_VERSION 1.10.0)
set(ROOT_LZ4_HASH "537512904744b35e232912055ccf8ec66d768639ff3abe5788d90d792ec5f48b")
# The sources come w/o CMakeList.txt as of version 1.10.0. We add ours, it's very simple.
set(ROOT_LZ4_PATCH_FILE ${CMAKE_CURRENT_SOURCE_DIR}/lz4_add_cmakelists.patch)

set(LZ4_VERSION ${ROOT_LZ4_VERSION} CACHE INTERNAL "" FORCE) # used in roottest

set(ROOT_LZ4_PREFIX ${CMAKE_BINARY_DIR}/builtins/LZ4-prefix)

set(LIBLZ4_LIBRARIES ${ROOT_LZ4_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}lz4-static${CMAKE_STATIC_LIBRARY_SUFFIX})

if(MSVC)
    # In the MSVC case, we forward the default C++ configuration flags because
    # they are changed by ROOT in SetUpWindows.cmake.
    set(lz4_extra_cmake_args
        -DCMAKE_CXXFLAGS_RELWITHDEBINFO=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
        -DCMAKE_CXXFLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
        -DCMAKE_CXXFLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG})
endif()

ExternalProject_Add(
    LZ4
    URL https://lcgpackages.web.cern.ch/tarFiles/sources/lz4-${ROOT_LZ4_VERSION}.tar.gz
    URL_HASH SHA256=${ROOT_LZ4_HASH}
    PATCH_COMMAND git apply ${ROOT_LZ4_PATCH_FILE}
    PREFIX ${ROOT_LZ4_PREFIX}
    CMAKE_ARGS -G ${CMAKE_GENERATOR} 
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DCMAKE_INSTALL_LIBDIR=<INSTALL_DIR>/lib
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    ${lz4_extra_cmake_args}
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config $<CONFIG> 
    INSTALL_COMMAND ${CMAKE_COMMAND} --install . --config $<CONFIG>
    LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 LOG_OUTPUT_ON_FAILURE 1
    BUILD_IN_SOURCE 1
    BUILD_BYPRODUCTS ${LIBLZ4_LIBRARIES}
    TIMEOUT 600
)


set(LIBLZ4_INCLUDE_DIR ${ROOT_LZ4_PREFIX}/include)
file(MAKE_DIRECTORY ${LIBLZ4_INCLUDE_DIR})

add_library(LibLZ4 STATIC IMPORTED GLOBAL)
add_library(LZ4::LZ4 ALIAS LibLZ4)
target_include_directories(LibLZ4 INTERFACE ${LIBLZ4_INCLUDE_DIR})
set_target_properties(LibLZ4 PROPERTIES IMPORTED_LOCATION ${LIBLZ4_LIBRARIES})
