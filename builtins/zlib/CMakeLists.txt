# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# **PLEASE UPDATE ALSO THE FOLLOWING LINE WHEN UPDATING THE VERSION**
# 3 Feb 2026, https://github.com/zlib-ng/zlib-ng/releases/tag/2.3.3
set(ROOT_ZLIB_VERSION 2.3.3)
set(ROOT_ZLIB_HASH "f9c65aa9c852eb8255b636fd9f07ce1c406f061ec19a2e7d508b318ca0c907d1")

set(ROOT_ZLIB_PREFIX ${CMAKE_BINARY_DIR}/builtins/ZLIB-prefix PARENT_SCOPE)
set(ROOT_ZLIB_PREFIX ${CMAKE_BINARY_DIR}/builtins/ZLIB-prefix)

# Here we need two cases because the library has two different names on Linux/macOS and Windows
if(MSVC)
     set(ROOT_ZLIB_LIBRARY ${ROOT_ZLIB_PREFIX}/lib/zlibstatic${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
     set(ROOT_ZLIB_LIBRARY ${ROOT_ZLIB_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}z${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

if(MSVC)
    # In the MSVC case, we forward the default C++ configuration flags because
    # they are changed by ROOT in SetUpWindows.cmake.
    set(zlib_extra_cmake_args
        -DCMAKE_CXXFLAGS_RELWITHDEBINFO=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
        -DCMAKE_CXXFLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
        -DCMAKE_CXXFLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG})
endif()

ExternalProject_Add(
    BUILTIN_ZLIB
    URL https://lcgpackages.web.cern.ch/lcgpackages/tarFiles/sources/zlib-ng-${ROOT_ZLIB_VERSION}.tar.gz
    URL_HASH SHA256=${ROOT_ZLIB_HASH}
    PREFIX ${ROOT_ZLIB_PREFIX}
    CMAKE_ARGS -G ${CMAKE_GENERATOR} 
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DCMAKE_INSTALL_LIBDIR=<INSTALL_DIR>/lib
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    -DCMAKE_C_VISIBILITY_PRESET=hidden
    -BUILD_TESTING=OFF
    -DWITH_GTEST=OFF
    -DBUILD_SHARED_LIBS=OFF
    -DZLIB_COMPAT=ON
    ${zlib_extra_cmake_args}
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config $<CONFIG>
    INSTALL_COMMAND ${CMAKE_COMMAND} --install . --config $<CONFIG>
    LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 LOG_OUTPUT_ON_FAILURE 1
    BUILD_BYPRODUCTS ${ROOT_ZLIB_LIBRARY}
    TIMEOUT 600
)

set(ZLIB_INCLUDE_DIR ${ROOT_ZLIB_PREFIX}/include)
file(MAKE_DIRECTORY ${ZLIB_INCLUDE_DIR})
set_property(GLOBAL APPEND PROPERTY ROOT_BUILTIN_TARGETS ZLIB::ZLIB)

add_library(ZLIB STATIC IMPORTED GLOBAL)
add_library(ZLIB::ZLIB ALIAS ZLIB)
target_include_directories(ZLIB INTERFACE ${ZLIB_INCLUDE_DIR})
set_target_properties(ZLIB PROPERTIES IMPORTED_LOCATION ${ROOT_ZLIB_LIBRARY})

add_dependencies(ZLIB BUILTIN_ZLIB)

# Set the canonical output of find_package according to
# https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#standard-variable-names
set(ZLIB_INCLUDE_DIRS ${ZLIB_INCLUDE_DIR} PARENT_SCOPE)
set(ZLIB_LIBRARIES ${ROOT_ZLIB_LIBRARY} PARENT_SCOPE)
set(ZLIB_FOUND TRUE PARENT_SCOPE)