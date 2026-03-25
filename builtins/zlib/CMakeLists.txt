# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# **PLEASE UPDATE ALSO THE FOLLOWING LINE WHEN UPDATING THE VERSION**
# 17 Feb 2026, https://github.com/madler/zlib/releases/tag/v1.3.2
set(ROOT_ZLIB_VERSION 1.3.2)
set(ROOT_ZLIB_HASH "bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16")

set(ROOT_ZLIB_PREFIX ${CMAKE_BINARY_DIR}/builtins/ZLIB-prefix PARENT_SCOPE)
set(ROOT_ZLIB_PREFIX ${CMAKE_BINARY_DIR}/builtins/ZLIB-prefix)

# Here we need two cases because the library has two different names on Linux/macOS and Windows
if(MSVC)
     set(ROOT_ZLIB_LIBRARY ${ROOT_ZLIB_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}zs${CMAKE_STATIC_LIBRARY_SUFFIX})
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
    URL https://lcgpackages.web.cern.ch/lcgpackages/tarFiles/sources/zlib-${ROOT_ZLIB_VERSION}.tar.gz
    URL_HASH SHA256=${ROOT_ZLIB_HASH}
    PREFIX ${ROOT_ZLIB_PREFIX}
    CMAKE_ARGS -G ${CMAKE_GENERATOR}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DCMAKE_INSTALL_LIBDIR=<INSTALL_DIR>/lib
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    -DCMAKE_C_VISIBILITY_PRESET=hidden
    -DZLIB_BUILD_TESTING=OFF
    -DZLIB_BUILD_SHARED=OFF
    ${zlib_extra_cmake_args}
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config $<CONFIG>
    INSTALL_COMMAND ${CMAKE_COMMAND} --install . --config $<CONFIG>
    LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 LOG_OUTPUT_ON_FAILURE 1 LOG_TEST 1
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
set(ZLIB_VERSION ${ROOT_ZLIB_VERSION} PARENT_SCOPE)
