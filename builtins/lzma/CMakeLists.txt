# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# **PLEASE UPDATE ALSO THE FOLLOWING LINE WHEN UPDATING THE VERSION**
# 17 Dec 2025, https://github.com/tukaani-project/xz/releases/tag/v5.8.2
set(ROOT_LZMA_VERSION 5.8.2)
set(ROOT_LZMA_HASH "ce09c50a5962786b83e5da389c90dd2c15ecd0980a258dd01f70f9e7ce58a8f1")

set(LZMA_TARGET LZMA)
set(ROOT_LZMA_PREFIX ${CMAKE_BINARY_DIR}/builtins/LZMA-prefix)

set(LIBLZMA_LIBRARIES ${ROOT_LZMA_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}lzma${CMAKE_STATIC_LIBRARY_SUFFIX})

if(MSVC)
    # In the MSVC case, we forward the default C++ configuration flags because
    # they are changed by ROOT in SetUpWindows.cmake.
    set(lzma_extra_cmake_args
        -DCMAKE_CXXFLAGS_RELWITHDEBINFO=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
        -DCMAKE_CXXFLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
        -DCMAKE_CXXFLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG})
    if(NOT CMAKE_GENERATOR MATCHES Ninja)
        if(winrtdebug)
            set(ROOT_LZMA_BUILD_COMMAND_FLAGS "--config Debug")
        else()
            set(ROOT_LZMA_BUILD_COMMAND_FLAGS "--config $<IF:$<CONFIG:Debug,RelWithDebInfo>,RelWithDebInfo,Release>")
        endif()
    endif()
else()
    if(CMAKE_OSX_SYSROOT)
        set(liblzma_flags "-isysroot\ ${CMAKE_OSX_SYSROOT}")
    endif()
    set(lzma_extra_cmake_args
        -DCMAKE_C_FLAGS=${liblzma_flags}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
endif()

ExternalProject_Add(
    LZMA
    URL ${lcgpackages}/xz-${ROOT_LZMA_VERSION}.tar.gz
    URL_HASH SHA256=${ROOT_LZMA_HASH}
    PREFIX ${ROOT_LZMA_PREFIX}
    CMAKE_ARGS -G ${CMAKE_GENERATOR} 
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DCMAKE_INSTALL_LIBDIR=<INSTALL_DIR>/lib
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    -DCMAKE_C_VISIBILITY_PRESET=hidden
    -DBUILD_SHARED_LIBS=OFF
    -DENABLE_SCRIPTS=OFF
    -DXZ_TOOL=OFF
    -DXZDEC=OFF
    -DLZMADEC=OFF
    -DLZMAINFO=OFF
    ${lzma_extra_cmake_args}
    BUILD_COMMAND ${CMAKE_COMMAND} --build . ${ROOT_LZMA_BUILD_COMMAND_FLAGS} --target liblzma
    INSTALL_COMMAND ${CMAKE_COMMAND} --install . ${ROOT_LZMA_BUILD_COMMAND_FLAGS} --component liblzma_Development
    LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 LOG_OUTPUT_ON_FAILURE 1
    BUILD_IN_SOURCE 1
    BUILD_BYPRODUCTS ${LIBLZMA_LIBRARIES}
    TIMEOUT 600
)


set(LIBLZMA_INCLUDE_DIR ${ROOT_LZMA_PREFIX}/include)
file(MAKE_DIRECTORY ${LIBLZMA_INCLUDE_DIR})

add_library(LibLZMA STATIC IMPORTED GLOBAL)
add_library(LibLZMA::LibLZMA ALIAS LibLZMA)
target_include_directories(LibLZMA INTERFACE ${LIBLZMA_INCLUDE_DIR})
set_target_properties(LibLZMA PROPERTIES IMPORTED_LOCATION ${LIBLZMA_LIBRARIES})

# Set the canonical output of find_package according to
# https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#standard-variable-names
set(LIBLZMA_INCLUDE_DIRS ${LibLZMA_INCLUDE_DIR} PARENT_SCOPE)
set(LIBLZMA_LIBRARIES ${LIBLZMA_LIBRARIES} PARENT_SCOPE)
set(LibLZMA_FOUND TRUE PARENT_SCOPE)
set(LibLZMA_VERSION ${ROOT_LZMA_VERSION} PARENT_SCOPE)
