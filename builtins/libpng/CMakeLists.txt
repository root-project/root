# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# **PLEASE UPDATE ALSO THE FOLLOWING LINE WHEN UPDATING THE VERSION**
# Feb 2026, https://github.com/pnggroup/libpng/releases/tag/v1.6.55
set(ROOT_LIBPNG_VERSION 1.6.55)
set(ROOT_LIBPNG_HASH "4b0abab6d219e95690ebe4db7fc9aa95f4006c83baaa022373c0c8442271283d")

set(ROOT_LIBPNG_PREFIX ${CMAKE_BINARY_DIR}/builtins/LIBPNG-prefix)

# Here we need two cases because the library has two different names on Linux/macOS and Windows
if(MSVC)
    if(winrtdebug)
        set(LIBPNG_POSTFIX d)
    endif()
    if(NOT CMAKE_GENERATOR MATCHES Ninja)
        if(winrtdebug)
            set(ROOT_LIBPNG_BUILD_COMMAND_FLAGS "--config Debug")
        else()
            set(ROOT_LIBPNG_BUILD_COMMAND_FLAGS "--config $<IF:$<CONFIG:Debug,RelWithDebInfo>,RelWithDebInfo,Release>")
        endif()
    endif()
    set(ROOT_LIBPNG_LIBRARY ${ROOT_LIBPNG_PREFIX}/lib/libpng16_static${LIBPNG_POSTFIX}${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
     set(ROOT_LIBPNG_LIBRARY ${ROOT_LIBPNG_PREFIX}/lib/libpng${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

# If zlib is not builtin, the string will be empty, nothing special to be 
# done: we rely on libpng CMake to find zlib.
# If zlib is builtin, we need to direct the builtin libpng to it.
# For that, the libpng recommended way (see their CMakeLists)
# is to use the CMake ZLIB_ROOT variable, which has nothing to do with ROOT 
# despite the name: it's a standard CMake convention to direct the search of
# the find macro.
# Therefore we set it to the prefix path of the builtin zlib in ROOT's 
# build directory.
if(builtin_zlib)
  set(ZLIB_ROOT_OPTION "-DZLIB_ROOT=${ROOT_ZLIB_PREFIX}")
endif()

ExternalProject_Add(
    BUILTIN_LIBPNG
    PREFIX ${ROOT_LIBPNG_PREFIX}
    URL ${lcgpackages}/libpng-${ROOT_LIBPNG_VERSION}.tar.gz
    URL_HASH SHA256=${ROOT_LIBPNG_HASH}
    CMAKE_ARGS -G ${CMAKE_GENERATOR}
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DCMAKE_INSTALL_LIBDIR=<INSTALL_DIR>/lib
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    -DCMAKE_C_VISIBILITY_PRESET=hidden
    -DPNG_SHARED=OFF
    -DPNG_STATIC=ON
    -DZLIB_LIBRARY=${ZLIB_LIBRARIES}
    -DZLIB_INCLUDE_DIR=${ZLIB_INCLUDE_DIRS}
    ${ZLIB_ROOT_OPTION}
    BUILD_COMMAND ${CMAKE_COMMAND} --build . ${ROOT_LIBPNG_BUILD_COMMAND_FLAGS}
    INSTALL_COMMAND ${CMAKE_COMMAND} --build . ${ROOT_LIBPNG_BUILD_COMMAND_FLAGS} --target install
    LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 LOG_OUTPUT_ON_FAILURE 1
    BUILD_BYPRODUCTS ${ROOT_LIBPNG_LIBRARY}
    TIMEOUT 600
)

file(MAKE_DIRECTORY ${ROOT_LIBPNG_PREFIX}/include)
add_library(PNG::PNG IMPORTED STATIC GLOBAL)
set_target_properties(PNG::PNG PROPERTIES
    IMPORTED_LOCATION ${ROOT_LIBPNG_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${ROOT_LIBPNG_PREFIX}/include
)
add_dependencies(PNG::PNG BUILTIN_LIBPNG)

if(builtin_zlib)
    add_dependencies(BUILTIN_LIBPNG BUILTIN_ZLIB)
endif()

# Set the canonical output of find_package according to
# https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#standard-variable-names
set(PNG_INCLUDE_DIRS ${ROOT_LIBPNG_PREFIX}/include PARENT_SCOPE)
set(PNG_LIBRARIES ${ROOT_LIBPNG_LIBRARY} PARENT_SCOPE)
set(PNG_FOUND TRUE PARENT_SCOPE)
set(PNG_VERSION ${ROOT_LIBPNG_VERSION} PARENT_SCOPE)
