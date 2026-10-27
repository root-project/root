# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# **PLEASE UPDATE ALSO THE FOLLOWING LINE WHEN UPDATING THE VERSION**
# 3 Jul 2026, https://github.com/libsdl-org/libtiff/releases/tag/v4.7.2
set(ROOT_LIBTIFF_VERSION 4.7.2)
set(ROOT_LIBTIFF_HASH "dd030ae5d6483033bd88d840fc97e36adc3022b844d9d896a3b5a50f2d660d6f")

set(ROOT_LIBTIFF_PREFIX ${CMAKE_BINARY_DIR}/builtins/LIBTIFF-prefix)
if(MSVC)
  if(winrtdebug)
    set(LIBTIFF_POSTFIX d)
  endif()
  if(NOT CMAKE_GENERATOR MATCHES Ninja)
    if(winrtdebug)
      set(ROOT_LIBTIFF_BUILD_COMMAND_FLAGS "--config Debug")
    else()
      set(ROOT_LIBTIFF_BUILD_COMMAND_FLAGS "--config $<IF:$<CONFIG:Debug,RelWithDebInfo>,RelWithDebInfo,Release>")
    endif()
  endif()
  set(ROOT_LIBTIFF_LIBRARY ${ROOT_LIBTIFF_PREFIX}/lib/tiff${LIBTIFF_POSTFIX}${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  set(ROOT_LIBTIFF_LIBRARY ${ROOT_LIBTIFF_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}tiff${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

# If zlib is not builtin, the string will be empty, nothing special to be 
# done: we rely on libtiff CMake to find zlib.
# If zlib is builtin, we need to direct the builtin libtiff to it.
if(builtin_zlib)
  set(ZLIB_ROOT_OPTION "-DZLIB_ROOT=${ROOT_ZLIB_PREFIX}")
endif()
if(builtin_zstd)
  set(ZSTD_ROOT_OPTION "-DZSTD_ROOT=${ROOT_ZSTD_PREFIX}")
endif()
if(builtin_lzma)
  set(LZMA_ROOT_OPTION "-DLIBLZMA_ROOT=${ROOT_LZMA_PREFIX}")
endif()
if(builtin_jpeg)
  set(JPEG_ROOT_OPTION "-Dlibjpeg-turbo_ROOT=${ROOT_LIBJPEG_PREFIX}")
endif()

ExternalProject_Add(
    BUILTIN_LIBTIFF
    PREFIX ${ROOT_LIBTIFF_PREFIX}
    URL https://root.cern/download/libtiff-${ROOT_LIBTIFF_VERSION}.tar.gz
    URL_HASH SHA256=${ROOT_LIBTIFF_HASH}
    CMAKE_ARGS -G ${CMAKE_GENERATOR}
    -DCMAKE_POLICY_VERSION_MINIMUM=3.12
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DCMAKE_INSTALL_LIBDIR=<INSTALL_DIR>/lib
    -DCMAKE_INSTALL_BINDIR=<INSTALL_DIR>/lib
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    -DCMAKE_C_VISIBILITY_PRESET=hidden
    -DBUILD_SHARED_LIBS=OFF
    -Dtiff-contrib=OFF
    -Dtiff-docs=OFF
    -Dtiff-opengl=OFF # if ON, link below to GL and dependency to builtin
    -Dtiff-tests=OFF
    -Dtiff-tools=OFF
    -Dcxx=OFF # We do not need library tiffxx.a (C++ stream API library)
    -Djpeg=ON
    -Dlerc=OFF
    -Dlibdeflate=OFF # if ON or unset, then you need to target link libs interface below
    -Dpixarlog=OFF
    -Dwebp=OFF
    ${ZLIB_ROOT_OPTION}
    ${ZSTD_ROOT_OPTION}
    ${LZMA_ROOT_OPTION}
    ${JPEG_ROOT_OPTION}
    BUILD_COMMAND ${CMAKE_COMMAND} --build . ${ROOT_LIBTIFF_BUILD_COMMAND_FLAGS}
    INSTALL_COMMAND ${CMAKE_COMMAND} --build . ${ROOT_LIBTIFF_BUILD_COMMAND_FLAGS} --target install
    LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 LOG_OUTPUT_ON_FAILURE 1
    BUILD_BYPRODUCTS ${ROOT_LIBTIFF_LIBRARY}
    TIMEOUT 600
)

file(MAKE_DIRECTORY ${ROOT_LIBTIFF_PREFIX}/include)
add_library(TIFF::TIFF IMPORTED STATIC GLOBAL)
set_target_properties(TIFF::TIFF PROPERTIES
    IMPORTED_LOCATION ${ROOT_LIBTIFF_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${ROOT_LIBTIFF_PREFIX}/include
)
add_dependencies(TIFF::TIFF BUILTIN_LIBTIFF)

if(builtin_zlib)
    add_dependencies(BUILTIN_LIBTIFF BUILTIN_ZLIB)
endif()
target_link_libraries(TIFF::TIFF INTERFACE ZLIB::ZLIB)
if(builtin_zstd)
    add_dependencies(BUILTIN_LIBTIFF BUILTIN_ZSTD)
endif()
target_link_libraries(TIFF::TIFF INTERFACE ZSTD::ZSTD)
if(builtin_lzma)
    add_dependencies(BUILTIN_LIBTIFF BUILTIN_LZMA)
endif()
target_link_libraries(TIFF::TIFF INTERFACE LibLZMA::LibLZMA)
if(builtin_jpeg)
    add_dependencies(BUILTIN_LIBTIFF BUILTIN_LIBJPEG)
endif()
target_link_libraries(TIFF::TIFF INTERFACE JPEG::JPEG)

# Set the canonical output of find_package according to
# https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#standard-variable-names
set(TIFF_INCLUDE_DIRS ${ROOT_LIBTIFF_PREFIX}/include PARENT_SCOPE)
set(TIFF_LIBRARIES ${ROOT_LIBTIFF_LIBRARY} PARENT_SCOPE)
set(TIFF_FOUND TRUE PARENT_SCOPE)
set(TIFF_VERSION ${ROOT_LIBTIFF_VERSION} PARENT_SCOPE)
