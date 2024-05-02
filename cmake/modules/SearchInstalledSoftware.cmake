# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

#---Check for installed packages depending on the build options/components enabled --
include(CheckCXXSourceCompiles)
include(CheckIncludeFileCXX)
include(ExternalProject)
include(FindPackageHandleStandardArgs)

set(lcgpackages http://lcgpackages.web.cern.ch/lcgpackages/tarFiles/sources)
string(REPLACE "-Werror " "" ROOT_EXTERNAL_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")

macro(find_package)
  if(NOT "${ARGV0}" IN_LIST ROOT_BUILTINS)
    _find_package(${ARGV})
  endif()
endmacro()

#---On MacOSX, try to find frameworks after standard libraries or headers------------
set(CMAKE_FIND_FRAMEWORK LAST)

#---If -Dshared=Off, prefer static libraries-----------------------------------------
if(NOT shared)
  if(WINDOWS)
    message(FATAL_ERROR "Option \"shared=Off\" not supported on Windows!")
  else()
    message("Preferring static libraries.")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a;${CMAKE_FIND_LIBRARY_SUFFIXES}")
  endif()
endif()


#---Check for Cocoa/Quartz graphics backend (MacOS X only)---------------------------
if(cocoa)
  if(APPLE)
    set(x11 OFF CACHE BOOL "Disabled because cocoa requested (${x11_description})" FORCE)
    set(builtin_freetype ON CACHE BOOL "Enabled because needed for Cocoa graphics (${builtin_freetype_description})" FORCE)
    if(NOT opengl)
      message(FATAL_ERROR "Option \"cocoa=ON\" requires \"opengl=ON\"!")
    endif()
  else()
    message(STATUS "Cocoa option can only be enabled on MacOSX platform")
    set(cocoa OFF CACHE BOOL "Disabled because only available on MacOSX (${cocoa_description})" FORCE)
  endif()
endif()

#---Check for Zlib ------------------------------------------------------------------
if(NOT builtin_zlib)
  message(STATUS "Looking for ZLib")
  # Clear cache variables, or LLVM may use old values for ZLIB
  foreach(suffix FOUND INCLUDE_DIR LIBRARY LIBRARY_DEBUG LIBRARY_RELEASE)
    unset(ZLIB_${suffix} CACHE)
  endforeach()
  if(fail-on-missing)
    find_package(ZLIB REQUIRED)
  else()
    find_package(ZLIB)
    if(NOT ZLIB_FOUND)
      message(STATUS "Zlib not found. Switching on builtin_zlib option")
      set(builtin_zlib ON CACHE BOOL "Enabled because Zlib not found (${builtin_zlib_description})" FORCE)
    endif()
  endif()
endif()

if(builtin_zlib)
  list(APPEND ROOT_BUILTINS ZLIB)
  add_subdirectory(builtins/zlib)
endif()

#---Check for nlohmann/json.hpp---------------------------------------------------------
if(NOT builtin_nlohmannjson)
  message(STATUS "Looking for nlohmann/json.hpp")
  if(fail-on-missing)
    find_package(nlohmann_json 3.9 REQUIRED)
  else()
    find_package(nlohmann_json 3.9 QUIET)
    if(nlohmann_json_FOUND)
      get_target_property(_nlohmann_json_incl nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
      message(STATUS "Found nlohmann/json.hpp in ${_nlohmann_json_incl} (found version ${nlohmann_json_VERSION})")
    else()
      message(STATUS "nlohmann/json.hpp not found. Switching on builtin_nlohmannjson option")
      set(builtin_nlohmannjson ON CACHE BOOL "Enabled because nlohmann/json.hpp not found" FORCE)
    endif()
  endif()
endif()

if(builtin_nlohmannjson)
  add_subdirectory(builtins/nlohmann)
endif()


#---Check for Unuran ------------------------------------------------------------------
if(unuran AND NOT builtin_unuran)
  message(STATUS "Looking for Unuran")
  if(fail-on-missing)
    find_Package(Unuran REQUIRED)
  else()
    find_Package(Unuran)
    if(NOT UNURAN_FOUND)
      message(STATUS "Unuran not found. Switching on builtin_unuran option")
      set(builtin_unuran ON CACHE BOOL "Enabled because Unuran not found (${builtin_unuran_description})" FORCE)
    endif()
  endif()
endif()

#---Check for Freetype---------------------------------------------------------------
if(NOT builtin_freetype)
  message(STATUS "Looking for Freetype")
  if(fail-on-missing)
    find_package(Freetype REQUIRED)
  else()
    find_package(Freetype)
    if(FREETYPE_FOUND)
      set(FREETYPE_INCLUDE_DIR ${FREETYPE_INCLUDE_DIR_freetype2})
    else()
      message(STATUS "FreeType not found. Switching on builtin_freetype option")
      set(builtin_freetype ON CACHE BOOL "Enabled because FreeType not found (${builtin_freetype_description})" FORCE)
    endif()
  endif()
endif()

if(builtin_freetype)
  set(freetype_version 2.12.1)
  message(STATUS "Building freetype version ${freetype_version} included in ROOT itself")
  set(FREETYPE_LIBRARY ${CMAKE_BINARY_DIR}/FREETYPE-prefix/src/FREETYPE/objs/.libs/${CMAKE_STATIC_LIBRARY_PREFIX}freetype${CMAKE_STATIC_LIBRARY_SUFFIX})
  if(WIN32)
    set(freetypebuild "Release")
    set(freetypelib freetype.lib)
    if(winrtdebug)
      set(freetypebuild "Debug")
      set(freetypelib freetyped.lib)
    endif()
    ExternalProject_Add(
      FREETYPE
      URL ${CMAKE_SOURCE_DIR}/graf2d/freetype/src/freetype-${freetype_version}.tar.gz
      URL_HASH SHA256=efe71fd4b8246f1b0b1b9bfca13cfff1c9ad85930340c27df469733bbb620938
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      CMAKE_ARGS -G ${CMAKE_GENERATOR} -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DFT_DISABLE_BZIP2=TRUE
      BUILD_COMMAND ${CMAKE_COMMAND} --build . --config ${freetypebuild}
      INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_if_different ${freetypebuild}/${freetypelib} ${FREETYPE_LIBRARY}
      LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 BUILD_IN_SOURCE 0
      BUILD_BYPRODUCTS ${FREETYPE_LIBRARY}
      TIMEOUT 600
    )
  else()
    set(_freetype_cflags -O)
    set(_freetype_cc ${CMAKE_C_COMPILER})
    if(CMAKE_SYSTEM_NAME STREQUAL AIX)
      set(_freetype_zlib --without-zlib)
    endif()
    set(_freetype_brotli "--with-brotli=no")
    if(CMAKE_OSX_SYSROOT)
      set(_freetype_cc "${_freetype_cc} -isysroot ${CMAKE_OSX_SYSROOT}")
    endif()
    ExternalProject_Add(
      FREETYPE
      URL ${CMAKE_SOURCE_DIR}/graf2d/freetype/src/freetype-${freetype_version}.tar.gz
      URL_HASH SHA256=efe71fd4b8246f1b0b1b9bfca13cfff1c9ad85930340c27df469733bbb620938
      CONFIGURE_COMMAND ./configure --prefix <INSTALL_DIR> --with-pic
                         --disable-shared --with-png=no --with-bzip2=no
                         --with-harfbuzz=no ${_freetype_brotli} ${_freetype_zlib}
                          "CC=${_freetype_cc}" CFLAGS=${_freetype_cflags}
      INSTALL_COMMAND ""
      LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 BUILD_IN_SOURCE 1
      BUILD_BYPRODUCTS ${FREETYPE_LIBRARY}
      TIMEOUT 600
    )
  endif()
  set(FREETYPE_INCLUDE_DIR ${CMAKE_BINARY_DIR}/FREETYPE-prefix/src/FREETYPE/include)
  set(FREETYPE_INCLUDE_DIRS ${FREETYPE_INCLUDE_DIR})
  set(FREETYPE_LIBRARIES ${FREETYPE_LIBRARY})
  set(FREETYPE_TARGET FREETYPE)
endif()

#---Check for PCRE-------------------------------------------------------------------
if(NOT builtin_pcre)
  message(STATUS "Looking for PCRE")
  # Clear cache before calling find_package(PCRE),
  # necessary to be able to toggle builtin_pcre and
  # not have find_package(PCRE) find builtin pcre.
  foreach(suffix FOUND INCLUDE_DIR PCRE_LIBRARY)
    unset(PCRE_${suffix} CACHE)
  endforeach()
  if(fail-on-missing)
    find_package(PCRE REQUIRED)
  else()
    find_package(PCRE)
    if(NOT PCRE_FOUND)
      message(STATUS "PCRE not found. Switching on builtin_pcre option")
      set(builtin_pcre ON CACHE BOOL "Enabled because PCRE not found (${builtin_pcre_description})" FORCE)
    endif()
  endif()
endif()

if(builtin_pcre)
  list(APPEND ROOT_BUILTINS PCRE)
  add_subdirectory(builtins/pcre)
endif()

#---Check for LZMA-------------------------------------------------------------------
if(NOT builtin_lzma)
  message(STATUS "Looking for LZMA")
  if(fail-on-missing)
    find_package(LibLZMA REQUIRED)
  else()
    find_package(LibLZMA)
    if(NOT LIBLZMA_FOUND)
      message(STATUS "LZMA not found. Switching on builtin_lzma option")
      set(builtin_lzma ON CACHE BOOL "Enabled because LZMA not found (${builtin_lzma_description})" FORCE)
    endif()
  endif()
endif()

if(builtin_lzma)
  set(lzma_version 5.2.4)
  set(LZMA_TARGET LZMA)
  message(STATUS "Building LZMA version ${lzma_version} included in ROOT itself")
  if(WIN32)
    set(LIBLZMA_LIBRARIES ${CMAKE_BINARY_DIR}/LZMA/src/LZMA/lib/liblzma.lib)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(LZMA_URL ${CMAKE_SOURCE_DIR}/core/lzma/src/xz-${lzma_version}-win64.tar.gz)
      set(LZMA_URL_HASH SHA256=76ba7cdff547141f6d6810c8600a9d782feca343debde378fc8f6a307cbfd1d2)
    else()
      set(LZMA_URL ${CMAKE_SOURCE_DIR}/core/lzma/src/xz-${lzma_version}-win32.tar.gz)
      set(LZMA_URL_HASH SHA256=a923ee68d836de5492d8de0fec467b9536f2543c8579ca11f4b5e6f46a8cda8c)
    endif()
    ExternalProject_Add(
      LZMA
      URL ${LZMA_URL}
      URL_HASH ${LZMA_URL_HASH}
      PREFIX LZMA
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      LOG_DOWNLOAD 1
      TIMEOUT 600
    )
    set(LIBLZMA_INCLUDE_DIR ${CMAKE_BINARY_DIR}/LZMA/src/LZMA/include)
  else()
    if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
      set(LIBLZMA_CFLAGS "-Wno-format-nonliteral")
      set(LIBLZMA_LDFLAGS "-Qunused-arguments")
    elseif( CMAKE_CXX_COMPILER_ID STREQUAL Intel)
      set(LIBLZMA_CFLAGS "-wd188 -wd181 -wd1292 -wd10006 -wd10156 -wd2259 -wd981 -wd128 -wd3179 -wd2102")
    endif()
    if(CMAKE_OSX_SYSROOT)
      set(LIBLZMA_CFLAGS "${LIBLZMA_CFLAGS} -isysroot ${CMAKE_OSX_SYSROOT}")
    endif()
    set(LIBLZMA_LIBRARIES ${CMAKE_BINARY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}lzma${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(LIBLZMA_CFLAGS "${LIBLZMA_CFLAGS} -O3")
    ExternalProject_Add(
      LZMA
      URL ${CMAKE_SOURCE_DIR}/core/lzma/src/xz-${lzma_version}.tar.gz
      URL_HASH SHA256=b512f3b726d3b37b6dc4c8570e137b9311e7552e8ccbab4d39d47ce5f4177145
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix <INSTALL_DIR> --libdir <INSTALL_DIR>/lib
                        --with-pic --disable-shared --quiet
                        --disable-scripts --disable-xz --disable-xzdec --disable-lzmadec --disable-lzmainfo --disable-lzma-links
                        CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} CFLAGS=${LIBLZMA_CFLAGS} LDFLAGS=${LIBLZMA_LDFLAGS}
      LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 BUILD_IN_SOURCE 1
      BUILD_BYPRODUCTS ${LIBLZMA_LIBRARIES}
      TIMEOUT 600
    )
    set(LIBLZMA_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include)
  endif()
endif()

#---Check for xxHash-----------------------------------------------------------------
if(NOT builtin_xxhash)
  message(STATUS "Looking for xxHash")
  if(fail-on-missing)
    find_package(xxHash 0.8 REQUIRED)
  else()
    find_package(xxHash 0.8)
    if(NOT xxHash_FOUND)
      message(STATUS "xxHash not found. Switching on builtin_xxhash option")
      set(builtin_xxhash ON CACHE BOOL "Enabled because xxHash not found (${builtin_xxhash_description})" FORCE)
    endif()
  endif()
endif()

if(builtin_xxhash)
  list(APPEND ROOT_BUILTINS xxHash)
  add_subdirectory(builtins/xxhash)
endif()

#---Check for ZSTD-------------------------------------------------------------------
if(NOT builtin_zstd)
  message(STATUS "Looking for ZSTD")
  foreach(suffix FOUND INCLUDE_DIR LIBRARY LIBRARIES LIBRARY_DEBUG LIBRARY_RELEASE)
    unset(ZSTD_${suffix} CACHE)
  endforeach()
  if(fail-on-missing)
    find_package(ZSTD REQUIRED)
    if(ZSTD_VERSION VERSION_LESS 1.0.0)
      message(FATAL "Version of installed ZSTD is too old: ${ZSTD_VERSION}. Please install newer version (>1.0.0)")
    endif()
  else()
    find_package(ZSTD)
    if(NOT ZSTD_FOUND)
      message(STATUS "ZSTD not found. Switching on builtin_zstd option")
      set(builtin_zstd ON CACHE BOOL "Enabled because ZSTD not found (${builtin_zstd_description})" FORCE)
    elseif(ZSTD_FOUND AND ZSTD_VERSION VERSION_LESS 1.0.0)
      message(STATUS "Version of installed ZSTD is too old: ${ZSTD_VERSION}. Switching on builtin_zstd option")
      set(builtin_zstd ON CACHE BOOL "Enabled because ZSTD not found (${builtin_zstd_description})" FORCE)
    endif()
  endif()
endif()

if(builtin_zstd)
  list(APPEND ROOT_BUILTINS ZSTD)
  add_subdirectory(builtins/zstd)
endif()

#---Check for LZ4--------------------------------------------------------------------
if(NOT builtin_lz4)
  message(STATUS "Looking for LZ4")
  foreach(suffix FOUND INCLUDE_DIR LIBRARY LIBRARY_DEBUG LIBRARY_RELEASE)
    unset(LZ4_${suffix} CACHE)
  endforeach()
  if(fail-on-missing)
    find_package(LZ4 REQUIRED)
  else()
    find_package(LZ4)
    if(NOT LZ4_FOUND)
      message(STATUS "LZ4 not found. Switching on builtin_lz4 option")
      set(builtin_lz4 ON CACHE BOOL "Enabled because LZ4 not found (${builtin_lz4_description})" FORCE)
    endif()
  endif()
endif()

if(builtin_lz4)
  list(APPEND ROOT_BUILTINS LZ4)
  add_subdirectory(builtins/lz4)
endif()

#---Check for X11 which is mandatory lib on Unix--------------------------------------
if(x11)
  message(STATUS "Looking for X11")
  if(X11_X11_INCLUDE_PATH)
    set(X11_FIND_QUIETLY 1)
  endif()
  find_package(X11 REQUIRED)
  if(X11_FOUND)
    list(REMOVE_DUPLICATES X11_INCLUDE_DIR)
    if(NOT X11_FIND_QUIETLY)
      message(STATUS "X11_INCLUDE_DIR: ${X11_INCLUDE_DIR}")
      message(STATUS "X11_LIBRARIES: ${X11_LIBRARIES}")
    endif()
  else()
    message(FATAL_ERROR "libX11 and X11 headers must be installed.")
  endif()
  if(X11_Xpm_FOUND)
    if(NOT X11_FIND_QUIETLY)
      message(STATUS "X11_Xpm_INCLUDE_PATH: ${X11_Xpm_INCLUDE_PATH}")
      message(STATUS "X11_Xpm_LIB: ${X11_Xpm_LIB}")
    endif()
  else()
    message(FATAL_ERROR "libXpm and Xpm headers must be installed.")
  endif()
  if(X11_Xft_FOUND)
    if(NOT X11_FIND_QUIETLY)
      message(STATUS "X11_Xft_INCLUDE_PATH: ${X11_Xft_INCLUDE_PATH}")
      message(STATUS "X11_Xft_LIB: ${X11_Xft_LIB}")
    endif()
  else()
    message(FATAL_ERROR "libXft and Xft headers must be installed.")
  endif()
  if(X11_Xext_FOUND)
    if(NOT X11_FIND_QUIETLY)
      message(STATUS "X11_Xext_INCLUDE_PATH: ${X11_Xext_INCLUDE_PATH}")
      message(STATUS "X11_Xext_LIB: ${X11_Xext_LIB}")
    endif()
  else()
    message(FATAL_ERROR "libXext and Xext headers must be installed.")
  endif()
endif()

#---Check for all kind of graphics includes needed by libAfterImage--------------------
if(asimage)
  if(NOT x11 AND NOT cocoa AND NOT WIN32)
    message(STATUS "Switching off 'asimage' because neither 'x11' nor 'cocoa' are enabled")
    set(asimage OFF CACHE BOOL "Disabled because neither x11 nor cocoa are enabled (${asimage_description})" FORCE)
  endif()
endif()
if(asimage)
  set(ASEXTRA_LIBRARIES)
  find_Package(GIF)
  if(GIF_FOUND)
    set(ASEXTRA_LIBRARIES ${ASEXTRA_LIBRARIES} ${GIF_LIBRARIES})
  endif()
  find_Package(TIFF)
  if(TIFF_FOUND)
    set(ASEXTRA_LIBRARIES ${ASEXTRA_LIBRARIES} ${TIFF_LIBRARIES})
  endif()
  find_Package(PNG)
  if(PNG_FOUND)
    set(ASEXTRA_LIBRARIES ${ASEXTRA_LIBRARIES} ${PNG_LIBRARIES})
    # Some missing variables needed for external PNG build
    set(PNG_LIBRARY_RELEASE ${PNG_LIBRARY})
    # apparently there will be two set of includes here (needs to be selected only last that was passed: PNG_INCLUDE_DIR)
    list(GET PNG_INCLUDE_DIRS 0 PNG_INCLUDE_DIR)
  endif()
  find_Package(JPEG)
  if(JPEG_FOUND)
    set(ASEXTRA_LIBRARIES ${ASEXTRA_LIBRARIES} ${JPEG_LIBRARIES})
  endif()

  #---AfterImage---------------------------------------------------------------
  set(AFTERIMAGE_LIBRARIES ${CMAKE_BINARY_DIR}/lib/libAfterImage${CMAKE_STATIC_LIBRARY_SUFFIX})
  if(WIN32)
    if(winrtdebug)
      set(astepbld "Debug")
    else()
      set(astepbld "Release")
    endif()
    ExternalProject_Add(
      AFTERIMAGE
      DOWNLOAD_COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/graf2d/asimage/src/libAfterImage AFTERIMAGE
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      CMAKE_ARGS -G ${CMAKE_GENERATOR} -DCMAKE_VERBOSE_MAKEFILE=ON -DFREETYPE_INCLUDE_DIR=${FREETYPE_INCLUDE_DIR} -DZLIB_INCLUDE_DIR=${ZLIB_INCLUDE_DIR}
      BUILD_COMMAND ${CMAKE_COMMAND} --build . --config ${astepbld}
      INSTALL_COMMAND  ${CMAKE_COMMAND} -E copy_if_different ${astepbld}/libAfterImage.lib <INSTALL_DIR>/lib/
      LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 BUILD_IN_SOURCE 0
      BUILD_BYPRODUCTS ${AFTERIMAGE_LIBRARIES}
      TIMEOUT 600
    )
    set(AFTERIMAGE_INCLUDE_DIR ${CMAKE_BINARY_DIR}/AFTERIMAGE-prefix/src/AFTERIMAGE)
  else()
    if(JPEG_FOUND)
      set(_jpeginclude --with-jpeg-includes=${JPEG_INCLUDE_DIR})
    else()
      set(_jpeginclude --with-builtin-jpeg)
    endif()
    if(PNG_FOUND)
      set(_pnginclude  --with-png-includes=${PNG_INCLUDE_DIR})
    else()
       set(_pnginclude  --with-builtin-png)
    endif()
    if(TIFF_FOUND)
      set(_tiffinclude --with-tiff-includes=${TIFF_INCLUDE_DIR})
    else()
      set(_tiffinclude --with-tiff=no)
    endif()
    if(cocoa)
      set(_jpeginclude --without-x --with-builtin-jpeg)
      set(_pnginclude  --with-builtin-png)
      set(_tiffinclude --with-tiff=no)
    endif()
    if(builtin_freetype)
      set(_ttf_include --with-ttf-includes=-I${FREETYPE_INCLUDE_DIR})
      set(_after_cflags "${_after_cflags} -DHAVE_FREETYPE_FREETYPE -DPNG_ARM_NEON_OPT=0")
    endif()
    if(CMAKE_OSX_SYSROOT)
      set(_after_cflags "${_after_cflags} -isysroot ${CMAKE_OSX_SYSROOT}")
    endif()
    ExternalProject_Add(
      AFTERIMAGE
      DOWNLOAD_COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/graf2d/asimage/src/libAfterImage AFTERIMAGE
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      CONFIGURE_COMMAND ./configure --prefix <INSTALL_DIR>
                        --libdir=<INSTALL_DIR>/lib
                        --with-ttf ${_ttf_include} --with-afterbase=no
                        --without-svg --disable-glx ${_after_mmx}
                        --with-builtin-ungif  --with-jpeg ${_jpeginclude}
                        --with-png ${_pnginclude} ${_tiffinclude}
                        CC=${CMAKE_C_COMPILER} CFLAGS=${_after_cflags}
      LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 BUILD_IN_SOURCE 1
      BUILD_BYPRODUCTS ${AFTERIMAGE_LIBRARIES}
      TIMEOUT 600
    )
    set(AFTERIMAGE_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include/libAfterImage)
  endif()
  if(builtin_freetype)
    add_dependencies(AFTERIMAGE FREETYPE)
  endif()
  set(AFTERIMAGE_TARGET AFTERIMAGE)
endif()

#---Check for GSL library---------------------------------------------------------------
if(mathmore OR builtin_gsl)
  if(builtin_gsl AND NO_CONNECTION)
    if(fail-on-missing)
      message(FATAL_ERROR "No internet connection. Please check your connection, or either disable the 'builtin_gsl' option or the 'fail-on-missing' to automatically disable options requiring internet access")
    else()
      message(STATUS "No internet connection, disabling 'builtin_gsl' option")
      set(builtin_gsl OFF CACHE BOOL "Disabled because there is no internet connection" FORCE)
    endif()
  endif()
  message(STATUS "Looking for GSL")
  if(NOT builtin_gsl)
    find_package(GSL 1.10)
    if(NOT GSL_FOUND)
      if(fail-on-missing)
        message(FATAL_ERROR "GSL package not found and 'mathmore' component if required ('fail-on-missing' enabled). "
                            "Alternatively, you can enable the option 'builtin_gsl' to build the GSL libraries internally.")
      else()
        message(STATUS "GSL not found. Set variable GSL_ROOT_DIR to point to your GSL installation")
        message(STATUS "               Alternatively, you can also enable the option 'builtin_gsl' to build the GSL libraries internally'")
        message(STATUS "               For the time being switching OFF 'mathmore' option")
        set(mathmore OFF CACHE BOOL "Disable because builtin_gsl disabled and external GSL not found (${mathmore_description})" FORCE)
      endif()
    endif()
  else()
    set(gsl_version 2.5)
    message(STATUS "Downloading and building GSL version ${gsl_version}")
    foreach(l gsl gslcblas)
      list(APPEND GSL_LIBRARIES ${CMAKE_BINARY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${l}${CMAKE_STATIC_LIBRARY_SUFFIX})
    endforeach()
    set(GSL_CBLAS_LIBRARY ${CMAKE_BINARY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gslcblas${CMAKE_STATIC_LIBRARY_SUFFIX})
    if(CMAKE_OSX_SYSROOT)
      set(_gsl_cppflags "-isysroot ${CMAKE_OSX_SYSROOT}")
      set(_gsl_ldflags  "-isysroot ${CMAKE_OSX_SYSROOT}")
    endif()
    ExternalProject_Add(
      GSL
      # http://mirror.switch.ch/ftp/mirror/gnu/gsl/gsl-${gsl_version}.tar.gz
      URL ${lcgpackages}/gsl-${gsl_version}.tar.gz
      URL_HASH SHA256=0460ad7c2542caaddc6729762952d345374784100223995eb14d614861f2258d
      SOURCE_DIR GSL-src # prevent "<gsl/...>" vs GSL/ macOS warning
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix <INSTALL_DIR>
                        --libdir=<INSTALL_DIR>/lib
                        --enable-shared=no --with-pic
                        CC=${CMAKE_C_COMPILER}
                        CFLAGS=${CMAKE_C_FLAGS}
                        CPPFLAGS=${_gsl_cppflags}
                        LDFLAGS=${_gsl_ldflags}
      LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1
      BUILD_BYPRODUCTS ${GSL_LIBRARIES}
      TIMEOUT 600
    )
    set(GSL_TARGET GSL)
    # FIXME: one need to find better way to extract path with GSL include files
    set(GSL_INCLUDE_DIR ${CMAKE_BINARY_DIR}/GSL-prefix/src/GSL-build)
    set(GSL_FOUND ON)
    set(mathmore ON CACHE BOOL "Enabled because builtin_gls requested (${mathmore_description})" FORCE)
  endif()
endif()

#---Check for OpenGL installation-------------------------------------------------------
if(opengl)
  message(STATUS "Looking for OpenGL")
  find_package(OpenGL)
  if(NOT OPENGL_FOUND OR NOT OPENGL_GLU_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "OpenGL package (with GLU) not found and opengl option required")
    elseif(cocoa)
      message(FATAL_ERROR "OpenGL package (with GLU) not found and opengl option required for \"cocoa=ON\"")
    else()
      message(STATUS "OpenGL (with GLU) not found. Switching off opengl option")
      set(opengl OFF CACHE BOOL "Disabled because OpenGL (with GLU) not found (${opengl_description})" FORCE)
    endif()
  endif()
endif()
# OpenGL should be working only with x11 (Linux),
# in case when -Dall=ON -Dx11=OFF, we will just disable opengl.
if(NOT WIN32 AND NOT APPLE)
  if(opengl AND NOT x11)
    message(STATUS "OpenGL was disabled, since it is requires x11 on Linux")
    set(opengl OFF CACHE BOOL "OpenGL requires x11" FORCE)
  endif()
endif()

#---Check for GLEW -------------------------------------------------------------------
# Opengl is "must" requirement for Glew.
if(opengl AND NOT builtin_glew)
  message(STATUS "Looking for GLEW")
  if(fail-on-missing)
    find_package(GLEW REQUIRED)
  else()
    find_package(GLEW)
    # Bug was reported on newer version of CMake on Mac OS X:
    # https://gitlab.kitware.com/cmake/cmake/-/issues/19662
    # https://github.com/microsoft/vcpkg/pull/7967
    if(GLEW_FOUND AND APPLE AND CMAKE_VERSION VERSION_GREATER 3.15)
      message(FATAL_ERROR "Please enable builtin Glew due bug in latest CMake (use cmake option -Dbuiltin_glew=ON).")
      unset(GLEW_FOUND)
    elseif(GLEW_FOUND AND NOT TARGET GLEW::GLEW)
      add_library(GLEW::GLEW UNKNOWN IMPORTED)
      set_target_properties(GLEW::GLEW PROPERTIES
      IMPORTED_LOCATION "${GLEW_LIBRARIES}"
      INTERFACE_INCLUDE_DIRECTORIES "${GLEW_INCLUDE_DIRS}")
    endif()
    if(NOT GLEW_FOUND)
      message(STATUS "GLEW not found. Switching on builtin_glew option")
      set(builtin_glew ON CACHE BOOL "Enabled because opengl requested and GLEW not found (${builtin_glew_description})" FORCE)
    endif()
  endif()
endif()

if(builtin_glew)
  list(APPEND ROOT_BUILTINS GLEW)
  add_subdirectory(builtins/glew)
endif()

#---Check for gl2ps ------------------------------------------------------------------
if(opengl AND NOT builtin_gl2ps)
  message(STATUS "Looking for gl2ps")
  if(fail-on-missing)
    find_Package(gl2ps REQUIRED)
  else()
    find_Package(gl2ps)
    if(NOT GL2PS_FOUND)
      message(STATUS "gl2ps not found. Switching on builtin_gl2ps option")
      set(builtin_gl2ps ON CACHE BOOL "Enabled because opengl requested and gl2ps not found (${builtin_gl2ps_description})" FORCE)
    endif()
  endif()
endif()

#---Check for Graphviz installation-------------------------------------------------------
if(gviz)
  message(STATUS "Looking for Graphviz")
  find_package(Graphviz)
  if(NOT GRAPHVIZ_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "Graphviz package not found and gviz option required")
    else()
      message(STATUS "Graphviz not found. Switching off gviz option")
      set(gviz OFF CACHE BOOL "Disabled because Graphviz not found (${gviz_description})" FORCE)
    endif()
  endif()
endif()

#---Check for XML Parser Support-----------------------------------------------------------
if(xml)
  message(STATUS "Looking for LibXml2")
  find_package(LibXml2)
  if(NOT LIBXML2_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "LibXml2 libraries not found and they are required (xml option enabled)")
    else()
      message(STATUS "LibXml2 not found. Switching off xml option")
      set(xml OFF CACHE BOOL "Disabled because LibXml2 not found (${xml_description})" FORCE)
    endif()
  endif()
endif()

#---Check for OpenSSL------------------------------------------------------------------
foreach(suffix FOUND INCLUDE_DIR INCLUDE_DIRS LIBRARY LIBRARIES VERSION)
  unset(OPENSSL_${suffix} CACHE)
endforeach()

if(ssl AND NOT builtin_openssl)
  if(fail-on-missing)
    find_package(OpenSSL REQUIRED)
  else()
    find_package(OpenSSL)
    if(NOT OPENSSL_FOUND)
      if(WIN32) # builtin OpenSSL does not work on Windows
        message(STATUS "Switching OFF 'ssl' option.")
        set(ssl OFF CACHE BOOL "Disabled because OpenSSL not found and builtin version does not work on Windows (${ssl_description})" FORCE)
      else()
        if(NO_CONNECTION)
          if(fail-on-missing)
            message(FATAL_ERROR "No internet connection and OpenSSL was not found. Please check your connection, or either disable the 'ssl' option or the 'fail-on-missing' to automatically disable options requiring internet access")
          else()
            message(STATUS "OpenSSL not found, and no internet connection. Disabing the 'ssl' option.")
            set(ssl OFF CACHE BOOL "Disabled because ssl requested and OpenSSL not found (${builtin_openssl_description}) and there is no internet connection" FORCE)
          endif()
        else()
          message(STATUS "OpenSSL not found, switching ON 'builtin_openssl' option.")
          set(builtin_openssl ON CACHE BOOL "Enabled because ssl requested and OpenSSL not found (${builtin_openssl_description})" FORCE)
        endif()
      endif()
    endif()
  endif()
endif()

if(builtin_openssl)
  if(NO_CONNECTION)
    if(fail-on-missing)
      message(FATAL_ERROR "No internet connection. Please check your connection, or either disable the 'builtin_openssl' option or the 'fail-on-missing' to automatically disable options requiring internet access")
    else()
      message(STATUS "No internet connection, disabling the 'ssl' and 'builtin_openssl' options")
      set(builtin_openssl OFF CACHE BOOL "Disabled because there is no internet connection" FORCE)
      set(ssl OFF CACHE BOOL "Disabled because there is no internet connection" FORCE)
    endif()
  else()
    list(APPEND ROOT_BUILTINS OpenSSL)
    add_subdirectory(builtins/openssl)
  endif()
endif()

#---Check for MySQL-------------------------------------------------------------------
if(mysql)
  message(STATUS "Looking for MySQL")
  find_package(MySQL)
  if(NOT MYSQL_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "MySQL libraries not found and they are required (mysql option enabled)")
    else()
      message(STATUS "MySQL not found. Switching off mysql option")
      set(mysql OFF CACHE BOOL "Disabled because MySQL not found (${mysql_description})" FORCE)
    endif()
  endif()
endif()

#---Check for FastCGI-----------------------------------------------------------
if(fcgi)
  message(STATUS "Looking for FastCGI")
  find_package(FastCGI)
  if(NOT FASTCGI_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "FastCGI library not found and they are required (fcgi option enabled)")
    else()
      message(STATUS "FastCGI not found. Switching off fcgi option")
      set(fcgi OFF CACHE BOOL "Disabled because FastCGI not found" FORCE)
    endif()
  endif()
endif()


#---Check for ODBC-------------------------------------------------------------------
if(odbc)
  message(STATUS "Looking for ODBC")
  find_package(ODBC)
  if(NOT ODBC_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "ODBC libraries not found and they are required (odbc option enabled)")
    else()
      message(STATUS "ODBC not found. Switching off odbc option")
      set(odbc OFF CACHE BOOL "Disabled because ODBC not found (${odbc_description})" FORCE)
    endif()
  endif()
endif()

#---Check for PostgreSQL-------------------------------------------------------------------
if(pgsql)
  message(STATUS "Looking for PostgreSQL")
  find_package(PostgreSQL)
  if(NOT PostgreSQL_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "PostgreSQL libraries not found and they are required (pgsql option enabled)")
    else()
      message(STATUS "PostgreSQL not found. Switching off pgsql option")
      set(pgsql OFF CACHE BOOL "Disabled because PostgreSQL not found (${pgsql_description})" FORCE)
    endif()
  endif()
endif()

#---Check for SQLite-------------------------------------------------------------------
if(sqlite)
  message(STATUS "Looking for SQLite")
  find_package(Sqlite)
  if(NOT SQLITE_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "SQLite libraries not found and they are required (sqlite option enabled)")
    else()
      message(STATUS "SQLite not found. Switching off sqlite option")
      set(sqlite OFF CACHE BOOL "Disabled because SQLite not found (${sqlite_description})" FORCE)
    endif()
  endif()
endif()

#---Check for Pythia8-------------------------------------------------------------------
if(pythia8)
  message(STATUS "Looking for Pythia8")
  find_package(Pythia8)
  if(NOT PYTHIA8_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "Pythia8 libraries not found and they are required (pythia8 option enabled)")
    else()
      message(STATUS "Pythia8 not found. Switching off pythia8 option")
      set(pythia8 OFF CACHE BOOL "Disabled because Pythia8 not found (${pythia8_description})" FORCE)
    endif()
  endif()
endif()

if(builtin_fftw3 AND NO_CONNECTION)
  if(fail-on-missing)
    message(FATAL_ERROR "No internet connection. Please check your connection, or either disable the 'builtin_fftw3' option or the 'fail-on-missing' to automatically disable options requiring internet access")
  else()
    message(STATUS "No internet connection, disabling 'builtin_fftw3' option")
    set(builtin_fftw3 OFF CACHE BOOL "Disabled because there is no internet connection" FORCE)
  endif()
endif()

#---Check for FFTW3-------------------------------------------------------------------
if(fftw3)
  if(NOT builtin_fftw3)
    message(STATUS "Looking for FFTW3")
    find_package(FFTW)
    if(NOT FFTW_FOUND)
      if(fail-on-missing)
        message(FATAL_ERROR "FFTW3 libraries not found and they are required (fftw3 option enabled)")
      else()
        message(STATUS "FFTW3 not found. Set [environment] variable FFTW_DIR to point to your FFTW3 installation")
        message(STATUS "                 Alternatively, you can also enable the option 'builtin_fftw3' to build FFTW3 internally'")
        message(STATUS "                 For the time being switching OFF 'fftw3' option")
        set(fftw3 OFF CACHE BOOL "Disabled because FFTW3 not found and builtin_fftw3 disabled (${fftw3_description})" FORCE)
      endif()
    endif()
  endif()
endif()
if(builtin_fftw3)
  set(FFTW_VERSION 3.3.8)
  message(STATUS "Downloading and building FFTW version ${FFTW_VERSION}")
  set(FFTW_LIBRARIES ${CMAKE_BINARY_DIR}/lib/libfftw3.a)
  ExternalProject_Add(
    FFTW3
    URL ${lcgpackages}/fftw-${FFTW_VERSION}.tar.gz
    URL_HASH SHA256=6113262f6e92c5bd474f2875fa1b01054c4ad5040f6b0da7c03c98821d9ae303
    INSTALL_DIR ${CMAKE_BINARY_DIR}
    CONFIGURE_COMMAND ./configure --prefix=<INSTALL_DIR>
    BUILD_COMMAND make CFLAGS=-fPIC
    LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1
    BUILD_IN_SOURCE 1
    BUILD_BYPRODUCTS ${FFTW_LIBRARIES}
    TIMEOUT 600
  )
  set(FFTW_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include)
  set(FFTW3_TARGET FFTW3)
  set(fftw3 ON CACHE BOOL "Enabled because builtin_fftw3 requested (${fftw3_description})" FORCE)
endif()

#---Check for fitsio-------------------------------------------------------------------
if(fitsio OR builtin_cfitsio)
  if(builtin_cfitsio AND NO_CONNECTION)
    if(fail-on-missing)
      message(FATAL_ERROR "No internet connection. Please check your connection, or either disable the 'builtin_cfitsio' option or the 'fail-on-missing' to automatically disable options requiring internet access")
    else()
      message(STATUS "No internet connection, disabling 'builtin_cfitsio' option")
      set(builtin_cfitsio OFF CACHE BOOL "Disabled because there is no internet connection" FORCE)
    endif()
  endif()
  if(builtin_cfitsio)
    set(cfitsio_version 3.450)
    string(REPLACE "." "" cfitsio_version_no_dots ${cfitsio_version})
    message(STATUS "Downloading and building CFITSIO version ${cfitsio_version}")
    set(CFITSIO_LIBRARIES ${CMAKE_BINARY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}cfitsio${CMAKE_STATIC_LIBRARY_SUFFIX})
    if(WIN32)
      if(winrtdebug)
        set(cfitsiobuild "Debug")
      else()
        set(cfitsiobuild "Release")
      endif()
      ExternalProject_Add(
        CFITSIO
        # ftp://heasarc.gsfc.nasa.gov/software/fitsio/c/cfitsio${cfitsio_version_no_dots}.tar.gz
        URL http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfit3450.zip
        URL_HASH SHA256=1d13073967654a48d47535ff33392656f252511ddf29059d7c7dc3ce8f2a1041
        INSTALL_DIR ${CMAKE_BINARY_DIR}
        CMAKE_ARGS -G ${CMAKE_GENERATOR} -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
        BUILD_COMMAND ${CMAKE_COMMAND} --build . --config ${cfitsiobuild}
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy ${cfitsiobuild}/cfitsio.dll <INSTALL_DIR>/bin
                COMMAND ${CMAKE_COMMAND} -E copy ${cfitsiobuild}/cfitsio.lib <INSTALL_DIR>/lib
        LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 BUILD_IN_SOURCE 0
        BUILD_BYPRODUCTS ${CFITSIO_LIBRARIES}
        TIMEOUT 600
      )
      set(CFITSIO_INCLUDE_DIR ${CMAKE_BINARY_DIR}/CFITSIO-prefix/src/CFITSIO)
      install(DIRECTORY ${CMAKE_BINARY_DIR}/bin/ DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT libraries FILES_MATCHING PATTERN "cfitsio*.dll")
    else()
      ExternalProject_Add(
        CFITSIO
        # ftp://heasarc.gsfc.nasa.gov/software/fitsio/c/cfitsio${cfitsio_version_no_dots}.tar.gz
        URL ${lcgpackages}/cfitsio${cfitsio_version_no_dots}.tar.gz
        URL_HASH SHA256=bf6012dbe668ecb22c399c4b7b2814557ee282c74a7d5dc704eb17c30d9fb92e
        INSTALL_DIR ${CMAKE_BINARY_DIR}
        CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix <INSTALL_DIR>
        LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1
        BUILD_IN_SOURCE 1
        BUILD_BYPRODUCTS ${CFITSIO_LIBRARIES}
        TIMEOUT 600
      )
      # We need to know which CURL_LIBRARIES were used in CFITSIO ExternalProject build
      # and which ${CURL_LIBRARIES} should be used after for linking in ROOT together with CFITSIO.
      # (curl is not strictly required in CFITSIO CMakeList.txt).
      find_package(CURL)
      if(CURL_FOUND)
        set(CFITSIO_LIBRARIES ${CFITSIO_LIBRARIES} ${CURL_LIBRARIES})
      endif()
      set(CFITSIO_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include)
    endif()
    set(fitsio ON CACHE BOOL "Enabled because builtin_cfitsio requested (${fitsio_description})" FORCE)
    set(CFITSIO_TARGET CFITSIO)
  else()
    message(STATUS "Looking for CFITSIO")
    if(fail-on-missing)
      find_package(CFITSIO REQUIRED)
    else()
      find_package(CFITSIO)
      if(NOT CFITSIO_FOUND)
        message(STATUS "CFITSIO not found. You can enable the option 'builtin_cfitsio' to build the library internally'")
        message(STATUS "                   For the time being switching off 'fitsio' option")
        set(fitsio OFF CACHE BOOL "Disabled because CFITSIO not found and builtin_cfitsio disabled (${fitsio_description})" FORCE)
      endif()
    endif()
  endif()
endif()

#---Check Shadow password support----------------------------------------------------
if(shadowpw)
  if(NOT EXISTS /etc/shadow)  #---TODO--The test always succeeds because the actual file is protected
    if(NOT CMAKE_SYSTEM_NAME MATCHES Linux)
      message(STATUS "Support Shadow password not found. Switching off shadowpw option")
      set(shadowpw OFF CACHE BOOL "Disabled because /etc/shadow not found (${shadowpw_description})" FORCE)
    endif()
  endif()
endif()

#---Check for Xrootd support---------------------------------------------------------

foreach(suffix FOUND INCLUDE_DIR INCLUDE_DIRS LIBRARY LIBRARIES)
  unset(XROOTD_${suffix} CACHE)
endforeach()

if(xrootd AND NOT builtin_xrootd)
  message(STATUS "Looking for XROOTD")
  find_package(XRootD)
  if(NOT XROOTD_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "XROOTD not found. Set environment variable XRDSYS to point to your XROOTD installation, "
                          "or include the installation of XROOTD in the CMAKE_PREFIX_PATH. "
                          "Alternatively, you can also enable the option 'builtin_xrootd' to build XROOTD internally")
    else()
      message(STATUS "XROOTD not found, enabling 'builtin_xrootd' option")
      set(builtin_xrootd ON CACHE BOOL "Enabled because xrootd is enabled, but external xrootd was not found (${xrootd_description})" FORCE)
    endif()
  endif()
endif()

if(builtin_xrootd AND NO_CONNECTION)
  if(fail-on-missing)
    message(FATAL_ERROR "No internet connection. Please check your connection, or either disable the 'builtin_xrootd' option or the 'fail-on-missing' to automatically disable options requiring internet access")
  else()
    message(STATUS "No internet connection, disabling 'builtin_xrootd' option")
    set(builtin_xrootd OFF CACHE BOOL "Disabled because there is no internet connection" FORCE)
    set(xrootd OFF CACHE BOOL "Disabled because there is no internet connection" FORCE)
  endif()
endif()
if(builtin_xrootd)
  list(APPEND ROOT_BUILTINS XROOTD)
  add_subdirectory(builtins/xrootd)
  set(xrootd ON CACHE BOOL "Enabled because builtin_xrootd requested (${xrootd_description})" FORCE)
endif()

#---check if netxng can be built-------------------------------
if(xrootd)
  set(netxng ON)
endif()

#---make sure non-builtin xrootd is not using builtin_openssl-----------
if(xrootd AND NOT builtin_xrootd AND builtin_openssl)
  if(fail-on-missing)
    message(FATAL_ERROR "Non-builtin XROOTD must not be used with builtin OpenSSL. If you want to use non-builtin XROOTD, please use the system OpenSSL")
  else()
    message(STATUS "Non-builtin XROOTD must not be used with builtin OpenSSL. Disabling the 'xrootd' option.")
    set(xrootd OFF CACHE BOOL "Disabled because non-builtin xrootd cannot be used with builtin OpenSSL" FORCE)
  endif()
endif()

#---Check for Apache Arrow
if(arrow)
  find_package(Arrow)
  if(NOT ARROW_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "Apache Arrow not found. Please set ARROW_HOME to point to your Arrow installation, "
                          "or include the installation of Arrow in the CMAKE_PREFIX_PATH.")
    else()
      message(STATUS "Apache Arrow API not found. Set variable ARROW_HOME to point to your Arrow installation, "
                     "or include the installation of Arrow in the CMAKE_PREFIX_PATH.")
      message(STATUS "For the time being switching OFF 'arrow' option")
      set(arrow OFF CACHE BOOL "Disabled because Apache Arrow API not found (${arrow_description})" FORCE)
    endif()
  else()
    if(${ARROW_VERSION} VERSION_GREATER_EQUAL 10.0.0 AND CMAKE_CXX_STANDARD LESS 17)
      if(fail-on-missing)
        message(FATAL_ERROR "The Apache Arrow version found on the system (${ARROW_VERSION}) requires at least CMAKE_CXX_STANDARD=17")
      else()
        message(STATUS "The Apache Arrow version found on the system (${ARROW_VERSION}) requires at least CMAKE_CXX_STANDARD=17")
        message(STATUS "For the time being switching OFF 'arrow' option")
        set(arrow OFF CACHE BOOL "Disabled because Apache Arrow Version ${ARROW_VERSION} requires CMAKE_CXX_STANDARD=17)" FORCE)
      endif()
    endif()
  endif()

endif()

#---Check for dCache-------------------------------------------------------------------
if(dcache)
  find_package(DCAP)
  if(NOT DCAP_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "dCap library not found and is required (dcache option enabled)")
    else()
      message(STATUS "dCap library not found. Set variable DCAP_DIR to point to your dCache installation")
      message(STATUS "For the time being switching OFF 'dcache' option")
      set(dcache OFF CACHE BOOL "Disabled because dCap not found (${dcache_description})" FORCE)
    endif()
  endif()
endif()

#---Check for ftgl if needed----------------------------------------------------------
if(opengl AND NOT builtin_ftgl)
  find_package(FTGL)
  if(NOT FTGL_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "ftgl library not found and is required ('builtin_ftgl' is OFF). Set varible FTGL_ROOT_DIR to installation location")
    else()
      message(STATUS "ftgl library not found. Set variable FTGL_ROOT_DIR to point to your installation")
      message(STATUS "For the time being switching ON 'builtin_ftgl' option")
      set(builtin_ftgl ON CACHE BOOL "Enabled because ftgl not found but opengl requested (${builtin_ftgl_description})" FORCE)
    endif()
  endif()
endif()

if(builtin_ftgl)
  # clear variables set to NOTFOUND to allow builtin FTGL to override them
  foreach(var FTGL_LIBRARIES FTGL_LIBRARY FTGL_LIBRARY_DEBUG FTGL_LIBRARY_RELEASE)
    unset(${var})
    unset(${var} CACHE)
  endforeach()
  set(FTGL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/graf3d/ftgl/inc)
  set(FTGL_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/graf3d/ftgl/inc)
  set(FTGL_CFLAGS -DBUILTIN_FTGL)
  set(FTGL_LIBRARIES FTGL)
endif()

#---Check for R/Rcpp/RInside--------------------------------------------------------------------
#added search of R packages here to remove multiples searches
if(r)
  message(STATUS "Looking for R")
  find_package(R COMPONENTS Rcpp RInside)
  if(NOT R_FOUND)
    if(fail-on-missing)
       message(FATAL_ERROR "R installation not found and is required ('r' option enabled)")
    else()
       message(STATUS "R installation not found. Set variable R_DIR to point to your R installation")
       message(STATUS "For the time being switching OFF 'r' option")
       set(r OFF CACHE BOOL "Disabled because R not found (${r_description})" FORCE)
    endif()
  endif()
endif()

#---Check for Davix library-----------------------------------------------------------

foreach(suffix FOUND INCLUDE_DIR INCLUDE_DIRS LIBRARY LIBRARIES)
  unset(DAVIX_${suffix} CACHE)
endforeach()

if(davix AND NOT builtin_davix)
  if(MSVC)
    message(FATAL_ERROR "Davix is not supported on Windows")
  endif()

  if(fail-on-missing)
    find_package(Davix 0.6.4 REQUIRED)
    if(DAVIX_VERSION VERSION_GREATER_EQUAL 0.6.8 AND DAVIX_VERSION VERSION_LESS 0.7.1)
      message(WARNING "Davix versions 0.6.8 to 0.7.0 have a bug and do not work with ROOT, please upgrade to 0.7.1 or later.")
    endif()
  else()
    find_package(Davix 0.6.4)
    if(NOT DAVIX_FOUND)
      find_package(libuuid)
      if(NOT libuuid_FOUND)
        message(STATUS "Davix dependency libuuid not found, switching OFF 'davix' option.")
      endif()
      find_package(LibXml2)
      if(NOT LIBXML2_FOUND)
        message(STATUS "Davix dependency libxml2 not found, switching OFF 'davix' option.")
      endif()
      find_package(OpenSSL)
      if(NOT (OPENSSL_FOUND OR builtin_openssl))
        message(STATUS "Davix dependency openssl not found, switching OFF 'davix' option.")
      endif()
      if(libuuid_FOUND AND LIBXML2_FOUND AND (OPENSSL_FOUND OR builtin_openssl))
        message(STATUS "Davix not found, switching ON 'builtin_davix' option.")
        set(builtin_davix ON CACHE BOOL "Enabled because external Davix not found but davix requested (${builtin_davix_description})" FORCE)
      else()
        set(davix OFF CACHE BOOL "Disabled because dependencies not found (${davix_description})" FORCE)
      endif()
    endif()
  endif()
endif()

if(builtin_davix)
  if(NO_CONNECTION)
    if(fail-on-missing)
      message(FATAL_ERROR "No internet connection. Please check your connection, or either disable the 'builtin_davix' option or the 'fail-on-missing' to automatically disable options requiring internet access")
    else()
      message(STATUS "No internet connection, disabling the 'davix' and 'builtin_davix' options")
      set(builtin_davix OFF CACHE BOOL "Disabled because there is no internet connection" FORCE)
      set(davix OFF CACHE BOOL "Disabled because there is no internet connection" FORCE)
      endif()
  else()
    list(APPEND ROOT_BUILTINS Davix)
    add_subdirectory(builtins/davix)
    set(davix ON CACHE BOOL "Enabled because builtin_davix is enabled)" FORCE)
  endif()
endif()

#---Check for TCMalloc---------------------------------------------------------------
if (tcmalloc)
  message(STATUS "Looking for tcmalloc")
  find_package(tcmalloc)
  if(NOT TCMALLOC_FOUND)
    message(STATUS "TCMalloc not found.")
  endif()
endif()

#---Check for JEMalloc---------------------------------------------------------------
if (jemalloc)
  if (tcmalloc)
   message(FATAL_ERROR "Both tcmalloc and jemalloc were selected: this is an inconsistent setup.")
  endif()
  message(STATUS "Looking for jemalloc")
  find_package(jemalloc)
  if(NOT JEMALLOC_FOUND)
    message(STATUS "JEMalloc not found.")
  endif()
endif()

#---Check for liburing----------------------------------------------------------------
if (uring)
  if(NOT CMAKE_SYSTEM_NAME MATCHES Linux)
    set(uring OFF CACHE BOOL "Disabled because liburing is only available on Linux" FORCE)
    message(STATUS "liburing was disabled because it is only available on Linux")
  else()
    message(STATUS "Looking for liburing")
    find_package(liburing)
    if(NOT LIBURING_FOUND)
      if(fail-on-missing)
        message(FATAL_ERROR "liburing not found and uring option required")
      else()
        message(STATUS "liburing not found. Switching off uring option")
        set(uring OFF CACHE BOOL "Disabled because liburing was not found (${uring_description})" FORCE)
      endif()
    endif()
  endif()
endif()

#---Check for DAOS----------------------------------------------------------------
if (daos AND daos_mock)
  message(FATAL_ERROR "Options `daos` and `daos_mock` are mutually exclusive; only one of them should be specified.")
endif()
if (testing AND NOT daos AND NOT WIN32)
  set(daos_mock ON CACHE BOOL "Enable `daos_mock` if `testing` option was set" FORCE)
endif()

if (daos OR daos_mock)
  message(STATUS "Looking for libuuid")
  if(fail-on-missing)
    find_package(libuuid REQUIRED)
  else()
    find_package(libuuid)
    if(NOT libuuid_FOUND)
      message(STATUS "libuuid not found. Disabling DAOS support")
      set(daos OFF CACHE BOOL "Disabled (libuuid not found)" FORCE)
      set(daos_mock OFF CACHE BOOL "Disabled (libuuid not found)" FORCE)
    endif()
  endif()
endif()
if (daos)
  message(STATUS "Looking for DAOS")
  if(fail-on-missing)
    find_package(DAOS REQUIRED)
  else()
    find_package(DAOS)
    if(NOT DAOS_FOUND)
      message(STATUS "libdaos not found. Disabling DAOS support")
      set(daos OFF CACHE BOOL "Disabled (libdaos not found)" FORCE)
    endif()
  endif()
endif()

#---Check for TBB---------------------------------------------------------------------
if(imt AND NOT builtin_tbb)
  message(STATUS "Looking for TBB")
  if(fail-on-missing)
    find_package(TBB 2020 REQUIRED)
  else()
    find_package(TBB 2020)
    if(NOT TBB_FOUND)
      message(STATUS "TBB not found, enabling 'builtin_tbb' option")
      set(builtin_tbb ON CACHE BOOL "Enabled because imt is enabled, but TBB was not found" FORCE)
    endif()
  endif()

  # Check that the found TBB does not use captured exceptions. If the header
  # <tbb/tbb_config.h> does not exist, assume that we have oneTBB newer than
  # version 2021, which does not have captured exceptions anyway.
  if(TBB_FOUND AND EXISTS "${TBB_INCLUDE_DIRS}/tbb/tbb_config.h")
    set(CMAKE_REQUIRED_INCLUDES "${TBB_INCLUDE_DIRS}")
    check_cxx_source_compiles("
#include <tbb/tbb_config.h>
#if TBB_USE_CAPTURED_EXCEPTION == 1
#error TBB uses tbb::captured_exception, not suitable for ROOT!
#endif
int main() { return 0; }" tbb_exception_result)
    if(NOT tbb_exception_result)
      if(fail-on-missing)
        message(FATAL_ERROR "Found TBB uses tbb::captured_exception, not suitable for ROOT!")
      endif()
      message(STATUS "Found TBB uses tbb::captured_exception, enabling 'builtin_tbb' option")
      set(builtin_tbb ON CACHE BOOL "Enabled because imt is enabled and found TBB is not suitable" FORCE)
    endif()
  endif()

  set(TBB_CXXFLAGS "-DTBB_SUPPRESS_DEPRECATED_MESSAGES=1")
endif()

if(builtin_tbb AND NO_CONNECTION)
  if(fail-on-missing)
    message(FATAL_ERROR "No internet connection. Please check your connection, or either disable the 'builtin_tbb' option or the 'fail-on-missing' to automatically disable options requiring internet access")
  else()
    message(STATUS "No internet connection, disabling 'builtin_tbb' and 'imt' options")
    set(builtin_tbb OFF CACHE BOOL "Disabled because there is no internet connection" FORCE)
    set(imt OFF CACHE BOOL "Disabled because 'builtin_tbb' was set but there is no internet connection" FORCE)
  endif()
endif()

if(builtin_tbb)
  set(tbb_url ${lcgpackages}/oneTBB-2021.9.0.tar.gz)
  set(tbb_sha256 1ce48f34dada7837f510735ff1172f6e2c261b09460e3bf773b49791d247d24e)

  if(MSVC)
    set(tbb_build Release)
    if(winrtdebug)
      set(tbb_build Debug)
      set(tbbsuffix "_debug")
    endif()
    set(TBB_LIBRARIES ${CMAKE_BINARY_DIR}/lib/tbb12${tbbsuffix}.lib)
    ExternalProject_Add(
      TBB
      URL ${tbb_url}
      URL_HASH SHA256=${tbb_sha256}
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      CMAKE_ARGS -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} -DTBB_ENABLE_IPO=OFF -DTBB_TEST=Off -DTBB_STRICT=Off -DTBBMALLOC_BUILD=Off -DTBBMALLOC_PROXY_BUILD=Off "-DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}" "-DCMAKE_INSTALL_LIBDIR=${CMAKE_BINARY_DIR}/lib" "-DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_BINARY_DIR}/include"
      BUILD_COMMAND ${CMAKE_COMMAND} --build . --config ${tbb_build}
      INSTALL_COMMAND ${CMAKE_COMMAND}  --install . --config ${tbb_build}
      LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 LOG_OUTPUT_ON_FAILURE 1
      BUILD_BYPRODUCTS ${TBB_LIBRARIES}
      TIMEOUT 600
    )
    install(DIRECTORY ${CMAKE_BINARY_DIR}/bin/ DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT libraries FILES_MATCHING PATTERN "tbb*.dll")
    install(DIRECTORY ${CMAKE_BINARY_DIR}/lib/ DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries FILES_MATCHING PATTERN "tbb*.lib")
  else()
    set(TBB_LIBRARIES ${CMAKE_BINARY_DIR}/lib/libtbb${CMAKE_SHARED_LIBRARY_SUFFIX})
    ExternalProject_Add(
      TBB
      URL ${tbb_url}
      URL_HASH SHA256=${tbb_sha256}
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      CMAKE_ARGS -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} -DTBB_ENABLE_IPO=OFF -DTBB_TEST=Off -DTBB_STRICT=Off -DTBBMALLOC_BUILD=Off -DTBBMALLOC_PROXY_BUILD=Off "-DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}" "-DCMAKE_INSTALL_LIBDIR=${CMAKE_BINARY_DIR}/lib" "-DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_BINARY_DIR}/include"
      LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 LOG_OUTPUT_ON_FAILURE 1
      BUILD_BYPRODUCTS ${TBB_LIBRARIES}
      TIMEOUT 600
    )
    install(DIRECTORY ${CMAKE_BINARY_DIR}/lib/ DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries FILES_MATCHING PATTERN "libtbb*")
  endif()

  ExternalProject_Add_Step(
     TBB tbb2externals
     COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/include/tbb ${CMAKE_BINARY_DIR}/ginclude/tbb
     COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/include/oneapi ${CMAKE_BINARY_DIR}/ginclude/oneapi
     DEPENDEES install
  )
  set(TBB_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/ginclude)
  set(TBB_CXXFLAGS "-DTBB_SUPPRESS_DEPRECATED_MESSAGES=1")
  # The following line is needed to generate the proper dependency with: BUILTINS TBB (in Imt)
  # and generated with this syntax: add_dependencies(${library} ${${arg1}_TARGET})
  set(TBB_TARGET TBB)
endif()

#---Check for Vc---------------------------------------------------------------------
if(builtin_vc)
  unset(Vc_FOUND)
  unset(Vc_FOUND CACHE)
  set(vc ON CACHE BOOL "Enabled because builtin_vc requested (${vc_description})" FORCE)
elseif(vc)
  if(fail-on-missing)
    find_package(Vc 1.4.4 CONFIG QUIET REQUIRED)
  else()
    find_package(Vc 1.4.4 CONFIG QUIET)
    if(NOT Vc_FOUND)
      message(STATUS "Vc library not found, support for it disabled.")
      message(STATUS "Please enable the option 'builtin_vc' to build Vc internally.")
      set(vc OFF CACHE BOOL "Disabled because Vc not found (${vc_description})" FORCE)
    endif()
  endif()
  if(Vc_FOUND)
    set_property(DIRECTORY APPEND PROPERTY INCLUDE_DIRECTORIES ${Vc_INCLUDE_DIR})
  endif()
endif()

if(vc AND NOT Vc_FOUND AND NO_CONNECTION)
  if(fail-on-missing)
    message(FATAL_ERROR "No internet connection. Please check your connection, or either disable the 'vc' option or the 'fail-on-missing' to automatically disable options requiring internet access")
  else()
    message(STATUS "No internet connection, disabling the 'vc' option")
    set(vc OFF CACHE BOOL "Disabled because there is no internet connection" FORCE)
  endif()
endif()

if(vc AND NOT Vc_FOUND)
  set(Vc_VERSION "1.4.4")
  set(Vc_PROJECT "Vc-${Vc_VERSION}")
  set(Vc_SRC_URI "${lcgpackages}/${Vc_PROJECT}.tar.gz")
  set(Vc_DESTDIR "${CMAKE_BINARY_DIR}/externals")
  set(Vc_ROOTDIR "${Vc_DESTDIR}/${CMAKE_INSTALL_PREFIX}")
  set(Vc_LIBNAME "${CMAKE_STATIC_LIBRARY_PREFIX}Vc${CMAKE_STATIC_LIBRARY_SUFFIX}")
  set(Vc_LIBRARY "${Vc_ROOTDIR}/lib/${Vc_LIBNAME}")

  ExternalProject_Add(VC
    URL     ${Vc_SRC_URI}
    URL_HASH SHA256=5933108196be44c41613884cd56305df320263981fe6a49e648aebb3354d57f3
    BUILD_IN_SOURCE 0
    BUILD_BYPRODUCTS ${Vc_LIBRARY}
    LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1
    CMAKE_ARGS -G ${CMAKE_GENERATOR}
               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_CXX_FLAGS=${ROOT_EXTERNAL_CXX_FLAGS}
               -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
    INSTALL_COMMAND env DESTDIR=${Vc_DESTDIR} ${CMAKE_COMMAND} --build . --target install
    TIMEOUT 600
  )

  set(VC_TARGET Vc)
  set(Vc_LIBRARIES Vc)
  set(Vc_INCLUDE_DIR ${Vc_ROOTDIR}/include)
  set(Vc_CMAKE_MODULES_DIR ${Vc_ROOTDIR}/lib/cmake/Vc)

  add_library(VcExt STATIC IMPORTED)
  set_property(TARGET VcExt PROPERTY IMPORTED_LOCATION ${Vc_LIBRARY})
  add_dependencies(VcExt VC)

  add_library(Vc INTERFACE)
  target_include_directories(Vc SYSTEM BEFORE INTERFACE $<BUILD_INTERFACE:${Vc_INCLUDE_DIR}>)
  target_link_libraries(Vc INTERFACE VcExt)

  find_package_handle_standard_args(Vc
    FOUND_VAR Vc_FOUND
    REQUIRED_VARS Vc_INCLUDE_DIR Vc_LIBRARIES Vc_CMAKE_MODULES_DIR
    VERSION_VAR Vc_VERSION)

  # FIXME: This is a workaround to let ROOT find the headers at runtime if
  # they are in the build directory. This is necessary until we decide how to
  # treat externals with headers used by ROOT
  if(NOT EXISTS ${CMAKE_BINARY_DIR}/include/Vc)
    if (NOT EXISTS ${CMAKE_BINARY_DIR}/include)
      execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/include)
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${Vc_INCLUDE_DIR}/Vc ${CMAKE_BINARY_DIR}/include/Vc)
  endif()
  # end of workaround

  install(DIRECTORY ${Vc_ROOTDIR}/ DESTINATION ".")
endif()

if(Vc_FOUND)
  # Missing from VcConfig.cmake
  set(Vc_INCLUDE_DIRS ${Vc_INCLUDE_DIR})
endif()

#---Check for VecCore--------------------------------------------------------------------
if(builtin_veccore)
  unset(VecCore_FOUND)
  unset(VecCore_FOUND CACHE)
  set(veccore ON CACHE BOOL "Enabled because builtin_veccore requested (${veccore_description})" FORCE)
elseif(veccore)
  if(vc)
    set(VecCore_COMPONENTS Vc)
  endif()
  if(fail-on-missing)
    find_package(VecCore 0.4.2 CONFIG QUIET REQUIRED COMPONENTS ${VecCore_COMPONENTS})
  else()
    find_package(VecCore 0.4.2 CONFIG QUIET COMPONENTS ${VecCore_COMPONENTS})
    if(NOT VecCore_FOUND)
      if(NO_CONNECTION)
        message(STATUS "VecCore not found and no internet connection, disabling the 'veccore' option")
        set(veccore OFF CACHE BOOL "Disabled because not found and No internet connection" FORCE)
      else()
        message(STATUS "VecCore not found, switching on 'builtin_veccore' option.")
        set(builtin_veccore ON CACHE BOOL "Enabled because veccore requested and not found externally (${builtin_veccore_description})" FORCE)
      endif()
    endif()
  endif()
  if(VecCore_FOUND)
      set_property(DIRECTORY APPEND PROPERTY INCLUDE_DIRECTORIES ${VecCore_INCLUDE_DIRS})
  endif()
endif()

if(builtin_veccore AND NO_CONNECTION)
  if(fail-on-missing)
    message(FATAL_ERROR "No internet connection. Please check your connection, or either disable the 'builtin_veccore' option or the 'fail-on-missing' to automatically disable options requiring internet access")
  else()
    message(STATUS "No internet connection, disabling the 'builtin_veccore' option")
    set(builtin_veccore OFF CACHE BOOL "Disabled because there is no internet connection" FORCE)
  endif()
endif()

if(builtin_veccore)
  set(VecCore_VERSION "0.7.0")
  set(VecCore_PROJECT "VecCore-${VecCore_VERSION}")
  set(VecCore_SRC_URI "${lcgpackages}/${VecCore_PROJECT}.tar.gz")
  set(VecCore_DESTDIR "${CMAKE_BINARY_DIR}/externals")
  set(VecCore_ROOTDIR "${VecCore_DESTDIR}/${CMAKE_INSTALL_PREFIX}")

  ExternalProject_Add(VECCORE
    URL     ${VecCore_SRC_URI}
    URL_HASH SHA256=61d9fc4be815c5c98088c2796763d3ed82ba4bad5a69b7892c1c2e7e1e53d311
    BUILD_IN_SOURCE 0
    LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1
    CMAKE_ARGS -G ${CMAKE_GENERATOR}
               -DBUILD_TESTING=OFF
               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_CXX_FLAGS=${ROOT_EXTERNAL_CXX_FLAGS}
               -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
    INSTALL_COMMAND env DESTDIR=${VecCore_DESTDIR} ${CMAKE_COMMAND} --build . --target install
    TIMEOUT 600
  )

  set(VECCORE_TARGET VecCore)
  set(VecCore_LIBRARIES VecCore)
  list(APPEND VecCore_INCLUDE_DIRS ${VecCore_ROOTDIR}/include)

  add_library(VecCore INTERFACE)
  target_include_directories(VecCore SYSTEM INTERFACE $<BUILD_INTERFACE:${VecCore_ROOTDIR}/include>)
  add_dependencies(VecCore VECCORE)

  if (Vc_FOUND)
    set(VecCore_Vc_FOUND True)
    set(VecCore_Vc_DEFINITIONS -DVECCORE_ENABLE_VC)
    set(VecCore_Vc_INCLUDE_DIR ${Vc_INCLUDE_DIR})
    set(VecCore_Vc_LIBRARIES ${Vc_LIBRARIES})

    set(VecCore_DEFINITIONS ${VecCore_Vc_DEFINITIONS})
    list(APPEND VecCore_INCLUDE_DIRS ${VecCore_Vc_INCLUDE_DIR})
    set(VecCore_LIBRARIES ${VecCore_LIBRARIES} ${Vc_LIBRARIES})
    target_link_libraries(VecCore INTERFACE ${Vc_LIBRARIES})
  endif()

  find_package_handle_standard_args(VecCore
    FOUND_VAR VecCore_FOUND
    REQUIRED_VARS VecCore_INCLUDE_DIRS VecCore_LIBRARIES
    VERSION_VAR VecCore_VERSION)

  # FIXME: This is a workaround to let ROOT find the headers at runtime if
  # they are in the build directory. This is necessary until we decide how to
  # treat externals with headers used by ROOT
  if(NOT EXISTS ${CMAKE_BINARY_DIR}/include/VecCore)
    if (NOT EXISTS ${CMAKE_BINARY_DIR}/include)
      execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/include)
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${VecCore_ROOTDIR}/include/VecCore ${CMAKE_BINARY_DIR}/include/VecCore)
  endif()
  # end of workaround

  install(DIRECTORY ${VecCore_ROOTDIR}/ DESTINATION ".")
endif()

if(builtin_vdt AND NO_CONNECTION)
  if(fail-on-missing)
    message(FATAL_ERROR "No internet connection. Please check your connection, or either disable the 'builtin_vdt' option or the 'fail-on-missing' to automatically disable options requiring internet access")
  else()
    message(STATUS "No internet connection, disabling the 'builtin_vdt' option")
    set(builtin_vdt OFF CACHE BOOL "Disabled because there is no internet connection" FORCE)
  endif()
endif()

#---Check for Vdt--------------------------------------------------------------------
if(vdt OR builtin_vdt)
  if(NOT builtin_vdt)
    message(STATUS "Looking for VDT")
    find_package(Vdt 0.4)
    if(NOT VDT_FOUND)
      if(fail-on-missing)
        message(FATAL_ERROR "VDT not found. Ensure that the installation of VDT is in the CMAKE_PREFIX_PATH")
      else()
        message(STATUS "VDT not found. Ensure that the installation of VDT is in the CMAKE_PREFIX_PATH")
        if(NO_CONNECTION)
          set(vdt OFF CACHE BOOL "Disabled because not found and no internet connection" FORCE)
        else()
          message(STATUS "               Switching ON 'builtin_vdt' option")
          set(builtin_vdt ON CACHE BOOL "Enabled because external vdt not found (${vdt_description})" FORCE)
        endif()
      endif()
    endif()
  endif()
  if(builtin_vdt)
    set(vdt_version 0.4.4)
    set(VDT_FOUND True)
    set(VDT_LIBRARIES ${CMAKE_BINARY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}vdt${CMAKE_SHARED_LIBRARY_SUFFIX})
    ExternalProject_Add(
      VDT
      URL ${lcgpackages}/vdt-${vdt_version}.tar.gz
      URL_HASH SHA256=8b1664b45ec82042152f89d171dd962aea9bb35ac53c8eebb35df1cb9c34e498
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      CMAKE_ARGS
        -DSSE=OFF # breaks on ARM without this
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
        -DCMAKE_CXX_FLAGS=${ROOT_EXTERNAL_CXX_FLAGS}
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1
      BUILD_BYPRODUCTS ${VDT_LIBRARIES}
      TIMEOUT 600
    )
    ExternalProject_Add_Step(
       VDT copy2externals
       COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/include/vdt ${CMAKE_BINARY_DIR}/ginclude/vdt
       DEPENDEES install
    )
    set(VDT_INCLUDE_DIR ${CMAKE_BINARY_DIR}/ginclude)
    set(VDT_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/ginclude)
    install(FILES ${CMAKE_BINARY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}vdt${CMAKE_SHARED_LIBRARY_SUFFIX}
            DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries)
    install(DIRECTORY ${CMAKE_BINARY_DIR}/include/vdt
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT extra-headers)
    set(vdt ON CACHE BOOL "Enabled because builtin_vdt enabled (${vdt_description})" FORCE)
    set_property(GLOBAL APPEND PROPERTY ROOT_BUILTIN_TARGETS VDT)
    add_library(VDT::VDT STATIC IMPORTED GLOBAL)
    set_target_properties(VDT::VDT
      PROPERTIES
        IMPORTED_LOCATION "${VDT_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${VDT_INCLUDE_DIRS}"
    )
  endif()
endif()

#---Check for VecGeom--------------------------------------------------------------------
if (vecgeom)
  message(STATUS "Looking for VecGeom")
  find_package(VecGeom 1.2 CONFIG)
  if(builtin_veccore)
    message(WARNING "ROOT must be built against the VecCore installation that was used to build VecGeom; builtin_veccore cannot be used. Option VecGeom will be disabled.")
    set(vecgeom OFF CACHE BOOL "Disabled because non-builtin VecGeom specified but its VecCore cannot be found" FORCE)
  elseif(builtin_veccore AND fail-on-missing)
    message(FATAL_ERROR "ROOT must be built against the VecCore installation that was used to build VecGeom; builtin_veccore cannot be used. Ensure that builtin_veccore option is OFF.")
  endif()
  if(NOT VecGeom_FOUND )
    if(fail-on-missing)
      message(FATAL_ERROR "VecGeom not found. Ensure that the installation of VecGeom is in the CMAKE_PREFIX_PATH")
    else()
      message(STATUS "VecGeom not found. Ensure that the installation of VecGeom is in the CMAKE_PREFIX_PATH")
      message(STATUS "              example: CMAKE_PREFIX_PATH=<VecGeom_install_path>/lib/cmake/VecGeom")
      message(STATUS "              For the time being switching OFF 'vecgeom' option")
      set(vecgeom OFF CACHE BOOL "Disabled because VecGeom not found (${vecgeom_description})" FORCE)
    endif()
  else()
    message(STATUS "   Found VecGeom " ${VecGeom_VERSION})
  endif()
endif()

#---Check for protobuf-------------------------------------------------------------------

if(tmva-sofie)
  message(STATUS "Looking for Protobuf")
  set(protobuf_MODULE_COMPATIBLE TRUE)
  find_package(Protobuf CONFIG)
  if(NOT Protobuf_FOUND)
    find_package(Protobuf MODULE)
  endif()
  if(NOT Protobuf_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "Protobuf libraries not found and they are required (tmva-sofie option enabled)")
    else()
      message(STATUS "Protobuf not found. Switching off tmva-sofie option")
      set(tmva-sofie OFF CACHE BOOL "Disabled because Protobuf not found" FORCE)
    endif()
  else()
    if(Protobuf_VERSION LESS 3.0)
      if(fail-on-missing)
        message(FATAL_ERROR "Protobuf libraries found but is less than the version required (3.0) (tmva-sofie option enabled)")
      else()
        message(STATUS "Protobuf found but its version is not high enough (>3.0). Switching off tmva-sofie option")
        set(tmva-sofie OFF CACHE BOOL "Disabled because found Protobuf version is not enough" FORCE)
      endif()
    else()
      if(NOT TARGET protobuf::protoc)
        if(fail-on-missing)
          message(FATAL_ERROR "Protobuf compiler not found (tmva-sofie option enabled)")
        else()
          message(STATUS "Protobuf compiler not found. Switching off tmva-sofie option")
          set(tmva-sofie OFF CACHE BOOL "Disabled because Protobuf compiler not found" FORCE)
        endif()
      endif()
    endif()
  endif()
endif()

### Look for package CuDNN. If both cudnn and tmva-gpu are set and cudnn was
### found, it implies the tmva-cudnn flag.
if (cudnn)
  if (fail-on-missing)
    find_package(CUDNN REQUIRED)
  else()
    find_package(CUDNN)
  endif()
  if (CUDNN_FOUND)
    message(STATUS "CuDNN library found: " ${CUDNN_LIBRARIES})
    ### set tmva-cudnn flag only if tmva-gpu is on!
    if (tmva-gpu)
      set(tmva-cudnn ON)
    endif()
  else()
    message(STATUS "CUDNN library not found")
    set(cudnn OFF CACHE BOOL "Disabled because cudnn is not found" FORCE)
  endif()
endif(cudnn)

#---TMVA and its dependencies------------------------------------------------------------
if(tmva)
  if(tmva-cpu AND imt)
    message(STATUS "Looking for BLAS for optional parts of TMVA")
    find_package(BLAS)
    if(NOT BLAS_FOUND)
      if (GSL_FOUND)
        message(STATUS "Using GSL CBLAS for optional parts of TMVA")
      else()
        set(tmva-cpu OFF CACHE BOOL "Disabled because BLAS was not found (${tmva-cpu_description})" FORCE)
      endif()
    endif()
  else()
    set(tmva-cpu OFF CACHE BOOL "Disabled because 'imt' is disabled (${tmva-cpu_description})" FORCE)
  endif()
  if(tmva-gpu AND NOT CMAKE_CUDA_COMPILER)
    set(tmva-gpu OFF CACHE BOOL "Disabled because cuda not found" FORCE)
  endif()
  if(tmva-gpu)
    # So far, TMVA is the only package that uses the CUDA toolkit. RooFit is
    # just compiling libraries with the NVidia compiler itself. If more ROOT
    # components depend on the CUDA toolkit, this should be moved.
    find_package(CUDAToolkit REQUIRED)
  endif()
  if(tmva-pymva)
    if(fail-on-missing AND (NOT Python3_NumPy_FOUND OR NOT Python3_Development_FOUND))
      message(FATAL_ERROR "TMVA: numpy python package or Python development package not found and tmva-pymva component required"
                          " (python executable: ${PYTHON_EXECUTABLE})")
    elseif(NOT Python3_NumPy_FOUND OR NOT Python3_Development_FOUND)
      message(STATUS "TMVA: Numpy or Python development package not found for python ${PYTHON_EXECUTABLE}. Switching off tmva-pymva option")
      set(tmva-pymva OFF CACHE BOOL "Disabled because Numpy or Python development package were not found (${tmva-pymva_description})" FORCE)
    endif()
  endif()
  if (R_FOUND)
    #Rmva is enable when r is found and tmva is on
    set(tmva-rmva ON)
  endif()
  if(tmva-rmva AND NOT R_FOUND)
    set(tmva-rmva  OFF CACHE BOOL "Disabled because R was not found (${tmva-rmva_description})"  FORCE)
  endif()
else()
  set(tmva-cpu   OFF CACHE BOOL "Disabled because 'tmva' is disabled (${tmva-cpu_description})"   FORCE)
  set(tmva-gpu   OFF CACHE BOOL "Disabled because 'tmva' is disabled (${tmva-gpu_description})"   FORCE)
  set(tmva-pymva OFF CACHE BOOL "Disabled because 'tmva' is disabled (${tmva-pymva_description})" FORCE)
  set(tmva-rmva  OFF CACHE BOOL "Disabled because 'tmva' is disabled (${tmva-rmva_description})"  FORCE)
endif(tmva)

#---Check for PyROOT---------------------------------------------------------------------
if(pyroot)

  if(NOT Python3_Development_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "PyROOT: Python development package not found and pyroot component required"
                          " (python executable: ${PYTHON_EXECUTABLE})")
    else()
      message(STATUS "PyROOT: Python development package not found for python ${PYTHON_EXECUTABLE}. Switching off pyroot option")
      set(pyroot OFF CACHE BOOL "Disabled because Python development package was not found for ${PYTHON_EXECUTABLE}" FORCE)
    endif()
  endif()

  message(STATUS "PyROOT: development package found. Building for version ${PYTHON_VERSION_STRING}")

endif()

#---Check for MPI---------------------------------------------------------------------
if (mpi)
  message(STATUS "Looking for MPI")
  find_package(MPI)
  if(NOT MPI_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "MPI not found. Ensure that the installation of MPI is in the CMAKE_PREFIX_PATH."
        " Example: CMAKE_PREFIX_PATH=<MPI_install_path> (e.g. \"/usr/local/mpich\")")
    else()
      message(STATUS "MPI not found. Ensure that the installation of MPI is in the CMAKE_PREFIX_PATH")
      message(STATUS "     Example: CMAKE_PREFIX_PATH=<MPI_install_path> (e.g. \"/usr/local/mpich\")")
      message(STATUS "     For the time being switching OFF 'mpi' option")
      set(mpi OFF CACHE BOOL "Disabled because MPI not found (${mpi_description})" FORCE)
    endif()
  endif()
endif()

#---Check for ZeroMQ when building RooFit::MultiProcess--------------------------------------------

if (roofit_multiprocess)
  if(NOT builtin_zeromq)
    message(STATUS "Looking for ZeroMQ (libzmq)")
    # Clear cache before calling find_package(ZeroMQ),
    # necessary to be able to toggle builtin_zeromq and
    # not have find_package(ZeroMQ) find builtin ZeroMQ.
    foreach(suffix FOUND INCLUDE_DIR INCLUDE_DIRS LIBRARY LIBRARIES)
      unset(ZeroMQ_${suffix} CACHE)
    endforeach()

    # Temporarily prefer config mode over module mode, so that a CMake-installed system version
    # gets detected before looking for an autotools-installed system version (which the
    # FindZeroMQ.cmake module does).
    set(CMAKE_FIND_PACKAGE_PREFER_CONFIG_ORIGINAL_VALUE ${CMAKE_FIND_PACKAGE_PREFER_CONFIG})
    set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)

    if(fail-on-missing)
      find_package(ZeroMQ 4.3.5 REQUIRED)
    else()
      find_package(ZeroMQ 4.3.5)
      if(NOT ZeroMQ_FOUND)
        message(STATUS "ZeroMQ not found. Switching on builtin_zeromq option")
        set(builtin_zeromq ON CACHE BOOL "Enabled because ZeroMQ not found (${builtin_zeromq_description})" FORCE)
        # If the ZeroMQ system version is too old, we can't use the system C++
        # headers either (note that find_package(ZeroMQ) not only checks if the
        # library exists, but also if it's a recent version with zmq_ppoll).
        set(builtin_cppzmq ON CACHE BOOL "Enabled because ZeroMQ not found (${builtin_cppzmq_description})" FORCE)
      endif()
    endif()

    # Reset default find_package mode
    set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ${CMAKE_FIND_PACKAGE_PREFER_CONFIG_ORIGINAL_VALUE})
    unset(CMAKE_FIND_PACKAGE_PREFER_CONFIG_ORIGINAL_VALUE)
  endif()

  if(builtin_zeromq)
    list(APPEND ROOT_BUILTINS ZeroMQ)
    add_subdirectory(builtins/zeromq/libzmq)
  endif()

  if(NOT builtin_cppzmq)
    message(STATUS "Looking for ZeroMQ C++ bindings (cppzmq)")
    # Clear cache before calling find_package(cppzmq),
    # necessary to be able to toggle builtin_cppzmq and
    # not have find_package(cppzmq) find builtin cppzmq.
    foreach(suffix FOUND INCLUDE_DIR INCLUDE_DIRS)
      unset(cppzmq_${suffix} CACHE)
    endforeach()
    if(fail-on-missing)
      find_package(cppzmq REQUIRED)
    else()
      find_package(cppzmq QUIET)
      if(NOT cppzmq_FOUND)
        message(STATUS "ZeroMQ C++ bindings not found. Switching on builtin_cppzmq option")
        set(builtin_cppzmq ON CACHE BOOL "Enabled because ZeroMQ C++ bindings not found (${builtin_cppzmq_description})" FORCE)
      endif()
    endif()
  endif()

  if(builtin_cppzmq)
    list(APPEND ROOT_BUILTINS cppzmq)
    add_subdirectory(builtins/zeromq/cppzmq)
  endif()
endif (roofit_multiprocess)

#---Check for googletest---------------------------------------------------------------
if (testing)
  if (NOT builtin_gtest)
    if(fail-on-missing)
      find_package(GTest REQUIRED)
    else()
      find_package(GTest)
      if(NOT GTEST_FOUND)
        if(NO_CONNECTION)
          if(fail-on-missing)
            message(FATAL_ERROR "No internet connection and GTest was not found. Please check your connection, or either disable the 'testing' option or the 'fail-on-missing' to automatically disable options requiring internet access")
          else()
            message(STATUS "GTest not found, and no internet connection. Disabing the 'testing' option.")
            set(testing OFF CACHE BOOL "Disabled because testing requested and GTest not found (${builtin_gtest_description}) and there is no internet connection" FORCE)
          endif()
        else()
          message(STATUS "GTest not found, switching ON 'builtin_gtest' option.")
          set(builtin_gtest ON CACHE BOOL "Enabled because testing requested and GTest not found (${builtin_gtest_description})" FORCE)
        endif()
      endif()
    endif()
  else()
    if(NO_CONNECTION)
      if(fail-on-missing)
        message(FATAL_ERROR "No internet connection. Please check your connection, or either disable the 'testing' option or the 'builtin_gtest' option or the 'fail-on-missing' option to automatically disable options requiring internet access")
      else()
        message(STATUS "No internet connection, disabling the 'testing' and 'builtin_gtest' options")
        set(testing OFF CACHE BOOL "Disabled because there is no internet connection" FORCE)
        set(builtin_gtest OFF CACHE BOOL "Disabled because there is no internet connection" FORCE)
      endif()
    endif()
  endif()
endif()

if (builtin_gtest)
  # FIXME: Remove our version of gtest in roottest. We can reuse this one.
  # Add googletest
  # http://stackoverflow.com/questions/9689183/cmake-googletest

  set(_gtest_byproduct_binary_dir
    ${CMAKE_CURRENT_BINARY_DIR}/googletest-prefix/src/googletest-build)
  set(_gtest_byproducts
    ${_gtest_byproduct_binary_dir}/lib/libgtest.a
    ${_gtest_byproduct_binary_dir}/lib/libgtest_main.a
    ${_gtest_byproduct_binary_dir}/lib/libgmock.a
    ${_gtest_byproduct_binary_dir}/lib/libgmock_main.a
    )

  if(MSVC)
    set(gtestbuild "Release")
    if(winrtdebug)
      set(gtestbuild "Debug")
    endif()
    set(EXTRA_GTEST_OPTS
      -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=${_gtest_byproduct_binary_dir}/lib/
      -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL:PATH=${_gtest_byproduct_binary_dir}/lib/
      -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=${_gtest_byproduct_binary_dir}/lib/
      -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO:PATH=${_gtest_byproduct_binary_dir}/lib/
      -Dgtest_force_shared_crt=ON
      BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${gtestbuild})
  endif()
  if(APPLE)
    set(EXTRA_GTEST_OPTS
      -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT})
  endif()

  ExternalProject_Add(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_SHALLOW 1
    GIT_TAG release-1.12.1
    UPDATE_COMMAND ""
    # # Force separate output paths for debug and release builds to allow easy
    # # identification of correct lib in subsequent TARGET_LINK_LIBRARIES commands
    # CMAKE_ARGS -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=DebugLibs
    #            -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
    #            -Dgtest_force_shared_crt=ON
    CMAKE_ARGS -G ${CMAKE_GENERATOR}
                  -DCMAKE_BUILD_TYPE=Release
                  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                  -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                  -DCMAKE_CXX_FLAGS=${ROOT_EXTERNAL_CXX_FLAGS}
                  -DCMAKE_AR=${CMAKE_AR}
                  -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
                  ${EXTRA_GTEST_OPTS}
    # Disable install step
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${_gtest_byproducts}
    # Wrap download, configure and build steps in a script to log output
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
    TIMEOUT 600
  )

  # Specify include dirs for gtest and gmock
  ExternalProject_Get_Property(googletest source_dir)
  set(GTEST_INCLUDE_DIR ${source_dir}/googletest/include)
  set(GMOCK_INCLUDE_DIR ${source_dir}/googlemock/include)
  # Create the directories. Prevents bug https://gitlab.kitware.com/cmake/cmake/issues/15052
  file(MAKE_DIRECTORY ${GTEST_INCLUDE_DIR} ${GMOCK_INCLUDE_DIR})

  # Libraries
  ExternalProject_Get_Property(googletest binary_dir)
  set(_G_LIBRARY_PATH ${binary_dir}/lib/)

  # Use gmock_main instead of gtest_main because it initializes gtest as well.
  # Note: The libraries are listed in reverse order of their dependancies.
  foreach(lib gtest gtest_main gmock gmock_main)
    add_library(${lib} IMPORTED STATIC GLOBAL)
    set_target_properties(${lib} PROPERTIES
      IMPORTED_LOCATION "${_G_LIBRARY_PATH}${CMAKE_STATIC_LIBRARY_PREFIX}${lib}${CMAKE_STATIC_LIBRARY_SUFFIX}"
    )
    add_dependencies(${lib} googletest)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND
        ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL 9)
      target_compile_options(${lib} INTERFACE -Wno-deprecated-copy)
    endif()
  endforeach()
  target_include_directories(gtest INTERFACE ${GTEST_INCLUDE_DIR})
  target_include_directories(gmock INTERFACE ${GMOCK_INCLUDE_DIR})

  set_property(TARGET gtest PROPERTY IMPORTED_LOCATION ${_G_LIBRARY_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX})
  set_property(TARGET gtest_main PROPERTY IMPORTED_LOCATION ${_G_LIBRARY_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main${CMAKE_STATIC_LIBRARY_SUFFIX})
  set_property(TARGET gmock PROPERTY IMPORTED_LOCATION ${_G_LIBRARY_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}gmock${CMAKE_STATIC_LIBRARY_SUFFIX})
  set_property(TARGET gmock_main PROPERTY IMPORTED_LOCATION ${_G_LIBRARY_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}gmock_main${CMAKE_STATIC_LIBRARY_SUFFIX})

endif()

if(webgui AND NOT builtin_openui5 AND NO_CONNECTION)
  if(fail-on-missing)
    message(FATAL_ERROR "No internet connection. Please check your connection, or either enable the 'builtin_openui5' option or the 'fail-on-missing' to automatically disable options requiring internet access")
  else()
    message(STATUS "No internet connection, switching to 'builtin_openui5' option")
    set(builtin_openui5 ON CACHE BOOL "Enabled because there is no internet connection" FORCE)
  endif()
endif()

#------------------------------------------------------------------------------------
if(webgui)
  if(NOT "$ENV{OPENUI5DIR}" STREQUAL "" AND EXISTS "$ENV{OPENUI5DIR}/resources/sap-ui-core.js")
     # create symbolic link on existing openui5 installation
     # should be used only for debug purposes to be able try different openui5 version
     # cannot be used for installation purposes
     message(STATUS "openui5 - use from $ENV{OPENUI5DIR}, only for debug purposes")
     file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/ui5)
     execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
        $ENV{OPENUI5DIR} ${CMAKE_BINARY_DIR}/ui5/distribution)
  else()
    if(builtin_openui5)
      ExternalProject_Add(
        OPENUI5
        URL ${CMAKE_SOURCE_DIR}/builtins/openui5/openui5.tar.gz
        URL_HASH SHA256=cc6791ac9c064c3e3f8281870063c549fa5e3f1252095ccc5d542c33db9cf8b2
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        SOURCE_DIR ${CMAKE_BINARY_DIR}/ui5/distribution
        TIMEOUT 600
      )
    else()
      ExternalProject_Add(
        OPENUI5
        URL https://github.com/SAP/openui5/releases/download/1.98.0/openui5-runtime-1.98.0.zip
        URL_HASH SHA256=3d3db9ba001141019aae2cdf6eb6d34d655a3652308e03a335794f1231d21d2f
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        SOURCE_DIR ${CMAKE_BINARY_DIR}/ui5/distribution
        TIMEOUT 600
      )
    endif()
    install(DIRECTORY ${CMAKE_BINARY_DIR}/ui5/distribution/ DESTINATION ${CMAKE_INSTALL_OPENUI5DIR}/distribution/ COMPONENT libraries FILES_MATCHING PATTERN "*")
  endif()
  ExternalProject_Add(
    RENDERCORE
    URL ${CMAKE_SOURCE_DIR}/builtins/rendercore/RenderCore-1.3.tar.gz
    URL_HASH SHA256=5f5d885c69b35cd71cf630fd7f37728650bb144f7bb176382d7f4d9dbc3e8066
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    SOURCE_DIR ${CMAKE_BINARY_DIR}/ui5/eve7/rcore
    TIMEOUT 600
  )
  ExternalProject_Add(
     MATHJAX
     URL ${CMAKE_SOURCE_DIR}/documentation/doxygen/mathjax.tar.gz
     URL_HASH SHA256=c5e22e60430a65963a87ab4dcc8856b9be5bd434d3b3871f27ee65b584c3c3ea
     CONFIGURE_COMMAND ""
     BUILD_COMMAND ""
     INSTALL_COMMAND ""
     SOURCE_DIR ${CMAKE_BINARY_DIR}/js/mathjax/
     TIMEOUT 600
  )
  install(DIRECTORY ${CMAKE_BINARY_DIR}/ui5/eve7/rcore/ DESTINATION ${CMAKE_INSTALL_OPENUI5DIR}/eve7/rcore/ COMPONENT libraries FILES_MATCHING PATTERN "*")
endif()

#------------------------------------------------------------------------------------
# Check if we need libatomic to use atomic operations in the C++ code. On ARM systems
# we generally do. First just test if CMake is able to compile a test executable
# using atomic operations without the help of a library. Only if it can't do we start
# looking for libatomic for the build.
#
check_cxx_source_compiles("
#include <atomic>
#include <cstdint>
int main() {
   std::atomic<int> a1;
   int a1val = a1.load();
   (void)a1val;
   std::atomic<uint64_t> a2;
   uint64_t a2val = a2.load(std::memory_order_relaxed);
   (void)a2val;
   return 0;
}
" ROOT_HAVE_CXX_ATOMICS_WITHOUT_LIB)
set(ROOT_ATOMIC_LIBS)
if(NOT ROOT_HAVE_CXX_ATOMICS_WITHOUT_LIB)
  find_library(ROOT_ATOMIC_LIB NAMES atomic
    HINTS ENV LD_LIBRARY_PATH
    DOC "Path to the atomic library to use during the build")
  mark_as_advanced(ROOT_ATOMIC_LIB)
  if(ROOT_ATOMIC_LIB)
    set(ROOT_ATOMIC_LIBS ${ROOT_ATOMIC_LIB})
  endif()
endif()

#------------------------------------------------------------------------------------
# Check if the pyspark package is installed on the system.
# Needed to run tests of the distributed RDataFrame module that use pyspark.
# The functionality has been tested with pyspark 2.4 and above.
if(test_distrdf_pyspark)
  message(STATUS "Looking for PySpark")

  if(fail-on-missing)
    find_package(PySpark 2.4 REQUIRED)
  else()

    find_package(PySpark 2.4)
    if(NOT PySpark_FOUND)
      message(STATUS "Switching OFF 'test_distrdf_pyspark' option")
      set(test_distrdf_pyspark OFF CACHE BOOL "Disabled because PySpark not found" FORCE)
    endif()

  endif()

endif()

#------------------------------------------------------------------------------------
# Check if the dask package is installed on the system.
# Needed to run tests of the distributed RDataFrame module that use dask.
if(test_distrdf_dask)
  message(STATUS "Looking for Dask")

  if(fail-on-missing)
    find_package(Dask 2022.08.1 REQUIRED)
  else()

    find_package(Dask 2022.08.1)
    if(NOT Dask_FOUND)
      message(STATUS "Switching OFF 'test_distrdf_dask' option")
      set(test_distrdf_dask OFF CACHE BOOL "Disabled because Dask not found" FORCE)
    endif()

  endif()

endif()
