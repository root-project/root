# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# **PLEASE UPDATE ALSO THE FOLLOWING LINE WHEN UPDATING THE VERSION**
# 20 Apr 2023, https://github.com/unuran/unuran/releases/tag/unuran-1.11.0
set(ROOT_UNURAN_VERSION 1.11.0)
set(ROOT_UNURAN_URL "https://github.com/unuran/unuran/archive/refs/tags/unuran-${ROOT_UNURAN_VERSION}.tar.gz") # TODO Move to LCG
set(ROOT_UNURAN_HASH "e46c15eff050150966988ec56969526b60ce0b97120a7821aa96703d0f175623")
set(ROOT_UNURAN_PREFIX ${CMAKE_BINARY_DIR}/builtins/UNURAN-prefix)
set(ROOT_UNURAN_LIBRARY ${ROOT_UNURAN_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}ftgl${CMAKE_STATIC_LIBRARY_SUFFIX})

include(ExternalProject)

# Clear cache variables set by find_package(UNURAN)
# to ensure that we use the builtin version
foreach(var UNURAN_LIBRARIES UNURAN_LIBRARY UNURAN_LIBRARY_DEBUG UNURAN_LIBRARY_RELEASE UNURAN_FOUND UNURAN_VERSION UNURAN_INCLUDE_DIR UNURAN_LIBRARY UNURAN_LIBRARIES)
  unset(${var})
  unset(${var} CACHE)
endforeach()

if(WIN32 AND NOT CMAKE_GENERATOR MATCHES Ninja)
  if(winrtdebug)
    set(ROOT_UNURAN_BUILD_COMMAND_FLAGS "--config Debug")
  else()
    set(ROOT_UNURAN_BUILD_COMMAND_FLAGS "--config $<IF:$<CONFIG:Debug,RelWithDebInfo>,RelWithDebInfo,Release>")
  endif()
endif()

ExternalProject_Add(
  BUILTIN_UNURAN
  PREFIX ${ROOT_UNURAN_PREFIX}
  URL ${ROOT_UNURAN_URL}
  URL_HASH SHA256=${ROOT_UNURAN_HASH}

  LOG_DOWNLOAD  TRUE
  LOG_CONFIGURE TRUE
  LOG_BUILD     TRUE
  LOG_INSTALL   TRUE
  LOG_OUTPUT_ON_FAILURE TRUE
  INSTALL_DIR ${ROOT_UNURAN_PREFIX}
  CONFIGURE_COMMAND ./autogen.sh COMMAND ./configure --prefix <INSTALL_DIR>
                    --libdir=<INSTALL_DIR>/lib
                    --disable-shared --with-pic
                    CC=${CMAKE_C_COMPILER}
                    CFLAGS=${CMAKE_C_FLAGS}
                    CPPFLAGS=${_gsl_cppflags}
                    LDFLAGS=${_gsl_ldflags}
                     COMMAND touch ./doc/version.texi COMMAND touch ./doc/version_win32.texi # See https://github.com/unuran/unuran/issues/1
  BUILD_IN_SOURCE TRUE
  BUILD_BYPRODUCTS
    ${ROOT_UNURAN_LIBRARY}
  TIMEOUT 600
)

set(ROOT_UNURAN_INCLUDE_DIR ${ROOT_UNURAN_PREFIX}/include)
file(MAKE_DIRECTORY ${ROOT_UNURAN_INCLUDE_DIR})
add_library(Unuran::unuran IMPORTED STATIC GLOBAL)
add_dependencies(Unuran::unuran BUILTIN_UNURAN)
target_compile_definitions(Unuran::unuran INTERFACE HAVE_CONFIG_H)
set_target_properties(Unuran::unuran PROPERTIES
    IMPORTED_LOCATION ${ROOT_UNURAN_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${ROOT_UNURAN_INCLUDE_DIR})

# Set the canonical output of find_package according to
# https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#standard-variable-names
set(Unuran_INCLUDE_DIRS ${ROOT_UNURAN_INCLUDE_DIR} PARENT_SCOPE)
set(Unuran_LIBRARIES ${ROOT_UNURAN_LIBRARY} PARENT_SCOPE)
set(Unuran_FOUND TRUE PARENT_SCOPE)
set(Unuran_VERSION ${ROOT_UNURAN_VERSION} PARENT_SCOPE)

# For compatibility with non-standard case FindUnuran.cmake
set(UNURAN_INCLUDE_DIRS ${ROOT_UNURAN_INCLUDE_DIR} PARENT_SCOPE)
set(UNURAN_LIBRARIES ${ROOT_UNURAN_LIBRARY} PARENT_SCOPE)
set(UNURAN_FOUND TRUE PARENT_SCOPE)
set(UNURAN_VERSION ${ROOT_UNURAN_VERSION} PARENT_SCOPE)

#  #---Define special compiler settings for unurun-----------------------------------------------------
#  set(UNR_CC ${CMAKE_C_COMPILER})
#  if(ROOT_ARCHITECTURE MATCHES hpuxia64acc)
#    set(UNR_CC "${UNR_CC} +DD64 -Ae")
#  elseif(ROOT_ARCHITECTURE MATCHES linuxppc64gcc)
#    set(UNR_CC "${UNR_CC} -m64 -fPIC")
#  elseif(ROOT_ARCHITECTURE MATCHES linuxx8664gcc)
#    set(UNR_CFLAGS "-m64 -fPIC")
#  elseif(ROOT_ARCHITECTURE MATCHES linuxicc)
#    set(UNR_CFLAGS "-m32")
#  elseif(ROOT_ARCHITECTURE MATCHES linuxx8664icc)
#    set(UNR_CFLAGS "-m64")
#  elseif(ROOT_ARCHITECTURE MATCHES win32 OR ROOT_ARCHITECTURE MATCHES win64)
#    set(UNR_CFLAGS "-MD -G5 -GX")
#  endif()
#  if(CMAKE_OSX_SYSROOT)
#    set(UNR_CFLAGS "${UNR_CFLAGS} -isysroot ${CMAKE_OSX_SYSROOT}")
#  endif()
#
##---We need to disable some warnings-------------------------------------------------------------------
#string(REPLACE -Wall "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
#if(${CMAKE_CXX_COMPILER_ID} MATCHES Clang)
#  ROOT_ADD_C_FLAG(CMAKE_C_FLAGS -Wno-parentheses-equality)
#  ROOT_ADD_C_FLAG(CMAKE_C_FLAGS -Wno-deprecated-non-prototype)
#endif()
#if(${CMAKE_CXX_COMPILER_ID} MATCHES GNU)
#  ROOT_ADD_C_FLAG(CMAKE_C_FLAGS -Wno-maybe-uninitialized)
#  ROOT_ADD_C_FLAG(CMAKE_C_FLAGS -Wno-alloc-size-larger-than)
#
#endif()
