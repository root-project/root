# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# **PLEASE UPDATE ALSO THE FOLLOWING LINE WHEN UPDATING THE VERSION**
# 21 Oct 2025, https://github.com/PCRE2Project/pcre2/releases/tag/pcre2-10.47
set(ROOT_PCRE2_VERSION 10.47)
set(ROOT_PCRE2_HASH "47fe8c99461250d42f89e6e8fdaeba9da057855d06eb7fc08d9ca03fd08d7bc7")

include(ExternalProject)

# Clear cache variables set by find_package(PCRE) and/or find_package(PCRE2)
# to ensure that we use the builtin version
foreach(var PCRE_FOUND PCRE_VERSION PCRE_INCLUDE_DIR PCRE_PCRE_LIBRARY PCRE_LIBRARIES PCRE2_FOUND PCRE2_VERSION PCRE2_INCLUDE_DIR PCRE2_PCRE2_LIBRARY PCRE2_LIBRARIES)
  unset(${var})
  unset(${var} CACHE)
endforeach()

if(WIN32)
  if(CMAKE_GENERATOR MATCHES Ninja)
    if (CMAKE_BUILD_TYPE MATCHES Debug)
      set(PCRE2_POSTFIX d)
    endif()
  else()
    if(winrtdebug)
      set(PCRE2_POSTFIX $<$<CONFIG:Debug>:d>)
      set(pcre2_config_kind "Debug")
    else()
      set(pcre2_config_kind "Release")
    endif()
    set(pcre2_config "--config ${pcre2_config_kind}")
  endif()
endif()

set(PCRE2_VERSION ${ROOT_PCRE2_VERSION} CACHE INTERNAL "" FORCE)
if(MSVC)
  set(PCRE2_LIBNAME ${CMAKE_STATIC_LIBRARY_PREFIX}pcre2-8-static${PCRE2_POSTFIX}${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  set(PCRE2_LIBNAME ${CMAKE_STATIC_LIBRARY_PREFIX}pcre2-8${PCRE2_POSTFIX}${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

# build byproducts only needed by Ninja
if("${CMAKE_GENERATOR}" STREQUAL "Ninja")
  set(PCRE2_BYPRODUCTS
    <BINARY_DIR>/interface/pcre2.h
    <BINARY_DIR>/${PCRE2_LIBNAME}
  )
endif()

ExternalProject_Add(BUILTIN_PCRE
  URL ${lcgpackages}/pcre2-${PCRE2_VERSION}.tar.bz2
  URL_HASH SHA256=${ROOT_PCRE2_HASH}

  LOG_DOWNLOAD  TRUE
  LOG_CONFIGURE TRUE
  LOG_BUILD     TRUE
  LOG_INSTALL   TRUE
  LOG_OUTPUT_ON_FAILURE TRUE

  CMAKE_CACHE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    -DCMAKE_GENERATOR:STRING=${CMAKE_GENERATOR}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    -DCMAKE_BUILD_SHARED_LIBS:BOOL=FALSE
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE
    -DPCRE2_BUILD_TESTS:BOOL=FALSE

  BUILD_COMMAND
    ${CMAKE_COMMAND} --build <BINARY_DIR> ${pcre2_config} --target pcre2-8-static

  BUILD_BYPRODUCTS
    ${PCRE2_BYPRODUCTS}

  INSTALL_COMMAND ""
  TIMEOUT 600
)

ExternalProject_Get_Property(BUILTIN_PCRE BINARY_DIR)

set(PCRE2_FOUND TRUE CACHE INTERNAL "" FORCE)
set(PCRE2_INCLUDE_DIR "${BINARY_DIR}/interface" CACHE INTERNAL "" FORCE)
if(WIN32)
  set(PCRE2_PCRE2_LIBRARY "${BINARY_DIR}/${pcre2_config_kind}/${PCRE2_LIBNAME}" CACHE INTERNAL "" FORCE)
else()
  set(PCRE2_PCRE2_LIBRARY "${BINARY_DIR}/${CMAKE_CFG_INTDIR}/${PCRE2_LIBNAME}" CACHE INTERNAL "" FORCE)
endif()
set(PCRE2_LIBRARIES "${PCRE2_PCRE2_LIBRARY}" CACHE INTERNAL "" FORCE)
set(PCRE2_TARGET PCRE2)

add_library(pcre2_builtin INTERFACE)
target_include_directories(pcre2_builtin INTERFACE $<BUILD_INTERFACE:${PCRE2_INCLUDE_DIR}>)
target_link_libraries(pcre2_builtin INTERFACE $<BUILD_INTERFACE:${PCRE2_LIBRARIES}>)
add_library(PCRE2::PCRE2 ALIAS pcre2_builtin)
add_dependencies(pcre2_builtin PCRE2)

set_property(GLOBAL APPEND PROPERTY ROOT_BUILTIN_TARGETS BUILTIN_PCRE)
