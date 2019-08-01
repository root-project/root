# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

include(ExternalProject)

# Clear cache variables set by find_package(PCRE)
# to ensure that we use the builtin version
foreach(var PCRE_FOUND PCRE_VERSION PCRE_INCLUDE_DIR PCRE_PCRE_LIBRARY PCRE_LIBRARIES)
  unset(${var} CACHE)
endforeach()

if(WIN32)
  if(winrtdebug)
    set(PCRE_POSTFIX $<$<CONFIG:Debug>:d>)
    set(pcre_config "Debug")
  else()
    set(pcre_config "Release")
  endif()
else()
  set(pcre_config $<CONFIG>)
endif()

set(PCRE_VERSION "8.42" CACHE INTERNAL "" FORCE)
set(PCRE_LIBNAME ${CMAKE_STATIC_LIBRARY_PREFIX}pcre${PCRE_POSTFIX}${CMAKE_STATIC_LIBRARY_SUFFIX})

# build byproducts only needed by Ninja
if("${CMAKE_GENERATOR}" STREQUAL "Ninja")
  set(PCRE_BYPRODUCTS
    <BINARY_DIR>/pcre.h
    <BINARY_DIR>/${PCRE_LIBNAME}
  )
endif()

ExternalProject_Add(PCRE
  URL ${CMAKE_CURRENT_SOURCE_DIR}/pcre-${PCRE_VERSION}.tar.gz
  URL_HASH SHA256=69acbc2fbdefb955d42a4c606dfde800c2885711d2979e356c0636efde9ec3b5

  LOG_DOWNLOAD  TRUE
  LOG_CONFIGURE TRUE
  LOG_BUILD     TRUE
  LOG_INSTALL   TRUE

  CMAKE_CACHE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    -DCMAKE_GENERATOR:STRING=${CMAKE_GENERATOR}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    -DCMAKE_BUILD_SHARED_LIBS:BOOL=FALSE
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE

  BUILD_COMMAND
    ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${pcre_config} --target pcre

  BUILD_BYPRODUCTS
    ${PCRE_BYPRODUCTS}

  INSTALL_COMMAND ""
)

ExternalProject_Get_Property(PCRE BINARY_DIR)

set(PCRE_FOUND TRUE CACHE INTERNAL "" FORCE)
set(PCRE_INCLUDE_DIR "${BINARY_DIR}" CACHE INTERNAL "" FORCE)
if(WIN32)
  set(PCRE_PCRE_LIBRARY "${BINARY_DIR}/${pcre_config}/${PCRE_LIBNAME}" CACHE INTERNAL "" FORCE)
else()
  set(PCRE_PCRE_LIBRARY "${BINARY_DIR}/${CMAKE_CFG_INTDIR}/${PCRE_LIBNAME}" CACHE INTERNAL "" FORCE)
endif()
set(PCRE_LIBRARIES "${PCRE_PCRE_LIBRARY}" CACHE INTERNAL "" FORCE)
set(PCRE_TARGET PCRE)

add_library(pcre_builtin INTERFACE)
target_include_directories(pcre_builtin INTERFACE $<BUILD_INTERFACE:${PCRE_INCLUDE_DIR}>)
target_link_libraries(pcre_builtin INTERFACE $<BUILD_INTERFACE:${PCRE_LIBRARIES}>)
add_library(PCRE::PCRE ALIAS pcre_builtin)
add_dependencies(pcre_builtin PCRE)

set_property(GLOBAL APPEND PROPERTY ROOT_BUILTIN_TARGETS PCRE)
