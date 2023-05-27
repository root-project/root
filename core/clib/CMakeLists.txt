# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/clib package
############################################################################

set_property(TARGET Core APPEND PROPERTY DICT_HEADERS
  strlcpy.h
  snprintf.h
  strtok.h
)

ROOT_OBJECT_LIBRARY(Clib
  src/Demangle.c
  src/attach.c
  src/detach.c
  src/getpagesize.c
  src/keys.c
  src/mcalloc.c
  src/mfree.c
  src/mmalloc.c
  src/mmapsup.c
  src/mmcheck.c
  src/mrealloc.c
  src/sbrksup.c
  src/snprintf.c
  src/strlcat.c
  src/strlcpy.c
)

target_include_directories(Clib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
  PRIVATE
    ${CMAKE_BINARY_DIR}/ginclude
    ${CMAKE_CURRENT_SOURCE_DIR}/res
    ${CMAKE_CURRENT_SOURCE_DIR}/../foundation/inc # for RConfig.hxx
)

target_include_directories(Core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_sources(Core PRIVATE $<TARGET_OBJECTS:Clib>)

ROOT_INSTALL_HEADERS()
