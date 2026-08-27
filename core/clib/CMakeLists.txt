# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/clib package
# This library contains string functions that first were introduced in BSD.
# None of its headers should be used in installed ROOT headers.
############################################################################

add_library(ROOTclib STATIC
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

target_include_directories(ROOTclib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
  PRIVATE
    ${CMAKE_BINARY_DIR}/ginclude # RConfigure.h
    ${CMAKE_CURRENT_SOURCE_DIR}/res
    ${CMAKE_CURRENT_SOURCE_DIR}/../foundation/inc # for RConfig.hxx
)
target_link_options(ROOTclib PRIVATE -fvisibility=hidden)
set_target_properties(ROOTclib PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
