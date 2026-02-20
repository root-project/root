# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/clib package
############################################################################

set(CLIB_HEADERS
  strlcpy.h
  snprintf.h
  strtok.h
)

set(RELATIVE_INC_HEADERS ${CLIB_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)

set(CLIB_RES_HEADERS
  mmalloc.h
  mmconfig.h
  mmprivate.h
)

set(RELATIVE_RES_HEADERS ${CLIB_RES_HEADERS})
list(TRANSFORM RELATIVE_RES_HEADERS PREPEND res/)

set_property(TARGET Core APPEND PROPERTY DICT_HEADERS
  ${CLIB_HEADERS}
)

ROOT_OBJECT_LIBRARY(Clib
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

if(NOT CMAKE_VERSION VERSION_LESS "3.23.0")
  target_sources(
      Clib
      PRIVATE
      FILE_SET private_header_files
      TYPE HEADERS
      BASE_DIRS inc/ res/
      FILES
          ${RELATIVE_INC_HEADERS}
          ${RELATIVE_RES_HEADERS}
  )
endif()

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
