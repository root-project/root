# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# - Locate FTGL library
# Defines:
#
#  FTGL_FOUND
#  FTGL_INCLUDE_DIR
#  FTGL_LIBRARY
#  FTGL_INCLUDE_DIRS (not cached)
#  FTGL_LIBRARIES (not cached)

find_path(FTGL_INCLUDE_DIR FTGL/ftgl.h
          HINTS $ENV{FTGL_ROOT_DIR}/include ${FTGL_ROOT_DIR}/include)

find_library(FTGL_LIBRARY NAMES ftgl
             HINTS $ENV{FTGL_ROOT_DIR}/lib ${FTGL_ROOT_DIR}/lib)

set(FTGL_INCLUDE_DIRS ${FTGL_INCLUDE_DIR})
set(FTGL_LIBRARIES ${FTGL_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FTGL DEFAULT_MSG FTGL_INCLUDE_DIR FTGL_LIBRARY)
mark_as_advanced(FTGL_FOUND FTGL_INCLUDE_DIR FTGL_LIBRARY)

if(FTGL_FOUND)
  set(FTGL_VERSION_SRC "${CMAKE_SOURCE_DIR}/cmake/modules/get_ftgl_version.cpp")
  set(VER_INCLUDE_DIRS ${FTGL_INCLUDE_DIR} ${FREETYPE_INCLUDE_DIR_ft2build})
  try_run(RUN_RESULT COMPILE_RESULT
      "${CMAKE_BINARY_DIR}"
      "${FTGL_VERSION_SRC}"
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${VER_INCLUDE_DIRS}"
      LINK_LIBRARIES ${FTGL_LIBRARY}
      COMPILE_OUTPUT_VARIABLE BUILD_LOG
      RUN_OUTPUT_VARIABLE FTGL_VERSION
  )
  if(COMPILE_RESULT AND RUN_RESULT EQUAL 0)
    message(STATUS "Detected FTGL version: ${FTGL_VERSION}")
  else()
    message(WARNING "Failed to detect FTGL version via compilation. ${BUILD_LOG}")
  endif()
  set(FTGL_VERSION  "${FTGL_VERSION}" CACHE INTERNAL "FTGL version")
endif()
