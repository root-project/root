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
  try_compile(FTGL_VERSION_API
      SOURCES "${FTGL_VERSION_SRC}"
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${VER_INCLUDE_DIRS}"
      LINK_LIBRARIES ${FTGL_LIBRARY}
      OUTPUT_VARIABLE FTGL_VERSION_API_LOG
  )
  if (FTGL_VERSION_API)
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
      message(SEND_ERROR "Could not detect FTGL version: ${BUILD_LOG}")
    endif()
  else()
    message(STATUS "Failed to detect FTGL version via compilation (due to ancient system libftgl version).")
    # ie the FTGL version is before 2.2.0dev (May 23, 2010) https://github.com/frankheckenbach/ftgl/commit/b066d7826070749499011c4f37c764b1610071ad where VERSION API was introduced
    # sourceforge last version is 2.1.3.rc5  (June 12, 2008) so widely used
    # Let's attempt to know if it's 2.1.3dev or earlier by looking at commits close to May 23 2008 when UTF8 support was added: https://github.com/ulrichard/ftgl/commit/84869ec7da984493b3cd268b9e56b80e7c78ac82#diff-1462936246cba8b5e9ee1c79a1207cea8efb0658be58cd25e7d3acd817ac0ac0R374
    # May 19, 2008 is close enough and defines new headers, so just check for their existence as proxy https://github.com/frankheckenbach/ftgl/commit/f7d0017882321606d9fdb9e2899e226254d5cc69#diff-4c5d896d82d5e966afbaa229162d7cf914a87a1b1cc3f844ce3ebbc495d71a05
    if(EXISTS ${FTGL_INCLUDE_DIR}/FTGL/FTBuffer.h)
      set(HAS_UTF8 TRUE)
    endif()
    if(HAS_UTF8)
      # Must be before 2.2.0 (May 23, 2010) and after 2.1.3.rc5 (May 23 / June 12, 2008)
      message(STATUS "Educated guess: FTGL system version may be 2.1.3.rc5 or later")
      set(FTGL_VERSION 2.1.3)
    else()
      # Must be before 2.1.3.rc5 (May 23, 2008)
      message(STATUS "Educated guess: FTGL system version is 2.1.2 or earlier")
      set(FTGL_VERSION 2.1.2)
    endif()
  endif()
  set(FTGL_VERSION "${FTGL_VERSION}" CACHE INTERNAL "FTGL version")
endif()
