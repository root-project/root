# Find the ZOPFLI includes and library.
#
# This module defines
# ZOPFLI_INCLUDE_DIR, where to locate ZOPFLI header files
# ZOPFLI_LIBRARIES, the libraries to link against to use ZOPFLI
# ZOPFLI_FOUND.  If false, you cannot build anything that requires ZOPFLI.

if(ZOPFLI_CONFIG_EXECUTABLE)
  set(ZOPFLI_FIND_QUIETLY 1)
endif()
set(ZOPFLI_FOUND 0)

if(NOT ZOPFLI_DIR)
  set(ZOPFLI_DIR $ENV{ZOPFLI_DIR})
endif()

find_path(ZOPFLI_INCLUDE_DIR zopfli.h PATHS
  ${ZOPFLI_DIR}/include
  /usr/include
  /usr/local/include
  /opt/zopfli/include
  NO_DEFAULT_PATH
  NO_SYSTEM_ENVIRONMENT_PATH
  DOC "Specify the directory containing zopfli.h"
)

find_library(ZOPFLI_LIBRARY NAMES zopfli PATHS
  ${ZOPFLI_DIR}/lib
  /usr/local/zopfli/lib
  /usr/local/lib
  /usr/lib/zopfli
  /usr/local/lib/zopfli
  /usr/zopfli/lib /usr/lib
  /usr/zopfli /usr/local/zopfli
  /opt/zopfli /opt/zopfli/lib
  NO_DEFAULT_PATH
  NO_SYSTEM_ENVIRONMENT_PATH
  DOC "Specify the zopfli library here."
)

if(ZOPFLI_INCLUDE_DIR)
  message(STATUS "Found ZOPFLI includes at ${ZOPFLI_INCLUDE_DIR}")
else()
  message(STATUS "ZOPFLI includes not found")
endif()

if(ZOPFLI_LIBRARY)
  message(STATUS "Found ZOPFLI library at ${ZOPFLI_LIBRARY}")
else()
  message(STATUS "ZOPFLI library not found")
endif()


if(ZOPFLI_INCLUDE_DIR AND ZOPFLI_LIBRARY)
  set(ZOPFLI_FOUND 1)
endif()

set(ZOPFLI_LIBRARIES ${ZOPFLI_LIBRARY})
mark_as_advanced(ZOPFLI_FOUND ZOPFLI_LIBRARY ZOPFLI_INCLUDE_DIR)
