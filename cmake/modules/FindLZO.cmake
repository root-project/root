# Find the LZO includes and library.
#
# This module defines
# LZO_INCLUDE_DIR, where to locate LZO header files
# LZO_LIBRARIES, the libraries to link against to use LZO
# LZO_FOUND.  If false, you cannot build anything that requires LZO.
#
# inspired by @zzxuanyuan's pull request https://github.com/root-mirror/root/pull/81

if(LZO_CONFIG_EXECUTABLE)
  set(LZO_FIND_QUIETLY 1)
endif()
set(LZO_FOUND 0)

if(NOT LZO_DIR)
  set(LZO_DIR $ENV{LZO_DIR})
endif()

find_path(LZO_INCLUDE_DIR lzo/lzoutil.h
  ${LZO_DIR}/include
  /usr/include
  /usr/local/include
  /opt/lzo/include
  DOC "Specify the directory containing lzoutil.h"
)

find_library(LZO_LIBRARY NAMES lzo2 PATHS
  ${LZO_DIR}/lib
  /usr/local/lzo/lib
  /usr/local/lib
  /usr/lib/lzo
  /usr/local/lib/lzo
  /usr/lzo/lib /usr/lib
  /usr/lzo /usr/local/lzo
  /opt/lzo /opt/lzo/lib
  DOC "Specify the lzo2 library here."
)

if(LZO_INCLUDE_DIR AND LZO_LIBRARY)
  set(LZO_FOUND 1)
  if(NOT LZO_FIND_QUIETLY)
     message(STATUS "Found LZO includes at ${LZO_INCLUDE_DIR}")
     message(STATUS "Found LZO library at ${LZO_LIBRARY}")
  endif()
endif()

set(LZO_LIBRARIES ${LZO_LIBRARY})
mark_as_advanced(LZO_FOUND LZO_LIBRARY LZO_INCLUDE_DIR)
