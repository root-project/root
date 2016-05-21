# Find the LZ4 includes and library.
#
# This module defines
# LZ4_INCLUDE_DIR, where to locate LZ4 header files
# LZ4_LIBRARIES, the libraries to link against to use LZ4
# LZ4_FOUND.  If false, you cannot build anything that requires LZ4.

if(LZ4_CONFIG_EXECUTABLE)
  set(LZ4_FIND_QUIETLY 1)
endif()
set(LZ4_FOUND 0)

find_path(LZ4_INCLUDE_DIR lz4.h
  $ENV{LZ4_DIR}/include
  /usr/include
  /usr/local/include
  /opt/lz4/include
  DOC "Specify the directory containing lz4.h"
)

find_library(LZ4_LIBRARY NAMES lz4 PATHS
  $ENV{LZ4_DIR}/lib
  /usr/local/lzo/lib
  /usr/local/lib
  /usr/lib/lzo
  /usr/local/lib/lzo
  /usr/lzo/lib /usr/lib
  /usr/lzo /usr/local/lzo
  /opt/lzo /opt/lzo/lib
  DOC "Specify the lz4 library here."
)

if(LZ4_INCLUDE_DIR AND LZ4_LIBRARY)
  set(LZ4_FOUND 1)
  if(NOT LZ4_FIND_QUIETLY)
     message(STATUS "Found LZ4 includes at ${LZ4_INCLUDE_DIR}")
     message(STATUS "Found LZ4 library at ${LZ4_LIBRARY}")
  endif()
endif()

set(LZ4_LIBRARIES ${LZ4_LIBRARY})
mark_as_advanced(LZ4_FOUND LZ4_LIBRARY LZ4_INCLUDE_DIR)
