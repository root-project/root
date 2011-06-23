# Find the LZMA includes and library.
# 
# This module defines
# LZMA_INCLUDE_DIR, where to locate LZMA header files
# LZMA_LIBRARIES, the libraries to link against to use Pythia6
# LZMA_FOUND.  If false, you cannot build anything that requires Pythia6.

if(LZMA_CONFIG_EXECUTABLE)
  set(LZMA_FIND_QUIETLY 1)
endif()
set(LZMA_FOUND 0)

find_path(LZMA_INCLUDE_DIR lzma.h
  $ENV{LZMA_DIR}/include
  /usr/local/include
  /usr/include/lzma
  /usr/local/include/lzma
  /opt/lzma/include
  DOC "Specify the directory containing lzma.h"
)

find_library(LZMA_LIBRARY NAMES lzma PATHS
  $ENV{LZMA_DIR}/lib
  /usr/local/lzma/lib
  /usr/local/lib
  /usr/lib/lzma
  /usr/local/lib/lzma
  /usr/lzma/lib /usr/lib
  /usr/lzma /usr/local/lzma
  /opt/lzma /opt/lzma/lib
  DOC "Specify the lzma library here."
)

if(LZMA_INCLUDE_DIR AND LZMA_LIBRARY)
  set(LZMA_FOUND 1 )
  if(NOT LZMA_FIND_QUIETLY)
     message(STATUS "Found LZMA includes at ${LZMA_INCLUDE_DIR}")
     message(STATUS "Found LZMA library at ${LZMA_LIBRARY}")
  endif()
endif()

set(LZMA_LIBRARIES ${LZMA_LIBRARY})
mark_as_advanced(LZMA_FOUND LZMA_LIBRARY LZMA_INCLUDE_DIR)
