# Find the Monalisa includes and library.
#
# This module defines
# MONALISA_INCLUDE_DIR, where to locate ApMon.h file
# MONALISA_LIBRARY, the libraries to link against to use Monalisa
# MONALISA_LIBRARIES, the libraries to link against to use Monalisa
# MONALISA_FOUND.  If false, you cannot build anything that requires Monalisa.

set(MONALISA_FOUND 0)
if(MONALISA_LIBRARY AND MONALISA_INCLUDE_DIR)
  set(MONALISA_FIND_QUIETLY TRUE)
endif()

find_path(MONALISA_INCLUDE_DIR ApMon.h
  ${MONALISA_DIR}/include
  $ENV{MONALISA_DIR}/include
  /usr/local/include
  /opt/alien/api/include
  /opt/monalisa/include
  /usr/include
  DOC "Specify the directory containing ApMon.h"
)

find_library(MONALISA_LIBRARY NAMES apmoncpp PATHS
  ${MONALISA_DIR}/lib
  $ENV{MONALISA_DIR}/lib
  /usr/local/lib
  /opt/alien/api/lib
  /opt/monalisa/lib
  /usr/lib
  DOC "Specify the libapmoncpp library here."
)

if(MONALISA_INCLUDE_DIR AND MONALISA_LIBRARY)
  set(MONALISA_FOUND 1 )
  if(NOT MONALISA_FIND_QUIETLY)
     message(STATUS "Found Monalisa includes at ${MONALISA_INCLUDE_DIR}")
     message(STATUS "Found Monalisa library at ${MONALISA_LIBRARY}")
  endif()
endif()

set(MONALISA_LIBRARIES ${MONALISA_LIBRARY})

mark_as_advanced(MONALISA_LIBRARY MONALISA_INCLUDE_DIR)
