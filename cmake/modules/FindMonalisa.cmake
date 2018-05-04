# Find the Monalisa includes and library.
#
# This module defines
# MONALISA_INCLUDE_DIR, where to locate ApMon.h file
# MONALISA_LIBRARY, the libraries to link against to use Monalisa
# MONALISA_LIBRARIES, the libraries to link against to use Monalisa
# MONALISA_FOUND.  If false, you cannot build anything that requires Monalisa.

if(MONALISA_LIBRARY AND MONALISA_INCLUDE_DIR)
  set(MONALISA_FIND_QUIETLY TRUE)
endif()

find_path(MONALISA_INCLUDE_DIR ApMon.h
  ${MONALISA}
  $ENV{MONALISA}
  ${MONALISA}/include
  $ENV{MONALISA}/include
  ${MONALISA_DIR}/include
  $ENV{MONALISA_DIR}/include
  /usr/local/include
  /opt/alien/api/include
  /opt/monalisa/include
  /usr/include
  DOC "Specify the directory containing ApMon.h"
)

find_library(MONALISA_LIBRARY NAMES apmoncpp PATHS
  ${MONALISA}/.libs
  $ENV{MONALISA}/.libs
  ${MONALISA_DIR}/lib
  $ENV{MONALISA_DIR}/lib
  /usr/local/lib
  /opt/alien/api/lib
  /opt/monalisa/lib
  /usr/lib
  DOC "Specify the libapmoncpp library here."
)

set(MONALISA_LIBRARIES ${MONALISA_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Monalisa
  FOUND_VAR MONALISA_FOUND REQUIRED_VARS MONALISA_LIBRARY MONALISA_INCLUDE_DIR)

mark_as_advanced(MONALISA_LIBRARY MONALISA_INCLUDE_DIR)
