# - Try to find CASTOR 
#  (See http://savannah.cern.ch/files/?group=castor)
#  Check for rfio_api.h, stager_api.h for CASTOR 2 and libshift
#
#  CASTOR_INCLUDE_DIR - where to find rfio_api.h, etc.
#  CASTOR_LIBRARIES   - List of libraries when using ....
#  CASTOR_FOUND       - True if CASTOR 2  libraries found.

set(CASTOR_FOUND FALSE)
set(CASTOR_LIBRARIES)

if(CASTOR_INCLUDE_DIR)
  set(CASTOR_FIND_QUIETLY 1)
endif()

find_path(CASTOR_INCLUDE_DIR NAMES rfio_api.h PATHS 
  ${CASTOR_DIR}/include $ENV{CASTOR_DIR}/include
  /cern/pro/include
  /cern/new/include
  /cern/old/include
  /opt/shift/include
  /usr/local/shift/include
  /usr/include/shift
  /usr/local/include/shift
  PATH_SUFFIXES shift
)

if(CASTOR_INCLUDE_DIR)
  file(READ ${CASTOR_INCLUDE_DIR}/patchlevel.h contents)
  string(REGEX MATCH   "BASEVERSION[ ]*[\"][ ]*([^ \"]+)" cont ${contents})
  string(REGEX REPLACE "BASEVERSION[ ]*[\"][ ]*([^ \"]+)" "\\1" CASTOR_VERSION ${cont})
endif()

set(locations ${CASTOR_DIR} $ENV{CASTOR_DIR}
              /cern/pro /cern/new /cern/old
              /opt/shift /usr/local/shift
              /usr/lib/shift /usr/local/lib/shift
)

find_library(CASTOR_shift_LIBRARY NAMES shift shiftmd HINTS ${locations})
find_library(CASTOR_rfio_LIBRARY NAMES castorrfio HINTS ${locations})
find_library(CASTOR_common_LIBRARY NAMES castorcommon HINTS ${locations})
find_library(CASTOR_client_LIBRARY NAMES castorclient castorClient HINTS ${locations})
find_library(CASTOR_ns_LIBRARY NAMES castorns HINTS ${locations})

if(CASTOR_shift_LIBRARY)
  message(STATUS "Found Castor LIB AT ${CASTOR_shift_LIBRARY}")
  set(CASTOR_LIBRARIES ${CASTOR_LIBRARIES} ${CASTOR_shift_LIBRARY})
endif()

if(CASTOR_INCLUDE_DIR AND CASTOR_LIBRARIES)
  set(CASTOR_FOUND TRUE)
  if(NOT CASTOR_FIND_QUIETLY)
    message(STATUS "Found Castor version ${CASTOR_VERSION} at ${CASTOR_INCLUDE_DIR}")
  endif()
endif()

mark_as_advanced(
  CASTOR_shift_LIBRARY
  CASTOR_rfio_LIBRARY
  CASTOR_common_LIBRARY
  CASTOR_client_LIBRARY
  CASTOR_ns_LIBRARY
  CASTOR_INCLUDE_DIR
)


