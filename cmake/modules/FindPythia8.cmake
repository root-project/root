# Find the Pythia8 includes and library.
# 
# This module defines
# PYTHIA8_INCLUDE_DIR, where to locate Pythia.h file
# PYTHIA8_LIBRARIES, the libraries to link against to use Pythia6
# PYTHIA8_FOUND.  If false, you cannot build anything that requires Pythia6.
# PYTHIA8_LIBRARY, where to find the libpythia8 library.

set(PYTHIA8_FOUND 0)

find_path(PYTHIA8_INCLUDE_DIR Pythia.h
  $ENV{PYTHIA8_DIR}/include
  /opt/pythia8/include
  /usr/local/include
  /usr/include
  /usr/include/pythia
  DOC "Specify the directory containing Pythia.h."
)

find_library(PYTHIA8_LIBRARY NAMES Pythia8 pythia8 PATHS
  $ENV{PYTHIA8_DIR}/lib
  /opt/pythia8/lib
  /usr/local/lib
  /usr/lib
  DOC "Specify the Pythia8 library here."
)

if(PYTHIA8_INCLUDE_DIR AND PYTHIA8_LIBRARY)
  set(PYTHIA8_FOUND 1 )
  message(STATUS "Found Pythia8 library at ${PYTHIA8_LIBRARY}")
endif()


set(PYTHIA8_LIBRARIES ${PYTHIA8_LIBRARY})

MARK_AS_ADVANCED( PYTHIA8_FOUND PYTHIA8_LIBRARY PYTHIA8_INCLUDE_DIR)
