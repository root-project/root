# Find the Pythia6 includes and library.
# 
# This module defines
# PYTHIA6_LIBRARIES, the libraries to link against to use Pythia6
# PYTHIA6_FOUND.  If false, you cannot build anything that requires Pythia6.
# PYTHIA6_LIBRARY, where to find the libPythia6 library.

set(PYTHIA6_FOUND 0)


find_library(PYTHIA6_LIBRARY NAMES pythia6 libPythia6 PATHS
  $ENV{PYTHIA6_DIR}/lib
  /cern/pro/lib 
  /opt/pythia 
  /opt/pythia6
  /usr/lib/pythia
  /usr/local/lib/pythia
  /usr/lib/pythia6
  /usr/local/lib/pythia6
  /usr/lib
  /usr/local/lib
  DOC "Specify the Pythia6 library here."
)

if(PYTHIA6_LIBRARY)
  message(STATUS "Found Pythia8 library at ${PYTHIA6_LIBRARY}")
  set(PYTHIA6_FOUND 1 )
endif()


set(PYTHIA6_LIBRARIES ${PYTHIA6_LIBRARY})

MARK_AS_ADVANCED( PYTHIA6_FOUND PYTHIA6_LIBRARY)
