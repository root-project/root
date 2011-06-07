# Find the FFTW includes and library.
# 
# This module defines
# FFTW_INCLUDE_DIR, where to locate Pythia.h file
# FFTW_LIBRARIES, the libraries to link against to use Pythia6
# FFTW_FOUND.  If false, you cannot build anything that requires Pythia6.
# FFTW_LIBRARY, where to find the libpythia8 library.

set(PYTHIA8_FOUND 0)

find_path(FFTW_INCLUDE_DIR fftw3.h
  $ENV{FFTW_DIR}/include
  /usr/include
  /usr/local/include
  /opt/include
  /usr/apps/include
  DOC "Specify the directory containing fftw3.h"
)

find_library(FFTW_LIBRARY NAMES libfftw3 libfftw3-3 PATHS
  $ENV{FFTW_DIR}/lib
  /usr/lib 
  /usr/local/lib
  /opt/lib
  /sw/lib
  DOC "Specify the fttw3 library here."
)

if(FFTW_INCLUDE_DIR AND FFTW_LIBRARY)
  set(FFTW_FOUND 1 )
  message(STATUS "Found fftw3 library at ${FFTW_LIBRARY}")
endif()


set(FFTW_LIBRARIES ${FFTW_LIBRARY})

MARK_AS_ADVANCED(FFTW_FOUND FFTW_LIBRARY FFTW_INCLUDE_DIR)
