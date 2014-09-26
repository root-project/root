# Find the FFTW includes and library.
# 
# This module defines
# FFTW_INCLUDE_DIR, where to locate fftw3.h file
# FFTW_LIBRARIES, the libraries to link against to use fftw3
# FFTW_FOUND.  If false, you cannot build anything that requires fftw3.
# FFTW_LIBRARY, where to find the libfftw3 library.

set(FFTW_FOUND 0)
if(FFTW_LIBRARY AND FFTW_INCLUDE_DIR)
  set(FFTW_FIND_QUIETLY TRUE)
endif()

find_path(FFTW_INCLUDE_DIR NAMES fftw3.h
  HINTS ${FFTW_DIR}/include $ENV{FFTW_DIR}/include
  DOC "Specify the directory containing fftw3.h"
)

find_library(FFTW_LIBRARY NAMES fftw3
  HINTS ${FFTW_DIR}/lib $ENV{FFTW_DIR}/lib
  DOC "Specify the fttw3 library here."
)

if(FFTW_INCLUDE_DIR AND FFTW_LIBRARY)
  set(FFTW_FOUND 1 )
  if(NOT FFTW_FIND_QUIETLY)
     message(STATUS "Found fftw3 includes at ${FFTW_INCLUDE_DIR}")
     message(STATUS "Found fftw3 library at ${FFTW_LIBRARY}")
  endif()
endif()

set(FFTW_LIBRARIES ${FFTW_LIBRARY})

mark_as_advanced(FFTW_FOUND FFTW_LIBRARY FFTW_INCLUDE_DIR)
