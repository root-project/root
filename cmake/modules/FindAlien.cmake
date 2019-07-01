# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# Find the Alien includes and library.
#
# This module defines
# ALIEN_INCLUDE_DIR, where to locate gapiUI.h file
# ALIEN_LIBRARY, the libraries to link against to use Alien
# ALIEN_LIBRARIES, the libraries to link against to use Alien
# ALIEN_FOUND.  If false, you cannot build anything that requires Alien.

set(ALIEN_FOUND 0)
if(ALIEN_LIBRARY AND ALIEN_INCLUDE_DIR)
  set(ALIEN_FIND_QUIETLY TRUE)
endif()

find_path(ALIEN_INCLUDE_DIR gapiUI.h PATHS
  ${ALIEN_DIR}/include $ENV{ALIEN_DIR}/include
  /usr/local/include
  /opt/alien/api/include
  /opt/monalisa/include
  /usr/include
  DOC "Specify the directory containing gapiUI.h"
)

find_library(ALIEN_LIBRARY NAMES gapiUI PATHS
  ${ALIEN_DIR}/lib $ENV{ALIEN_DIR}/lib
  /usr/local/lib
  /opt/alien/api/lib
  /opt/monalisa/lib
  /usr/lib
  DOC "Specify the libgapiUI library here."
)

if(ALIEN_INCLUDE_DIR AND ALIEN_LIBRARY)
  set(ALIEN_FOUND 1 )
  if(NOT ALIEN_FIND_QUIETLY)
     message(STATUS "Found Alien includes at ${ALIEN_INCLUDE_DIR}")
     message(STATUS "Found Alien library at ${ALIEN_LIBRARY}")
  endif()
endif()

set(ALIEN_LIBRARIES ${ALIEN_LIBRARY})

mark_as_advanced(ALIEN_LIBRARY ALIEN_INCLUDE_DIR)
