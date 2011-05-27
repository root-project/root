# - Try to find PLUTO instalation
# Once done this will define
#
#  PLUTO_FOUND - system has GEANT3
#  PLUTO_INCLUDE_DIR - the GEANT3 include directory
#  PLUTO_LIBRARIES - The libraries needed to use GEANT3
#  PLUTO_DEFINITIONS - Compiler switches required for using GEANT3
#

if (PLUTO_INCLUDE_DIR AND PLUTO_LIBRARIES)
  SET (PLUTO_INCLUDE_DIR PLUTO_INCLUDE_DIR-NOTFOUND)
  SET (PLUTO_LIB PLUTO_LIB-NOTFOUND)
  SET (PLUTO_DUMMY_LIB PLUTO_DUMMY_LIB-NOTFOUND)
endif (PLUTO_INCLUDE_DIR AND PLUTO_LIBRARIES)

MESSAGE(STATUS "Looking for Pluto...")

FIND_PATH(PLUTO_INCLUDE_DIR NAMES PChannel.h PATHS
  ${SIMPATH}/generators/pluto/src
  ${SIMPATH}/generators/pluto
  ${SIMPATH}/generators/pluto/include
  NO_DEFAULT_PATH
)

FIND_PATH(PLUTO_LIBRARY_DIR NAMES libPluto.so PATHS
  ${SIMPATH}/generators/lib
  ${SIMPATH}/generators/pluto
  NO_DEFAULT_PATH
)

if (PLUTO_INCLUDE_DIR AND PLUTO_LIBRARY_DIR)
   set(PLUTO_FOUND TRUE)
endif (PLUTO_INCLUDE_DIR AND PLUTO_LIBRARY_DIR)

if (PLUTO_FOUND)
  if (NOT PLUTO_FIND_QUIETLY)
    MESSAGE(STATUS "Looking for Pluto... - found ${PLUTO_LIBRARY_DIR}")
#    message(STATUS "Found PLUTO: ${PLUTO_LIBRARY_DIR}")
    SET(LD_LIBRARY_PATH ${LD_LIBRARY_PATH} ${PLUTO_LIBRARY_DIR})
  endif (NOT PLUTO_FIND_QUIETLY)
else (PLUTO_FOUND)
  if (PLUTO_FIND_REQUIRED)
    message(FATAL_ERROR "Looking for Pluto... - Not found")
  endif (PLUTO_FIND_REQUIRED)
endif (PLUTO_FOUND)

