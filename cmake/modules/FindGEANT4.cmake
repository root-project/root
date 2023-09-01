# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# - Try to find GEANT4
# Once done this will define
#
#  GEANT4_FOUND - system has GEANT4
#  GEANT4_INCLUDE_DIR - the GEANT4 include directory
#  GEANT4_LIBRARIES - The libraries needed to use GEANT4
#  GEANT4_DEFINITIONS - Compiler switches required for using GEANT4
#

if (GEANT4_INCLUDE_DIR AND GEANT4_LIBRARY_DIR)
  SET (GEANT4_INCLUDE_DIR GEANT4_INCLUDE_DIR-NOTFOUND)
  SET (GEANT4_LIB_DIR GEANT4_LIB_DIR-NOTFOUND)
  SET (GEANT4_PLISTS_LIB_DIR GEANT4_PLISTS_LIB_DIR-NOTFOUND)
  SET (GEANT4_DIR GEANT4_DIR-NOTFOUND)
endif (GEANT4_INCLUDE_DIR AND GEANT4_LIBRARY_DIR)

MESSAGE(STATUS "Looking for GEANT4...")

FIND_PATH(GEANT4_DIR NAMES env.sh PATHS
  ${SIMPATH}/transport/geant4
  ${SIMPATH}/transport/geant4/source
  NO_DEFAULT_PATH
)

FIND_PATH(GEANT4_INCLUDE_DIR NAMES G4Event.hh PATHS
  ${SIMPATH}/transport/geant4/include
  NO_DEFAULT_PATH
)

SET(GEANT4_INCLUDE_DIR
${SIMPATH}/transport/geant4/include
${SIMPATH}/transport/geant4/source/interfaces/common/include
${SIMPATH}/transport/geant4/physics_lists/hadronic/Packaging/include
${SIMPATH}/transport/geant4/physics_lists/hadronic/QGSP/include
)

FIND_PATH(GEANT4_LIB_DIR NAMES libG4baryons.so libG4baryons.dylib PATHS
  ${SIMPATH}/transport/geant4/lib/Linux-g++
  ${SIMPATH}/transport/geant4/lib/Linux-icc
  ${SIMPATH}/transport/geant4/lib
  NO_DEFAULT_PATH
)

IF (GEANT4_LIB_DIR)
  SET(GEANT4_LIBRARY_DIR ${GEANT4_LIB_DIR})
ENDIF (GEANT4_LIB_DIR)

if (GEANT4_INCLUDE_DIR AND GEANT4_LIBRARY_DIR)
   set(GEANT4_FOUND TRUE)
endif (GEANT4_INCLUDE_DIR AND GEANT4_LIBRARY_DIR)

if (GEANT4_FOUND)
  if (NOT GEANT4_FIND_QUIETLY)
    MESSAGE(STATUS "Looking for GEANT4... - found ${GEANT4_LIBRARY_DIR}")
#    message(STATUS "Found ${GEANT4_LIBRARY_DIR}")
  endif (NOT GEANT4_FIND_QUIETLY)
  SET(LD_LIBRARY_PATH ${LD_LIBRARY_PATH} ${GEANT4_LIBRARY_DIR})
else (GEANT4_FOUND)
  if (GEANT4_FIND_REQUIRED)
    message(FATAL_ERROR "Looking for GEANT4... - Not found")
  endif (GEANT4_FIND_REQUIRED)
endif (GEANT4_FOUND)
