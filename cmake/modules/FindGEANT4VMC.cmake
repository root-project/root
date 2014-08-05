# - Try to find GEANT4VMC
# Once done this will define
#
#  GEANT4VMC_FOUND - system has GEANT3
#  GEANT4VMC_INCLUDE_DIR - the GEANT3 include directory
#  GEANT4VMC_LIBRARIES - The libraries needed to use GEANT3
#  GEANT4VMC_DEFINITIONS - Compiler switches required for using GEANT3
#

if (GEANT4VMC_INCLUDE_DIR AND GEANT4VMC_LIBRARY_DIR)
  SET (GEANT4VMC_INCLUDE_DIR GEANT4VMC_INCLUDE_DIR-NOTFOUND)
  SET (GEANT4VMC_LIB_DIR GEANT4VMC_LIB_DIR-NOTFOUND)
  SET (GEANT4VMC_PLISTS_LIB_DIR GEANT4VMC_PLISTS_LIB_DIR-NOTFOUND)
endif (GEANT4VMC_INCLUDE_DIR AND GEANT4VMC_LIBRARY_DIR)

MESSAGE(STATUS "Looking for GEANT4VMC...")

FIND_PATH(GEANT4VMC_INCLUDE_DIR NAMES TG4G3Units.h PATHS
  ${SIMPATH}/transport/geant4_vmc/include
   NO_DEFAULT_PATH
)

set(GEANT4VMC_INCLUDE_DIR
${SIMPATH}/transport/geant4_vmc/source/global/include
${SIMPATH}/transport/geant4_vmc/source/geometry/include
${SIMPATH}/transport/geant4_vmc/source/digits+hits/include
${SIMPATH}/transport/geant4_vmc/source/physics/include
${SIMPATH}/transport/geant4_vmc/source/event/include
${SIMPATH}/transport/geant4_vmc/source/run/include
${SIMPATH}/transport/geant4_vmc/source/interfaces/include
${SIMPATH}/transport/geant4_vmc/source/visualization/include
${SIMPATH}/transport/geant4_vmc/include
${SIMPATH}/transport/vgm/packages/BaseVGM/include
${SIMPATH}/transport/vgm/packages/ClhepVGM/include
${SIMPATH}/transport/vgm/packages/Geant4GM/include
${SIMPATH}/transport/vgm/packages/RootGM/include
${SIMPATH}/transport/vgm/packages/VGM/include
${SIMPATH}/transport/vgm/packages/XmlVGM/include
)


FIND_PATH(GEANT4VMC_LIBRARY_DIR NAMES libgeant4vmc.so libgeant4vmc.dylib PATHS
  ${SIMPATH}/transport/geant4_vmc/lib/tgt_linux
  ${SIMPATH}/transport/geant4_vmc/lib/tgt_linuxicc
  ${SIMPATH}/transport/geant4_vmc/lib/tgt_linuxx8664gcc
  ${SIMPATH}/transport/geant4_vmc/lib
  NO_DEFAULT_PATH
)

# check for existence of header file, which is needed in CbmRunConfiguration
# The file is only present in old versions of VMC
FIND_FILE(GEANT4_MODULAR_PHYSICS_LIST TG4ModularPhysicsList.h PATHS
  ${GEANT4VMC_INCLUDE_DIR}
  NO_DEFAULT_PATH
)

if (GEANT4VMC_INCLUDE_DIR AND GEANT4VMC_LIBRARY_DIR)
  if (NOT GEANT4VMC_FIND_QUIETLY)
    MESSAGE(STATUS "Looking for GEANT4VMC... - found  ${GEANT4VMC_LIBRARY_DIR}")
  endif (NOT GEANT4VMC_FIND_QUIETLY)
else (GEANT4VMC_INCLUDE_DIR AND GEANT4VMC_LIBRARY_DIR)
  if (GEANT4VMC_FIND_REQUIRED)
    message(FATAL_ERROR "Looking for GEANT4VMC... - Not found ")
  endif (GEANT4VMC_FIND_REQUIRED)
endif (GEANT4VMC_INCLUDE_DIR AND GEANT4VMC_LIBRARY_DIR)


MESSAGE(STATUS "Looking for VGM...")

FIND_PATH(VGM_LIBRARY_DIR NAMES libBaseVGM.so libBaseVGM.dylib PATHS
  ${SIMPATH}/transport/vgm/lib/Linux-g++
  ${SIMPATH}/transport/vgm.2.08.04/lib/Linux-g++
  ${SIMPATH}/transport/vgm/lib/Linux-icc
  ${SIMPATH}/transport/vgm/lib
  NO_DEFAULT_PATH
)

if (VGM_LIBRARY_DIR)
  if (NOT GEANT4VMC_FIND_QUIETLY)
    MESSAGE(STATUS "Looking for VGM... - found  ${VGM_LIBRARY_DIR}")
  endif (NOT GEANT4VMC_FIND_QUIETLY)
else (VGM_LIBRARY_DIR)
  if (GEANT4VMC_FIND_REQUIRED)
    message(FATAL_ERROR "Looking for VGM... - Not found ")
  endif (GEANT4VMC_FIND_REQUIRED)
endif (VGM_LIBRARY_DIR)


if (GEANT4VMC_INCLUDE_DIR AND GEANT4VMC_LIBRARY_DIR AND VGM_LIBRARY_DIR)
   set(GEANT4VMC_FOUND TRUE)
endif (GEANT4VMC_INCLUDE_DIR AND GEANT4VMC_LIBRARY_DIR AND VGM_LIBRARY_DIR)

if (GEANT4VMC_FOUND)
  SET(LD_LIBRARY_PATH ${LD_LIBRARY_PATH} ${GEANT4VMC_LIBRARY_DIR}
      ${VGM_LIBRARY_DIR})
endif (GEANT4VMC_FOUND)

