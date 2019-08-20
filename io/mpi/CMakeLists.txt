# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT io/mpi package
############################################################################

ROOT_LINKER_LIBRARY(RMPI
  src/TClientInfo.cxx
  src/TMPIFile.cxx
  $<TARGET_OBJECTS:RootPcmObjs>
  LIBRARIES
    ${CMAKE_DL_LIBS}
    MPI::MPI_CXX
  DEPENDENCIES
    RIO
)

ROOT_GENERATE_DICTIONARY(G__RMPI
  TClientInfo.h
  TMPIFile.h
  STAGE1
  NOTARGET
  MODULE
    RMPI
  LINKDEF
    LinkDef.h
  OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    RIO
)

ROOT_INSTALL_HEADERS()
