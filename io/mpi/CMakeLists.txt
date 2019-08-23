# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT io/mpi package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(RMPI
  HEADERS
     TClientInfo.h
     TMPIFile.h
  SOURCES
     src/TClientInfo.cxx
     src/TMPIFile.cxx
  DEPENDENCIES
     RIO
)

target_link_libraries(RMPI
  PRIVATE
    MPI::MPI_CXX
    $<$<PLATFORM_ID:Windows>:ws2_32.lib>
)
