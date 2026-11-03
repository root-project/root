# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT io/mpi package
############################################################################

set(RMPI_HEADERS
  TMPIClientInfo.h
  TMPIFile.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(RMPI
  HEADERS
     ${RMPI_HEADERS}
  SOURCES
     src/TMPIClientInfo.cxx
     src/TMPIFile.cxx
  DEPENDENCIES
     RIO
     MathCore
)

target_include_directories(RMPI PUBLIC ${MPI_CXX_HEADER_DIR})
target_link_libraries(RMPI
  PRIVATE
    MPI::MPI_CXX
    $<$<PLATFORM_ID:Windows>:ws2_32.lib>
)

set(RELATIVE_INC_HEADERS ${RMPI_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)

target_sources(RMPI
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

