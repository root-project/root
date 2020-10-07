# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT io/iolite package
############################################################################

if (WIN32)
  set(rawfile_local_sources src/RRawFileWin.cxx)
else ()
  set(rawfile_local_sources src/RRawFileUnix.cxx)
endif ()

ROOT_LINKER_LIBRARY(ROOTIOLite
  ${rawfile_local_sources}
  src/RRawFile.cxx
  DEPENDENCIES
    Foundation
)

ROOT_INSTALL_HEADERS()
