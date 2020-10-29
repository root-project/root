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

if (uring)
  set(rawfile_local_headers ROOT/RIoUring.hxx)
endif ()


ROOT_STANDARD_LIBRARY_PACKAGE(ROOTIOLite
  STAGE1
  HEADERS
    ${rawfile_local_headers}
    ROOT/RRawFile.hxx
    ROOT/RRawFileUnix.hxx
    ROOT/RRawFileWin.hxx
  SOURCES
    ${rawfile_local_sources}
    src/RRawFile.cxx
  LIBRARIES
    ROOTFoundation
  DEPENDENCIES
    Core
  )
