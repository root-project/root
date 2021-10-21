# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT net/netxng package
# @author Lukasz Janyst <ljanyst@cern.ch>
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(NetxNG
  HEADERS
    TNetXNGFile.h
    TNetXNGFileStager.h
    TNetXNGSystem.h
    ROOT/RRawFileNetXNG.hxx
  SOURCES
    src/TNetXNGFile.cxx
    src/TNetXNGFileStager.cxx
    src/TNetXNGSystem.cxx
    src/RRawFileNetXNG.cxx
  LIBRARIES
    Xrootd::Xrootd
  DEPENDENCIES
    Net
    RIO
    Thread
  BUILTINS
    XROOTD
)

target_include_directories(NetxNG SYSTEM PRIVATE ${XROOTD_INCLUDE_DIRS})

target_compile_options(NetxNG PRIVATE -Wno-shadow)

ROOT_ADD_TEST_SUBDIRECTORY(test)
