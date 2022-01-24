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
# The generated dictionary source file G__NetxNG.cxx, compiled into the OBJECT
# library G__NetxNG, depends on XRootD headers via our TNetXNGFile.h. To avoid
# warnings, the XRootD include directories must be added as SYSTEM. We cannot do
# this automatically in the macros as CMake has no target property to find out
# SYSTEM include directories, so we must do it manually here.
target_include_directories(G__NetxNG SYSTEM PRIVATE ${XROOTD_INCLUDE_DIRS})

target_compile_options(NetxNG PRIVATE -Wno-shadow)

ROOT_ADD_TEST_SUBDIRECTORY(test)
