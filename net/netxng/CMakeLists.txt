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
  DEPENDENCIES
    Net
    RIO
    Thread
)

target_link_libraries(NetxNG PRIVATE XRootD::XrdCl XRootD::XrdUtils)
target_compile_options(NetxNG PRIVATE -Wno-shadow)

# When linking against the XRootD target, XRootD includes become "-isystem".
# By linking explicitly here, we suppress a warning during dictionary compilation.
target_link_libraries(G__NetxNG PRIVATE XRootD::XrdCl)

ROOT_ADD_TEST_SUBDIRECTORY(test)
