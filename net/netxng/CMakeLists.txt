# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT net/netxng package
# @author Lukasz Janyst <ljanyst@cern.ch>
############################################################################

set(NETXNG_HEADERS
  TNetXNGFile.h
  TNetXNGFileStager.h
  TNetXNGSystem.h
  ROOT/RRawFileNetXNG.hxx
)

ROOT_STANDARD_LIBRARY_PACKAGE(NetxNG
  HEADERS
    ${NETXNG_HEADERS}
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

set(RELATIVE_INC_HEADERS ${NETXNG_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(NetxNG
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

target_link_libraries(NetxNG PRIVATE XRootD::XrdCl XRootD::XrdUtils)
target_compile_options(NetxNG PRIVATE -Wno-shadow)

# When linking against the XRootD target, XRootD includes become "-isystem".
# By linking explicitly here, we suppress a warning during dictionary compilation.
target_link_libraries(G__NetxNG PRIVATE XRootD::XrdCl)

ROOT_ADD_TEST_SUBDIRECTORY(test)
