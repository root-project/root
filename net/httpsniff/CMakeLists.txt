# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT net/httpsniff package
# @author Sergey Linev, GSI
############################################################################

set(RHTTPSNIFF_HEADERS
  TRootSnifferFull.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(RHTTPSniff
  HEADERS
    ${RHTTPSNIFF_HEADERS}
  SOURCES
    src/TRootSnifferFull.cxx
  DEPENDENCIES
    Gpad
    Graf
    Hist
    RHTTP
    RIO
    Thread
    Tree
    XMLIO
)

set(RELATIVE_INC_HEADERS ${RHTTPSNIFF_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(RHTTPSniff
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)


ROOT_ADD_TEST_SUBDIRECTORY(test)
