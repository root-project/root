# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT net/davix package
# @author Pere Mato, CERN
############################################################################

set(RDAVIX_HEADERS
  ROOT/RRawFileDavix.hxx
  TDavixFile.h
  TDavixSystem.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(RDAVIX
  HEADERS
    ${RDAVIX_HEADERS}
  SOURCES
    src/RRawFileDavix.cxx
    src/TDavixFile.cxx
    src/TDavixSystem.cxx
  LIBRARIES
    Davix::Davix
    Thread
  DEPENDENCIES
    Net
    RIO
)

set(RELATIVE_INC_HEADERS ${RDAVIX_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(RDAVIX
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/ src/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
    src/TDavixFileInternal.h
    src/utils.h
)

target_compile_options(RDAVIX PRIVATE -Wno-deprecated-declarations)

ROOT_ADD_TEST_SUBDIRECTORY(test)
