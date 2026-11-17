# Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT ntuplebrowse package
# @author Jakob Blomer CERN
############################################################################

set(NTUPLEBROWSE_HEADERS
  ROOT/RNTupleBrowseUtils.hxx
  ROOT/RNTupleClassicBrowse.hxx
  ROOT/RNTupleDrawVisitor.hxx
  ROOT/RNTupleTreeMap.hxx
)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTNTupleBrowse
HEADERS
  ${NTUPLEBROWSE_HEADERS}
SOURCES
  src/RNTupleBrowseUtils.cxx
  src/RNTupleClassicBrowse.cxx
  src/RNTupleDrawVisitor.cxx
  src/RNTupleTreeMap.cxx
LINKDEF
  LinkDef.h
DEPENDENCIES
  Core
  Gpad
  Graf
  Hist
  ROOTNTuple
  ROOTNTupleUtil
  ROOTTreeMap
)

set(RELATIVE_INC_HEADERS ${NTUPLEBROWSE_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(ROOTNTupleBrowse
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

ROOT_ADD_TEST_SUBDIRECTORY(test)
