# Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT ntuplebrowse package
# @author Jakob Blomer CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTNTupleBrowse
HEADERS
  ROOT/RNTupleBrowseUtils.hxx
  ROOT/RNTupleClassicBrowse.hxx
  ROOT/RNTupleDrawVisitor.hxx
  ROOT/RNTupleTreeMap.hxx
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

ROOT_ADD_TEST_SUBDIRECTORY(test)
