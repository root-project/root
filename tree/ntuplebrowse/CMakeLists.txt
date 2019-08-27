# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT ntuplebrowse package
# @author Simon Leisibach CERN
############################################################################

if(NOT root7)
  return()
endif()


ROOT_STANDARD_LIBRARY_PACKAGE(ROOTNTupleBrowse
HEADERS
  ROOT/RBrowseVisitor.hxx
  ROOT/RNTupleBrowser.hxx
SOURCES
  v7/src/RBrowseVisitor.cxx
  v7/src/RNTupleBrowser.cxx
LINKDEF
  LinkDef.h
DEPENDENCIES
  ROOTNTuple
  Hist
)

ROOT_ADD_TEST_SUBDIRECTORY(v7/test)


