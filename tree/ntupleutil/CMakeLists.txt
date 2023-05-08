# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT ntuple extra utils package
# @author Jakob Blomer CERN
############################################################################

if(NOT root7)
  return()
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTNTupleUtil
HEADERS
  ROOT/RNTupleImporter.hxx
SOURCES
  v7/src/RNTupleImporter.cxx
LINKDEF
  LinkDef.h
DEPENDENCIES
  ROOTNTuple
  Tree
)

if(MSVC)
  target_compile_definitions(ROOTNTupleUtil PRIVATE _USE_MATH_DEFINES)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(v7/test)
