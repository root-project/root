# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf2d/primitives package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTGraphicsPrimitives
  HEADERS
    ROOT/RBox.hxx
    ROOT/RLegend.hxx
    ROOT/RLine.hxx
    ROOT/RMarker.hxx
    ROOT/RText.hxx
  SOURCES
    v7/src/RBox.cxx
    v7/src/RLegend.cxx
    v7/src/RLine.cxx
    v7/src/RMarker.cxx
    v7/src/RText.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Core
    RIO
    ROOTGpadv7
)

ROOT_ADD_TEST_SUBDIRECTORY(v7/test)
