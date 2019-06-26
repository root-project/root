# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT hist/histdraw package
############################################################################

if(NOT root7)
  return()
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTHistDraw
  HEADERS
    ROOT/RHistDrawable.hxx
    ROOT/RHistDrawingOpts.hxx
  SOURCES
    v7/src/RHistDrawable.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Hist
    ROOTGpadv7
    ROOTGraphicsPrimitives
)

ROOT_ADD_TEST_SUBDIRECTORY(v7/test)
