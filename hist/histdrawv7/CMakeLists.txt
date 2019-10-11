# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT hist/histdraw package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTHistDraw
  HEADERS
    ROOT/RHistDrawable.hxx
  SOURCES
    src/RHistDrawable.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Core
    MathCore
    Hist
    ROOTGpadv7
)

ROOT_ADD_TEST_SUBDIRECTORY(test)
