# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/browsable
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTBrowsable
  HEADERS
    ROOT/RBrowsable.hxx
    ROOT/RBrowsableSysFile.hxx
    ROOT/RBrowserItem.hxx
    ROOT/RDrawableProvider.hxx
  SOURCES
    src/RBrowsable.cxx
    src/RBrowsableSysFile.cxx
    src/RDrawableProvider.cxx
  DEPENDENCIES
    RIO
)

# provider for drawing of v7 histograms on RCanvas

ROOT_LINKER_LIBRARY(ROOTHistDrawProvider
     src/RDrawableRHist.cxx
  DEPENDENCIES
     ROOTBrowsable
     ROOTGpadv7
     ROOTHistDraw
)
