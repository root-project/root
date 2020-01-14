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
    ROOT/Browsable/RHolder.hxx
    ROOT/Browsable/RElement.hxx
    ROOT/Browsable/RLevelIter.hxx
    ROOT/Browsable/RWrapper.hxx
    ROOT/Browsable/RGroup.hxx
    ROOT/Browsable/RProvider.hxx
    ROOT/RBrowsable.hxx
    ROOT/RBrowsableSysFile.hxx
    ROOT/RBrowserItem.hxx
    ROOT/RDrawableProvider.hxx
    ROOT/RBrowsableTObject.hxx
    ROOT/RBrowsableTDirectory.hxx
  SOURCES
    src/RElement.cxx
    src/RLevelIter.cxx
    src/RProvider.cxx
    src/RBrowsable.cxx
    src/RGroup.cxx
    src/RBrowsableSysFile.cxx
    src/RDrawableProvider.cxx
    src/RBrowsableTObject.cxx
    src/RBrowsableTDirectory.cxx
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

# provider for generic drawing of TObject on RCanvas/TCanvas

ROOT_LINKER_LIBRARY(ROOTObjectDrawProvider
     src/RDrawableTObject.cxx
  DEPENDENCIES
     ROOTBrowsable
     RIO
     Gpad
     ROOTGpadv7
)

# provider for drawing of TLeaf on RCanvas/TCanvas

ROOT_LINKER_LIBRARY(ROOTTreeDrawProvider
     src/RDrawableTTree.cxx
  DEPENDENCIES
     ROOTBrowsable
     RIO
     Tree
     Gpad
     Hist
     ROOTGpadv7
)
