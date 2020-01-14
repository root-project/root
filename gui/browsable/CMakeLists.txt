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
    ROOT/Browsable/RAnyObjectHolder.hxx
    ROOT/Browsable/RElement.hxx
    ROOT/Browsable/RGroup.hxx
    ROOT/Browsable/RHolder.hxx
    ROOT/Browsable/RItem.hxx
    ROOT/Browsable/RLevelIter.hxx
    ROOT/Browsable/RProvider.hxx
    ROOT/Browsable/RShared.hxx
    ROOT/Browsable/RSysFile.hxx
    ROOT/Browsable/RSysFileItem.hxx
    ROOT/Browsable/RUnique.hxx
    ROOT/Browsable/RWrapper.hxx
    ROOT/Browsable/TKeyItem.hxx
    ROOT/Browsable/TObjectHolder.hxx
    ROOT/Browsable/TObjectItem.hxx
    ROOT/RDrawableProvider.hxx
  SOURCES
    src/RDrawableProvider.cxx
    src/RElement.cxx
    src/RGroup.cxx
    src/RLevelIter.cxx
    src/RProvider.cxx
    src/RSysFile.cxx
    src/TDirectoryElement.cxx
    src/TObjectElement.cxx
    src/TObjectHolder.cxx
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
