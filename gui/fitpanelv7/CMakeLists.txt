# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/fitpanelv7
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTFitPanelv7
  HEADERS
    ROOT/RFitPanel.hxx
    ROOT/RFitPanelModel.hxx
  SOURCES
    RFitPanel.cxx
    RFitPanelModel.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Hist
    Gpad
    ROOTGpadv7
    ROOTHistDraw
    ROOTWebDisplay
)
