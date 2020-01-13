# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/browserv7
############################################################################


ROOT_STANDARD_LIBRARY_PACKAGE(ROOTBrowserv7
  HEADERS
    ROOT/RBrowsableTObject.hxx
    ROOT/RBrowsableTDirectory.hxx
    ROOT/RBrowser.hxx
    ROOT/RFileDialog.hxx
  SOURCES
    src/RBrowsableTObject.cxx
    src/RBrowsableTDirectory.cxx
    src/RDrawableRHist.cxx
    src/RBrowser.cxx
    src/RFileDialog.cxx
  DEPENDENCIES
    ROOTBrowsable
    ROOTWebDisplay
    RIO
    Gpad
    WebGui6
    Tree
    ROOTGpadv7
    ROOTHistDraw
)
