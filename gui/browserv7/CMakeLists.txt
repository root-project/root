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
    ROOT/RBrowserItem.hxx
    ROOT/RBrowser.hxx
  SOURCES
    src/RBrowser.cxx
  DEPENDENCIES
    ROOTWebDisplay
    RIO
    Gpad
    WebGui6
    
)
