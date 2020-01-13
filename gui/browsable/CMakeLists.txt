# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/browserv7
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTBrowsable
  HEADERS
    ROOT/RBrowsable.hxx
    ROOT/RBrowsableSysFile.hxx
    ROOT/RBrowserItem.hxx
  SOURCES
    src/RBrowsable.cxx
    src/RBrowsableSysFile.cxx
  DEPENDENCIES
    RIO
)
