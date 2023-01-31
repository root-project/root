# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTGeomViewer
  HEADERS
    ROOT/RGeomData.hxx
    ROOT/RGeomViewer.hxx
    ROOT/RGeoPainter.hxx
  SOURCES
    src/RGeomData.cxx
    src/RGeomViewer.cxx
    src/RGeoPainter.cxx
  DEPENDENCIES
    Core
    Geom
    RCsg
    ROOTWebDisplay
    ROOTBrowserv7
)
