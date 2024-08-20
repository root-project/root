# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

set (EXTRA_DICT_OPTS)
if (runtime_cxxmodules AND WIN32)
  set (EXTRA_DICT_OPTS NO_CXXMODULE)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTGeomViewer
  HEADERS
    ROOT/RGeomData.hxx
    ROOT/RGeomHierarchy.hxx
    ROOT/RGeomViewer.hxx
    ROOT/RGeoPainter.hxx
  SOURCES
    src/RGeomData.cxx
    src/RGeomHierarchy.cxx
    src/RGeomViewer.cxx
    src/RGeoPainter.cxx
  DEPENDENCIES
    Core
    Geom
    RCsg
    GeomPainter
    ROOTWebDisplay
    ROOTBrowserv7
  ${EXTRA_DICT_OPTS}
)
