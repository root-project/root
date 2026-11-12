# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

set (EXTRA_DICT_OPTS)
if (runtime_cxxmodules AND WIN32)
  set (EXTRA_DICT_OPTS NO_CXXMODULE)
endif()

set(GEOMVIEWER_HEADERS
  ROOT/RGeomData.hxx
  ROOT/RGeomHierarchy.hxx
  ROOT/RGeomViewer.hxx
  ROOT/RGeoPainter.hxx
)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTGeomViewer
  HEADERS
    ${GEOMVIEWER_HEADERS}
  SOURCES
    src/RGeomData.cxx
    src/RGeomHierarchy.cxx
    src/RGeomViewer.cxx
    src/RGeoPainter.cxx
  DEPENDENCIES
    Core
    Geom
    GeomPainter
    ROOTWebDisplay
    ROOTBrowserv7
  LIBRARIES
    ROOTCsg
  ${EXTRA_DICT_OPTS}
)

set(RELATIVE_INC_HEADERS ${GEOMVIEWER_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)

target_sources(ROOTGeomViewer
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)
