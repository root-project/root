# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT geom/geopainter package
# @author Pere Mato, CERN
############################################################################

set (GEOMCHECKER_HEADERS
  TGeoOverlap.h
  TGeoChecker.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(GeomChecker
  HEADERS
    ${GEOMCHECKER_HEADERS}
  SOURCES
    src/TGeoOverlap.cxx
    src/TGeoChecker.cxx
  DEPENDENCIES
    Geom
    Gpad
    Graf3d
    Hist
    RIO
    Tree
)


set(RELATIVE_INC_HEADERS ${GEOMCHECKER_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)

target_sources(GeomChecker
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)
