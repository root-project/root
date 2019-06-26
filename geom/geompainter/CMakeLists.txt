# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT geom/geopainter package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(GeomPainter
  HEADERS
    TGeoChecker.h
    TGeoOverlap.h
    TGeoPainter.h
    TGeoTrack.h
  SOURCES
    src/TGeoChecker.cxx
    src/TGeoOverlap.cxx
    src/TGeoPainter.cxx
    src/TGeoTrack.cxx
  DEPENDENCIES
    Geom
    Gpad
    Graf3d
    Hist
    RIO
    Tree
)
