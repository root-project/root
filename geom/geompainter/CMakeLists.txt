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
