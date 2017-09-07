############################################################################
# CMakeLists.txt file for building ROOT geom/geopainter package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(GeomPainter
                              HEADERS T*.h
                              DEPENDENCIES Geom Tree Graf3d Hist Gpad RIO)
