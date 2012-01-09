############################################################################
# CMakeLists.txt file for building ROOT geom/geopainter package
# @author Pere Mato, CERN
############################################################################

set(libname GeomPainter)

ROOT_USE_PACKAGE(geom/geom)
ROOT_USE_PACKAGE(graf3d/g3d)
ROOT_USE_PACKAGE(tree/tree)


ROOT_GENERATE_DICTIONARY(G__${libname} T*.h LINKDEF LinkDef.h)

ROOT_GENERATE_ROOTMAP(${libname} LINKDEF LinkDef.h DEPENDENCIES Geom Tree Graf3d Hist Gpad RIO MathCore)
ROOT_LINKER_LIBRARY(${libname} *.cxx G__${libname}.cxx DEPENDENCIES Geom Tree Graf3d Hist Gpad RIO)
ROOT_INSTALL_HEADERS()

