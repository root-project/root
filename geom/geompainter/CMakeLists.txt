############################################################################
# CMakeLists.txt file for building ROOT geom/geopainter package
# @author Pere Mato, CERN
############################################################################

set(libname GeomPainter)

ROOT_GENERATE_DICTIONARY(G__${libname} T*.h MODULE ${libname} LINKDEF LinkDef.h DEPENDENCIES Geom Tree Graf3d Hist Gpad RIO)


ROOT_LINKER_LIBRARY(${libname} *.cxx G__${libname}.cxx DEPENDENCIES Geom Tree Graf3d Hist Gpad RIO)
ROOT_INSTALL_HEADERS()

