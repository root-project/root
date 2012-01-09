############################################################################
# CMakeLists.txt file for building ROOT geom/gdml package
############################################################################

ROOT_USE_PACKAGE(geom/geom)
ROOT_USE_PACKAGE(io/xml)

ROOT_GENERATE_DICTIONARY(G__Gdml *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(Gdml LINKDEF LinkDef.h DEPENDENCIES Geom XMLIO Hist RIO)
ROOT_LINKER_LIBRARY(Gdml *.cxx G__Gdml.cxx DEPENDENCIES Geom XMLIO Hist RIO)
ROOT_INSTALL_HEADERS()

