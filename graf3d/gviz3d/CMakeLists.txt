############################################################################
# CMakeLists.txt file for building ROOT graf3d/gviz3d package
############################################################################

ROOT_USE_PACKAGE(gui/gui)
ROOT_USE_PACKAGE(gui/ged)
ROOT_USE_PACKAGE(graf3d/gl)
ROOT_USE_PACKAGE(geom/geom)

ROOT_GENERATE_DICTIONARY(G__Gviz3d *.h MODULE Gviz3d LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(Gviz3d *.cxx G__Gviz3d.cxx LIBRARIES GLEW ${FTGL_LIBRARIES} DEPENDENCIES Gui Ged Geom RGL)
ROOT_INSTALL_HEADERS()
