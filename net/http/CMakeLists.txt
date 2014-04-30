############################################################################
# CMakeLists.txt file for building ROOT net/http package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(core)
ROOT_USE_PACKAGE(tree/tree)

add_definitions(-DHTTP_WITHOUT_FASTCGI)

ROOT_GENERATE_DICTIONARY(G__HTTP *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(RHTTP LINKDEF LinkDef.h DEPENDENCIES Thread Tree Hist Gpad Graf RIO XMLIO)
ROOT_LINKER_LIBRARY(RHTTP *.cxx civetweb/civetweb.c G__HTTP.cxx LIBRARIES DEPENDENCIES Thread Tree Hist Gpad Graf RIO XMLIO)

ROOT_INSTALL_HEADERS()

