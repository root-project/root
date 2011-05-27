############################################################################
# CMakeLists.txt file for building ROOT io/xml package
############################################################################

ROOT_USE_PACKAGE(io/io)

ROOT_GENERATE_DICTIONARY(G__XML *.h  LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(XMLIO LINKDEF LinkDef.h DEPENDENCIES RIO)

ROOT_LINKER_LIBRARY(XMLIO *.cxx G__XML.cxx DEPENDENCIES RIO )
ROOT_INSTALL_HEADERS()

