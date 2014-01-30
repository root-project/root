############################################################################
# CMakeLists.txt file for building ROOT io/xml package
############################################################################

ROOT_USE_PACKAGE(io/io)

ROOT_GENERATE_DICTIONARY(G__XML *.h MODULE XMLIO LINKDEF LinkDef.h)


ROOT_LINKER_LIBRARY(XMLIO *.cxx G__XML.cxx DEPENDENCIES RIO )
ROOT_INSTALL_HEADERS()

