############################################################################
# CMakeLists.txt file for building ROOT io/xml package
############################################################################

ROOT_GENERATE_DICTIONARY(G__XMLIO *.h MODULE XMLIO LINKDEF LinkDef.h DEPENDENCIES RIO )


ROOT_LINKER_LIBRARY(XMLIO *.cxx G__XMLIO.cxx DEPENDENCIES RIO )
ROOT_INSTALL_HEADERS()

