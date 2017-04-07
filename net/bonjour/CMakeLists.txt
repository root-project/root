############################################################################
# CMakeLists.txt file for building ROOT net/bonjour package
# @author Pere Mato, CERN
############################################################################

include_directories(${BONJOUR_INCLUDE_DIR})

ROOT_GENERATE_DICTIONARY(G__BONJ *.h MODULE Bonjour LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(Bonjour *.cxx G__BONJ.cxx LIBRARIES ${BONJOUR_LIBRARIES} DEPENDENCIES Core)

ROOT_INSTALL_HEADERS()


