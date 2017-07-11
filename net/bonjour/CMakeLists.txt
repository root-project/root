############################################################################
# CMakeLists.txt file for building ROOT net/bonjour package
# @author Pere Mato, CERN
############################################################################

include_directories(${BONJOUR_INCLUDE_DIR})

ROOT_GENERATE_DICTIONARY(G__Bonjour *.h MODULE Bonjour LINKDEF LinkDef.h DEPENDENCIES Core)

ROOT_LINKER_LIBRARY(Bonjour *.cxx G__Bonjour.cxx LIBRARIES ${BONJOUR_LIBRARIES} DEPENDENCIES Core)

ROOT_INSTALL_HEADERS()


