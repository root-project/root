############################################################################
# CMakeLists.txt file for building ROOT net/alien package
# @author Pere Mato, CERN
############################################################################

include_directories(${ALIEN_INCLUDE_DIR})


ROOT_GENERATE_DICTIONARY(G__Alien *.h MODULE RAliEn LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(RAliEn *.cxx G__Alien.cxx LIBRARIES ${ALIEN_LIBRARIES}
                                               DEPENDENCIES XMLIO Netx Tree Proof)

ROOT_INSTALL_HEADERS()


