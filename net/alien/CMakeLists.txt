############################################################################
# CMakeLists.txt file for building ROOT net/alien package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(core)
ROOT_USE_PACKAGE(net/netx)
ROOT_USE_PACKAGE(proof/proofd)
ROOT_USE_PACKAGE(io/xml)


include_directories(${ALIEN_INCLUDE_DIR})


ROOT_GENERATE_DICTIONARY(G__Alien *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(RAliEn LINKDEF LinkDef.h DEPENDENCIES XMLIO Netx Tree Proof)
ROOT_LINKER_LIBRARY(RAliEn *.cxx G__Alien.cxx LIBRARIES ${ALIEN_LIBRARIES} 
                                               DEPENDENCIES XMLIO Netx Tree Proof)

ROOT_INSTALL_HEADERS()


