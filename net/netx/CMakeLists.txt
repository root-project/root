############################################################################
# CMakeLists.txt file for building ROOT net/netx package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(net/net)
ROOT_USE_PACKAGE(proof/proofd)

include_directories(${XROOTD_INCLUDE_DIRS})
add_definitions(${XROOTD_CFLAGS})
#---------------------------------------------------------------------------

ROOT_GENERATE_DICTIONARY(G__Netx *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(Netx LINKDEF LinkDef.h DEPENDENCIES Net RIO Thread)
ROOT_LINKER_LIBRARY(Netx *.cxx G__Netx.cxx LIBRARIES ${XROOTD_LIBRARIES} DEPENDENCIES Net RIO Thread)

ROOT_INSTALL_HEADERS()

