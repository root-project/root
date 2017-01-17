############################################################################
# CMakeLists.txt file for building ROOT net/netxng package
# @author Lukasz Janyst <ljanyst@cern.ch>
############################################################################

include_directories(${XROOTD_INCLUDE_DIRS})
add_definitions(${XROOTD_CFLAGS})

ROOT_GENERATE_DICTIONARY(G__NetxNG *.h MODULE NetxNG LINKDEF LinkDef.h)

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-shadow)
ROOT_LINKER_LIBRARY(NetxNG *.cxx G__NetxNG.cxx LIBRARIES ${XROOTD_LIBRARIES} DEPENDENCIES Net RIO Thread BUILTINS XROOTD)

ROOT_INSTALL_HEADERS()
