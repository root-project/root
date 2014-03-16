############################################################################
# CMakeLists.txt file for building ROOT net/netxng package
# @author Lukasz Janyst <ljanyst@cern.ch>
############################################################################

ROOT_USE_PACKAGE(net/net)

include_directories(${XROOTD_INCLUDE_DIRS})
add_definitions(${XROOTD_CFLAGS})

ROOT_GENERATE_DICTIONARY(G__NetxNG *.h MODULE NetxNG LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(NetxNG *.cxx G__NetxNG.cxx LIBRARIES ${XROOTD_LIBRARIES} DEPENDENCIES Net RIO Thread)

ROOT_INSTALL_HEADERS()

if(builtin_xrootd)
  add_dependencies(NetxNG XROOTD)
endif()
