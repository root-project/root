############################################################################
# CMakeLists.txt file for building ROOT net/netx package
# @author Pere Mato, CERN
############################################################################

include_directories(${XROOTD_INCLUDE_DIRS})
add_definitions(${XROOTD_CFLAGS})
#---------------------------------------------------------------------------

ROOT_GENERATE_DICTIONARY(G__Netx *.h MODULE Netx LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(Netx *.cxx G__Netx.cxx LIBRARIES ${XROOTD_LIBRARIES} DEPENDENCIES Net RIO Thread)

if(builtin_xrootd)
  ROOT_ADD_BUILTIN_DEPENDENCIES(Netx XROOTD)
endif()

ROOT_INSTALL_HEADERS()
