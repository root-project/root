############################################################################
# CMakeLists.txt file for building ROOT net/netx package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(net/net)

#---Using or nor an external installation of XROOTD-------------------------
if(NOT XROOTD_FOUND)
  set(XROOTD_LIBRARY_DIR ${CMAKE_SOURCE_DIR}/net/xrootd/src/xrootd/lib)
  set(XROOTD_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/net/xrootd/src/xrootd/src)
endif()
include_directories(${XROOTD_INCLUDE_DIR})
link_directories(${XROOTD_LIBRARY_DIR})
#---------------------------------------------------------------------------

include_directories(${CMAKE_SOURCE_DIR}/net/xrootd/src/xrootd/src)
if(WIN32)
  add_definitions(-DNOGDI)
  set(xrdclient libXrdClient)
endif()

ROOT_GENERATE_DICTIONARY(G__Netx *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(Netx LINKDEF LinkDef.h DEPENDENCIES Net RIO Thread)
ROOT_LINKER_LIBRARY(Netx *.cxx G__Netx.cxx LIBRARIES ${xrdclient} DEPENDENCIES Net RIO Thread)

ROOT_INSTALL_HEADERS()

