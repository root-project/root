############################################################################
# CMakeLists.txt file for building ROOT proof/proofx package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(proof/proofd)
ROOT_USE_PACKAGE(proof/proofplayer)

if(WIN32)
  set(headers TXProofMgr.h TXSlave.h TXSocket.h TXSocketHandler.h TXHandler.h)
  set(sources TXProofMgr.cxx TXSlave.cxx TXSocket.cxx TXSocketHandler.cxx TXHandler.cxx 
               ../proofd/src/XrdProofConn.cxx ../proofd/src/XrdProofPhyConn.cxx ../proofd/src/XProofProtUtils.cxx)
  set(libs libXrdClient)
else()
  file(GLOB headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/inc ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)
  file(GLOB sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)
  set(libs XrdOuc XrdSys XrdNet XrdClient)
endif()

#---Using or nor an external installation of XROOTD-------------------------
if(NOT XROOTD_FOUND)
  set(XROOTD_LIBRARY_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  set(XROOTD_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/net/xrootd/src/xrootd/src)
endif()

include_directories(${XROOTD_INCLUDE_DIR})
link_directories(${XROOTD_LIBRARY_DIR})

ROOT_GENERATE_DICTIONARY(G__Proofx ${headers} LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(Proofx LINKDEF LinkDef.h DEPENDENCIES Net Proof Thread)
ROOT_LINKER_LIBRARY(Proofx ${sources} G__Proofx.cxx LIBRARIES ${libs} XrdProofd DEPENDENCIES Net Proof Thread )

ROOT_INSTALL_HEADERS()
