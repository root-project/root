
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
else()
  file(GLOB headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/inc ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)
  file(GLOB sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)
endif()

include_directories(${XROOTD_INCLUDE_DIR})
link_directories(${XROOTD_LIBRARY_DIR})
add_definitions(${XROOTD_CFLAGS})

ROOT_GENERATE_DICTIONARY(G__Proofx ${headers} LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(Proofx LINKDEF LinkDef.h DEPENDENCIES Net Proof Thread)
ROOT_LINKER_LIBRARY(Proofx ${sources} G__Proofx.cxx LIBRARIES ${XROOTD_LIBRARIES} XrdProofd DEPENDENCIES Net Proof Thread )

ROOT_INSTALL_HEADERS()
