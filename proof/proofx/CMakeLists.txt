# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT proof/proofx package
# @author Pere Mato, CERN
############################################################################

add_definitions(${XROOTD_CFLAGS})
include_directories(${XROOTD_INCLUDE_DIRS})
include_directories(AFTER ${CMAKE_CURRENT_SOURCE_DIR}/../proofd/inc)
link_directories(${XROOTD_LIBRARY_DIR})
set_source_files_properties(src/TXSocket.cxx COMPILE_FLAGS -Wno-address)

if(UNIX)
  set(PROOFX_EXTRA_HEADERS TXUnixSocket.h)
  set(PROOFX_EXTRA_SOURCES src/TXUnixSocket.cxx)
elseif(WIN32)
  set(PROOFX_EXTRA_SOURCES
      ../proofd/src/XrdProofConn.cxx
      ../proofd/src/XrdProofPhyConn.cxx
      ../proofd/src/XProofProtUtils.cxx)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(Proofx
  HEADERS
    TXHandler.h
    TXProofMgr.h
    TXProofServ.h
    TXSlave.h
    TXSocket.h
    TXSocketHandler.h
    ${PROOFX_EXTRA_HEADERS}
  SOURCES
    src/TXHandler.cxx
    src/TXProofMgr.cxx
    src/TXProofServ.cxx
    src/TXSlave.cxx
    src/TXSocket.cxx
    src/TXSocketHandler.cxx
    ${PROOFX_EXTRA_SOURCES}
  LIBRARIES
    ${XROOTD_LIBRARIES}
    XrdProofd
  DEPENDENCIES
    Net
    Proof
    Thread
  BUILTINS
    XROOTD
)
