# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT proof/proofd package
# @author Pere Mato, CERN
############################################################################

set(PROOFD_SOURCES
  src/XpdObject.cxx
  src/XProofProtUtils.cxx
  src/XrdProofConn.cxx
  src/XrdProofdAdmin.cxx
  src/XrdProofdAux.cxx
  src/XrdProofdClient.cxx
  src/XrdProofdClientMgr.cxx
  src/XrdProofdConfig.cxx
  src/XrdProofdManager.cxx
  src/XrdProofdNetMgr.cxx
  src/XrdProofdPriorityMgr.cxx
  src/XrdProofdProofServ.cxx
  src/XrdProofdProofServMgr.cxx
  src/XrdProofdProtocol.cxx
  src/XrdProofdResponse.cxx
  src/XrdProofdSandbox.cxx
  src/XrdProofGroup.cxx
  src/XrdProofPhyConn.cxx
  src/XrdProofSched.cxx
  src/XrdProofWorker.cxx
  src/XrdROOT.cxx
  ../../net/rpdutils/src/rpdconn.cxx
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../net/rpdutils/res)

if(WIN32)
  ROOT_LINKER_LIBRARY(XrdProofd
    XProofProtUtils.cxx
    LIBRARIES
      ROOT::XRootD
  )
else()
  if (NOT XROOTD_NOMAIN)
    ROOT_EXECUTABLE(xproofd
      ${PROOFD_SOURCES}
      LIBRARIES
        ROOT::XRootD
    )
  endif()

  ROOT_LINKER_LIBRARY(XrdProofd
    ${PROOFD_SOURCES}
    LIBRARIES
      ROOT::XRootD
      rpdutil
      ${CMAKE_THREAD_LIBS_INIT}
      ${CMAKE_DL_LIBS}
  )

  ROOT_EXECUTABLE(proofexecv
    proofexecv.cxx
    ${CMAKE_SOURCE_DIR}/net/rpdutils/src/rpdconn.cxx
    ${CMAKE_SOURCE_DIR}/net/rpdutils/src/rpdpriv.cxx
  )

  target_include_directories(proofexecv PRIVATE
    ${CMAKE_SOURCE_DIR}/core/base/inc    # for RConfig.hxx
    ${CMAKE_BINARY_DIR}/ginclude         # for RConfigure.h
  )

endif()
