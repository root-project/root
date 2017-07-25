############################################################################
# CMakeLists.txt file for building ROOT proof/proofd package
# @author Pere Mato, CERN
############################################################################

include_directories(${XROOTD_INCLUDE_DIRS})
include_directories(AFTER ${CMAKE_CURRENT_SOURCE_DIR}/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../net/rpdutils/res)
add_definitions(${XROOTD_CFLAGS})
#
# This should not be needed any longer after fix 039136791216af4620b9041e89d60c3ac0bab2f7
# Uncomment if not the case (G Ganis, 18/5/2016)
# ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-undefined-bool-conversion)

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-nonnull-compare)

if(WIN32)
  ROOT_LINKER_LIBRARY(XrdProofd XProofProtUtils.cxx LIBRARIES ${XROOTD_LIBRARIES})
else()
  ROOT_EXECUTABLE(proofd proofd.cxx ${CMAKE_SOURCE_DIR}/core/clib/src/strlcat.c
                         LIBRARIES ${XROOTD_LIBRARIES} rpdutil rsa ${GLOBUS_LIBRARIES} ${OPENSSL_LIBRARIES} ${CRYPTLIBS}
                         BUILTINS XROOTD OPENSSL)
  if (NOT XROOTD_NOMAIN)
     ROOT_EXECUTABLE(xproofd X*.cxx ${CMAKE_SOURCE_DIR}/net/rpdutils/src/rpdconn.cxx
                         LIBRARIES ${XROOTD_LIBRARIES} ${SYSLIBS} BUILTINS XROOTD)
  endif ()
  ROOT_LINKER_LIBRARY(XrdProofd X*.cxx ${CMAKE_SOURCE_DIR}/net/rpdutils/src/rpdconn.cxx
                         LIBRARIES ${XROOTD_LIBRARIES} rpdutil ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS} BUILTINS XROOTD)
  set_target_properties(XrdProofd PROPERTIES INTERFACE_LINK_LIBRARIES "")
  ROOT_EXECUTABLE(proofexecv proofexecv.cxx ${CMAKE_SOURCE_DIR}/net/rpdutils/src/rpdconn.cxx ${CMAKE_SOURCE_DIR}/net/rpdutils/src/rpdpriv.cxx
                         LIBRARIES ${SYSLIBS})
endif()
