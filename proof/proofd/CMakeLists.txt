############################################################################
# CMakeLists.txt file for building ROOT proof/proofd package
# @author Pere Mato, CERN
############################################################################

include_directories(${XROOTD_INCLUDE_DIRS})
include_directories(AFTER ${CMAKE_CURRENT_SOURCE_DIR}/inc)
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
                         LIBRARIES ${XROOTD_LIBRARIES} rpdutil rsa ${GLOBUS_LIBRARIES} ${OPENSSL_LIBRARIES} ${CRYPTLIBS})
  if (NOT XROOTD_NOMAIN)
     ROOT_EXECUTABLE(xproofd X*.cxx ${CMAKE_SOURCE_DIR}/net/rpdutils/src/rpdconn.cxx
                         LIBRARIES ${XROOTD_LIBRARIES} ${SYSLIBS})
  endif ()
  ROOT_LINKER_LIBRARY(XrdProofd X*.cxx ${CMAKE_SOURCE_DIR}/net/rpdutils/src/rpdconn.cxx
                         LIBRARIES ${XROOTD_LIBRARIES} rpdutil ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
  set_target_properties(XrdProofd PROPERTIES INTERFACE_LINK_LIBRARIES "")
  ROOT_EXECUTABLE(proofexecv proofexecv.cxx ${CMAKE_SOURCE_DIR}/net/rpdutils/src/rpdconn.cxx ${CMAKE_SOURCE_DIR}/net/rpdutils/src/rpdpriv.cxx
                         LIBRARIES ${SYSLIBS})
endif()

if(builtin_xrootd)
  if(NOT TARGET XROOTDLIBS)
    add_custom_command(OUTPUT ${XROOTD_LIBRARIES} DEPENDS XROOTD)
    add_custom_target(XROOTDLIBS DEPENDS ${XROOTD_LIBRARIES})
  endif()
  add_dependencies(XrdProofd XROOTDLIBS)
  if (NOT XROOTD_NOMAIN)
     add_dependencies(xproofd XROOTDLIBS)
  endif ()
  add_dependencies(proofd XROOTDLIBS)
endif()

if(builtin_openssl)
  ROOT_ADD_BUILTIN_DEPENDENCIES(proofd OPENSSL)
endif()


ROOT_INSTALL_HEADERS(OPTIONS REGEX inc/X EXCLUDE)
