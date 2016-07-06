############################################################################
# CMakeLists.txt file for building ROOT proof/proofd package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(proof/proof)
ROOT_USE_PACKAGE(net/rpdutils)
ROOT_USE_PACKAGE(net/auth)

include_directories(${XROOTD_INCLUDE_DIRS})
add_definitions(${XROOTD_CFLAGS})
ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-nonnull-compare)
ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-undefined-bool-conversion)

if(WIN32)
  ROOT_LINKER_LIBRARY(XrdProofd XProofProtUtils.cxx LIBRARIES ${XROOTD_LIBRARIES})
else()
  ROOT_EXECUTABLE(proofd proofd.cxx ${CMAKE_SOURCE_DIR}/core/clib/src/strlcat.c 
                         LIBRARIES ${XROOTD_LIBRARIES} rpdutil rsa ${GLOBUS_LIBRARIES} ${OPENSSL_LIBRARIES} ${CMAKE_DL_LIBS} ${CRYPTLIBS} )
  if (NOT XROOTD_NOMAIN)
     ROOT_EXECUTABLE(xproofd X*.cxx ${CMAKE_SOURCE_DIR}/net/rpdutils/src/rpdconn.cxx
                         LIBRARIES ${XROOTD_LIBRARIES} ${SYSLIBS})
  endif ()
  ROOT_LINKER_LIBRARY(XrdProofd X*.cxx ${CMAKE_SOURCE_DIR}/net/rpdutils/src/rpdconn.cxx
                         LIBRARIES ${XROOTD_LIBRARIES} rpdutil ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
  set_target_properties(XrdProofd PROPERTIES LINK_INTERFACE_LIBRARIES "" )
endif()

if(builtin_xrootd)
  add_dependencies(XrdProofd XROOTD)
  if (NOT XROOTD_NOMAIN)
     add_dependencies(xproofd XROOTD)
  endif ()
  add_dependencies(proofd XROOTD)
endif()

if(builtin_openssl)
  ROOT_ADD_BUILTIN_DEPENDENCIES(proofd OPENSSL)
endif()


ROOT_INSTALL_HEADERS()
