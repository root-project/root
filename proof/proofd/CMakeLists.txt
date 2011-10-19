############################################################################
# CMakeLists.txt file for building ROOT proof/proofd package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(proof/proof)
ROOT_USE_PACKAGE(net/rpdutils)
ROOT_USE_PACKAGE(net/auth)

if(WIN32)
  ROOT_LINKER_LIBRARY(XrdProofd XProofProtUtils.cxx LIBRARIES libXrdClient)
else()
  set(libs Xrd XrdOuc XrdNet XrdNetUtil XrdSys XrdClient XrdSut ${BONJOUR_LIBRARIES})
  set(exelibs libXrd.a libXrdClient.a libXrdNet.a libXrdNetUtil.a libXrdOuc.a libXrdSys.a libXrdSut.a)

  ROOT_EXECUTABLE(proofd proofd.cxx ${CMAKE_SOURCE_DIR}/core/clib/src/strlcat.c 
                         LIBRARIES ${exelibs} rpdutil rsa ${OPENSSL_LIBRARIES} ${CRYPTLIBS} )
  ROOT_EXECUTABLE(xproofd X*.cxx ${CMAKE_SOURCE_DIR}/net/rpdutils/src/rpdconn.cxx 
                         LIBRARIES ${exelibs} ${SYSLIBS})
  ROOT_LINKER_LIBRARY(XrdProofd X*.cxx ${CMAKE_SOURCE_DIR}/net/rpdutils/src/rpdconn.cxx 
                         LIBRARIES ${libs} rpdutil ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
  set_target_properties(XrdProofd PROPERTIES LINK_INTERFACE_LIBRARIES "" )
  
  add_dependencies(proofd XROOTD)
  add_dependencies(xproofd XROOTD)
  add_dependencies(XrdProofd XROOTD)
endif()

ROOT_INSTALL_HEADERS()
