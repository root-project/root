############################################################################
# CMakeLists.txt file for building ROOT net/auth package
# @author Pere Mato, CERN
############################################################################

add_library(rsa STATIC src/rsaaux.cxx src/rsalib.cxx src/rsafun.cxx)
target_include_directories(rsa PUBLIC res)
set_property(TARGET rsa PROPERTY POSITION_INDEPENDENT_CODE ON)

ROOT_STANDARD_LIBRARY_PACKAGE(RootAuth
  HEADERS
    AuthConst.h
    TAuthenticate.h
    THostAuth.h
    TRootAuth.h
    TRootSecContext.h
  SOURCES
    TAuthenticate.cxx
    THostAuth.cxx
    TRootAuth.cxx
    TRootSecContext.cxx
  LINKDEF
    LinkDefRoot.h
  LIBRARIES
    ${CRYPTLIBS} rsa
  DEPENDENCIES
    Net
    RIO
)

if(ssl)
  target_link_libraries(RootAuth PUBLIC ${OPENSSL_LIBRARIES})
endif()
