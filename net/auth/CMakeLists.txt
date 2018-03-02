############################################################################
# CMakeLists.txt file for building ROOT net/auth package
# @author Pere Mato, CERN
############################################################################

ROOT_GLOB_HEADERS(headers inc/*.h)
list(REMOVE_ITEM headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/LinkDefAFS.h
                         ${CMAKE_CURRENT_SOURCE_DIR}/inc/LinkDefRoot.h
                         ${CMAKE_CURRENT_SOURCE_DIR}/inc/AFSAuth.h
                         ${CMAKE_CURRENT_SOURCE_DIR}/inc/AFSAuthTypes.h
                         ${CMAKE_CURRENT_SOURCE_DIR}/inc/TAFS.h)

ROOT_GLOB_SOURCES(sources src/*.cxx)
list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/DaemonUtils.cxx
                         ${CMAKE_CURRENT_SOURCE_DIR}/src/AFSAuth.cxx
                         ${CMAKE_CURRENT_SOURCE_DIR}/src/TAFS.cxx)

if(MACOSX_SSK_DEPRECATED)
  add_definitions(-Wno-deprecated-declarations)
endif()

#---static library needed for rootd and proofd --------------------------------------------
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/res)

add_library(rsa STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/rsaaux.cxx
                       ${CMAKE_CURRENT_SOURCE_DIR}/src/rsalib.cxx
                       ${CMAKE_CURRENT_SOURCE_DIR}/src/rsafun.cxx)
add_dependencies(rsa move_headers)

if(ssl)
  include_directories(${OPENSSL_INCLUDE_DIR})
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(RootAuth
                              NO_INSTALL_HEADERS
                              HEADERS ${headers}
                              SOURCES ${sources}
                              LINKDEF LinkDefRoot.h
                              LIBRARIES ${CRYPTLIBS}
                              DEPENDENCIES Net RIO)

if(afs)
  ROOT_STANDARD_LIBRARY_PACKAGE(AFSAuth
                                NO_INSTALL_HEADERS
                                HEADERS AFSAuth.h AFSAuthTypes.h TAFS.h
                                LINKDEF LinkDefAFS.h
                                LIBRARIES ${CRYPTLIBS}
                                DEPENDENCIES Net RIO)
else()
  set(installoptions FILTER "AFS")
endif()

ROOT_INSTALL_HEADERS(${installoptions})


