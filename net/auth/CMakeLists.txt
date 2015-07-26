############################################################################
# CMakeLists.txt file for building ROOT net/auth package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(net/net)
ROOT_USE_PACKAGE(io/io)

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
add_library(rsa STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/rsaaux.cxx
                       ${CMAKE_CURRENT_SOURCE_DIR}/src/rsalib.cxx
                       ${CMAKE_CURRENT_SOURCE_DIR}/src/rsafun.cxx)
add_dependencies(rsa move_headers)

if(ssl)
  include_directories(${OPENSSL_INCLUDE_DIR})
endif()

ROOT_GENERATE_DICTIONARY(G__RootAuth ${headers} MODULE RootAuth LINKDEF LinkDefRoot.h)
ROOT_LINKER_LIBRARY(RootAuth ${sources} G__RootAuth.cxx LIBRARIES ${CRYPTLIBS} DEPENDENCIES Net RIO )

if(afs)
  ROOT_GENERATE_DICTIONARY(G__AFSAuth AFSAuth.h AFSAuthTypes.h TAFS.h MODULE AFSAuth LINKDEF LinkDefAFS.h)
  ROOT_LINKER_LIBRARY(AFSAuth AFSAuth.cxx TAFS.cxx G__AFSAuth.cxx LIBRARIES ${CRYPTLIBS} DEPENDENCIES  Net RIO )
else()
  set(installoptions OPTIONS REGEX "AFS" EXCLUDE)
endif()

ROOT_INSTALL_HEADERS(${installoptions})


