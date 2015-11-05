############################################################################
# CMakeLists.txt file for building ROOT net/net package
############################################################################

ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(math/mathcore)


ROOT_GLOB_HEADERS(headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/inc inc/*.h)
ROOT_GLOB_SOURCES(sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/src src/*.cxx)

if(NOT ssl)
  list(REMOVE_ITEM headers TSSLSocket.h)
  list(REMOVE_ITEM sources TSSLSocket.cxx)
  set(ssllib)
else()
  include_directories(${OPENSSL_INCLUDE_DIR})
  set(ssllib ${OPENSSL_LIBRARIES} ${CMAKE_DL_LIBS})
  add_definitions(-DR__SSL)
endif()

if(NOT ssl OR NOT CRYPTLIBS)
  list(REMOVE_ITEM headers TS3HTTPRequest.h)
  list(REMOVE_ITEM headers TS3WebFile.h)
  list(REMOVE_ITEM sources TS3HTTPRequest.cxx)
  list(REMOVE_ITEM sources TS3WebFile.cxx)
  add_definitions(-DR__NO_CRYPTO)
endif()

if(MACOSX_SSL_DEPRECATED)
  set_source_files_properties(src/TSSLSocket.cxx COMPILE_FLAGS "-Wno-deprecated-declarations" )
endif()

ROOT_GENERATE_DICTIONARY(G__Net ${headers} LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(Net LINKDEF LinkDef.h DEPENDENCIES MathCore RIO )
ROOT_LINKER_LIBRARY(Net ${sources} G__Net.cxx LIBRARIES ${ssllib} ${CRYPTLIBS} DEPENDENCIES MathCore RIO )

if(builtin_openssl)
  ROOT_ADD_BUILTIN_DEPENDENCIES(Net OPENSSL)
endif()

ROOT_INSTALL_HEADERS()
