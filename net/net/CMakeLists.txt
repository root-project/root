############################################################################
# CMakeLists.txt file for building ROOT net/net package
############################################################################

ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(math/mathcore)


file(GLOB headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/inc inc/*.h)
file(GLOB sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/src src/*.cxx)

if(NOT ssl)
  list(REMOVE_ITEM headers TSSLSocket.h)
  list(REMOVE_ITEM sources TSSLSocket.cxx)
  set(ssllib)
else()
  set(ssllib ${OPENSSL_LIBRARIES})
  add_definitions(-DR__SSL)
endif()

if(NOT crypto)
  list(REMOVE_ITEM headers TAS3File.h)
  list(REMOVE_ITEM headers TGSFile.h)
  list(REMOVE_ITEM headers THTTPMessage.h)
  list(REMOVE_ITEM sources TAS3File.cxx)
  list(REMOVE_ITEM sources TGSFile.cxx)
  list(REMOVE_ITEM sources THTTPMessage.cxx)
  add_definitions(-DR__NO_CRYPTO)
endif()

ROOT_GENERATE_DICTIONARY(G__Net ${headers} LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(Net LINKDEF LinkDef.h DEPENDENCIES MathCore RIO )
ROOT_LINKER_LIBRARY(Net ${sources} G__Net.cxx LIBRARIES ${ssllib} ${CRYPTLIBS} DEPENDENCIES MathCore RIO )

ROOT_INSTALL_HEADERS()
