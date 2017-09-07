############################################################################
# CMakeLists.txt file for building ROOT net/net package
############################################################################

ROOT_GLOB_HEADERS(headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/inc inc/*.h)
ROOT_GLOB_SOURCES(sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/src src/*.cxx)

if(NOT ssl)
  list(REMOVE_ITEM headers TSSLSocket.h)
  list(REMOVE_ITEM sources TSSLSocket.cxx)
  set(ssllib)
else()
  set(ssllib ${OPENSSL_LIBRARIES} ${CMAKE_DL_LIBS})
  add_definitions(-DR__SSL)
  include_directories(${OPENSSL_INCLUDE_DIR})
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

ROOT_STANDARD_LIBRARY_PACKAGE(Net
                              HEADERS ${headers}
                              SOURCES ${sources}
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              LIBRARIES ${ssllib} ${CRYPTLIBS}
                              DEPENDENCIES RIO BUILTINS OPENSSL)

