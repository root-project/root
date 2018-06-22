############################################################################
# CMakeLists.txt file for building ROOT net/net package
############################################################################

ROOT_GLOB_HEADERS(headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/inc inc/*.h)
ROOT_GLOB_SOURCES(sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/src src/*.cxx)

if(ssl)
  add_definitions(-DR__SSL)
  include_directories(${OPENSSL_INCLUDE_DIR})
else()
  list(REMOVE_ITEM headers TSSLSocket.h)
  list(REMOVE_ITEM sources TSSLSocket.cxx)
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
                              DEPENDENCIES RIO)

if(ssl)
  target_link_libraries(Net ${OPENSSL_LIBRARIES} ${CRYPTLIBS})
endif()
