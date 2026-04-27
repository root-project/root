# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT net/net package
############################################################################

if(ssl)
  add_definitions(-DR__SSL)

  set(NET_SSL_HEADERS
    TSSLSocket.h
  )

  set(NET_SSL_SOURCES
    src/TSSLSocket.cxx
  )

  if(MACOSX_SSL_DEPRECATED)
    set_source_files_properties(src/TSSLSocket.cxx
      COMPILE_FLAGS -Wno-deprecated-declarations)
  endif()
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(Net
  HEADERS
    NetErrors.h
    RRemoteProtocol.h
    TApplicationRemote.h
    TApplicationServer.h
    TFileStager.h
    TMessage.h
    TMonitor.h
    TParallelMergingFile.h
    TPServerSocket.h
    TPSocket.h
    TSecContext.h
    TServerSocket.h
    TSocket.h
    TSQLColumnInfo.h
    TSQLMonitoring.h
    TSQLResult.h
    TSQLRow.h
    TSQLServer.h
    TSQLStatement.h
    TSQLTableInfo.h
    TUDPSocket.h
    ${NET_SSL_HEADERS}
  SOURCES
    src/NetErrors.cxx
    src/TApplicationRemote.cxx
    src/TApplicationServer.cxx
    src/TFileStager.cxx
    src/TMessage.cxx
    src/TMonitor.cxx
    src/TParallelMergingFile.cxx
    src/TPServerSocket.cxx
    src/TPSocket.cxx
    src/TSecContext.cxx
    src/TServerSocket.cxx
    src/TSocket.cxx
    src/TSQLColumnInfo.cxx
    src/TSQLMonitoring.cxx
    src/TSQLResult.cxx
    src/TSQLRow.cxx
    src/TSQLServer.cxx
    src/TSQLStatement.cxx
    src/TSQLTableInfo.cxx
    src/TUDPSocket.cxx
    ${NET_SSL_SOURCES}
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    RIO
  LIBRARIES
    ${ROOT_ATOMIC_LIBS}
)

if(ssl)
  target_include_directories(Net PRIVATE ${OPENSSL_INCLUDE_DIR})
  target_link_libraries(Net PRIVATE ${OPENSSL_LIBRARIES})
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)