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
    TS3HTTPRequest.h
    TS3WebFile.h
  )

  set(NET_SSL_SOURCES
    src/TSSLSocket.cxx
    src/TS3HTTPRequest.cxx
    src/TS3WebFile.cxx
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
    TFTP.h
    TGrid.h
    TGridCollection.h
    TGridJDL.h
    TGridJob.h
    TGridJobStatus.h
    TGridJobStatusList.h
    TGridResult.h
    TMessage.h
    TMonitor.h
    TNetFile.h
    TNetFileStager.h
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
    TWebFile.h
    ${NET_SSL_HEADERS}
  SOURCES
    src/NetErrors.cxx
    src/TApplicationRemote.cxx
    src/TApplicationServer.cxx
    src/TFileStager.cxx
    src/TFTP.cxx
    src/TGrid.cxx
    src/TGridJDL.cxx
    src/TGridJob.cxx
    src/TGridJobStatus.cxx
    src/TGridJobStatusList.cxx
    src/TGridResult.cxx
    src/TMessage.cxx
    src/TMonitor.cxx
    src/TNetFile.cxx
    src/TNetFileStager.cxx
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
    src/TWebFile.cxx
    ${NET_SSL_SOURCES}
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    RIO
)

if(ssl)
  target_include_directories(Net PRIVATE ${OPENSSL_INCLUDE_DIR})
  target_link_libraries(Net PRIVATE ${OPENSSL_LIBRARIES})
endif()
