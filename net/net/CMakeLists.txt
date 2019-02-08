############################################################################
# CMakeLists.txt file for building ROOT net/net package
############################################################################

set(headers
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
   TS3HTTPRequest.h
   TS3WebFile.h
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
   TSSLSocket.h
   TUDPSocket.h
   TWebFile.h
)
set(sources
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
   src/TS3HTTPRequest.cxx
   src/TS3WebFile.cxx
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
   src/TSSLSocket.cxx
   src/TUDPSocket.cxx
   src/TWebFile.cxx
)

if(ssl)
  add_definitions(-DR__SSL)
  include_directories(${OPENSSL_INCLUDE_DIR})
else()
  list(REMOVE_ITEM headers TSSLSocket.h)
  list(REMOVE_ITEM sources src/TSSLSocket.cxx)
endif()

if(NOT ssl OR NOT CRYPTLIBS)
  list(REMOVE_ITEM headers TS3HTTPRequest.h)
  list(REMOVE_ITEM headers TS3WebFile.h)
  list(REMOVE_ITEM sources src/TS3HTTPRequest.cxx)
  list(REMOVE_ITEM sources src/TS3WebFile.cxx)
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
  target_link_libraries(Net PRIVATE ${OPENSSL_LIBRARIES})
endif()
