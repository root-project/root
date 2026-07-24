# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT net/net package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Net
  HEADERS
    NetErrors.h
    RRemoteProtocol.h
    TApplicationRemote.h
    TApplicationServer.h
    TFileStager.h
    TGrid.h
    TGridCollection.h
    TGridJDL.h
    TGridJob.h
    TGridJobStatus.h
    TGridJobStatusList.h
    TGridResult.h
    TMessage.h
    TMonitor.h
    TParallelMergingFile.h
    TPServerSocket.h
    TPSocket.h
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
  SOURCES
    src/NetErrors.cxx
    src/TApplicationRemote.cxx
    src/TApplicationServer.cxx
    src/TGrid.cxx
    src/TGridJDL.cxx
    src/TGridJob.cxx
    src/TGridJobStatus.cxx
    src/TGridJobStatusList.cxx
    src/TGridResult.cxx
    src/TFileStager.cxx
    src/TMessage.cxx
    src/TMonitor.cxx
    src/TParallelMergingFile.cxx
    src/TPServerSocket.cxx
    src/TPSocket.cxx
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
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    RIO
  LIBRARIES
    ${ROOT_ATOMIC_LIBS}
)
ROOT_ADD_TEST_SUBDIRECTORY(test)
