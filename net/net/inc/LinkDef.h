/* @(#)root/net:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CLING__

#pragma link C++ enum EMessageTypes;
#pragma link C++ enum ESockOptions;
#pragma link C++ enum ESendRecvOptions;

#pragma link C++ global ROOT::Deprecated::gGrid;
#pragma link C++ global gGridJobStatusList;

#pragma link C++ class TServerSocket;
#pragma link C++ class TSocket;
#pragma link C++ class TPServerSocket;
#pragma link C++ class TPSocket;
#pragma link C++ class TMessage;
#pragma link C++ class TMonitor;
#pragma link C++ class TNetFile;
#pragma link C++ class ROOT::Deprecated::TNetFileStager;
#pragma link C++ class TNetSystem;
#pragma link C++ class ROOT::Deprecated::TWebFile;
#pragma link C++ class ROOT::Deprecated::TWebSystem;
#pragma link C++ class TFTP;
#pragma link C++ class TSQLServer;
#pragma link C++ class TSQLResult;
#pragma link C++ class TSQLRow;
#pragma link C++ class TSQLStatement;
#pragma link C++ class TSQLTableInfo;
#pragma link C++ class TSQLColumnInfo;
#pragma link C++ class TSQLMonitoringWriter;
#pragma link C++ class ROOT::Deprecated::TGrid;
#pragma link C++ class ROOT::Deprecated::TGridResult+;
#pragma link C++ class ROOT::Deprecated::TGridJDL+;
#pragma link C++ class ROOT::Deprecated::TGridJob+;
#pragma link C++ class ROOT::Deprecated::TGridJobStatus+;
#pragma link C++ class ROOT::Deprecated::TGridJobStatusList+;
#pragma link C++ class ROOT::Deprecated::TGridCollection+;
#pragma link C++ class ROOT::Deprecated::TSecContext;
#pragma link C++ class ROOT::Deprecated::TSecContextCleanup;
#pragma link C++ class TFileStager;
#pragma link C++ class TApplicationRemote;
#pragma link C++ class TApplicationServer;
#pragma link C++ class TUDPSocket;
#pragma link C++ class TParallelMergingFile+;

#ifdef R__SSL
#pragma link C++ class ROOT::Deprecated::TS3HTTPRequest+;
#pragma link C++ class ROOT::Deprecated::TS3WebFile+;
#pragma link C++ class TSSLSocket;
#endif

#pragma read sourceClass="TGridCollection" version="[-1]" targetClass="ROOT::Deprecated::TGridCollection"
#pragma read sourceClass="TGridJDL" version="[-1]" targetClass="ROOT::Deprecated::TGridJDL"
#pragma read sourceClass="TGridJob" version="[-1]" targetClass="ROOT::Deprecated::TGridJob"
#pragma read sourceClass="TGridJobStatus" version="[-1]" targetClass="ROOT::Deprecated::TGridJobStatus"
#pragma read sourceClass="TGridJobStatusList" version="[-1]" targetClass="ROOT::Deprecated::TGridJobStatusList"
#pragma read sourceClass="TGridResult" version="[-1]" targetClass="ROOT::Deprecated::TGridResult"

#endif
