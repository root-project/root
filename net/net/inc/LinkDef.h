/* @(#)root/net:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__

#pragma link C++ enum EMessageTypes;
#pragma link C++ enum ESockOptions;
#pragma link C++ enum ESendRecvOptions;

#pragma link C++ global gGrid;
#pragma link C++ global gGridJobStatusList;

#pragma link C++ global kSrvAuth;
#pragma link C++ global kSrvNoAuth;

#pragma link C++ class TServerSocket;
#pragma link C++ class TSocket;
#pragma link C++ class TPServerSocket;
#pragma link C++ class TPSocket;
#pragma link C++ class TMessage;
#pragma link C++ class TMonitor;
#pragma link C++ class TNetFile;
#pragma link C++ class TNetFileStager;
#pragma link C++ class TNetSystem;
#pragma link C++ class TWebFile;
#pragma link C++ class TWebSystem;
#pragma link C++ class TFTP;
#pragma link C++ class TSQLServer;
#pragma link C++ class TSQLResult;
#pragma link C++ class TSQLRow;
#pragma link C++ class TSQLStatement;
#pragma link C++ class TSQLTableInfo;
#pragma link C++ class TSQLColumnInfo;
#pragma link C++ class TSQLMonitoringWriter;
#pragma link C++ class TGrid;
#pragma link C++ class TGridResult+;
#pragma link C++ class TGridJDL+;
#pragma link C++ class TGridJob+;
#pragma link C++ class TGridJobStatus+;
#pragma link C++ class TGridJobStatusList+;
#pragma link C++ class TGridCollection+;
#pragma link C++ class TSecContext;
#pragma link C++ class TSecContextCleanup;
#pragma link C++ class TFileStager;
#pragma link C++ class TApplicationRemote;
#pragma link C++ class TApplicationServer;
#pragma link C++ class TUdpSocket;
#ifndef R__NO_CRYPTO
#pragma link C++ class THTTPMessage+;
#pragma link C++ class TAS3File+;
#pragma link C++ class TGSFile+;
#endif
#ifdef R__SSL
#pragma link C++ class TSSLSocket;
#endif
#pragma link C++ class TParallelMergingFile+;

#endif
