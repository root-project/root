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

#pragma link C++ class TServerSocket;
#pragma link C++ class TSocket;
#pragma link C++ class TPServerSocket;
#pragma link C++ class TPSocket;
#pragma link C++ class TMessage;
#pragma link C++ class TMonitor;
#pragma link C++ class TSQLServer;
#pragma link C++ class TSQLResult;
#pragma link C++ class TSQLRow;
#pragma link C++ class TSQLStatement;
#pragma link C++ class TSQLTableInfo;
#pragma link C++ class TSQLColumnInfo;
#pragma link C++ class TSQLMonitoringWriter;
#pragma link C++ class ROOT::Deprecated::TSecContext;
#pragma link C++ class ROOT::Deprecated::TSecContextCleanup;
#pragma link C++ class TFileStager;
#pragma link C++ class TApplicationRemote;
#pragma link C++ class TApplicationServer;
#pragma link C++ class TUDPSocket;
#pragma link C++ class TParallelMergingFile+;

#ifdef R__SSL
#pragma link C++ class TSSLSocket;
#endif

#endif
