/* @(#)root/net:$Name:  $:$Id: LinkDef.h,v 1.9 2004/02/19 00:11:18 rdm Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ enum EMessageTypes;
#pragma link C++ enum ESockOptions;
#pragma link C++ enum ESendRecvOptions;

#pragma link C++ global gGrid;

#pragma link C++ struct Grid_Result_t;

#pragma link C++ class TInetAddress;
#pragma link C++ class TAuthenticate;
#pragma link C++ class TServerSocket;
#pragma link C++ class TSocket;
#pragma link C++ class TPServerSocket;
#pragma link C++ class TPSocket;
#pragma link C++ class TMessage;
#pragma link C++ class TMonitor;
#pragma link C++ class TUrl;
#pragma link C++ class TNetFile;
#pragma link C++ class TNetSystem;
#pragma link C++ class TWebFile;
#pragma link C++ class TCache;
#pragma link C++ class TFTP;
#pragma link C++ class TSQLServer;
#pragma link C++ class TSQLResult;
#pragma link C++ class TSQLRow;
#pragma link C++ class TGrid;
#pragma link C++ class TGridResult;
#pragma link C++ class TGridProof;
#pragma link C++ class THostAuth;
#pragma link C++ class TSecContext;
#pragma link C++ class TSecContextCleanup;

#endif
