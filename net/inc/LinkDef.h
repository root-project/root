/* @(#)root/net:$Name:  $:$Id: LinkDef.h,v 1.2 2000/11/27 10:46:20 rdm Exp $ */

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

#pragma link C++ class TInetAddress;
#pragma link C++ class TAuthenticate;
#pragma link C++ class TServerSocket;
#pragma link C++ class TSocket;
#pragma link C++ class TMessage;
#pragma link C++ class TMonitor;
#pragma link C++ class TUrl;
#pragma link C++ class TNetFile;
#pragma link C++ class TWebFile;
#pragma link C++ class TCache;
#pragma link C++ class TSQLServer;
#pragma link C++ class TSQLResult;
#pragma link C++ class TSQLRow;

#endif
