// @(#)root/net:$Id$
// Author: Fons Rademakers   19/1/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPServerSocket
#define ROOT_TPServerSocket


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPServerSocket                                                       //
//                                                                      //
// This class implements parallel server sockets. A parallel server     //
// socket waits for requests to come in over the network. It performs   //
// some operation based on that request and then possibly returns a     //
// full duplex parallel socket to the requester. The actual work is     //
// done via the TSystem class (either TUnixSystem or TWinNTSystem).     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TServerSocket.h"

class TPSocket;


class TPServerSocket : public TServerSocket {

private:
   Int_t  fTcpWindowSize; // size of tcp window (for window scaling)

   TPServerSocket(const TPServerSocket &) = delete;
   void operator=(const TPServerSocket &) = delete;

public:
   TPServerSocket(Int_t port, Bool_t reuse = kFALSE,
                  Int_t backlog = kDefaultBacklog,
                  Int_t tcpwindowsize = -1);
   TPServerSocket(const char *service, Bool_t reuse = kFALSE,
                  Int_t backlog = kDefaultBacklog,
                  Int_t tcpwindowsize = -1);

   virtual ~TPServerSocket() {}

   TSocket *Accept(UChar_t Opt = kSrvNoAuth) override;

   ClassDefOverride(TPServerSocket,0)  // Parallel server socket
};

#endif
