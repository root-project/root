// @(#)root/proofx:$Id$
// Author: G. Ganis Oct 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXSocketHandler
#define ROOT_TXSocketHandler

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXSocketHandler                                                      //
//                                                                      //
// Input handler for xproofd sockets. These sockets cannot be directly  //
// monitored on their descriptor, because the reading activity goes via //
// the reader thread. This class allows to handle this problem.         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TSysEvtHandler
#include "TSysEvtHandler.h"
#endif
#ifndef ROOT_TXSocket
#include "TXSocket.h"
#endif

class TXSocketHandler : public TFileHandler {

// friend class TXSocket;

   TFileHandler   *fHandler;    // Handler associated to the input socket
   TSocket        *fInputSock;  // Input socket from client or master

   void    SetHandler(TFileHandler *h, TSocket *s)
                                   { fHandler = h; fInputSock = s; }

   static TXSocketHandler *fgSocketHandler; // Input socket handler

   TXSocketHandler(TFileHandler *h, TSocket *s) :
                   TFileHandler(TXSocket::fgPipe.GetRead(), 1)
                   { fHandler = h; fInputSock = s; }
public:
   virtual ~TXSocketHandler() { }

   Bool_t  Notify();
   Bool_t  ReadNotify() { return Notify(); }

   static TXSocketHandler *GetSocketHandler(TFileHandler *h = 0, TSocket *s = 0);

   ClassDef(TXSocketHandler, 0) //Input handler class for xproofd sockets
};

#endif
