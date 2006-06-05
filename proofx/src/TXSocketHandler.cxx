// @(#)root/proofx:$Name:  $:$Id: TXSocketHandler.cxx,v 1.4 2006/04/19 10:57:44 rdm Exp $
// Author: Gerardo Ganis  12/12/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXSocketHandler                                                      //
//                                                                      //
// Input handler for xproofd sockets. These sockets cannot be directly  //
// monitored on their descriptor, because the reading activity goes via //
// the reader thread. This class allows to handle this problem.         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMonitor.h"
#include "TProof.h"
#include "TSlave.h"
#include "TXSocketHandler.h"
#include "TXHandler.h"


ClassImp(TXSocketHandler)

// Unique instance of the socket input handler
TXSocketHandler *TXSocketHandler::fgSocketHandler = 0;

//______________________________________________________________________________
Bool_t TXSocketHandler::Notify()
{
   // Set readiness on the monitor

   if (gDebug > 2)
      TXSocket::DumpReadySock();

   // Get the socket
   TXSocket *s = 0;
   {  R__LOCKGUARD(&TXSocket::fgReadyMtx);
      s = (TXSocket *) TXSocket::fgReadySock.Last();
      if (gDebug > 2)
         Info("Notify", "ready socket %p (input socket: %p)", s, fInputSock);
   }

   // If empty, nothing to do
   if (!s) {
      Warning("Notify","socket-ready list is empty!");
      return kTRUE;
   }

   // Handle this input
   s->fHandler->HandleInput();

   // We are done
   return kTRUE;
}

//_______________________________________________________________________
TXSocketHandler *TXSocketHandler::GetSocketHandler(TFileHandler *h, TSocket *s)
{
   // Get an instance of the input socket handler with 'h' as handler,
   // connected to socket 's'.
   // Create the instance, if not already existing

   if (!fgSocketHandler)
      fgSocketHandler = new TXSocketHandler(h, s);
   else
      if (h && s)
         fgSocketHandler->SetHandler(h, s);

   return fgSocketHandler;
}
