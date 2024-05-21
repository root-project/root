// @(#)root/proofx:$Id$
// Author: Gerardo Ganis  12/12/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TXSocketHandler
\ingroup proofx

Input handler for XProofD sockets. These sockets cannot be directly
monitored on their descriptor, because the reading activity goes via
the internal reader thread. This class allows to handle the related issue.

*/

#include "TMonitor.h"
#include "TProof.h"
#include "TSlave.h"
#include "TXSocketHandler.h"
#include "TXHandler.h"
#include "TList.h"


ClassImp(TXSocketHandler);

// Unique instance of the socket input handler
TXSocketHandler *TXSocketHandler::fgSocketHandler = 0;

////////////////////////////////////////////////////////////////////////////////
/// Set readiness on the monitor

Bool_t TXSocketHandler::Notify()
{
   if (gDebug > 2)
      TXSocket::fgPipe.DumpReadySock();

   // Get the socket
   TXSocket *s = TXSocket::fgPipe.GetLastReady();
   if (gDebug > 2)
      Info("Notify", "ready socket %p (%s) (input socket: %p) (fFileNum: %d)",
                     s, (s ? s->GetTitle() : "***undef***"), fInputSock, fFileNum);

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

////////////////////////////////////////////////////////////////////////////////
/// Get an instance of the input socket handler with 'h' as handler,
/// connected to socket 's'.
/// Create the instance, if not already existing

TXSocketHandler *TXSocketHandler::GetSocketHandler(TFileHandler *h, TSocket *s)
{
   if (!fgSocketHandler)
      fgSocketHandler = new TXSocketHandler(h, s);
   else
      if (h && s)
         fgSocketHandler->SetHandler(h, s);

   return fgSocketHandler;
}
