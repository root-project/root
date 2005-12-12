/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXSocketHandler                                                      //
//                                                                      //
// Authors: G. Ganis, CERN, 2005                                        //
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

 // Unique instance of the socket input handler
TXSocketHandler *TXSocketHandler::fgSocketHandler = 0;

//______________________________________________________________________________
Bool_t TXSocketHandler::Notify()
{
   // Set readiness on the monitor

   if (gDebug > 2)
      TXSocket::DumpReadySock();

   // Get the socket
   TSocket *s = 0;
   {  R__LOCKGUARD(&TXSocket::fgReadyMtx);
      s = (TSocket *) TXSocket::fgReadySock.Last();
      if (gDebug > 2)
         Info("Notify", "ready socket %p (input socket: %p)", s, fInputSock);
   }

   // If empty, nothing to do
   if (!s) {
      Warning("Notify","socket-ready list is empty!");
      return kTRUE;
   }

   Bool_t notdone = kTRUE;

   // Check if it is the input handler first
   if (s == fInputSock) {
      // Input handler in TXProofServ
      if (gDebug > 2)
         Info("Notify","calling input handler for socket %p",s);
      if (fHandler)
         fHandler->Notify();
      notdone = kFALSE;
   }

   // If not, check if the socket belongs to a TProof instance
   if (notdone) {
      // Get the reference proof, if any
      TProof *proof = (((TXSocket *)s)->fReference) ? 
                      dynamic_cast<TProof *>(((TXSocket *)s)->fReference) : 0; 
      if (proof) {

         // Attach to the monitor instance, if any
         TMonitor *mon =
            (proof && proof->fCurrentMonitor) ? proof->fCurrentMonitor : 0;

         if (gDebug > 2)
            Info("Notify","proof: %p, mon: %p", proof, mon);

         if (mon && mon->GetListOfActives()->FindObject(s)) {
            // Synchronous collection in TProof
            if (gDebug > 2)
               Info("Notify","posting monitor %p with socket %p", mon, s);
            mon->SetReady(s);
            notdone = kFALSE;
         } else if (proof->GetListOfSlaves()) {
            TIter nxsl(proof->GetListOfSlaves());
            TSlave *sl = 0;
            while ((sl = (TSlave *)nxsl())) {
               if (gDebug > 2)
                  Info("Notify","slave: %p, sock: %p", sl, sl->GetSocket());
               if (s == sl->GetSocket())
                   break;
            }
            if (sl) {
               // Asynchronous collection in TProof
               if (gDebug > 2)
                  Info("Notify","calling TProof::CollectInputFrom for socket %p",s);
               proof->CollectInputFrom(s);
               notdone = kFALSE;
            } else
               Info("Notify","socket %p not found in fSlaves list",s);
         }
      } else {
         Warning("Notify",
                 "reference to proof missing; socket: %p", s);
      }
   }

   if (notdone)
      Warning("Notify",
              "unassigned ready socket %p (input socket: %p)",s,fInputSock);

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
