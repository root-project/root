// @(#)root/proofx:$Id$
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
// TXUnixSocket                                                         //
//                                                                      //
// Implementation of TXSocket using PF_UNIX sockets.                    //
// Used for the internal connection between coordinator and proofserv.  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "XpdSysPthread.h"

#include "TXUnixSocket.h"
#include "TEnv.h"
#include "XrdProofPhyConn.h"

ClassImp(TXUnixSocket)

//_____________________________________________________________________________
TXUnixSocket::TXUnixSocket(const char *url,
                           Int_t psid, Char_t capver, TXHandler *handler, int fd)
             : TXSocket(0,'i',psid,capver,0,-1,handler)
{
   // Constructor

   // Initialization
   if (url) {

      // Create connection
      fConn = new XrdProofPhyConn(url, psid, capver, this, 0, fd);
      if (!(fConn->IsValid())) {
         Error("TXUnixSocket", "severe error occurred while opening a connection"
                               " to server [%s]", fUrl.Data());
         return;
      }

      // Fill some info
      fUser = fConn->fUser.c_str();
      fHost = fConn->fHost.c_str();
      fPort = fConn->fPort;
      fXrdProofdVersion = fConn->fRemoteProtocol;
      fRemoteProtocol = fConn->fRemoteProtocol;

      // Save also updated url
      TSocket::fUrl = fConn->fUrl.GetUrl().c_str();

      // This is needed for the reader thread to signal an interrupt
      fPid = gSystem->GetPid();
   }
}

//______________________________________________________________________________
Int_t TXUnixSocket::Reconnect()
{
   // Try reconnection after failure

   if (gDebug > 0) {
      Info("Reconnect", "%p: %p: %d: trying to reconnect on %s", this,
                        fConn, (fConn ? fConn->IsValid() : 0), fUrl.Data());
   }

   Int_t tryreconnect = gEnv->GetValue("TXSocket.Reconnect", 0);
   if (tryreconnect == 0 || fXrdProofdVersion < 1005) {
      if (tryreconnect == 0)
         Info("Reconnect","%p: reconnection attempts explicitly disabled!", this);
      else
         Info("Reconnect","%p: server does not support reconnections (protocol: %d < 1005)",
                          this, fXrdProofdVersion);
      return -1;
   }

   if (fConn && !fConn->IsValid()) {

      // Block any other attempt to use this connection
      XrdSysMutexHelper l(fConn->fMutex);

      fConn->Close();
      int maxtry, timewait;
      XrdProofConn::GetRetryParam(maxtry, timewait);
      XrdProofConn::SetRetryParam(300, 1);
      fConn->Connect();
      XrdProofConn::SetRetryParam();
   }

   if (gDebug > 0) {
      Info("Reconnect", "%p: %p: attempt %s", this, fConn,
                        ((fConn && fConn->IsValid()) ? "succeeded!" : "failed"));
   }

   // Done
   return ((fConn && fConn->IsValid()) ? 0 : -1);
}
