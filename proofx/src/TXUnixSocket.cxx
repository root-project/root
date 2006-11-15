// @(#)root/proofx:$Name:  $:$Id: TXUnixSocket.cxx,v 1.5 2006/07/31 10:27:48 rdm Exp $
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

#include "TXUnixSocket.h"
#include "XrdProofPhyConn.h"

ClassImp(TXUnixSocket)

//_____________________________________________________________________________
TXUnixSocket::TXUnixSocket(const char *url,
                           Int_t psid, Char_t capver, TXHandler *handler)
             : TXSocket(0,'i',psid,capver,0,-1,handler)
{
   // Constructor

   // Initialization
   if (url) {

      // Create connection
      fConn = new XrdProofPhyConn(url, psid, capver, this);
      if (!(fConn->IsValid())) {
         Error("TXUnixSocket", "severe error occurred while opening a connection"
                               " to server [%s]", fUrl.Data());
         return;
      }

      // Fill some info
      fUser = fConn->fUser.c_str();
      fHost = fConn->fHost.c_str();
      fPort = fConn->fPort;

      // Save also updated url
      TSocket::fUrl = fConn->fUrl.GetUrl().c_str();

      // This is needed for the reader thread to signal an interrupt
      fPid = gSystem->GetPid();
   }
}
