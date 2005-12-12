/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXUnixSocket
#define ROOT_TXUnixSocket

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXUnixSocket                                                         //
//                                                                      //
// Authors: G. Ganis, CERN, 2005                                        //
//                                                                      //
// Implementation of TXSocket using PF_UNIX sockets.                    //
// Used for the internal connection between coordinator and proofserv.  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TXSocket
#include "TXSocket.h"
#endif
#include <list>

class TXUnixSocket  : public TXSocket {

friend class TXProofServ;

public:

   TXUnixSocket(const char *u, Int_t psid = -1, Char_t ver = -1);
   virtual ~TXUnixSocket() { fSessionID = -1; }

   Int_t               GetClientID()
                      { return (fClientIDs.size() > 0) ? fClientIDs.front() : -1; }
   Int_t               GetClientIDSize() { return fClientIDs.size(); }

   void                RemoveClientID() { if (fClientIDs.size() > 1)
                                              fClientIDs.pop_front(); }
   void                SetClientID(Int_t cid) { fClientIDs.push_front(cid); }

private:

   std::list<Int_t>    fClientIDs; 

   ClassDef(TXUnixSocket, 0) //Connection class for Xrd PROOF using UNIX sockets
};

#endif
