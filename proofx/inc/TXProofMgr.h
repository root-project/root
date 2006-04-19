// @(#)root/proofx:$Name:  $:$Id: TXProofMgr.h,v 1.2 2006/02/26 16:09:57 rdm Exp $
// Author: G. Ganis, Nov 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXProofMgr
#define ROOT_TXProofMgr


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXProofMgr                                                           //
//                                                                      //
// The PROOF manager interacts with the PROOF server coordinator to     //
// create or destroy a PROOF session, attach to or detach from          //
// existing one, and to monitor any client activity on the cluster.     //
// At most one manager instance per server is allowed.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualProofMgr
#include "TVirtualProofMgr.h"
#endif
#ifndef ROOT_TUrl
#include "TUrl.h"
#endif
#ifndef ROOT_TXHandler
#include "TXHandler.h"
#endif

class TXSocket;

class TXProofMgr : public TVirtualProofMgr, public TXHandler {

private:

   TXSocket   *fSocket; // Connection to XRD

   Int_t Init(Int_t loglevel = -1);

public:
   TXProofMgr(const char *url, Int_t loglevel = -1, const char *alias = "");
   virtual ~TXProofMgr();

   Bool_t      HandleError();

   Bool_t      IsValid() const { return fSocket; }

   TVirtualProof *AttachSession(Int_t id, Bool_t gui = kFALSE);
   void        DetachSession(Int_t, Option_t * = "");
   Bool_t      MatchUrl(const char *url);
   TList      *QuerySessions(Option_t *opt = "S");
   void        ShowWorkers();

   ClassDef(TXProofMgr,0)  // XrdProofd PROOF manager interface
};

#endif
