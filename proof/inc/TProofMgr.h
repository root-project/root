// @(#)root/proof:$Name:  $:$Id: TProofMgr.h,v 1.1 2005/12/10 16:51:57 rdm Exp $
// Author: G. Ganis, Nov 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofMgr
#define ROOT_TProofMgr


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofMgr                                                            //
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

class TProofMgr : public TVirtualProofMgr {

public:
   TProofMgr(const char *url, Int_t loglevel = -1, const char *alias = "");
   virtual ~TProofMgr() { }

   Bool_t      IsValid() const { return kTRUE; }

   TVirtualProof *AttachSession(Int_t id, Bool_t gui = kFALSE);
   void           DetachSession(Int_t, Option_t * = "");
   TList         *QuerySessions(Option_t *opt = "S");
   Int_t          Reset(const char *usr = 0);

   ClassDef(TProofMgr,0)  // PROOF session manager
};

#endif
