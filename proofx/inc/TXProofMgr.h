// @(#)root/proofx:$Name:  $:$Id: TXProofMgr.h,v 1.7 2006/11/28 12:10:52 rdm Exp $
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

#ifndef ROOT_TProofMgr
#include "TProofMgr.h"
#endif
#ifndef ROOT_TUrl
#include "TUrl.h"
#endif
#ifndef ROOT_TXHandler
#include "TXHandler.h"
#endif

//
// XPROOF client version: increase whenever a non backward compatible
// change occur
//  ->1      first version being tested by ALICE
const Int_t kXPROOF_Protocol = 1; 

class TXSocket;

class TXProofMgr : public TProofMgr, public TXHandler {

private:

   TXSocket   *fSocket; // Connection to XRD

   Int_t Init(Int_t loglevel = -1);

public:
   TXProofMgr(const char *url, Int_t loglevel = -1, const char *alias = "");
   virtual ~TXProofMgr();

   Bool_t      HandleError(const void *in = 0);

   Bool_t      IsValid() const { return fSocket; }

   TProof     *AttachSession(Int_t id, Bool_t gui = kFALSE);
   void        DetachSession(Int_t, Option_t * = "");
   TProofLog  *GetSessionLogs(Int_t ridx = 0, const char *stag = 0);
   Bool_t      MatchUrl(const char *url);
   TList      *QuerySessions(Option_t *opt = "S");
   TObjString *ReadBuffer(const char *file, Long64_t ofs, Int_t len);
   Int_t       Reset(const char *usr = 0);
   void        ShowWorkers();

   ClassDef(TXProofMgr,0)  // XrdProofd PROOF manager interface
};

#endif
