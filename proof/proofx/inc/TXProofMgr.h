// @(#)root/proofx:$Id$
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

class TStopwatch;
class TXSocket;

class TXProofMgr : public TProofMgr, public TXHandler {

private:

   TXSocket   *fSocket; // Connection to XRD

   Int_t Init(Int_t loglevel = -1);

   void CpProgress(const char *pfx, Long64_t bytes,
                   Long64_t size, TStopwatch *watch, Bool_t cr = kFALSE);
   TObjString *Exec(Int_t action,
                    const char *what, const char *how, const char *where);

public:
   TXProofMgr(const char *url, Int_t loglevel = -1, const char *alias = "");
   virtual ~TXProofMgr();

   Bool_t      HandleInput(const void *);
   Bool_t      HandleError(const void *in = 0);

   Bool_t      IsValid() const { return fSocket; }
   void        SetInvalid();

   TProof     *AttachSession(Int_t id, Bool_t gui = kFALSE)
                      { return TProofMgr::AttachSession(id, gui); }
   TProof     *AttachSession(TProofDesc *d, Bool_t gui = kFALSE);
   void        DetachSession(Int_t, Option_t * = "");
   void        DetachSession(TProof *, Option_t * = "");
   TProofLog  *GetSessionLogs(Int_t ridx = 0, const char *stag = 0,
                              const char *pattern = "-v \"| SvcMsg\"",
                              Bool_t rescan = kFALSE);
   Bool_t      MatchUrl(const char *url);
   void        ShowROOTVersions();
   TList      *QuerySessions(Option_t *opt = "S");
   TObjString *ReadBuffer(const char *file, Long64_t ofs, Int_t len);
   TObjString *ReadBuffer(const char *file, const char *pattern);
   Int_t       Reset(Bool_t hard = kFALSE, const char *usr = 0);
   Int_t       SendMsgToUsers(const char *msg, const char *usr = 0);
   void        SetROOTVersion(const char *tag);
   void        ShowWorkers();

   // Remote file system actions
   Int_t       Cp(const char *src, const char *dst = 0, const char *opts = 0);
   void        Find(const char *what = "~/", const char *how = "-type f", const char *where = 0);
   void        Grep(const char *what, const char *how = 0, const char *where = 0);
   void        Ls(const char *what = "~/", const char *how = 0, const char *where = 0);
   void        More(const char *what, const char *how = 0, const char *where = 0);
   Int_t       Rm(const char *what, const char *how = 0, const char *where = 0);
   void        Tail(const char *what, const char *how = 0, const char *where = 0);
   Int_t       Md5sum(const char *what, TString &sum, const char *where = 0);
   Int_t       Stat(const char *what, FileStat_t &st, const char *where = 0);

   Int_t       GetFile(const char *remote, const char *local, const char *opt = 0);
   Int_t       PutFile(const char *local, const char *remote, const char *opt = 0);

   ClassDef(TXProofMgr,0)  // XrdProofd PROOF manager interface
};

#endif
