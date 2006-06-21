// @(#)root/base:$Name:  $:$Id: TVirtualProofMgr.h,v 1.3 2006/06/02 15:14:35 rdm Exp $
// Author: G. Ganis, Nov 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualProofMgr
#define ROOT_TVirtualProofMgr


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualProofMgr                                                     //
//                                                                      //
// Abstract interface to the manager for PROOF sessions.                //
// The PROOF manager interacts with the PROOF server coordinator to     //
// create or destroy a PROOF session, attach to or detach from          //
// existing one, and to monitor any client activity on the cluster.     //
// At most one manager instance per server is allowed.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TUrl
#include "TUrl.h"
#endif

class TList;
class TVirtualProof;
class TVirtualProofDesc;
class TVirtualProofMgr;

typedef TVirtualProofMgr
     *(*TVirtualProofMgr_t)(const char *, Int_t, const char *);

class TVirtualProofMgr : public TNamed {

public:
   enum EServType { kProofd = 0, kXProofd = 1 };

private:
   static TVirtualProofMgr_t fgTProofMgrHook[2]; // Constructor hooks for TProofMgr
   static TVirtualProofMgr_t GetProofMgrHook(const char *type);

protected:
   Int_t          fRemoteProtocol; // Protocol number run by the daemon server
   EServType      fServType;       // Type of server: old-proofd, XrdProofd
   TList         *fSessions;       // PROOF session managed by this server
   TUrl           fUrl;            // Server URL

   static TList   fgListOfManagers; // Sub-list of TROOT::ListOfProofs for managers

   TVirtualProofMgr() : fRemoteProtocol(-1),
                        fServType(kXProofd), fSessions(0), fUrl() { }

public:
   TVirtualProofMgr(const char *url, Int_t /*loglevel*/ = 0, const char * /*alias*/ = 0);
   virtual ~TVirtualProofMgr();

   virtual Bool_t      IsProofd() const { return (fServType == kProofd); }
   virtual Bool_t      IsValid() const = 0;

   virtual TVirtualProof *AttachSession(Int_t, Bool_t = kFALSE) = 0;
   virtual TVirtualProof *CreateSession(const char * = 0, const char * = 0, Int_t = -1);
   virtual void        DetachSession(Int_t, Option_t * = "") = 0;
   virtual TVirtualProofDesc *GetProofDesc(Int_t id);
   virtual Int_t       GetRemoteProtocol() const { return fRemoteProtocol; }
   virtual const char *GetUrl() { return fUrl.GetUrl(); }
   virtual Bool_t      MatchUrl(const char *url);
   virtual TList      *QuerySessions(Option_t *opt = "S") = 0;
   virtual Int_t       Reset(const char *usr = 0) = 0;
   virtual void        ShowWorkers();
   virtual void        SetAlias(const char *alias="") { TNamed::SetTitle(alias); }
   virtual void        ShutdownSession(Int_t id) { DetachSession(id,"S"); }
   virtual void        ShutdownSession(TVirtualProof *p);

   static TList       *GetListOfManagers();

   static void SetTProofMgrHook(TVirtualProofMgr_t pmh, const char *type = 0);

   static TVirtualProofMgr *Create(const char *url, Int_t loglevel = -1,
                                   const char *alias = 0, Bool_t xpd = kTRUE);

   ClassDef(TVirtualProofMgr,0)  // Abstract PROOF manager interface
};


//
// Metaclass describing the essentials of a PROOF session
//
class TVirtualProofDesc : public TNamed {
public:
   enum EStatus { kUnknown = -1, kIdle = 0, kRunning =1, kShutdown = 2};

private:
   Int_t          fLocalId;  // ID in the local list
   Int_t          fStatus;   // Session status (see EStatus)
   TVirtualProof *fProof;    // Related instance of TVirtualProof
   Int_t          fRemoteId; // Remote ID assigned by the coordinator to the proofserv
   TString        fUrl;      // Url of the connection

public:
   TVirtualProofDesc(const char *tag = 0, const char *alias = 0, const char *url = 0,
                     Int_t id = -1, Int_t remid = -1, Int_t status = kIdle, TVirtualProof *p = 0)
                    : TNamed(tag, alias),
                      fLocalId(id), fProof(p), fRemoteId(remid), fUrl(url) { SetStatus(status); }
   virtual ~TVirtualProofDesc() { }

   Int_t          GetLocalId() const { return fLocalId; }
   TVirtualProof *GetProof() const { return fProof; }
   Int_t          GetRemoteId() const { return fRemoteId; }
   Int_t          GetStatus() const { return fStatus; }
   const char    *GetUrl() const { return fUrl; }

   Bool_t         IsIdle() const { return (fStatus == kIdle) ? kTRUE : kFALSE; }
   Bool_t         IsRunning() const { return (fStatus == kRunning) ? kTRUE : kFALSE; }
   Bool_t         IsShuttingDown() const { return (fStatus == kShutdown) ? kTRUE : kFALSE; }

   Bool_t         MatchId(Int_t id) const { return (fLocalId == id); }

   void           Print(Option_t *opt = "") const;

   void           SetStatus(Int_t st) { fStatus = (st < kIdle || st > kShutdown) ? -1 : st; }

   void           SetProof(TVirtualProof *p) { fProof = p; }
   void           SetRemoteId(Int_t id) { fRemoteId = id; }

   ClassDef(TVirtualProofDesc,2)  // Abstract description of a proof session
};

#endif
