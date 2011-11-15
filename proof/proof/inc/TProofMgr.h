// @(#)root/proof:$Id$
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

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TUrl
#include "TUrl.h"
#endif
#ifndef ROOT_TSystem
#include "TSystem.h"
#endif

class TList;
class TObjString;
class TProof;
class TProofDesc;
class TProofLog;
class TProofMgr;
class TSignalHandler;

typedef TProofMgr *(*TProofMgr_t)(const char *, Int_t, const char *);

class TProofMgr : public TNamed {

public:
   enum EServType { kProofd = 0, kXProofd = 1, kProofLite = 2 };

private:
   TProofMgr(const TProofMgr&); // Not implemented
   TProofMgr& operator=(const TProofMgr&); // Not implemented

   static TProofMgr_t fgTXProofMgrHook; // Constructor hooks for TXProofMgr
   static TProofMgr_t GetXProofMgrHook();

protected:
   Int_t          fRemoteProtocol; // Protocol number run by the daemon server
   EServType      fServType;       // Type of server: old-proofd, XrdProofd
   TList         *fSessions;       // PROOF session managed by this server
   TUrl           fUrl;            // Server URL

   TSignalHandler *fIntHandler;    // Interrupt signal handler (ctrl-c)

   static TList   fgListOfManagers; // Sub-list of TROOT::ListOfProofs for managers

   TProofMgr() : fRemoteProtocol(-1),
                        fServType(kXProofd), fSessions(0), fUrl(), fIntHandler(0) { }

public:
   TProofMgr(const char *url, Int_t loglevel = -1, const char *alias = "");
   virtual ~TProofMgr();

   virtual Bool_t      IsLite() const { return (fServType == kProofLite); }
   virtual Bool_t      IsProofd() const { return (fServType == kProofd); }
   virtual Bool_t      IsValid() const { return kTRUE; }
   virtual void        SetInvalid() { }
           void        Close() { SetInvalid(); }

   virtual TProof     *AttachSession(Int_t, Bool_t = kFALSE);
   virtual TProof     *AttachSession(TProofDesc *, Bool_t = kFALSE);
   virtual TProof     *CreateSession(const char * = 0, const char * = 0, Int_t = -1);
   virtual void        DetachSession(Int_t, Option_t * = "");
   virtual void        DetachSession(TProof *, Option_t * = "");
   virtual void        DiscardSession(TProof *p);
   virtual TProofDesc *GetProofDesc(Int_t id);
   virtual TProofDesc *GetProofDesc(TProof *p);
   virtual Int_t       GetRemoteProtocol() const { return fRemoteProtocol; }
   virtual TProofLog  *GetSessionLogs(Int_t = 0, const char * = 0,
                                      const char * = "-v \"| SvcMsg\"", Bool_t = kFALSE)
                                      { return (TProofLog *)0; }
   virtual const char *GetUrl() { return fUrl.GetUrl(); }
   virtual Bool_t      MatchUrl(const char *url);
   virtual TList      *QuerySessions(Option_t *opt = "S");
   virtual TObjString *ReadBuffer(const char *, Long64_t, Int_t)
                                        { return (TObjString *)0; }
   virtual TObjString *ReadBuffer(const char *, const char *)
                                        { return (TObjString *)0; }
   virtual Int_t       Reset(Bool_t hard = kFALSE, const char *usr = 0);
   virtual void        ShowWorkers();
   virtual Int_t       SendMsgToUsers(const char *, const char * = 0);
   virtual void        SetAlias(const char *alias="") { TNamed::SetTitle(alias); }
   virtual void        SetROOTVersion(const char *) { }
   virtual void        ShowROOTVersions() { }
   virtual void        ShutdownSession(Int_t id) { DetachSession(id,"S"); }
   virtual void        ShutdownSession(TProof *p) { DetachSession(p,"S"); }

   // Remote file system actions
   virtual Int_t       Cp(const char *, const char * = 0, const char * = 0) { return -1; }
   virtual void        Find(const char * = "~/", const char * = 0, const char * = 0) { }
   virtual void        Grep(const char *, const char * = 0, const char * = 0) { }
   virtual void        Ls(const char * = "~/", const char * = 0, const char * = 0) { }
   virtual void        More(const char *, const char * = 0, const char * = 0) { }
   virtual Int_t       Rm(const char *, const char * = 0, const char * = 0) { return -1; }
   virtual void        Tail(const char *, const char * = 0, const char * = 0) { }
   virtual Int_t       Md5sum(const char *, TString &, const char * = 0) { return -1; }
   virtual Int_t       Stat(const char *, FileStat_t &, const char * = 0) { return -1; }

   virtual Int_t       GetFile(const char *, const char *, const char * = 0) { return -1; }
   virtual Int_t       PutFile(const char *, const char *, const char * = 0) { return -1; }

   static TList       *GetListOfManagers();

   static void         SetTXProofMgrHook(TProofMgr_t pmh);

   static TProofMgr   *Create(const char *url, Int_t loglevel = -1,
                              const char *alias = 0, Bool_t xpd = kTRUE);
   static Int_t        Ping(const char *url, Bool_t checkxrd = kFALSE);

   ClassDef(TProofMgr,0)  // Abstract PROOF manager interface
};

//
// Metaclass describing the essentials of a PROOF session
//
class TProofDesc : public TNamed {
public:
   enum EStatus { kUnknown = -1, kIdle = 0, kRunning =1, kShutdown = 2};

private:
   TProofDesc(const TProofDesc&); // Not implemented
   TProofDesc& operator=(const TProofDesc&); // Not implemented

   Int_t          fLocalId;  // ID in the local list
   Int_t          fStatus;   // Session status (see EStatus)
   TProof        *fProof;    // Related instance of TProof
   Int_t          fRemoteId; // Remote ID assigned by the coordinator to the proofserv
   TString        fUrl;      // Url of the connection

public:
   TProofDesc(const char *tag = 0, const char *alias = 0, const char *url = 0,
                     Int_t id = -1, Int_t remid = -1, Int_t status = kIdle, TProof *p = 0)
                    : TNamed(tag, alias),
     fLocalId(id), fStatus(0), fProof(p), fRemoteId(remid), fUrl(url) { SetStatus(status); }
   virtual ~TProofDesc() { }

   Int_t          GetLocalId() const { return fLocalId; }
   TProof        *GetProof() const { return fProof; }
   Int_t          GetRemoteId() const { return fRemoteId; }
   Int_t          GetStatus() const { return fStatus; }
   const char    *GetUrl() const { return fUrl; }

   Bool_t         IsIdle() const { return (fStatus == kIdle) ? kTRUE : kFALSE; }
   Bool_t         IsRunning() const { return (fStatus == kRunning) ? kTRUE : kFALSE; }
   Bool_t         IsShuttingDown() const { return (fStatus == kShutdown) ? kTRUE : kFALSE; }

   Bool_t         MatchId(Int_t id) const { return (fLocalId == id); }

   void           Print(Option_t *opt = "") const;

   void           SetStatus(Int_t st) { fStatus = (st < kIdle || st > kShutdown) ? -1 : st; }

   void           SetProof(TProof *p) { fProof = p; }
   void           SetRemoteId(Int_t id) { fRemoteId = id; }

   ClassDef(TProofDesc,1)  // Small class describing a proof session
};

#endif
