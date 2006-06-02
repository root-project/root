// @(#)root/base:$Name:  $:$Id: TVirtualProof.h,v 1.31 2006/05/26 15:13:01 rdm Exp $
// Author: Fons Rademakers   16/09/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualProof
#define ROOT_TVirtualProof


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualProof                                                        //
//                                                                      //
// Abstract interface to the Parallel ROOT Facility, PROOF.             //
// For more information on PROOF see the TProof class.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualProofMgr
#include "TVirtualProofMgr.h"
#endif
#ifndef ROOT_TQObject
#include "TQObject.h"
#endif

class TList;
class TDSet;
class TEventList;
class TTree;
class TDSet;
class TDrawFeedback;
class TChain;
class TQueryResult;

// Global object with default PROOF session
class TVirtualProof;
R__EXTERN TVirtualProof *gProof;

// Special type for the hook to the TProof constructor, needed to avoid
// using the plugin manager
typedef TVirtualProof *(*TProof_t)(const char *, const char *, const char *,
                                   Int_t, const char *);

class TVirtualProof : public TNamed, public TQObject {

public:
   // PROOF status bits
   enum EStatusBits {
      kUsingSessionGui = BIT(14)
   };
   enum EQueryMode {
      kSync = 0,
      kAsync = 1
   };
   enum EUploadOpt {
      kAppend             = 0x1,
      kOverwriteDataSet   = 0x2,
      kNoOverwriteDataSet = 0x4,
      kOverwriteAllFiles  = 0x8,
      kOverwriteNoFiles   = 0x10,
      kAskUser            = 0x0
   };
   enum EUploadDataSetAnswer {
      kError = -1,
      kDataSetExists = -2
   };
   enum EUploadPackageOpt {
      kUntar             = 0x0,  //Untar over existing dir [default]
      kRemoveOld         = 0x1   //Remove existing dir with same name
   };

private:
   static TProof_t              fgProofHook; // Hook to TProof constructor

protected:
   TVirtualProofMgr::EServType  fServType;  // Type of server: proofd, XrdProofd
   TVirtualProofMgr            *fManager;   // Manager to which this session belongs (if any)
   EQueryMode                   fQueryMode; // default query mode

   TVirtualProof() : fServType(TVirtualProofMgr::kXProofd), fManager(0), fQueryMode(kSync) { }

public:
   TVirtualProof(const char * /*masterurl*/, const char * /*conffile*/ = 0,
                 const char * /*confdir*/ = 0, Int_t /*loglevel*/ = 0)
      : fServType(TVirtualProofMgr::kXProofd), fManager(0), fQueryMode(kSync) { }
   virtual ~TVirtualProof() { Emit("~TVirtualProof()"); }

   virtual void        cd(Int_t = -1) { gProof = this; }

   virtual void        SetAlias(const char *alias="") { TNamed::SetTitle(alias); }

   virtual Int_t       Ping() = 0;
   virtual Int_t       Exec(const char *cmd) = 0;
   virtual Int_t       Process(TDSet *set, const char *selector,
                               Option_t *option = "",
                               Long64_t nentries = -1,
                               Long64_t firstentry = 0,
                               TEventList *evl = 0) = 0;
   virtual Int_t       DrawSelect(TDSet *set, const char *varexp,
                                  const char *selection = "",
                                  Option_t *option = "",
                                  Long64_t nentries = -1,
                                  Long64_t firstentry = 0) = 0;
   virtual Int_t       Archive(Int_t query, const char *url) = 0;
   virtual Int_t       Archive(const char *queryref, const char *url = 0) = 0;
   virtual Int_t       CleanupSession(const char *sessiontag) = 0;
   virtual Int_t       Finalize(Int_t qry = -1, Bool_t force = kFALSE) = 0;
   virtual Int_t       Finalize(const char *queryref, Bool_t force = kFALSE) = 0;
   virtual Int_t       Remove(Int_t query) = 0;
   virtual Int_t       Remove(const char *queryref, Bool_t all = kFALSE) = 0;
   virtual Int_t       Retrieve(Int_t query, const char *path = 0) = 0;
   virtual Int_t       Retrieve(const char *queryref, const char *path = 0) = 0;

   virtual void        StopProcess(Bool_t abort) = 0;
   virtual void        AddInput(TObject *obj) = 0;
   virtual void        ClearInput() = 0;
   virtual TObject    *GetOutput(const char *name) = 0;
   virtual TList      *GetOutputList() = 0;

   virtual Int_t       SetParallel(Int_t nodes = 99999) = 0;
   virtual void        SetLogLevel(Int_t level, UInt_t mask = 0xFFFFFFFF) = 0;

   virtual void        Close(Option_t *option="") = 0;
   virtual void        Print(Option_t *option="") const = 0;

   //-- cache and package management
   virtual void        ShowCache(Bool_t all = kFALSE) = 0;
   virtual void        ClearCache() = 0;
   virtual TList      *GetListOfPackages() = 0;
   virtual TList      *GetListOfEnabledPackages() = 0;
   virtual void        ShowPackages(Bool_t all = kFALSE) = 0;
   virtual void        ShowEnabledPackages(Bool_t all = kFALSE) = 0;
   virtual Int_t       ClearPackages() = 0;
   virtual Int_t       ClearPackage(const char *package) = 0;
   virtual Int_t       EnablePackage(const char *package) = 0;
   virtual Int_t       UploadPackage(const char *par, EUploadPackageOpt opt = kUntar) = 0;

   virtual Int_t       AddDynamicPath(const char *libpath) = 0;
   virtual Int_t       AddIncludePath(const char *incpath) = 0;
   virtual Int_t       RemoveDynamicPath(const char *libpath) = 0;
   virtual Int_t       RemoveIncludePath(const char *incpath) = 0;

   //-- dataset management
   virtual Int_t       UploadDataSet(const char *dataset,
                                     const char *files,
                                     const char *dest,
                                     Int_t opt = kAskUser,
                                     TList *skippedFiles = 0) = 0;

   virtual Int_t       UploadDataSetFromFile(const char *file,
                                             const char *dest,
                                             const char *dataset,
                                             Int_t opt = kAskUser) = 0;
   virtual TList      *GetDataSets() = 0;
   virtual void        ShowDataSets() = 0;
   virtual void        ShowDataSet(const char *dataset) = 0;
   virtual Int_t       RemoveDataSet(const char *dateset) = 0;
   virtual Int_t       VerifyDataSet(const char *dataset) = 0;
   virtual TList      *GetDataSet(const char *dataset) = 0;

   virtual const char *GetMaster() const = 0;
   virtual const char *GetConfDir() const = 0;
   virtual const char *GetConfFile() const = 0;
   virtual const char *GetUser() const = 0;
   virtual const char *GetWorkDir() const = 0;
   virtual const char *GetSessionTag() const { return GetName(); }
   virtual const char *GetImage() const = 0;
   virtual const char *GetUrl() = 0;
   virtual Int_t       GetPort() const = 0;
   virtual Int_t       GetRemoteProtocol() const = 0;
   virtual Int_t       GetClientProtocol() const = 0;
   virtual Int_t       GetStatus() const = 0;
   virtual Int_t       GetLogLevel() const = 0;
   virtual Int_t       GetParallel() const = 0;
   virtual Int_t       GetSessionID() const { return -1; }
   virtual TList      *GetSlaveInfo() = 0;

   virtual EQueryMode  GetQueryMode() const { return fQueryMode; }
   virtual void        SetQueryType(EQueryMode mode) { fQueryMode = mode; }

   virtual TVirtualProofMgr::EServType   GetServType() const { return fServType; }

   virtual Long64_t    GetBytesRead() const = 0;
   virtual Float_t     GetRealTime() const = 0;
   virtual Float_t     GetCpuTime() const = 0;

   virtual Bool_t      IsProofd() const { return (fServType == TVirtualProofMgr::kProofd); }
   virtual Bool_t      IsMaster() const = 0;
   virtual Bool_t      IsValid() const = 0;
   virtual Bool_t      IsParallel() const = 0;
   virtual Bool_t      IsDataReady(Long64_t &totalbytes, Long64_t &bytesready) = 0;
   virtual Bool_t      IsIdle() const = 0;

   virtual void        AddFeedback(const char *name) = 0;
   virtual void        RemoveFeedback(const char *name) = 0;
   virtual void        ClearFeedback() = 0;
   virtual void        ShowFeedback() const = 0;
   virtual TList      *GetFeedbackList() const = 0;

   virtual TList      *GetListOfQueries(Option_t *opt = "") = 0;
   virtual Int_t       GetNumberOfQueries() = 0;
   virtual Int_t       GetNumberOfDrawQueries() = 0;
   virtual TList      *GetQueryResults() = 0;
   virtual TQueryResult *GetQueryResult(const char *ref) = 0;
   virtual void        GetMaxQueries() = 0;
   virtual void        SetMaxDrawQueries(Int_t max) = 0;
   virtual void        ShowQueries(Option_t *opt = "") = 0;

   virtual void        SetActive(Bool_t active = kTRUE) = 0;

   virtual void        LogMessage(const char *msg, Bool_t all) = 0; //*SIGNAL*
   virtual void        Progress(Long64_t total, Long64_t processed) = 0; //*SIGNAL*
   virtual void        Feedback(TList *objs) = 0; //*SIGNAL*
   virtual void        QueryResultReady(const char *ref) = 0; //*SIGNAL*
   virtual void        ResetProgressDialog(const char *sel, Int_t sz,
                                   Long64_t fst, Long64_t ent) = 0; //*SIGNAL*
   virtual void        StartupMessage(const char *msg, Bool_t status,
                                      Int_t done, Int_t total) = 0; //*SIGNAL*

   virtual void        GetLog(Int_t start = -1, Int_t end = -1) = 0;
   virtual void        PutLog(TQueryResult *qr) = 0;
   virtual void        ShowLog(Int_t qry = -1) = 0;
   virtual void        ShowLog(const char *queryref) = 0;
   virtual Bool_t      SendingLogToWindow() const = 0;
   virtual void        SendLogToWindow(Bool_t mode) = 0;

   virtual void        ResetProgressDialogStatus() = 0;

   virtual TTree      *GetTreeHeader(TDSet* dset) = 0;
   virtual TList      *GetOutputNames() = 0;

   virtual void        AddChain(TChain* chain) = 0;
   virtual void        RemoveChain(TChain* chain) = 0;

   virtual TDrawFeedback *CreateDrawFeedback() = 0;
   virtual void        SetDrawFeedbackOption(TDrawFeedback *f, Option_t *opt) = 0;
   virtual void        DeleteDrawFeedback(TDrawFeedback *f) = 0;

   virtual void        Detach(Option_t * = "") = 0;

   virtual TVirtualProofMgr *GetManager() { return fManager; }
   virtual void        SetManager(TVirtualProofMgr *mgr) { fManager = mgr; }

   static void         SetTProofHook(TProof_t proofhook);
   static TProof_t     GetTProofHook();

   static TVirtualProof *Open(const char *cluster = 0, const char *conffile = 0,
                              const char *confdir = 0, Int_t loglevel = 0);

   ClassDef(TVirtualProof,0)  // Abstract PROOF interface
};
#endif
