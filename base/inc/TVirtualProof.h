// @(#)root/base:$Name:  $:$Id: TVirtualProof.h,v 1.18 2005/08/30 10:47:31 rdm Exp $
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

#ifndef ROOT_TObject
#include "TObject.h"
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


class TVirtualProof : public TObject, public TQObject {

public:
   enum EQueryMode { kSync = 0, kAsync = 1 };

protected:
   EQueryMode   fQueryMode;    // default query mode

   TVirtualProof() : fQueryMode(kSync) { }

public:
   TVirtualProof(const char * /*masterurl*/, const char * /*conffile*/ = 0,
                 const char * /*confdir*/ = 0, Int_t /*loglevel*/ = 0)
      : fQueryMode(kSync) { }
   virtual ~TVirtualProof() { Emit("~TVirtualProof()"); }

   virtual Int_t       Ping() = 0;
   virtual Int_t       Exec(const char *cmd) = 0;
   virtual Int_t       Process(TDSet *set, const char *selector,
                               Option_t *option = "",
                               Long64_t nentries = -1,
                               Long64_t firstentry = 0,
                               TEventList *evl = 0) = 0;
   virtual Int_t       DrawSelect(TDSet *set, const char *varexp,
                                  const char *selection,
                                  Option_t *option = "",
                                  Long64_t nentries = -1,
                                  Long64_t firstentry = 0) = 0;
   virtual Int_t       Finalize() = 0;
   virtual Int_t       Retrieve(Int_t query) = 0;
   virtual Int_t       Archive(Int_t query, const char *url) = 0;

   virtual void        StopProcess(Bool_t abort) = 0;
   virtual void        AddInput(TObject *obj) = 0;
   virtual void        ClearInput() = 0;
   virtual TObject    *GetOutput(const char *name) = 0;
   virtual TList      *GetOutputList() = 0;

   virtual Int_t       SetParallel(Int_t nodes = 99999) = 0;
   virtual void        SetLogLevel(Int_t level, UInt_t mask = 0xFFFFFFFF) = 0;

   virtual void        Close(Option_t *option="") = 0;
   virtual void        Print(Option_t *option="") const = 0;

   virtual void        ShowCache(Bool_t all = kFALSE) = 0;
   virtual void        ClearCache() = 0;
   virtual void        ShowPackages(Bool_t all = kFALSE) = 0;
   virtual void        ShowEnabledPackages(Bool_t all = kFALSE) = 0;
   virtual Int_t       ClearPackages() = 0;
   virtual Int_t       ClearPackage(const char *package) = 0;
   virtual Int_t       EnablePackage(const char *package) = 0;
   virtual Int_t       UploadPackage(const char *par) = 0;

   virtual const char *GetMaster() const = 0;
   virtual const char *GetConfDir() const = 0;
   virtual const char *GetConfFile() const = 0;
   virtual const char *GetUser() const = 0;
   virtual const char *GetWorkDir() const = 0;
   virtual const char *GetImage() const = 0;
   virtual Int_t       GetPort() const = 0;
   virtual Int_t       GetRemoteProtocol() const = 0;
   virtual Int_t       GetClientProtocol() const = 0;
   virtual Int_t       GetStatus() const = 0;
   virtual Int_t       GetLogLevel() const = 0;
   virtual Int_t       GetParallel() const = 0;
   virtual TList      *GetSlaveInfo() = 0;

   virtual EQueryMode  GetQueryMode() const { return fQueryMode; }
   virtual void        SetQueryType(EQueryMode mode) { fQueryMode = mode; }

   virtual Long64_t    GetBytesRead() const = 0;
   virtual Float_t     GetRealTime() const = 0;
   virtual Float_t     GetCpuTime() const = 0;

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

   virtual void        GetListOfQueries() = 0;
   virtual void        ShowQueries(Option_t *opt = "") = 0;

   virtual void        SetActive(Bool_t active = kTRUE) = 0;

   virtual void        LogMessage(const char *msg, Bool_t all) = 0; //*SIGNAL*
   virtual void        Progress(Long64_t total, Long64_t processed) = 0; //*SIGNAL*
   virtual void        Feedback(TList *objs) = 0; //*SIGNAL*
   virtual void        ResetProgressDialog(const char *sel, Int_t sz,
                                   Long64_t fst, Long64_t ent) = 0; //*SIGNAL*

   virtual void        GetLog(Int_t start = -1, Int_t end = -1) = 0;
   virtual void        ShowLog(Int_t qry = -1) = 0;
   virtual Bool_t      GetLogToWindow() const = 0;
   virtual void        SetLogToWindow(Bool_t mode) = 0;

   virtual void        ResetProgressDialogStatus() = 0;

   virtual TTree      *GetTreeHeader(TDSet* dset) = 0;
   virtual TList      *GetOutputNames() = 0;

   virtual void        AddChain(TChain* chain) = 0;
   virtual void        RemoveChain(TChain* chain) = 0;

   virtual TDrawFeedback *CreateDrawFeedback() = 0;
   virtual void           SetDrawFeedbackOption(TDrawFeedback *f, Option_t *opt) = 0;
   virtual void           DeleteDrawFeedback(TDrawFeedback *f) = 0;

   ClassDef(TVirtualProof,0)  // Abstract PROOF interface
};

R__EXTERN TVirtualProof *gProof;

#endif
