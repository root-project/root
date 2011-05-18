// @(#)root/proof:$Id$
// Author: Fons Rademakers   15/03/07

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualProofPlayer
#define ROOT_TVirtualProofPlayer


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualProofPlayer                                                  //
//                                                                      //
// Abstract interface for the PROOF player.                             //
// See the concrete implementations under 'proofplayer' for details.    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TQObject
#include "TQObject.h"
#endif

class TDSet;
class TDSetElement;
class TEventList;
class TQueryResult;
class TDrawFeedback;
class TList;
class TSlave;
class TMessage;
class TProof;
class TSocket;
class TVirtualPacketizer;
class TProofProgressStatus;
class TProofProgressInfo;


class TVirtualProofPlayer : public TObject, public TQObject {

public:
   // TDSet status bits
   enum EExitStatus { kFinished, kStopped, kAborted };

   TVirtualProofPlayer() { }
   virtual ~TVirtualProofPlayer() { }

   virtual Long64_t  Process(TDSet *set,
                             const char *selector, Option_t *option = "",
                             Long64_t nentries = -1, Long64_t firstentry = 0) = 0;
   virtual Long64_t  Finalize(Bool_t force = kFALSE, Bool_t sync = kFALSE) = 0;
   virtual Long64_t  Finalize(TQueryResult *qr) = 0;
   virtual Long64_t  DrawSelect(TDSet *set, const char *varexp,
                                const char *selection, Option_t *option = "",
                                Long64_t nentries = -1, Long64_t firstentry = 0) = 0;
   virtual Int_t     GetDrawArgs(const char *var, const char *sel, Option_t *opt,
                                 TString &selector, TString &objname) = 0;
   virtual void      HandleGetTreeHeader(TMessage *mess) = 0;
   virtual void      HandleRecvHisto(TMessage *mess) = 0;

   virtual void      StopProcess(Bool_t abort, Int_t timeout = -1) = 0;
   virtual void      AddInput(TObject *inp) = 0;
   virtual void      ClearInput() = 0;
   virtual TObject  *GetOutput(const char *name) const = 0;
   virtual TList    *GetOutputList() const = 0;
   virtual TList    *GetInputList() const = 0;
   virtual TList    *GetListOfResults() const = 0;
   virtual void      AddQueryResult(TQueryResult *q) = 0;
   virtual TQueryResult *GetCurrentQuery() const = 0;
   virtual TQueryResult *GetQueryResult(const char *ref) = 0;
   virtual void      RemoveQueryResult(const char *ref) = 0;
   virtual void      SetCurrentQuery(TQueryResult *q) = 0;
   virtual void      SetMaxDrawQueries(Int_t max) = 0;
   virtual void      RestorePreviousQuery() =0 ;
   virtual Int_t     AddOutputObject(TObject *obj) = 0;
   virtual void      AddOutput(TList *out) = 0;   // Incorporate a list
   virtual void      StoreOutput(TList *out) = 0;   // Adopts the list
   virtual void      StoreFeedback(TObject *slave, TList *out) = 0; // Adopts the list
   virtual void      Progress(Long64_t total, Long64_t processed) = 0; // *SIGNAL*
   virtual void      Progress(TSlave *, Long64_t total, Long64_t processed) = 0;
   virtual void      Progress(Long64_t total, Long64_t processed, Long64_t bytesread,
                              Float_t initTime, Float_t procTime,
                              Float_t evtrti, Float_t mbrti) = 0; // *SIGNAL*
   virtual void      Progress(TSlave *, Long64_t total, Long64_t processed,
                              Long64_t bytesread, Float_t initTime, Float_t procTime,
                              Float_t evtrti, Float_t mbrti) = 0; // *SIGNAL*
   virtual void      Progress(TProofProgressInfo *) = 0; // *SIGNAL*
   virtual void      Progress(TSlave *, TProofProgressInfo *) = 0; // *SIGNAL*
   virtual void      Feedback(TList *objs) = 0; // *SIGNAL*

   virtual TDrawFeedback *CreateDrawFeedback(TProof *p) = 0;
   virtual void           SetDrawFeedbackOption(TDrawFeedback *f, Option_t *opt) = 0;
   virtual void           DeleteDrawFeedback(TDrawFeedback *f) = 0;

   virtual TDSetElement *GetNextPacket(TSlave *slave, TMessage *r) = 0;

   virtual Int_t     ReinitSelector(TQueryResult *qr) = 0;

   virtual void      UpdateAutoBin(const char *name,
                                   Double_t& xmin, Double_t& xmax,
                                   Double_t& ymin, Double_t& ymax,
                                   Double_t& zmin, Double_t& zmax) = 0;

   virtual void MergeOutput() = 0;

   virtual Bool_t    IsClient() const = 0;

   virtual EExitStatus GetExitStatus() const = 0;
   virtual Long64_t    GetEventsProcessed() const = 0;
   virtual void        AddEventsProcessed(Long64_t ev) = 0;
   virtual TProofProgressStatus*  GetProgressStatus() const = 0;

   virtual void      SetDispatchTimer(Bool_t on = kTRUE) = 0;
   virtual void      SetStopTimer(Bool_t on = kTRUE,
                                  Bool_t abort = kFALSE, Int_t timeout = 0) = 0;
   virtual void      SetInitTime() = 0;
   virtual Long64_t  GetCacheSize() = 0;
   virtual Int_t     GetLearnEntries() = 0;

   virtual TVirtualPacketizer *GetPacketizer() const { return 0; }

   static TVirtualProofPlayer *Create(const char *player, TProof *p, TSocket *s = 0);

   ClassDef(TVirtualProofPlayer,0)  // Abstract PROOF player
};

#endif
