// @(#)root/proofplayer:$Id$
// Author: Maarten Ballintijn   07/01/02

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofPlayer
#define ROOT_TProofPlayer


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofPlayer                                                         //
//                                                                      //
// This internal class and its subclasses steer the processing in PROOF.//
// Instances of the TProofPlayer class are created on the worker nodes  //
// per session and do the processing.                                   //
// Instances of its subclass - TProofPlayerRemote are created per each  //
// query on the master(s) and on the client. On the master(s),          //
// TProofPlayerRemote coordinate processing, check the dataset, create  //
// the packetizer and take care of merging the results of the workers.  //
// The instance on the client collects information on the input         //
// (dataset and selector), it invokes the Begin() method and finalizes  //
// the query by calling Terminate().                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualProofPlayer.h"
#include "TArrayL64.h"
#include "TArrayF.h"
#include "TArrayI.h"
#include "TList.h"
#include "TSystem.h"
#include "TQueryResult.h"
#include "TProofProgressStatus.h"
#include "TError.h"

#include <mutex>

class TSelector;
class TSocket;
class TVirtualPacketizer;
class TSlave;
class TEventIter;
class TProofStats;
class TStatus;
class TTimer;
class THashList;
class TH1;
class TFile;
class TStopwatch;

//------------------------------------------------------------------------

class TProofPlayer : public TVirtualProofPlayer {

private:
   TList        *fAutoBins;  // Map of min/max values by name for slaves

protected:
   TList        *fInput;           //-> list with input objects
   THashList    *fOutput;          //   list with output objects
   TSelector    *fSelector;        //!  the latest selector
   Bool_t        fCreateSelObj;    //!  kTRUE when fSelector has been created locally
   TClass       *fSelectorClass;   //!  class of the latest selector
   TTimer       *fFeedbackTimer;   //!  timer for sending intermediate results
   Long_t        fFeedbackPeriod;  //!  period (ms) for sending intermediate results
   TEventIter   *fEvIter;          //!  iterator on events or objects
   TStatus      *fSelStatus;       //!  status of query in progress
   EExitStatus   fExitStatus;      //   exit status
   Long64_t      fTotalEvents;     //   number of events requested
   TProofProgressStatus *fProgressStatus; // the progress status object;

   Long64_t      fReadBytesRun;   //! Bytes read in this run
   Long64_t      fReadCallsRun;   //! Read calls in this run
   Long64_t      fProcessedRun;   //! Events processed in this run

   TList        *fQueryResults;    //List of TQueryResult
   TQueryResult *fQuery;           //Instance of TQueryResult currently processed
   TQueryResult *fPreviousQuery;   //Previous instance of TQueryResult processed
   Int_t         fDrawQueries;     //Number of Draw queries in the list
   Int_t         fMaxDrawQueries;  //Max number of Draw queries kept

   TTimer       *fStopTimer;       //Timer associated with a stop request
   std::mutex    fStopTimerMtx;    //To protect the stop timer

   TTimer       *fDispatchTimer;    //Dispatch pending events while processing

   TTimer       *fProcTimeTimer;    //Notifies reaching of allowed max proc time
   TStopwatch   *fProcTime;         //Packet proc time

   TString       fOutputFilePath;   //Path to file with (partial) results of the query
   TFile        *fOutputFile;       //TFile object attached to fOutputFilePath
   Long_t        fSaveMemThreshold; //Threshold for saving output to file
   Bool_t        fSavePartialResults; //Whether to save the partial results
   Bool_t        fSaveResultsPerPacket; //Whether to save partial results after each packet

   static THashList *fgDrawInputPars;  // List of input parameters to be kept on drawing actions

   void         *GetSender() override { return this; }  //used to set gTQSender

   virtual Int_t DrawCanvas(TObject *obj); // Canvas drawing via libProofDraw

   virtual void SetupFeedback();  // specialized setup
   
   void  MergeOutput(Bool_t savememvalues = kFALSE) override;

public:   // fix for broken compilers so TCleanup can call StopFeedback()
   virtual void StopFeedback();   // specialized teardown

protected:
   class TCleanup {
   private:
      TProofPlayer *fPlayer;
   public:
      TCleanup(TProofPlayer *p) : fPlayer(p) { }
      ~TCleanup() { fPlayer->StopFeedback(); }
   };

   Int_t  AssertSelector(const char *selector_file);
   Bool_t CheckMemUsage(Long64_t &mfreq, Bool_t &w80r, Bool_t &w80v, TString &wmsg);

   void MapOutputListToDataMembers() const;

public:
   enum EStatusBits { kDispatchOneEvent = BIT(15), kIsProcessing = BIT(16),
                      kMaxProcTimeReached = BIT(17), kMaxProcTimeExtended = BIT(18) };

   TProofPlayer(TProof *proof = 0);
   ~TProofPlayer() override;

   Long64_t  Process(TDSet *set,
                     const char *selector, Option_t *option = "",
                     Long64_t nentries = -1, Long64_t firstentry = 0) override;
   Long64_t  Process(TDSet *set,
                     TSelector *selector, Option_t *option = "",
                     Long64_t nentries = -1, Long64_t firstentry = 0) override;
   Bool_t JoinProcess(TList *workers) override;
   TVirtualPacketizer *GetPacketizer() const override { return 0; }
   Long64_t  Finalize(Bool_t force = kFALSE, Bool_t sync = kFALSE) override;
   Long64_t  Finalize(TQueryResult *qr) override;
   Long64_t  DrawSelect(TDSet *set, const char *varexp,
                        const char *selection, Option_t *option = "",
                        Long64_t nentries = -1, Long64_t firstentry = 0) override;
   Int_t     GetDrawArgs(const char *var, const char *sel, Option_t *opt,
                         TString &selector, TString &objname) override;
   void      HandleGetTreeHeader(TMessage *mess) override;
   void      HandleRecvHisto(TMessage *mess) override;
   void      FeedBackCanvas(const char *name, Bool_t create);

   void      StopProcess(Bool_t abort, Int_t timeout = -1) override;
   void      AddInput(TObject *inp) override;
   void      ClearInput() override;
   TObject  *GetOutput(const char *name) const override;
   TList    *GetOutputList() const override;
   TList    *GetInputList() const override { return fInput; }
   TList    *GetListOfResults() const override { return fQueryResults; }
   void      AddQueryResult(TQueryResult *q) override;
   TQueryResult *GetCurrentQuery() const override { return fQuery; }
   TQueryResult *GetQueryResult(const char *ref) override;
   void      RemoveQueryResult(const char *ref) override;
   void      SetCurrentQuery(TQueryResult *q) override;
   void      SetMaxDrawQueries(Int_t max) override { fMaxDrawQueries = max; }
   void      RestorePreviousQuery() override { fQuery = fPreviousQuery; }
   Int_t     AddOutputObject(TObject *obj) override;
   void      AddOutput(TList *out) override;   // Incorporate a list
   void      StoreOutput(TList *out) override;   // Adopts the list
   void      StoreFeedback(TObject *slave, TList *out) override; // Adopts the list
   void      Progress(Long64_t total, Long64_t processed) override; // *SIGNAL*
   void      Progress(TSlave *, Long64_t total, Long64_t processed) override
                { Progress(total, processed); }
   void      Progress(Long64_t total, Long64_t processed, Long64_t bytesread,
                      Float_t initTime, Float_t procTime,
                      Float_t evtrti, Float_t mbrti) override; // *SIGNAL*
   void      Progress(TSlave *, Long64_t total, Long64_t processed, Long64_t bytesread,
                      Float_t initTime, Float_t procTime,
                      Float_t evtrti, Float_t mbrti) override
                { Progress(total, processed, bytesread, initTime, procTime,
                           evtrti, mbrti); } // *SIGNAL*
   void      Progress(TProofProgressInfo *pi) override; // *SIGNAL*
   void      Progress(TSlave *, TProofProgressInfo *pi) override { Progress(pi); } // *SIGNAL*
   void      Feedback(TList *objs) override; // *SIGNAL*

   TDrawFeedback *CreateDrawFeedback(TProof *p) override;
   void           SetDrawFeedbackOption(TDrawFeedback *f, Option_t *opt) override;
   void           DeleteDrawFeedback(TDrawFeedback *f) override;

   TDSetElement *GetNextPacket(TSlave *slave, TMessage *r) override;

   Int_t     ReinitSelector(TQueryResult *qr) override;

   void      UpdateAutoBin(const char *name,
                           Double_t& xmin, Double_t& xmax,
                           Double_t& ymin, Double_t& ymax,
                           Double_t& zmin, Double_t& zmax) override;

   Bool_t    IsClient() const override { return kFALSE; }

   void      SetExitStatus(EExitStatus st) override { fExitStatus = st; }
   EExitStatus GetExitStatus() const override { return fExitStatus; }
   Long64_t    GetEventsProcessed() const override { return fProgressStatus->GetEntries(); }
   void        AddEventsProcessed(Long64_t ev) override { fProgressStatus->IncEntries(ev); }

   void      SetDispatchTimer(Bool_t on = kTRUE) override;
   void      SetStopTimer(Bool_t on = kTRUE,
                          Bool_t abort = kFALSE, Int_t timeout = 0) override;

   void      SetInitTime() override { }

   void      SetMerging(Bool_t = kTRUE) override { }

   Long64_t  GetCacheSize() override;
   Int_t     GetLearnEntries() override;

   void      SetOutputFilePath(const char *fp) override { fOutputFilePath = fp; }
   Int_t     SavePartialResults(Bool_t queryend = kFALSE, Bool_t force = kFALSE) override;

   void              SetProcessing(Bool_t on = kTRUE);
   TProofProgressStatus  *GetProgressStatus() const override { return fProgressStatus; }

   void      UpdateProgressInfo() override;

   ClassDefOverride(TProofPlayer,0)  // Basic PROOF player
};


//------------------------------------------------------------------------

class TProofPlayerLocal : public TProofPlayer {

private:
   Bool_t   fIsClient;

protected:
   void SetupFeedback() override { }
   void StopFeedback() override { }

public:
   TProofPlayerLocal(Bool_t client = kTRUE) : fIsClient(client) { }
   ~TProofPlayerLocal() override { }

   Bool_t         IsClient() const override { return fIsClient; }
   Long64_t  Process(const char *selector, Long64_t nentries = -1, Option_t *option = "");
   Long64_t  Process(TSelector *selector, Long64_t nentries = -1, Option_t *option = "");
   Long64_t  Process(TDSet *set,
                     const char *selector, Option_t *option = "",
                     Long64_t nentries = -1, Long64_t firstentry = 0) override {
             return TProofPlayer::Process(set, selector, option, nentries, firstentry); }
   Long64_t  Process(TDSet *set,
                     TSelector *selector, Option_t *option = "",
                     Long64_t nentries = -1, Long64_t firstentry = 0) override {
             return TProofPlayer::Process(set, selector, option, nentries, firstentry); }
   ClassDefOverride(TProofPlayerLocal,0)  // PROOF player running on client
};


//------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofPlayerRemote                                                   //
//                                                                      //
// Instances of TProofPlayerRemote are created per each query on the    //
// master(s) and on the client. On the master(s), TProofPlayerRemote    //
// coordinate processing, check the dataset, create the packetizer      //
// and take care of merging the results of the workers.                 //
// The instance on the client collects information on the input         //
// (dataset and selector), it invokes the Begin() method and finalizes  //
// the query by calling Terminate().                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


class TProofPlayerRemote : public TProofPlayer {

protected:
   TProof             *fProof;         // link to associated PROOF session
   TList              *fOutputLists;   // results returned by slaves
   TList              *fFeedback;      // reference for use on master
   TList              *fFeedbackLists; // intermediate results
   TVirtualPacketizer *fPacketizer;    // transform TDSet into packets for slaves
   Bool_t              fMergeFiles;    // is True when merging output files centrally is needed
   TDSet              *fDSet;          //!tdset for current processing
   ErrorHandlerFunc_t  fErrorHandler;  // Store previous handler when redirecting output
   Bool_t              fMergeTH1OneByOne;  // If kTRUE forces TH1 merge one-by-one [kTRUE]
   TH1                *fProcPackets;    //!Histogram with packets being processed (owned by TPerfStats)
   TMessage           *fProcessMessage; // Process message to replay when adding new workers dynamically
   TString             fSelectorFileName; // Current Selector's name, set by Process()

   TStopwatch         *fMergeSTW;      // Merging stop watch
   Int_t               fNumMergers;    // Number of submergers

   Bool_t  HandleTimer(TTimer *timer) override;
   Int_t           InitPacketizer(TDSet *dset, Long64_t nentries,
                                  Long64_t first, const char *defpackunit,
                                  const char *defpackdata);
   TList          *MergeFeedback();
   Bool_t          MergeOutputFiles();
   void            NotifyMemory(TObject *obj);
   void            SetLastMergingMsg(TObject *obj);
   virtual Bool_t  SendSelector(const char *selector_file); //send selector to slaves
   TProof         *GetProof() const { return fProof; }
   void            SetupFeedback() override;  // specialized setup
   void            StopFeedback() override;   // specialized teardown
   void            SetSelectorDataMembersFromOutputList();

public:
   TProofPlayerRemote(TProof *proof = 0) : fProof(proof), fOutputLists(0), fFeedback(0),
                                           fFeedbackLists(0), fPacketizer(0),
                                           fMergeFiles(kFALSE), fDSet(0), fErrorHandler(0),
                                           fMergeTH1OneByOne(kTRUE), fProcPackets(0),
                                           fProcessMessage(0), fMergeSTW(0), fNumMergers(0) 
                                           { fProgressStatus = new TProofProgressStatus(); }
   ~TProofPlayerRemote() override;   // Owns the fOutput list
   Long64_t Process(TDSet *set, const char *selector,
                            Option_t *option = "", Long64_t nentries = -1,
                            Long64_t firstentry = 0) override;
   Long64_t Process(TDSet *set, TSelector *selector,
                            Option_t *option = "", Long64_t nentries = -1,
                            Long64_t firstentry = 0) override;
   Bool_t JoinProcess(TList *workers) override;
   Long64_t Finalize(Bool_t force = kFALSE, Bool_t sync = kFALSE) override;
   Long64_t Finalize(TQueryResult *qr) override;
   Long64_t       DrawSelect(TDSet *set, const char *varexp,
                             const char *selection, Option_t *option = "",
                             Long64_t nentries = -1, Long64_t firstentry = 0) override;

   void           RedirectOutput(Bool_t on = kTRUE);
   void           StopProcess(Bool_t abort, Int_t timeout = -1) override;
   void           StoreOutput(TList *out) override;   // Adopts the list
   void   StoreFeedback(TObject *slave, TList *out) override; // Adopts the list
   Int_t          Incorporate(TObject *obj, TList *out, Bool_t &merged);
   TObject       *HandleHistogram(TObject *obj, Bool_t &merged);
   Bool_t         HistoSameAxis(TH1 *h0, TH1 *h1);
   Int_t          AddOutputObject(TObject *obj) override;
   void           AddOutput(TList *out) override;   // Incorporate a list
   void   MergeOutput(Bool_t savememvalues = kFALSE) override;
   void           Progress(Long64_t total, Long64_t processed) override; // *SIGNAL*
   void           Progress(TSlave*, Long64_t total, Long64_t processed) override
                     { Progress(total, processed); }
   void           Progress(Long64_t total, Long64_t processed, Long64_t bytesread,
                           Float_t initTime, Float_t procTime,
                           Float_t evtrti, Float_t mbrti) override; // *SIGNAL*
   void           Progress(TSlave *, Long64_t total, Long64_t processed, Long64_t bytesread,
                           Float_t initTime, Float_t procTime,
                           Float_t evtrti, Float_t mbrti) override
                      { Progress(total, processed, bytesread, initTime, procTime,
                           evtrti, mbrti); } // *SIGNAL*
   void           Progress(TProofProgressInfo *pi) override; // *SIGNAL*
   void           Progress(TSlave *, TProofProgressInfo *pi) override { Progress(pi); } // *SIGNAL*
   void           Feedback(TList *objs) override; // *SIGNAL*
   TDSetElement  *GetNextPacket(TSlave *slave, TMessage *r) override;
   TVirtualPacketizer *GetPacketizer() const override { return fPacketizer; }

   Bool_t         IsClient() const override;

   void           SetInitTime() override;

   void           SetMerging(Bool_t on = kTRUE) override;

   ClassDefOverride(TProofPlayerRemote,0)  // PROOF player running on master server
};


//------------------------------------------------------------------------

class TProofPlayerSlave : public TProofPlayer {

private:
   TSocket *fSocket;
   TList   *fFeedback;  // List of objects to send updates of

   Bool_t HandleTimer(TTimer *timer) override;

protected:
   void SetupFeedback() override;
   void StopFeedback() override;

public:
   TProofPlayerSlave(TSocket *socket = 0) : fSocket(socket), fFeedback(0) { }

   void  HandleGetTreeHeader(TMessage *mess) override;

   ClassDefOverride(TProofPlayerSlave,0)  // PROOF player running on slave server
};


//------------------------------------------------------------------------

class TProofPlayerSuperMaster : public TProofPlayerRemote {

private:
   TArrayL64 fSlaveProgress;
   TArrayL64 fSlaveTotals;
   TArrayL64 fSlaveBytesRead;
   TArrayF   fSlaveInitTime;
   TArrayF   fSlaveProcTime;
   TArrayF   fSlaveEvtRti;
   TArrayF   fSlaveMBRti;
   TArrayI   fSlaveActW;
   TArrayI   fSlaveTotS;
   TArrayF   fSlaveEffS;
   TList     fSlaves;
   Bool_t    fReturnFeedback;

protected:
   Bool_t HandleTimer(TTimer *timer) override;
   void   SetupFeedback() override;

public:
   TProofPlayerSuperMaster(TProof *proof = 0) :
      TProofPlayerRemote(proof), fReturnFeedback(kFALSE) { }
   ~TProofPlayerSuperMaster() override { }

   Long64_t Process(TDSet *set, const char *selector,
                    Option_t *option = "", Long64_t nentries = -1,
                    Long64_t firstentry = 0) override;
   Long64_t Process(TDSet *set, TSelector *selector,
                    Option_t *option = "", Long64_t nentries = -1,
                    Long64_t firstentry = 0) override
                    { return TProofPlayerRemote::Process(set, selector, option,
                                                         nentries, firstentry); }
   void  Progress(Long64_t total, Long64_t processed) override
                    { TProofPlayerRemote::Progress(total, processed); }
   void  Progress(Long64_t total, Long64_t processed, Long64_t bytesread,
                  Float_t initTime, Float_t procTime,
                  Float_t evtrti, Float_t mbrti) override
                    { TProofPlayerRemote::Progress(total, processed, bytesread,
                                                   initTime, procTime, evtrti, mbrti); }
   void  Progress(TProofProgressInfo *pi) override { TProofPlayerRemote::Progress(pi); }
   void  Progress(TSlave *sl, Long64_t total, Long64_t processed) override;
   void  Progress(TSlave *sl, Long64_t total, Long64_t processed, Long64_t bytesread,
                  Float_t initTime, Float_t procTime,
                  Float_t evtrti, Float_t mbrti) override;
   void  Progress(TSlave *sl, TProofProgressInfo *pi) override;

   ClassDefOverride(TProofPlayerSuperMaster,0)  // PROOF player running on super master
};

#endif
