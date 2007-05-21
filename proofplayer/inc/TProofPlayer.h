// @(#)root/proofplayer:$Name:  $:$Id: TProofPlayer.h,v 1.44 2007/03/19 01:36:56 rdm Exp $
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
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualProofPlayer
#include "TVirtualProofPlayer.h"
#endif
#ifndef ROOT_TArrayL64
#include "TArrayL64.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_TSystem
#include "TSystem.h"
#endif
#ifndef ROOT_TQueryResult
#include "TQueryResult.h"
#endif

class TSelector;
class TProof;
class TSocket;
class TVirtualPacketizer;
class TSlave;
class TEventIter;
class TProofStats;
class TMutex;
class TStatus;
class TTimer;


//------------------------------------------------------------------------

class TProofPlayer : public TVirtualProofPlayer {

private:
   TList        *fAutoBins;  // Map of min/max values by name for slaves

protected:
   TList        *fInput;           //-> list with input objects
   TList        *fOutput;          //   list with output objects
   TSelector    *fSelector;        //!  the latest selector
   TClass       *fSelectorClass;   //!  class of the latest selector
   TTimer       *fFeedbackTimer;   //!  timer for sending intermediate results
   Long_t        fFeedbackPeriod;  //!  period (ms) for sending intermediate results
   TEventIter   *fEvIter;          //!  iterator on events or objects
   TStatus      *fSelStatus;       //!  status of query in progress
   EExitStatus   fExitStatus;      //   exit status
   Long64_t      fEventsProcessed; //   number of events processed
   Long64_t      fTotalEvents;     //   number of events requested

   TList        *fQueryResults;    //List of TQueryResult
   TQueryResult *fQuery;           //Instance of TQueryResult currently processed
   TQueryResult *fPreviousQuery;   //Previous instance of TQueryResult processed
   Int_t         fDrawQueries;     //Number of Draw queries in the list
   Int_t         fMaxDrawQueries;  //Max number of Draw queries kept

   TTimer       *fStopTimer;       //Timer associated with a stop request
   TMutex       *fStopTimerMtx;    //To protect the stop timer

   TTimer       *fDispatchTimer;    //Dispatch pending events while processing

   void         *GetSender() { return this; }  //used to set gTQSender

   virtual void SetupFeedback();  // specialized setup

public:   // fix for broken compilers so TCleanup can call StopFeedback()
   virtual void StopFeedback();   // specialized teardown

protected:
   class TCleanup {
   private:
      TProofPlayer *fPlayer;
   public:
      TCleanup(TProofPlayer *p) : fPlayer(p) { }
      ~TCleanup() { gSystem->Syslog(kLogErr, "!!!cleanup!!!"); fPlayer->StopFeedback(); }
   };

public:
   enum EStatusBits { kDispatchOneEvent = BIT(15) };

   TProofPlayer(TProof *proof = 0);
   virtual ~TProofPlayer();

   Long64_t  Process(TDSet *set,
                     const char *selector, Option_t *option = "",
                     Long64_t nentries = -1, Long64_t firstentry = 0,
                     TEventList *evl = 0);
   Long64_t  Finalize(Bool_t force = kFALSE, Bool_t sync = kFALSE);
   Long64_t  Finalize(TQueryResult *qr);
   Long64_t  DrawSelect(TDSet *set, const char *varexp,
                        const char *selection, Option_t *option = "",
                        Long64_t nentries = -1, Long64_t firstentry = 0);
   void      HandleGetTreeHeader(TMessage *mess);
   void      HandleRecvHisto(TMessage *mess);

   void      StopProcess(Bool_t abort, Int_t timeout = -1);
   void      AddInput(TObject *inp);
   void      ClearInput();
   TObject  *GetOutput(const char *name) const;
   TList    *GetOutputList() const;
   TList    *GetInputList() const { return fInput; }
   TList    *GetListOfResults() const { return fQueryResults; }
   void      AddQueryResult(TQueryResult *q);
   TQueryResult *GetCurrentQuery() const { return fQuery; }
   TQueryResult *GetQueryResult(const char *ref);
   void      RemoveQueryResult(const char *ref);
   void      SetCurrentQuery(TQueryResult *q);
   void      SetMaxDrawQueries(Int_t max) { fMaxDrawQueries = max; }
   void      RestorePreviousQuery() { fQuery = fPreviousQuery; }
   Int_t     AddOutputObject(TObject *obj);
   void      AddOutput(TList *out);   // Incorporate a list
   void      StoreOutput(TList *out);   // Adopts the list
   void      StoreFeedback(TObject *slave, TList *out); // Adopts the list
   void      Progress(Long64_t total, Long64_t processed); // *SIGNAL*
   void      Progress(TSlave *, Long64_t total, Long64_t processed)
                { Progress(total, processed); }
   void      Progress(Long64_t total, Long64_t processed, Long64_t bytesread,
                      Float_t initTime, Float_t procTime,
                      Float_t evtrti, Float_t mbrti); // *SIGNAL*
   void      Feedback(TList *objs); // *SIGNAL*

   TDrawFeedback *CreateDrawFeedback(TProof *p);
   void           SetDrawFeedbackOption(TDrawFeedback *f, Option_t *opt);
   void           DeleteDrawFeedback(TDrawFeedback *f);

   TDSetElement *GetNextPacket(TSlave *slave, TMessage *r);

   Int_t     ReinitSelector(TQueryResult *qr);

   void      UpdateAutoBin(const char *name,
                           Double_t& xmin, Double_t& xmax,
                           Double_t& ymin, Double_t& ymax,
                           Double_t& zmin, Double_t& zmax);

   Bool_t    IsClient() const { return kFALSE; }

   EExitStatus GetExitStatus() const { return fExitStatus; }
   Long64_t    GetEventsProcessed() const { return fEventsProcessed; }
   void        AddEventsProcessed(Long64_t ev) { fEventsProcessed += ev; }

   void      SetDispatchTimer(Bool_t on = kTRUE);
   void      SetStopTimer(Bool_t on = kTRUE,
                          Bool_t abort = kFALSE, Int_t timeout = 0);

   ClassDef(TProofPlayer,0)  // Basic PROOF player
};


//------------------------------------------------------------------------

class TProofPlayerLocal : public TProofPlayer {

public:
   TProofPlayerLocal(TProof *) { }

   ClassDef(TProofPlayerLocal,0)  // PROOF player running on client
};


//------------------------------------------------------------------------

class TProofPlayerRemote : public TProofPlayer {

private:
   TProof             *fProof;         // link to associated PROOF session
   TList              *fOutputLists;   // results returned by slaves
   TList              *fFeedback;      // reference for use on master
   TList              *fFeedbackLists; // intermediate results
   TVirtualPacketizer *fPacketizer;    // transform TDSet into packets for slaves
   TDSet              *fDSet;          //!tdset for current processing

   TList              *MergeFeedback();

protected:
   virtual Bool_t  HandleTimer(TTimer *timer);
   virtual Bool_t  SendSelector(const char *selector_file); //send selector to slaves
   TProof         *GetProof() const { return fProof; }
   void            SetupFeedback();  // specialized setup
   void            StopFeedback();   // specialized teardown

public:
   TProofPlayerRemote(TProof *proof = 0) : fProof(proof), fOutputLists(0), fFeedback(0),
                                           fFeedbackLists(0), fPacketizer(0) {}
   virtual ~TProofPlayerRemote();   // Owns the fOutput list

   Long64_t       Process(TDSet *set, const char *selector,
                          Option_t *option = "", Long64_t nentries = -1,
                          Long64_t firstentry = 0, TEventList *evl = 0);
   Long64_t       Finalize(Bool_t force = kFALSE, Bool_t sync = kFALSE);
   Long64_t       Finalize(TQueryResult *qr);
   Long64_t       DrawSelect(TDSet *set, const char *varexp,
                             const char *selection, Option_t *option = "",
                             Long64_t nentries = -1, Long64_t firstentry = 0);

   void           StopProcess(Bool_t abort, Int_t timeout = -1);
   void           StoreOutput(TList *out);   // Adopts the list
   void           StoreFeedback(TObject *slave, TList *out); // Adopts the list
   Int_t          Incorporate(TObject *obj, TList *out, Bool_t &merged);
   Int_t          AddOutputObject(TObject *obj);
   void           AddOutput(TList *out);   // Incorporate a list
   void           MergeOutput();
   void           Progress(Long64_t total, Long64_t processed); // *SIGNAL*
   void           Progress(TSlave*, Long64_t total, Long64_t processed)
                     { Progress(total, processed); }
   void           Progress(Long64_t total, Long64_t processed, Long64_t bytesread,
                           Float_t initTime, Float_t procTime,
                           Float_t evtrti, Float_t mbrti); // *SIGNAL*
   void           Feedback(TList *objs); // *SIGNAL*
   TDSetElement  *GetNextPacket(TSlave *slave, TMessage *r);

   Bool_t         IsClient() const;

   ClassDef(TProofPlayerRemote,0)  // PROOF player running on master server
};


//------------------------------------------------------------------------

class TProofPlayerSlave : public TProofPlayer {

private:
   TSocket *fSocket;
   TList   *fFeedback;  // List of objects to send updates of

   Bool_t HandleTimer(TTimer *timer);

protected:
   void SetupFeedback();
   void StopFeedback();

public:
   TProofPlayerSlave(TSocket *socket = 0) : fSocket(socket), fFeedback(0) { }

   void  HandleGetTreeHeader(TMessage *mess);

   ClassDef(TProofPlayerSlave,0)  // PROOF player running on slave server
};


//------------------------------------------------------------------------

class TProofPlayerSuperMaster : public TProofPlayerRemote {

private:
   TArrayL64 fSlaveProgress;
   TArrayL64 fSlaveTotals;
   TList     fSlaves;
   Bool_t    fReturnFeedback;

protected:
   Bool_t HandleTimer(TTimer *timer);
   void   SetupFeedback();

public:
   TProofPlayerSuperMaster(TProof *proof = 0) :
      TProofPlayerRemote(proof), fReturnFeedback(kFALSE) { }
   virtual ~TProofPlayerSuperMaster() { }

   Long64_t Process(TDSet *set, const char *selector,
                    Option_t *option = "", Long64_t nentries = -1,
                    Long64_t firstentry = 0, TEventList *evl = 0);
   void  Progress(Long64_t total, Long64_t processed)
                    { TProofPlayerRemote::Progress(total, processed); }
   void  Progress(Long64_t total, Long64_t processed, Long64_t bytesread,
                  Float_t initTime, Float_t procTime,
                  Float_t evtrti, Float_t mbrti)
                    { TProofPlayerRemote::Progress(total, processed, bytesread,
                                                   initTime, procTime, evtrti, mbrti); }
   void  Progress(TSlave *sl, Long64_t total, Long64_t processed);

   ClassDef(TProofPlayerSuperMaster,0)  // PROOF player running on super master
};

#endif
