// @(#)root/proof:$Name:  $:$Id: TProofPlayer.h,v 1.26 2005/03/13 15:06:50 rdm Exp $
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

#ifndef ROOT_TArrayL
#include "TArrayL.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TObjString
#include "TObjString.h"
#endif
#ifndef ROOT_TQObject
#include "TQObject.h"
#endif
#ifndef ROOT_TSystem
#include "TSystem.h"
#endif


class TSelector;
class TDSet;
class TDSetElement;
class TSlave;
class TEventList;
class TProof;
class TSocket;
class TVirtualPacketizer;
class TMessage;
class TSlave;
class TEventIter;
class TProofStats;
class TStatus;


//------------------------------------------------------------------------

class TProofPlayer : public TObject, public TQObject {

private:
   TList      *fAutoBins;  // Map of min/max values by name for slaves

protected:
   TList      *fInput;         //-> list with input objects
   TList      *fOutput;        //   list with output objects
   TSelector  *fSelector;      //!  the latest selector
   TClass     *fSelectorClass; //!  class of the latest selector
   TTimer     *fFeedbackTimer; //!  timer for sending intermediate results
   TEventIter *fEvIter;        //!  iterator on events or objects
   TStatus    *fSelStatus;     //!  status of query in progress

   void       *GetSender() { return this; }  //used to set gTQSender

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
   TProofPlayer();
   virtual ~TProofPlayer();

   virtual Long64_t  Process(TDSet *set,
                             const char *selector, Option_t *option = "",
                             Long64_t nentries = -1, Long64_t firstentry = 0,
                             TEventList *evl = 0);
   virtual Long64_t  DrawSelect(TDSet *set, const char *varexp,
                                const char *selection, Option_t *option = "",
                                Long64_t nentries = -1, Long64_t firstentry = 0);

   virtual void      StopProcess(Bool_t abort);
   virtual void      AddInput(TObject *inp);
   virtual void      ClearInput();
   virtual TObject  *GetOutput(const char *name) const;
   virtual TList    *GetOutputList() const;
   virtual TList    *GetInputList() const { return fInput; }
   virtual void      StoreOutput(TList *out);   // Adopts the list
   virtual void      StoreFeedback(TObject *slave, TList *out); // Adopts the list
   virtual void      Progress(Long64_t total, Long64_t processed); // *SIGNAL*
   virtual void      Progress(TSlave *, Long64_t total, Long64_t processed)
                        { Progress(total, processed); }
   virtual void      Feedback(TList *objs); // *SIGNAL*

   virtual TDSetElement *GetNextPacket(TSlave *slave, TMessage *r);

   void              UpdateAutoBin(const char *name,
                                   Double_t& xmin, Double_t& xmax,
                                   Double_t& ymin, Double_t& ymax,
                                   Double_t& zmin, Double_t& zmax);

   virtual Bool_t    IsClient() const { return kFALSE; }

   ClassDef(TProofPlayer,0)  // Abstract PROOF player
};


//------------------------------------------------------------------------

class TProofPlayerLocal : public TProofPlayer {

public:
   TProofPlayerLocal() { }

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
   TProof         *GetProof() const { return fProof; }
   virtual Bool_t  SendSelector(const char *selector_file); //send selector to slaves
   virtual void    SetupFeedback();  // specialized setup
   virtual void    StopFeedback();   // specialized teardown

public:
   TProofPlayerRemote(TProof *proof = 0) : fProof(proof), fOutputLists(0), fFeedback(0),
                                           fFeedbackLists(0), fPacketizer(0) {}
   virtual ~TProofPlayerRemote();   // Owns the fOutput list

   Long64_t       Process(TDSet *set, const char *selector,
                          Option_t *option = "", Long64_t nentries = -1,
                          Long64_t firstentry = 0, TEventList *evl = 0);
   Long64_t       DrawSelect(TDSet *set, const char *varexp,
                             const char *selection, Option_t *option = "",
                             Long64_t nentries = -1, Long64_t firstentry = 0);

   void           StopProcess(Bool_t abort);
   void           StoreOutput(TList *out);   // Adopts the list
   void           StoreFeedback(TObject *slave, TList *out); // Adopts the list
   void           MergeOutput();
   void           Progress(Long64_t total, Long64_t processed); // *SIGNAL*
   void           Progress(TSlave *, Long64_t total, Long64_t processed)
                     { Progress(total, processed); }
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

   virtual Bool_t HandleTimer(TTimer *timer);

protected:
   void SetupFeedback();
   void StopFeedback();

public:
   TProofPlayerSlave();
   TProofPlayerSlave(TSocket *socket);

   Long64_t DrawSelect(TDSet *set, const char *varexp,
                       const char *selection, Option_t *option = "",
                       Long64_t nentries = -1, Long64_t firstentry = 0);

   ClassDef(TProofPlayerSlave,0)  // PROOF player running on slave server
};


//------------------------------------------------------------------------

class TProofPlayerSuperMaster : public TProofPlayerRemote {

private:
   TArrayL fSlaveProgress;
   TArrayL fSlaveTotals;
   TList   fSlaves;
   Bool_t  fReturnFeedback;

protected:
   virtual Bool_t HandleTimer(TTimer *timer);
   virtual void   SetupFeedback();

public:
   TProofPlayerSuperMaster(TProof *proof = 0) :
      TProofPlayerRemote(proof), fReturnFeedback(kFALSE) { }
   virtual ~TProofPlayerSuperMaster() { }

   virtual Long64_t Process(TDSet *set, const char *selector,
                            Option_t *option = "", Long64_t nentries = -1,
                            Long64_t firstentry = 0, TEventList *evl = 0);
   virtual void  Progress(Long64_t total, Long64_t processed)
                    { TProofPlayerRemote::Progress(total, processed); }
   virtual void  Progress(TSlave *sl, Long64_t total, Long64_t processed);

   ClassDef(TProofPlayerSuperMaster,0)  // PROOF player running on super master
};

#endif
