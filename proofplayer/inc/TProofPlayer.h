// @(#)root/proof:$Name:  $:$Id: TProofPlayer.h,v 1.22 2004/07/09 01:34:51 rdm Exp $
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


class TList;
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
   virtual void      StoreOutput(TList *out);   // Adopts the list
   virtual void      StoreFeedback(TObject *slave, TList *out); // Adopts the list
   virtual void      Progress(Long64_t total, Long64_t processed); // *SIGNAL*
   virtual void      Feedback(TList *objs); // *SIGNAL*

   virtual TDSetElement *GetNextPacket(TSlave *slave, TMessage *r);
   void              UpdateAutoBin(const char *name,
                        Double_t& xmin, Double_t& xmax,
                        Double_t& ymin, Double_t& ymax,
                        Double_t& zmin, Double_t& zmax);


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

   virtual Bool_t      HandleTimer(TTimer *timer);
   TList              *MergeFeedback();

protected:
   virtual void        SetupFeedback();  // specialized setup
   virtual void        StopFeedback();   // specialized teardown

public:
   TProofPlayerRemote(TProof *proof = 0) : fProof(proof), fOutputLists(0), fFeedback(0),
                                           fFeedbackLists(0), fPacketizer(0) {}
   ~TProofPlayerRemote();   // Owns the fOutput list

   Long64_t       Process(TDSet *set, const char *selector,
                          Option_t *option = "", Long64_t nentries = -1,
                          Long64_t firstentry = 0, TEventList *evl = 0);
   void           StopProcess(Bool_t abort);
   void           StoreOutput(TList *out);   // Adopts the list
   void           StoreFeedback(TObject *slave, TList *out); // Adopts the list
   void           MergeOutput();
   TDSetElement  *GetNextPacket(TSlave *slave, TMessage *r);

   ClassDef(TProofPlayerRemote,0)  // PROOF player running on master server
};


//------------------------------------------------------------------------

class TProofPlayerSlave : public TProofPlayer {
private:
   TSocket *fSocket;
   TList   *fFeedback;  // List of objects to send updates of

   virtual Bool_t      HandleTimer(TTimer *timer);

protected:
   void SetupFeedback();
   void StopFeedback();

public:
   TProofPlayerSlave();
   TProofPlayerSlave(TSocket *socket);

   Long64_t    DrawSelect(TDSet *set, const char *varexp,
                       const char *selection, Option_t *option = "",
                       Long64_t nentries = -1, Long64_t firstentry = 0);

   ClassDef(TProofPlayerSlave,0)  // PROOF player running on slave server
};

#endif
