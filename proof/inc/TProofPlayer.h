// @(#)root/proof:$Name:  $:$Id: TProofPlayer.h,v 1.7 2002/10/07 10:43:51 rdm Exp $
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


typedef long Long64_t;

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


//------------------------------------------------------------------------

class TProofPlayer : public TObject, public TQObject {

private:
   TList      *fAutoBins;  // Map of min/max values by name for slaves

protected:
   TList      *fInput;     //-> list with input objects
   TList      *fOutput;    //   list with output objects
   TSelector  *fSelector;  //!  The latest selector

   void       *GetSender() { return this; }  //used to set gTQSender

public:
   TProofPlayer();
   virtual ~TProofPlayer();

   virtual Int_t     Process(TDSet *set,
                             const char *selector,
                             Long64_t nentries = -1, Long64_t first = 0,
                             TEventList *evl = 0);

   virtual void      AddInput(TObject *inp);
   virtual void      ClearInput();
   virtual TObject  *GetOutput(const char *name) const;
   virtual TList    *GetOutputList() const;
   virtual void      StoreOutput(TList *out);   // Adopts the list
   virtual void      Progress(Long64_t total, Long64_t processed); //*SIGNAL*

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
   TProof             *fProof;       // Link to associated PROOF session
   TList              *fOutputLists; // Results returned by slaves
   TVirtualPacketizer *fPacketizer;  // Transform TDSet into packets for slaves

public:
   TProofPlayerRemote() { fProof = 0; fOutputLists = 0; }
   TProofPlayerRemote(TProof *proof);
   ~TProofPlayerRemote();   // Owns the fOutput list

   Int_t Process(TDSet *set,
                 const char *selector,
                 Long64_t nentries = -1, Long64_t first = 0,
                 TEventList *evl = 0);
   void  StoreOutput(TList *out);   // Adopts the list
   void  MergeOutput();

   TDSetElement *GetNextPacket(TSlave *slave, TMessage *r);

   ClassDef(TProofPlayerRemote,0)  // PROOF player running on master server
};


// -------------------------------------------------------------------

class TProofPlayerSlave : public TProofPlayer {
private:
   TSocket *fSocket;

public:
   TProofPlayerSlave();
   TProofPlayerSlave(TSocket *socket);

   ClassDef(TProofPlayerSlave,0)  // PROOF player running on slave server
};

#endif
