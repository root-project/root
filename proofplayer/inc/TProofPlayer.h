// @(#)root/proof:$Name:  $:$Id: TProofPlayer.h,v 1.2 2002/02/12 17:53:18 rdm Exp $
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

class TList;
class TSelector;
class TDSet;
class TDSetElement;
class TSlave;
class TEventList;
class TProof;
class TSocket;


//------------------------------------------------------------------------

class TProofPlayer : public TObject {

protected:
   TList      *fInput;     //-> list with input objects
   TList      *fOutput;    //   list with output objects
   TSelector  *fSelector;  //!  The latest selector

public:
   TProofPlayer();
   virtual ~TProofPlayer();

   virtual Int_t     Process(TDSet *set,
                             const char *selector,
                             Int_t nentries = -1, Int_t first = 0,
                             TEventList *evl = 0);

   virtual void      AddInput(TObject *inp);
   virtual void      ClearInput();
   virtual TObject  *GetOutput(const char *name) const;
   virtual TList    *GetOutputList() const;
   virtual void      StoreOutput(TList *out);   // Adopts the list

   virtual TDSetElement *GetNextPacket(TSlave *slave);


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
   TProof        *fProof;        // Link to associated PROOF session
   TList         *fOutputLists;  // Results returned by slaves

   // currently here -- for packet generation
   TDSet         *fSet;          // TDSet to split in packets
   TDSetElement  *fElem;         // Element currently being processed
   Double_t       fFirst;
   Double_t       fNum;
   Double_t       fCur;

public:
   TProofPlayerRemote() { fProof = 0; fOutputLists = 0; }
   TProofPlayerRemote(TProof *proof);
   ~TProofPlayerRemote();   // Owns the fOutput list

   Int_t Process(TDSet *set,
                 const char *selector,
                 Int_t nentries = -1, Int_t first = 0,
                 TEventList *evl = 0);
   void  StoreOutput(TList *out);   // Adopts the list
   void  MergeOutput();

   TDSetElement *GetNextPacket(TSlave *slave);

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
