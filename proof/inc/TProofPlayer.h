// @(#)root/proof:$Name:  $:$Id: TProofPlayer.h,v 1.1 2002/01/15 00:45:20 rdm Exp $
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
class TEventList;
class TProof;


//------------------------------------------------------------------------

class TProofPlayer : public TObject {

protected:
   TList   *fInput;    //-> list with input objects
   TList   *fOutput;   //   list with output objects

public:
   TProofPlayer();
   virtual ~TProofPlayer();

   virtual Int_t     Process(TDSet *set,
                             const char *selector,
                             Int_t nentries = -1, Int_t first = 0,
                             TEventList *evl = 0) = 0;

   virtual void      AddInput(TObject *inp);
   virtual void      ClearInput();
   virtual TObject  *GetOutput(const char *name) const;
   virtual TList    *GetOutputList() const;

   ClassDef(TProofPlayer,1)  // Abstract PROOF player
};


//------------------------------------------------------------------------

class TProofPlayerLocal : public TProofPlayer {

public:
   TProofPlayerLocal() { }

   Int_t Process(TDSet *set,
                 const char *selector,
                 Int_t nentries = -1, Int_t first = 0,
                 TEventList *evl = 0);

   ClassDef(TProofPlayerLocal,1)  // PROOF player running on client
};


//------------------------------------------------------------------------

class TProofPlayerRemote : public TProofPlayer {

private:
   TProof  *fProof;   // link to associated PROOF session

public:
   TProofPlayerRemote() { fProof = 0; }
   TProofPlayerRemote(TProof *proof);

   Int_t  Process(TDSet *set,
                  const char *selector,
                  Int_t nentries = -1, Int_t first = 0,
                  TEventList *evl = 0);

   ClassDef(TProofPlayerRemote,1)  // PROOF player running on master server
};


// -------------------------------------------------------------------

class TProofPlayerSlave : public TProofPlayer {

public:
   TProofPlayerSlave();

   Int_t  Process(TDSet *set,
                  const char *selector,
                  Int_t nentries = -1, Int_t first = 0,
                  TEventList *evl = 0);

   ClassDef(TProofPlayerSlave,1)  // PROOF player running on slave server
};

#endif
