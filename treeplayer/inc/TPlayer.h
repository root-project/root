// @(#)root/treeplayer:$Name:$:$Id:$
// Author: Maarten Ballintijn   07/01/02

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPlayer
#define ROOT_TPlayer


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPlayer                                                              //
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

class TPlayer : public TObject {

protected:
   TList   *fInput;    // list with input objects
   TList   *fOutput;   // list with output objects

public:
   TPlayer();
   virtual ~TPlayer();

   virtual Int_t     Process(TDSet *set,
                             const char *selector,
                             Int_t nentries = -1, Int_t first = 0,
                             TEventList *evl = 0) = 0;

   virtual void      AddInput( TObject *inp );
   virtual void      ClearInput();
   virtual TObject  *GetOutput(const char *name) const;
   virtual TList    *GetOutputList() const;

   ClassDef(TPlayer,1)  // Player ABC
};


//------------------------------------------------------------------------

class TPlayerLocal : public TPlayer {

public:
   TPlayerLocal() { }

   Int_t Process(TDSet *set,
                 const char *selector,
                 Int_t nentries = -1, Int_t first = 0,
                 TEventList *evl = 0);

   ClassDef(TPlayerLocal,1)  // Player running on client
};


//------------------------------------------------------------------------

class TPlayerRemote : public TPlayer {

private:
   TProof  *fProof;   // link to associated PROOF session

public:
   TPlayerRemote() { fProof = 0; }
   TPlayerRemote(TProof *proof);

   Int_t  Process(TDSet *set,
                  const char *selector,
                  Int_t nentries = -1, Int_t first = 0,
                  TEventList *evl = 0);

   ClassDef(TPlayerRemote,1)  // Player running on PROOF master server
};


// -------------------------------------------------------------------

class TPlayerSlave : public TPlayer {

public:
   TPlayerSlave();

   Int_t  Process(TDSet *set,
                  const char *selector,
                  Int_t nentries = -1, Int_t first = 0,
                  TEventList *evl = 0);

   ClassDef(TPlayerSlave,1)  // Player running on PROOF slave server
};

#endif
