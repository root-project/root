// @(#)root/proof:$Name:  $:$Id: TEventIter.h,v 1.1 2002/01/18 14:24:09 rdm Exp $
// Author: Maarten Ballintijn   07/01/02

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEventIter
#define ROOT_TEventIter

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TEventIter                                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TDSet;
class TSelector;
class TFile;
class TIter;
class TTree;
class TSocket;


//------------------------------------------------------------------------

class TEventIter : public TObject {
public:
            TEventIter();
   virtual ~TEventIter();

   virtual Bool_t Init(TSelector *selector) = 0;
   virtual Bool_t GetNextEvent(TSelector *selector) = 0;
   virtual Double_t GetEntry() const = 0;

   ClassDef(TEventIter,1)  // Event iterator used by TProofPlayer's
};

//------------------------------------------------------------------------

class TEventIterLocal : public TEventIter {

private:
   Bool_t   fIsTree;
   TList   *fFiles;
   TIter   *fNextFile;

   TFile   *fFile;
   Double_t fMaxEntry;
   Double_t fEntry;

   TString  fClassName;
   TList   *fKeys;
   TIter   *fNextKey;
   TObject *fObj;

   TTree   *fTree;

   Bool_t GetNextEventTree(TSelector *selector);
   Bool_t GetNextEventObj(TSelector *selector);
   Bool_t LoadNextTree();

public:
   TEventIterLocal();
   TEventIterLocal(TDSet *set);
  ~TEventIterLocal();

   Bool_t   Init(TSelector *selector);
   Bool_t   GetNextEvent(TSelector *selector);
   Double_t GetEntry() const { return fEntry; };

   ClassDef(TEventIterLocal,1)  // Event iterator used by local TProofPlayer
};


//------------------------------------------------------------------------

class TEventIterSlave : public TEventIter {
private:
   TSocket  *fSocket;
   Bool_t   fIsTree;

   TFile   *fFile;
   Double_t fEntry;
   Double_t fMaxEntry;

   TString  fClassName;
   TList   *fKeys;
   TIter   *fNextKey;
   TObject *fObj;

   TTree   *fTree;

public:
   TEventIterSlave();
   TEventIterSlave(TSocket *socket);
  ~TEventIterSlave();

   Bool_t Init(TSelector *selector);
   Bool_t GetNextEvent(TSelector *selector);
   Double_t GetEntry() const { return fEntry; };

   ClassDef(TEventIterSlave,1)  // Event iterator used by TProofPlayer on Slave
};


#endif


