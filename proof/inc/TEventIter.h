// @(#)root/proof:$Name:  $:$Id: TEventIter.h,v 1.2 2002/02/12 17:53:18 rdm Exp $
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
class TDirectory;
class TSelector;
class TList;
class TIter;
class TTree;


//------------------------------------------------------------------------

class TEventIter : public TObject {

protected:
   TDSet       *fDSet;      // data set over which to iterate
   TDirectory  *fDir;       // directory containing the objects
   TSelector   *fSel;       // selector to by used
   Double_t     fFirst;     // first entry to process
   Double_t     fNum;       // number of entries to process
   Double_t     fCur;       // current entry

public:
   TEventIter();
   TEventIter(TDSet *dset, TDirectory *dir, TSelector *sel);
   virtual ~TEventIter();

   virtual Bool_t GetNextEvent() = 0;
   virtual Bool_t InitRange(Double_t first, Double_t num) = 0;

   static TEventIter *Create(TDSet *dset, TDirectory *dir, TSelector *sel);

   ClassDef(TEventIter,1)  // Event iterator used by TProofPlayer's
};


//------------------------------------------------------------------------

class TEventIterObj : public TEventIter {

private:
   TString  fClassName;    // class name of objects to iterate over
   TList   *fKeys;         // list of keys
   TIter   *fNextKey;      // next key in directory
   TObject *fObj;          // object found

public:
   TEventIterObj();
   TEventIterObj(TDSet *dset, TDirectory *dir, TSelector *sel);
   ~TEventIterObj();

   Bool_t GetNextEvent();
   Bool_t InitRange(Double_t first, Double_t num);

   ClassDef(TEventIterObj,1)  // Event iterator for objects
};


//------------------------------------------------------------------------

class TEventIterTree : public TEventIter {

private:
   TString  fTreeName;  // name of the tree object to iterate over
   TTree   *fTree;      // tree we are iterating over

public:
   TEventIterTree();
   TEventIterTree(TDSet *dset, TDirectory *dir, TSelector *sel);
   ~TEventIterTree();

   Bool_t GetNextEvent();
   Bool_t InitRange(Double_t first, Double_t num);

   ClassDef(TEventIterTree,1)  // Event iterator for Trees
};

#endif
