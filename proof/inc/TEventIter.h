// @(#)root/proof:$Name:  $:$Id: TEventIter.h,v 1.3 2002/03/13 01:52:20 rdm Exp $
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

typedef long Long64_t;

class TDSet;
class TDSetElement;
class TFile;
class TDirectory;
class TSelector;
class TList;
class TIter;
class TTree;


//------------------------------------------------------------------------

class TEventIter : public TObject {

protected:
   TDSet         *fDSet;      // data set over which to iterate

   TDSetElement  *fElem;      // Current Element

   TString        fFilename;  // Name of the current file
   TFile         *fFile;      // Current file
   TString        fPath;      // Path to current TDirectory
   TDirectory    *fDir;       // directory containing the objects or the TTree
   Long64_t       fElemFirst; // first entry to process for this element
   Long64_t       fElemNum;   // number of entries to process for this element
   Long64_t       fElemCur;   // current entry for this element

   TSelector     *fSel;       // selector to be used
   Long64_t       fFirst;     // first entry to process
   Long64_t       fNum;       // number of entries to process
   Long64_t       fCur;       // current entry

   Int_t    LoadDir();        // Load the directory pointed to by fElem

public:
   TEventIter();
   TEventIter(TDSet *dset, TSelector *sel, Long64_t first, Long64_t num);
   virtual ~TEventIter();

   virtual Long64_t  GetNextEvent() = 0;

   static TEventIter *Create(TDSet *dset, TSelector *sel, Long64_t first, Long64_t num);

   ClassDef(TEventIter,0)  // Event iterator used by TProofPlayer's
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
   TEventIterObj(TDSet *dset, TSelector *sel, Long64_t first, Long64_t num);
   ~TEventIterObj();

   Long64_t GetNextEvent();

   ClassDef(TEventIterObj,0)  // Event iterator for objects
};


//------------------------------------------------------------------------

class TEventIterTree : public TEventIter {

private:
   TString  fTreeName;  // name of the tree object to iterate over
   TTree   *fTree;      // tree we are iterating over

public:
   TEventIterTree();
   TEventIterTree(TDSet *dset, TSelector *sel, Long64_t first, Long64_t num);
   ~TEventIterTree();

   Long64_t GetNextEvent();

   ClassDef(TEventIterTree,0)  // Event iterator for Trees
};

#endif
