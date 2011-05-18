// @(#)root/proofplayer:$Id$
// Author: Maarten Ballintijn   07/01/02
// Modified: Long Tran-Thanh    04/09/07  (Addition of TEventIterUnit)

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
// Special iterator class used in TProofPlayer to iterate over events   //
// or objects in the packets.                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TDSet;
class TDSetElement;
class TFile;
class TDirectory;
class TSelector;
class TList;
class TIter;
class TTree;
class TTreeCache;
class TEventList;
class TEntryList;

//------------------------------------------------------------------------

class TEventIter : public TObject {

protected:
   TDSet         *fDSet;         // data set over which to iterate

   TDSetElement  *fElem;         // Current Element

   TString        fFilename;     // Name of the current file
   TFile         *fFile;         // Current file
   Long64_t       fOldBytesRead; // last reported number of bytes read
   TString        fPath;         // Path to current TDirectory
   TDirectory    *fDir;          // directory containing the objects or the TTree
   Long64_t       fElemFirst;    // first entry to process for this element
   Long64_t       fElemNum;      // number of entries to process for this element
   Long64_t       fElemCur;      // current entry for this element

   TSelector     *fSel;          // selector to be used
   Long64_t       fFirst;        // first entry to process
   Long64_t       fNum;          // number of entries to process
   Long64_t       fCur;          // current entry
   Bool_t         fStop;         // termination of run requested
   TEventList    *fEventList;    //! eventList for processing
   Int_t          fEventListPos; //! current position in the eventList
   TEntryList    *fEntryList;    //! entry list for processing
   Long64_t       fEntryListPos; //! current position in the entrylist

   Int_t          LoadDir();     // Load the directory pointed to by fElem

public:
   TEventIter();
   TEventIter(TDSet *dset, TSelector *sel, Long64_t first, Long64_t num);
   virtual ~TEventIter();

   virtual Long64_t  GetCacheSize() = 0;
   virtual Int_t     GetLearnEntries() = 0;
   virtual Long64_t  GetNextEvent() = 0;
   virtual void      StopProcess(Bool_t abort);

   static TEventIter *Create(TDSet *dset, TSelector *sel, Long64_t first, Long64_t num);

   ClassDef(TEventIter,0)  // Event iterator used by TProofPlayer's
};


//------------------------------------------------------------------------

class TEventIterUnit : public TEventIter {

private:
 Long64_t fNum;
 Long64_t fCurrent;


public:
   TEventIterUnit();
   TEventIterUnit(TDSet *dset, TSelector *sel, Long64_t num);
   ~TEventIterUnit() { }

   Long64_t GetCacheSize() {return -1;}
   Int_t    GetLearnEntries() {return -1;}
   Long64_t GetNextEvent();

   ClassDef(TEventIterUnit,0)  // Event iterator for objects
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

   Long64_t GetCacheSize() {return -1;}
   Int_t    GetLearnEntries() {return -1;}
   Long64_t GetNextEvent();

   ClassDef(TEventIterObj,0)  // Event iterator for objects
};


//------------------------------------------------------------------------
class TEventIterTree : public TEventIter {

private:
   TString     fTreeName;     // name of the tree object to iterate over
   TTree      *fTree;         // tree we are iterating over
   TTreeCache *fTreeCache;    // instance of the tree cache for the tree
   Bool_t      fTreeCacheIsLearning; // Whether cache is in learning phase
   Bool_t      fUseTreeCache; // Control usage of the tree cache
   Long64_t    fCacheSize;    // Cache size
   Bool_t      fUseParallelUnzip; // Control usage of parallel unzip
   Bool_t      fDontCacheFiles; // Control OS caching of read files (Mac Os X only)
   TList      *fFileTrees;    // Files && Trees currently open

   // Auxilliary class to keep track open files and loaded trees
   class TFileTree : public TNamed {
   public:
      Bool_t    fUsed;
      Bool_t    fIsLocal;
      TFile    *fFile;
      TList    *fTrees;
      TFileTree(const char *name, TFile *f, Bool_t islocal);
      virtual ~TFileTree();
   };

   TTree* Load(TDSetElement *elem, Bool_t &localfile);
   TTree* GetTrees(TDSetElement *elem);
public:
   TEventIterTree();
   TEventIterTree(TDSet *dset, TSelector *sel, Long64_t first, Long64_t num);
   ~TEventIterTree();

   Long64_t GetCacheSize();
   Int_t    GetLearnEntries();
   Long64_t GetNextEvent();

   ClassDef(TEventIterTree,0)  // Event iterator for Trees
};

#endif
