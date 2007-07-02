// @(#)root/proofplayer:$Name:  $:$Id: TEventIter.cxx,v 1.34 2007/07/02 07:52:40 ganis Exp $
// Author: Maarten Ballintijn   07/01/02

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TEventIter                                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TEventIter.h"
#include "TCollection.h"
#include "TDSet.h"
#include "TFile.h"
#include "TKey.h"
#include "TProofDebug.h"
#include "TSelector.h"
#include "TTimeStamp.h"
#include "TTree.h"
#include "TVirtualPerfStats.h"
#include "TEventList.h"
#include "TMap.h"
#include "TObjString.h"
#include "TRegexp.h"

#include "TError.h"


ClassImp(TEventIter)

//______________________________________________________________________________
TEventIter::TEventIter()
{
   // Default constructor

   fDSet  = 0;
   fElem  = 0;
   fFile  = 0;
   fDir   = 0;
   fSel   = 0;
   fFirst = 0;
   fCur   = -1;
   fNum   = 0;
   fStop  = kFALSE;
   fOldBytesRead = 0;
}

//______________________________________________________________________________
TEventIter::TEventIter(TDSet *dset, TSelector *sel, Long64_t first, Long64_t num)
   : fDSet(dset), fSel(sel)
{
   // Constructor

   fElem  = 0;
   fFile  = 0;
   fDir   = 0;
   fFirst = first;
   fCur   = -1;
   fNum   = num;
   fStop  = kFALSE;
   fEventList = 0;
   fEventListPos = 0;
   fOldBytesRead = 0;
}

//______________________________________________________________________________
TEventIter::~TEventIter()
{
   // Destructor

   delete fFile;
}

//______________________________________________________________________________
void TEventIter::StopProcess(Bool_t /*abort*/)
{
   // Set flag to stop the process

   fStop = kTRUE;
}

//______________________________________________________________________________
TEventIter *TEventIter::Create(TDSet *dset, TSelector *sel, Long64_t first, Long64_t num)
{
   // Create and instance of the appropriate iterator

   if ( dset->IsTree() ) {
      return new TEventIterTree(dset, sel, first, num);
   } else {
      return new TEventIterObj(dset, sel, first, num);
   }
}

//______________________________________________________________________________
Int_t TEventIter::LoadDir()
{
   // Load directory

   Int_t ret = 0;

   // Check Filename
   if ( fFile == 0 || fFilename != fElem->GetFileName() ) {
      fDir = 0;
      delete fFile; fFile = 0;

      fFilename = fElem->GetFileName();

      TDirectory *dirsave = gDirectory;

      Double_t start = 0;
      if (gPerfStats != 0) start = TTimeStamp();

      fFile = TFile::Open(fFilename);

      if (gPerfStats != 0) {
         gPerfStats->FileOpenEvent(fFile, fFilename, double(TTimeStamp())-start);
         fOldBytesRead = 0;
      }

      if (dirsave) dirsave->cd();

      if (!fFile || fFile->IsZombie() ) {
         if (fFile)
            Error("Process","Cannot open file: %s (%s)",
               fFilename.Data(), strerror(fFile->GetErrno()) );
         else
            Error("Process","Cannot open file: %s (errno unavailable)",
               fFilename.Data());
         // cleanup ?
         return -1;
      }
      PDB(kLoop,2) Info("LoadDir","Opening file: %s", fFilename.Data() );
      ret = 1;
   }

   // Check Directory
   if ( fDir == 0 || fPath != fElem->GetDirectory() ) {
      TDirectory *dirsave = gDirectory;

      fPath = fElem->GetDirectory();
      if ( !fFile->cd(fPath) ) {
         Error("Process","Cannot cd to: %s",
            fPath.Data() );
         return -1;
      }
      PDB(kLoop,2) Info("Process","Cd to: %s", fPath.Data() );
      fDir = gDirectory;
      if (dirsave) dirsave->cd();
      ret = 1;
   }

   return ret;
}

//------------------------------------------------------------------------


ClassImp(TEventIterObj)

//______________________________________________________________________________
TEventIterObj::TEventIterObj()
{
   // Default ctor.

   fKeys     = 0;
   fNextKey  = 0;
   fObj      = 0;
}

//______________________________________________________________________________
TEventIterObj::TEventIterObj(TDSet *dset, TSelector *sel, Long64_t first, Long64_t num)
   : TEventIter(dset,sel,first,num)
{
   // Constructor

   fClassName = dset->GetType();
   fKeys     = 0;
   fNextKey  = 0;
   fObj      = 0;
}


//______________________________________________________________________________
TEventIterObj::~TEventIterObj()
{
   // Destructor

   // delete fKeys ?
   delete fNextKey;
   delete fObj;
}

//______________________________________________________________________________
Long64_t TEventIterObj::GetNextEvent()
{
   // Get next event

   if (fStop || fNum == 0) return -1;

   while ( fElem == 0 || fElemNum == 0 || fCur < fFirst-1 ) {

      if (gPerfStats != 0 && fFile != 0) {
         Long64_t bytesRead = fFile->GetBytesRead();
         gPerfStats->SetBytesRead(bytesRead - fOldBytesRead);
         fOldBytesRead = bytesRead;
      }

      fElem = fDSet->Next(fKeys->GetSize());
      if (fElem->GetEventList()) {
         Error("GetNextEvent", "EventLists not implemented");
         return -1;
      }

      if ( fElem == 0 ) {
         fNum = 0;
         return -1;
      }

      Int_t r = LoadDir();

      if ( r == -1 ) {

         // Error has been reported
         fNum = 0;
         return -1;

      } else if ( r == 1 ) {

         // New file and/or directory
         fKeys = fDir->GetListOfKeys();
         fNextKey = new TIter(fKeys);
      }

      // Validate values for this element
      fElemFirst = fElem->GetFirst();
      fElemNum = fElem->GetNum();
      fEventList = fElem->GetEventList();
      fEventListPos = 0;
      if (fEventList) {
         fElemNum = fEventList->GetN();
      }

      Long64_t num = fKeys->GetSize();

      if ( fElemFirst > num ) {
         Error("GetNextEvent","First (%d) higher then number of keys (%d) in %d",
            fElemFirst, num, fElem->GetName() );
         fNum = 0;
         return -1;
      }

      if ( fElemNum == -1 ) {
         fElemNum = num - fElemFirst;
      } else if ( fElemFirst+fElemNum  > num ) {
         Error("GetNextEvent","Num (%d) + First (%d) larger then number of keys (%d) in %s",
            fElemNum, fElemFirst, num, fElem->GetDirectory() );
         fElemNum = num - fElemFirst;
      }

      // Skip this element completely?
      if ( fCur + fElemNum < fFirst ) {
         fCur += fElemNum;
         continue;
      }

      // Position within this element. TODO: more efficient?
      fNextKey->Reset();
      for(fElemCur = -1; fElemCur < fElemFirst-1 ; fElemCur++, fNextKey->Next());
   }

   --fElemNum;
   ++fElemCur;
   --fNum;
   ++fCur;
   TKey *key = (TKey*) fNextKey->Next();
   TDirectory *dirsave = gDirectory;
   fDir->cd();
   fObj = key->ReadObj();
   if (dirsave) dirsave->cd();
   fSel->SetObject( fObj );

   return fElemCur;
}

//------------------------------------------------------------------------

ClassImp(TEventIterTree)

//______________________________________________________________________________
TEventIterTree::TEventIterTree()
{
   // Default ctor.

   fTree = 0;
}

//______________________________________________________________________________
TEventIterTree::TEventIterTree(TDSet *dset, TSelector *sel, Long64_t first, Long64_t num)
   : TEventIter(dset,sel,first,num)
{
   // Constructor

   fTreeName = dset->GetObjName();
   fTree = 0;
   fAcquiredTrees = new TList;
}

//______________________________________________________________________________
TEventIterTree::~TEventIterTree()
{
   // Destructor

   SafeDelete(fTree);
   ReleaseAllTrees();
   SafeDelete(fAcquiredTrees);
}

//______________________________________________________________________________
void TEventIterTree::ReleaseAllTrees()
{
   // Release all acquired trees.

   fAcquiredTrees->SetOwner(kFALSE);
   fAcquiredTrees->Clear();
}

//______________________________________________________________________________
TTree* TEventIterTree::GetTrees(TDSetElement *elem)
{
   // Create a Tree for the main TDSetElement and for all the friends.
   // Returns the main tree or 0 in case of an error.

   TTree* main = Load(elem);
   if (!main)
      return 0;

   fAcquiredTrees->AddFirst(main);

   TList *friends = elem->GetListOfFriends();
   if (friends) {
      TIter nxf(friends);
      TPair *p = 0;
      while ((p = (TPair *) nxf())) {
         TDSetElement *dse = (TDSetElement *) p->Key();
         TObjString *str = (TObjString *) p->Value();
         TTree* friendTree = Load(dse);
         if (friendTree) {
            fAcquiredTrees->AddFirst(friendTree);
            main->AddFriend(friendTree, str->GetName());
         } else {
            return 0;
         }
      }
   }
   return main;
}

//______________________________________________________________________________
TTree* TEventIterTree::Load(TDSetElement *e)
{
   // Load a tree from s TDSetElement

   if (!e) {
      Error("Load","undefined element", e->GetFileName());
      return (TTree *)0;
   }

   const char *fn = e->GetFileName();
   const char *dn = e->GetDirectory();
   const char *tn = e->GetObjName();

   // Open the file
   TFile *f = TFile::Open(fn);
   if (!f) {
      Error("GetTrees","file '%s' could not be open", fn);
      return (TTree *)0;
   }
   // Change dir, if required
   if (dn && !f->cd(dn)) {
      Error("Load","Cannot cd to: %s", dn);
      return (TTree *)0;
   }
   PDB(kLoop,2)
      Info("Load","cd to: %s", dn);

   // If a wild card we will use the first object of the type
   // requested compatible with the reg expression we got
   TString on(tn);
   TString sreg(tn);
   if (sreg.Length() <= 0 || sreg == "" || sreg.Contains("*")) {
      if (sreg.Contains("*"))
         sreg.ReplaceAll("*", ".*");
      else
         sreg = ".*";
      TRegexp re(sreg);
      if (f->GetListOfKeys()) {
         TIter nxk(f->GetListOfKeys());
         TKey *k = 0;
         while ((k = (TKey *) nxk())) {
            if (!strcmp(k->GetClassName(), "TTree")) {
               TString kn(k->GetName());
               if (kn.Index(re) != kNPOS) {
                  on = kn;
                  break;
               }
            }
         }
      }
   }

   // Point to the key
   TKey *key = f->GetKey(on);
   if (key == 0) {
      Error("Load", "Cannot find tree \"%s\" in %s", tn, fn);
      return (TTree*)0;
   }

   PDB(kLoop,2)
      Info("Load", "Reading: %s", tn);
   TDirectory *dirsave = gDirectory;
   f->cd();
   TTree *tree = dynamic_cast<TTree*> (key->ReadObj());
   if (dirsave)
      dirsave->cd();
   if (tree == 0) {
      Error("Load", "Cannot <dynamic_cast> obj to tree \"%s\"", tn);
      return (TTree*)0;
   }

   // Done
   return tree;
}

//______________________________________________________________________________
Long64_t TEventIterTree::GetNextEvent()
{
   // Get next event

   if (fStop || fNum == 0) return -1;

   Bool_t attach = kFALSE;

   while ( fElem == 0 || fElemNum == 0 || fCur < fFirst-1 ) {

      if (gPerfStats != 0 && fTree != 0) {
         Long64_t totBytesRead = fTree->GetCurrentFile()->GetBytesRead();
         Long64_t bytesRead = totBytesRead - fOldBytesRead;
         gPerfStats->SetBytesRead(bytesRead);
         fOldBytesRead = totBytesRead;
      }

      SafeDelete(fElem);
      if (fTree) {
         fElem = fDSet->Next(fTree->GetEntries());
      } else {
         fElem = fDSet->Next();
      }

      if ( fElem == 0 ) {
         fNum = 0;
         return -1;
      }
      ReleaseAllTrees();

      TTree *newTree = GetTrees(fElem);
      if (!newTree) {
         // Error has been reported
         fNum = 0;
         return -1;
      }
      if (newTree != fTree) {
         SafeDelete(fTree);
         fTree = newTree;
         attach = kTRUE;
         fOldBytesRead = fTree->GetCurrentFile()->GetBytesRead();
      }

      // Validate values for this element
      fElemFirst = fElem->GetFirst();
      fElemNum = fElem->GetNum();
      fEventList = fElem->GetEventList();
      fEventListPos = 0;
      if (fEventList)
         fElemNum = fEventList->GetN();

      Long64_t num = (Long64_t) fTree->GetEntries();

      if (!fEventList) {
         if ( fElemFirst > num ) {
            Error("GetNextEvent","First (%d) higher then number of entries (%d) in %s",
               fElemFirst, num, fElem->GetObjName() );
            fNum = 0;
            return -1;
         }
         if ( fElemNum == -1 ) {
            fElemNum = num - fElemFirst;
         } else if ( fElemFirst+fElemNum  > num ) {
            Error("GetNextEvent","Num (%d) + First (%d) larger then number of entries (%d) in %s",
               fElemNum, fElemFirst, num, fElem->GetName() );
            fElemNum = num - fElemFirst;
         }

         // Skip this element completely?
         if ( fCur + fElemNum < fFirst ) {
            fCur += fElemNum;
            continue;
         }
         // Position within this element. TODO: more efficient?
         fElemCur = fElemFirst-1;
      }
   }

   if ( attach ) {
      PDB(kLoop,1) Info("GetNextEvent","Call Init(%p)",fTree);
      fSel->Init(fTree);
      fSel->Notify();
      if (fSel->GetAbort() == TSelector::kAbortProcess) {
         // the error has been reported
         return -1;
      }
      attach = kFALSE;
   }
   if (!fEventList) {
      --fElemNum;
      ++fElemCur;
      --fNum;
      ++fCur;
      return fElemCur;
   }
   else {
      --fElemNum;
      Long64_t rv = fEventList->GetEntry(fEventListPos);
      fEventListPos++;
      return rv;
   }
}
