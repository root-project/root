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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TEventIter                                                           //
//                                                                      //
// Special iterator class used in TProofPlayer to iterate over events   //
// or objects in the packets.                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TEnv.h"
#include "TEventIter.h"
#include "TCollection.h"
#include "TDSet.h"
#include "TFile.h"
#include "TKey.h"
#include "TProofDebug.h"
#include "TSelector.h"
#include "TTimeStamp.h"
#include "TTree.h"
#include "TTreeCache.h"
#include "TVirtualPerfStats.h"
#include "TEventList.h"
#include "TEntryList.h"
#include "TList.h"
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
   fEventList = 0;
   fEntryList = 0;
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
   fEntryList = 0;
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

   if (dset->TestBit(TDSet::kEmpty)) {
      return new TEventIterUnit(dset, sel, num);
   }  else if (dset->IsTree()) {
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

      // Take into acoount possible prefixes
      TFile::EFileType typ = TFile::kDefault;
      TString fname = gEnv->GetValue("Path.Localroot","");
      if (!fname.IsNull())
         typ = TFile::GetType(fFilename, "", &fname);
      if (typ != TFile::kLocal)
         fname = fFilename;
      fFile = TFile::Open(fname);

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


ClassImp(TEventIterUnit)

//______________________________________________________________________________
TEventIterUnit::TEventIterUnit()
{
   // Default constructor

   fDSet = 0;
   fElem = 0;
   fSel = 0;
   fNum = 0;
   fCurrent = 0;
   fStop = kFALSE;
}

//______________________________________________________________________________
TEventIterUnit::TEventIterUnit(TDSet* dset, TSelector *sel, Long64_t num)
{
   // Main constructor

   fDSet = dset;
   fElem = 0;
   fSel = sel;
   fNum = num;
   fCurrent = 0;
   fStop = kFALSE;
}

//______________________________________________________________________________
Long64_t TEventIterUnit::GetNextEvent()
{
   // Get next event

   if (fStop || fNum == 0)
      return -1;

   while (fElem == 0 || fCurrent == 0) {

      if (!(fElem = fDSet->Next()))
         return -1;

      if (!fElem->TestBit(TDSetElement::kEmpty)) {
         Error("GetNextEvent", "data element must be set to kEmtpy");
         return -1;
      }

      fNum = fElem->GetNum();
      if (!(fCurrent = fNum)) {
         fNum = 0;
         return -1;
      }
   }
   --fCurrent;
   return fCurrent;
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
      if (fElem->GetEntryList()) {
         Error("GetNextEvent", "Entry- or event-list not available");
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
      fEntryList = dynamic_cast<TEntryList *>(fElem->GetEntryList());
      fEventList = (fEntryList) ? (TEventList *)0
                                : dynamic_cast<TEventList *>(fElem->GetEntryList());
      fEventListPos = 0;
      if (fEventList)
         fElemNum = fEventList->GetN();

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
      for(fElemCur = -1; fElemCur < fElemFirst-1 ; fElemCur++, fNextKey->Next()) { }
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

//______________________________________________________________________________
TEventIterTree::TFileTree::TFileTree(const char *name, TFile *f, Bool_t islocal)
               : TNamed(name, ""), fUsed(kFALSE), fIsLocal(islocal), fFile(f)
{
   // Default ctor.

   fTrees = new TList;
   fTrees->SetOwner();
}
//______________________________________________________________________________
TEventIterTree::TFileTree::~TFileTree()
{
   // Default cdtor.

   // Avoid destroying the cache; must be placed before deleting the trees
   fFile->SetCacheRead(0);
   SafeDelete(fTrees);
   SafeDelete(fFile);
}

ClassImp(TEventIterTree)

//______________________________________________________________________________
TEventIterTree::TEventIterTree()
{
   // Default ctor.

   fTree = 0;
   fTreeCache = 0;
}

//______________________________________________________________________________
TEventIterTree::TEventIterTree(TDSet *dset, TSelector *sel, Long64_t first, Long64_t num)
   : TEventIter(dset,sel,first,num)
{
   // Constructor

   fTreeName = dset->GetObjName();
   fTree = 0;
   fTreeCache = 0;
   fFileTrees = new TList;
   fFileTrees->SetOwner();
   fUseTreeCache = gEnv->GetValue("ProofPlayer.UseTreeCache", 1);
   fCacheSize = gEnv->GetValue("ProofPlayer.CacheSize", 10000000);
}

//______________________________________________________________________________
TEventIterTree::~TEventIterTree()
{
   // Destructor

   // The cache is deleted in here
   SafeDelete(fFileTrees);
   SafeDelete(fTreeCache);
}

//______________________________________________________________________________
TTree* TEventIterTree::GetTrees(TDSetElement *elem)
{
   // Create a Tree for the main TDSetElement and for all the friends.
   // Returns the main tree or 0 in case of an error.

   // Reset used flags
   TIter nxft(fFileTrees);
   TFileTree *ft = 0;
   while ((ft = (TFileTree *)nxft()))
      ft->fUsed = kFALSE;

   Bool_t localfile = kFALSE;
   TTree* main = Load(elem, localfile);

   if (main && main != fTree) {
      // Set the file cache
      if (!localfile) {
         if (fUseTreeCache) {
            TFile *curfile = main->GetCurrentFile();
            if (!fTreeCache) {
               main->SetCacheSize(fCacheSize);
               fTreeCache = (TTreeCache *)curfile->GetCacheRead();
            } else {
               curfile->SetCacheRead(fTreeCache);
               main->SetCacheSize(fCacheSize);
               fTreeCache->UpdateBranches(main, kTRUE);
            }
         } else {
            // Disable the cache
            main->SetCacheSize(-1);
         }
      } else {
         fTreeCache = 0;
      }

      Bool_t loc = kFALSE;
      // Also the friends
      TList *friends = elem->GetListOfFriends();
      if (friends) {
         TIter nxf(friends);
         TPair *p = 0;
         while ((p = (TPair *) nxf())) {
            TDSetElement *dse = (TDSetElement *) p->Key();
            TObjString *str = (TObjString *) p->Value();
            TTree* friendTree = Load(dse, loc);
            if (friendTree) {
               main->AddFriend(friendTree, str->GetName());
            } else {
               return 0;
            }
         }
      }
   }

   // Remove instances not used
   nxft.Reset();
   while ((ft = (TFileTree *)nxft())) {
      if (!(ft->fUsed)) {
         fFileTrees->Remove(ft);
         delete ft;
      }
   }

   // Done, successfully or not
   return main;
}

//______________________________________________________________________________
TTree* TEventIterTree::Load(TDSetElement *e, Bool_t &localfile)
{
   // Load a tree from s TDSetElement

   if (!e) {
      Error("Load","undefined element", e->GetFileName());
      return (TTree *)0;
   }

   const char *fn = e->GetFileName();
   const char *dn = e->GetDirectory();
   const char *tn = e->GetObjName();

   TFile *f = 0;

   // Check if the file is already open
   TString names(fn);
   TString name;
   Ssiz_t from = 0;
   TFileTree *ft = 0;
   while (names.Tokenize(name,from,"|")) {
      TString key(TUrl(name).GetFileAndOptions());
      if ((ft = (TFileTree *) fFileTrees->FindObject(key.Data()))) {
         f = ft->fFile;
         break;
      }
   }

   // Open the file, if needed
   if (!f) {
      TFile::EFileType typ = TFile::kDefault;
      TString fname = gEnv->GetValue("Path.Localroot","");
      if (!fname.IsNull())
         typ = TFile::GetType(fn, "", &fname);
      if (typ != TFile::kLocal) {
         fname = fn;
      } else {
         localfile = kTRUE;
      }

      // Open the file
      f = TFile::Open(fname);
      if (!f) {
         Error("GetTrees","file '%s' ('%s') could not be open", fn, fname.Data());
         return (TTree *)0;
      }

      // Create TFileTree instance in the list
      ft = new TFileTree(TUrl(f->GetName()).GetFileAndOptions(), f, localfile);
      fFileTrees->Add(ft);
   } else {
      // Fill locality boolean
      localfile = (ft) ? ft->fIsLocal : localfile;
   }

   // Check if the tree is already loaded
   if (ft && ft->fTrees->GetSize() > 0) {
      TTree *t = 0;
      if (!strcmp(tn, "*"))
         t = (TTree *) ft->fTrees->First();
      else
         t = (TTree *) ft->fTrees->FindObject(tn);
      if (t) {
         ft->fUsed = kTRUE;
         return t;
      }
   }

   TDirectory *dd = f;
   // Change dir, if required
   if (dn && !(dd = f->GetDirectory(dn))) {
      Error("Load","Cannot get to: %s", dn);
      return (TTree *)0;
   }
   PDB(kLoop,2)
      Info("Load","got directory: %s", dn);

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
      if (dd->GetListOfKeys()) {
         TIter nxk(dd->GetListOfKeys());
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
   TKey *key = dd->GetKey(on);
   if (key == 0) {
      Error("Load", "Cannot find tree \"%s\" in %s", tn, fn);
      return (TTree*)0;
   }

   PDB(kLoop,2)
      Info("Load", "Reading: %s", tn);

   TTree *tree = dynamic_cast<TTree*> (key->ReadObj());
   dd->cd();

   if (tree == 0) {
      Error("Load", "Cannot <dynamic_cast> obj to tree \"%s\"", tn);
      return (TTree*)0;
   }

   // Add track in the cache
   ft->fTrees->Add(tree);
   ft->fUsed = kTRUE;

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
      while (!fElem) {
         if (fTree) {
            fElem = fDSet->Next(fTree->GetEntries());
         } else {
            fElem = fDSet->Next();
         }

         if (!fElem) {
            // End of processing
            fNum = 0;
            return -1;
         }

         TTree *newTree = GetTrees(fElem);
         if (newTree) {
            if (newTree != fTree) {
               // The old tree is wonwd by TFileTree and will be deleted there
               fTree = newTree;
               attach = kTRUE;
               fOldBytesRead = fTree->GetCurrentFile()->GetBytesRead();
            }
            // Set range to be analysed
            if (fTreeCache)
               fTreeCache->SetEntryRange(fElem->GetFirst(),
                                         fElem->GetFirst() + fElem->GetNum() - 1);
         } else {
            // Could not open this element: ask for another one
            SafeDelete(fElem);
            // The current tree, if any, is not valid anymore
            fTree = 0;
         }
      }

      // Validate values for this element
      fElemFirst = fElem->GetFirst();
      fElemNum = fElem->GetNum();
      fEntryList = dynamic_cast<TEntryList *>(fElem->GetEntryList());
      fEventList = (fEntryList) ? (TEventList *)0
                                : dynamic_cast<TEventList *>(fElem->GetEntryList());
      fEntryListPos = fElemFirst;
      fEventListPos = 0;
      if (fEntryList)
         fElemNum = fEntryList->GetEntriesToProcess();
      else if (fEventList)
         fElemNum = fEventList->GetN();

      Long64_t num = (Long64_t) fTree->GetEntries();

      if (!fEntryList && !fEventList) {
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
      PDB(kLoop,1) Info("GetNextEvent","Call Init(%p) and Notify()",fTree);
      fSel->Init(fTree);
      fSel->Notify();
      TIter next(fSel->GetOutputList());
      TEntryList *elist=0;
      while ((elist=(TEntryList*)next())){
         if (elist->InheritsFrom(TEntryList::Class()))
            elist->SetTree(fTree->GetName(), fElem->GetFileName());
      }
      if (fSel->GetAbort() == TSelector::kAbortProcess) {
         // the error has been reported
         return -1;
      }
      attach = kFALSE;
   }
   Long64_t rv;

   if (fEntryList){
      --fElemNum;
      rv = fEntryList->GetEntry(fEntryListPos);
      fEntryListPos++;
   } else if (fEventList) {
      --fElemNum;
      rv = fEventList->GetEntry(fEventListPos);
      fEventListPos++;
   } else {
      --fElemNum;
      ++fElemCur;
      --fNum;
      ++fCur;
      rv = fElemCur;
   }

   // For prefetching
   fTree->LoadTree(rv);

   return rv;
}
