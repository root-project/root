// @(#)root/proof:$Name:  $:$Id: TEventIter.cxx,v 1.17 2005/09/16 08:48:38 rdm Exp $
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


TEventIterTree::TFileCache *TEventIterTree::TFileCache::fgInstance = 0;
TEventIterTree::TDirectoryCache *TEventIterTree::TDirectoryCache::fgInstance = 0;


ClassImp(TEventIter)

//______________________________________________________________________________
TEventIter::TEventIter()
{
   fDSet  = 0;
   fElem  = 0;
   fFile  = 0;
   fDir   = 0;
   fSel   = 0;
   fFirst = 0;
   fCur   = -1;
   fNum   = 0;
   fStop  = kFALSE;
}

//______________________________________________________________________________
TEventIter::TEventIter(TDSet *dset, TSelector *sel, Long64_t first, Long64_t num)
   : fDSet(dset), fSel(sel)
{
   fElem  = 0;
   fFile  = 0;
   fDir   = 0;
   fFirst = first;
   fCur   = -1;
   fNum   = num;
   fStop  = kFALSE;
   fEventList = 0;
   fEventListPos = 0;
}

//______________________________________________________________________________
TEventIter::~TEventIter()
{
   delete fFile;
}

//______________________________________________________________________________
void TEventIter::StopProcess(Bool_t /*abort*/)
{
   fStop = kTRUE;
}

//______________________________________________________________________________
TEventIter *TEventIter::Create(TDSet *dset, TSelector *sel, Long64_t first, Long64_t num)
{
   if ( dset->IsTree() ) {
      return new TEventIterTree(dset, sel, first, num);
   } else {
      return new TEventIterObj(dset, sel, first, num);
   }
}

//______________________________________________________________________________
Int_t TEventIter::LoadDir()
{
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
   fClassName = dset->GetType();
   fKeys     = 0;
   fNextKey  = 0;
   fObj      = 0;
}


//______________________________________________________________________________
TEventIterObj::~TEventIterObj()
{
   // delete fKeys ?
   delete fNextKey;
   delete fObj;
}

//______________________________________________________________________________
Long64_t TEventIterObj::GetNextEvent()
{
   if (fStop || fNum == 0) return -1;

   while ( fElem == 0 || fElemNum == 0 || fCur < fFirst-1 ) {

      if (gPerfStats != 0 && fFile != 0) {
         Long64_t bytesRead = fFile->GetBytesRead();
         gPerfStats->SetBytesRead(bytesRead - fOldBytesRead);
         fOldBytesRead = bytesRead;
      }

      fElem = fDSet->Next();
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
   fTreeName = dset->GetObjName();
   fTree = 0;
}

//______________________________________________________________________________
TEventIterTree::~TEventIterTree()
{
   ReleaseAllTrees();
}

//______________________________________________________________________________
void TEventIterTree::ReleaseAllTrees() {
   // release all acquired trees.
   for (std::list<TTree*>::iterator i = fAcquiredTrees.begin(); i != fAcquiredTrees.end(); ++i) {
      fTreeCache.Release(*i);
   }
   fAcquiredTrees.clear();
}

//______________________________________________________________________________
TTree* TEventIterTree::GetTrees(TDSetElement *elem)
{
   // Create a Tree for the main TDSetElement and for all the friends.
   // Returns the main tree or 0 in case of an error.

   TTree* main = fTreeCache.Acquire(elem->GetFileName(),
                                  elem->GetDirectory(), elem->GetObjName());
   if (!main)
      return 0;
   fAcquiredTrees.push_front(main);

   TDSetElement::FriendsList_t* friends = elem->GetListOfFriends();
   for (TDSetElement::FriendsList_t::iterator i = friends->begin();
                i != friends->end(); ++i) {
      TTree* friendTree = fTreeCache.Acquire(i->first->GetFileName(),
                                  i->first->GetDirectory(), i->first->GetObjName());
      if (friendTree) {
         fAcquiredTrees.push_front(friendTree);
         main->AddFriend(friendTree, i->second);
      }
      else {
         ReleaseAllTrees();
         return 0;
      }
   }
   return main;
}

//______________________________________________________________________________
Long64_t TEventIterTree::GetNextEvent()
{

   if (fStop || fNum == 0) return -1;

   Bool_t attach = kFALSE;

   while ( fElem == 0 || fElemNum == 0 || fCur < fFirst-1 ) {

      if (gPerfStats != 0 && fFile != 0) {
         Long64_t bytesRead = fFile->GetBytesRead();
         gPerfStats->SetBytesRead(bytesRead - fOldBytesRead);
         fOldBytesRead = bytesRead;
      }

      fElem = fDSet->Next();

      if ( fElem == 0 ) {
         fNum = 0;
         return -1;
      }
      ReleaseAllTrees();

      fTree = GetTrees(fElem);
      if (!fTree) {
         // Error has been reported
         fNum = 0;
         return -1;
      }
      attach = kTRUE;

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
      fSel->Init( fTree );
      if( !fSel->Notify()) {
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
      int rv = fEventList->GetEntry(fEventListPos);
      fEventListPos++;
      return rv;
   }
}

//______________________________________________________________________________
TEventIterTree::TFileCache::ObjectAndBool_t TEventIterTree::TFileCache::Load(const TString &fileName)
{
   // Loads a file given its filename. See TObjectCache::Load().

   TDirectory *dirsave = gDirectory;

   Double_t start = 0;
   if (gPerfStats != 0) start = TTimeStamp();

   TFile* file = TFile::Open(fileName.Data());

   if (gPerfStats != 0) {
      gPerfStats->FileOpenEvent(file, fileName, double(TTimeStamp())-start);
   }

   if (dirsave) dirsave->cd();

   if (!file || file->IsZombie() ) {
      if (file) {
         ::Error("TEventIterTree::TFileCache::Load","Cannot open file: %s (%s)",
                 fileName.Data(), strerror(file->GetErrno()) );
         delete file;
         file = 0;
      }
      else
         ::Error("TEventIterTree::TFileCache::Load",
                 "Cannot open file: %s (errno unavailable)", fileName.Data());
      return std::make_pair((TFile*)0, kFALSE);
   }
   PDB(kLoop,2) ::Info("TEventIterTree::TFileCache::Load","Opening file: %s", fileName.Data());

   return std::make_pair(file, kTRUE);
}

//______________________________________________________________________________
void TEventIterTree::TFileCache::Unload(TFile* &f)
{
   // Deletes the file. See TObjectCache::Load().

   delete f;
}

//______________________________________________________________________________
TEventIterTree::TFileCache* TEventIterTree::TFileCache::Instance()
{
   // Returns an instance (only one in the system) of the class.

   if (fgInstance == 0)
      fgInstance = new TFileCache();
   return fgInstance;
}

//______________________________________________________________________________
TDirectory* TEventIterTree::TDirectoryCache::Acquire(const TString& fileName, const TString& dirName)
{
   // Included for more clear syntax. See TObjectCache::Acquire.

   return TObjectCache<TCacheKey, TCacheObject>
               ::Acquire(std::make_pair(fileName, dirName));
}

//______________________________________________________________________________
TEventIterTree::TDirectoryCache::ObjectAndBool_t TEventIterTree::TDirectoryCache::Load(const TCacheKey &k)
{
   // Loads a directory given the file name and the directory name.
   // See TObjectCache::Acquire().

   const TString fileName = k.first;
   const TString dirName = k.second;
   using namespace std;
   TDirectory *dirsave = gDirectory;

   TFile *file = TFileCache::Instance()->Acquire(fileName);
   if (!file)
      return make_pair((TDirectory*)0, kFALSE);

   if (!file->cd(dirName)) {
     :: Error("TEventIterTree::TDirectoryCache::Load","Cannot cd to: %s", dirName.Data() );
      TFileCache::Instance()->Release(file);
      return make_pair((TDirectory*)0, kFALSE);
   }
   PDB(kLoop,2) ::Info("TEventIterTree::TDirectoryCache::Load","Cd to: %s", dirName.Data() );

   TDirectory *dir = gDirectory;
   if (dirsave) dirsave->cd();
   fDirectoryFiles[dir] = file;
   return make_pair(dir, kTRUE);
}

//______________________________________________________________________________
void TEventIterTree::TDirectoryCache::Unload(TDirectory* &dir)
{
   // Releases the file in which the directory was stored.
   // The directory itself is not deleted. It will be deleted when the file
   // is closed.

   TFileCache::Instance()->Release(fDirectoryFiles[dir]);
   fDirectoryFiles.erase(dir);
}

//______________________________________________________________________________
TEventIterTree::TDirectoryCache* TEventIterTree::TDirectoryCache::Instance()
{
   // Returns an instance (only one in the system) of the class.

   if (fgInstance == 0)
      fgInstance = new TDirectoryCache();
   return fgInstance;
}

//______________________________________________________________________________
void TEventIterTree::TTreeCache::Unload(TTree* &tree)
{
   // Deleted the tree. Releases the file in which it was stored.

   delete tree;
   TDirectoryCache::Instance()->Release(fTreeDirectories[tree]);
   fTreeDirectories.erase(tree);
}

//______________________________________________________________________________
TTree* TEventIterTree::TTreeCache::Acquire(const TString& fileName, const TString& dirName, const TString& treeName)
{
   // Included for more clear syntax. See TObjectCache::Acquire.

   return TObjectCache<TCacheKey, TCacheObject>
               ::Acquire(std::make_pair(fileName, std::make_pair(dirName, treeName)));
}

//______________________________________________________________________________
TEventIterTree::TTreeCache::ObjectAndBool_t TEventIterTree::TTreeCache::Load(const TCacheKey &k)
{
   // Loads a tree given the file name where it's stored, the directory name in the file
   // and the tree name. See TObjectCache::Acquire().

   const TString fileName = k.first;
   const TString dirName = k.second.first;
   const TString treeName = k.second.second;
   using namespace std;

   TDirectory *dir = TDirectoryCache::Instance()->Acquire(fileName, dirName);
   if (!dir)
      return make_pair((TTree*)0, kFALSE);

   TKey *key = dir->GetKey(treeName);

   if (key == 0) {
      ::Error("TEventIterTree::TTreeCache::Load","Cannot find tree \"%s\" in %s",
            treeName.Data(), fileName.Data());
      TDirectoryCache::Instance()->Release(dir);
      return make_pair((TTree*)0, kFALSE);
   }

   PDB(kLoop,2) ::Info("TEventIterTree::TTreeCache::Load","Reading: %s", treeName.Data() );
   TDirectory *dirsave = gDirectory;
   dir->cd();
   TTree *tree = dynamic_cast<TTree*> (key->ReadObj());
   if (dirsave) dirsave->cd();
   if (tree == 0) {
      TDirectoryCache::Instance()->Release(dir);
      return make_pair((TTree*)0, kFALSE);
   }
   fTreeDirectories[tree] = dir;
   return make_pair(tree, kTRUE);
}
