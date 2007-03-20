// @(#)root/proofplayer:$Name:  $:$Id: TObjectCache.cxx,v 1.3 2006/07/01 11:39:37 rdm Exp $
// Author: M. Biskup, G. Ganis  2/4/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjectCache                                                         //
//                                                                      //
// Manage small caches for TDirectories, TFiles or TTrees using LRU     //
// strategy.                                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObjectCache.h"
#include "TFile.h"
#include "TKey.h"
#include "TProofDebug.h"
#include "TVirtualPerfStats.h"
#include "TTimeStamp.h"
#include "TTree.h"
#include "TRegexp.h"

TFileCache *TFileCache::fgInstance = 0;
TDirectoryCache *TDirectoryCache::fgInstance = 0;

//______________________________________________________________________________
TFileCache::ObjectAndBool_t TFileCache::Load(const TString &fileName)
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
         ::Error("TFileCache::Load","Cannot open file: %s (%s)",
                 fileName.Data(), strerror(file->GetErrno()) );
         delete file;
         file = 0;
      }
      else
         ::Error("TFileCache::Load",
                 "Cannot open file: %s (errno unavailable)", fileName.Data());
      return std::make_pair((TFile*)0, kFALSE);
   }
   PDB(kLoop,2) ::Info("TFileCache::Load","Opening file: %s", fileName.Data());

   return std::make_pair(file, kTRUE);
}

//______________________________________________________________________________
void TFileCache::Unload(TFile* &f)
{
   // Deletes the file. See TObjectCache::Load().

   delete f;
}

//______________________________________________________________________________
TFileCache* TFileCache::Instance()
{
   // Returns an instance (only one in the system) of the class.

   if (fgInstance == 0)
      fgInstance = new TFileCache();
   return fgInstance;
}

//______________________________________________________________________________
TDirectoryCache::ObjectAndBool_t TDirectoryCache::Load(const TCacheKey &k)
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
     :: Error("TDirectoryCache::Load","Cannot cd to: %s", dirName.Data() );
      TFileCache::Instance()->Release(file);
      return make_pair((TDirectory*)0, kFALSE);
   }
   PDB(kLoop,2) ::Info("TDirectoryCache::Load","Cd to: %s", dirName.Data() );

   TDirectory *dir = gDirectory;
   if (dirsave) dirsave->cd();
   fDirectoryFiles[dir] = file;
   return make_pair(dir, kTRUE);
}

//______________________________________________________________________________
void TDirectoryCache::Unload(TDirectory* &dir)
{
   // Releases the file in which the directory was stored.
   // The directory itself is not deleted. It will be deleted when the file
   // is closed.

   TFileCache::Instance()->Release(fDirectoryFiles[dir]);
   fDirectoryFiles.erase(dir);
}

//______________________________________________________________________________
TDirectoryCache* TDirectoryCache::Instance()
{
   // Returns an instance (only one in the system) of the class.

   if (fgInstance == 0)
      fgInstance = new TDirectoryCache();
   return fgInstance;
}

//______________________________________________________________________________
void TTreeFileCache::Unload(TTree* &tree)
{
   // Deleted the tree. Releases the file in which it was stored.

   delete tree;
   TDirectoryCache::Instance()->Release(fTreeDirectories[tree]);
   fTreeDirectories.erase(tree);
}

//______________________________________________________________________________
TTreeFileCache::ObjectAndBool_t TTreeFileCache::Load(const TCacheKey &k)
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

   TString on(treeName);
   TString sreg(treeName);
   // If a wild card we will use the first object of the type
   // requested compatible with the reg expression we got
   if (sreg.Length() <= 0 || sreg == "" || sreg.Contains("*")) {
      if (sreg.Contains("*"))
         sreg.ReplaceAll("*", ".*");
      else
         sreg = ".*";
      TRegexp re(sreg);
      if (dir->GetListOfKeys()) {
         TIter nxk(dir->GetListOfKeys());
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

   TKey *key = dir->GetKey(on);

   if (key == 0) {
      ::Error("TTreeFileCache::Load","Cannot find tree \"%s\" in %s",
            treeName.Data(), fileName.Data());
      TDirectoryCache::Instance()->Release(dir);
      return make_pair((TTree*)0, kFALSE);
   }

   PDB(kLoop,2) ::Info("TTreeFileCache::Load","Reading: %s", treeName.Data() );
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
