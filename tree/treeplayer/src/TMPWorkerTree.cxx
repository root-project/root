/* @(#)root/multiproc:$Id$ */
// Author: Enrico Guiraud July 2015
// Modified: G Ganis Jan 2017

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "MPCode.h"
#include "MPSendRecv.h"
#include "TError.h"
#include "TMPWorkerTree.h"
#include "TSystem.h"
#include "TEnv.h"
#include <string>

//////////////////////////////////////////////////////////////////////////
///
/// \class TMPWorkerTree
///
/// This class works in conjuction with TMPClient, reacting to messages
/// received from it as specified by the Notify and HandleInput methods.
/// When TMPClient::Fork is called, a TMPWorker instance is passed to it
/// which will take control of the ROOT session in the children processes.
///
/// After forking, every time a message is sent or broadcast to the workers,
/// TMPWorker::Notify is called and the message is retrieved.
/// Messages exchanged between TMPClient and TMPWorker should be sent with
/// the MPSend() standalone function.\n
/// If the code of the message received is above 1000 (i.e. it is an MPCode)
/// the qualified TMPWorker::HandleInput method is called, that takes care
/// of handling the most generic type of messages. Otherwise the unqualified
/// (possibly overridden) version of HandleInput is called, allowing classes
/// that inherit from TMPWorker to manage their own protocol.\n
/// An application's worker class should inherit from TMPWorker and implement
/// a HandleInput method that overrides TMPWorker's.\n
///
//////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////////
/// Class constructors.
/// Note that this does not set variables like fPid or fS (worker's socket).\n
/// These operations are handled by the Init method, which is called after
/// forking.\n
/// This separation is in place because the instantiation of a worker
/// must be done once _before_ forking, while the initialization of the
/// members must be done _after_ forking by each of the children processes.
TMPWorkerTree::TMPWorkerTree()
              : TMPWorker(), fFileNames(), fTreeName(), fTree(nullptr), fFile(nullptr),
                fTreeCache(0), fTreeCacheIsLearning(kFALSE),
                fUseTreeCache(kTRUE), fCacheSize(-1)
{
   Setup();
}

TMPWorkerTree::TMPWorkerTree(const std::vector<std::string>& fileNames,
                             const std::string& treeName,
                             unsigned nWorkers, ULong64_t maxEntries)
              : TMPWorker(nWorkers, maxEntries),
                fFileNames(fileNames), fTreeName(treeName), fTree(nullptr), fFile(nullptr),
                fTreeCache(0), fTreeCacheIsLearning(kFALSE),
                fUseTreeCache(kTRUE), fCacheSize(-1)
{
   Setup();
}

TMPWorkerTree::TMPWorkerTree(TTree *tree, unsigned nWorkers, ULong64_t maxEntries)
              : TMPWorker(nWorkers, maxEntries), fTree(tree),
                fTreeCache(0), fTreeCacheIsLearning(kFALSE),
                fUseTreeCache(kTRUE), fCacheSize(-1)
{
   Setup();
}

TMPWorkerTree::~TMPWorkerTree()
{
   // Properly close the open file, if any
   CloseFile();
}

//////////////////////////////////////////////////////////////////////////
/// Auxilliary method for common initializations
void TMPWorkerTree::Setup()
{
  Int_t uc = gEnv->GetValue("MultiProc.UseTreeCache", 0);
  if (uc != 1) fUseTreeCache = kFALSE;
  fCacheSize = gEnv->GetValue("MultiProc.CacheSize", -1);
}

//////////////////////////////////////////////////////////////////////////
/// Handle file closing.

void TMPWorkerTree::CloseFile()
{
   // Avoid destroying the cache; must be placed before deleting the trees
   if (fFile) {
      if (fTree) fFile->SetCacheRead(0, fTree);
      delete fFile ;
      fFile = 0;
   }
}

//////////////////////////////////////////////////////////////////////////
/// Handle file opening.

TFile *TMPWorkerTree::OpenFile(const std::string& fileName)
{

   TFile *fp = TFile::Open(fileName.c_str());
   if (fp == nullptr || fp->IsZombie()) {
      std::stringstream ss;
      ss << "could not open file " << fileName;
      std::string errmsg = ss.str();
      SendError(errmsg, PoolCode::kProcError);
      return nullptr;
   }

   return fp;
}

//////////////////////////////////////////////////////////////////////////
/// Retrieve a tree from an open file.

TTree *TMPWorkerTree::RetrieveTree(TFile *fp)
{
   //retrieve the TTree with the specified name from file
   //we are not the owner of the TTree object, the file is!
   TTree *tree = nullptr;
   if(fTreeName == "") {
      // retrieve the first TTree
      // (re-adapted from TEventIter.cxx)
      if (fp->GetListOfKeys()) {
         for(auto k : *fp->GetListOfKeys()) {
            TKey *key = static_cast<TKey*>(k);
            if (!strcmp(key->GetClassName(), "TTree") || !strcmp(key->GetClassName(), "TNtuple"))
               tree = static_cast<TTree*>(fp->Get(key->GetName()));
         }
      }
   } else {
      tree = static_cast<TTree*>(fp->Get(fTreeName.c_str()));
   }
   if (tree == nullptr) {
      std::stringstream ss;
      ss << "cannot find tree with name " << fTreeName << " in file " << fp->GetName();
      std::string errmsg = ss.str();
      SendError(errmsg, PoolCode::kProcError);
      return nullptr;
   }

   return tree;
}

//////////////////////////////////////////////////////////////////////////
/// Tree cache handling

void TMPWorkerTree::SetupTreeCache(TTree *tree)
{
   if (fUseTreeCache) {
      TFile *curfile = tree->GetCurrentFile();
      if (curfile) {
         if (!fTreeCache) {
            tree->SetCacheSize(fCacheSize);
            fTreeCache = (TTreeCache *)curfile->GetCacheRead(tree);
            if (fCacheSize < 0) fCacheSize = tree->GetCacheSize();
         } else {
            fTreeCache->ResetCache();
            curfile->SetCacheRead(fTreeCache, tree);
            fTreeCache->UpdateBranches(tree);
         }
         if (fTreeCache) {
            fTreeCacheIsLearning = fTreeCache->IsLearning();
            if (fTreeCacheIsLearning)
               Info("SetupTreeCache","the tree cache is in learning phase");
         }
      } else {
         Warning("SetupTreeCache", "default tree does not have a file attached: corruption? Tree cache untouched");
      }
   } else {
      // Disable the cache
      tree->SetCacheSize(0);
   }
}
