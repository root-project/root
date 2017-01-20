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
/// This class works in conjuction with TTreeProcessorMP, reacting to messages
/// received from it as specified by the Notify and HandleInput methods.
///
/// \class TMPWorkerTreeFunc
///
/// Templated derivation of TMPWorkerTree handlign generic function tree processing. 
///
/// \class TMPWorkerTreeSel
///
/// Templated derivation of TMPWorkerTree handlign selector tree processing. 
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
              : TMPWorker(nWorkers, maxEntries), fTree(tree), fFile(nullptr),
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
      SendError(errmsg, MPCode::kProcError);
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
      SendError(errmsg, MPCode::kProcError);
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

//////////////////////////////////////////////////////////////////////////
/// Init overload definign max entries

void TMPWorkerTree::Init(int fd, unsigned workerN) {

   TMPWorker::Init(fd, workerN);
   fMaxNEntries = EvalMaxEntries(fMaxNEntries);
}

//////////////////////////////////////////////////////////////////////////
/// Max entries evaluation

ULong64_t TMPWorkerTree::EvalMaxEntries(ULong64_t maxEntries)
{
   // E.g.: when dividing 10 entries between 3 workers, the first
   //       two will process 10/3 == 3 entries, the last one will process
   //       10 - 2*(10/3) == 4 entries.
   if(GetNWorker() < fNWorkers-1)
      return maxEntries/fNWorkers;
   else
      return maxEntries - (fNWorkers-1)*(maxEntries/fNWorkers);
}

//////////////////////////////////////////////////////////////////////////
/// Generic input handling

void TMPWorkerTree::HandleInput(MPCodeBufPair& msg)
{
   unsigned code = msg.first;

   if (code == MPCode::kProcRange
         || code == MPCode::kProcFile
         || code == MPCode::kProcTree) {
      //execute fProcFunc on a file or a range of entries in a file
      Process(code, msg);
   } else if (code == MPCode::kSendResult) {
      //send back result
      SendResult();
   } else {
      //unknown code received
      std::string reply = "S" + std::to_string(GetNWorker());
      reply += ": unknown code received: " + std::to_string(code);
      MPSend(GetSocket(), MPCode::kError, reply.data());
   }
}



//////////////////////////////////////////////////////////////////////////
/// Selector processing SendResult and Process overload

void TMPWorkerTreeSel::SendResult()
{
   //send back result
   fSelector.SlaveTerminate();
   MPSend(GetSocket(), MPCode::kProcResult, fSelector.GetOutputList());
}

/// Selector specialization
void TMPWorkerTreeSel::Process(unsigned int code, MPCodeBufPair& msg)
{
   //evaluate the index of the file to process in fFileNames
   //(we actually don't need the parameter if code == kProcTree)
   unsigned fileN = 0;
   unsigned nProcessed = 0;
   TTree *tree = 0;
   Long64_t start = 0;
   Long64_t finish = 0;
   bool setupcache = true;

   if (code ==  MPCode::kProcTree) {

      // The tree must be defined at this level
      if(fTree == nullptr) {
         std::cout << "tree undefined!\n" ;
         //errors are handled inside RetrieveTree
         return;
      }

      //retrieve the total number of entries ranges processed so far by TPool
      nProcessed = ReadBuffer<unsigned>(msg.second.get());

      //create entries range
      //example: for 21 entries, 4 workers we want ranges 0-5, 5-10, 10-15, 15-21
      //and this worker must take the rangeN-th range
      unsigned nEntries = fTree->GetEntries();
      unsigned nBunch = nEntries / fNWorkers;
      unsigned rangeN = nProcessed % fNWorkers;
      start = rangeN*nBunch + 1;
      if(rangeN < (fNWorkers-1))
         finish = (rangeN+1)*nBunch;
      else
         finish = nEntries;

      //process tree
      tree = fTree;
      CloseFile(); // May not be needed
      if (fTree->GetCurrentFile()) {
         // We need to reopen the file locally (TODO: to understand and fix this)
         if ((fFile = TFile::Open(fTree->GetCurrentFile()->GetName())) && !fFile->IsZombie()) {
            if (!(tree = (TTree *) fFile->Get(fTree->GetName()))) {
               std::string errmsg = "unable to retrieve tree from open file " +
                                    std::string(fTree->GetCurrentFile()->GetName());
               SendError(errmsg);
            }
            fTree = tree;
         } else {
            //errors are handled inside OpenFile
            std::string errmsg = "unable to open file " + std::string(fTree->GetCurrentFile()->GetName());
            SendError(errmsg);
         }
      }

   } else {

      if (code == MPCode::kProcRange) {
         //retrieve the total number of entries ranges processed so far by TPool
         nProcessed = ReadBuffer<unsigned>(msg.second.get());
         //evaluate the file and the entries range to process
         fileN = nProcessed / fNWorkers;
      } else {
         //evaluate the file and the entries range to process
         fileN = ReadBuffer<unsigned>(msg.second.get());
      }

      // Open the file
      fFile = OpenFile(fFileNames[fileN]);
      if (fFile == nullptr) {
         //errors are handled inside OpenFile
         std::string errmsg = "unable to open file " + fFileNames[fileN];
         SendError(errmsg);
         return;
      }

      //retrieve the TTree with the specified name from file
      //we are not the owner of the TTree object, the file is!
      tree = RetrieveTree(fFile);
      if (tree == nullptr) {
         //errors are handled inside RetrieveTree
         std::string errmsg = "unable to retrieve tree from open file " + fFileNames[fileN];
         SendError(errmsg);
         return;
      }

      // Prepare to setup the cache, if required
      setupcache = (tree != fTree) ? true : false;

      // Store as reference
      fTree = tree;

      //create entries range
      if (code == MPCode::kProcRange) {
         //example: for 21 entries, 4 workers we want ranges 0-5, 5-10, 10-15, 15-21
         //and this worker must take the rangeN-th range
         unsigned nEntries = tree->GetEntries();
         unsigned nBunch = nEntries / fNWorkers;
         if(nEntries % fNWorkers) nBunch++;
         unsigned rangeN = nProcessed % fNWorkers;
         start = rangeN*nBunch + 1;
         if(rangeN < (fNWorkers-1))
            finish = (rangeN+1)*nBunch;
         else
            finish = nEntries;
      } else {
         start = 0;
         finish = tree->GetEntries();
      }
   }

   // Setup the cache, if required
   if (setupcache) SetupTreeCache(fTree);

   //check if we are going to reach the max of entries
   //change finish accordingly
   if (fMaxNEntries)
      if (fProcessedEntries + finish - start > fMaxNEntries)
         finish = start + fMaxNEntries - fProcessedEntries;

   if(fFirstEntry){
     fSelector.SlaveBegin(nullptr);
     fFirstEntry = false;
   }

   fSelector.Init(tree);
   fSelector.Notify();
   for(Long64_t entry = start; entry<finish; ++entry) {
      fSelector.Process(entry);
   }

   //update the number of processed entries
   fProcessedEntries += finish - start;

   MPSend(GetSocket(), MPCode::kIdling);

   return;
}

