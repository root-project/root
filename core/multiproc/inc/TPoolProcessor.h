/* @(#)root/multiproc:$Id$ */
// Author: Enrico Guiraud September 2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPoolProcessor
#define ROOT_TPoolProcessor

#include "TMPWorker.h"
#include "PoolUtils.h"
#include "MPCode.h"
#include "MPSendRecv.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TEventList.h"
#include "TEntryList.h"
#include "TTree.h"
#include "TFile.h"
#include "TKey.h"
#include "TH1.h"
#include <memory>
#include <string>
#include <sstream>
#include <type_traits> //std::result_of


//If the user lambda returns a TH1F*, TTree*, TEventList*, we incur in the
//problem of that object being automatically owned by the current open file.
//For these three types, we call SetDirectory(nullptr) to detach the returned
//object from the file we are reading the TTree from.
//Note: the only sane case in which this should happen is when a TH1F* is
//returned.
template<class T, typename std::enable_if<std::is_pointer<T>::value && std::is_constructible<TObject*, T>::value>::type* = nullptr>
void DetachRes(T res)
{
   auto th1p = dynamic_cast<TH1*>(res);
   if(th1p != nullptr) {
      th1p->SetDirectory(nullptr);
      return;
   }

   auto ttreep = dynamic_cast<TTree*>(res);
   if(ttreep != nullptr) {
      ttreep->SetDirectory(nullptr);
      return;
   }

   auto tentrylist = dynamic_cast<TEntryList*>(res);
   if(tentrylist != nullptr) {
      tentrylist->SetDirectory(nullptr);
      return;
   }

   auto teventlist = dynamic_cast<TEventList*>(res);
   if(teventlist != nullptr) {
      teventlist->SetDirectory(nullptr);
      return;
   }

   return;
}


template<class F>
class TPoolProcessor : public TMPWorker {
public:
   TPoolProcessor(F procFunc, const std::vector<std::string>& fileNames, const std::string& treeName, unsigned nWorkers, ULong64_t maxEntries);
   TPoolProcessor(F procFunc, TTree *tree, unsigned nWorkers, ULong64_t maxEntries);
   ~TPoolProcessor() {}

   void HandleInput(MPCodeBufPair& msg); ///< Execute instructions received from a TPool client
   void Init(int fd, unsigned workerN);

private:
   void Process(unsigned code, MPCodeBufPair& msg);
   TFile *OpenFile(const std::string& fileName);
   TTree *RetrieveTree(TFile *fp);
   ULong64_t EvalMaxEntries(ULong64_t maxEntries);

   F fProcFunc; ///< the function to be executed
   std::vector<std::string> fFileNames; ///< the files to be processed by all workers
   std::string fTreeName; ///< the name of the tree to be processed
   TTree *fTree; ///< pointer to the tree to be processed. It is only used if the tree is directly passed to TProcPool::Process as argument
   unsigned fNWorkers; ///< the number of workers spawned
   ULong64_t fMaxNEntries; ///< the maximum number of entries to be processed by this worker
   ULong64_t fProcessedEntries; ///< the number of entries processed by this worker so far
   typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type fReducedResult; ///< the results of the executions of fProcFunc merged together
   bool fCanReduce; ///< true if fReducedResult can be reduced with a new result, false until we have produced one result
};


template<class F>
TPoolProcessor<F>::TPoolProcessor(F procFunc, const std::vector<std::string>& fileNames, const std::string& treeName, unsigned nWorkers, ULong64_t maxEntries) : TMPWorker(), fProcFunc(procFunc),
   fFileNames(fileNames), fTreeName(treeName), fTree(nullptr),
   fNWorkers(nWorkers), fMaxNEntries(maxEntries),
   fProcessedEntries(0), fReducedResult(), fCanReduce(false)
{}


template<class F>
TPoolProcessor<F>::TPoolProcessor(F procFunc, TTree *tree, unsigned nWorkers, ULong64_t maxEntries) :
   TMPWorker(), fProcFunc(procFunc),
   fFileNames(), fTreeName(), fTree(tree),
   fNWorkers(nWorkers), fMaxNEntries(maxEntries),
   fProcessedEntries(0), fReducedResult(), fCanReduce(false)
{}


template<class F>
void TPoolProcessor<F>::HandleInput(MPCodeBufPair& msg)
{
   unsigned code = msg.first;

   if (code == PoolCode::kProcRange
         || code == PoolCode::kProcFile
         || code == PoolCode::kProcTree) {
      //execute fProcFunc on a file or a range of entries in a file
      Process(code, msg);
   } else if (code == PoolCode::kSendResult) {
      //send back result
      MPSend(GetSocket(), PoolCode::kProcResult, fReducedResult);
   } else {
      //unknown code received
      std::string reply = "S" + std::to_string(GetNWorker());
      reply += ": unknown code received: " + std::to_string(code);
      MPSend(GetSocket(), MPCode::kError, reply.data());
   }
}


template<class F>
void TPoolProcessor<F>::Init(int fd, unsigned workerN) {
   TMPWorker::Init(fd, workerN);

   fMaxNEntries = EvalMaxEntries(fMaxNEntries);
}


template<class F>
void TPoolProcessor<F>::Process(unsigned code, MPCodeBufPair& msg)
{
   //evaluate the index of the file to process in fFileNames
   //(we actually don't need the parameter if code == kProcTree)
   unsigned fileN = 0;
   unsigned nProcessed = 0;
   if (code == PoolCode::kProcRange || code == PoolCode::kProcTree) {
      if (code == PoolCode::kProcTree && !fTree) {
         // This must be defined
         std::cerr << "[S]: Process:kProcTree fTree undefined!\n";
         return;
      }
      //retrieve the total number of entries ranges processed so far by TPool
      nProcessed = ReadBuffer<unsigned>(msg.second.get());
      //evaluate the file and the entries range to process
      fileN = nProcessed / fNWorkers;
   } else {
      //evaluate the file and the entries range to process
      fileN = ReadBuffer<unsigned>(msg.second.get());
   }

   std::unique_ptr<TFile> fp;
   TTree *tree = nullptr;
   if (code != PoolCode::kProcTree ||
      (code == PoolCode::kProcTree && fTree->GetCurrentFile())) {
      //open file
     if (code == PoolCode::kProcTree && fTree->GetCurrentFile()) {
         // Single tree from file: we need to reopen, because file descriptor gets invalidated across Fork
         fp.reset(OpenFile(fTree->GetCurrentFile()->GetName()));
      } else {
         fp.reset(OpenFile(fFileNames[fileN]));
      }
      if (fp == nullptr) {
         //errors are handled inside OpenFile
         return;
      }

      //retrieve the TTree with the specified name from file
      //we are not the owner of the TTree object, the file is!
      tree = RetrieveTree(fp.get());
      if(tree == nullptr) {
         //errors are handled inside RetrieveTree
         return;
      }
   } else {
      // Tree in memory: OK
      tree = fTree;
   }

   //create entries range
   Long64_t start = 0;
   Long64_t finish = 0;
   if (code == PoolCode::kProcRange || code == PoolCode::kProcTree) {
      //example: for 21 entries, 4 workers we want ranges 0-5, 5-10, 10-15, 15-21
      //and this worker must take the rangeN-th range
      unsigned nEntries = tree->GetEntries();
      unsigned nBunch = nEntries / fNWorkers;
      unsigned rangeN = nProcessed % fNWorkers;
      start = rangeN*nBunch;
      if(rangeN < (fNWorkers-1))
         finish = (rangeN+1)*nBunch;
      else
         finish = nEntries;
   } else {
      start = 0;
      finish = tree->GetEntries();
   }

   //check if we are going to reach the max of entries
   //change finish accordingly
   if (fMaxNEntries)
      if (fProcessedEntries + finish - start > fMaxNEntries)
         finish = start + fMaxNEntries - fProcessedEntries;

   // create a TTreeReader that reads this range of entries
   TTreeReader reader(tree);
   //Set first entry to start-1 so that the next call to TTreeReader::Next() sets the entry to the right value
   TTreeReader::EEntryStatus status = reader.SetEntriesRange(start-1, finish);
   if(status != TTreeReader::kEntryValid) {
      std::string reply = "S" + std::to_string(GetNWorker());
      reply += ": could not set TTreeReader to range " + std::to_string(start) + " " + std::to_string(finish);
      MPSend(GetSocket(), PoolCode::kProcError, reply.data());
      return;
   }
   
   //execute function
   auto res = fProcFunc(reader);

   //detach result from file if needed (currently needed for TH1, TTree, TEventList)
   DetachRes(res);

   //update the number of processed entries
   fProcessedEntries += finish - start;

   if(fCanReduce) {
      fReducedResult = static_cast<decltype(fReducedResult)>(PoolUtils::ReduceObjects({res, fReducedResult})); //TODO try not to copy these into a vector, do everything by ref. std::vector<T&>?
   } else {
      fCanReduce = true;
      fReducedResult = res;
   }
      
   if(fMaxNEntries == fProcessedEntries)
      //we are done forever
      MPSend(GetSocket(), PoolCode::kProcResult, fReducedResult);
   else
      //we are done for now
      MPSend(GetSocket(), PoolCode::kIdling);
}


template<class F>
TFile *TPoolProcessor<F>::OpenFile(const std::string& fileName)
{

   TFile *fp = TFile::Open(fileName.c_str());
   if (fp == nullptr || fp->IsZombie()) {
      std::string reply = "S" + std::to_string(GetNWorker());
      reply.append(": could not open file ");
      reply.append(fileName);
      MPSend(GetSocket(), PoolCode::kProcError, reply.data());
      return nullptr;
   }

   return fp;
}


template<class F>
TTree *TPoolProcessor<F>::RetrieveTree(TFile *fp)
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
      std::string reply = "S" + std::to_string(GetNWorker());
      std::stringstream ss;
      ss << ": cannot find tree with name " << fTreeName << " in file " << fp->GetName();
      reply.append(ss.str());
      MPSend(GetSocket(), PoolCode::kProcError, reply.data());
      return nullptr;
   }

   return tree;
}


template<class F>
ULong64_t TPoolProcessor<F>::EvalMaxEntries(ULong64_t maxEntries)
{
   //e.g.: when dividing 10 entries between 3 workers, the first
   //two will process 10/3 == 3 entries, the last one will process
   //10 - 2*(10/3) == 4 entries.
   if(GetNWorker() < fNWorkers-1)
      return maxEntries/fNWorkers;
   else
      return maxEntries - (fNWorkers-1)*(maxEntries/fNWorkers);
}

#endif
