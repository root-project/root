/* @(#)root/multiproc:$Id$ */
// Author: G Ganis Jan 2017

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMPWorkerTree
#define ROOT_TMPWorkerTree

#include "TMPWorker.h"
#include "TFile.h"
#include "TEntryList.h"
#include "TEventList.h"
#include "TH1.h"
#include "TKey.h"
#include "TSelector.h"
#include "TTree.h"
#include "TTreeCache.h"
#include "TTreeReader.h"

#include <memory> //unique_ptr
#include <string>
#include <sstream>
#include <type_traits> //std::result_of
#include <unistd.h> //pid_t

class TMPWorkerTree : public TMPWorker {
   /// \cond
//   ClassDef(TMPWorkerTree, 0);
   /// \endcond
public:
   TMPWorkerTree();
   TMPWorkerTree(const std::vector<std::string>& fileNames, const std::string& treeName, unsigned nWorkers, ULong64_t maxEntries);
   TMPWorkerTree(TTree *tree, unsigned nWorkers, ULong64_t maxEntries);
   virtual ~TMPWorkerTree();

   // It doesn't make sense to copy a TMPWorker (each one has a uniq_ptr to its socket)
   TMPWorkerTree(const TMPWorkerTree &) = delete;
   TMPWorkerTree &operator=(const TMPWorkerTree &) = delete;

protected:

   void         CloseFile();
   ULong64_t    EvalMaxEntries(ULong64_t maxEntries);
   void         HandleInput(MPCodeBufPair& msg); ///< Execute instructions received from a MP client
   void         Init(int fd, unsigned workerN);
   TFile       *OpenFile(const std::string& fileName);
   virtual void Process(unsigned, MPCodeBufPair&) { }
   TTree       *RetrieveTree(TFile *fp);
   virtual void SendResult() { }
   void         Setup();
   void         SetupTreeCache(TTree *tree);

   std::vector<std::string> fFileNames; ///< the files to be processed by all workers
   std::string fTreeName;               ///< the name of the tree to be processed
   TTree *fTree;                        ///< pointer to the tree to be processed. It is only used if the tree is directly passed to TProcessExecutor::Process as argument
   TFile *fFile;                        ///< last open file

private:

   // TTree cache handling
   TTreeCache *fTreeCache;              ///< instance of the tree cache for the tree
   Bool_t      fTreeCacheIsLearning;    ///< Whether cache is in learning phase
   Bool_t      fUseTreeCache;           ///< Control usage of the tree cache
   Long64_t    fCacheSize;              ///< Cache size
};

template<class F>
class TMPWorkerTreeFunc : public TMPWorkerTree {
public:
   TMPWorkerTreeFunc(F procFunc, const std::vector<std::string>& fileNames,
                                 const std::string& treeName, unsigned nWorkers, ULong64_t maxEntries)
                  : TMPWorkerTree(fileNames, treeName, nWorkers, maxEntries),
                    fProcFunc(procFunc), fReducedResult(), fCanReduce(false) {}
   TMPWorkerTreeFunc(F procFunc, TTree *tree, unsigned nWorkers, ULong64_t maxEntries)
                  : TMPWorkerTree(tree, nWorkers, maxEntries),
                    fProcFunc(procFunc), fReducedResult(), fCanReduce(false) {}
   virtual ~TMPWorkerTreeFunc() {}

private:
   void Process(unsigned code, MPCodeBufPair& msg);
   void SendResult();

   F  fProcFunc; ///< copy the function to be executed
   typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type fReducedResult; ///< the results of the executions of fProcFunc merged together
   bool fCanReduce; ///< true if fReducedResult can be reduced with a new result, false until we have produced one result
};

class TMPWorkerTreeSel : public TMPWorkerTree {
public:
   TMPWorkerTreeSel(TSelector &selector, const std::vector<std::string>& fileNames,
                                         const std::string& treeName, unsigned nWorkers, ULong64_t maxEntries)
                  : TMPWorkerTree(fileNames, treeName, nWorkers, maxEntries),
                    fSelector(selector), fFirstEntry(true) {}
   TMPWorkerTreeSel(TSelector &selector, TTree *tree, unsigned nWorkers, ULong64_t maxEntries)
                  : TMPWorkerTree(tree, nWorkers, maxEntries),
                    fSelector(selector), fFirstEntry(true) {}
   virtual ~TMPWorkerTreeSel() {}

private:
   void Process(unsigned code, MPCodeBufPair& msg);
   void SendResult();

   TSelector &fSelector; ///< pointer to the selector to be used to process the tree. It is null if we are not using a TSelector.
   bool fFirstEntry = true;
};

//////////////////////////////////////////////////////////////////////////
/// Auxilliary templated functions
/// If the user lambda returns a TH1F*, TTree*, TEventList*, we incur in the
/// problem of that object being automatically owned by the current open file.
/// For these three types, we call SetDirectory(nullptr) to detach the returned
/// object from the file we are reading the TTree from.
/// Note: the only sane case in which this should happen is when a TH1F* is
/// returned.
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

//////////////////////////////////////////////////////////////////////////
/// Generic function processing SendResult and Process overload

template<class F>
void TMPWorkerTreeFunc<F>::SendResult()
{
   //send back result
   MPSend(GetSocket(), MPCode::kProcResult, fReducedResult);
}

template<class F>
void TMPWorkerTreeFunc<F>::Process(unsigned code, MPCodeBufPair& msg)
{
   //evaluate the index of the file to process in fFileNames
   //(we actually don't need the parameter if code == kProcTree)
   unsigned fileN = 0;
   unsigned nProcessed = 0;
   if (code == MPCode::kProcRange || code == MPCode::kProcTree) {
      if (code == MPCode::kProcTree && !fTree) {
         // This must be defined
         Error("TMPWorkerTreeFunc::Process", "[S]: Process:kProcTree fTree undefined!\n");
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
   if (code != MPCode::kProcTree ||
      (code == MPCode::kProcTree && fTree->GetCurrentFile())) {
      //open file
     if (code == MPCode::kProcTree && fTree->GetCurrentFile()) {
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

   // Setup the cache, if required
   SetupTreeCache(tree);

   //create entries range
   Long64_t start = 0;
   Long64_t finish = 0;
   if (code == MPCode::kProcRange || code == MPCode::kProcTree) {
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
   TTreeReader::EEntryStatus status = reader.SetEntriesRange(start, finish);
   if(status != TTreeReader::kEntryValid) {
      std::string reply = "S" + std::to_string(GetNWorker());
      reply += ": could not set TTreeReader to range " + std::to_string(start) + " " + std::to_string(finish);
      MPSend(GetSocket(), MPCode::kProcError, reply.data());
      return;
   }

   //execute function
   auto res = fProcFunc(reader);

   //detach result from file if needed (currently needed for TH1, TTree, TEventList)
   DetachRes(res);

   //update the number of processed entries
   fProcessedEntries += finish - start;

   if(fCanReduce) {
      PoolUtils::ReduceObjects<TObject *> redfunc;
      fReducedResult = static_cast<decltype(fReducedResult)>(redfunc({res, fReducedResult})); //TODO try not to copy these into a vector, do everything by ref. std::vector<T&>?
   } else {
      fCanReduce = true;
      fReducedResult = res;
   }

   if(fMaxNEntries == fProcessedEntries)
      //we are done forever
      MPSend(GetSocket(), MPCode::kProcResult, fReducedResult);
   else
      //we are done for now
      MPSend(GetSocket(), MPCode::kIdling);
}

#endif
