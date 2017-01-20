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
#include "TKey.h"
#include "TTree.h"
#include "TTreeCache.h"

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
   virtual void Process(unsigned code, MPCodeBufPair& msg);
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

#endif
