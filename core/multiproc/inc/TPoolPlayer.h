/* @(#)root/multiproc:$Id$ */
// Author: Enrico Guiraud September 2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPoolPlayer
#define ROOT_TPoolPlayer

#include "TMPWorker.h"
#include "TTree.h"
#include "TSelector.h"
#include "MPSendRecv.h"

class TPoolPlayer : public TMPWorker {
   public:
   TPoolPlayer(TSelector &selector, TTree *tree, unsigned nWorkers, ULong64_t maxEntries) :
      TMPWorker(), fSelector(selector), fFileNames(), fTreeName(), fTree(tree),
      fNWorkers(nWorkers), fMaxNEntries(maxEntries),
      fProcessedEntries(0)
   {}
   TPoolPlayer(TSelector &selector, const std::vector<std::string>& fileNames,
               const std::string& treeName, unsigned nWorkers, ULong64_t maxEntries) :
      TMPWorker(), fSelector(selector), fFileNames(fileNames), fTreeName(treeName), fTree(nullptr),
      fNWorkers(nWorkers), fMaxNEntries(maxEntries),
      fProcessedEntries(0)
   {}

   ~TPoolPlayer() {}

   void HandleInput(MPCodeBufPair& msg); ///< Execute instructions received from a TPool client
   void Init(int fd, unsigned nWorkers);

   private:
   void ProcTree(MPCodeBufPair& msg); ///< Run fSelector->Process over the tree entries, send back result
   void ProcDataSet(unsigned int code, MPCodeBufPair& msg); ///< Run fSelector->Process over a data set

   TSelector &fSelector; ///< pointer to the selector to be used to process the tree. It is null if we are not using a TSelector.
   std::vector<std::string> fFileNames; ///< the files to be processed by all workers
   std::string fTreeName; ///< the name of the tree to be processed
   TTree *fTree; ///< tree to be processed. It is only used if the tree is directly passed to TProcessExecutor::Process as argument
   unsigned fNWorkers; ///< the number of workers spawned
   ULong64_t fMaxNEntries; ///< the maximum number of entries to be processed by this worker
   ULong64_t fProcessedEntries; ///< the number of entries processed by this worker so far
   bool fFirstEntry = true;
};

#endif
