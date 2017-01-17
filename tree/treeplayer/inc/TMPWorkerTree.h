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
#if 0
#include "MPSendRecv.h" //MPCodeBufPair
#include "EPoolCode.h"
#include "TSysEvtHandler.h" //TFileHandler
#endif
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
   //it doesn't make sense to copy a TMPWorker (each one has a uniq_ptr to its socket)
   TMPWorkerTree(const TMPWorkerTree &) = delete;
   TMPWorkerTree &operator=(const TMPWorkerTree &) = delete;

protected:
   std::vector<std::string> fFileNames; ///< the files to be processed by all workers
   std::string fTreeName; ///< the name of the tree to be processed
   TTree *fTree; ///< pointer to the tree to be processed. It is only used if the tree is directly passed to TProcessExecutor::Process as argument
   TFile *fFile; ///< last open file

   void   CloseFile();
   TFile *OpenFile(const std::string& fileName);
   TTree *RetrieveTree(TFile *fp);

   void   Setup();
   void   SetupTreeCache(TTree *tree);

private:
//   virtual void HandleInput(MPCodeBufPair &msg);

   // TTree cache handling
   TTreeCache *fTreeCache;    // instance of the tree cache for the tree
   Bool_t      fTreeCacheIsLearning; // Whether cache is in learning phase
   Bool_t      fUseTreeCache; // Control usage of the tree cache
   Long64_t    fCacheSize;    // Cache size
};

#endif
