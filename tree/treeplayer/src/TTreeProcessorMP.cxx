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

#include "TEnv.h"
#include "ROOT/TTreeProcessorMP.hxx"
#include "TPoolPlayer.h"

//////////////////////////////////////////////////////////////////////////
///
/// \class TTreeProcessorMP
/// \brief This class provides a simple interface to process a TTree dataset
///        in parallel with multi-process technology 
///
/// ###TTreeProcessorMP::ProcTree 
/// (To be written/updated)
/// The two possible usages of the Map method are:\n
/// * Map(F func, unsigned nTimes): func is executed nTimes with no arguments
/// * Map(F func, T& args): func is executed on each element of the collection of arguments args
///
/// For either signature, func is executed as many times as needed by a pool of
/// fNWorkers workers; the number of workers can be passed to the constructor
/// or set via SetNWorkers. It defaults to the number of cores.\n
/// A collection containing the result of each execution is returned.\n
/// **Note:** the user is responsible for the deletion of any object that might
/// be created upon execution of func, returned objects included: TTreeProcessorMP never
/// deletes what it returns, it simply forgets it.\n
/// **Note:** that the usage of TTreeProcessorMP::Map is indicated only when the task to be
/// executed takes more than a few seconds, otherwise the overhead introduced
/// by Map will outrun the benefits of parallel execution on most machines.
///
/// \param func
/// \parblock
/// a lambda expression, an std::function, a loaded macro, a
/// functor class or a function that takes zero arguments (for the first signature)
/// or one (for the second signature).
/// \endparblock
/// \param args
/// \parblock
/// a standard container (vector, list, deque), an initializer list
/// or a pointer to a TCollection (TList*, TObjArray*, ...).
/// \endparblock
/// **Note:** the version of TTreeProcessorMP::Map that takes a TCollection* as argument incurs
/// in the overhead of copying data from the TCollection to an STL container. Only
/// use it when absolutely necessary.\n
/// **Note:** in cases where the function to be executed takes more than
/// zero/one argument but all are fixed except zero/one, the function can be wrapped
/// in a lambda or via std::bind to give it the right signature.\n
/// **Note:** the user should take care of initializing random seeds differently in each
/// process (e.g. using the process id in the seed). Otherwise several parallel executions
/// might generate the same sequence of pseudo-random numbers.
///
/// #### Return value:
/// If T derives from TCollection Map returns a TObjArray, otherwise it
/// returns an std::vector. In both cases, the elements in the container
/// will be the objects returned by func.
///
///
/// #### Examples:
///
/// ~~~{.cpp}
/// root[] TTreeProcessorMP pool; auto hists = pool.Map(CreateHisto, 10);
/// root[] TTreeProcessorMP pool(2); auto squares = pool.Map([](int a) { return a*a; }, {1,2,3});
/// ~~~
///
/// ###TTreeProcessorMP::MapReduce
/// This set of methods behaves exactly like Map, but takes an additional
/// function as a third argument. This function is applied to the set of
/// objects returned by the corresponding Map execution to "squash" them
/// to a single object.
///
/// ####Examples:
/// ~~~{.cpp}
/// root[] TTreeProcessorMP pool; auto ten = pool.MapReduce([]() { return 1; }, 10, [](std::vector<int> v) { return std::accumulate(v.begin(), v.end(), 0); })
/// root[] TTreeProcessorMP pool; auto hist = pool.MapReduce(CreateAndFillHists, 10, PoolUtils::ReduceObjects);
/// ~~~
///
//////////////////////////////////////////////////////////////////////////

namespace ROOT {
//////////////////////////////////////////////////////////////////////////
/// Class constructor.
/// nWorkers is the number of times this ROOT session will be forked, i.e.
/// the number of workers that will be spawned.
TTreeProcessorMP::TTreeProcessorMP(unsigned nWorkers) : TMPClient(nWorkers)
{
   Reset();
}

//////////////////////////////////////////////////////////////////////////
/// TSelector-based tree processing: memory resident tree
TList* TTreeProcessorMP::ProcTree(TTree& tree, TSelector& selector, ULong64_t nToProcess)
{
   //prepare environment
   Reset();
   unsigned nWorkers = GetNWorkers();
   selector.Begin(nullptr);

   //fork
   TPoolPlayer worker(selector, &tree, nWorkers, nToProcess/nWorkers);
   bool ok = Fork(worker);
   if(!ok) {
      Error("TTreeProcessorMP::ProcTree", "[E][C] Could not fork. Aborting operation");
      return nullptr;
   }

   //divide entries equally between workers
   fTaskType = ETask::kProcByRange;

   //tell workers to start processing entries
   fNToProcess = nWorkers; //this is the total number of ranges that will be processed by all workers cumulatively
   std::vector<unsigned> args(nWorkers);
   std::iota(args.begin(), args.end(), 0);
   fNProcessed = Broadcast(PoolCode::kProcTree, args);
   if (fNProcessed < nWorkers)
      Error("TTreeProcessorMP::ProcTree", "[E][C] There was an error while sending tasks to workers."
                                   " Some entries might not be processed.");

   //collect results, distribute new tasks
   std::vector<TObject*> outLists;
   Collect(outLists);

   // The first element must be a TList instead of a TSelector List, to avoid duplicate problems with merging
   FixLists(outLists);

   PoolUtils::ReduceObjects<TObject *> redfunc;
   auto outList = static_cast<TList*>(redfunc(outLists));

   TList *selList = selector.GetOutputList();
   for(auto obj : *outList) {
      selList->Add(obj);
   }
   outList->SetOwner(false);
   delete outList;

   selector.Terminate();

   //clean-up and return
   ReapWorkers();
   fTaskType = ETask::kNoTask;
   return outList;
}

//////////////////////////////////////////////////////////////////////////
/// TSelector-based tree processing: dataset as a vector of files
TList* TTreeProcessorMP::ProcTree(const std::vector<std::string>& fileNames, TSelector& selector, const std::string& treeName, ULong64_t nToProcess)
{

   //prepare environment
   Reset();
   unsigned nWorkers = GetNWorkers();
   selector.Begin(nullptr);

   //fork
   TPoolPlayer worker(selector, fileNames, treeName, nWorkers, nToProcess);
   bool ok = Fork(worker);
   if (!ok) {
      Error("TTreeProcessorMP::ProcTree", "[E][C] Could not fork. Aborting operation");
      return nullptr;
   }

   Int_t procByFile = gEnv->GetValue("MultiProc.TestProcByFile", 0);

   if (procByFile) {
      if (fileNames.size() < nWorkers) {
         // TTree entry granularity: for each file, we divide entries equally between workers
         fTaskType = ETask::kProcByRange;
         // Tell workers to start processing entries
         fNToProcess = nWorkers*fileNames.size(); //this is the total number of ranges that will be processed by all workers cumulatively
         std::vector<unsigned> args(nWorkers);
         std::iota(args.begin(), args.end(), 0);
         fNProcessed = Broadcast(PoolCode::kProcRange, args);
         if (fNProcessed < nWorkers)
            Error("TTreeProcessorMP::ProcTree", "[E][C] There was an error while sending tasks to workers."
                                         " Some entries might not be processed");
      } else {
         // File granularity: each worker processes one whole file as a single task
         fTaskType = ETask::kProcByFile;
         fNToProcess = fileNames.size();
         std::vector<unsigned> args(nWorkers);
         std::iota(args.begin(), args.end(), 0);
         fNProcessed = Broadcast(PoolCode::kProcFile, args);
         if (fNProcessed < nWorkers)
            Error("TTreeProcessorMP::ProcTree", "[E][C] There was an error while sending tasks to workers."
                                         " Some entries might not be processed.");
      }
   } else {
      // TTree entry granularity: for each file, we divide entries equally between workers
      fTaskType = ETask::kProcByRange;
      // Tell workers to start processing entries
      fNToProcess = nWorkers*fileNames.size(); //this is the total number of ranges that will be processed by all workers cumulatively
      std::vector<unsigned> args(nWorkers);
      std::iota(args.begin(), args.end(), 0);
      fNProcessed = Broadcast(PoolCode::kProcRange, args);
      if (fNProcessed < nWorkers)
         Error("TTreeProcessorMP::ProcTree", "[E][C] There was an error while sending tasks to workers."
                                      " Some entries might not be processed.");
   }

   // collect results, distribute new tasks
   std::vector<TObject*> outLists;
   Collect(outLists);

   // The first element must be a TList instead of a TSelector List, to avoid duplicate problems with merging
   FixLists(outLists);

   PoolUtils::ReduceObjects<TObject *> redfunc;
   auto outList = static_cast<TList*>(redfunc(outLists));

   TList *selList = selector.GetOutputList();
   for(auto obj : *outList) {
      selList->Add(obj);
   }
   outList->SetOwner(false);
   delete outList;

   selector.Terminate();

   //clean-up and return
   ReapWorkers();
   fTaskType = ETask::kNoTask;
   return outList;

}

//////////////////////////////////////////////////////////////////////////
/// TSelector-based tree processing: dataset as a TFileCollection
TList* TTreeProcessorMP::ProcTree(TFileCollection& files, TSelector& selector, const std::string& treeName, ULong64_t nToProcess)
{
   std::vector<std::string> fileNames(files.GetNFiles());
   unsigned count = 0;
   for(auto f : *static_cast<THashList*>(files.GetList()))
      fileNames[count++] = static_cast<TFileInfo*>(f)->GetCurrentUrl()->GetUrl();

   TList *rl = ProcTree(fileNames, selector, treeName, nToProcess);
   return rl;
}

//////////////////////////////////////////////////////////////////////////
/// TSelector-based tree processing: dataset as a TChain
TList* TTreeProcessorMP::ProcTree(TChain& files, TSelector& selector, const std::string& treeName, ULong64_t nToProcess)
{
   TObjArray* filelist = files.GetListOfFiles();
   std::vector<std::string> fileNames(filelist->GetEntries());
   unsigned count = 0;
   for(auto f : *filelist)
      fileNames[count++] = f->GetTitle();

   return ProcTree(fileNames, selector, treeName, nToProcess);
}

//////////////////////////////////////////////////////////////////////////
/// TSelector-based tree processing: dataset as a single file
TList* TTreeProcessorMP::ProcTree(const std::string& fileName, TSelector& selector, const std::string& treeName, ULong64_t nToProcess)
{
   std::vector<std::string> singleFileName(1, fileName);
   return ProcTree(singleFileName, selector, treeName, nToProcess);
}

/// Fix list of lists before merging (to avoid errors about duplicated objects)
void TTreeProcessorMP::FixLists(std::vector<TObject*> &lists) {

   // The first element must be a TList instead of a TSelector List, to avoid duplicate problems with merging
   TList *firstlist = new TList;
   TList *oldlist = (TList *) lists[0];
   TIter nxo(oldlist);
   TObject *o = 0;
   while ((o = nxo())) { firstlist->Add(o); }
   oldlist->SetOwner(kFALSE);
   lists.erase(lists.begin());
   lists.insert(lists.begin(), firstlist);
   delete oldlist;
}

//////////////////////////////////////////////////////////////////////////
/// Reset TTreeProcessorMP's state.
void TTreeProcessorMP::Reset()
{
   fNProcessed = 0;
   fNToProcess = 0;
   fTaskType = ETask::kNoTask;
}


//////////////////////////////////////////////////////////////////////////
/// Reply to a worker who just sent a result.
/// If another argument to process exists, tell the worker. Otherwise
/// send a shutdown order.
void TTreeProcessorMP::ReplyToFuncResult(TSocket *s)
{
   if (fNProcessed < fNToProcess) {
      //this cannot be a "greedy worker" task
      if (fTaskType == ETask::kMap)
         MPSend(s, PoolCode::kExecFunc);
      else if (fTaskType == ETask::kMapWithArg)
         MPSend(s, PoolCode::kExecFuncWithArg, fNProcessed);
      ++fNProcessed;
   } else //whatever the task is, we are done
      MPSend(s, MPCode::kShutdownOrder);
}


//////////////////////////////////////////////////////////////////////////
/// Reply to a worker who is idle.
/// If another argument to process exists, tell the worker. Otherwise
/// ask for a result
void TTreeProcessorMP::ReplyToIdle(TSocket *s)
{
   if (fNProcessed < fNToProcess) {
      //we are executing a "greedy worker" task
      if (fTaskType == ETask::kMapWithArg)
         MPSend(s, PoolCode::kExecFuncWithArg, fNProcessed);
      else if (fTaskType == ETask::kMap)
         MPSend(s, PoolCode::kExecFunc);
      else if (fTaskType == ETask::kProcByRange)
         MPSend(s, PoolCode::kProcRange, fNProcessed);
      else if (fTaskType == ETask::kProcByFile)
         MPSend(s, PoolCode::kProcFile, fNProcessed);
      ++fNProcessed;
   } else
      MPSend(s, PoolCode::kSendResult);
}

} // namespace ROOT
