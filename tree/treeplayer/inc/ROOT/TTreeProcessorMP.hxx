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

#ifndef ROOT_TTreeProcessorMP
#define ROOT_TTreeProcessorMP

#include "MPCode.h"
#include "MPSendRecv.h"
#include "PoolUtils.h"
#include "TChain.h"
#include "TChainElement.h"
#include "TError.h"
#include "TFileCollection.h"
#include "TFileInfo.h"
#include "THashList.h"
#include "TMPClient.h"
#include "TMPWorkerTree.h"
#include "TSelector.h"
#include "TTreeReader.h"
#include <algorithm> //std::generate
#include <numeric> //std::iota
#include <string>
#include <type_traits> //std::result_of, std::enable_if
#include <functional> //std::reference_wrapper
#include <vector>

namespace ROOT {

class TTreeProcessorMP : private TMPClient {
public:
   explicit TTreeProcessorMP(unsigned nWorkers = 0); //default number of workers is the number of processors
   ~TTreeProcessorMP() = default;
   //it doesn't make sense for a TTreeProcessorMP to be copied
   TTreeProcessorMP(const TTreeProcessorMP &) = delete;
   TTreeProcessorMP &operator=(const TTreeProcessorMP &) = delete;

   // Process
   // these versions requires that procFunc returns a ptr to TObject or inheriting classes and takes a TTreeReader& (both enforced at compile-time)
   template<class F> auto Process(const std::vector<std::string>& fileNames, F procFunc, const std::string& treeName = "", ULong64_t nToProcess = 0) -> typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type;
   template<class F> auto Process(const std::string& fileName, F procFunc, const std::string& treeName = "", ULong64_t nToProcess = 0) -> typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type;
   template<class F> auto Process(TFileCollection& files, F procFunc, const std::string& treeName = "", ULong64_t nToProcess = 0) -> typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type;
   template<class F> auto Process(TChain& files, F procFunc, const std::string& treeName = "", ULong64_t nToProcess = 0) -> typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type;
   template<class F> auto Process(TTree& tree, F procFunc, ULong64_t nToProcess = 0) -> typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type;
   // these versions require a TSelector
   TList* Process(const std::vector<std::string>& fileNames, TSelector& selector, const std::string& treeName = "", ULong64_t nToProcess = 0);
   TList* Process(const std::string &fileName, TSelector& selector, const std::string& treeName = "", ULong64_t nToProcess = 0);
   TList* Process(TFileCollection& files, TSelector& selector, const std::string& treeName = "", ULong64_t nToProcess = 0);
   TList* Process(TChain& files, TSelector& selector, const std::string& treeName = "", ULong64_t nToProcess = 0);
   TList* Process(TTree& tree, TSelector& selector, ULong64_t nToProcess = 0);

   void SetNWorkers(unsigned n) { TMPClient::SetNWorkers(n); }
   unsigned GetNWorkers() const { return TMPClient::GetNWorkers(); }

private:
   template<class T> void Collect(std::vector<T> &reslist);
   template<class T> void HandlePoolCode(MPCodeBufPair &msg, TSocket *sender, std::vector<T> &reslist);

   void FixLists(std::vector<TObject*> &lists);
   void Reset();
   void ReplyToIdle(TSocket *s);

   unsigned fNProcessed; ///< number of arguments already passed to the workers
   unsigned fNToProcess; ///< total number of arguments to pass to the workers

   /// A collection of the types of tasks that TTreeProcessorMP can execute.
   /// It is used to interpret in the right way and properly reply to the
   /// messages received (see, for example, TTreeProcessorMP::HandleInput)
   enum class ETask : unsigned char {
      kNoTask,        ///< no task is being executed
      kProcByRange,   ///< a Process method is being executed and each worker will process a certain range of each file
      kProcByFile     ///< a Process method is being executed and each worker will process a different file
   };

   ETask fTaskType = ETask::kNoTask; ///< the kind of task that is being executed, if any
};

template<class F>
auto TTreeProcessorMP::Process(const std::vector<std::string>& fileNames, F procFunc, const std::string& treeName, ULong64_t nToProcess) -> typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type
{
   using retType = typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type;
   static_assert(std::is_constructible<TObject*, retType>::value, "procFunc must return a pointer to a class inheriting from TObject, and must take a reference to TTreeReader as the only argument");

   //prepare environment
   Reset();
   unsigned nWorkers = GetNWorkers();

   //fork
   TMPWorkerTreeFunc<F> worker(procFunc, fileNames, treeName, nWorkers, nToProcess);
   bool ok = Fork(worker);
   if(!ok) {
      Error("TTreeProcessorMP::Process", "[E][C] Could not fork. Aborting operation.");
      return nullptr;
   }

   if(fileNames.size() < nWorkers) {
      //TTree entry granularity. For each file, we divide entries equally between workers
      fTaskType = ETask::kProcByRange;
      //Tell workers to start processing entries
      fNToProcess = nWorkers*fileNames.size(); //this is the total number of ranges that will be processed by all workers cumulatively
      std::vector<unsigned> args(nWorkers);
      std::iota(args.begin(), args.end(), 0);
      fNProcessed = Broadcast(MPCode::kProcRange, args);
      if(fNProcessed < nWorkers)
         Error("TTreeProcessorMP::Process", "[E][C] There was an error while sending tasks to workers. Some entries might not be processed.");
   } else {
      //file granularity. each worker processes one whole file as a single task
      fTaskType = ETask::kProcByFile;
      fNToProcess = fileNames.size();
      std::vector<unsigned> args(nWorkers);
      std::iota(args.begin(), args.end(), 0);
      fNProcessed = Broadcast(MPCode::kProcFile, args);
      if(fNProcessed < nWorkers)
         Error("TTreeProcessorMP::Process", "[E][C] There was an error while sending tasks to workers. Some entries might not be processed.");
   }

   //collect results, distribute new tasks
   std::vector<TObject*> reslist;
   Collect(reslist);

   //merge
   PoolUtils::ReduceObjects<TObject *> redfunc;
   auto res = redfunc(reslist);

   //clean-up and return
   ReapWorkers();
   fTaskType = ETask::kNoTask;
   return static_cast<retType>(res);
}


template<class F>
auto TTreeProcessorMP::Process(const std::string& fileName, F procFunc, const std::string& treeName, ULong64_t nToProcess) -> typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type
{
   std::vector<std::string> singleFileName(1, fileName);
   return Process(singleFileName, procFunc, treeName, nToProcess);
}


template<class F>
auto TTreeProcessorMP::Process(TFileCollection& files, F procFunc, const std::string& treeName, ULong64_t nToProcess) -> typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type
{
   std::vector<std::string> fileNames(files.GetNFiles());
   unsigned count = 0;
   for(auto f : *static_cast<THashList*>(files.GetList()))
      fileNames[count++] = static_cast<TFileInfo*>(f)->GetCurrentUrl()->GetUrl();

   return Process(fileNames, procFunc, treeName, nToProcess);
}


template<class F>
auto TTreeProcessorMP::Process(TChain& files, F procFunc, const std::string& treeName, ULong64_t nToProcess) -> typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type
{
   TObjArray* filelist = files.GetListOfFiles();
   std::vector<std::string> fileNames(filelist->GetEntries());
   unsigned count = 0;
   for(auto f : *filelist)
      fileNames[count++] = f->GetTitle();

   return Process(fileNames, procFunc, treeName, nToProcess);
}


template<class F>
auto TTreeProcessorMP::Process(TTree& tree, F procFunc, ULong64_t nToProcess) -> typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type
{
   using retType = typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type;
   static_assert(std::is_constructible<TObject*, retType>::value, "procFunc must return a pointer to a class inheriting from TObject, and must take a reference to TTreeReader as the only argument");

   //prepare environment
   Reset();
   unsigned nWorkers = GetNWorkers();

   //fork
   TMPWorkerTreeFunc<F> worker(procFunc, &tree, nWorkers, nToProcess);
   bool ok = Fork(worker);
   if(!ok) {
      Error("TTreeProcessorMP::Process", "[E][C] Could not fork. Aborting operation.");
      return nullptr;
   }

   //divide entries equally between workers
   fTaskType = ETask::kProcByRange;

   //tell workers to start processing entries
   fNToProcess = nWorkers; //this is the total number of ranges that will be processed by all workers cumulatively
   std::vector<unsigned> args(nWorkers);
   std::iota(args.begin(), args.end(), 0);
   fNProcessed = Broadcast(MPCode::kProcTree, args);
   if(fNProcessed < nWorkers)
      Error("TTreeProcessorMP::Process", "[E][C] There was an error while sending tasks to workers. Some entries might not be processed.");

   //collect results, distribute new tasks
   std::vector<TObject*> reslist;
   Collect(reslist);

   //merge
   PoolUtils::ReduceObjects<TObject *> redfunc;
   auto res = redfunc(reslist);

   //clean-up and return
   ReapWorkers();
   fTaskType = ETask::kNoTask;
   return static_cast<retType>(res);
}

//////////////////////////////////////////////////////////////////////////
/// Handle message and reply to the worker
template<class T>
void TTreeProcessorMP::HandlePoolCode(MPCodeBufPair &msg, TSocket *s, std::vector<T> &reslist)
{
   unsigned code = msg.first;
   if (code == MPCode::kIdling) {
      ReplyToIdle(s);
   } else if(code == MPCode::kProcResult) {
      if(msg.second != nullptr)
         reslist.push_back(std::move(ReadBuffer<T>(msg.second.get())));
      MPSend(s, MPCode::kShutdownOrder);
   } else if(code == MPCode::kProcError) {
      const char *str = ReadBuffer<const char*>(msg.second.get());
      Error("TTreeProcessorMP::HandlePoolCode", "[E][C] a worker encountered an error: %s\n"
                                         "Continuing execution ignoring these entries.", str);
      ReplyToIdle(s);
      delete [] str;
   } else {
      // UNKNOWN CODE
      Error("TTreeProcessorMP::HandlePoolCode", "[W][C] unknown code received from server. code=%d", code);
   }
}

//////////////////////////////////////////////////////////////////////////
/// Listen for messages sent by the workers and call the appropriate handler function.
/// TTreeProcessorMP::HandlePoolCode is called on messages with a code < 1000 and
/// TMPClient::HandleMPCode is called on messages with a code >= 1000.
template<class T>
void TTreeProcessorMP::Collect(std::vector<T> &reslist)
{
   TMonitor &mon = GetMonitor();
   mon.ActivateAll();
   while (mon.GetActive() > 0) {
      TSocket *s = mon.Select();
      MPCodeBufPair msg = MPRecv(s);
      if (msg.first == MPCode::kRecvError) {
         Error("TTreeProcessorMP::Collect", "[E][C] Lost connection to a worker");
         Remove(s);
      } else if (msg.first < 1000)
         HandlePoolCode(msg, s, reslist);
      else
         HandleMPCode(msg, s);
   }
}

} // ROOT namespace

#endif
