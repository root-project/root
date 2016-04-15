/* @(#)root/multiproc:$Id$ */
// Author: Enrico Guiraud July 2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProcPool
#define ROOT_TProcPool

#include "TMPClient.h"
#include "MPSendRecv.h"
#include "TPoolWorker.h"
#include "PoolUtils.h"
#include "MPCode.h"
#include "TPoolProcessor.h"
#include "TTreeReader.h"
#include "TFileCollection.h"
#include "TChain.h"
#include "TChainElement.h"
#include "THashList.h"
#include "TFileInfo.h"
#include <vector>
#include <string>
#include <initializer_list>
#include <type_traits> //std::result_of, std::enable_if
#include <numeric> //std::iota
#include <algorithm> //std::generate
#include <functional> //std::reference_wrapper
#include <iostream>
#include "TPool.h"

class TProcPool : public TPool<TProcPool>, private TMPClient {
public:
   explicit TProcPool(unsigned nWorkers = 0); //default number of workers is the number of processors
   ~TProcPool() {}
   //it doesn't make sense for a TProcPool to be copied
   TProcPool(const TProcPool &) = delete;
   TProcPool &operator=(const TProcPool &) = delete;

   // Map
   template<class F, class Cond = noReferenceCond<F>>
   auto Map(F func, unsigned nTimes) -> std::vector<typename std::result_of<F()>::type>;
   /// \cond
   template<class F, class INTEGER, class Cond = noReferenceCond<F, INTEGER>>
   auto Map(F func, ROOT::TSeq<INTEGER> args) -> std::vector<typename std::result_of<F(INTEGER)>::type>;
   template<class F, class T, class Cond = noReferenceCond<F, T>>
   auto Map(F func, std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type>;
   /// \endcond
   using TPool<TProcPool>::Map;

   // ProcTree
   // this version requires that procFunc returns a ptr to TObject or inheriting classes and takes a TTreeReader& (both enforced at compile-time)
   template<class F> auto ProcTree(const std::vector<std::string>& fileNames, F procFunc, const std::string& treeName = "", ULong64_t nToProcess = 0) -> typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type;
   template<class F> auto ProcTree(const std::string& fileName, F procFunc, const std::string& treeName = "", ULong64_t nToProcess = 0) -> typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type;
   template<class F> auto ProcTree(TFileCollection& files, F procFunc, const std::string& treeName = "", ULong64_t nToProcess = 0) -> typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type;
   template<class F> auto ProcTree(TChain& files, F procFunc, const std::string& treeName = "", ULong64_t nToProcess = 0) -> typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type;
   template<class F> auto ProcTree(TTree& tree, F procFunc, ULong64_t nToProcess = 0) -> typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type;

   void SetNWorkers(unsigned n) { TMPClient::SetNWorkers(n); }
   unsigned GetNWorkers() const { return TMPClient::GetNWorkers(); }

   template<class T, class BINARYOP> auto Reduce(const std::vector<T> &objs, BINARYOP redfunc)-> decltype(redfunc(objs.front(), objs.front())) = delete;
   using TPool<TProcPool>::Reduce;

private:
   template<class T> void Collect(std::vector<T> &reslist);
   template<class T> void HandlePoolCode(MPCodeBufPair &msg, TSocket *sender, std::vector<T> &reslist);

   void Reset();
   void ReplyToFuncResult(TSocket *s);
   void ReplyToIdle(TSocket *s);

   unsigned fNProcessed; ///< number of arguments already passed to the workers
   unsigned fNToProcess; ///< total number of arguments to pass to the workers

   /// A collection of the types of tasks that TProcPool can execute.
   /// It is used to interpret in the right way and properly reply to the
   /// messages received (see, for example, TProcPool::HandleInput)
   enum class ETask : unsigned char {
      kNoTask,   ///< no task is being executed
      kMap,          ///< a Map method with no arguments is being executed
      kMapWithArg,   ///< a Map method with arguments is being executed
      kProcByRange,   ///< a ProcTree method is being executed and each worker will process a certain range of each file
      kProcByFile,    ///< a ProcTree method is being executed and each worker will process a different file
   };

   ETask fTaskType = ETask::kNoTask; ///< the kind of task that is being executed, if any
};


/************ TEMPLATE METHODS IMPLEMENTATION ******************/

//////////////////////////////////////////////////////////////////////////
/// Execute func (with no arguments) nTimes in parallel.
/// A vector containg executions' results is returned.
/// Functions that take more than zero arguments can be executed (with
/// fixed arguments) by wrapping them in a lambda or with std::bind.
template<class F, class Cond>
auto TProcPool::Map(F func, unsigned nTimes) -> std::vector<typename std::result_of<F()>::type>
{
   using retType = decltype(func());
   //prepare environment
   Reset();
   fTaskType = ETask::kMap;

   //fork max(nTimes, fNWorkers) times
   unsigned oldNWorkers = GetNWorkers();
   if (nTimes < oldNWorkers)
      SetNWorkers(nTimes);
   TPoolWorker<F> worker(func);
   bool ok = Fork(worker);
   SetNWorkers(oldNWorkers);
   if (!ok)
   {
      std::cerr << "[E][C] Could not fork. Aborting operation\n";
      return std::vector<retType>();
   }

   //give out tasks
   fNToProcess = nTimes;
   std::vector<retType> reslist;
   reslist.reserve(fNToProcess);
   fNProcessed = Broadcast(PoolCode::kExecFunc, fNToProcess);

   //collect results, give out other tasks if needed
   Collect(reslist);

   //clean-up and return
   ReapWorkers();
   fTaskType = ETask::kNoTask;
   return reslist;
}

// tell doxygen to ignore this (\endcond closes the statement)
/// \cond

// actual implementation of the Map method. all other calls with arguments eventually
// call this one
template<class F, class T, class Cond>
auto TProcPool::Map(F func, std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type>
{
   //check whether func is callable
   using retType = decltype(func(args.front()));
   //prepare environment
   Reset();
   fTaskType = ETask::kMapWithArg;

   //fork max(args.size(), fNWorkers) times
   //N.B. from this point onwards, args is filled with undefined (but valid) values, since TPoolWorker moved its content away
   unsigned oldNWorkers = GetNWorkers();
   if (args.size() < oldNWorkers)
      SetNWorkers(args.size());
   TPoolWorker<F, T> worker(func, args);
   bool ok = Fork(worker);
   SetNWorkers(oldNWorkers);
   if (!ok)
   {
      std::cerr << "[E][C] Could not fork. Aborting operation\n";
      return std::vector<retType>();
   }

   //give out tasks
   fNToProcess = args.size();
   std::vector<retType> reslist;
   reslist.reserve(fNToProcess);
   std::vector<unsigned> range(fNToProcess);
   std::iota(range.begin(), range.end(), 0);
   fNProcessed = Broadcast(PoolCode::kExecFuncWithArg, range);

   //collect results, give out other tasks if needed
   Collect(reslist);

   //clean-up and return
   ReapWorkers();
   fTaskType = ETask::kNoTask;
   return reslist;
}

template<class F, class INTEGER, class Cond>
auto TProcPool::Map(F func, ROOT::TSeq<INTEGER> args) -> std::vector<typename std::result_of<F(INTEGER)>::type>
{
   std::vector<INTEGER> vargs(args.size());
   std::copy(args.begin(), args.end(), vargs.begin());
   const auto &reslist = Map(func, vargs);
   return reslist;
}
// tell doxygen to stop ignoring code
/// \endcond


template<class F>
auto TProcPool::ProcTree(const std::vector<std::string>& fileNames, F procFunc, const std::string& treeName, ULong64_t nToProcess) -> typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type
{
   using retType = typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type;
   static_assert(std::is_constructible<TObject*, retType>::value, "procFunc must return a pointer to a class inheriting from TObject, and must take a reference to TTreeReader as the only argument");

   //prepare environment
   Reset();
   unsigned nWorkers = GetNWorkers();

   //fork
   TPoolProcessor<F> worker(procFunc, fileNames, treeName, nWorkers, nToProcess);
   bool ok = Fork(worker);
   if(!ok) {
      std::cerr << "[E][C] Could not fork. Aborting operation\n";
      return nullptr;
   }

   if(fileNames.size() < nWorkers) {
      //TTree entry granularity. For each file, we divide entries equally between workers
      fTaskType = ETask::kProcByRange;
      //Tell workers to start processing entries
      fNToProcess = nWorkers*fileNames.size(); //this is the total number of ranges that will be processed by all workers cumulatively
      std::vector<unsigned> args(nWorkers);
      std::iota(args.begin(), args.end(), 0);
      fNProcessed = Broadcast(PoolCode::kProcRange, args);
      if(fNProcessed < nWorkers)
         std::cerr << "[E][C] There was an error while sending tasks to workers. Some entries might not be processed.\n";
   } else {
      //file granularity. each worker processes one whole file as a single task
      fTaskType = ETask::kProcByFile;
      fNToProcess = fileNames.size();
      std::vector<unsigned> args(nWorkers);
      std::iota(args.begin(), args.end(), 0);
      fNProcessed = Broadcast(PoolCode::kProcFile, args);
      if(fNProcessed < nWorkers)
         std::cerr << "[E][C] There was an error while sending tasks to workers. Some entries might not be processed.\n";
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
auto TProcPool::ProcTree(const std::string& fileName, F procFunc, const std::string& treeName, ULong64_t nToProcess) -> typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type
{
   std::vector<std::string> singleFileName(1, fileName);
   return ProcTree(singleFileName, procFunc, treeName, nToProcess);
}


template<class F>
auto TProcPool::ProcTree(TFileCollection& files, F procFunc, const std::string& treeName, ULong64_t nToProcess) -> typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type
{
   std::vector<std::string> fileNames(files.GetNFiles());
   unsigned count = 0;
   for(auto f : *static_cast<THashList*>(files.GetList()))
      fileNames[count++] = static_cast<TFileInfo*>(f)->GetCurrentUrl()->GetFile();

   return ProcTree(fileNames, procFunc, treeName, nToProcess);
}


template<class F>
auto TProcPool::ProcTree(TChain& files, F procFunc, const std::string& treeName, ULong64_t nToProcess) -> typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type
{
   TObjArray* filelist = files.GetListOfFiles();
   std::vector<std::string> fileNames(filelist->GetEntries());
   unsigned count = 0;
   for(auto f : *filelist)
      fileNames[count++] = f->GetTitle();

   return ProcTree(fileNames, procFunc, treeName, nToProcess);
}


template<class F>
auto TProcPool::ProcTree(TTree& tree, F procFunc, ULong64_t nToProcess) -> typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type
{
   using retType = typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type;
   static_assert(std::is_constructible<TObject*, retType>::value, "procFunc must return a pointer to a class inheriting from TObject, and must take a reference to TTreeReader as the only argument");

   //prepare environment
   Reset();
   unsigned nWorkers = GetNWorkers();

   //fork
   TPoolProcessor<F> worker(procFunc, &tree, nWorkers, nToProcess);
   bool ok = Fork(worker);
   if(!ok) {
      std::cerr << "[E][C] Could not fork. Aborting operation\n";
      return nullptr;
   }

   //divide entries equally between workers
   fTaskType = ETask::kProcByRange;

   //tell workers to start processing entries
   fNToProcess = nWorkers; //this is the total number of ranges that will be processed by all workers cumulatively
   std::vector<unsigned> args(nWorkers);
   std::iota(args.begin(), args.end(), 0);
   fNProcessed = Broadcast(PoolCode::kProcTree, args);
   if(fNProcessed < nWorkers)
      std::cerr << "[E][C] There was an error while sending tasks to workers. Some entries might not be processed.\n";

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
/// Listen for messages sent by the workers and call the appropriate handler function.
/// TProcPool::HandlePoolCode is called on messages with a code < 1000 and
/// TMPClient::HandleMPCode is called on messages with a code >= 1000.
template<class T>
void TProcPool::Collect(std::vector<T> &reslist)
{
   TMonitor &mon = GetMonitor();
   mon.ActivateAll();
   while (mon.GetActive() > 0) {
      TSocket *s = mon.Select();
      MPCodeBufPair msg = MPRecv(s);
      if (msg.first == MPCode::kRecvError) {
         std::cerr << "[E][C] Lost connection to a worker\n";
         Remove(s);
      } else if (msg.first < 1000)
         HandlePoolCode(msg, s, reslist);
      else
         HandleMPCode(msg, s);
   }
}


//////////////////////////////////////////////////////////////////////////
/// Handle message and reply to the worker
template<class T>
void TProcPool::HandlePoolCode(MPCodeBufPair &msg, TSocket *s, std::vector<T> &reslist)
{
   unsigned code = msg.first;
   if (code == PoolCode::kFuncResult) {
      reslist.push_back(std::move(ReadBuffer<T>(msg.second.get())));
      ReplyToFuncResult(s);
   } else if (code == PoolCode::kIdling) {
      ReplyToIdle(s);
   } else if(code == PoolCode::kProcResult) {
      if(msg.second != nullptr)
         reslist.push_back(std::move(ReadBuffer<T>(msg.second.get())));
      MPSend(s, MPCode::kShutdownOrder);
   } else if(code == PoolCode::kProcError) {
      const char *str = ReadBuffer<const char*>(msg.second.get());
      std::cerr << "[E][C] a worker encountered an error: " << str << "\n"
                << "Continuing execution ignoring these entries.\n";
      ReplyToIdle(s);
      delete [] str;
   } else {
      // UNKNOWN CODE
      std::cerr << "[W][C] unknown code received from server. code=" << code << "\n";
   }
}

#endif
