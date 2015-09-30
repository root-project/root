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
#include "TCollection.h"
#include "TPoolWorker.h"
#include "TObjArray.h"
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

class TProcPool : private TMPClient {
public:
   explicit TProcPool(unsigned nWorkers = 0); //default number of workers is the number of processors
   ~TProcPool() {}
   //it doesn't make sense for a TProcPool to be copied
   TProcPool(const TProcPool &) = delete;
   TProcPool &operator=(const TProcPool &) = delete;

   // Map
   //these late return types allow for a compile-time check of compatibility between function signatures and args,
   //and a compile-time check that the argument list implements a front() method (all STL sequence containers have it)
   template<class F> auto Map(F func, unsigned nTimes) -> std::vector<decltype(func())>;
   template<class F, class T> auto Map(F func, T &args) -> std::vector < decltype(++(args.begin()), args.end(), func(args.front())) >;
   /// \cond doxygen should ignore these methods
   template<class F> TObjArray Map(F func, TCollection &args);
   template<class F, class T> auto Map(F func, std::initializer_list<T> args) -> std::vector<decltype(func(*args.begin()))>;
   template<class F, class T> auto Map(F func, std::vector<T> &args) -> std::vector<decltype(func(args.front()))>;
   /// \endcond

   // MapReduce
   // the late return types also check at compile-time whether redfunc is compatible with func,
   // other than checking that func is compatible with the type of arguments.
   // a static_assert check in TProcPool::Reduce is used to check that redfunc is compatible with the type returned by func
   template<class F, class R> auto MapReduce(F func, unsigned nTimes, R redfunc) -> decltype(func());
   template<class F, class T, class R> auto MapReduce(F func, T &args, R redfunc) -> decltype(++(args.begin()), args.end(), func(args.front()));
   /// \cond doxygen should ignore these methods
   template<class F, class R> auto MapReduce(F func, TCollection &args, R redfunc) -> decltype(func(nullptr));
   template<class F, class T, class R> auto MapReduce(F func, std::initializer_list<T> args, R redfunc) -> decltype(func(*args.begin()));
   template<class F, class T, class R> auto MapReduce(F func, std::vector<T> &args, R redfunc) -> decltype(func(args.front()));
   /// \endcond

   // Process
   // this version requires that procFunc returns a ptr to TObject or inheriting classes and takes a TTreeReader& (both enforced at compile-time)
   template<class F> TObject* Process(const std::vector<std::string>& fileNames, F procFunc, const std::string& treeName = "", ULong64_t nToProcess = 0);
   template<class F> TObject* Process(const std::string& fileName, F procFunc, const std::string& treeName = "", ULong64_t nToProcess = 0);
   template<class F> TObject* Process(TFileCollection& files, F procFunc, const std::string& treeName = "", ULong64_t nToProcess = 0);
   template<class F> TObject* Process(TChain& files, F procFunc, const std::string& treeName = "", ULong64_t nToProcess = 0);
   template<class F> TObject* Process(TTree& tree, F procFunc, ULong64_t nToProcess = 0);

   void SetNWorkers(unsigned n) { TMPClient::SetNWorkers(n); }
   unsigned GetNWorkers() const { return TMPClient::GetNWorkers(); }

private:
   template<class T> void Collect(std::vector<T> &reslist);
   template<class T> void HandlePoolCode(MPCodeBufPair &msg, TSocket *sender, std::vector<T> &reslist);

   void Reset();
   template<class T, class R> T Reduce(const std::vector<T> &objs, R redfunc);
   void ReplyToFuncResult(TSocket *s);
   void ReplyToIdle(TSocket *s);

   unsigned fNProcessed; ///< number of arguments already passed to the workers
   unsigned fNToProcess; ///< total number of arguments to pass to the workers

   enum class ETask : unsigned {
      kNoTask = 0,
      kMap,
      kMapWithArg,
      kMapRed,
      kMapRedWithArg,
      kProcRange,
      kProcFile,
   } fTask; ///< the kind of task that is being executed, if any
};


/************ TEMPLATE METHODS IMPLEMENTATION ******************/

//////////////////////////////////////////////////////////////////////////
/// Execute func (with no arguments) nTimes in parallel.
/// A vector containg executions' results is returned.
/// Functions that take more than zero arguments can be executed (with
/// fixed arguments) by wrapping them in a lambda or with std::bind.
template<class F>
auto TProcPool::Map(F func, unsigned nTimes) -> std::vector<decltype(func())>
{
   using retType = decltype(func());
   //prepare environment
   Reset();
   fTask = ETask::kMap;

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
   fTask = ETask::kNoTask;
   return reslist;
}


//////////////////////////////////////////////////////////////////////////
/// Execute func in parallel distributing the elements of the args collection between the workers.
/// See class description for the valid types of collections and containers that can be used.
/// A vector containing each execution's result is returned. The user is responsible of deleting
/// objects that might be created upon the execution of func, returned objects included.
/// **Note:** the collection of arguments is modified by Map and should be considered empty or otherwise
/// invalidated after Map's execution (std::move might be applied to it).
template<class F, class T>
auto TProcPool::Map(F func, T &args) -> std::vector < decltype(++(args.begin()), args.end(), func(args.front())) >
{
   std::vector<typename T::value_type> vargs(
      std::make_move_iterator(std::begin(args)),
      std::make_move_iterator(std::end(args))
   );
   const auto &reslist = Map(func, vargs);
   return reslist;
}


// tell doxygen to ignore this (\endcond closes the statement)
/// \cond
template<class F>
TObjArray TProcPool::Map(F func, TCollection &args)
{
   // check the function returns something from which we can build a TObject*
   static_assert(std::is_constructible<TObject *, typename std::result_of<F(TObject *)>::type>::value,
                 "func should return a pointer to TObject or derived classes");

   //build vector with same elements as args
   std::vector<TObject *> vargs(args.GetSize());
   auto it = vargs.begin();
   for (auto o : args) {
      *it = o;
      ++it;
   }

   //call Map
   const auto &reslist = Map(func, vargs);

   //build TObjArray with same elements as reslist
   TObjArray resarray;
   for (const auto &res : reslist)
      resarray.Add(res);
   return resarray;
}


template<class F, class T>
auto TProcPool::Map(F func, std::initializer_list<T> args) -> std::vector<decltype(func(*args.begin()))>
{
   std::vector<T> vargs(std::move(args));
   const auto &reslist = Map(func, vargs);
   return reslist;
}


// actual implementation of the Map method. all other calls with arguments eventually
// call this one
template<class F, class T>
auto TProcPool::Map(F func, std::vector<T> &args) -> std::vector<decltype(func(args.front()))>
{
   //check whether func is callable
   using retType = decltype(func(args.front()));
   //prepare environment
   Reset();
   fTask = ETask::kMapWithArg;

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
   fTask = ETask::kNoTask;
   return reslist;
}
// tell doxygen to stop ignoring code
/// \endcond


//////////////////////////////////////////////////////////////////////////
/// This method behaves just like Map, but an additional redfunc function
/// must be provided. redfunc is applied to the vector Map would return and
/// must return the same type as func. In practice, redfunc can be used to
/// "squash" the vector returned by Map into a single object by merging,
/// adding, mixing the elements of the vector.
template<class F, class R>
auto TProcPool::MapReduce(F func, unsigned nTimes, R redfunc) -> decltype(func())
{
   using retType = decltype(func());
   //prepare environment
   Reset();
   fTask = ETask::kMapRed;

   //fork max(nTimes, fNWorkers) times
   unsigned oldNWorkers = GetNWorkers();
   if (nTimes < oldNWorkers)
      SetNWorkers(nTimes);
   TPoolWorker<F, void, R> worker(func, redfunc);
   bool ok = Fork(worker);
   SetNWorkers(oldNWorkers);
   if (!ok) {
      std::cerr << "[E][C] Could not fork. Aborting operation\n";
      return retType();
   }

   //give workers their first task
   fNToProcess = nTimes;
   std::vector<retType> reslist;
   reslist.reserve(fNToProcess);
   fNProcessed = Broadcast(PoolCode::kExecFunc, fNToProcess);

   //collect results/give workers their next task
   Collect(reslist);

   //clean-up and return
   ReapWorkers();
   fTask = ETask::kNoTask;
   return redfunc(reslist);
}

//////////////////////////////////////////////////////////////////////////
/// This method behaves just like Map, but an additional redfunc function
/// must be provided. redfunc is applied to the vector Map would return and
/// must return the same type as func. In practice, redfunc can be used to
/// "squash" the vector returned by Map into a single object by merging,
/// adding, mixing the elements of the vector.
template<class F, class T, class R>
auto TProcPool::MapReduce(F func, T &args, R redfunc) -> decltype(++(args.begin()), args.end(), func(args.front()))
{
   std::vector<typename T::value_type> vargs(
      std::make_move_iterator(std::begin(args)),
      std::make_move_iterator(std::end(args))
   );
   const auto &reslist = MapReduce(func, vargs, redfunc);
   return reslist;
}

/// \cond doxygen should ignore these methods
template<class F, class R>
auto TProcPool::MapReduce(F func, TCollection &args, R redfunc) -> decltype(func(nullptr))
{
   //build vector with same elements as args
   std::vector<TObject *> vargs(args.GetSize());
   auto it = vargs.begin();
   for (auto o : args) {
      *it = o;
      ++it;
   }

   //call MapReduce
   auto res = MapReduce(func, vargs, redfunc); //TODO useless copying by value here, but it looks like the return type of this MapReduce is a reference otherwise

   return res;
}


template<class F, class T, class R>
auto TProcPool::MapReduce(F func, std::initializer_list<T> args, R redfunc) -> decltype(func(*args.begin()))
{
   std::vector<T> vargs(std::move(args));
   const auto &reslist = MapReduce(func, vargs, redfunc);
   return reslist;
}


template<class F, class T, class R>
auto TProcPool::MapReduce(F func, std::vector<T> &args, R redfunc) -> decltype(func(args.front()))
{
   using retType = decltype(func(args.front()));
   //prepare environment
   Reset();
   fTask = ETask::kMapRedWithArg;

   //fork max(args.size(), fNWorkers) times
   unsigned oldNWorkers = GetNWorkers();
   if (args.size() < oldNWorkers)
      SetNWorkers(args.size());
   TPoolWorker<F, T, R> worker(func, args, redfunc);
   bool ok = Fork(worker);
   SetNWorkers(oldNWorkers);
   if (!ok) {
      std::cerr << "[E][C] Could not fork. Aborting operation\n";
      return retType();
   }

   //give workers their first task
   fNToProcess = args.size();
   std::vector<retType> reslist;
   reslist.reserve(fNToProcess);
   std::vector<unsigned> range(fNToProcess);
   std::iota(range.begin(), range.end(), 0);
   fNProcessed = Broadcast(PoolCode::kExecFuncWithArg, range);

   //collect results/give workers their next task
   Collect(reslist);

   ReapWorkers();
   fTask = ETask::kNoTask;
   return redfunc(reslist);
}
/// \endcond


template<class F>
TObject* TProcPool::Process(const std::vector<std::string>& fileNames, F procFunc, const std::string& treeName, ULong64_t nToProcess)
{
   static_assert(std::is_constructible<TObject*, typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type>::value, "procFunc must return a pointer to a class inheriting from TObject, and must take a reference to TTreeReader as the only argument");

   //prepare environment
   Reset();
   unsigned nWorkers = GetNWorkers();

   //fork
   TPoolProcessor<F> worker(procFunc, fileNames, treeName, nWorkers, nToProcess/nWorkers);
   bool ok = Fork(worker);
   if(!ok) {
      std::cerr << "[E][C] Could not fork. Aborting operation\n";
      return nullptr;
   }

   if(fileNames.size() < nWorkers) {
      //TTree entry granularity. For each file, we divide entries equally between workers
      fTask = ETask::kProcRange;
      //Tell workers to start processing entries
      fNToProcess = nWorkers*fileNames.size(); //this is the total number of ranges that will be processed by all workers cumulatively
      std::vector<unsigned> args(nWorkers);
      std::iota(args.begin(), args.end(), 0);
      fNProcessed = Broadcast(PoolCode::kProcRange, args);
      if(fNProcessed < nWorkers)
         std::cerr << "[E][C] There was an error while sending tasks to workers. Some entries might not be processed.\n";
   } else {
      //file granularity. each worker processes one whole file as a single task
      fTask = ETask::kProcFile;
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
   TObject* res = PoolUtils::ReduceObjects(reslist);

   //clean-up and return
   ReapWorkers();
   fTask = ETask::kNoTask;
   return res;
}


template<class F>
TObject* TProcPool::Process(const std::string& fileName, F procFunc, const std::string& treeName, ULong64_t nToProcess)
{
   std::vector<std::string> singleFileName(1, fileName);
   return Process(singleFileName, procFunc, treeName, nToProcess);
}


template<class F>
TObject* TProcPool::Process(TFileCollection& files, F procFunc, const std::string& treeName, ULong64_t nToProcess)
{
   std::vector<std::string> fileNames(files.GetNFiles());
   unsigned count = 0;
   for(auto f : *static_cast<THashList*>(files.GetList()))
      fileNames[count++] = static_cast<TFileInfo*>(f)->GetCurrentUrl()->GetFile();

   return Process(fileNames, procFunc, treeName, nToProcess);
}


template<class F>
TObject* TProcPool::Process(TChain& files, F procFunc, const std::string& treeName, ULong64_t nToProcess)
{
   TObjArray* filelist = files.GetListOfFiles();
   std::vector<std::string> fileNames(filelist->GetEntries());
   unsigned count = 0;
   for(auto f : *filelist)
      fileNames[count++] = f->GetTitle();

   return Process(fileNames, procFunc, treeName, nToProcess);
}


template<class F>
TObject* TProcPool::Process(TTree& tree, F procFunc, ULong64_t nToProcess)
{
   static_assert(std::is_constructible<TObject*, typename std::result_of<F(std::reference_wrapper<TTreeReader>)>::type>::value, "procFunc must return a pointer to a class inheriting from TObject, and must take a reference to TTreeReader as the only argument");

   //prepare environment
   Reset();
   unsigned nWorkers = GetNWorkers();

   //fork
   TPoolProcessor<F> worker(procFunc, &tree, nWorkers, nToProcess/nWorkers);
   bool ok = Fork(worker);
   if(!ok) {
      std::cerr << "[E][C] Could not fork. Aborting operation\n";
      return nullptr;
   }

   //divide entries equally between workers
   fTask = ETask::kProcRange;

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
   TObject* res = PoolUtils::ReduceObjects(reslist);

   //clean-up and return
   ReapWorkers();
   fTask = ETask::kNoTask;
   return res;
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

/// Check that redfunc has the right signature and call it on objs
template<class T, class R>
T TProcPool::Reduce(const std::vector<T> &objs, R redfunc)
{
   // check we can apply reduce to objs
   static_assert(std::is_same<decltype(redfunc(objs)), T>::value, "redfunc does not have the correct signature");

   return redfunc(objs);
}

#endif
