/* @(#)root/multiproc:$Id$ */
// Author: Enrico Guiraud July 2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPool
#define ROOT_TPool

#include "TMPClient.h"
#include "TCollection.h"
#include "MPSendRecv.h"
#include "TPoolWorker.h"
#include "TSocket.h"
#include "TObjArray.h"
#include "PoolCode.h"
#include "MPCode.h"
#include "TClass.h"
#include <vector>
#include <initializer_list>
#include <type_traits> //std::result_of, std::enable_if
#include <typeinfo> //typeid
#include <iostream>
#include <numeric> //std::iota

//////////////////////////////////////////////////////////////////////////
///
/// This namespace contains pre-defined functions to be used in
/// conjuction with TPool::Map and TPool::MapReduce.
///
//////////////////////////////////////////////////////////////////////////
namespace PoolUtils {
   TObject *ReduceObjects(const std::vector<TObject *> &objs);
}

class TPool : private TMPClient {
public:
   explicit TPool(unsigned nWorkers = 0); //default number of workers is the number of processors
   ~TPool() {}
   //it doesn't make sense for a TPool to be copied
   TPool(const TPool &) = delete;
   TPool &operator=(const TPool &) = delete;

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
   // a static_assert check in TPool::Reduce is used to check that redfunc is compatible with the type returned by func
   template<class F, class R> auto MapReduce(F func, unsigned nTimes, R redfunc) -> decltype(func());
   template<class F, class T, class R> auto MapReduce(F func, T &args, R redfunc) -> decltype(++(args.begin()), args.end(), func(args.front()));
   /// \cond doxygen should ignore these methods
   template<class F, class R> auto MapReduce(F func, TCollection &args, R redfunc) -> decltype(func(nullptr));
   template<class F, class T, class R> auto MapReduce(F func, std::initializer_list<T> args, R redfunc) -> decltype(func(*args.begin()));
   template<class F, class T, class R> auto MapReduce(F func, std::vector<T> &args, R redfunc) -> decltype(func(args.front()));
   /// \endcond

   inline void SetNWorkers(unsigned n) { TMPClient::SetNWorkers(n); }
   inline unsigned GetNWorkers() const { return TMPClient::GetNWorkers(); }
   /// Return true if this process is the parent/client/master process, false otherwise

private:
   template<class T> void Collect(std::vector<T> &reslist);
   template<class T> void HandlePoolCode(MPCodeBufPair &msg, TSocket *sender, std::vector<T> &reslist);

   void Reset();
   template<class T, class R> T Reduce(const std::vector<T> &objs, R redfunc);
   void ReplyToResult(TSocket *s);
   void ReplyToIdle(TSocket *s);

   unsigned fNProcessed; ///< number of arguments already passed to the workers
   unsigned fNToProcess; ///< total number of arguments to pass to the workers
   bool fWithArg; ///< true if arguments are passed to Map
   bool fWithReduce; ///< true if MapReduce has been called
};


/************ TEMPLATE METHODS IMPLEMENTATION ******************/

//////////////////////////////////////////////////////////////////////////
/// Execute func (with no arguments) nTimes in parallel.
/// A vector containg executions' results is returned.
/// Functions that take more than zero arguments can be executed (with
/// fixed arguments) by wrapping them in a lambda or with std::bind.
template<class F>
auto TPool::Map(F func, unsigned nTimes) -> std::vector<decltype(func())>
{
   using retType = decltype(func());
   //prepare environment
   Reset();
   fWithArg = false;

   //fork max(nTimes, fNWorkers) times
   unsigned oldNWorkers = GetNWorkers();
   if (nTimes < oldNWorkers)
      SetNWorkers(nTimes);
   TPoolWorker<F> worker(func);
   unsigned ok = Fork(worker);
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
   ReapServers();
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
auto TPool::Map(F func, T &args) -> std::vector < decltype(++(args.begin()), args.end(), func(args.front())) >
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
TObjArray TPool::Map(F func, TCollection &args)
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
auto TPool::Map(F func, std::initializer_list<T> args) -> std::vector<decltype(func(*args.begin()))>
{
   std::vector<T> vargs(std::move(args));
   const auto &reslist = Map(func, vargs);
   return reslist;
}


// actual implementation of the Map method. all other calls with arguments eventually
// call this one
template<class F, class T>
auto TPool::Map(F func, std::vector<T> &args) -> std::vector<decltype(func(args.front()))>
{
   //check whether func is callable
   using retType = decltype(func(args.front()));
   //prepare environment
   Reset();
   fWithArg = true;

   //fork max(args.size(), fNWorkers) times
   //N.B. from this point onwards, args is filled with undefined (but valid) values, since TPoolWorker moved its content away
   unsigned oldNWorkers = GetNWorkers();
   if (args.size() < oldNWorkers)
      SetNWorkers(args.size());
   TPoolWorker<F, T> worker(func, args);
   unsigned ok = Fork(worker);
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
   ReapServers();
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
auto TPool::MapReduce(F func, unsigned nTimes, R redfunc) -> decltype(func())
{
   using retType = decltype(func());
   //prepare environment
   Reset();
   fWithArg = false;
   fWithReduce = true;

   //fork max(nTimes, fNWorkers) times
   unsigned oldNWorkers = GetNWorkers();
   if (nTimes < oldNWorkers)
      SetNWorkers(nTimes);
   TPoolWorker<F, void, R> worker(func, redfunc);
   unsigned ok = Fork(worker);
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
   ReapServers();
   return redfunc(reslist);
}

//////////////////////////////////////////////////////////////////////////
/// This method behaves just like Map, but an additional redfunc function
/// must be provided. redfunc is applied to the vector Map would return and
/// must return the same type as func. In practice, redfunc can be used to
/// "squash" the vector returned by Map into a single object by merging,
/// adding, mixing the elements of the vector.
template<class F, class T, class R>
auto TPool::MapReduce(F func, T &args, R redfunc) -> decltype(++(args.begin()), args.end(), func(args.front()))
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
auto TPool::MapReduce(F func, TCollection &args, R redfunc) -> decltype(func(nullptr))
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
auto TPool::MapReduce(F func, std::initializer_list<T> args, R redfunc) -> decltype(func(*args.begin()))
{
   std::vector<T> vargs(std::move(args));
   const auto &reslist = MapReduce(func, vargs, redfunc);
   return reslist;
}


template<class F, class T, class R>
auto TPool::MapReduce(F func, std::vector<T> &args, R redfunc) -> decltype(func(args.front()))
{
   using retType = decltype(func(args.front()));
   //prepare environment
   Reset();
   fWithArg = true;
   fWithReduce = true;

   //fork max(args.size(), fNWorkers) times
   unsigned oldNWorkers = GetNWorkers();
   if (args.size() < oldNWorkers)
      SetNWorkers(args.size());
   TPoolWorker<F, T, R> worker(func, args, redfunc);
   unsigned ok = Fork(worker);
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

   ReapServers();
   return redfunc(reslist);
}
/// \endcond


//////////////////////////////////////////////////////////////////////////
/// Listen for messages sent by the workers and call the appropriate handler function.
/// TPool::HandlePoolCode is called on messages with a code < 1000 and
/// TMPClient::HandleMPCode is called on messages with a code >= 1000.
template<class T>
void TPool::Collect(std::vector<T> &reslist)
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
/// Handle message and reply to the worker (actual code implemented in ReplyToResult
template<class T>
void TPool::HandlePoolCode(MPCodeBufPair &msg, TSocket *s, std::vector<T> &reslist)
{
   unsigned code = msg.first;
   if (code == PoolCode::kFuncResult) {
      reslist.push_back(std::move(ReadBuffer<T>(msg.second.get())));
      ReplyToResult(s);
   } else if (code == PoolCode::kIdling) {
      ReplyToIdle(s);
   } else {
      // UNKNOWN CODE
      std::cerr << "[W][C] unknown code received from server. code=" << code << "\n";
   }
}

/// Check that redfunc has the right signature and call it on objs
template<class T, class R>
T TPool::Reduce(const std::vector<T> &objs, R redfunc)
{
   // check we can apply reduce to objs
   static_assert(std::is_same<decltype(redfunc(objs)), T>::value, "redfunc does not have the correct signature");

   return redfunc(objs);
}

#endif
