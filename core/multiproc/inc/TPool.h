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
#include "TPoolServer.h"
#include "TSocket.h"
#include "TObjArray.h"
#include "EPoolCode.h"
#include "EMPCode.h"
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
   /// This function calls Merge on the objects inside the std::vector
   TObject* ReduceObjects(const std::vector<TObject *>& objs);
}

//////////////////////////////////////////////////////////////////////////
///
/// This class provides a simple interface to execute the same task
/// multiple times in parallel, possibly with different arguments every
/// time. This mimics the behaviour of python's pool.Map method.
///
/// TPool::Map
/// The two main usages of the Map method are:
/// Map(F func, unsigned nTimes): func is executed nTimes with no arguments\n
/// Map(F func, T& args): func is executed on each element of the collection of arguments args\n
/// For either signature, func is executed as many times as needed by a pool
/// fNWorkers workers; the number of workers can be passed to the constructor
/// or set via SetNWorkers. fNWorkers defaults to the number of cores.\n
/// A collection containing the result of each execution is returned.
/// Note that the user is responsible for the deletion of any object that might
/// be created upon execution of func, returned objects included: TPool never
/// deletes what it returns, it simply forgets it.\n
/// Note that the usage of TPool::Map is indicated only when the task to be
/// executed takes more than a few seconds, otherwise the overhead introduced
/// by Map will outrun the benefits of parallel execution.
///
/// Valid argument types for Map:\n
/// func can be a lambda expression, an std::function, a loaded macro, a
/// functor class or a function, as long as it takes at most one argument.
/// args can be a standard container (vector, list, deque), an initializer list
/// or a pointer to a TCollection (TList*, TObjArray*, ...).\n
/// N.B. the version of TPool::Map that takes a TCollection* as argument incurs
/// in the overhead of copying data from the TCollection to an STL container. Only
/// use it when absolutely necessary.\n
/// Note that in cases where the function to be executed takes more than
/// zero/one argument but only zero/one is relevant for Map,
/// the function can easily be wrapped in a lambda or via std::bind to give
/// it the right signature.
/// N.B.2
/// The user should take care of initializing random seeds differently in each
/// process (e.g. using the process id in the seed).
///
/// Return value of Map:
/// If T derives from TCollection Map returns a TObjArray, otherwise it
/// returns a standard vector. In both cases, the elements of the container
/// will be the objects returned by func, if any.
///
/// Examples:
/// root[] TPool pool; auto hists = pool.Map(CreateHisto, 10);
/// root[] TPool pool(2); auto squares = pool.Map([](int a) { return a*a; }, {1,2,3});
///
/// TPool::MapReduce
/// This set of methods behaves exactly like Map, but takes an additional
/// function as a third argument. This function is applied to the set of
/// objects returned by the corresponding Map execution to "squash" them
/// to a single object.
///
/// Examples:
/// root[] TPool pool; auto ten = pool.MapReduce([]() { return 1; }, 10, [](std::vector<int> v) { return std::accumulate(v.begin(), v.end(), 0); })
/// root[] TPool pool; auto hist = pool.MapReduce(CreateAndFillHists, 10, PoolUtils::ReduceObjects);
///
//////////////////////////////////////////////////////////////////////////
class TPool : private TMPClient {
public:
   explicit TPool(unsigned nWorkers = 0); //default number of workers is the number of processors
   ~TPool() {}
   //it doesn't make sense for a TPool to be copied
   TPool(const TPool&) = delete;
   TPool& operator=(const TPool&) = delete;

   // Map
   //these late return types allow for a compile-time check of compatibility between function signatures and args,
   //and a compile-time check that the argument list implements a front() method (all STL sequence containers have it)
   template<class F> auto Map(F func, unsigned nTimes) -> std::vector<decltype(func())>;
   template<class F, class T> auto Map(F func, T& args) -> std::vector<decltype(func(args.front()))>;
   /// \cond doxygen should ignore these methods
   template<class F> TObjArray Map(F func, TCollection *args);
   template<class F, class T> auto Map(F func, std::initializer_list<T> args) -> std::vector<decltype(func(*args.begin()))>;
   template<class F, class T> auto Map(F func, std::vector<T>& args) -> std::vector<decltype(func(args.front()))>;
   /// \endcond

   // MapReduce
   // the late return types also check at compile-time whether redfunc is compatible with func,
   // other than checking that func is compatible with the type of arguments.
   // a static_assert check in TPool::Reduce is used to check that redfunc is compatible with the type returned by func
   template<class F, class R> auto MapReduce(F func, unsigned nTimes, R redfunc) -> decltype(func());
   template<class F, class T, class R> auto MapReduce(F func, T &args, R redfunc) -> decltype(func(args.front()));
   /// \cond doxygen should ignore these methods
   template<class F, class R> auto MapReduce(F func, TCollection *args, R redfunc) -> decltype(func(nullptr));
   template<class F, class T, class R> auto MapReduce(F func, std::initializer_list<T> args, R redfunc) -> decltype(func(*args.begin()));
   template<class F, class T, class R> auto MapReduce(F func, std::vector<T>& args, R redfunc) -> decltype(func(args.front()));
   /// \endcond

   inline void SetNWorkers(unsigned n) { TMPClient::SetNWorkers(n); }
   inline unsigned GetNWorkers() const { return TMPClient::GetNWorkers(); }
   /// Return true if this process is the parent/client/master process, false otherwise
   inline bool GetIsParent() const { return TMPClient::GetIsParent(); }

private:
   template<class T> void PoolCollect(std::vector<T> &reslist);
   template<class T> void HandlePoolCode(MPCodeBufPair& msg, TSocket *sender, std::vector<T> &reslist);

   //this version reads classes from the message
   template<class T,
      typename std::enable_if<std::is_class<T>::value>::type* = nullptr>
         T ReadBuffer(std::shared_ptr<TBufferFile> buf);
   //this version reads built-in types from the message
   template<class T,
      typename std::enable_if<!std::is_class<T>::value && !std::is_pointer<T>::value>::type* = nullptr>
         T ReadBuffer(std::shared_ptr<TBufferFile> buf);
   //this version reads std::string and c-strings from the message
   template<class T,
      typename std::enable_if<std::is_same<const char *, T>::value>::type* = nullptr>
         T ReadBuffer(std::shared_ptr<TBufferFile> buf);
   //this version reads a TObject* from the message
   template<class T,
      typename std::enable_if<std::is_pointer<T>::value && std::is_constructible<TObject *, T>::value>::type* = nullptr>
         T ReadBuffer(std::shared_ptr<TBufferFile> buf);
   
   void Reset();
   template<class T, class R> T Reduce(const std::vector<T>& objs, R redfunc);
   void ReplyToResult(TSocket *s);

   unsigned fNProcessed; ///< number of arguments already passed to the workers
   unsigned fNToProcess; ///< total number of arguments to pass to the workers
   bool  fWithArg; ///< true if arguments are passed to Map
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
   //check whether func is callable
   using retType = decltype(func());
   //prepare environment
   Reset();
   fWithArg = false;

   //fork
   unsigned ok = Fork(new TPoolServer<F>(func));
   if (!ok) {
      std::cerr << "[E][C] Could not fork. Aborting operation\n";
      return std::vector<retType>();
   }
   if (!GetIsParent()) //servers return here
      return std::vector<retType>();

   //collect results  
   fNToProcess = nTimes;
   std::vector<retType> reslist;
   reslist.reserve(fNToProcess);
   fNProcessed = Broadcast(EPoolCode::kExecFunc, fNToProcess);
   if(fNProcessed < GetNWorkers() && fNProcessed == fNToProcess) { //this happens when fNToProcess is less than the number of workers
      //tell idle workers to shutdown. Idle workers are the ones whose sockets are still active after the broadcast
      TIter next(GetMonitor().GetListOfActives());
      TSocket *s = nullptr;
      while((s = (TSocket*)next())) {
         MPSend(s, EMPCode::kShutdownOrder);
      }
   }
   PoolCollect(reslist);
   ReapServers();
   return reslist;
}


//////////////////////////////////////////////////////////////////////////
/// Execute func in parallel distributing the elements of the args collection between the workers.
/// See class description for the valid types of collections and containers that can be used.
/// A vector containing each execution's result is returned. The user is responsible of deleting
/// objects that might be created upon the execution of func, returned objects included.
/// N.B.
/// The collection of arguments is modified by Map and should be considered empty or otherwise
/// invalidated after Map's execution (std::move might be applied to it).\n
template<class F, class T>
auto TPool::Map(F func, T &args) -> std::vector<decltype(func(args.front()))>
{
   std::vector<typename T::value_type> vargs(std::make_move_iterator(std::begin(args)), std::make_move_iterator(std::end(args)));
   const auto& reslist = Map(func, vargs);
   return reslist;
}


// tell doxygen to ignore this (\endcond closes the statement)
/// \cond
template<class F>
TObjArray TPool::Map(F func, TCollection *args)
{
   // check the function returns something from which we can build a TObject*
   static_assert(std::is_constructible<TObject*, typename std::result_of<F(TObject*)>::type>::value,
      "func should return a TObject*");
   
   //build vector with same elements as args
   std::vector<TObject *> vargs(args->GetSize());
   auto it = vargs.begin();
   TIter next(args);
   TObject *o;
   while ((o = next())) {
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

   //fork
   //N.B. from this point onwards, args is filled with undefined (but valid) values, since TPoolServer moved its content away
   unsigned ok = Fork(new TPoolServer<F, T>(func, args));
   if (!ok)
      return std::vector<retType>();
   if (!GetIsParent()) //servers return here
      return std::vector<retType>();

   //collect results
   fNToProcess = args.size();
   std::vector<retType> reslist;
   reslist.reserve(fNToProcess);
   std::vector<unsigned> range(fNToProcess);
   std::iota(range.begin(), range.end(), 0);
   fNProcessed = Broadcast(EPoolCode::kExecFuncWithArg, range);
   if(fNProcessed < GetNWorkers() && fNProcessed == fNToProcess) { //this happens when fNToProcess is less than the number of workers
      //tell idle workers to shutdown. Idle workers are the ones whose sockets are still active after the broadcast
      TIter next(GetMonitor().GetListOfActives());
      TSocket *s = nullptr;
      while((s = (TSocket*)next())) {
         MPSend(s, EMPCode::kShutdownOrder);
      }
   }
   PoolCollect(reslist);
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
   const auto& reslist = Map(func, nTimes);
   auto res = Reduce(reslist, redfunc);
   return res;
}

//////////////////////////////////////////////////////////////////////////
/// This method behaves just like Map, but an additional redfunc function
/// must be provided. redfunc is applied to the vector Map would return and
/// must return the same type as func. In practice, redfunc can be used to
/// "squash" the vector returned by Map into a single object by merging,
/// adding, mixing the elements of the vector.
template<class F, class T, class R>
auto TPool::MapReduce(F func, T &args, R redfunc) -> decltype(func(args.front()))
{
   const auto& reslist = Map(func, args);
   auto res = Reduce(reslist, redfunc);
   return res;
}

/// \cond doxygen should ignore these methods
template<class F, class R>
auto TPool::MapReduce(F func, TCollection *args, R redfunc) -> decltype(func(nullptr))
{
   //build vector with same elements as args
   std::vector<TObject *> vargs(args->GetSize());
   TIter next(args);
   auto it = vargs.begin();
   TObject *o;
   while ((o = next())) {
      *it = o;
      ++it;
   }

   //call Map
   const auto &reslist = Map(func, vargs);

   //call Reduce
   auto res = Reduce(reslist, redfunc);

   return res;
}


template<class F, class T, class R>
auto TPool::MapReduce(F func, std::initializer_list<T> args, R redfunc) -> decltype(func(*args.begin()))
{
   const auto& reslist = Map(func, args);
   auto res = Reduce(reslist, redfunc);
   return res;
}


template<class F, class T, class R>
auto TPool::MapReduce(F func, std::vector<T> &args, R redfunc) -> decltype(func(args.front()))
{
   const auto& reslist = Map(func, args);
   auto res = Reduce(reslist, redfunc);
   return res;
}
/// \endcond


//////////////////////////////////////////////////////////////////////////
/// Listen for messages sent by the workers and call the appropriate handler function.
/// TPool::HandlePoolCode is called on messages with a code < 1000 and
/// TMPClient::HandleMPCode is called on messages with a code >= 1000.
template<class T>
void TPool::PoolCollect(std::vector<T> &reslist)
{
   TMonitor &mon = GetMonitor();
   mon.ActivateAll();
   while (mon.GetActive() > 0) {
      TSocket *s = mon.Select();
      MPCodeBufPair msg = MPRecv(s);
      if(msg.first == EMPCode::kRecvError) {
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
void TPool::HandlePoolCode(MPCodeBufPair& msg, TSocket *s, std::vector<T> &reslist)
{
   unsigned code = msg.first;
   if (code == EPoolCode::kFuncResult) {
      reslist.push_back(std::move(ReadBuffer<T>(msg.second)));
      ReplyToResult(s);
   } else {
      // UNKNOWN CODE
      std::cerr << "[W][C] unknown code received from server. code=" << code << "\n";
   }
}


//////////////////////////////////////////////////////////////////////////
/// One of the template functions used to read objects from messages.
/// Different implementations are provided for different types of objects:
/// classes, non-pointer built-ins and const char*. Reading pointers is
/// not implemented.
template<class T, typename std::enable_if<std::is_class<T>::value>::type*>
T TPool::ReadBuffer(std::shared_ptr<TBufferFile> buf)
{
   TClass* c = TClass::GetClass(typeid(T));
   T* objp = (T*)buf->ReadObjectAny(c);
   T obj = *objp; //yes, this is slow. how do we return a T without leaking memory otherwise?
   delete objp;
   return obj;
}

/// \cond
template<class T, typename std::enable_if<!std::is_class<T>::value && !std::is_pointer<T>::value>::type*>
T TPool::ReadBuffer(std::shared_ptr<TBufferFile> buf)
{
   //read built-in type
   T obj;
   *(buf) >> obj;
   return obj;
}

template<class T, typename std::enable_if<std::is_same<const char *, T>::value>::type*>
T TPool::ReadBuffer(std::shared_ptr<TBufferFile> buf)
{
   //read c-string
   char *c = new char[buf->BufferSize()];
   buf->ReadString(c, buf->BufferSize());
   return c;
}

template<class T, typename std::enable_if<std::is_pointer<T>::value && std::is_constructible<TObject *, T>::value>::type*>
T TPool::ReadBuffer(std::shared_ptr<TBufferFile> buf)
{
   //read TObject*
   return (TObject*)buf->ReadObjectAny(TClass::GetClass(typeid(typename std::remove_pointer<T>::type)));
}
/// \endcond

/// Check that redfunc has the right signature and call it on objs
template<class T, class R>
T TPool::Reduce(const std::vector<T>& objs, R redfunc)
{
   // check we can apply reduce to objs
   static_assert(std::is_same<decltype(redfunc(objs)), T>::value, "redfunc does not have the correct signature");

   return redfunc(objs);
}

#endif
