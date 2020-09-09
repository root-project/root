// @(#)root/thread:$Id$
// Author: Xavier Valls September 2020

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TExecutor
#define ROOT_TExecutor

#include "ROOT/TExecutorBaseImpl.hxx"
#include "ROOT/TSequentialExecutor.hxx"
#ifdef R__USE_IMT
#include "ROOT/TThreadExecutor.hxx"
#endif
#include "ROOT/TProcessExecutor.hxx"
#include "TROOT.h"
#include "ExecutionPolicy.hxx"
#include <memory>
#include <thread>

namespace ROOT{

namespace Internal{
class TExecutor: public TExecutorBaseImpl<TExecutor> {
public:

   explicit TExecutor(unsigned nProcessingUnits = 0) :
      TExecutor(ROOT::IsImplicitMTEnabled() ? ROOT::Internal::ExecutionPolicy::kMultithread : ROOT::Internal::ExecutionPolicy::kSerial, nProcessingUnits) {}

   explicit TExecutor(ROOT::Internal::ExecutionPolicy execPolicy, unsigned nProcessingUnits = 0) : fExecPolicy(execPolicy) {
      fExecPolicy = execPolicy;
      switch(fExecPolicy) {
         case ROOT::Internal::ExecutionPolicy::kSerial:
            fSeqPool = std::unique_ptr<ROOT::TSequentialExecutor>(new ROOT::TSequentialExecutor());
            break;
#ifdef R__USE_IMT
         case ROOT::Internal::ExecutionPolicy::kMultithread:
            fThreadPool = std::unique_ptr<ROOT::TThreadExecutor>(new ROOT::TThreadExecutor(nProcessingUnits));
            break;
#endif
         case ROOT::Internal::ExecutionPolicy::kMultiprocess:
            fProcPool = std::unique_ptr<ROOT::TProcessExecutor>(new ROOT::TProcessExecutor(nProcessingUnits));
            break;
      }
   }

   TExecutor(TExecutor &) = delete;
   TExecutor &operator=(TExecutor &) = delete;

   using TExecutorBaseImpl<TExecutor>::Map;
   template<class F, class Cond = noReferenceCond<F>>
   auto Map(F func, unsigned nTimes) -> std::vector<typename std::result_of<F()>::type>;
   template<class F, class INTEGER, class Cond = noReferenceCond<F, INTEGER>>
   auto Map(F func, ROOT::TSeq<INTEGER> args) -> std::vector<typename std::result_of<F(INTEGER)>::type>;
   template<class F, class T, class Cond = noReferenceCond<F, T>>
   auto Map(F func, std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type>;

   // // MapReduce
   // // the late return types also check at compile-time whether redfunc is compatible with func,
   // // other than checking that func is compatible with the type of arguments.
   // // a static_assert check in TExecutor::Reduce is used to check that redfunc is compatible with the type returned by func
   using TExecutorBaseImpl<TExecutor>::MapReduce;
   template<class F, class R, class Cond = noReferenceCond<F>>
   auto MapReduce(F func, unsigned nTimes, R redfunc) -> typename std::result_of<F()>::type;
   template<class F, class R, class Cond = noReferenceCond<F>>
   auto MapReduce(F func, unsigned nTimes, R redfunc, unsigned nChunks) -> typename std::result_of<F()>::type;
   template<class F, class INTEGER, class R, class Cond = noReferenceCond<F, INTEGER>>
   auto MapReduce(F func, ROOT::TSeq<INTEGER> args, R redfunc, unsigned nChunks) -> typename std::result_of<F(INTEGER)>::type;
   /// \cond
   template<class F, class T, class R, class Cond = noReferenceCond<F, T>>
   auto MapReduce(F func, std::initializer_list<T> args, R redfunc, unsigned nChunks) -> typename std::result_of<F(T)>::type;
   /// \endcond
   template<class F, class T, class R, class Cond = noReferenceCond<F, T>>
   auto MapReduce(F func, std::vector<T> &args, R redfunc) -> typename std::result_of<F(T)>::type;
   template<class F, class T, class R, class Cond = noReferenceCond<F, T>>
   auto MapReduce(F func, std::vector<T> &args, R redfunc, unsigned nChunks) -> typename std::result_of<F(T)>::type;

   using TExecutorBaseImpl<TExecutor>::Reduce;
   template<class T, class R> auto Reduce(const std::vector<T> &objs, R redfunc) -> decltype(redfunc(objs));

protected:
   template<class F, class R, class Cond = noReferenceCond<F>>
   auto Map(F func, unsigned nTimes, R redfunc, unsigned nChunks) -> std::vector<typename std::result_of<F()>::type>;
   template<class F, class INTEGER, class R, class Cond = noReferenceCond<F, INTEGER>>
   auto Map(F func, ROOT::TSeq<INTEGER> args, R redfunc, unsigned nChunks) -> std::vector<typename std::result_of<F(INTEGER)>::type>;
   template<class F, class T, class R, class Cond = noReferenceCond<F, T>>
   auto Map(F func, std::vector<T> &args, R redfunc, unsigned nChunks) -> std::vector<typename std::result_of<F(T)>::type>;
   template<class F, class T, class R, class Cond = noReferenceCond<F, T>>
   auto Map(F func, std::initializer_list<T> args, R redfunc, unsigned nChunks) -> std::vector<typename std::result_of<F(T)>::type>;

private:
      ROOT::Internal::ExecutionPolicy fExecPolicy;
#ifdef R__USE_IMT
      std::unique_ptr<ROOT::TThreadExecutor> fThreadPool;
#endif
      std::unique_ptr<ROOT::TProcessExecutor> fProcPool;
      std::unique_ptr<ROOT::TSequentialExecutor> fSeqPool;
};


//////////////////////////////////////////////////////////////////////////
   /// Execute func (with no arguments) nTimes in parallel.
   /// A vector containg executions' results is returned.
   /// Functions that take more than zero arguments can be executed (with
   /// fixed arguments) by wrapping them in a lambda or with std::bind.
   template<class F, class Cond>
   auto TExecutor::Map(F func, unsigned nTimes) -> std::vector<typename std::result_of<F()>::type> {
      using retType = decltype(func());
      std::vector<retType> res;;
      switch(fExecPolicy){
         case ROOT::Internal::ExecutionPolicy::kSerial:
            res = fSeqPool->Map(func, nTimes);
            break;
#ifdef R__USE_IMT
         case ROOT::Internal::ExecutionPolicy::kMultithread:
            res = fThreadPool->Map(func, nTimes);
            break;
#endif
         case ROOT::Internal::ExecutionPolicy::kMultiprocess:
            res = fProcPool->Map(func, nTimes);
            break;
      }
      return res;
   }

   //////////////////////////////////////////////////////////////////////////
   /// Execute func in parallel, taking an element of a
   /// sequence as argument.
   /// A vector containg executions' results is returned.
   template<class F, class INTEGER, class Cond>
   auto TExecutor::Map(F func, ROOT::TSeq<INTEGER> args) -> std::vector<typename std::result_of<F(INTEGER)>::type> {
      using retType = decltype(func(args.front()));
      std::vector<retType> res;

      switch(fExecPolicy){
         case ROOT::Internal::ExecutionPolicy::kSerial:
            res = fSeqPool->Map(func, args);
            break;
#ifdef R__USE_IMT
         case ROOT::Internal::ExecutionPolicy::kMultithread:
            res = fThreadPool->Map(func, args);
            break;
#endif
         case ROOT::Internal::ExecutionPolicy::kMultiprocess:
            res = fProcPool->Map(func, args);
            break;
      }
      return res;
   }

   //////////////////////////////////////////////////////////////////////////
   /// Execute func (with no arguments) nTimes in parallel.
   /// Divides and groups the executions in nChunks (if it doesn't make sense will reduce the number of chunks) with partial reduction;
   /// A vector containg partial reductions' results is returned.
   template<class F, class R, class Cond>
   auto TExecutor::Map(F func, unsigned nTimes, R redfunc, unsigned nChunks) -> std::vector<typename std::result_of<F()>::type> {
      using retType = decltype(func());
      std::vector<retType> res;;
      switch(fExecPolicy){
         case ROOT::Internal::ExecutionPolicy::kSerial:
            res = fSeqPool->Map(func, nTimes, redfunc, 1);
            break;
#ifdef R__USE_IMT
         case ROOT::Internal::ExecutionPolicy::kMultithread:
            res = fThreadPool->Map(func, nTimes, redfunc, nChunks);
            break;
#endif
         case ROOT::Internal::ExecutionPolicy::kMultiprocess:
            res = fProcPool->Map(func, nTimes, redfunc, nChunks);
            break;
      }
      return res;
   }

   //////////////////////////////////////////////////////////////////////////
   /// Execute func in parallel, taking an element of an
   /// std::vector as argument.
   /// A vector containg executions' results is returned.
   // actual implementation of the Map method. all other calls with arguments eventually
   // call this one
   template<class F, class T, class Cond>
   auto TExecutor::Map(F func, std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type> {
      // //check whether func is callable
      using retType = decltype(func(args.front()));
      std::vector<retType> res;;
      switch(fExecPolicy){
         case ROOT::Internal::ExecutionPolicy::kSerial:
            res = fSeqPool->Map(func, args);
            break;
#ifdef R__USE_IMT
         case ROOT::Internal::ExecutionPolicy::kMultithread:
            res = fThreadPool->Map(func, args);
            break;
#endif
         case ROOT::Internal::ExecutionPolicy::kMultiprocess:
            res = fProcPool->Map(func, args);
            break;
      }
      return res;
   }

   //////////////////////////////////////////////////////////////////////////
   /// Execute func in parallel, taking an element of a
   /// sequence as argument.
   /// Divides and groups the executions in nChunks (if it doesn't make sense will reduce the number of chunks) with partial reduction\n
   /// A vector containg partial reductions' results is returned.
   template<class F, class INTEGER, class R, class Cond>
   auto TExecutor::Map(F func, ROOT::TSeq<INTEGER> args, R redfunc, unsigned nChunks) -> std::vector<typename std::result_of<F(INTEGER)>::type> {
      using retType = decltype(func(args.front()));
      std::vector<retType> res;;
      switch(fExecPolicy){
         case ROOT::Internal::ExecutionPolicy::kSerial:
            res = fSeqPool->Map(func, args, redfunc, 1);
            break;
#ifdef R__USE_IMT
         case ROOT::Internal::ExecutionPolicy::kMultithread:
            res = fThreadPool->Map(func, args, redfunc, nChunks);
            break;
#endif
         case ROOT::Internal::ExecutionPolicy::kMultiprocess:
            res = fProcPool->Map(func, args, redfunc, nChunks);
            break;
      }
      return res;
   }

   /// \cond
   //////////////////////////////////////////////////////////////////////////
   /// Execute func in parallel, taking an element of an
   /// std::vector as argument. Divides and groups the executions in nChunks with partial reduction.
   /// If it doesn't make sense will reduce the number of chunks.\n
   /// A vector containg partial reductions' results is returned.
   template<class F, class T, class R, class Cond>
   auto TExecutor::Map(F func, std::vector<T> &args, R redfunc, unsigned nChunks) -> std::vector<typename std::result_of<F(T)>::type> {
      using retType = decltype(func(args.front()));
      std::vector<retType> res;;
      switch(fExecPolicy){
         case ROOT::Internal::ExecutionPolicy::kSerial:
            res = fSeqPool->Map(func, args, redfunc, 1);
            break;
#ifdef R__USE_IMT
         case ROOT::Internal::ExecutionPolicy::kMultithread:
            res = fThreadPool->Map(func, args, redfunc, nChunks);
            break;
#endif
         case ROOT::Internal::ExecutionPolicy::kMultiprocess:
            res = fProcPool->Map(func, args, redfunc, nChunks);
            break;
      }
      return res;
   }

    //////////////////////////////////////////////////////////////////////////
   /// Execute func in parallel, taking an element of an
   /// std::initializer_list as an argument. Divides and groups the executions in nChunks with partial reduction.
   /// If it doesn't make sense will reduce the number of chunks.\n
   /// A vector containg partial reductions' results is returned.
   template<class F, class T, class R, class Cond>
   auto TExecutor::Map(F func, std::initializer_list<T> args, R redfunc, unsigned nChunks) -> std::vector<typename std::result_of<F(T)>::type> {
      std::vector<T> vargs(std::move(args));
      const auto &reslist = Map(func, vargs, redfunc, nChunks);
      return reslist;
   }
/// \endcond


   //////////////////////////////////////////////////////////////////////////
   /// This method behaves just like Map, but an additional redfunc function
   /// must be provided. redfunc is applied to the vector Map would return and
   /// must return the same type as func. In practice, redfunc can be used to
   /// "squash" the vector returned by Map into a single object by merging,
   /// adding, mixing the elements of the vector.\n
   /// The fourth argument indicates the number of chunks we want to divide our work in.
   template<class F, class R, class Cond>
   auto TExecutor::MapReduce(F func, unsigned nTimes, R redfunc) -> typename std::result_of<F()>::type {
      return Reduce(Map(func, nTimes), redfunc);
   }

   template<class F, class R, class Cond>
   auto TExecutor::MapReduce(F func, unsigned nTimes, R redfunc, unsigned nChunks) -> typename std::result_of<F()>::type {
      return Reduce(Map(func, nTimes, redfunc, nChunks), redfunc);
   }

   template<class F, class INTEGER, class R, class Cond>
   auto TExecutor::MapReduce(F func, ROOT::TSeq<INTEGER> args, R redfunc, unsigned nChunks) -> typename std::result_of<F(INTEGER)>::type {
      return Reduce(Map(func, args, redfunc, nChunks), redfunc);
   }
   /// \cond
   template<class F, class T, class R, class Cond>
   auto TExecutor::MapReduce(F func, std::initializer_list<T> args, R redfunc, unsigned nChunks) -> typename std::result_of<F(T)>::type {
      return Reduce(Map(func, args, redfunc, nChunks), redfunc);
   }
   /// \endcond

   template<class F, class T, class R, class Cond>
   auto TExecutor::MapReduce(F func, std::vector<T> &args, R redfunc) -> typename std::result_of<F(T)>::type {
      return Reduce(Map(func, args), redfunc);
   }

   template<class F, class T, class R, class Cond>
   auto TExecutor::MapReduce(F func, std::vector<T> &args, R redfunc, unsigned nChunks) -> typename std::result_of<F(T)>::type {
      return Reduce(Map(func, args, redfunc, nChunks), redfunc);
   }

   //////////////////////////////////////////////////////////////////////////
   /// "Reduce" an std::vector into a single object by passing a
   /// function as the second argument defining the reduction operation.
   template<class T, class R>
   auto TExecutor::Reduce(const std::vector<T> &objs, R redfunc) -> decltype(redfunc(objs))
   {
      // check we can apply reduce to objs
      static_assert(std::is_same<decltype(redfunc(objs)), T>::value, "redfunc does not have the correct signature");
      return redfunc(objs);
   }
}
}
#endif
