// @(#)root/thread:$Id$
// Author: Xavier Valls November 2017

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TSequentialExecutor
#define ROOT_TSequentialExecutor

#include "RConfigure.h"

#include "ROOT/TExecutor.hxx"
#include <numeric>
#include <vector>

namespace ROOT {

   class TSequentialExecutor: public TExecutor<TSequentialExecutor> {
   public:
      explicit TSequentialExecutor(){};

      TSequentialExecutor(TSequentialExecutor &) = delete;
      TSequentialExecutor &operator=(TSequentialExecutor &) = delete;

      template<class F>
      void Foreach(F func, unsigned nTimes);
      template<class F, class INTEGER>
      void Foreach(F func, ROOT::TSeq<INTEGER> args);
      /// \cond
      template<class F, class T>
      void Foreach(F func, std::initializer_list<T> args);
      /// \endcond
      template<class F, class T>
      void Foreach(F func, std::vector<T> &args);

      using TExecutor<TSequentialExecutor>::Map;
      template<class F, class Cond = noReferenceCond<F>>
      auto Map(F func, unsigned nTimes) -> std::vector<typename std::result_of<F()>::type>;
      template<class F, class INTEGER, class Cond = noReferenceCond<F, INTEGER>>
      auto Map(F func, ROOT::TSeq<INTEGER> args) -> std::vector<typename std::result_of<F(INTEGER)>::type>;
      template<class F, class T, class Cond = noReferenceCond<F, T>>
      auto Map(F func, std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type>;

      // // MapReduce
      // // the late return types also check at compile-time whether redfunc is compatible with func,
      // // other than checking that func is compatible with the type of arguments.
      // // a static_assert check in TSequentialExecutor::Reduce is used to check that redfunc is compatible with the type returned by func
      using TExecutor<TSequentialExecutor>::MapReduce;
      template<class F, class R, class Cond = noReferenceCond<F>>
      auto MapReduce(F func, unsigned nTimes, R redfunc) -> typename std::result_of<F()>::type;
      template<class F, class T, class R, class Cond = noReferenceCond<F, T>>
      auto MapReduce(F func, std::vector<T> &args, R redfunc) -> typename std::result_of<F(T)>::type;
      
      using TExecutor<TSequentialExecutor>::Reduce;
      template<class T, class R> auto Reduce(const std::vector<T> &objs, R redfunc) -> decltype(redfunc(objs));
   };

   /************ TEMPLATE METHODS IMPLEMENTATION ******************/

   //////////////////////////////////////////////////////////////////////////
   /// Execute func (with no arguments) nTimes.
   /// Functions that take more than zero arguments can be executed (with
   /// fixed arguments) by wrapping them in a lambda or with std::bind.
   template<class F>
   void TSequentialExecutor::Foreach(F func, unsigned nTimes) {
      for (auto i = 0U; i < nTimes; ++i) func();
   }

   //////////////////////////////////////////////////////////////////////////
   /// Execute func, taking an element of a
   /// sequence as argument.
   template<class F, class INTEGER>
   void TSequentialExecutor::Foreach(F func, ROOT::TSeq<INTEGER> args) {
       for(auto i : args) func(i);
   }

   /// \cond
   //////////////////////////////////////////////////////////////////////////
   /// Execute func, taking an element of a
   /// initializer_list as argument.
   template<class F, class T>
   void TSequentialExecutor::Foreach(F func, std::initializer_list<T> args) {
       std::vector<T> vargs(std::move(args));
       Foreach(func, vargs);
   }
   /// \endcond

   //////////////////////////////////////////////////////////////////////////
   /// Execute func, taking an element of an
   /// std::vector as argument.
   template<class F, class T>
   void TSequentialExecutor::Foreach(F func, std::vector<T> &args) {
        unsigned int nToProcess = args.size();
        for(auto i: ROOT::TSeqI(nToProcess)) func(args[i]);
   }

   //////////////////////////////////////////////////////////////////////////
   /// Execute func (with no arguments) nTimes.
   /// A vector containg executions' results is returned.
   /// Functions that take more than zero arguments can be executed (with
   /// fixed arguments) by wrapping them in a lambda or with std::bind.
   template<class F, class Cond>
   auto TSequentialExecutor::Map(F func, unsigned nTimes) -> std::vector<typename std::result_of<F()>::type> {
      using retType = decltype(func());
      std::vector<retType> reslist(nTimes);
      for(auto i: ROOT::TSeqI(nTimes)) reslist[i] = func();
      return reslist;
   }

   //////////////////////////////////////////////////////////////////////////
   /// Execute func, taking an element of a
   /// sequence as argument.
   /// A vector containg executions' results is returned.
   template<class F, class INTEGER, class Cond>
   auto TSequentialExecutor::Map(F func, ROOT::TSeq<INTEGER> args) -> std::vector<typename std::result_of<F(INTEGER)>::type> {
      using retType = decltype(func(*args.begin()));
      std::vector<retType> reslist(args.size());
      for(auto i: args) reslist[i] = func(i);
      return reslist;
   }

   //////////////////////////////////////////////////////////////////////////
   /// Execute func, taking an element of an
   /// std::vector as argument.
   /// A vector containg executions' results is returned.
   // actual implementation of the Map method. all other calls with arguments eventually
   // call this one
   template<class F, class T, class Cond>
   auto TSequentialExecutor::Map(F func, std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type> {
      // //check whether func is callable
      using retType = decltype(func(args.front()));
      unsigned int nToProcess = args.size();
      std::vector<retType> reslist(nToProcess);
      for(auto i: ROOT::TSeqI(nToProcess)) reslist[i] = func(args[i]);
      return reslist;
   }

   template<class F, class R, class Cond>
   auto TSequentialExecutor::MapReduce(F func, unsigned nTimes, R redfunc) -> typename std::result_of<F()>::type {
      return Reduce(Map(func, nTimes), redfunc);
   }

   template<class F, class T, class R, class Cond>
   auto TSequentialExecutor::MapReduce(F func, std::vector<T> &args, R redfunc) -> typename std::result_of<F(T)>::type {
      return Reduce(Map(func, args), redfunc);
   }

   //////////////////////////////////////////////////////////////////////////
   /// "Reduce" an std::vector into a single object by passing a
   /// function as the second argument defining the reduction operation.
   template<class T, class R>
   auto TSequentialExecutor::Reduce(const std::vector<T> &objs, R redfunc) -> decltype(redfunc(objs))
   {
      // check we can apply reduce to objs
      static_assert(std::is_same<decltype(redfunc(objs)), T>::value, "redfunc does not have the correct signature");
      return redfunc(objs);
   }

} // namespace ROOT
#endif
