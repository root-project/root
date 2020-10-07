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

#include "ROOT/TExecutorCRTP.hxx"
#include "ExecutionPolicy.hxx"
#include <numeric>
#include <vector>

namespace ROOT {

   class TSequentialExecutor: public TExecutorCRTP<TSequentialExecutor> {
   public:
      explicit TSequentialExecutor(){};
      explicit TSequentialExecutor(ROOT::Internal::ExecutionPolicy, unsigned): TSequentialExecutor(){};

      TSequentialExecutor(const TSequentialExecutor &) = delete;
      TSequentialExecutor &operator=(const TSequentialExecutor &) = delete;

      template<class F>
      void Foreach(F func, unsigned nTimes);
      template<class F, class INTEGER>
      void Foreach(F func, ROOT::TSeq<INTEGER> args);
      template<class F, class T>
      void Foreach(F func, std::initializer_list<T> args);
      template<class F, class T>
      void Foreach(F func, std::vector<T> &args);
      template<class F, class T>
      void Foreach(F func, const std::vector<T> &args);

      using TExecutorCRTP<TSequentialExecutor>::Map;
      template<class F, class Cond = noReferenceCond<F>>
      auto Map(F func, unsigned nTimes) -> std::vector<typename std::result_of<F()>::type>;
      template<class F, class INTEGER, class Cond = noReferenceCond<F, INTEGER>>
      auto Map(F func, ROOT::TSeq<INTEGER> args) -> std::vector<typename std::result_of<F(INTEGER)>::type>;
      template<class F, class T, class Cond = noReferenceCond<F, T>>
      auto Map(F func, std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type>;
      template<class F, class T, class Cond = noReferenceCond<F, T>>
      auto Map(F func, const std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type>;

      // // MapReduce
      // // the late return types also check at compile-time whether redfunc is compatible with func,
      // // other than checking that func is compatible with the type of arguments.
      // // a static_assert check in TSequentialExecutor::Reduce is used to check that redfunc is compatible with the type returned by func
      using TExecutorCRTP<TSequentialExecutor>::MapReduce;
      template<class F, class R, class Cond = noReferenceCond<F>>
      auto MapReduce(F func, unsigned nTimes, R redfunc) -> typename std::result_of<F()>::type;
      template<class F, class T, class R, class Cond = noReferenceCond<F, T>>
      auto MapReduce(F func, std::vector<T> &args, R redfunc) -> typename std::result_of<F(T)>::type;
      template<class F, class T, class R, class Cond = noReferenceCond<F, T>>
      auto MapReduce(F func, const std::vector<T> &args, R redfunc) -> typename std::result_of<F(T)>::type;

      using TExecutorCRTP<TSequentialExecutor>::Reduce;

      //////////////////////////////////////////////////////////////////////////
      /// \brief Return the number of workers in the sequential executor: a single one.
      ///
      /// \return The number of workers in the pool, one.
      unsigned GetPoolSize() { return 1u; }
   };

   /************ TEMPLATE METHODS IMPLEMENTATION ******************/

   //////////////////////////////////////////////////////////////////////////
   /// \brief Execute a function without arguments several times, dividing the execution in nChunks.
   ///
   /// \param func Function to be executed.
   /// \param nTimes Number of times function should be called.
   template<class F>
   void TSequentialExecutor::Foreach(F func, unsigned nTimes) {
      for (auto i = 0U; i < nTimes; ++i) func();
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Execute a function over a sequence of indexes, dividing the execution in nChunks.
   ///
   /// \param func Function to be executed. Must take an element of the sequence passed assecond argument as a parameter.
   /// \param args Sequence of indexes to execute `func` on.
   template<class F, class INTEGER>
   void TSequentialExecutor::Foreach(F func, ROOT::TSeq<INTEGER> args) {
         for(auto i : args) func(i);
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Execute a function over the elements of an initializer_list, dividing the execution in nChunks.
   ///
   /// \param func Function to be executed on the elements of the initializer_list passed as second parameter.
   /// \param args initializer_list for a vector to apply `func` on.
   template<class F, class T>
   void TSequentialExecutor::Foreach(F func, std::initializer_list<T> args) {
         std::vector<T> vargs(std::move(args));
         Foreach(func, vargs);
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Execute a function over the elements of a vector, dividing the execution in nChunks.
   ///
   /// \param func Function to be executed on the elements of the vector passed as second parameter.
   /// \param args Vector of elements passed as an argument to `func`.
   template<class F, class T>
   void TSequentialExecutor::Foreach(F func, std::vector<T> &args) {
         unsigned int nToProcess = args.size();
         for(auto i: ROOT::TSeqI(nToProcess)) func(args[i]);
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Execute a function over the elements of an immutable vector, dividing the execution in nChunks.
   ///
   /// \param func Function to be executed on the elements of the immutable vector passed as second parameter.
   /// \param args Immutable vector of elements passed as an argument to `func`.
   template<class F, class T>
   void TSequentialExecutor::Foreach(F func, const std::vector<T> &args) {
         unsigned int nToProcess = args.size();
         for(auto i: ROOT::TSeqI(nToProcess)) func(args[i]);
   }

   //////////////////////////////////////////////////////////////////////////
   /// \copydoc TExecutorCRTP::Map(F func,unsigned nTimes)
   template<class F, class Cond>
   auto TSequentialExecutor::Map(F func, unsigned nTimes) -> std::vector<typename std::result_of<F()>::type> {
      using retType = decltype(func());
      std::vector<retType> reslist(nTimes);
      for(auto i: ROOT::TSeqI(nTimes)) reslist[i] = func();
      return reslist;
   }

   //////////////////////////////////////////////////////////////////////////
   /// \copydoc TExecutorCRTP::Map(F func,ROOT::TSeq<INTEGER> args)
   template<class F, class INTEGER, class Cond>
   auto TSequentialExecutor::Map(F func, ROOT::TSeq<INTEGER> args) -> std::vector<typename std::result_of<F(INTEGER)>::type> {
      using retType = decltype(func(*args.begin()));
      std::vector<retType> reslist(args.size());
      for(auto i: args) reslist[i] = func(i);
      return reslist;
   }

   //////////////////////////////////////////////////////////////////////////
   /// \copydoc TExecutorCRTP::Map(F func,std::vector<T> &args)
   template<class F, class T, class Cond>
   auto TSequentialExecutor::Map(F func, std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type> {
      // //check whether func is callable
      using retType = decltype(func(args.front()));
      unsigned int nToProcess = args.size();
      std::vector<retType> reslist(nToProcess);
      for(auto i: ROOT::TSeqI(nToProcess)) reslist[i] = func(args[i]);
      return reslist;
   }

   //////////////////////////////////////////////////////////////////////////
   /// \copydoc TExecutorCRTP::Map(F func,const std::vector<T> &args)
   template<class F, class T, class Cond>
   auto TSequentialExecutor::Map(F func, const std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type> {
      // //check whether func is callable
      using retType = decltype(func(args.front()));
      unsigned int nToProcess = args.size();
      std::vector<retType> reslist(nToProcess);
      for(auto i: ROOT::TSeqI(nToProcess)) reslist[i] = func(args[i]);
      return reslist;
   }

   //////////////////////////////////////////////////////////////////////////
   /// \copydoc TExecutorCRTP::MapReduce(F func,unsigned nTimes,R redfunc)
   template<class F, class R, class Cond>
   auto TSequentialExecutor::MapReduce(F func, unsigned nTimes, R redfunc) -> typename std::result_of<F()>::type {
      return Reduce(Map(func, nTimes), redfunc);
   }

   //////////////////////////////////////////////////////////////////////////
   /// \copydoc TExecutorCRTP::MapReduce(F func,std::vector<T> &args,R redfunc)
   template<class F, class T, class R, class Cond>
   auto TSequentialExecutor::MapReduce(F func, std::vector<T> &args, R redfunc) -> typename std::result_of<F(T)>::type {
      return Reduce(Map(func, args), redfunc);
   }

   //////////////////////////////////////////////////////////////////////////
   /// \copydoc TExecutorCRTP::MapReduce(F func,const std::vector<T> &args,R redfunc)
   template<class F, class T, class R, class Cond>
   auto TSequentialExecutor::MapReduce(F func, const std::vector<T> &args, R redfunc) -> typename std::result_of<F(T)>::type {
      return Reduce(Map(func, args), redfunc);
   }

} // namespace ROOT
#endif
