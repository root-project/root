// @(#)root/thread:$Id$
// Author: Xavier Valls March 2016

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TThreadExecutor
#define ROOT_TThreadExecutor

#include "RConfigure.h"

// exclude in case ROOT does not have IMT support
#ifndef R__USE_IMT
// No need to error out for dictionaries.
# if !defined(__ROOTCLING__) && !defined(G__DICTIONARY)
#  error "Cannot use ROOT::TThreadExecutor without defining R__USE_IMT."
# endif
#else

#include "ROOT/TExecutor.hxx"
#include "ROOT/TPoolManager.hxx"
#include "TError.h"
#include <functional>
#include <memory>
#include <numeric>

namespace ROOT {

   class TThreadExecutor: public TExecutor<TThreadExecutor> {
   public:

      explicit TThreadExecutor(UInt_t nThreads = 0u);

      TThreadExecutor(TThreadExecutor &) = delete;
      TThreadExecutor &operator=(TThreadExecutor &) = delete;

      template<class F>
      void Foreach(F func, unsigned nTimes, unsigned nChunks = 0);
      template<class F, class INTEGER>
      void Foreach(F func, ROOT::TSeq<INTEGER> args, unsigned nChunks = 0);
      /// \cond
      template<class F, class T>
      void Foreach(F func, std::initializer_list<T> args, unsigned nChunks = 0);
      /// \endcond
      template<class F, class T>
      void Foreach(F func, std::vector<T> &args, unsigned nChunks = 0);
      template<class F, class T>
      void Foreach(F func, const std::vector<T> &args, unsigned nChunks = 0);

      using TExecutor<TThreadExecutor>::Map;
      template<class F, class Cond = noReferenceCond<F>>
      auto Map(F func, unsigned nTimes) -> std::vector<typename std::result_of<F()>::type>;
      template<class F, class INTEGER, class Cond = noReferenceCond<F, INTEGER>>
      auto Map(F func, ROOT::TSeq<INTEGER> args) -> std::vector<typename std::result_of<F(INTEGER)>::type>;
      template<class F, class T, class Cond = noReferenceCond<F, T>>
      auto Map(F func, std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type>;

      // // MapReduce
      // // the late return types also check at compile-time whether redfunc is compatible with func,
      // // other than checking that func is compatible with the type of arguments.
      // // a static_assert check in TThreadExecutor::Reduce is used to check that redfunc is compatible with the type returned by func
      using TExecutor<TThreadExecutor>::MapReduce;
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

      using TExecutor<TThreadExecutor>::Reduce;
      template<class T, class BINARYOP> auto Reduce(const std::vector<T> &objs, BINARYOP redfunc) -> decltype(redfunc(objs.front(), objs.front()));
      template<class T, class R> auto Reduce(const std::vector<T> &objs, R redfunc) -> decltype(redfunc(objs));

      unsigned GetPoolSize();

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
      void   ParallelFor(unsigned start, unsigned end, unsigned step, const std::function<void(unsigned int i)> &f);
      double ParallelReduce(const std::vector<double> &objs, const std::function<double(double a, double b)> &redfunc);
      float  ParallelReduce(const std::vector<float> &objs, const std::function<float(float a, float b)> &redfunc);
      template<class T, class R>
      auto SeqReduce(const std::vector<T> &objs, R redfunc) -> decltype(redfunc(objs));

      std::shared_ptr<ROOT::Internal::TPoolManager> fSched = nullptr;
   };

   /************ TEMPLATE METHODS IMPLEMENTATION ******************/

   //////////////////////////////////////////////////////////////////////////
   /// Execute func (with no arguments) nTimes in parallel.
   /// Functions that take more than zero arguments can be executed (with
   /// fixed arguments) by wrapping them in a lambda or with std::bind.
   template<class F>
   void TThreadExecutor::Foreach(F func, unsigned nTimes, unsigned nChunks) {
      if (nChunks == 0) {
         ParallelFor(0U, nTimes, 1, [&](unsigned int){func();});
         return;
      }

      unsigned step = (nTimes + nChunks - 1) / nChunks;
      auto lambda = [&](unsigned int i)
      {
         for (unsigned j = 0; j < step && (i + j) < nTimes; j++) {
            func();
         }
      };
      ParallelFor(0U, nTimes, step, lambda);
   }

   //////////////////////////////////////////////////////////////////////////
   /// Execute func in parallel, taking an element of a
   /// sequence as argument.
   template<class F, class INTEGER>
   void TThreadExecutor::Foreach(F func, ROOT::TSeq<INTEGER> args, unsigned nChunks) {
      if (nChunks == 0) {
         ParallelFor(*args.begin(), *args.end(), args.step(), [&](unsigned int i){func(i);});
         return;
      }
      unsigned start = *args.begin();
      unsigned end = *args.end();
      unsigned seqStep = args.step();
      unsigned step = (end - start + nChunks - 1) / nChunks; //ceiling the division

      auto lambda = [&](unsigned int i)
      {
         for (unsigned j = 0; j < step && (i + j) < end; j+=seqStep) {
            func(i + j);
         }
      };
      ParallelFor(start, end, step, lambda);
   }

   /// \cond
   //////////////////////////////////////////////////////////////////////////
   /// Execute func in parallel, taking an element of a
   /// initializer_list as argument.
   template<class F, class T>
   void TThreadExecutor::Foreach(F func, std::initializer_list<T> args, unsigned nChunks) {
      std::vector<T> vargs(std::move(args));
      Foreach(func, vargs, nChunks);
   }
   /// \endcond

   //////////////////////////////////////////////////////////////////////////
   /// Execute func in parallel, taking an element of an
   /// std::vector as argument.
   template<class F, class T>
   void TThreadExecutor::Foreach(F func, std::vector<T> &args, unsigned nChunks) {
      unsigned int nToProcess = args.size();
      if (nChunks == 0) {
         ParallelFor(0U, nToProcess, 1, [&](unsigned int i){func(args[i]);});
         return;
      }

      unsigned step = (nToProcess + nChunks - 1) / nChunks; //ceiling the division
      auto lambda = [&](unsigned int i)
      {
         for (unsigned j = 0; j < step && (i + j) < nToProcess; j++) {
            func(args[i + j]);
         }
      };
      ParallelFor(0U, nToProcess, step, lambda);
   }

   //////////////////////////////////////////////////////////////////////////
   /// Execute func in parallel, taking an element of a std::vector as argument.
   template<class F, class T>
   void TThreadExecutor::Foreach(F func, const std::vector<T> &args, unsigned nChunks) {
      unsigned int nToProcess = args.size();
      if (nChunks == 0) {
         ParallelFor(0U, nToProcess, 1, [&](unsigned int i){func(args[i]);});
         return;
      }

      unsigned step = (nToProcess + nChunks - 1) / nChunks; //ceiling the division
      auto lambda = [&](unsigned int i)
      {
         for (unsigned j = 0; j < step && (i + j) < nToProcess; j++) {
            func(args[i + j]);
         }
      };
      ParallelFor(0U, nToProcess, step, lambda);
   }

   //////////////////////////////////////////////////////////////////////////
   /// Execute func (with no arguments) nTimes in parallel.
   /// A vector containg executions' results is returned.
   /// Functions that take more than zero arguments can be executed (with
   /// fixed arguments) by wrapping them in a lambda or with std::bind.
   template<class F, class Cond>
   auto TThreadExecutor::Map(F func, unsigned nTimes) -> std::vector<typename std::result_of<F()>::type> {
      using retType = decltype(func());
      std::vector<retType> reslist(nTimes);
      auto lambda = [&](unsigned int i)
      {
         reslist[i] = func();
      };
      ParallelFor(0U, nTimes, 1, lambda);

      return reslist;
   }

   //////////////////////////////////////////////////////////////////////////
   /// Execute func in parallel, taking an element of a
   /// sequence as argument.
   /// A vector containg executions' results is returned.
   template<class F, class INTEGER, class Cond>
   auto TThreadExecutor::Map(F func, ROOT::TSeq<INTEGER> args) -> std::vector<typename std::result_of<F(INTEGER)>::type> {
      unsigned start = *args.begin();
      unsigned end = *args.end();
      unsigned seqStep = args.step();

      using retType = decltype(func(start));
      std::vector<retType> reslist(args.size());
      auto lambda = [&](unsigned int i)
      {
         reslist[i] = func(i);
      };
      ParallelFor(start, end, seqStep, lambda);

      return reslist;
   }

   //////////////////////////////////////////////////////////////////////////
   /// Execute func (with no arguments) nTimes in parallel.
   /// Divides and groups the executions in nChunks (if it doesn't make sense will reduce the number of chunks) with partial reduction;
   /// A vector containg partial reductions' results is returned.
   template<class F, class R, class Cond>
   auto TThreadExecutor::Map(F func, unsigned nTimes, R redfunc, unsigned nChunks) -> std::vector<typename std::result_of<F()>::type> {
      if (nChunks == 0)
      {
         return Map(func, nTimes);
      }

      unsigned step = (nTimes + nChunks - 1) / nChunks;
      // Avoid empty chunks
      unsigned actualChunks = (nTimes + step - 1) / step;
      using retType = decltype(func());
      std::vector<retType> reslist(actualChunks);
      auto lambda = [&](unsigned int i)
      {
         std::vector<retType> partialResults(std::min(nTimes-i, step));
         for (unsigned j = 0; j < step && (i + j) < nTimes; j++) {
            partialResults[j] = func();
         }
         reslist[i / step] = Reduce(partialResults, redfunc);
      };
      ParallelFor(0U, nTimes, step, lambda);

      return reslist;
   }

   //////////////////////////////////////////////////////////////////////////
   /// Execute func in parallel, taking an element of an
   /// std::vector as argument.
   /// A vector containg executions' results is returned.
   // actual implementation of the Map method. all other calls with arguments eventually
   // call this one
   template<class F, class T, class Cond>
   auto TThreadExecutor::Map(F func, std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type> {
      // //check whether func is callable
      using retType = decltype(func(args.front()));

      unsigned int nToProcess = args.size();
      std::vector<retType> reslist(nToProcess);

      auto lambda = [&](unsigned int i)
      {
         reslist[i] = func(args[i]);
      };

      ParallelFor(0U, nToProcess, 1, lambda);

      return reslist;
   }

   //////////////////////////////////////////////////////////////////////////
   /// Execute func in parallel, taking an element of a
   /// sequence as argument.
   /// Divides and groups the executions in nChunks (if it doesn't make sense will reduce the number of chunks) with partial reduction\n
   /// A vector containg partial reductions' results is returned.
   template<class F, class INTEGER, class R, class Cond>
   auto TThreadExecutor::Map(F func, ROOT::TSeq<INTEGER> args, R redfunc, unsigned nChunks) -> std::vector<typename std::result_of<F(INTEGER)>::type> {
      if (nChunks == 0)
      {
         return Map(func, args);
      }

      unsigned start = *args.begin();
      unsigned end = *args.end();
      unsigned seqStep = args.step();
      unsigned step = (end - start + nChunks - 1) / nChunks; //ceiling the division
      // Avoid empty chunks
      unsigned actualChunks = (end - start + step - 1) / step;

      using retType = decltype(func(start));
      std::vector<retType> reslist(actualChunks);
      auto lambda = [&](unsigned int i)
      {
         std::vector<retType> partialResults(std::min(end-i, step));
         for (unsigned j = 0; j < step && (i + j) < end; j+=seqStep) {
            partialResults[j] = func(i + j);
         }
         reslist[i / step] = Reduce(partialResults, redfunc);
      };
      ParallelFor(start, end, step, lambda);

      return reslist;
   }

/// \cond
    //////////////////////////////////////////////////////////////////////////
   /// Execute func in parallel, taking an element of an
   /// std::vector as argument. Divides and groups the executions in nChunks with partial reduction.
   /// If it doesn't make sense will reduce the number of chunks.\n
   /// A vector containg partial reductions' results is returned.
   template<class F, class T, class R, class Cond>
   auto TThreadExecutor::Map(F func, std::vector<T> &args, R redfunc, unsigned nChunks) -> std::vector<typename std::result_of<F(T)>::type> {
      if (nChunks == 0)
      {
         return Map(func, args);
      }

      unsigned int nToProcess = args.size();
      unsigned step = (nToProcess + nChunks - 1) / nChunks; //ceiling the division
      // Avoid empty chunks
      unsigned actualChunks = (nToProcess + step - 1) / step;

      using retType = decltype(func(args.front()));
      std::vector<retType> reslist(actualChunks);
      auto lambda = [&](unsigned int i)
      {
         std::vector<T> partialResults(step);
         for (unsigned j = 0; j < step && (i + j) < nToProcess; j++) {
            partialResults[j] = func(args[i + j]);
         }
         reslist[i / step] = Reduce(partialResults, redfunc);
      };

      ParallelFor(0U, nToProcess, step, lambda);

      return reslist;
   }

    //////////////////////////////////////////////////////////////////////////
   /// Execute func in parallel, taking an element of an
   /// std::initializer_list as an argument. Divides and groups the executions in nChunks with partial reduction.
   /// If it doesn't make sense will reduce the number of chunks.\n
   /// A vector containg partial reductions' results is returned.
   template<class F, class T, class R, class Cond>
   auto TThreadExecutor::Map(F func, std::initializer_list<T> args, R redfunc, unsigned nChunks) -> std::vector<typename std::result_of<F(T)>::type> {
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
   auto TThreadExecutor::MapReduce(F func, unsigned nTimes, R redfunc) -> typename std::result_of<F()>::type {
      return Reduce(Map(func, nTimes), redfunc);
   }

   template<class F, class R, class Cond>
   auto TThreadExecutor::MapReduce(F func, unsigned nTimes, R redfunc, unsigned nChunks) -> typename std::result_of<F()>::type {
      return Reduce(Map(func, nTimes, redfunc, nChunks), redfunc);
   }

   template<class F, class INTEGER, class R, class Cond>
   auto TThreadExecutor::MapReduce(F func, ROOT::TSeq<INTEGER> args, R redfunc, unsigned nChunks) -> typename std::result_of<F(INTEGER)>::type {
      return Reduce(Map(func, args, redfunc, nChunks), redfunc);
   }
   /// \cond
   template<class F, class T, class R, class Cond>
   auto TThreadExecutor::MapReduce(F func, std::initializer_list<T> args, R redfunc, unsigned nChunks) -> typename std::result_of<F(T)>::type {
      return Reduce(Map(func, args, redfunc, nChunks), redfunc);
   }
   /// \endcond

   template<class F, class T, class R, class Cond>
   auto TThreadExecutor::MapReduce(F func, std::vector<T> &args, R redfunc) -> typename std::result_of<F(T)>::type {
      return Reduce(Map(func, args), redfunc);
   }

   template<class F, class T, class R, class Cond>
   auto TThreadExecutor::MapReduce(F func, std::vector<T> &args, R redfunc, unsigned nChunks) -> typename std::result_of<F(T)>::type {
      return Reduce(Map(func, args, redfunc, nChunks), redfunc);
   }

   //////////////////////////////////////////////////////////////////////////
   /// "Reduce" an std::vector into a single object in parallel by passing a
   /// binary operator as the second argument to act on pairs of elements of the std::vector.
   template<class T, class BINARYOP>
   auto TThreadExecutor::Reduce(const std::vector<T> &objs, BINARYOP redfunc) -> decltype(redfunc(objs.front(), objs.front()))
   {
      // check we can apply reduce to objs
      static_assert(std::is_same<decltype(redfunc(objs.front(), objs.front())), T>::value, "redfunc does not have the correct signature");
      return ParallelReduce(objs, redfunc);
   }

   //////////////////////////////////////////////////////////////////////////
   /// "Reduce" an std::vector into a single object by passing a
   /// function as the second argument defining the reduction operation.
   template<class T, class R>
   auto TThreadExecutor::Reduce(const std::vector<T> &objs, R redfunc) -> decltype(redfunc(objs))
   {
      // check we can apply reduce to objs
      static_assert(std::is_same<decltype(redfunc(objs)), T>::value, "redfunc does not have the correct signature");
      return SeqReduce(objs, redfunc);
   }

   template<class T, class R>
   auto TThreadExecutor::SeqReduce(const std::vector<T> &objs, R redfunc) -> decltype(redfunc(objs))
   {
      return redfunc(objs);
   }

} // namespace ROOT

#endif   // R__USE_IMT
#endif
