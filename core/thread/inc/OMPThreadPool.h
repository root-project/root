// @(#)root/thread:$Id$
// Author: Xavier Valls March 2016

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_OMPThreadPool
#define ROOT_OMPThreadPool

#include "TPool.h"
#include <omp.h>

class OMPThreadPool: public TPool<OMPThreadPool> {
public:
   explicit OMPThreadPool() = default;

   explicit OMPThreadPool(size_t nThreads){
     omp_set_dynamic(0);     // Explicitly disable dynamic teams
     omp_set_num_threads(4);
   }

   ~OMPThreadPool() {
      omp_set_dynamic(1);
      omp_set_num_threads(omp_get_max_threads());
   }


   template<class F, class Cond = noReferenceCond<F>>
   auto Map(F func, unsigned nTimes) -> std::vector<typename std::result_of<F()>::type>;
   /// \cond
   template<class F, class INTEGER, class Cond = noReferenceCond<F, INTEGER>>
   auto Map(F func, ROOT::TSeq<INTEGER> args) -> std::vector<typename std::result_of<F(INTEGER)>::type>;
   template<class F, class T, class Cond = noReferenceCond<F, T>>
   auto Map(F func, std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type>;
   // / \endcond
   using TPool<OMPThreadPool>::Map;

   template<class T, class BINARYOP> auto Reduce(const std::vector<T> &objs, BINARYOP redfunc) -> decltype(redfunc(objs.front(), objs.front())) = delete;
   using TPool<OMPThreadPool>::Reduce;

private:

};

/************ TEMPLATE METHODS IMPLEMENTATION ******************/

//////////////////////////////////////////////////////////////////////////
/// Execute func (with no arguments) nTimes in parallel.
/// A vector containg executions' results is returned.
/// Functions that take more than zero arguments can be executed (with
/// fixed arguments) by wrapping them in a lambda or with std::bind.
template<class F, class Cond>
auto OMPThreadPool::Map(F func, unsigned nTimes) -> std::vector<typename std::result_of<F()>::type>
{
   using retType = decltype(func());
   std::vector<retType> reslist(nTimes);

   #pragma omp parallel for
   for(auto i=0U; i<nTimes; i++)
         reslist[i] = func();
   return reslist;
}

template<class F, class INTEGER, class Cond>
auto OMPThreadPool::Map(F func, ROOT::TSeq<INTEGER> args) -> std::vector<typename std::result_of<F(INTEGER)>::type>
{
   unsigned start = *args.begin();
   unsigned end = *args.end();
   using retType = decltype(func(start));
   std::vector<retType> reslist(end-start);

   #pragma omp parallel for
   for(auto i=start; i<end; i++)
         reslist[i] = func(i);
   return reslist;
}

// tell doxygen to ignore this (\endcond closes the statement)
/// \cond

// actual implementation of the Map method. all other calls with arguments eventually
// call this one
template<class F, class T, class Cond>
auto OMPThreadPool::Map(F func, std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type>
{
   // //check whether func is callable
   using retType = decltype(func(args.front()));

   unsigned int fNToProcess = args.size();
   std::vector<retType> reslist(fNToProcess);


   #pragma omp parallel for
   for(auto i=0U; i<fNToProcess; i++)
         reslist[i] = func(args[i]);
   return reslist;
}

// // tell doxygen to stop ignoring code
// /// \endcond

#endif
