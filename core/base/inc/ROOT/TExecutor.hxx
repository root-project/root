// @(#)root/thread:$Id$
// Author: Xavier Valls March 2016

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TExecutor
#define ROOT_TExecutor

#include "ROOT/TSeq.hxx"
#include "TList.h"
#include <vector>

//////////////////////////////////////////////////////////////////////////
///
/// \class ROOT::TExecutor
/// \brief This class defines an interface to execute the same task
/// multiple times in parallel, possibly with different arguments every
/// time. The classes implementing it mimic the behaviour of python's pool.Map method. 
///
/// ###ROOT::TExecutor::Map
/// The two possible usages of the Map method are:\n
/// * Map(F func, unsigned nTimes): func is executed nTimes with no arguments
/// * Map(F func, T& args): func is executed on each element of the collection of arguments args
///
/// For either signature, func is executed as many times as needed by a pool of
/// nThreads threads; It defaults to the number of cores.\n
/// A collection containing the result of each execution is returned.\n
/// **Note:** the user is responsible for the deletion of any object that might
/// be created upon execution of func, returned objects included: ROOT::TExecutor never
/// deletes what it returns, it simply forgets it.\n
///
/// \param func
/// \parblock
/// a lambda expression, an std::function, a loaded macro, a
/// functor class or a function that takes zero arguments (for the first signature)
/// or one (for the second signature).
/// \endparblock
/// \param args
/// \parblock
/// a standard vector, a ROOT::TSeq of integer type or an initializer list for the second signature.
/// An integer only for the first.\n
/// \endparblock
///
/// **Note:** in cases where the function to be executed takes more than
/// zero/one argument but all are fixed except zero/one, the function can be wrapped
/// in a lambda or via std::bind to give it the right signature.\n
///
/// #### Return value:
/// An std::vector. The elements in the container
/// will be the objects returned by func.

namespace ROOT {

template<class subc>
class TExecutor {
public:
   explicit TExecutor() = default;
   explicit TExecutor(size_t /* nThreads */ ){};

   template< class F, class... T>
   using noReferenceCond = typename std::enable_if<"Function can't return a reference" && !(std::is_reference<typename std::result_of<F(T...)>::type>::value)>::type;

   // // Map
   // //these late return types allow for a compile-time check of compatibility between function signatures and args,
   // //and a compile-time check that the argument list implements a front() method (all STL sequence containers have it)
   template<class F, class Cond = noReferenceCond<F>>
   auto Map(F func, unsigned nTimes) -> std::vector<typename std::result_of<F()>::type>;
   template<class F, class INTEGER, class Cond = noReferenceCond<F, INTEGER>>
   auto Map(F func, ROOT::TSeq<INTEGER> args) -> std::vector<typename std::result_of<F(INTEGER)>::type>;
   /// \cond
   template<class F, class T, class Cond = noReferenceCond<F, T>>
   auto Map(F func, std::initializer_list<T> args) -> std::vector<typename std::result_of<F(T)>::type>;
   /// \endcond
   template<class F, class T, class Cond = noReferenceCond<F, T>>
   auto Map(F func, std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type>;

   // // MapReduce
   // // the late return types also check at compile-time whether redfunc is compatible with func,
   // // other than checking that func is compatible with the type of arguments.
   // // a static_assert check in TExecutor<subc>::Reduce is used to check that redfunc is compatible with the type returned by func
   template<class F, class INTEGER, class R, class Cond = noReferenceCond<F, INTEGER>>
   auto MapReduce(F func, ROOT::TSeq<INTEGER> args, R redfunc) -> typename std::result_of<F(INTEGER)>::type;
   /// \cond
   template<class F, class T, class R, class Cond = noReferenceCond<F, T>>
   auto MapReduce(F func, std::initializer_list<T> args, R redfunc) -> typename std::result_of<F(T)>::type;
   /// \endcond
   template<class F, class T, class Cond = noReferenceCond<F, T>>
   T* MapReduce(F func, std::vector<T*> &args);

   template<class T> T* Reduce(const std::vector<T*> &mergeObjs);

private:
  inline subc & Derived()
  {
    return *static_cast<subc*>(this);
  }
};

//////////////////////////////////////////////////////////////////////////
/// Execute func (with no arguments) nTimes in parallel.
/// A vector containg executions' results is returned.
/// Functions that take more than zero arguments can be executed (with
/// fixed arguments) by wrapping them in a lambda or with std::bind.
template<class subc> template<class F, class Cond>
auto TExecutor<subc>::Map(F func, unsigned nTimes) -> std::vector<typename std::result_of<F()>::type>
{
   return Derived().Map(func, nTimes);
}

//////////////////////////////////////////////////////////////////////////
/// Execute func in parallel, taking an element of a
/// sequence as argument. Divides and groups the executions in nChunks with partial reduction;
/// A vector containg partial reductions' results is returned.
template<class subc> template<class F, class INTEGER, class Cond>
auto TExecutor<subc>::Map(F func, ROOT::TSeq<INTEGER> args) -> std::vector<typename std::result_of<F(INTEGER)>::type>
{
  return Derived().Map(func, args);
}

//////////////////////////////////////////////////////////////////////////
/// Execute func in parallel, taking an element of the std::initializer_list
/// as argument. Divides and groups the executions in nChunks with partial reduction;
/// A vector containg partial reductions' results is returned.
template<class subc> template<class F, class T, class Cond>
auto TExecutor<subc>::Map(F func, std::initializer_list<T> args) -> std::vector<typename std::result_of<F(T)>::type>
{
   std::vector<T> vargs(std::move(args));
   return Map(func, vargs);
}

//////////////////////////////////////////////////////////////////////////
/// Execute func in parallel, taking an element of an
/// std::vector as argument.
/// A vector containg executions' results is returned.
// actual implementation of the Map method. all other calls with arguments eventually
// call this one
template<class subc> template<class F, class T, class Cond>
auto TExecutor<subc>::Map(F func, std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type>
{
   return Derived().Map(func, args);
}

//////////////////////////////////////////////////////////////////////////
/// This method behaves just like Map, but an additional redfunc function
/// must be provided. redfunc is applied to the vector Map would return and
/// must return the same type as func. In practice, redfunc can be used to
/// "squash" the vector returned by Map into a single object by merging,
/// adding, mixing the elements of the vector.
template<class subc> template<class F, class INTEGER, class R, class Cond>
auto TExecutor<subc>::MapReduce(F func, ROOT::TSeq<INTEGER> args, R redfunc) -> typename std::result_of<F(INTEGER)>::type
{
  std::vector<INTEGER> vargs(args.size());
  std::copy(args.begin(), args.end(), vargs.begin());
  return Derived().MapReduce(func, vargs, redfunc);
}

template<class subc> template<class F, class T, class R, class Cond>
auto TExecutor<subc>::MapReduce(F func, std::initializer_list<T> args, R redfunc) -> typename std::result_of<F(T)>::type
{
   std::vector<T> vargs(std::move(args));
   return Derived().MapReduce(func, vargs, redfunc);
}

template<class subc> template<class F, class T, class Cond>
T* TExecutor<subc>::MapReduce(F func, std::vector<T*> &args)
{
   return Derived().Reduce(Map(func, args));
}

//////////////////////////////////////////////////////////////////////////
/// "Reduce" an std::vector into a single object by using the object's Merge
//Reduction for objects with the Merge() method
template<class subc> template<class T>
T* TExecutor<subc>::Reduce(const std::vector<T*> &mergeObjs)
{
   TList l;
  for(unsigned i =1; i<mergeObjs.size(); i++){
    l.Add(mergeObjs[i]);
  }
  // use clone to return a new object 
  auto retHist = dynamic_cast<T*>((mergeObjs.front())->Clone());
  if (retHist) retHist->Merge(&l);
  return retHist;
}

} // end namespace ROOT

#endif
