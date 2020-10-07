// @(#)root/core/base:$Id$
// Author: Xavier Valls March 2016

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TExecutorCRTP
#define ROOT_TExecutorCRTP

#include "ROOT/TSeq.hxx"
#include "TList.h"
#include <vector>

//////////////////////////////////////////////////////////////////////////
///
/// \class ROOT::TExecutorCRTP
/// \brief This class defines an interface to execute the same task
/// multiple times, possibly in parallel and with different arguments every
/// time. The classes implementing it mimic the behaviour of python's pool.Map method.
///
/// ###ROOT::TExecutorCRTP<subc>::Map
/// The two possible usages of the Map method are:\n
/// * `Map(F func, unsigned nTimes)`: func is executed nTimes with no arguments
/// * `Map(F func, T& args)`: func is executed on each element of the collection of arguments args
///
/// For either signature, func is executed as many times as needed by a pool of
/// n workers; It defaults to the number of cores.\n
/// A collection containing the result of each execution is returned.\n
/// **Note:** the user is responsible for the deletion of any object that might
/// be created upon execution of func, returned objects included. ROOT::TExecutorCRTP derived classes
///  never delete what they return, they simply forget it.\n
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
///
/// ### ROOT::TExecutorCRTP<subc>::MapReduce
/// This set of methods behaves exactly like Map, but takes an additional
/// function as a third argument. This function is applied to the set of
/// objects returned by the corresponding Map execution to "squash" them
/// to a single object. This function should be independent of the size of
/// the vector returned by Map due to optimization of the number of chunks.
///
/// #### Examples:
/// ~~~{.cpp}
/// root[] ROOT::TProcessExecutor pool; auto ten = pool.MapReduce([]() { return 1; }, 10, [](std::vector<int> v) { return std::accumulate(v.begin(), v.end(), 0); })
/// root[] ROOT::TThreadExecutor pool; auto hist = pool.MapReduce(CreateAndFillHists, 10, PoolUtils::ReduceObjects);
/// ~~~
///
//////////////////////////////////////////////////////////////////////////


namespace ROOT {

template<class subc>
class TExecutorCRTP {
public:
   explicit TExecutorCRTP() = default;
   explicit TExecutorCRTP(size_t /* nWorkers */ ){};

   /// type definition in used in templated functions for not allowing mapping functions that return references.
   /// The resulting vector elements must be assignable, references aren't.
   template< class F, class... T>
   using noReferenceCond = typename std::enable_if<"Function can't return a reference" && !(std::is_reference<typename std::result_of<F(T...)>::type>::value)>::type;

   // Map
   // These trailing return types allow for a compile time check of compatibility between function signatures and args,
   // and a compile time check that the argument list implements a front() method (all STL sequence containers have it)
   template<class F, class Cond = noReferenceCond<F>>
   auto Map(F func, unsigned nTimes) -> std::vector<typename std::result_of<F()>::type>;
   template<class F, class INTEGER, class Cond = noReferenceCond<F, INTEGER>>
   auto Map(F func, ROOT::TSeq<INTEGER> args) -> std::vector<typename std::result_of<F(INTEGER)>::type>;
   template<class F, class T, class Cond = noReferenceCond<F, T>>
   auto Map(F func, std::initializer_list<T> args) -> std::vector<typename std::result_of<F(T)>::type>;
   template<class F, class T, class Cond = noReferenceCond<F, T>>
   auto Map(F func, std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type>;
   template<class F, class T, class Cond = noReferenceCond<F, T>>
   auto Map(F func, const std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type>;

   // MapReduce
   // The trailing return types also check at compile time whether redfunc is compatible with func,
   // other than checking that func is compatible with the type of arguments.
   // a static_assert check in TExecutorCRTP<subc>::Reduce is used to check that redfunc is compatible with the type returned by func
   template<class F, class INTEGER, class R, class Cond = noReferenceCond<F, INTEGER>>
   auto MapReduce(F func, ROOT::TSeq<INTEGER> args, R redfunc) -> typename std::result_of<F(INTEGER)>::type;
   template<class F, class T, class R, class Cond = noReferenceCond<F, T>>
   auto MapReduce(F func, std::initializer_list<T> args, R redfunc) -> typename std::result_of<F(T)>::type;
   template<class F, class T, class R, class Cond = noReferenceCond<F, T>>
   auto MapReduce(F func, const std::vector<T> &args, R redfunc) -> typename std::result_of<F(T)>::type;
   template<class F, class T, class R, class Cond = noReferenceCond<F, T>>
   auto MapReduce(F func, std::vector<T> &args, R redfunc) -> typename std::result_of<F(T)>::type;
   template<class F, class T,class Cond = noReferenceCond<F, T>>
   T* MapReduce(F func, std::vector<T*> &args);
   template<class F, class T,class Cond = noReferenceCond<F, T>>
   T* MapReduce(F func, const std::vector<T*> &args);

   template<class T> T* Reduce(const std::vector<T*> &mergeObjs);
   template<class T, class R> auto Reduce(const std::vector<T> &objs, R redfunc) -> decltype(redfunc(objs));

   unsigned GetPoolSize();

private:
   inline subc & Derived()
   {
     return *static_cast<subc*>(this);
   }
};

//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function without arguments several times.
///
/// \param func Function to be executed.
/// \param nTimes Number of times function should be called.
/// \return A vector with the results of the function calls.
/// Functions that take arguments can be executed (with
/// fixed arguments) by wrapping them in a lambda or with std::bind.
template<class subc> template<class F, class Cond>
auto TExecutorCRTP<subc>::Map(F func, unsigned nTimes) -> std::vector<typename std::result_of<F()>::type>
{
   return Derived().Map(func, nTimes);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function over a sequence of indexes.
///
/// \param func Function to be executed. Must take an element of the sequence passed assecond argument as a parameter.
/// \param args Sequence of indexes to execute `func` on.
/// \return A vector with the results of the function calls.
template<class subc> template<class F, class INTEGER, class Cond>
auto TExecutorCRTP<subc>::Map(F func, ROOT::TSeq<INTEGER> args) -> std::vector<typename std::result_of<F(INTEGER)>::type>
{
   return Derived().Map(func, args);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function over the elements of an initializer_list.
///
/// \param func Function to be executed on the elements of the initializer_list passed as second parameter.
/// \param args initializer_list for a vector to apply `func` on.
/// \return A vector with the results of the function calls.
template<class subc> template<class F, class T, class Cond>
auto TExecutorCRTP<subc>::Map(F func, std::initializer_list<T> args) -> std::vector<typename std::result_of<F(T)>::type>
{
   std::vector<T> vargs(std::move(args));
   const auto &reslist = Map(func, vargs);
   return reslist;
}

//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function over the elements of a vector
///
/// \param func Function to be executed on the elements of the vector passed as second parameter.
/// \param args Vector of elements passed as an argument to `func`.
/// \return A vector with the results of the function calls.
template<class subc> template<class F, class T, class Cond>
auto TExecutorCRTP<subc>::Map(F func, std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type>
{
   return Derived().Map(func, args);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function over the elements of an immutable vector
///
/// \param func Function to be executed on the elements of the vector passed as second parameter.
/// \param args Vector of elements passed as an argument to `func`.
/// \return A vector with the results of the function calls.
template<class subc> template<class F, class T, class Cond>
auto TExecutorCRTP<subc>::Map(F func, const std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type>
{
   return Derived().Map(func, args);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function over a sequence of indexes (Map) and accumulate the results into a single value (Reduce).
///
/// \param func Function to be executed. Must take an element of the sequence passed assecond argument as a parameter.
/// \param args Sequence of indexes to execute `func` on.
/// \param redfunc Reduction function to combine the results of the calls to `func`. Must return the same type as `func`.
/// \return A value result of "reducing" the vector returned by the Map operation into a single object.
template<class subc> template<class F, class INTEGER, class R, class Cond>
auto TExecutorCRTP<subc>::MapReduce(F func, ROOT::TSeq<INTEGER> args, R redfunc) -> typename std::result_of<F(INTEGER)>::type
{
   std::vector<INTEGER> vargs(args.size());
   std::copy(args.begin(), args.end(), vargs.begin());
   return Derived().MapReduce(func, vargs, redfunc);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function over the elements of an initializer_list (Map) and accumulate the results into a single value (Reduce).
///
/// \param func Function to be executed on the elements of the initializer_list passed as second parameter.
/// \param args initializer_list for a vector to apply `func` on.
/// \param redfunc Reduction function to combine the results of the calls to `func`. Must return the same type as `func`.
/// \return A value result of "reducing" the vector returned by the Map operation into a single object.
template<class subc> template<class F, class T, class R, class Cond>
auto TExecutorCRTP<subc>::MapReduce(F func, std::initializer_list<T> args, R redfunc) -> typename std::result_of<F(T)>::type
{
   std::vector<T> vargs(std::move(args));
   return Derived().MapReduce(func, vargs, redfunc);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function over the elements of a vector (Map) and accumulate the results into a single value (Reduce).
///
/// \param func Function to be executed on the elements of the vector passed as second parameter.
/// \param args Vector of elements passed as an argument to `func`.
/// \param redfunc Reduction function to combine the results of the calls to `func`. Must return the same type as `func`.
/// \return A value result of "reducing" the vector returned by the Map operation into a single object.
template<class subc> template<class F, class T, class R, class Cond>
auto TExecutorCRTP<subc>::MapReduce(F func, std::vector<T> &args, R redfunc) -> typename std::result_of<F(T)>::type
{
   return Derived().MapReduce(func, args, redfunc);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function over the elements of an immutable vector (Map) and accumulate the results into a single value (Reduce).
///
/// \param func Function to be executed on the elements of the vector passed as second parameter.
/// \param args Immutable vector of elements passed as an argument to `func`.
/// \param redfunc Reduction function to combine the results of the calls to `func`. Must return the same type as `func`.
/// \return A value result of "reducing" the vector returned by the Map operation into a single object.
template<class subc> template<class F, class T, class R, class Cond>
auto TExecutorCRTP<subc>::MapReduce(F func, const std::vector<T> &args, R redfunc) -> typename std::result_of<F(T)>::type
{
   return Derived().MapReduce(func, args, redfunc);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function over the TObject-inheriting elements of a vector (Map) and merge the objects into a single one (Reduce).
///
/// \param func Function to be executed on the elements of the vector passed as second parameter.
/// \param args Vector of elements passed as an argument to `func`.
/// \param redfunc Reduction function to combine the results of the calls to `func`. Must return the same type as `func`.
/// \return A value result of "reducing" the vector returned by the Map operation into a single object.
template<class subc> template<class F, class T, class Cond>
T* TExecutorCRTP<subc>::MapReduce(F func, std::vector<T*> &args)
{
   return Derived().Reduce(Map(func, args));
}

//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function over the TObject-inheriting elements of an immutable vector (Map) and merge the objects into a single one (Reduce).
///
/// \param func Function to be executed on the elements of the vector passed as second parameter.
/// \param args Immutableector of elements passed as an argument to `func`.
/// \param redfunc Reduction function to combine the results of the calls to `func`. Must return the same type as `func`.
/// \return A value result of "reducing" the vector returned by the Map operation into a single object.
template<class subc> template<class F, class T, class Cond>
T* TExecutorCRTP<subc>::MapReduce(F func, const std::vector<T*> &args)
{
   return Derived().Reduce(Map(func, args));
}

//////////////////////////////////////////////////////////////////////////
/// \brief "Reduce" an std::vector into a single object by using the object's Merge method.
///
/// \param mergeObjs A vector of ROOT objects implementing the Merge method
/// \return An object result of merging the vector elements into one.
template<class subc> template<class T>
T* TExecutorCRTP<subc>::Reduce(const std::vector<T*> &mergeObjs)
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

//////////////////////////////////////////////////////////////////////////
/// \brief "Reduce" an std::vector into a single object by passing a
/// function as the second argument defining the reduction operation.
///
/// \param objs A vector of elements to combine.
/// \param redfunc Reduction function to combine the elements of the vector `objs`
/// \return A value result of combining the vector elements into a single object of the same type.
template<class subc> template<class T, class R>
auto TExecutorCRTP<subc>::Reduce(const std::vector<T> &objs, R redfunc) -> decltype(redfunc(objs))
{
   // check we can apply reduce to objs
   static_assert(std::is_same<decltype(redfunc(objs)), T>::value, "redfunc does not have the correct signature");
   return redfunc(objs);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Return the number of pooled workers.
///
/// \return The number of workers in the pool.
template<class subc>
unsigned TExecutorCRTP<subc>::GetPoolSize()
{
   return Derived().GetPoolSize();
}

} // end namespace ROOT
#endif
