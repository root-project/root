// @(#)root/thread:$Id$
// Author: Xavier Valls September 2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TExecutor
#define ROOT_TExecutor

#include "ROOT/RConfig.hxx"
#include "ROOT/RMakeUnique.hxx"
#include "ROOT/TExecutorCRTP.hxx"
#include "ROOT/TSeq.hxx"
#include "ROOT/TSequentialExecutor.hxx"
#ifdef R__USE_IMT
#include "ROOT/TThreadExecutor.hxx"
#endif
#ifndef R__WIN32
#include "ROOT/TProcessExecutor.hxx"
#endif
#include "TROOT.h"
#include "ExecutionPolicy.hxx"

#include <initializer_list>
#include <memory>
#include <thread>
#include <type_traits> //std::enable_if, std::result_of
#include <stdexcept> //std::invalid_argument
#include <utility> //std::move

//////////////////////////////////////////////////////////////////////////
///
/// \class ROOT::Internal::TExecutor
/// \brief This class defines an interface to execute the same task
/// multiple times, sequentially or in parallel depending on the execution policy passed
/// as a first parameter on construction, and possibly with different arguments every time.
/// The classes implementing it mimic the behaviour of python's pool.Map method.
///
/// ###ROOT::Internal::TExecutor::Map
/// The two possible usages of the Map method are:\n
/// * `Map(F func, unsigned nTimes)`: func is executed nTimes with no arguments
/// * `Map(F func, T& args)`: func is executed on each element of the collection of arguments args
///
/// For either signature, func is executed as many times as needed by a pool of
/// nThreads threads; It defaults to the number of cores.\n
/// A collection containing the result of each execution is returned.\n
/// **Note:** the user is responsible for the deletion of any object that might
/// be created upon execution of func, returned objects included. ROOT::::Internal::TExecutor never
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
///
/// ### ROOT::Internal::TExecutor::MapReduce
/// This set of methods behaves exactly like Map, but takes an additional
/// function as a third argument. This function is applied to the set of
/// objects returned by the corresponding Map execution to "squash" them
/// to a single object. This function should be independent of the size of
/// the vector returned by Map due to optimization of the number of chunks.
///
/// #### Examples:
/// ~~~{.cpp}
/// root[] ROOT::Internal::TExecutor pool; auto ten = pool.MapReduce([]() { return 1; }, 10, [](std::vector<int> v) { return std::accumulate(v.begin(), v.end(), 0); })
/// root[] ROOT::Internal::TExecutor pool(ROOT::Internal::ExecutionPolicy::kMultiProcess); auto hist = pool.MapReduce(CreateAndFillHists, 10, PoolUtils::ReduceObjects);
/// ~~~
///
//////////////////////////////////////////////////////////////////////////


namespace ROOT{

namespace Internal{
class TExecutor: public TExecutorCRTP<TExecutor> {
   friend TExecutorCRTP;
#ifdef R__USE_IMT
   friend TThreadExecutor;
#endif
public:

   /// \brief Class constructor. Sets the default execution policy and initializes the corresponding executor.
   /// Defaults to multithreaded execution policy if ROOT is compiled with IMT=ON and IsImplicitMTEnabled. Otherwise it defaults to a serial execution policy
   /// \param nWorkers [optional] Number of parallel workers, only taken into account if the execution policy is kMultithread
   explicit TExecutor(unsigned nWorkers = 0) :
      TExecutor(ROOT::IsImplicitMTEnabled() ? ROOT::Internal::ExecutionPolicy::kMultiThread : ROOT::Internal::ExecutionPolicy::kSequential, nWorkers) {}

   /// \brief Class constructor. Sets the execution policy and initializes the corresponding executor.
   /// \param execPolicy Execution policy(kMultithread, kMultiprocess, kSerial) to process the data
   /// \param nWorkers [optional] Number of parallel workers, only taken into account if the execution policy is kMultithread
   explicit TExecutor(ROOT::Internal::ExecutionPolicy execPolicy, unsigned nWorkers = 0) : fExecPolicy(execPolicy) {
      fExecPolicy = execPolicy;
      switch(fExecPolicy) {
         case ROOT::Internal::ExecutionPolicy::kSequential:
            fSequentialExecutor = std::make_unique<ROOT::TSequentialExecutor>();
            break;
#ifdef R__USE_IMT
         case ROOT::Internal::ExecutionPolicy::kMultiThread:
            fThreadExecutor = std::make_unique<ROOT::TThreadExecutor>(nWorkers);
            break;
#endif
#ifndef R__WIN32
         case ROOT::Internal::ExecutionPolicy::kMultiProcess:
            fProcessExecutor = std::make_unique<ROOT::TProcessExecutor>(nWorkers);
            break;
#endif
         default:
            throw std::invalid_argument(
               "Invalid execution policy. Potential issues: * kMultithread policy not available when ROOT is compiled with IMT=OFF.\n * kMultiprocess policy not available in Windows");
      }
   }

   TExecutor(const TExecutor &) = delete;
   TExecutor &operator=(const TExecutor &) = delete;

   /// Return the execution policy the executor is set to
   ROOT::Internal::ExecutionPolicy Policy(){ return fExecPolicy; }

   // Map
   //
   using TExecutorCRTP<TExecutor>::Map;

   // MapReduce
   // the late return types also check at compile-time whether redfunc is compatible with func,
   // other than checking that func is compatible with the type of arguments.
   // a static_assert check in TExecutor::Reduce is used to check that redfunc is compatible with the type returned by func
   using TExecutorCRTP<TExecutor>::MapReduce;
   template<class F, class R, class Cond = noReferenceCond<F>>
   auto MapReduce(F func, unsigned nTimes, R redfunc, unsigned nChunks) -> typename std::result_of<F()>::type;
   template<class F, class INTEGER, class R, class Cond = noReferenceCond<F, INTEGER>>
   auto MapReduce(F func, ROOT::TSeq<INTEGER> args, R redfunc, unsigned nChunks) -> typename std::result_of<F(INTEGER)>::type;
   template<class F, class T, class R, class Cond = noReferenceCond<F, T>>
   auto MapReduce(F func, std::initializer_list<T> args, R redfunc, unsigned nChunks) -> typename std::result_of<F(T)>::type;
   template<class F, class T, class R, class Cond = noReferenceCond<F, T>>
   auto MapReduce(F func, std::vector<T> &args, R redfunc, unsigned nChunks) -> typename std::result_of<F(T)>::type;
   template<class F, class T, class R, class Cond = noReferenceCond<F, T>>
   auto MapReduce(F func, const std::vector<T> &args, R redfunc, unsigned nChunks) -> typename std::result_of<F(T)>::type;

   // Reduce
   //
   using TExecutorCRTP<TExecutor>::Reduce;

unsigned GetPoolSize() const;

private:
   // Implementation of the Map functions declared in the parent class (TExecutorCRTP)
   //
   template<class F, class Cond = noReferenceCond<F>>
   auto MapImpl(F func, unsigned nTimes) -> std::vector<typename std::result_of<F()>::type>;
   template<class F, class INTEGER, class Cond = noReferenceCond<F, INTEGER>>
   auto MapImpl(F func, ROOT::TSeq<INTEGER> args) -> std::vector<typename std::result_of<F(INTEGER)>::type>;
   template<class F, class T, class Cond = noReferenceCond<F, T>>
   auto MapImpl(F func, std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type>;
   template<class F, class T, class Cond = noReferenceCond<F, T>>
   auto MapImpl(F func, const std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type>;

   // Extension of the Map interfaces with chunking, specific to this class and
   // only available from a MapReduce call.
   template<class F, class R, class Cond = noReferenceCond<F>>
   auto Map(F func, unsigned nTimes, R redfunc, unsigned nChunks) -> std::vector<typename std::result_of<F()>::type>;
   template<class F, class INTEGER, class R, class Cond = noReferenceCond<F, INTEGER>>
   auto Map(F func, ROOT::TSeq<INTEGER> args, R redfunc, unsigned nChunks) -> std::vector<typename std::result_of<F(INTEGER)>::type>;
   template<class F, class T, class R, class Cond = noReferenceCond<F, T>>
   auto Map(F func, std::vector<T> &args, R redfunc, unsigned nChunks) -> std::vector<typename std::result_of<F(T)>::type>;
   template<class F, class T, class R, class Cond = noReferenceCond<F, T>>
   auto Map(F func, const std::vector<T> &args, R redfunc, unsigned nChunks) -> std::vector<typename std::result_of<F(T)>::type>;
   template<class F, class T, class R, class Cond = noReferenceCond<F, T>>
   auto Map(F func, std::initializer_list<T> args, R redfunc, unsigned nChunks) -> std::vector<typename std::result_of<F(T)>::type>;

   ROOT::Internal::ExecutionPolicy fExecPolicy;
#ifdef R__USE_IMT
   std::unique_ptr<ROOT::TThreadExecutor> fThreadExecutor;
#else
   #define fThreadExecutor fSequentialExecutor
#endif
#ifndef R__WIN32
   std::unique_ptr<ROOT::TProcessExecutor> fProcessExecutor;
#else
   #define fProcessExecutor fSequentialExecutor
#endif
   std::unique_ptr<ROOT::TSequentialExecutor> fSequentialExecutor;

   /// \brief Helper class to get the correct return type from the Map function,
   /// necessary to infer the ResolveExecutorAndMap function type
   template<class F, class CONTAINER>
   struct MapRetType {
      using type = typename std::result_of<F(typename CONTAINER::value_type)>::type;
   };

   template<class F>
   struct MapRetType<F, unsigned> {
      using type = typename std::result_of<F()>::type;
   };


   /// \brief Function called from Map to select and execute the correct Executor
   /// according to the set Execution Policy.
   template<class F, class T>
   auto ResolveExecutorAndMap(F func, T args) -> std::vector<typename MapRetType<F, T>::type> {
      std::vector<typename MapRetType<F, T>::type> res;
      switch(fExecPolicy) {
         case ROOT::Internal::ExecutionPolicy::kSequential:
            res = fSequentialExecutor->Map(func, args);
            break;
         case ROOT::Internal::ExecutionPolicy::kMultiThread:
            res = fThreadExecutor->Map(func, args);
            break;
         case ROOT::Internal::ExecutionPolicy::kMultiProcess:
            res = fProcessExecutor->Map(func, args);
            break;
         default:
            break;
      }
      return res;
   }
};


//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function without arguments several times.
/// Implementation of the Map method.
///
/// \copydetails TExecutorCRTP::Map(F func,unsigned nTimes)
template<class F, class Cond>
auto TExecutor::MapImpl(F func, unsigned nTimes) -> std::vector<typename std::result_of<F()>::type> {
   return ResolveExecutorAndMap(func, nTimes);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function over a sequence of indexes.
/// Implementation of the Map method.
///
/// \copydetails TExecutorCRTP::Map(F func,ROOT::TSeq<INTEGER> args)
template<class F, class INTEGER, class Cond>
auto TExecutor::MapImpl(F func, ROOT::TSeq<INTEGER> args) -> std::vector<typename std::result_of<F(INTEGER)>::type> {
   return ResolveExecutorAndMap(func, args);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Execute func (with no arguments) nTimes, dividing the execution in nChunks and providing a result per chunk if
/// the execution policy is multithreaded. Otherwise, it ignores the two last arguments and performs a normal Map operation.
///
/// \param func Function to be executed.
/// \param nTimes Number of times function should be called.
/// \param redfunc Reduction function, used both to generate the partial results and the end result. Must return the same type as `func`.
/// \param nChunks Number of chunks to split the input data for processing.
/// \return A vector with the results of the function calls.
template<class F, class R, class Cond>
auto TExecutor::Map(F func, unsigned nTimes, R redfunc, unsigned nChunks) -> std::vector<typename std::result_of<F()>::type> {
   if (fExecPolicy == ROOT::Internal::ExecutionPolicy::kMultiThread) {
      return fThreadExecutor->Map(func, nTimes, redfunc, nChunks);
   }
   return Map(func, nTimes);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function over the elements of a vector.
/// Implementation of the Map method.
///
/// \copydetails TExecutorCRTP::Map(F func,std::vector<T> &args)
template<class F, class T, class Cond>
auto TExecutor::MapImpl(F func, std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type> {
   return ResolveExecutorAndMap(func, args);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function over the elements of an immutable vector.
/// Implementation of the Map method.
///
/// \copydetails TExecutorCRTP::Map(F func,const std::vector<T> &args)
template<class F, class T, class Cond>
auto TExecutor::MapImpl(F func, const std::vector<T> &args) -> std::vector<typename std::result_of<F(T)>::type> {
   return ResolveExecutorAndMap(func, args);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function over a sequence of indexes, dividing the execution in nChunks and providing a result per chunk if
/// the execution policy is multithreaded. Otherwise, it ignores the two last arguments and performs a normal Map operation.
///
/// \param func Function to be executed. Must take an element of the sequence passed assecond argument as a parameter.
/// \param args Sequence of indexes to execute `func` on.
/// \param redfunc Reduction function, used to combine the results of the calls to `func` into partial results. Must return the same type as `func`.
/// \param nChunks Number of chunks to split the input data for processing.
/// \return A vector with the results of the function calls.
template<class F, class INTEGER, class R, class Cond>
auto TExecutor::Map(F func, ROOT::TSeq<INTEGER> args, R redfunc, unsigned nChunks) -> std::vector<typename std::result_of<F(INTEGER)>::type> {
   if (fExecPolicy == ROOT::Internal::ExecutionPolicy::kMultiThread) {
      return fThreadExecutor->Map(func, args, redfunc, nChunks);
   }
   return Map(func, args);
}



//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function over the elements of a vector, dividing the execution in nChunks and providing a result per chunk if
/// the execution policy is multithreaded. Otherwise, it ignores the two last arguments and performs a normal Map operation.
///
/// \param func Function to be executed on the elements of the vector passed as second parameter.
/// \param args Vector of elements passed as an argument to `func`.
/// \param redfunc Reduction function, used to combine the results of the calls to `func` into partial results. Must return the same type as `func`.
/// \param nChunks Number of chunks to split the input data for processing.
/// \return A vector with the results of the function calls.
template<class F, class T, class R, class Cond>
auto TExecutor::Map(F func, std::vector<T> &args, R redfunc, unsigned nChunks) -> std::vector<typename std::result_of<F(T)>::type> {
   if (fExecPolicy == ROOT::Internal::ExecutionPolicy::kMultiThread) {
      return fThreadExecutor->Map(func, args, redfunc, nChunks);
   }
   return Map(func, args);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function over the elements of an immutable vector, dividing the execution in nChunks and providing a result per chunk if
/// the execution policy is multithreaded. Otherwise, it ignores the two last arguments and performs a normal Map operation.
///
/// \param func Function to be executed on the elements of the vector passed as second parameter.
/// \param args Immutable vector of elements passed as an argument to `func`.
/// \param redfunc Reduction function, used to combine the results of the calls to `func` into partial results. Must return the same type as `func`.
/// \param nChunks Number of chunks to split the input data for processing.
/// \return A vector with the results of the function calls.
template<class F, class T, class R, class Cond>
auto TExecutor::Map(F func, const std::vector<T> &args, R redfunc, unsigned nChunks) -> std::vector<typename std::result_of<F(T)>::type> {
   if (fExecPolicy == ROOT::Internal::ExecutionPolicy::kMultiThread) {
      return fThreadExecutor->Map(func, args, redfunc, nChunks);
   }
   return Map(func, args);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function over the elements of an initializer_list, dividing the execution in nChunks and providing a result per chunk if
/// the execution policy is multithreaded. Otherwise, it ignores the two last arguments and performs a normal Map operation.
///
/// \param func Function to be executed on the elements of the initializer_list passed as second parameter.
/// \param args initializer_list for a vector to apply `func` on.
/// \param redfunc Reduction function, used to combine the results of the calls to `func` into partial results. Must return the same type as `func`.
/// \param nChunks Number of chunks to split the input data for processing.
/// \return A vector with the results of the function calls.
template<class F, class T, class R, class Cond>
auto TExecutor::Map(F func, std::initializer_list<T> args, R redfunc, unsigned nChunks) -> std::vector<typename std::result_of<F(T)>::type> {
   std::vector<T> vargs(std::move(args));
   const auto &reslist = Map(func, vargs, redfunc, nChunks);
   return reslist;
}


//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function `nTimes` (Map) and accumulate the results into a single value (Reduce).
/// Benefits from partial reduction into `nChunks` intermediate results if the execution policy is multithreaded.
/// Otherwise, it ignores the two last arguments and performs a normal Map operation.
///
/// \param func Function to be executed. Must take an element of the sequence passed as second argument as a parameter.
/// \param nTimes Number of times function should be called.
/// \param redfunc Reduction function to combine the results of the calls to `func` into partial results, and these
/// into a final result. Must return the same type as `func`.
/// \param nChunks Number of chunks to split the input data for processing.
/// \return A value result of "reducing" the vector returned by the Map operation into a single object.
template<class F, class R, class Cond>
auto TExecutor::MapReduce(F func, unsigned nTimes, R redfunc, unsigned nChunks) -> typename std::result_of<F()>::type {
   return Reduce(Map(func, nTimes, redfunc, nChunks), redfunc);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function over a sequence of indexes (Map) and accumulate the results into a single value (Reduce).
/// Benefits from partial reduction into `nChunks` intermediate results if the execution policy is multithreaded.
/// Otherwise, it ignores the two last arguments and performs a normal Map operation.
///
/// \param func Function to be executed. Must take an element of the sequence passed assecond argument as a parameter.
/// \param args Sequence of indexes to execute `func` on.
/// \param redfunc Reduction function to combine the results of the calls to `func` into partial results, and these
/// into a final result. Must return the same type as `func`.
/// \param nChunks Number of chunks to split the input data for processing.
/// \return A value result of "reducing" the vector returned by the Map operation into a single object.
template<class F, class INTEGER, class R, class Cond>
auto TExecutor::MapReduce(F func, ROOT::TSeq<INTEGER> args, R redfunc, unsigned nChunks) -> typename std::result_of<F(INTEGER)>::type {
   return Reduce(Map(func, args, redfunc, nChunks), redfunc);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function over the elements of an initializer_list (Map) and accumulate the results into a single value (Reduce).
/// Benefits from partial reduction into `nChunks` intermediate results if the execution policy is multithreaded.
/// Otherwise, it ignores the two last arguments and performs a normal Map operation.
///
/// \param func Function to be executed. Must take an element of the sequence passed assecond argument as a parameter.
/// \param args initializer_list for a vector to apply `func` on.
/// \param redfunc Reduction function to combine the results of the calls to `func` into partial results, and these
/// into a final result. Must return the same type as `func`.
/// \param nChunks Number of chunks to split the input data for processing.
/// \return A value result of "reducing" the vector returned by the Map operation into a single object.
template<class F, class T, class R, class Cond>
auto TExecutor::MapReduce(F func, std::initializer_list<T> args, R redfunc, unsigned nChunks) -> typename std::result_of<F(T)>::type {
   return Reduce(Map(func, args, redfunc, nChunks), redfunc);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function over the elements of a vector (Map) and accumulate the results into a single value (Reduce).
/// Benefits from partial reduction into `nChunks` intermediate results if the execution policy is multithreaded.
/// Otherwise, it ignores the two last arguments and performs a normal Map operation.
///
/// \param func Function to be executed. Must take an element of the sequence passed assecond argument as a parameter.
/// \param args Vector of elements passed as an argument to `func`.
/// \param redfunc Reduction function to combine the results of the calls to `func` into partial results, and these
/// into a final result. Must return the same type as `func`.
/// \param nChunks Number of chunks to split the input data for processing.
/// \return A value result of "reducing" the vector returned by the Map operation into a single object.
template<class F, class T, class R, class Cond>
auto TExecutor::MapReduce(F func, std::vector<T> &args, R redfunc, unsigned nChunks) -> typename std::result_of<F(T)>::type {
   return Reduce(Map(func, args, redfunc, nChunks), redfunc);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Execute a function over the elements of an immutable vector (Map) and accumulate the results into a single value (Reduce).
/// Benefits from partial reduction into `nChunks` intermediate results if the execution policy is multithreaded.
/// Otherwise, it ignores the two last arguments and performs a normal Map operation.
///
/// \param func Function to be executed. Must take an element of the sequence passed assecond argument as a parameter.
/// \param args Immutable vector, whose elements are passed as an argument to `func`.
/// \param redfunc Reduction function to combine the results of the calls to `func` into partial results, and these
/// into a final result. Must return the same type as `func`.
/// \param nChunks Number of chunks to split the input data for processing.
/// \return A value result of "reducing" the vector returned by the Map operation into a single object.
template<class F, class T, class R, class Cond>
auto TExecutor::MapReduce(F func, const std::vector<T> &args, R redfunc, unsigned nChunks) -> typename std::result_of<F(T)>::type {
   return Reduce(Map(func, args, redfunc, nChunks), redfunc);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Return the number of pooled workers.
///
/// \return The number of workers in the pool in the executor used as a backend.

unsigned TExecutor::GetPoolSize() const
{
   unsigned poolSize{0u};
   switch(fExecPolicy){
      case ROOT::Internal::ExecutionPolicy::kSequential:
         poolSize = fSequentialExecutor->GetPoolSize();
         break;
      case ROOT::Internal::ExecutionPolicy::kMultiThread:
         poolSize = fThreadExecutor->GetPoolSize();
         break;
      case ROOT::Internal::ExecutionPolicy::kMultiProcess:
         poolSize = fProcessExecutor->GetPoolSize();
         break;
      default:
         break;
   }
   return poolSize;
}

} // namespace Internal
} // namespace ROOT

#endif
