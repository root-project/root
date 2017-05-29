/* @(#)root/multiproc:$Id$ */
// Author: Enrico Guiraud July 2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMPWorkerExecutor
#define ROOT_TMPWorkerExecutor

#include "MPCode.h"
#include "MPSendRecv.h"
#include "PoolUtils.h"
#include "TMPWorker.h"
#include <string>
#include <vector>

//////////////////////////////////////////////////////////////////////////
///
/// \class TMPWorkerExecutor
///
/// This class works together with TProcessExecutor to allow the execution of
/// functions in server processes. Depending on the exact task that the
/// worker is required to execute, a different version of the class
/// can be called.
///
/// ### TMPWorkerExecutor<F, T, R>
/// The most general case, used by
/// TProcessExecutor::MapReduce(F func, T& args, R redfunc).
/// This worker is build with:
/// * a function of signature F (the one to be executed)
/// * a collection of arguments of type T on which to apply the function
/// * a reduce function with signature R to be used to squash many
/// returned values together.
///
/// ### Partial specializations
/// A few partial specializations are provided for less general cases:
/// * TMPWorkerExecutor<F, T, void> handles the case of a function that takes
/// one argument and does not perform reduce operations
/// (TProcessExecutor::Map(F func, T& args)).
/// * TMPWorkerExecutor<F, void, R> handles the case of a function that takes
/// no arguments, to be executed a specified amount of times, which
/// returned values are squashed together (reduced)
/// (TProcessExecutor::Map(F func, unsigned nTimes, R redfunc))
/// * TMPWorkerExecutor<F, void, void> handles the case of a function that takes
/// no arguments and whose arguments are not "reduced"
/// (TProcessExecutor::Map(F func, unsigned nTimes))
///
/// Since all the important data are passed to TMPWorkerExecutor at construction
/// time, the kind of messages that client and workers have to exchange
/// are usually very simple.
///
//////////////////////////////////////////////////////////////////////////

// Quick guide to TMPWorkerExecutor:
// For each TProcessExecutor::Map and TProcessExecutor::MapReduce signature
// there's a corresponding
// specialization of TMPWorkerExecutor:
// * Map(func, nTimes) --> TMPWorkerExecutor<F, void, void>
// * Map(func, args)   --> TMPWorkerExecutor<F, T, void>
// * MapReduce(func, nTimes, redfunc) --> TMPWorkerExecutor<F, void, R>
// * MapReduce(func, args, redfunc)   --> TMPWorkerExecutor<F, T, R>
// I thought about having four different classes (with different names)
// instead of four specializations of the same class template, but it really
// makes no difference in the end since the different classes would be class
// templates anyway, and I would have to find a meaningful name for each one.
// About code replication: looking carefully, it can be noticed that there's
// very little code replication since the different versions of TMPWorkerExecutor
// all behave slightly differently, in incompatible ways (e.g. they all need
// different data members, different signatures for the ctors, and so on).

template<class F, class T = void, class R = void>
class TMPWorkerExecutor : public TMPWorker {
public:
   // TProcessExecutor is in charge of checking the signatures for incompatibilities:
   // we trust that decltype(redfunc(std::vector<decltype(func(args[0]))>)) == decltype(args[0])
   // TODO document somewhere that fReducedResult must have a default ctor
   TMPWorkerExecutor(F func, const std::vector<T> &args, R redfunc) :
      TMPWorker(), fFunc(func), fArgs(args), fRedFunc(redfunc),
      fReducedResult(), fCanReduce(false)
   {}
   ~TMPWorkerExecutor() {}

   void HandleInput(MPCodeBufPair &msg) ///< Execute instructions received from a TProcessExecutor client
   {
      unsigned code = msg.first;
      TSocket *s = GetSocket();
      std::string reply = "S" + std::to_string(GetNWorker());
      if (code == MPCode::kExecFuncWithArg) {
         unsigned n;
         msg.second->ReadUInt(n);
         // execute function on argument n
         const auto &res = fFunc(fArgs[n]);
         // tell client we're done
         MPSend(s, MPCode::kIdling);
         // reduce arguments if possible
         if (fCanReduce) {
            using FINAL = decltype(fReducedResult);
            using ORIGINAL = decltype(fRedFunc({res, fReducedResult}));
            fReducedResult = ROOT::Internal::PoolUtils::ResultCaster<ORIGINAL, FINAL>::CastIfNeeded(fRedFunc({res, fReducedResult})); //TODO try not to copy these into a vector, do everything by ref. std::vector<T&>?
         } else {
            fCanReduce = true;
            fReducedResult = res;
         }
      } else if (code == MPCode::kSendResult) {
         MPSend(s, MPCode::kFuncResult, fReducedResult);
      } else {
         reply += ": unknown code received: " + std::to_string(code);
         MPSend(s, MPCode::kError, reply.c_str());
      }
   }

private:
   F fFunc; ///< the function to be executed
   std::vector<T> fArgs; ///< a vector containing the arguments that must be passed to fFunc
   R fRedFunc; ///< the reduce function
   decltype(fFunc(fArgs.front())) fReducedResult; ///< the result of the execution
   bool fCanReduce; ///< true if fReducedResult can be reduced with a new result, false until we have produced one result
};


template<class F, class R>
class TMPWorkerExecutor<F, void, R> : public TMPWorker {
public:
   TMPWorkerExecutor(F func, R redfunc) :
      TMPWorker(), fFunc(func), fRedFunc(redfunc),
      fReducedResult(), fCanReduce(false)
   {}
   ~TMPWorkerExecutor() {}

   void HandleInput(MPCodeBufPair &msg) ///< Execute instructions received from a TProcessExecutor client
   {
      unsigned code = msg.first;
      TSocket *s = GetSocket();
      std::string reply = "S" + std::to_string(GetNWorker());
      if (code == MPCode::kExecFunc) {
         // execute function
         const auto &res = fFunc();
         // tell client we're done
         MPSend(s, MPCode::kIdling);
         // reduce arguments if possible
         if (fCanReduce) {
            fReducedResult = fRedFunc({res, fReducedResult}); //TODO try not to copy these into a vector, do everything by ref. std::vector<T&>?
         } else {
            fCanReduce = true;
            fReducedResult = res;
         }
      } else if (code == MPCode::kSendResult) {
         MPSend(s, MPCode::kFuncResult, fReducedResult);
      } else {
         reply += ": unknown code received: " + std::to_string(code);
         MPSend(s, MPCode::kError, reply.c_str());
      }
   }

private:
   F fFunc; ///< the function to be executed
   R fRedFunc; ///< the reduce function
   decltype(fFunc()) fReducedResult; ///< the result of the execution
   bool fCanReduce; ///< true if fReducedResult can be reduced with a new result, false until we have produced one result
};

template<class F, class T>
class TMPWorkerExecutor<F, T, void> : public TMPWorker {
public:
   TMPWorkerExecutor(F func, const std::vector<T> &args) : TMPWorker(), fFunc(func), fArgs(std::move(args)) {}
   ~TMPWorkerExecutor() {}
   void HandleInput(MPCodeBufPair &msg) ///< Execute instructions received from a TProcessExecutor client
   {
      unsigned code = msg.first;
      TSocket *s = GetSocket();
      std::string reply = "S" + std::to_string(GetNWorker());
      if (code == MPCode::kExecFuncWithArg) {
         unsigned n;
         msg.second->ReadUInt(n);
         MPSend(s, MPCode::kFuncResult, fFunc(fArgs[n]));
      } else {
         reply += ": unknown code received: " + std::to_string(code);
         MPSend(s, MPCode::kError, reply.c_str());
      }
   }

private:
   F fFunc; ///< the function to be executed
   std::vector<T> fArgs; ///< a vector containing the arguments that must be passed to fFunc
};


// doxygen should ignore this specialization
/// \cond
// The most generic class template is meant to handle functions that
// must be executed by passing one argument to them.
// This partial specialization is used to handle the case
// of functions which must be executed without passing any argument.
template<class F>
class TMPWorkerExecutor<F, void, void> : public TMPWorker {
public:
   explicit TMPWorkerExecutor(F func) : TMPWorker(), fFunc(func) {}
   ~TMPWorkerExecutor() {}
   void HandleInput(MPCodeBufPair &msg)
   {
      unsigned code = msg.first;
      TSocket *s = GetSocket();
      std::string myId = "S" + std::to_string(GetPid());
      if (code == MPCode::kExecFunc) {
         MPSend(s, MPCode::kFuncResult, fFunc());
      } else {
         MPSend(s, MPCode::kError, (myId + ": unknown code received: " + std::to_string(code)).c_str());
      }
   }

private:
   F fFunc;
};
/// \endcond

#endif
