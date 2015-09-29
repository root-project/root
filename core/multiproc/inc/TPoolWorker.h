/* @(#)root/multiproc:$Id$ */
// Author: Enrico Guiraud July 2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPoolWorker
#define ROOT_TPoolWorker

#include "TMPWorker.h"
#include "PoolCode.h"
#include "MPCode.h"
#include "MPSendRecv.h"
#include <string>
#include <vector>

// Quick guide to TPoolWorker:
// For each TProcPool::Map and TProcPool::MapReduce signature
// there's a corresponding
// specialization of TPoolWorker:
// * Map(func, nTimes) --> TPoolWorker<F, void, void>
// * Map(func, args)   --> TPoolWorker<F, T, void>
// * MapReduce(func, nTimes, redfunc) --> TPoolWorker<F, void, R>
// * MapReduce(func, args, redfunc)   --> TPoolWorker<F, T, R>
// I thought about having four different classes (with different names)
// instead of four specializations of the same class template, but it really
// makes no difference in the end since the different classes would be class
// templates anyway, and I would have to find a meaningful name for each one.
// About code replication: looking carefully, it can be noticed that there's
// very little code replication since the different versions of TPoolWorker
// all behave slightly differently, in incompatible ways (e.g. they all need
// different data members, different signatures for the ctors, and so on).

template<class F, class T = void, class R = void>
class TPoolWorker : public TMPWorker {
public:
   // TProcPool is in charge of checking the signatures for incompatibilities:
   // we trust that decltype(redfunc(std::vector<decltype(func(args[0]))>)) == decltype(args[0])
   // TODO document somewhere that fReducedResult must have a default ctor
   TPoolWorker(F func, const std::vector<T> &args, R redfunc) :
      TMPWorker(), fFunc(func), fArgs(std::move(args)), fRedFunc(redfunc),
      fReducedResult(), fCanReduce(false)
   {}
   ~TPoolWorker() {}

   void HandleInput(MPCodeBufPair &msg) ///< Execute instructions received from a TProcPool client
   {
      unsigned code = msg.first;
      TSocket *s = GetSocket();
      std::string reply = "S" + std::to_string(GetPid());
      if (code == PoolCode::kExecFuncWithArg) {
         unsigned n;
         msg.second->ReadUInt(n);
         // execute function on argument n
         const auto &res = fFunc(fArgs[n]);
         // tell client we're done
         MPSend(s, PoolCode::kIdling);
         // reduce arguments if possible
         if (fCanReduce) {
            fReducedResult = fRedFunc({res, fReducedResult}); //TODO try not to copy these into a vector, do everything by ref. std::vector<T&>?
         } else {
            fCanReduce = true;
            fReducedResult = res;
         }
      } else if (code == PoolCode::kSendResult) {
         MPSend(s, PoolCode::kFuncResult, fReducedResult);
      } else {
         reply += ": unknown code received: " + std::to_string(code);
         MPSend(s, MPCode::kError, reply.data());
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
class TPoolWorker<F, void, R> : public TMPWorker {
public:
   TPoolWorker(F func, R redfunc) :
      TMPWorker(), fFunc(func), fRedFunc(redfunc),
      fReducedResult(), fCanReduce(false)
   {}
   ~TPoolWorker() {}

   void HandleInput(MPCodeBufPair &msg) ///< Execute instructions received from a TProcPool client
   {
      unsigned code = msg.first;
      TSocket *s = GetSocket();
      std::string reply = "S" + std::to_string(GetPid());
      if (code == PoolCode::kExecFunc) {
         // execute function
         const auto &res = fFunc();
         // tell client we're done
         MPSend(s, PoolCode::kIdling);
         // reduce arguments if possible
         if (fCanReduce) {
            fReducedResult = fRedFunc({res, fReducedResult}); //TODO try not to copy these into a vector, do everything by ref. std::vector<T&>?
         } else {
            fCanReduce = true;
            fReducedResult = res;
         }
      } else if (code == PoolCode::kSendResult) {
         MPSend(s, PoolCode::kFuncResult, fReducedResult);
      } else {
         reply += ": unknown code received: " + std::to_string(code);
         MPSend(s, MPCode::kError, reply.data());
      }
   }

private:
   F fFunc; ///< the function to be executed
   R fRedFunc; ///< the reduce function
   decltype(fFunc()) fReducedResult; ///< the result of the execution
   bool fCanReduce; ///< true if fReducedResult can be reduced with a new result, false until we have produced one result
};

template<class F, class T>
class TPoolWorker<F, T, void> : public TMPWorker {
public:
   TPoolWorker(F func, const std::vector<T> &args) : TMPWorker(), fFunc(func), fArgs(std::move(args)) {}
   ~TPoolWorker() {}
   void HandleInput(MPCodeBufPair &msg) ///< Execute instructions received from a TProcPool client
   {
      unsigned code = msg.first;
      TSocket *s = GetSocket();
      std::string reply = "S" + std::to_string(GetPid());
      if (code == PoolCode::kExecFuncWithArg) {
         unsigned n;
         msg.second->ReadUInt(n);
         MPSend(s, PoolCode::kFuncResult, fFunc(fArgs[n]));
      } else {
         reply += ": unknown code received: " + std::to_string(code);
         MPSend(s, MPCode::kError, reply.data());
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
class TPoolWorker<F, void, void> : public TMPWorker {
public:
   explicit TPoolWorker(F func) : TMPWorker(), fFunc(func) {}
   ~TPoolWorker() {}
   void HandleInput(MPCodeBufPair &msg)
   {
      unsigned code = msg.first;
      TSocket *s = GetSocket();
      std::string myId = "S" + std::to_string(GetPid());
      if (code == PoolCode::kExecFunc) {
         MPSend(s, PoolCode::kFuncResult, fFunc());
      } else {
         MPSend(s, MPCode::kError, (myId + ": unknown code received: " + std::to_string(code)).c_str());
      }
   }

private:
   F fFunc;
};
/// \endcond

#endif
