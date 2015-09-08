/* @(#)root/multiproc:$Id$ */
// Author: Enrico Guiraud July 2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPoolServer
#define ROOT_TPoolServer

#include "TMPServer.h"
#include "EPoolCode.h"
#include "EMPCode.h"
#include "MPSendRecv.h"
#include <string>
#include <vector>

//////////////////////////////////////////////////////////////////////////
///
/// This class works together with TPool to allow the execution of
/// functions in server processes. The partial specialization <class F, void>
/// handles the case in which no arguments should be passed to the function.
///
/// The function to be executed and optionally a collection of arguments
/// are passed to TPoolServer's constructor, so that the only information
/// that must be exchanged between servers and client is whether to execute
/// the function and the index of the argument to pass to it.
///
//////////////////////////////////////////////////////////////////////////

template<class F, class T = void>
class TPoolServer : public TMPServer {
public:
   TPoolServer(F func, const std::vector<T> &args) : TMPServer(), fFunc(func), fArgs(std::move(args)) {}
   ~TPoolServer() {}
   void HandleInput(MPCodeBufPair& msg) ///< Execute instructions received from a TPool client
   {
      unsigned code = msg.first;
      TSocket *s = GetSocket();
      std::string reply = "S" + std::to_string(GetPid());
      if (code == EPoolCode::kExecFuncWithArg) {
         unsigned n;
         msg.second->ReadUInt(n);
         MPSend(s, EPoolCode::kFuncResult, fFunc(fArgs[n]));
      } else {
         reply += ": unknown code received: " + std::to_string(code);
         MPSend(s, EMPCode::kError, reply.data());
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
class TPoolServer<F, void> : public TMPServer {
public:
   explicit TPoolServer(F func) : TMPServer(), fFunc(func) {}
   ~TPoolServer() {}
   void HandleInput(MPCodeBufPair& msg)
   {
      unsigned code = msg.first;
      TSocket *s = GetSocket();
      std::string myId = "S" + std::to_string(GetPid());
      if (code == EPoolCode::kExecFunc) {
         //EXECUTE FUNCTION WITHOUT ARGUMENTS
         MPSend(s, EPoolCode::kFuncResult, fFunc());
      } else {
         MPSend(s, EMPCode::kError, (myId + ": unknown code received: " + std::to_string(code)).c_str());
      }
   }

private:
   F fFunc;
};
/// \endcond

#endif
