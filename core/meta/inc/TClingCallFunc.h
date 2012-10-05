// @(#)root/core/meta:$Id$
// Author: Paul Russo   30/07/2012

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_CallFunc
#define ROOT_CallFunc

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClingCallFunc                                                       //
//                                                                      //
// Emulation of the CINT CallFunc class.                                //
//                                                                      //
// The CINT C++ interpreter provides an interface for calling           //
// functions through the generated wrappers in dictionaries with        //
// the CallFunc class. This class provides the same functionality,      //
// using an interface as close as possible to CallFunc but the          //
// function metadata and calling service comes from the Cling           //
// C++ interpreter and the Clang C++ compiler, not CINT.                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClingMethodInfo.h"

#include "llvm/ExecutionEngine/GenericValue.h"
#include "cling/Interpreter/StoredValueRef.h"

namespace cling {
class Interpreter;
}

namespace llvm {
class Function;
}

#include <vector>

class TClingClassInfo;

class TClingCallFunc {

private:

   cling::Interpreter               *fInterp; // Cling interpreter, we do *not* own.
   TClingMethodInfo                 *fMethod; // Current method, we own.
   llvm::Function                   *fEEFunc; // Execution Engine function for current method, we do *not* own.
   void                             *fEEAddr; // Pointer to actual compiled code, we do *not* own.
   std::vector<llvm::GenericValue>   fArgs; // Arguments to pass to function.
   std::vector<cling::StoredValueRef> fArgVals; // Arguments' storage.

public:

   ~TClingCallFunc() { delete fMethod; }

   explicit TClingCallFunc(cling::Interpreter *interp)
      : fInterp(interp), fMethod(0), fEEFunc(0), fEEAddr(0)
   {
      fMethod = new TClingMethodInfo(interp);
   }

   TClingCallFunc(const TClingCallFunc &rhs)
      : fInterp(rhs.fInterp), fMethod(0), fEEFunc(rhs.fEEFunc),
        fEEAddr(rhs.fEEAddr), fArgs(rhs.fArgs)
   {
      fMethod = new TClingMethodInfo(*rhs.fMethod);
   }

   TClingCallFunc &operator=(const TClingCallFunc &rhs)
   {
      if (this != &rhs) {
         fInterp = rhs.fInterp;
         delete fMethod;
         fMethod = new TClingMethodInfo(*rhs.fMethod);
         fEEFunc = rhs.fEEFunc;
         fEEAddr = rhs.fEEAddr;
         fArgs = rhs.fArgs;
      }
      return *this;
   }

   void                Exec(void *address) const;
   long                ExecInt(void *address) const;
   long long           ExecInt64(void *address) const;
   double              ExecDouble(void *address) const;
   TClingMethodInfo   *FactoryMethod() const;
   void                Init();
   void               *InterfaceMethod() const;
   bool                IsValid() const;
   void                ResetArg();
   void                SetArg(long param);
   void                SetArg(double param);
   void                SetArg(long long param);
   void                SetArg(unsigned long long param);
   void                SetArgArray(long *paramArr, int nparam);
   void                SetArgs(const char *params);
   void                SetFunc(const TClingClassInfo *info, const char *method, const char *params, long *offset);
   void                SetFunc(const TClingMethodInfo *info);
   void                SetFuncProto(const TClingClassInfo *info, const char *method, const char *proto, long *offset);
   void                Init(const clang::FunctionDecl *);
   llvm::GenericValue  Invoke(const std::vector<llvm::GenericValue> &ArgValues) const;

};

#endif // ROOT_CallFunc
