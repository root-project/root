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

   cling::Interpreter                *fInterp;  // Cling interpreter, we do *not* own.
   TClingMethodInfo                  *fMethod;  // Current method, we own.
   llvm::Function                    *fEEFunc;  // Execution Engine function for current method, we do *not* own.
   void                              *fEEAddr;  // Pointer to actual compiled code, we do *not* own.
   std::vector<cling::StoredValueRef> fArgVals; // Argument values parsed and evaluated from a text string.
                                                // We must keep this around because it is the owner of
                                                // any storage created by constructor calls in the evaluated
                                                // text string.  For example, if "2,TNamed(),3.0" is passed,
                                                // the second argument is a reference to a constructed
                                                // object of type TNamed (in the compiler this would be
                                                // a temporary, but cling makes it the return value of a
                                                // wrapper function call, so we must store it).  All of the
                                                // value parts stored here are copied immediately into fArgs,
                                                // but fArgs does not become the owner. 
   std::vector<llvm::GenericValue>    fArgs;    // Arguments to pass to the JIT when calling the function.
                                                // Some arguments may be set by value with one of the SetArg()
                                                // overloads, and some may be set by evaluating a text string
                                                // with SetArgs() or SetFunc() (in which case the value
                                                // stored here is a copy of the value stored in fArgVals).
                                                // We do *not* own the data stored here.


   std::string ExprToString(const clang::Expr* expr) const;
   void EvaluateArgList(const std::string &ArgList);
   cling::StoredValueRef EvaluateExpression(const clang::Expr* expr) const;

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
