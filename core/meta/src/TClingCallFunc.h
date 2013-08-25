// @(#)root/core/meta:$Id$
// Author: Paul Russo   30/07/2012

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
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
#include "TClingClassInfo.h"

#include "llvm/ExecutionEngine/GenericValue.h"
#include "cling/Interpreter/StoredValueRef.h"

#include <vector>

namespace clang {
class Expr;
class FuncionDecl;
class CXXMethodDecl;
}
namespace cling {
class Interpreter;
class Value;
}

namespace llvm {
class Function;
}

class TInterpreterValue;

class TClingCallFunc {

private:

   cling::Interpreter                *fInterp;  // Cling interpreter, we do *not* own.
   TClingMethodInfo                  *fMethod;  // Current method, we own. If it is virtual this holds the trampoline.
   const clang::CXXMethodDecl        *fMethodAsWritten;
   llvm::Function                    *fEEFunc;  // Execution Engine function for current method, we do *not* own.
   void                              *fEEAddr;  // Pointer to actual compiled code, we do *not* own.
   mutable std::vector<cling::StoredValueRef> fArgVals; // Argument values parsed and evaluated from a text string.
                                                // We must keep this around because it is the owner of
                                                // any storage created by constructor calls in the evaluated
                                                // text string.  For example, if "2,TNamed(),3.0" is passed,
                                                // the second argument is a reference to a constructed
                                                // object of type TNamed (in the compiler this would be
                                                // a temporary, but cling makes it the return value of a
                                                // wrapper function call, so we must store it).  All of the
                                                // value parts stored here are copied immediately into fArgs,
                                                // but fArgs does not become the owner. 
   //   std::vector<llvm::GenericValue>    fArgs;    // Arguments to pass to the JIT when calling the function.
                                                // Some arguments may be set by value with one of the SetArg()
                                                // overloads, and some may be set by evaluating a text string
                                                // with SetArgs() or SetFunc() (in which case the value
                                                // stored here is a copy of the value stored in fArgVals).
                                                // We do *not* own the data stored here.
   bool                                fIgnoreExtraArgs;


private:
   llvm::GenericValue convertIntegralToArg(const cling::Value& val, 
                                           const llvm::Type* targetType) const;
   std::string ExprToString(const clang::Expr* expr) const;
   void EvaluateArgList(const std::string &ArgList);
   cling::StoredValueRef EvaluateExpr(const clang::Expr* E) const;
   bool IsTrampolineFunc() const { return fMethodAsWritten; }
   bool DoesThatTrampolineFuncReturn() const;
   bool DoesThatFuncReturnATemporary() const;
   bool IsMemberFunc() const;
   const clang::FunctionDecl* GetOriginalDecl() const;

   void PushArg(const cling::Value& value) const;
   void PushArg(cling::StoredValueRef value) const;
   void SetThisPtr(const clang::CXXMethodDecl* MD, void* address) const;
   void SetReturnPtr(const clang::FunctionDecl* FD, void* address) const;
   void SetReturnPtr(cling::StoredValueRef val) const {
      fArgVals[1] = val;
   }
   cling::StoredValueRef& GetReturnPtr() const { return fArgVals[1]; }
   size_t GetArgValsSize() const { 
      return fArgVals.size() - !fArgVals[0].isValid() - !fArgVals[1].isValid();
   }

   void PreallocatePtrs() const {
      // fArgVals[0] to be used for this ptr.
      fArgVals.push_back(cling::StoredValueRef::invalidValue());
      // fArgVals[1] to be used for return result ptr.
      fArgVals.push_back(cling::StoredValueRef::invalidValue());
   }

   void BuildTrampolineFunc(clang::CXXMethodDecl* MD);
   void CodeGenDecl(const clang::FunctionDecl* FD);

public:

   ~TClingCallFunc() { 
      delete fMethod;
   }

   explicit TClingCallFunc(cling::Interpreter *interp)
      : fInterp(interp), fMethodAsWritten(0), fEEFunc(0), fEEAddr(0), 
        fIgnoreExtraArgs(false)
   {
      fMethod = new TClingMethodInfo(interp);
      
      PreallocatePtrs();
   }

   TClingCallFunc(const TClingCallFunc &rhs)
      : fInterp(rhs.fInterp), fMethodAsWritten(0), fEEFunc(rhs.fEEFunc), 
        fEEAddr(rhs.fEEAddr), fArgVals(rhs.fArgVals),
        fIgnoreExtraArgs(rhs.fIgnoreExtraArgs)
   {
      fMethod = new TClingMethodInfo(*rhs.fMethod);
   }

   TClingCallFunc &operator=(const TClingCallFunc &rhs)
   {
      if (this != &rhs) {
         fInterp = rhs.fInterp;
         delete fMethod;
         fMethod = new TClingMethodInfo(*rhs.fMethod);
         fMethodAsWritten = rhs.fMethodAsWritten;
         fEEFunc = rhs.fEEFunc;
         fEEAddr = rhs.fEEAddr;
         fArgVals = rhs.fArgVals;
         fIgnoreExtraArgs = rhs.fIgnoreExtraArgs;
      }
      return *this;
   }

   void                Exec(void *address, TInterpreterValue* interpVal = 0) const;
   long                ExecInt(void *address) const;
   long long           ExecInt64(void *address) const;
   double              ExecDouble(void *address) const;
   TClingMethodInfo   *FactoryMethod() const;
   void                IgnoreExtraArgs(bool ignore) { fIgnoreExtraArgs = ignore; }
   void                Init();
   void               *InterfaceMethod() const;
   bool                IsValid() const;
   void                ResetArg();
   void                SetArg(long arg);
   void                SetArg(double arg);
   void                SetArg(long long arg);
   void                SetArg(unsigned long long arg);
   void                SetArgArray(long *argArr, int narg);
   void                SetArgs(const char *args);
   void                SetFunc(const TClingClassInfo *info, const char *method, const char *arglist, long *poffset);
   void                SetFunc(const TClingClassInfo *info, const char *method, const char *arglist, bool objectIsConst, long *poffset);
   void                SetFunc(const TClingMethodInfo *info);
   void                SetFuncProto(const TClingClassInfo *info, const char *method, const char *proto, long *poffset, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch);
   void                SetFuncProto(const TClingClassInfo *info, const char *method, const char *proto, bool objectIsConst, long *poffset, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch);
   void                SetFuncProto(const TClingClassInfo *info, const char *method, const llvm::SmallVector<clang::QualType, 4> &proto, long *poffset, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch);
   void                SetFuncProto(const TClingClassInfo *info, const char *method, const llvm::SmallVector<clang::QualType, 4> &proto, bool objectIsConst, long *poffset, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch);
   void                Init(const clang::FunctionDecl *);
   void                Invoke(cling::StoredValueRef* result = 0) const;
};

#endif // ROOT_CallFunc
