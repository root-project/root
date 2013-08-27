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
   mutable std::vector<cling::StoredValueRef> fArgVals; // Stored function arguments, we own.
   bool                                fIgnoreExtraArgs;


private:
   void EvaluateArgList(const std::string &ArgList);
   cling::StoredValueRef EvaluateExpr(const clang::Expr* E) const;

public:

   ~TClingCallFunc() { 
      delete fMethod;
   }

   explicit TClingCallFunc(cling::Interpreter *interp)
      : fInterp(interp), fIgnoreExtraArgs(false)
   {
      fMethod = new TClingMethodInfo(interp);
   }

   TClingCallFunc(const TClingCallFunc &rhs)
      : fInterp(rhs.fInterp), fArgVals(rhs.fArgVals),
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
};

#endif // ROOT_CallFunc
