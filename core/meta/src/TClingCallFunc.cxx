// @(#)root/core/meta:$Id$
// Author: Paul Russo   30/07/2012
// Author: Vassil Vassilev   9/02/2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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

#include "TClingCallFunc.h"

#include "TClingClassInfo.h"
#include "TClingMethodInfo.h"
#include "TInterpreterValue.h"

#include "TError.h"
#include "TCling.h"

#include "cling/Interpreter/CompilationOptions.h"
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/LookupHelper.h"
#include "cling/Interpreter/StoredValueRef.h"
#include "cling/Interpreter/Transaction.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Lookup.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"

#include "clang/Sema/SemaInternal.h"

#include <string>
#include <vector>

using namespace ROOT;

// This ought to be declared by the implementer .. oh well...
extern void unresolvedSymbol();

cling::StoredValueRef TClingCallFunc::EvaluateExpr(const clang::Expr* E) const
{
   // Evaluate an Expr* and return its cling::StoredValueRef
   cling::StoredValueRef valref;

   using namespace clang;
   ASTContext& C = fInterp->getSema().getASTContext();
   llvm::APSInt res;
   if (E->EvaluateAsInt(res, C, /*AllowSideEffects*/Expr::SE_NoSideEffects)) {
      llvm::GenericValue gv;
      gv.IntVal = res;
      cling::Value val(gv, C.IntTy, fInterp->getLLVMType(C.IntTy));
      return cling::StoredValueRef::bitwiseCopy(C, val);
   }

   // TODO: Build a wrapper around the expression to avoid decompilation and 
   // compilation and other string operations.
   cling::Interpreter::CompilationResult cr 
      = fInterp->evaluate(ExprToString(E), valref);
   if (cr == cling::Interpreter::kSuccess)
      return valref;
   return cling::StoredValueRef::invalidValue();
}

void TClingCallFunc::Exec(void *address, TInterpreterValue* interpVal /* =0 */) const
{
   if (!IsValid()) {
      Error("TClingCallFunc::Exec", "Attempt to execute while invalid.");
      return;
   }

   // In case the function we are calling returns a temporary the user needs to
   // provide proper storage for it.
   if (fEEFunc->hasStructRetAttr() && !interpVal) {
      Error("TClingCallFunc::Invoke", 
            "Trying to invoke a function that returns a temporary without providing a storage.");
      return;
   }
   else if (interpVal && !fEEFunc->hasStructRetAttr()) {
      Error("TClingCallFunc::Invoke", 
            "Trying to invoke a function that doesn't return a temporary but storage is provided.");
      return;
   }


   const clang::FunctionDecl* FD = GetOriginalDecl();
   if (!IsMemberFunc()) {
      // Free function or static member function.
      // For the trampoline we need to insert a fake this ptr, because it
      // has to be our first argument for the trampoline.

      // If a value is given pass it as an argument to invoke
      if (interpVal) {
         Invoke();
         reinterpret_cast<cling::StoredValueRef&>(interpVal->Get()) = GetReturnPtr();
      }
      else
         Invoke();
      return;
   }

   // Member function.
   const clang::CXXMethodDecl* MD = llvm::cast<const clang::CXXMethodDecl>(FD);
   if (llvm::isa<clang::CXXConstructorDecl>(MD)) {
      // Constructor.
      Error("TClingCallFunc::Exec", "Constructor must be called with ExecInt!");
      return;
   }
   if (!address) {
      Error("TClingCallFunc::Exec",
            "calling member function with no object pointer!");
      return;
   }
   SetThisPtr(MD, address);
   // If a value is given pass it as an argument to invoke
   if (interpVal) {
      assert(!GetReturnPtr().isValid()
             && "We already have set the return result!?");
      Invoke();
      reinterpret_cast<cling::StoredValueRef&>(interpVal->Get()) = GetReturnPtr();
   }
   else
      Invoke();
}

Long_t TClingCallFunc::ExecInt(void *address) const
{
   // Yes, the function name has Int in it, but it
   // returns a long.  This is a matter of CINT history.
   if (!IsValid()) {
      Error("TClingCallFunc::ExecInt", "Attempt to execute while invalid.");
      return 0L;
   }
   const clang::FunctionDecl* FD = GetOriginalDecl();
   TCling* clingInterp = static_cast<TCling*>(gCling);
   if (!IsMemberFunc()) {
      // Free function or static member function.
      cling::StoredValueRef val;
      Invoke(&val);
      // In case this is a temporary we need to extend its lifetime by 
      // registering it to the list of temporaries.
      if (DoesThatFuncReturnATemporary()) {
         std::string s = "The function " + FD->getNameAsString() +
            "returns a temporary. Consider using Exec with TInterpreterValue!";
         Info("TClingCallFunc::ExecInt", "%s", s.c_str());
         clingInterp->RegisterTemporary(val);
      }

      return val.get().simplisticCastAs<long>();
   }

   if (const clang::CXXConstructorDecl *CD =
       llvm::dyn_cast<clang::CXXConstructorDecl>(FD)) {
      //
      // We are simulating evaluating the expression:
      //
      //      new MyClass(args...)
      //
      // and we return the allocated address.
      //
      clang::ASTContext &Ctx = CD->getASTContext();
      const clang::RecordDecl *RD  = CD->getParent();
      if (!RD->getDefinition()) {
         // Forward-declared class, we do not know what the size is.
         return 0L;
      }
      //
      //  If we are not doing a placement new, then
      //  find and call an operator new to allocate
      //  the memory for the object.
      //
      if (!address) {
         // TODO: Use ASTContext::getTypeSize() for the size of a type
        const clang::ASTRecordLayout &Layout = Ctx.getASTRecordLayout(RD);
        int64_t size = Layout.getSize().getQuantity();
        const cling::LookupHelper& LH = fInterp->getLookupHelper();
        // TODO: Use Sema::FindAllocationFunctions instead of the lookup.
        const clang::FunctionDecl *newFunc = 0;
        if ((newFunc = LH.findFunctionProto(RD, "operator new",
              "std::size_t"))) {
           // We have a member function operator new.
           // Note: An operator new that is a class member does not take
           //       a this pointer as the first argument, unlike normal
           //       member functions.
        }
        else if ((newFunc = LH.findFunctionProto(Ctx.getTranslationUnitDecl(),
              "operator new", "std::size_t"))) {
           // We have a global operator new.
        }
        else {
           Error("TClingCallFunc::ExecInt",
                 "in constructor call and could not find an operator new");
           return 0L;
        }
        TClingCallFunc cf(fInterp);
        cf.fMethod = new TClingMethodInfo(fInterp, newFunc);
        cf.Init(newFunc);
        cf.SetArg(static_cast<long>(size));
        // Note: This may throw!
        cling::StoredValueRef val;
        cf.Invoke(&val);
        address =
           reinterpret_cast<void*>(val.get().simplisticCastAs<unsigned long>());
        // Note: address is guaranteed to be non-zero here, otherwise
        //       the operator new would have thrown a bad_alloc exception.
      }
      //
      //  Call the constructor, either passing the address we were given,
      //  or the address we got from the operator new as the this pointer.
      //
      SetThisPtr(CD, address);
      //fArgVals[0] = cling::StoredValueRef::bitwiseCopy(C, thisPtr);

      Invoke();
      // And return the address of the object.
      return reinterpret_cast<long>(address);
   }

   // Member function
   const clang::CXXMethodDecl* MD = llvm::cast<const clang::CXXMethodDecl>(FD);

   // FIXME: Need to treat member operator new special, it takes no this ptr.
   if (!address) {
      Error("TClingCallFunc::ExecInt",
            "Calling member function with no object pointer!");
      return 0L;
   }
   SetThisPtr(MD, address);

   // Intentionally not initialized, so valgrind can report function calls
   // that don't set this.
   long returnStorage;

   if (IsTrampolineFunc() && DoesThatTrampolineFuncReturn()) {
      // Provide return storage
      SetReturnPtr(MD, &returnStorage);
   }

   cling::StoredValueRef val;
   Invoke(&val);
   // In case this is a temporary we need to extend its lifetime by 
   // registering it to the list of temporaries.
   if (DoesThatFuncReturnATemporary()) {
      std::string s = "The function " + FD->getNameAsString() +
         "returns a temporary. Consider using Exec with TInterpreterValue!";
      Info("TClingCallFunc::ExecInt", "%s", s.c_str());
      clingInterp->RegisterTemporary(val);
   }

   if (IsTrampolineFunc() && DoesThatTrampolineFuncReturn())
      return returnStorage;

   // FIXME: Why don't we use cling::Value::isVoid interface?
   if (val.get().getClangType()->isVoidType()) {
      // CINT was silently return 0 in this case,
      // for now emulate this behavior for backward compatibility ...
      return 0;
   }
   return val.get().simplisticCastAs<long>();
}

long long TClingCallFunc::ExecInt64(void *address) const
{
   if (!IsValid()) {
      Error("TClingCallFunc::ExecInt64", "Attempt to execute while invalid.");
      return 0LL;
   }
   const clang::FunctionDecl* FD = GetOriginalDecl();
   if (!IsMemberFunc()) {
      // Free function or static member function.
      cling::StoredValueRef val;
      Invoke(&val);
      return val.get().simplisticCastAs<long long>();
   }

   // Member function.
   const clang::CXXMethodDecl* MD = llvm::cast<const clang::CXXMethodDecl>(FD);
   if (llvm::dyn_cast<clang::CXXConstructorDecl>(MD)) {
      // Constructor.
      Error("TClingCallFunc::Exec", "Constructor must be called with ExecInt!");
      return 0LL;
   }
   if (!address) {
      Error("TClingCallFunc::Exec",
            "Calling member function with no object pointer!");
      return 0LL;
   }

   SetThisPtr(MD, address);

   // Intentionally not initialized, so valgrind can report function calls
   // that don't set this.
   long long returnStorage;
   if (IsTrampolineFunc() && DoesThatTrampolineFuncReturn()) {
      // Provide return storage
      SetReturnPtr(MD, &returnStorage);
   }

   cling::StoredValueRef val;
   Invoke(&val);
   if (IsTrampolineFunc() && DoesThatTrampolineFuncReturn()) {
      // Provide return storage
      return returnStorage;
   }
   return val.get().simplisticCastAs<long long>();
}

double TClingCallFunc::ExecDouble(void *address) const
{
   if (!IsValid()) {
      Error("TClingCallFunc::ExecDouble", "Attempt to execute while invalid.");
      return 0.0;
   }
   const clang::FunctionDecl *FD = GetOriginalDecl();
   if (!IsMemberFunc()) {
      // Free function or static member function.
      cling::StoredValueRef val;
      Invoke(&val);
      return val.get().simplisticCastAs<double>();
   }
   // Member function.
   const clang::CXXMethodDecl* MD = llvm::cast<const clang::CXXMethodDecl>(FD);
   if (llvm::isa<clang::CXXConstructorDecl>(FD)) {
      // Constructor.
      Error("TClingCallFunc::Exec", "Constructor must be called with ExecInt!");
      return 0.0;
   }
   if (!address) {
      Error("TClingCallFunc::Exec",
            "Calling member function with no object pointer!");
      return 0.0;
   }

   SetThisPtr(MD, address);

   // Intentionally not initialized, so valgrind can report function calls
   // that don't set this.
   double returnStorage;
   // For the trampoline we need to insert a fake this ptr, because it
   // has to be our first argument for the trampoline.
   if (IsTrampolineFunc() && DoesThatTrampolineFuncReturn()) {
      SetReturnPtr(fMethodAsWritten, &returnStorage);
   }

   cling::StoredValueRef val;
   Invoke(&val);
   if (IsTrampolineFunc() && DoesThatTrampolineFuncReturn()) {
      return returnStorage;
   }
   return val.get().simplisticCastAs<double>();
}

TClingMethodInfo *TClingCallFunc::FactoryMethod() const
{
   return new TClingMethodInfo(*fMethod);
}

void TClingCallFunc::Init()
{
   delete fMethod;
   fMethod = 0;
   fMethodAsWritten = 0;
   fEEFunc = 0;
   fEEAddr = 0;
   ResetArg();
}

void *TClingCallFunc::InterfaceMethod() const
{
   if (!IsValid()) {
      return 0;
   }
   return const_cast<void*>(reinterpret_cast<const void*>(GetOriginalDecl()));
}

bool TClingCallFunc::IsValid() const
{
   return fEEAddr;
}

void TClingCallFunc::ResetArg()
{
   fArgVals.clear();
   PreallocatePtrs();
}

void TClingCallFunc::SetArg(long param)
{
   clang::ASTContext& C = fInterp->getSema().getASTContext();
   llvm::GenericValue gv;
   clang::QualType QT = C.LongTy;
   gv.IntVal = llvm::APInt(C.getTypeSize(QT), param);
   PushArg(cling::Value(gv, QT, fInterp->getLLVMType(QT)));
}

void TClingCallFunc::SetArg(double param)
{
   clang::ASTContext& C = fInterp->getSema().getASTContext();
   llvm::GenericValue gv;
   clang::QualType QT = C.DoubleTy;
   gv.DoubleVal = param;
   PushArg(cling::Value(gv, QT, fInterp->getLLVMType(QT)));
}

void TClingCallFunc::SetArg(long long param)
{
   clang::ASTContext& C = fInterp->getSema().getASTContext();
   llvm::GenericValue gv;
   clang::QualType QT = C.LongLongTy;
   gv.IntVal = llvm::APInt(C.getTypeSize(QT), param);
   PushArg(cling::Value(gv, QT, fInterp->getLLVMType(QT)));
}

void TClingCallFunc::SetArg(unsigned long long param)
{
   clang::ASTContext& C = fInterp->getSema().getASTContext();
   llvm::GenericValue gv;
   clang::QualType QT = C.UnsignedLongLongTy;
   gv.IntVal = llvm::APInt(C.getTypeSize(QT), param);
   PushArg(cling::Value(gv, QT, fInterp->getLLVMType(QT)));
}

void TClingCallFunc::SetArgArray(long *paramArr, int nparam)
{
   ResetArg();
   for (int i = 0; i < nparam; ++i) {
      SetArg(paramArr[i]);
   }
}

void TClingCallFunc::EvaluateArgList(const std::string &ArgList)
{
   ResetArg();
   llvm::SmallVector<clang::Expr*, 4> exprs;
   fInterp->getLookupHelper().findArgList(ArgList, exprs);
   for (llvm::SmallVector<clang::Expr*, 4>::const_iterator I = exprs.begin(),
         E = exprs.end(); I != E; ++I) {
      cling::StoredValueRef val = EvaluateExpr(*I);
      if (!val.isValid()) {
         // Bad expression, all done.
         break;
      }
      PushArg(val);
   }
}

void TClingCallFunc::SetArgs(const char *params)
{
   ResetArg();
   EvaluateArgList(params);
}

void TClingCallFunc::SetFunc(const TClingClassInfo* info, const char* method,
                             const char* arglist, long* poffset)
{
   SetFunc(info,method,arglist,false,poffset);
}

void TClingCallFunc::SetFunc(const TClingClassInfo* info,
                             const char* method,
                             const char* arglist,  bool objectIsConst, 
                             long* poffset)
{
   delete fMethod;
   fMethod = new TClingMethodInfo(fInterp);
   fEEFunc = 0;
   fEEAddr = 0;
   if (poffset) {
      *poffset = 0L;
   }
   ResetArg();
   if (!info->IsValid()) {
      Error("TClingCallFunc::SetFunc", "Class info is invalid!");
      return;
   }
   if (!strcmp(arglist, ")")) {
      // CINT accepted a single right paren as meaning no arguments.
      arglist = "";
   }
   *fMethod = info->GetMethodWithArgs(method, arglist, objectIsConst, poffset);
   if (!fMethod->IsValid()) {
      //Error("TClingCallFunc::SetFunc", "Could not find method %s(%s)", method,
      //      arglist);
      return;
   }
   const clang::FunctionDecl *decl = fMethod->GetMethodDecl();
   Init(decl);
   if (!IsValid()) {
      //Error("TClingCallFunc::SetFunc", "Method %s(%s) has no body.", method,
      //      arglist);
   }
   // FIXME: The arglist was already parsed by the lookup, we should
   //        enhance the lookup to return the resulting expression
   //        list so we do not need to parse it again here.
   EvaluateArgList(arglist);
}

void TClingCallFunc::SetFunc(const TClingMethodInfo *info)
{
   delete fMethod;
   fMethod = 0;
   fEEFunc = 0;
   fEEAddr = 0;
   ResetArg();
   fMethod = new TClingMethodInfo(*info);
   if (!fMethod->IsValid()) {
      return;
   }
   Init(fMethod->GetMethodDecl());
   //if (!IsValid()) {
   //   Error("TClingCallFunc::SetFunc", "Method has no body.");
   //}
}

void TClingCallFunc::SetFuncProto(const TClingClassInfo *info,
                                  const char *method, const char *proto,
                                  long *poffset,
                                  EFunctionMatchMode mode /* = kConversionMatch */
                                  )
{
   SetFuncProto(info,method,proto,false,poffset, mode);
}

void TClingCallFunc::SetFuncProto(const TClingClassInfo *info,
                                  const char *method, const char *proto,
                                  bool objectIsConst,
                                  long *poffset,
                                  EFunctionMatchMode mode /* =kConversionMatch */
                                  )
{
   delete fMethod;
   fMethod = new TClingMethodInfo(fInterp);
   fEEFunc = 0;
   fEEAddr = 0;
   if (poffset) {
      *poffset = 0L;
   }
   ResetArg();
   if (!info->IsValid()) {
      Error("TClingCallFunc::SetFuncProto", "Class info is invalid!");
      return;
   }
   *fMethod = info->GetMethod(method, proto, objectIsConst, poffset, mode);
   if (!fMethod->IsValid()) {
      //Error("TClingCallFunc::SetFuncProto", "Could not find method %s(%s)",
      //      method, proto);
      return;
   }
   const clang::FunctionDecl *decl = fMethod->GetMethodDecl();
   Init(decl);
   if (!IsValid()) {
      //Error("TClingCallFunc::SetFuncProto", "Method %s(%s) has no body.",
      //      method, proto);
   }
}

void TClingCallFunc::SetFuncProto(const TClingClassInfo *info,
                                  const char *method,
                                  const llvm::SmallVector<clang::QualType, 4> &proto,
                                  long *poffset,
                                  EFunctionMatchMode mode /* = kConversionMatch */
                                  )
{
   SetFuncProto(info,method,proto,false,poffset, mode);
}

void TClingCallFunc::SetFuncProto(const TClingClassInfo *info,
                                  const char *method,
                                  const llvm::SmallVector<clang::QualType, 4> &proto,
                                  bool objectIsConst,
                                  long *poffset,
                                  EFunctionMatchMode mode /* =kConversionMatch */
                                  )
{
   delete fMethod;
   fMethod = new TClingMethodInfo(fInterp);
   fEEFunc = 0;
   fEEAddr = 0;
   if (poffset) {
      *poffset = 0L;
   }
   ResetArg();
   if (!info->IsValid()) {
      Error("TClingCallFunc::SetFuncProto", "Class info is invalid!");
      return;
   }
   *fMethod = info->GetMethod(method, proto, objectIsConst, poffset, mode);
   if (!fMethod->IsValid()) {
      //Error("TClingCallFunc::SetFuncProto", "Could not find method %s(%s)",
      //      method, proto);
      return;
   }
   const clang::FunctionDecl *decl = fMethod->GetMethodDecl();
   Init(decl);
   if (!IsValid()) {
      //Error("TClingCallFunc::SetFuncProto", "Method %s(%s) has no body.",
      //      method, proto);
   }
}

