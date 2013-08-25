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

llvm::GenericValue TClingCallFunc::convertIntegralToArg(const cling::Value& val,
                                                        const llvm::Type* targetType) const {
   // Do "extended" integral conversion, at least as CINT's dictionaries
   // would have done it: everything is a long, then gets cast to whatever
   // type is expected. "Whatever type" is an llvm type here, thus we do
   // integral conversion, integral to ptr, integral to floating point.

   assert(val.isValid() && "cling::Value must not be invalid!");

   // If both are not builtins but pointers no conversion needed.
   if (val.getLLVMType()->isPointerTy() && targetType->isPointerTy())
      return val.getGV();

   clang::QualType ClangQT = val.getClangType();

   // We support primitive array to ptr conversions like:
   // const char[N] to const char* 
   if (ClangQT->isArrayType() 
       && ClangQT->getArrayElementTypeNoTypeQual()->isCharType()) {
      return val.getGV();
   }

   assert((ClangQT->isBuiltinType() || ClangQT->isPointerType() || ClangQT->isReferenceType())
          && "Conversions supported only on builtin types!");

   llvm::GenericValue result;
   switch(val.getLLVMType()->getTypeID()) {
   case llvm::Type::IntegerTyID : {
      const llvm::APInt& intVal = val.getGV().IntVal;
      switch(targetType->getTypeID()) {
      case llvm::Type::IntegerTyID :
         // We need to match the bitwidths, because this makes LLVM build 
         // different types. Eg. BitWidth = 64 -> i64, BitWidth = 32 -> i32. 
         // If we don't do a bitwidth conversion, later on when making the call 
         // the types mismatch and we are screwed.
         if (ClangQT->isSignedIntegerType())
            result.IntVal
               = intVal.sextOrTrunc(targetType->getPrimitiveSizeInBits());
         else
            result.IntVal 
               = intVal.zextOrTrunc(targetType->getPrimitiveSizeInBits());

         return result;
      case llvm::Type::HalfTyID : /*16 bit floating point*/ 
      case llvm::Type::FloatTyID : 
         if (ClangQT->isSignedIntegerType())
            result.FloatVal = (float)intVal.getSExtValue();
         else
            result.FloatVal = (float)intVal.getZExtValue();               

         return result;
      case llvm::Type::DoubleTyID :
         if (ClangQT->isSignedIntegerType())
            result.DoubleVal = (double)intVal.getSExtValue();
         else
            result.DoubleVal = (double)intVal.getZExtValue();               

         return result;
      case llvm::Type::PointerTyID :
         if (ClangQT->isSignedIntegerType())
            return llvm::PTOGV((void*)intVal.getSExtValue());
         return llvm::PTOGV((void*)intVal.getZExtValue());
      default: 
         return val.getGV();
      }
      break;
   }
   case llvm::Type::PointerTyID : {
      switch(targetType->getTypeID()) {
      case llvm::Type::IntegerTyID: {
         clang::ASTContext& C = fInterp->getSema().getASTContext();
         unsigned long res = (unsigned long)val.getGV().PointerVal;
         llvm::APInt LongAPInt(C.getTypeSize(C.UnsignedLongTy), res);
         result.IntVal = LongAPInt;
         return result;
      }
      default:
         return val.getGV();
      }
      break;
   }

   case llvm::Type::DoubleTyID: {
      switch (targetType->getTypeID()) {
      case llvm::Type::HalfTyID : /*16 bit floating point*/ 
      case llvm::Type::FloatTyID :
         result.FloatVal = (float)val.getGV().DoubleVal;
         return result;

      case llvm::Type::IntegerTyID : {
         // We need to match the bitwidths, because this makes LLVM build 
         // different types. Eg. BitWidth = 64 -> i64, BitWidth = 32 -> i32. 
         // If we don't do a bitwidth conversion, later on when making the call 
         // the types mismatch and we are screwed.
         result.IntVal = llvm::APInt(targetType->getPrimitiveSizeInBits(),
                                     val.getGV().DoubleVal,
                                     true);
         return result;
      }
      default:
         return val.getGV();
      }
   }

   case llvm::Type::HalfTyID : /*16 bit floating point*/ 
   case llvm::Type::FloatTyID: {
      switch (targetType->getTypeID()) {
      case llvm::Type::DoubleTyID :
         result.DoubleVal = val.getGV().FloatVal;
         return result;

      case llvm::Type::IntegerTyID : {
         // We need to match the bitwidths, because this makes LLVM build 
         // different types. Eg. BitWidth = 64 -> i64, BitWidth = 32 -> i32. 
         // If we don't do a bitwidth conversion, later on when making the call 
         // the types mismatch and we are screwed.
         result.IntVal = llvm::APInt(targetType->getPrimitiveSizeInBits(),
                                     val.getGV().FloatVal,
                                     true);
         return result;
      }
      default:
         return val.getGV();
      }
   }

   default : 
      return val.getGV();
   }

   llvm_unreachable("Must be able to convert.");
   // Make the compiler happy:
   return val.getGV();
}

std::string TClingCallFunc::ExprToString(const clang::Expr* expr) const
{
   // Get a string representation of an expression

   clang::PrintingPolicy policy(fInterp->getCI()->
                                getASTContext().getPrintingPolicy());
   policy.SuppressTagKeyword = true;
   policy.SuppressUnwrittenScope = false;
   policy.SuppressInitializers = false;
   policy.AnonymousTagLocations = false;

   std::string buf;
   llvm::raw_string_ostream out(buf);
   expr->printPretty(out, /*Helper=*/0, policy, /*Indentation=*/0);
   out << ';'; // no value printing
   out.flush();
   return buf;
}

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

bool TClingCallFunc::DoesThatTrampolineFuncReturn() const { 
   assert (IsTrampolineFunc() && "Cannot be called on non-trampolines.");
   // If the underlying virtual function returns non void, we have synthesized
   // a default return storage argument __Res.
   return !fMethodAsWritten->getResultType()->isVoidType();
}

bool TClingCallFunc::DoesThatFuncReturnATemporary() const {
   return fEEFunc->hasStructRetAttr();
}

bool TClingCallFunc::IsMemberFunc() const {
   using namespace clang;

   // Trampolines are non member functions, however we treat them as they are,
   // because in their signature we have arg0, which represents the this ptr. 
   // Plus the this ptr in the trampolines is passed in as the member function's
   // this ptr.
   if (IsTrampolineFunc())
      return true;
   const Decl *D = fMethod->GetMethodDecl();
   if (const CXXMethodDecl* MD = dyn_cast<CXXMethodDecl>(D)) {
      return !MD->isStatic();
   }

   return false;
}

const clang::FunctionDecl* TClingCallFunc::GetOriginalDecl() const {
   return (fMethodAsWritten) ? fMethodAsWritten : fMethod->GetMethodDecl();
}

void TClingCallFunc::PushArg(const cling::Value& value) const {
   clang::ASTContext& C = fInterp->getSema().getASTContext();
   fArgVals.push_back(cling::StoredValueRef::bitwiseCopy(C, value));
}

void TClingCallFunc::PushArg(cling::StoredValueRef value) const {
   fArgVals.push_back(value);
}

void TClingCallFunc::SetThisPtr(const clang::CXXMethodDecl* MD, 
                                void* address) const {
   clang::ASTContext& C = fInterp->getSema().getASTContext(); 
   clang::QualType QT = MD->getThisType(C);
   cling::Value val(llvm::PTOGV(address), QT, fEEFunc->arg_begin()->getType());
   fArgVals[0] = cling::StoredValueRef::bitwiseCopy(C, val);
}

void TClingCallFunc::SetReturnPtr(const clang::FunctionDecl* FD, 
                                  void* address) const {
   clang::ASTContext& C = fInterp->getSema().getASTContext();
   clang::QualType QT;
   if (IsTrampolineFunc())
      QT = llvm::cast<clang::CXXMethodDecl>(FD)->getThisType(C);
   else
      QT = FD->getResultType();
   cling::Value val(llvm::PTOGV(address), QT, fEEFunc->arg_begin()->getType());
   fArgVals[1] = cling::StoredValueRef::bitwiseCopy(C, val);
}

void TClingCallFunc::BuildTrampolineFunc(clang::CXXMethodDecl* MD) {
   // In case of virtual function we need a trampoline for the vtable 
   // evaluation like:
   // 1. void unique_name(CLASS* This, Arg1* a, ...[, CLASSRes* result = 0]) {
   // 2.  if (result)
   // 3.    *result = This->Function(*a, b, c);
   // 4.  else
   // 5.    This->Function(*a, b, c);
   // 6. }
   //
   // if the virtual returns a reference the CLASSRes* becomes CLASSRes**
   // and line 3 becomes:
   // 3.    *result = &This->Function(*a, b, c);
   //
   assert(MD && "MD cannot be null.");
   assert(MD->isVirtual() && "MD has to be virtual.");
   using namespace clang;
   Sema& S = fInterp->getSema();
   ASTContext& C = S.getASTContext();

   std::string FunctionName = "__callfunc_vtable_trampoline";
   fInterp->createUniqueName(FunctionName);
   IdentifierInfo& II = C.Idents.get(FunctionName);
   SourceLocation Loc;

   // Could trigger deserialization of decls.
   cling::Interpreter::PushTransactionRAII RAII(fInterp);

   // Copy-pasted and adapted from cling's DeclExtractor.cpp
   FunctionDecl* TrampolineFD 
      = dyn_cast_or_null<FunctionDecl>(S.ImplicitlyDefineFunction(Loc, II, S.TUScope));
   if (TrampolineFD) {
      TrampolineFD->setImplicit(false); // Better for debugging
      // We can't PushDeclContext, because we don't have scope.
      Sema::ContextRAII pushedDC(S, TrampolineFD);

      // NOTE:
      // We know that our function returns an int, however we are not going
      // to add a return statement, because we use that function as a 
      // trampoline to call. Valgrind will be happy because LLVM
      // generates a return result, which is (false?) initialized.
      llvm::SmallVector<Stmt*, 2> Stmts;
      llvm::SmallVector<ParmVarDecl*, 4> Params;
      // Pass in parameter This
      QualType ThisQT = MD->getThisType(C);
      ParmVarDecl* ThisPtrParm 
         = ParmVarDecl::Create(C, TrampolineFD, Loc, Loc, 
                               &C.Idents.get("This"), ThisQT,
                               C.getTrivialTypeSourceInfo(ThisQT, Loc),
                               SC_None, /*DefaultArg*/0);
      Params.push_back(ThisPtrParm);

      // Recreate the same arg nodes for the trampoline and use them 
      // instead
      ParmVarDecl* PVD = 0;
      Expr* defaultArg = 0;
      for (FunctionDecl::param_const_iterator I = MD->param_begin(), 
              E = MD->param_end(); I != E; ++I) {
         // For now instead of creating a copy - just forward to the AST 
         // node of the original function
         defaultArg = (*I)->getDefaultArg();
         PVD = ParmVarDecl::Create(C, TrampolineFD, Loc, Loc, 
                                   (*I)->getIdentifier(), (*I)->getType(),
                                   C.getTrivialTypeSourceInfo((*I)->getType(), Loc),
                                   SC_None, defaultArg);
         Params.push_back(PVD);
      }
         
      // Create the parameter for the result type
      ParmVarDecl* ResParm = 0;
      if (!MD->getResultType()->isVoidType()) {
         // We must default the parameter to zero.
         ExprResult Zero = S.ActOnIntegerConstant(Loc, 0);
         // Since we deref it it always has to be ptr type.
         QualType ResQT = MD->getResultType();
         // In case the return result is a reference type we have to transform
         // it to pointer. 
         if (ResQT->isReferenceType())
            ResQT = C.getPointerType(ResQT.getNonReferenceType());
         ResQT = C.getPointerType(ResQT);
         ResParm 
            = ParmVarDecl::Create(C, TrampolineFD, Loc, Loc,
                                  &C.Idents.get("__Res"), ResQT,
                                  C.getTrivialTypeSourceInfo(ResQT, Loc),
                                  SC_None, Zero.take());

         Params.push_back(ResParm);
      }

            
      // Change the type of the function to consider the new paramlist.
      llvm::SmallVector<QualType, 4> ParamQTs;
      for(size_t i = 0; i < Params.size(); ++i)
         ParamQTs.push_back(Params[i]->getType());

      const FunctionProtoType* Proto 
         = llvm::cast<FunctionProtoType>(TrampolineFD->getType().getTypePtr());
      QualType NewFDQT = C.getFunctionType(C.VoidTy, ParamQTs,
                                           Proto->getExtProtoInfo());
      TrampolineFD->setType(NewFDQT);
      TrampolineFD->setParams(Params);

      // Copy-pasted and adapted from cling's DynamicLookup.cpp
      CXXScopeSpec CXXSS;
      LookupResult MemberLookup(S, MD->getDeclName(),
                                Loc, Sema::LookupMemberName);
      // Add the declaration. No lookup is performed, we know that this
      // decl doesn't exist yet.
      MemberLookup.addDecl(MD, AS_public);
      MemberLookup.resolveKind();
      Expr* ExprBase 
         = S.BuildDeclRefExpr(/*ThisPTR*/Params[0], ThisQT, VK_LValue, Loc
                              ).take();
         
      // Synthesize implicit casts if needed.
      bool isArrow = true;
      ExprBase 
         = S.PerformMemberExprBaseConversion(ExprBase,isArrow).take();
      Expr* MemberExpr 
         = S.BuildMemberReferenceExpr(ExprBase, ThisQT, Loc, isArrow, 
                                      CXXSS, Loc, 
                                      /*FirstQualifierInScope=*/0,
                                      MemberLookup, /*TemplateArgs=*/0
                                      ).take();
      // Build the arguments as declrefs to the params
      llvm::SmallVector<Expr*, 4> ExprArgs;
      ExprResult Arg;
      DeclarationNameInfo NameInfo;
      // We need to skip the first argument, it is the artificial this ptr
      // and the last if exist
      size_t ParamsSize = (ResParm) ? Params.size() -1 : Params.size();
      for (size_t i = 1; i < ParamsSize; ++i) {
         NameInfo = DeclarationNameInfo(Params[i]->getDeclName(), 
                                        Params[i]->getLocStart());
         Arg = S.BuildDeclarationNameExpr(CXXSS, NameInfo, Params[i]);
         ExprArgs.push_back(Arg.take());
      }
      // Set the warning to ignore (they are by definition spurrious
      // for a trampoline function).
      DiagnosticsEngine& Diag = fInterp->getCI()->getDiagnostics();
      Diag.setDiagnosticMapping(clang::diag::warn_format_nonliteral_noargs,
                                clang::diag::MAP_IGNORE, SourceLocation());
      Diag.setDiagnosticMapping(clang::diag::warn_format_nonliteral,
                                clang::diag::MAP_IGNORE, SourceLocation());
      // Build the actual call
      ExprResult theCall = S.ActOnCallExpr(S.TUScope, MemberExpr, Loc,
                                           MultiExprArg(ExprArgs.data(), 
                                                        ExprArgs.size()),
                                           Loc
                                           );
      // Create the if stmt
      if (ResParm) {
         // 1. if (result)
         // 2.   *result = This->Function(*a, b, c);
         // 3. else
         // 4.   This->Function(*a, b, c)
         // or in case of return by ref:
         // line 2 becomes:
         //   *result = &This->Function(*a, b, c)
         //
         NameInfo = DeclarationNameInfo(ResParm->getDeclName(), 
                                        ResParm->getLocStart());
         ExprResult DRE, LHS, RHS, BoolCond, BinOp;
         DRE = S.BuildDeclarationNameExpr(CXXSS, NameInfo, ResParm);
         LHS = S.BuildUnaryOp(/*Scope*/0, Loc, UO_Deref, DRE.take());
         // If it was a reference we have made **__Res and we need to take the
         // address of the call. i.e *__Res = &theCall(...) 
         RHS = theCall;
         if (MD->getResultType()->isReferenceType())
            RHS = S.BuildUnaryOp(/*Scope*/0, Loc, UO_AddrOf, RHS.take());
           
         BinOp = S.BuildBinOp(/*Scope*/0, Loc, BO_Assign, LHS.take(), 
                              RHS.take());
         // Add the needed casts needed for turning the result into bool
         // condition
         BoolCond = S.ActOnBooleanCondition(/*Scope*/0, Loc, DRE.get());
         Sema::FullExprArg FullCond(S.MakeFullExpr(BoolCond.get(), Loc));
         // Note that here we reuse the call expr from the then part into
         // the else part of the if stmt. In general that is dangerous if
         // they are in different decl contexts and so on, but here it 
         // shouldn't be.
         StmtResult IfStmt = S.ActOnIfStmt(Loc, FullCond, /*CondVar*/0, 
                                           BinOp.take(), Loc, 
                                           theCall.take());
         Stmts.push_back(IfStmt.take()); 
      }
      else
         Stmts.push_back(theCall.take());

      CompoundStmt* CS = new (C) CompoundStmt(C, Stmts, Loc, Loc);

      TrampolineFD->setBody(CS);
   
      fMethodAsWritten = llvm::cast<clang::CXXMethodDecl>(MD);
      // We also need to update the current method. It should point to 
      // the trampoline. 
      fMethod = new TClingMethodInfo(fInterp, TrampolineFD);
      // There is no code generated for the wrapper, force codegen on it
      CodeGenDecl(TrampolineFD);
   }
}

void TClingCallFunc::CodeGenDecl(const clang::FunctionDecl* FD) {
   if (!FD)
      return;
   bool needInstantiation = false;
   if (!FD->isDefined(FD)) {
      if (FD->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate) {
         // We have a function template instance, let's check the
         // template.
         const clang::FunctionDecl *tmplt = FD->getInstantiatedFromMemberFunction();
         if (tmplt && !tmplt->isDefined(tmplt)) {
            return;
         }
         if (FD->getTemplateSpecializationInfo()) {
            clang::FunctionTemplateDecl *tmpltDecl = FD->getTemplateSpecializationInfo()->getTemplate();
            if (tmpltDecl && !tmpltDecl->hasBody()) {
               return;
            }
         }
         if (FD->isImplicitlyInstantiable()) {
            needInstantiation = true;
         }
      } else {
         // Not an error; the caller might just check whether this function can
         // be called at all.
         //Error("CodeGenDecl", "Cannot codegen %s: no definition available!",
         //      FD->getNameAsString().c_str());
         return;
      }
   }

   if (needInstantiation) {
      // Could trigger deserialization of decls.
      cling::Interpreter::PushTransactionRAII RAII(fInterp);
      clang::Sema &S = fInterp->getSema();
      clang::FunctionDecl *FDmod = const_cast<clang::FunctionDecl*>(FD);
      S.InstantiateFunctionDefinition(clang::SourceLocation(), FDmod,
                                      /*Recursive=*/ true,
                                      /*DefinitionRequired=*/ true);
   }
   cling::CompilationOptions CO;
   CO.DeclarationExtraction = 0;
   CO.ValuePrinting = cling::CompilationOptions::VPDisabled;
   CO.ResultEvaluation = 0;
   CO.DynamicScoping = 0;
   CO.Debug = 0;
   CO.CodeGeneration = 1;

   cling::Transaction T(CO, FD->getASTContext());

   T.append(const_cast<clang::FunctionDecl*>(FD));
   T.setState(cling::Transaction::kCompleted);

   fInterp->emitAllDecls(&T);
   assert(T.getState() == cling::Transaction::kCommitted
          && "Compilation should never fail!");
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

void TClingCallFunc::Init(const clang::FunctionDecl *FD)
{
   fEEFunc = 0;
   fEEAddr = 0;
   bool isMemberFunc = true;
   clang::CXXMethodDecl *MD 
      = llvm::dyn_cast<clang::CXXMethodDecl>(const_cast<clang::FunctionDecl*>(FD));

   // If MD is virual function we will need a trampoline, evaluating the vtable.
   if (MD && MD->isVirtual())
      BuildTrampolineFunc(MD);

   const clang::DeclContext *DC = FD->getDeclContext();
   while (DC->getDeclKind() == clang::Decl::LinkageSpec)
      DC = DC->getParent();
   if (DC->isTranslationUnit() || DC->isNamespace() || (MD && MD->isStatic())) {
      // Free function or static member function.
      isMemberFunc = false;
   }

   //
   //  Mangle the function name, if necessary.
   //
   std::string FuncName;
   if (IsTrampolineFunc())
      fInterp->maybeMangleDeclName(fMethod->GetMethodDecl(), FuncName);
   else
      fInterp->maybeMangleDeclName(FD, FuncName);
      
   //
   //  Check the execution engine for the function.
   //
   llvm::ExecutionEngine *EE = fInterp->getExecutionEngine();
   fEEFunc = EE->FindFunctionNamed(FuncName.c_str());

   // In many cases if the Decl in unused no code is generated by cling/clang.
   // In such cases when we have the Decl in the AST but we don't have code 
   // produced yet we have to force that to happen.
   if (!fEEFunc)
      CodeGenDecl(const_cast<clang::FunctionDecl*>(FD));

   //
   //  Check again the execution engine for the function.
   //
   fEEFunc = EE->FindFunctionNamed(FuncName.c_str());
   if (fEEFunc) {
      // Execution engine had it, get the mapping.
      fEEAddr = EE->getPointerToFunction(fEEFunc);
   }
   else {
      // Execution engine does not have it, check
      // the loaded shareable libraries.

      // Avoid spurious error message if we look for an
      // unimplemented (but declared) function.
      fInterp->suppressLazyFunctionCreatorDiags(true);
      void *FP = EE->getPointerToNamedFunction(FuncName,
                 /*AbortOnFailure=*/false);
      fInterp->suppressLazyFunctionCreatorDiags(false);
      if (FP == unresolvedSymbol) {
         // We failed to find an implementation for the function, the 
         // interface requires the 'address' to be zero.
         fEEAddr = 0;
      } else if (FP) {
         fEEAddr = FP;

         // Create a llvm function we can use to call it with later.
         //FIXME: No need of LLVMContext for doing such.
         llvm::LLVMContext &Context = *fInterp->getLLVMContext();
         llvm::SmallVector<llvm::Type *, 8> Params;
         if (isMemberFunc) {
            // Force the invisible this pointer arg to pointer to char.
            Params.push_back(llvm::PointerType::getUnqual(
                                llvm::IntegerType::get(Context, CHAR_BIT)));
         }
         for (unsigned I = 0U; I < FD->getNumParams(); ++I) {
            const clang::ParmVarDecl *PVD = FD->getParamDecl(I);
            clang::QualType QT = PVD->getType();
            llvm::Type *argtype = const_cast<llvm::Type*>(fInterp->getLLVMType(QT));
            if (argtype == 0) {
               // We are not in good shape, quit while we are still alive.
               return;
            }
            Params.push_back(argtype);
         }
         llvm::Type *ReturnType = 0;
         if (llvm::isa<clang::CXXConstructorDecl>(FD)) {
            // Force the return type of a constructor to be long.
            ReturnType = llvm::IntegerType::get(Context, sizeof(long) *
                                                CHAR_BIT);
         }
         else {
            ReturnType = const_cast<llvm::Type*>(fInterp->getLLVMType(FD->getResultType()));
         }
         if (ReturnType) {
            
            // Create the llvm function type.
            llvm::FunctionType *FT = llvm::FunctionType::get(ReturnType, Params,
                                     /*isVarArg=*/false);
            // Create the ExecutionEngine function.
            // Note: We use weak linkage here so lookup failure does not abort.
            llvm::Function *F = llvm::Function::Create(FT,
                                llvm::GlobalValue::ExternalWeakLinkage,
                                FuncName, fInterp->getModule());
            // FIXME: This probably does not work for Windows!
            // See ASTContext::getFunctionType() for proper way to set it.
            // Actually this probably is not needed.
            F->setCallingConv(llvm::CallingConv::C);
            // Map the created ExecutionEngine function to the
            // address found in the shareable library, so the next
            // time we do a lookup it will be found.
            EE->addGlobalMapping(F, FP);
            // Set our state.
            fEEFunc = F;
         }
      }
   }
}

void TClingCallFunc::Invoke(cling::StoredValueRef* result /*= 0*/) const
{
   if (result) 
      *result = cling::StoredValueRef::invalidValue();

   // It should make no difference if we are working with the trampoline or
   // the real decl.
   const clang::FunctionDecl *FD = fMethod->GetMethodDecl();

   // If the function returns a temporary adjust the temporary's lifetime.
   if (fEEFunc->hasStructRetAttr()) {
      assert(!GetReturnPtr().isValid()
             && "We already have set the return result!?");
      clang::ASTContext& C = fInterp->getSema().getASTContext();
      llvm::Type* LLVMRetTy = fEEFunc->getReturnType();
      clang::QualType ClangRetTy = FD->getResultType();
      SetReturnPtr(cling::StoredValueRef::allocate(C, ClangRetTy, LLVMRetTy));
   }

   // We are going to loop over the JIT function args.
   llvm::FunctionType *ft = fEEFunc->getFunctionType();
     
   unsigned num_params = ft->getNumParams();
   unsigned min_args = FD->getMinRequiredArguments();
   // For trampolines we pass in the first argument as a fake this ptr.
   unsigned num_given_args = GetArgValsSize();

   if (IsMemberFunc() && !IsTrampolineFunc()) {
      // Adjust for the hidden this pointer first argument.
      ++min_args;
   }

   if (num_given_args < min_args) {
      // Not all required arguments given.
      Error("TClingCallFunc::Invoke",
            "Not enough function arguments given to %s (min: %u max:%u, given: %u)",
            fMethod->Name(), min_args, num_params, num_given_args);
      return;
   }
   else if (num_given_args > num_params) {
      if (!fIgnoreExtraArgs) {// FIXME: This should not be an error.
         Error("TClingCallFunc::Invoke",
               "Too many function arguments given to %s (min: %u max: %u, given: %u)",
               fMethod->Name(), min_args, num_params, num_given_args);
         return;
      }
   }

   // This will be the arguments actually passed to the JIT function.
   std::vector<llvm::GenericValue> Args;

   if (num_given_args < num_params) {
      // This means that we have synthesized artificial default arg __Res
      if (IsTrampolineFunc() && DoesThatTrampolineFuncReturn()) {
         if (!fArgVals[1].isValid())
            SetReturnPtr(fMethodAsWritten, /*address*/0);
         // Update the count of given arguments.
         num_given_args = GetArgValsSize();
      }

      // If there are still arguments to be resolved
      unsigned num_default_args_to_resolve = num_params - num_given_args;
      unsigned arg_index;
      const clang::ParmVarDecl* pvd = 0;
      for (size_t i = 0; i < num_default_args_to_resolve; ++i) {
         // If this was a member function, skip the this ptr - it has already 
         // been handled.
         arg_index = num_given_args + i                           // count in func decl
                     - int(IsMemberFunc() && !IsTrampolineFunc()) // tramp 'this' adj.
                     - int(DoesThatFuncReturnATemporary())        // return-by-value adj.
                     - int(IsTrampolineFunc() && DoesThatTrampolineFuncReturn());    // tramp return adj.
         //arg_index = num_given_args - min_args + i;
         //arg_index = num_given_args + i - IsMemberFunc() + IsTrampolineFunc();
         // Use the default value from the decl.
         pvd = FD->getParamDecl(arg_index);
         assert(pvd->hasDefaultArg() && "No default for argument!");

         // TODO: Optimize here.
         const clang::Expr* expr = pvd->getDefaultArg();
         cling::StoredValueRef valref = EvaluateExpr(expr);
         if (valref.isValid()) {
            // const cling::Value& val = valref.get();
            // if (!val.getClangType()->isIntegralType(context) &&
            //       !val.getClangType()->isRealFloatingType() &&
            //       !val.getClangType()->canDecayToPointerType()) {
            //    // Invalid argument type.
            //    Error("TClingCallFunc::Invoke", "Default for argument %lu: %s",
            //          i, ExprToString(expr).c_str());
            //    Error("TClingCallFunc::Invoke",
            //          "is not of integral, floating, or pointer type!");
            //    return;
            // }
            //const llvm::Type *ty = ft->getParamType(arg_index);
            PushArg(valref.get());
            //Args.push_back(convertIntegralToArg(val, ty));
         }
         else {
            Error("TClingCallFunc::Invoke",
                  "Could not evaluate default for argument %u: %s",
                  arg_index, ExprToString(expr).c_str());
            return;
         }
      } // for default arg
   } // if less args than params


   // FIXME: Redesign the Trampoline function to take the return argument __Res
   // as a first argument and that if statement and few more will disappear.
   //
   // LLVM return by-value takes the variable where the result should be stored
   // as a first argument.
   if (fEEFunc->hasStructRetAttr())
      Args.push_back(fArgVals[1].get().getGV());


   // Add the this ptr if valid.
   if(fArgVals[0].isValid())
      Args.push_back(fArgVals[0].get().getGV());

   unsigned numPassedArgs = fArgVals.size();
   assert(numPassedArgs >= 2 && "Too few actual parameters.");
   numPassedArgs -= 2;
   unsigned trampolineRetArg = 0;
   if (IsTrampolineFunc() && fArgVals[1].isValid()
       && !fEEFunc->hasStructRetAttr())
      trampolineRetArg = 1;

   // After the following loop we would have numPassedArgs more arguments
   // in Args than we have now. Complain now if that's too much or not
   // enough.
   if (numPassedArgs + Args.size() + trampolineRetArg > ft->getNumParams()) {
      if (!fIgnoreExtraArgs) {
         Error("TClingCallFunc::Invoke",
               "Passed %d parameters to ExecutionEngine's function"
               " taking %d parameters!", numPassedArgs, ft->getNumParams());
         return;
      } else {
         if (ft->getNumParams() < Args.size()) {
            Error("TClingCallFunc::Invoke",
                  "Passed %d special parameters to ExecutionEngine's function"
                  " taking %d parameters!", (int)Args.size(),
                  ft->getNumParams());
            return;
         }
         numPassedArgs = ft->getNumParams() - Args.size();
      }
   } else if (numPassedArgs + Args.size() + trampolineRetArg
              < ft->getNumParams()) {
      Error("TClingCallFunc::Invoke",
            "Passed only %d parameters to ExecutionEngine's function"
            " taking %d parameters!", numPassedArgs, ft->getNumParams());
      return;
   }

   // If it is a trampoline do not iterate over the last
   for (size_t i = 2, e = numPassedArgs + 2; i < e; ++i) {
      // if (IsTrampolineFunc()) {
      //    // If it is the last argument of a non-void function, that should be
      //    // the arg for storing the result.
      //    if (i == e-1 && !ft->getReturnType()->isVoidTy()) {
      //       // Add return ptr arg
      //       if (i == num_given_args)
      //          // We have a user-provided return value address.
      //          Args.push_back(fArgVals[e].get().getGV());
      //       else
      //          // NULL
      //          Args.push_back(llvm::PTOGV(0));
      //       continue;
      //    }
      // }
      // We have a user-provided argument value.
      // If this was a member function, skip the this ptr - it has already been
      // handled. Likewise, if return-by-val, skip return address.
      unsigned eearg_index = i - 2 + (int)IsMemberFunc() + (int)DoesThatFuncReturnATemporary();
      const llvm::Type *ty = ft->getParamType(eearg_index);
      if (ty != fArgVals[i].get().getLLVMType())
         Args.push_back(convertIntegralToArg(fArgVals[i].get(), ty));
      else
         Args.push_back(fArgVals[i].get().getGV());
   }

   // Add the return ptr if valid.
   if(fArgVals[1].isValid() && !fEEFunc->hasStructRetAttr()) {
      assert(IsTrampolineFunc() && "Should be only valid for trampolines.");
      Args.push_back(fArgVals[1].get().getGV());
   }

   llvm::GenericValue return_val = fInterp->getExecutionEngine()->runFunction(fEEFunc, Args);
   // if fEEFunc->hasStructRetAttr() we already have the return result
   if (result) {
      if (DoesThatFuncReturnATemporary()) {
         *result = GetReturnPtr();
         return;
      }
      using namespace cling;
      const clang::ASTContext& C = fInterp->getSema().getASTContext();
      if (ft->getReturnType()->getTypeID() == llvm::Type::PointerTyID) {
         // Note: The cint interface requires pointers and references to be
         //       returned as unsigned long.
         Value convVal(return_val, FD->getResultType(), 
                       fEEFunc->getReturnType());

         clang::QualType QT = C.UnsignedLongTy;
         const llvm::Type* ty = fInterp->getLLVMType(QT);
         llvm::GenericValue ulong_return_val = convertIntegralToArg(convVal, ty);
         *result = StoredValueRef::bitwiseCopy(C, Value(ulong_return_val, QT, ty));
      } else {
         Value V = Value(return_val, GetOriginalDecl()->getResultType(), 
                         ft->getReturnType());
         *result = StoredValueRef::bitwiseCopy(C, V);
      }
   }
}
