// @(#)root/core/meta:$Id$
// Author: Paul Russo   30/07/2012

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
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

#include "TError.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/LookupHelper.h"
#include "cling/Interpreter/StoredValueRef.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/LLVMContext.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Type.h"

#include <string>
#include <vector>

// This ought to be declared by the implementer .. oh well...
extern void unresolvedSymbol();

void TClingCallFunc::Exec(void *address) const
{
   if (!IsValid()) {
      return;
   }
   const clang::Decl *D = fMethod->GetMethodDecl();
   const clang::CXXMethodDecl *MD = llvm::dyn_cast<clang::CXXMethodDecl>(D);
   const clang::DeclContext *DC = D->getDeclContext();
   if (DC->isTranslationUnit() || DC->isNamespace() || (MD && MD->isStatic())) {
      // Free function or static member function.
      Invoke(fArgs);
   }
   else {
      // Member function.
      if (const clang::CXXConstructorDecl *CD =
               llvm::dyn_cast<clang::CXXConstructorDecl>(D)) {
         clang::ASTContext &Context = CD->getASTContext();
         const clang::RecordDecl *RD = llvm::cast<clang::RecordDecl>(DC);
         if (!RD->getDefinition()) {
            // Forward-declared class, we do not know what the size is.
            return;
         }
         const clang::ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
         int64_t size = Layout.getSize().getQuantity();
         address = malloc(size);
      }
      else {
         if (!address) {
            Error("TClingCallFunc::Exec",
                  "calling member function with no object pointer!\n");
         }
      }
      std::vector<llvm::GenericValue> args;
      llvm::GenericValue this_ptr;
      this_ptr.IntVal = llvm::APInt(sizeof(unsigned long) * CHAR_BIT,
                                    reinterpret_cast<unsigned long>(address));
      args.push_back(this_ptr);
      args.insert(args.end(), fArgs.begin(), fArgs.end());
      Invoke(args);
   }
}

long TClingCallFunc::ExecInt(void *address) const
{
   // Yes, the function name has Int in it, but it
   // returns a long.  This is a matter of CINT history.
   if (!IsValid()) {
      return 0L;
   }
   llvm::GenericValue val;
   const clang::Decl *D = fMethod->GetMethodDecl();
   const clang::CXXMethodDecl *MD = llvm::dyn_cast<clang::CXXMethodDecl>(D);
   const clang::DeclContext *DC = D->getDeclContext();
   if (DC->isTranslationUnit() || DC->isNamespace() || (MD && MD->isStatic())) {
      // Free function or static member function.
      val = Invoke(fArgs);
   }
   else {
      // Member function.
      if (const clang::CXXConstructorDecl *CD =
               llvm::dyn_cast<clang::CXXConstructorDecl>(D)) {
         clang::ASTContext &Context = CD->getASTContext();
         const clang::RecordDecl *RD = llvm::cast<clang::RecordDecl>(DC);
         if (!RD->getDefinition()) {
            // Forward-declared class, we do not know what the size is.
            return 0L;
         }
         const clang::ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
         int64_t size = Layout.getSize().getQuantity();
         address = malloc(size); // this is bad, we really need to call the class' own operator new but oh well

         std::vector<llvm::GenericValue> args;
         llvm::GenericValue this_ptr;
         this_ptr.IntVal = llvm::APInt(sizeof(unsigned long) * CHAR_BIT,
                                       reinterpret_cast<unsigned long>(address));
         args.push_back(this_ptr);
         args.insert(args.end(), fArgs.begin(), fArgs.end());
         val  = Invoke(args);
         
         // We don't really mean to call the constructor and return its (lack of)
         // return value, we meant to execute and return 'new TypeOf(...)' and
         // return the allocated address, so here you go:
         return (long)(address);
      }
      else {
         if (!address) {
            fprintf(stderr, "TClingCallFunc::Exec: error: "
                    "calling member function with no object pointer!\n");
         }
      }
      std::vector<llvm::GenericValue> args;
      llvm::GenericValue this_ptr;
      this_ptr.IntVal = llvm::APInt(sizeof(unsigned long) * CHAR_BIT,
                                    reinterpret_cast<unsigned long>(address));
      args.push_back(this_ptr);
      args.insert(args.end(), fArgs.begin(), fArgs.end());
      val  = Invoke(args);
   }
   return static_cast<long>(val.IntVal.getSExtValue());
}

long long TClingCallFunc::ExecInt64(void *address) const
{
   if (!IsValid()) {
      return 0LL;
   }
   llvm::GenericValue val;
   const clang::Decl *D = fMethod->GetMethodDecl();
   const clang::CXXMethodDecl *MD = llvm::dyn_cast<clang::CXXMethodDecl>(D);
   const clang::DeclContext *DC = D->getDeclContext();
   if (DC->isTranslationUnit() || DC->isNamespace() || (MD && MD->isStatic())) {
      // Free function or static member function.
      val = Invoke(fArgs);
   }
   else {
      // Member function.
      if (const clang::CXXConstructorDecl *CD =
               llvm::dyn_cast<clang::CXXConstructorDecl>(D)) {
         clang::ASTContext &Context = CD->getASTContext();
         const clang::RecordDecl *RD = llvm::cast<clang::RecordDecl>(DC);
         if (!RD->getDefinition()) {
            // Forward-declared class, we do not know what the size is.
            return 0LL;
         }
         const clang::ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
         int64_t size = Layout.getSize().getQuantity();
         address = malloc(size);
      }
      else {
         if (!address) {
            fprintf(stderr, "TClingCallFunc::Exec: error: "
                    "calling member function with no object pointer!\n");
         }
      }
      std::vector<llvm::GenericValue> args;
      llvm::GenericValue this_ptr;
      this_ptr.IntVal = llvm::APInt(sizeof(unsigned long) * CHAR_BIT,
                                    reinterpret_cast<unsigned long>(address));
      args.push_back(this_ptr);
      args.insert(args.end(), fArgs.begin(), fArgs.end());
      val = Invoke(args);
   }
   return static_cast<long long>(val.IntVal.getSExtValue());
}

double TClingCallFunc::ExecDouble(void *address) const
{
   if (!IsValid()) {
      return 0.0;
   }
   llvm::GenericValue val;
   const clang::Decl *D = fMethod->GetMethodDecl();
   const clang::CXXMethodDecl *MD = llvm::dyn_cast<clang::CXXMethodDecl>(D);
   const clang::DeclContext *DC = D->getDeclContext();
   if (DC->isTranslationUnit() || DC->isNamespace() || (MD && MD->isStatic())) {
      // Free function or static member function.
      val = Invoke(fArgs);
   }
   else {
      // Member function.
      if (const clang::CXXConstructorDecl *CD =
               llvm::dyn_cast<clang::CXXConstructorDecl>(D)) {
         clang::ASTContext &Context = CD->getASTContext();
         const clang::RecordDecl *RD = llvm::cast<clang::RecordDecl>(DC);
         if (!RD->getDefinition()) {
            // Forward-declared class, we do not know what the size is.
            return 0.0;
         }
         const clang::ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
         int64_t size = Layout.getSize().getQuantity();
         address = malloc(size);
      }
      else {
         if (!address) {
            fprintf(stderr, "TClingCallFunc::Exec: error: "
                    "calling member function with no object pointer!\n");
         }
      }
      std::vector<llvm::GenericValue> args;
      llvm::GenericValue this_ptr;
      this_ptr.IntVal = llvm::APInt(sizeof(unsigned long) * CHAR_BIT,
                                    reinterpret_cast<unsigned long>(address));
      args.push_back(this_ptr);
      args.insert(args.end(), fArgs.begin(), fArgs.end());
      val = Invoke(args);
   }
   return val.DoubleVal;
}

TClingMethodInfo *TClingCallFunc::FactoryMethod() const
{
   return new TClingMethodInfo(*fMethod);
}

void TClingCallFunc::Init()
{
   delete fMethod;
   fMethod = 0;
   fEEFunc = 0;
   fEEAddr = 0;
   fArgVals.clear();
   fArgs.clear();
}

void *TClingCallFunc::InterfaceMethod() const
{
   if (!IsValid()) {
      return 0;
   }
   return fEEAddr;
}

bool TClingCallFunc::IsValid() const
{
   return fEEAddr;
}

void TClingCallFunc::ResetArg()
{
   fArgVals.clear();
   fArgs.clear();
}

void TClingCallFunc::SetArg(long param)
{
   llvm::GenericValue gv;
   gv.IntVal = llvm::APInt(sizeof(long) * CHAR_BIT, param);
   fArgs.push_back(gv);
}

void TClingCallFunc::SetArg(double param)
{
   llvm::GenericValue gv;
   gv.DoubleVal = param;
   fArgs.push_back(gv);
}

void TClingCallFunc::SetArg(long long param)
{
   llvm::GenericValue gv;
   gv.IntVal = llvm::APInt(sizeof(long long) * CHAR_BIT, param);
   fArgs.push_back(gv);
}

void TClingCallFunc::SetArg(unsigned long long param)
{
   llvm::GenericValue gv;
   gv.IntVal = llvm::APInt(sizeof(unsigned long long) * CHAR_BIT, param);
   fArgs.push_back(gv);
}

void TClingCallFunc::SetArgArray(long *paramArr, int nparam)
{
   for (int i = 0; i < nparam; ++i) {
      llvm::GenericValue gv;
      gv.IntVal = llvm::APInt(sizeof(long) * CHAR_BIT, paramArr[i]);
      fArgs.push_back(gv);
   }
}

static void evaluateArgList(cling::Interpreter *interp,
                            const std::string &ArgList,
                            std::vector<cling::StoredValueRef> &EvaluatedArgs)
{
   clang::PrintingPolicy Policy(interp->getCI()->
      getASTContext().getPrintingPolicy());
   Policy.SuppressTagKeyword = true;
   Policy.SuppressUnwrittenScope = true;
   Policy.SuppressInitializers = true;
   Policy.AnonymousTagLocations = false;

   llvm::SmallVector<clang::Expr*, 4> exprs;
   interp->getLookupHelper().findArgList(ArgList, exprs);
   for (llvm::SmallVector<clang::Expr*, 4>::const_iterator I = exprs.begin(),
         E = exprs.end(); I != E; ++I) {
      std::string empty;
      llvm::raw_string_ostream tmp(empty);
      (*I)->printPretty(tmp, /*PrinterHelper=*/0, Policy, /*Indentation=*/0);
      cling::StoredValueRef val;
      cling::Interpreter::CompilationResult cres =
         interp->evaluate(tmp.str(), &val);
      if (cres != cling::Interpreter::kSuccess) {
         // Bad expression, all done.
         break;
      }
      EvaluatedArgs.push_back(val);
   }
}

void TClingCallFunc::SetArgs(const char *params)
{
   evaluateArgList(fInterp, params, fArgVals);
   clang::ASTContext &Context = fInterp->getCI()->getASTContext();
   for (unsigned I = 0U, E = fArgVals.size(); I < E; ++I) {
      const cling::Value& val = fArgVals[I].get();
      if (!val.type->isIntegralType(Context) &&
            !val.type->isRealFloatingType() && !val.type->isPointerType()) {
         // Invalid argument type.
         Error("TClingCallFunc::SetArgs", "Given arguments: %s", params);
         Error("TClingCallFunc::SetArgs", "Argument number %u is not of "
               "integral, floating, or pointer type!", I);
         break;
      }
      fArgs.push_back(val.value);
   }
}

void TClingCallFunc::SetFunc(const TClingClassInfo *info, const char *method, const char *params, long *offset)
{
   delete fMethod;
   fMethod = new TClingMethodInfo(fInterp);
   fEEFunc = 0;
   fEEAddr = 0;
   const cling::LookupHelper& lh = fInterp->getLookupHelper();
   const clang::FunctionDecl *decl = lh.findFunctionArgs(info->GetDecl(), method, params);
   if (!decl) {
      return;
   }
   fMethod->Init(decl);
   Init(decl);
   if (offset) {
      offset = 0L;
   }
   // FIXME: We should eliminate the double parse here!
   fArgVals.clear();
   fArgs.clear();
   evaluateArgList(fInterp, params, fArgVals);
   clang::ASTContext &Context = fInterp->getCI()->getASTContext();
   for (unsigned I = 0U, E = fArgVals.size(); I < E; ++I) {
      const cling::Value& val = fArgVals[I].get();
      if (!val.type->isIntegralType(Context) &&
            !val.type->isRealFloatingType() && !val.type->isPointerType()) {
         // Invalid argument type, cint skips it, strange.
         continue;
      }
      fArgs.push_back(val.value);
   }
}

void TClingCallFunc::SetFunc(const TClingMethodInfo *info)
{
   delete fMethod;
   fMethod = 0;
   fEEFunc = 0;
   fEEAddr = 0;
   fMethod = new TClingMethodInfo(*info);
   if (!fMethod->IsValid()) {
      return;
   }
   Init(fMethod->GetMethodDecl());
}

void TClingCallFunc::SetFuncProto(const TClingClassInfo *info, const char *method,
                                  const char *proto, long *offset)
{
   delete fMethod;
   fMethod = new TClingMethodInfo(fInterp);
   fEEFunc = 0;
   fEEAddr = 0;
   if (!info->IsValid()) {
      return;
   }
   const cling::LookupHelper& lh = fInterp->getLookupHelper();
   const clang::FunctionDecl *FD = lh.findFunctionProto(info->GetDecl(), method, proto);
   if (!FD) {
      return;
   }
   fMethod->Init(FD);
   Init(FD);
   if (offset) {
      offset = 0L;
   }
}

static llvm::Type *getLLVMTypeFromBuiltin(llvm::LLVMContext &Context,
                                          clang::ASTContext &ASTCtx,
                                          const clang::BuiltinType* PBT)
{
   llvm::Type *TY = 0;
   if (PBT->isInteger()) {
      uint64_t BTBits = ASTCtx.getTypeInfo(PBT).first;
      TY = llvm::IntegerType::get(Context, BTBits);
   } else switch (PBT->getKind()) {
      case clang::BuiltinType::Half:
      case clang::BuiltinType::ObjCId:
      case clang::BuiltinType::ObjCClass:
      case clang::BuiltinType::ObjCSel:
      case clang::BuiltinType::Dependent:
      case clang::BuiltinType::Overload:
      case clang::BuiltinType::BoundMember:
      case clang::BuiltinType::PseudoObject:
      case clang::BuiltinType::UnknownAny:
      case clang::BuiltinType::BuiltinFn:
      case clang::BuiltinType::ARCUnbridgedCast:
         Error("TClingCallFunc::getLLVMTypeFromBuiltin()",
               "Not implemented (kind %d)!", (int) PBT->getKind());
         break;
      case clang::BuiltinType::Void:
         TY = llvm::Type::getVoidTy(Context);
         break;
      case clang::BuiltinType::Float:
         TY = llvm::Type::getFloatTy(Context);
         break;
      case clang::BuiltinType::Double:
         TY = llvm::Type::getDoubleTy(Context);
         break;
      case clang::BuiltinType::LongDouble:
         TY = llvm::Type::getFP128Ty(Context);
         break;
      case clang::BuiltinType::NullPtr:
         TY = llvm::IntegerType::get(Context, CHAR_BIT);
         break;
      default:
         // everything else should be ints - what are we missing?
         Error("TClingCallFunc::getLLVMTypeFromBuiltin()",
               "Logic error (missing kind %d)!", (int)PBT->getKind());
         break;         
   }
   return TY;
}

static llvm::Type *getLLVMType(llvm::LLVMContext &Context,
                               clang::ASTContext &ASTCtx,
                               clang::QualType QT)
{
   llvm::Type *TY = 0;
   QT = QT.getCanonicalType();
   const clang::BuiltinType *BT = QT->getAs<clang::BuiltinType>();
   // Note: nullptr is a builtin type.
   if (QT->isPointerType() || QT->isReferenceType()) {
      clang::QualType PT = QT->getPointeeType();
      PT = PT.getCanonicalType();
      const clang::BuiltinType *PBT = llvm::dyn_cast<clang::BuiltinType> (PT);
      if (PBT) {
         // Pointer to something simple, preserve that.
         if (PT->isVoidType()) {
            // We have pointer to void, llvm cannot handle that,
            // force it to pointer to char.
            TY = llvm::PointerType::getUnqual(
                    llvm::IntegerType::get(Context, CHAR_BIT));
         }
         else {
            // We have pointer to clang builtin type, preserve that.
            llvm::Type *llvm_pt = getLLVMTypeFromBuiltin(Context, ASTCtx, PBT);
            TY = llvm::PointerType::getUnqual(llvm_pt);
         }
      }
      else {
         // Force it to pointer to char.
         TY = llvm::PointerType::getUnqual(
                 llvm::IntegerType::get(Context, CHAR_BIT));
      }
   }
   else if (QT->isRealFloatingType()) {
      TY = getLLVMTypeFromBuiltin(Context, ASTCtx, BT);
   }
   else if (QT->isIntegralOrEnumerationType()) {
      if (BT) {
         TY = getLLVMTypeFromBuiltin(Context, ASTCtx, BT);
      }
      else {
         const clang::EnumType *ET = QT->getAs<clang::EnumType>();
         clang::QualType IT = ET->getDecl()->getIntegerType();
         IT = IT.getCanonicalType();
         const clang::BuiltinType *IBT = llvm::dyn_cast<clang::BuiltinType>(IT);
         TY = getLLVMTypeFromBuiltin(Context, ASTCtx, IBT);
      }
   }
   else if (QT->isVoidType()) {
      TY = llvm::Type::getVoidTy(Context);
   }
   return TY;
}

void TClingCallFunc::Init(const clang::FunctionDecl *FD)
{
   fEEFunc = 0;
   fEEAddr = 0;
   bool isMemberFunc = true;
   const clang::CXXMethodDecl *MD = llvm::dyn_cast<clang::CXXMethodDecl>(FD);
   const clang::DeclContext *DC = FD->getDeclContext();
   if (DC->isTranslationUnit() || DC->isNamespace() || (MD && MD->isStatic())) {
      // Free function or static member function.
      isMemberFunc = false;
   }
   //
   //  Mangle the function name, if necessary.
   //
   const char *FuncName = 0;
   std::string MangledName;
   llvm::raw_string_ostream OS(MangledName);
   clang::ASTContext& ASTCtx = fInterp->getCI()->getASTContext();
   llvm::OwningPtr<clang::MangleContext> Mangle(ASTCtx.createMangleContext());
   if (!Mangle->shouldMangleDeclName(FD)) {
      clang::IdentifierInfo *II = FD->getIdentifier();
      FuncName = II->getNameStart();
   }
   else {
      if (const clang::CXXConstructorDecl *D =
               llvm::dyn_cast<clang::CXXConstructorDecl>(FD)) {
         //Ctor_Complete,          // Complete object ctor
         //Ctor_Base,              // Base object ctor
         //Ctor_CompleteAllocating // Complete object allocating ctor (unused)
         Mangle->mangleCXXCtor(D, clang::Ctor_Complete, OS);
      }
      else if (const clang::CXXDestructorDecl *D =
                  llvm::dyn_cast<clang::CXXDestructorDecl>(FD)) {
         //Dtor_Deleting, // Deleting dtor
         //Dtor_Complete, // Complete object dtor
         //Dtor_Base      // Base object dtor
         Mangle->mangleCXXDtor(D, clang::Dtor_Deleting, OS);
      }
      else {
         Mangle->mangleName(FD, OS);
      }
      OS.flush();
      FuncName = MangledName.c_str();
   }
   //
   //  Check the execution engine for the function.
   //
   llvm::ExecutionEngine *EE = fInterp->getExecutionEngine();
   fEEFunc = EE->FindFunctionNamed(FuncName);
   if (fEEFunc) {
      // Execution engine had it, get the mapping.
      fEEAddr = EE->getPointerToFunction(fEEFunc);
   }
   else {
      // Execution engine does not have it, check
      // the loaded shareable libraries.
      // NOTE: This issues a spurious error message if we look for an
      // unimplemented (but declared) function.
      void *FP = EE->getPointerToNamedFunction(FuncName,
                 /*AbortOnFailure=*/false);
      if (FP == unresolvedSymbol) {
         // We failed to find an implementation for the function, the 
         // interface requires the 'address' to be zero.
         fEEAddr = 0;
      } else if (FP) {
         // Create a llvm function we can use to call it with later.
         llvm::LLVMContext &Context = *fInterp->getLLVMContext();
         unsigned NumParams = FD->getNumParams();
         llvm::SmallVector<llvm::Type *, 8> Params;
         if (isMemberFunc) {
            // Force the invisible this pointer arg to pointer to char.
            Params.push_back(llvm::PointerType::getUnqual(
                                llvm::IntegerType::get(Context, CHAR_BIT)));
         }
         for (unsigned I = 0U; I < NumParams; ++I) {
            const clang::ParmVarDecl *PVD = FD->getParamDecl(I);
            clang::QualType QT = PVD->getType();
            Params.push_back(getLLVMType(Context, ASTCtx, QT));
         }
         llvm::Type *ReturnType = 0;
         if (llvm::isa<clang::CXXConstructorDecl>(FD)) {
            // Force the return type of a constructor to be long.
            ReturnType = llvm::IntegerType::get(Context, sizeof(long) *
                                                CHAR_BIT);
         }
         else {
            ReturnType = getLLVMType(Context, ASTCtx, FD->getResultType());
         }
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
         fEEAddr = FP;
      }
   }
}

llvm::GenericValue
TClingCallFunc::Invoke(const std::vector<llvm::GenericValue> &ArgValues) const
{
   // FIXME: We need to think about thunks for the this pointer adjustment,
   //        and the return pointer adjustment for covariant return types.
   //if (!IsValid()) {
   //   return;
   //}
   unsigned long num_given_args = static_cast<unsigned long>(ArgValues.size());
   const clang::FunctionDecl *fd = fMethod->GetMethodDecl();
   const clang::CXXMethodDecl *md = llvm::dyn_cast<clang::CXXMethodDecl>(fd);
   const clang::DeclContext *dc = fd->getDeclContext();
   bool isMemberFunction = true;
   if (dc->isTranslationUnit() || dc->isNamespace() || (md && md->isStatic())) {
      isMemberFunction = false;
   }
   unsigned num_params = fd->getNumParams();
   unsigned min_args = fd->getMinRequiredArguments();
   if (isMemberFunction) {
      // Adjust for the hidden this pointer first argument.
      ++num_params;
      ++min_args;
   }
   if (num_given_args < min_args) {
      // Not all required arguments given.
      Error("TClingCallFunc::Invoke()",
            "Not enough function arguments given (min: %u max:%u, given: %lu)",
            min_args, num_params, num_given_args);
      llvm::GenericValue bad_val;
      return bad_val;
   }
   else if (num_given_args > num_params) {
      Error("TClingCallFunc::Invoke()",
            "Too many function arguments given (min: %u max: %u, given: %lu)",
            min_args, num_params, num_given_args);
      llvm::GenericValue bad_val;
      return bad_val;
   }
   // We need a printing policy for handling default arguments.
   clang::ASTContext &context = fd->getASTContext();
   clang::PrintingPolicy policy(context.getPrintingPolicy());
   policy.SuppressTagKeyword = true;
   policy.SuppressUnwrittenScope = true;
   policy.SuppressInitializers = false;
   policy.AnonymousTagLocations = false;
   // This will be the arguments actually passed to the JIT function.
   std::vector<llvm::GenericValue> Args;
   // We are going to loop over the JIT function args.
   llvm::FunctionType *ft = fEEFunc->getFunctionType();
   for (unsigned i = 0U, e = ft->getNumParams(); i < e; ++i) {
      if (i < num_given_args) {
         // We have a user-provided argument value.
         //
         //  FIXME: We must convert the passed args, which are using
         //         the types required by the CINT interface, to the
         //         types the JIT expects.  This used to be done by
         //         the function wrappers in the CINT dictionary.
         //
         llvm::Type *ty = ft->getParamType(i);
         if (ty->getTypeID() == llvm::Type::PointerTyID) {
            // The cint interface passes these as integers, and we must
            // convert them to pointers because GenericValue stores
            // integer and pointer values in different data members.
            Args.push_back(llvm::PTOGV(reinterpret_cast<void *>(
                                          ArgValues[i].IntVal.getSExtValue())));
         }
         else {
            // FIXME: This is too naive, we may need to do some downcasting
            //        here if the actual type to pass to the JIT function
            //        is bool, char, short, int, enum, or float.
            Args.push_back(ArgValues[i]);
         }
      }
      else {
         // Use the default value from the decl.
         const clang::ParmVarDecl* pvd = 0;
         if (!isMemberFunction) {
            pvd = fd->getParamDecl(i);
         }
         else {
            // Compensate for the undeclared added this pointer value.
            pvd = fd->getParamDecl(i-1);
         }
         //assert(pvd->hasDefaultArg() && "No default for argument!");
         const clang::Expr* expr = pvd->getDefaultArg();
         static std::string buf;
         buf.clear();
         llvm::raw_string_ostream out(buf);
         expr->printPretty(out, /*Helper=*/0, policy, /*Indentation=*/0);
         cling::StoredValueRef valref;
         cling::Interpreter::CompilationResult cr =
            fInterp->evaluate(out.str(), &valref);
         if (cr == cling::Interpreter::kSuccess) {
            const cling::Value& val = valref.get();
            if (!val.type->isIntegralType(context) &&
                  !val.type->isRealFloatingType() &&
                  !val.type->isPointerType()) {
               // Invalid argument type.
               Error("TClingCallFunc::Invoke",
                     "Default for argument %u: %s", i, out.str().c_str());
               Error("TClingCallFunc::Invoke",
                     "is not of integral, floating, or pointer type!");
               llvm::GenericValue bad_val;
               return bad_val;
            }
            // FIXME: This is too naive, we may need to do some conversions.
            Args.push_back(val.value);
         }
         else {
            Error("TClingCallFunc::Invoke",
                  "Could not evaluate default for argument %u: %s",
                  i, out.str().c_str());
            llvm::GenericValue bad_val;
            return bad_val;
         }
      }
   }
   llvm::GenericValue return_val;
   return_val = fInterp->getExecutionEngine()->runFunction(fEEFunc, Args);
   if (ft->getReturnType()->getTypeID() == llvm::Type::PointerTyID) {
      // Note: The cint interface requires pointers to be
      //       returned as unsigned long.
      llvm::GenericValue converted_return_val;
      converted_return_val.IntVal =
         llvm::APInt(sizeof(unsigned long) * CHAR_BIT,
            reinterpret_cast<unsigned long>(GVTOP(return_val)));
      return converted_return_val;
   }
   return return_val;
}

