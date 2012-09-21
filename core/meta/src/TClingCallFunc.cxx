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
#include "cling/Interpreter/Value.h"

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
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Type.h"

#include <string>
#include <vector>

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
                            std::vector<cling::Value> &EvaluatedArgs)
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
      cling::Value val;
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
   std::vector<cling::Value> Args;
   evaluateArgList(fInterp, params, Args);
   clang::ASTContext &Context = fInterp->getCI()->getASTContext();
   for (unsigned I = 0U, E = Args.size(); I < E; ++I) {
      cling::Value val = Args[I];
      if (!val.type->isIntegralType(Context) &&
            !val.type->isRealFloatingType() && !val.type->isPointerType()) {
         // Invalid argument type.
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
   fArgs.clear();
   std::vector<cling::Value> Args;
   evaluateArgList(fInterp, params, Args);
   clang::ASTContext &Context = fInterp->getCI()->getASTContext();
   for (unsigned I = 0U, E = Args.size(); I < E; ++I) {
      cling::Value val = Args[I];
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

static llvm::Type *getLLVMTypeFromBuiltinKind(llvm::LLVMContext &Context,
      clang::BuiltinType::Kind BuiltinKind)
{
   llvm::Type *TY = 0;
   switch (BuiltinKind) {
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
      case clang::BuiltinType::Char16:
      case clang::BuiltinType::Char32:
         // We do not use these, make gcc be quiet.
         break;
      case clang::BuiltinType::Void:
         TY = llvm::Type::getVoidTy(Context);
         break;
      case clang::BuiltinType::Bool:
         TY = llvm::IntegerType::get(Context, sizeof(bool) * CHAR_BIT);
         break;
      case clang::BuiltinType::Char_U:
         TY = llvm::IntegerType::get(Context, sizeof(unsigned char) * CHAR_BIT);
         break;
      case clang::BuiltinType::UChar:
         TY = llvm::IntegerType::get(Context, sizeof(unsigned char) * CHAR_BIT);
         break;
      case clang::BuiltinType::WChar_U:
         TY = llvm::IntegerType::get(Context, sizeof(unsigned wchar_t) *
                                     CHAR_BIT);
         break;
#if 0
      case clang::BuiltinType::Char16:
         TY = llvm::IntegerType::get(Context, sizeof(char16_t) * CHAR_BIT);
         break;
      case clang::BuiltinType::Char32:
         TY = llvm::IntegerType::get(Context, sizeof(char32_t) * CHAR_BIT);
         break;
#endif // 0
      case clang::BuiltinType::UShort:
         TY = llvm::IntegerType::get(Context, sizeof(unsigned short) *
                                     CHAR_BIT);
         break;
      case clang::BuiltinType::UInt:
         TY = llvm::IntegerType::get(Context, sizeof(unsigned int) * CHAR_BIT);
         break;
      case clang::BuiltinType::ULong:
         TY = llvm::IntegerType::get(Context, sizeof(unsigned long) * CHAR_BIT);
         break;
      case clang::BuiltinType::ULongLong:
         TY = llvm::IntegerType::get(Context, sizeof(unsigned long long) *
                                     CHAR_BIT);
         break;
      case clang::BuiltinType::UInt128:
         TY = llvm::IntegerType::get(Context, sizeof(__uint128_t) * CHAR_BIT);
         break;
      case clang::BuiltinType::Char_S:
         TY = llvm::IntegerType::get(Context, sizeof(signed char) * CHAR_BIT);
         break;
      case clang::BuiltinType::SChar:
         TY = llvm::IntegerType::get(Context, sizeof(signed char) * CHAR_BIT);
         break;
      case clang::BuiltinType::WChar_S:
         TY = llvm::IntegerType::get(Context, sizeof(signed wchar_t) *
                                     CHAR_BIT);
         break;
      case clang::BuiltinType::Short:
         TY = llvm::IntegerType::get(Context, sizeof(signed short) * CHAR_BIT);
         break;
      case clang::BuiltinType::Int:
         TY = llvm::IntegerType::get(Context, sizeof(signed int) * CHAR_BIT);
         break;
      case clang::BuiltinType::Long:
         TY = llvm::IntegerType::get(Context, sizeof(signed long) * CHAR_BIT);
         break;
      case clang::BuiltinType::LongLong:
         TY = llvm::IntegerType::get(Context, sizeof(signed long long) *
                                     CHAR_BIT);
         break;
      case clang::BuiltinType::Int128:
         TY = llvm::IntegerType::get(Context, sizeof(__int128_t) * CHAR_BIT);
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
         TY = llvm::PointerType::getUnqual(llvm::IntegerType::get(Context,
                                           CHAR_BIT));
         break;
   }
   return TY;
}

static llvm::Type *getLLVMType(llvm::LLVMContext &Context, clang::QualType QT)
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
            clang::BuiltinType::Kind kind = PBT->getKind();
            llvm::Type *llvm_pt = getLLVMTypeFromBuiltinKind(Context, kind);
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
      clang::BuiltinType::Kind kind = BT->getKind();
      TY = getLLVMTypeFromBuiltinKind(Context, kind);
   }
   else if (QT->isIntegralOrEnumerationType()) {
      if (BT) {
         clang::BuiltinType::Kind kind = BT->getKind();
         TY = getLLVMTypeFromBuiltinKind(Context, kind);
      }
      else {
         const clang::EnumType *ET = QT->getAs<clang::EnumType>();
         clang::QualType IT = ET->getDecl()->getIntegerType();
         IT = IT.getCanonicalType();
         const clang::BuiltinType *IBT = llvm::dyn_cast<clang::BuiltinType>(IT);
         clang::BuiltinType::Kind kind = IBT->getKind();
         TY = getLLVMTypeFromBuiltinKind(Context, kind);
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
   llvm::OwningPtr<clang::MangleContext> Mangle(fInterp->getCI()->
         getASTContext().createMangleContext());
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
      void *FP = EE->getPointerToNamedFunction(FuncName,
                 /*AbortOnFailure=*/false);
      //if (FP == unresolvedSymbol) {
      //   // The ExecutionContext will refuse to do anything after this,
      //   // so we must force it back to normal.
      //   fInterp->resetUnresolved();
      //}
      //else 
      if (FP) {
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
            Params.push_back(getLLVMType(Context, QT));
         }
         llvm::Type *ReturnType = 0;
         if (llvm::isa<clang::CXXConstructorDecl>(FD)) {
            // Force the return type of a constructor to be long.
            ReturnType = llvm::IntegerType::get(Context, sizeof(long) *
                                                CHAR_BIT);
         }
         else {
            ReturnType = getLLVMType(Context, FD->getResultType());
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

llvm::GenericValue TClingCallFunc::Invoke(const std::vector<llvm::GenericValue> &ArgValues) const
{
   // FIXME: We need to think about thunks for the this pointer adjustment,
   //        and the return pointer adjustment.
   //if (!IsValid()) {
   //   return;
   //}
   std::vector<llvm::GenericValue> Args;
   llvm::FunctionType *FT = fEEFunc->getFunctionType();
   for (unsigned I = 0U, E = FT->getNumParams(); I < E; ++I) {
      llvm::Type *TY = FT->getParamType(I);
      if (TY->getTypeID() == llvm::Type::PointerTyID) {
         // The cint interface passes these as integers, and we must
         // convert them to pointers because GenericValue stores
         // integer and pointer values in different data members.
         Args.push_back(llvm::PTOGV(reinterpret_cast<void *>(
                                       ArgValues[I].IntVal.getSExtValue())));
      }
      else {
         Args.push_back(ArgValues[I]);
      }
   }
   llvm::GenericValue val;
   val = fInterp->getExecutionEngine()->runFunction(fEEFunc, Args);
   if (FT->getReturnType()->getTypeID() == llvm::Type::PointerTyID) {
      //The cint interface requires pointers to be return as unsigned long.
      llvm::GenericValue gv;
      gv.IntVal = llvm::APInt(sizeof(unsigned long) * CHAR_BIT,
                              reinterpret_cast<unsigned long>(GVTOP(val)));
      return gv;
   }
   return val;
}

