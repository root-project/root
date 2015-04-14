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
// TClingMethodInfo                                                     //
//                                                                      //
// Emulation of the CINT MethodInfo class.                              //
//                                                                      //
// The CINT C++ interpreter provides an interface to metadata about     //
// a function through the MethodInfo class.  This class provides the    //
// same functionality, using an interface as close as possible to       //
// MethodInfo but the typedef metadata comes from the Clang C++         //
// compiler, not CINT.                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClingMethodInfo.h"

#include "TClingCallFunc.h"
#include "TClingClassInfo.h"
#include "TClingMethodArgInfo.h"
#include "TDictionary.h"
#include "TClingTypeInfo.h"
#include "TError.h"
#include "TMetaUtils.h"
#include "TCling.h"
#include "ThreadLocalStorage.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/Type.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Sema/Sema.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace clang;

class TClingMethodInfo::SpecIterator
{
public:
   typedef clang::FunctionTemplateDecl::spec_iterator Iterator;

   SpecIterator(Iterator begin, Iterator end) : fIter(begin), fEnd(end) {}
   explicit SpecIterator(clang::FunctionTemplateDecl *decl) : fIter(decl->spec_begin()), fEnd(decl->spec_end()) {}

   FunctionDecl *operator* () const { return *fIter; }
   FunctionDecl *operator-> () const { return *fIter; }
   SpecIterator & operator++ () { ++fIter; return *this; }
   SpecIterator   operator++ (int) {
      SpecIterator tmp(fIter,fEnd);
      ++(*this);
      return tmp;
   }
   bool operator!() { return fIter == fEnd; }
   operator bool() { return fIter != fEnd; }

private:

   Iterator fIter;
   Iterator fEnd;
};

TClingMethodInfo::TClingMethodInfo(const TClingMethodInfo &rhs) :
   fInterp(rhs.fInterp),
   fContexts(rhs.fContexts),
   fFirstTime(rhs.fFirstTime),
   fContextIdx(rhs.fContextIdx),
   fIter(rhs.fIter),
   fTitle(rhs.fTitle),
   fTemplateSpecIter(nullptr),
   fSingleDecl(rhs.fSingleDecl)
{
   if (rhs.fTemplateSpecIter) {
      // The SpecIterator query the decl.
      R__LOCKGUARD(gInterpreterMutex);
      fTemplateSpecIter = new SpecIterator(*rhs.fTemplateSpecIter);
   }
}


TClingMethodInfo::TClingMethodInfo(cling::Interpreter *interp,
                                   TClingClassInfo *ci)
   : fInterp(interp), fFirstTime(true), fContextIdx(0U), fTitle(""),
     fTemplateSpecIter(0), fSingleDecl(0)
{
   R__LOCKGUARD(gInterpreterMutex);

   if (!ci || !ci->IsValid()) {
      return;
   }
   clang::CXXRecordDecl *cxxdecl = llvm::dyn_cast<clang::CXXRecordDecl>(const_cast<clang::Decl*>(ci->GetDecl()));
   if (cxxdecl) {
      // Make sure we have an entry for all the implicit function.

      // Could trigger deserialization of decls.
      cling::Interpreter::PushTransactionRAII RAII(interp);

      fInterp->getSema().ForceDeclarationOfImplicitMembers(cxxdecl);
   }
   clang::DeclContext *dc =
      llvm::cast<clang::DeclContext>(const_cast<clang::Decl*>(ci->GetDecl()));
   dc->collectAllContexts(fContexts);
   // Could trigger deserialization of decls.
   cling::Interpreter::PushTransactionRAII RAII(interp);
   fIter = dc->decls_begin();
   InternalNext();
   fFirstTime = true;
}

TClingMethodInfo::TClingMethodInfo(cling::Interpreter *interp,
                                   const clang::FunctionDecl *FD)
   : fInterp(interp), fFirstTime(true), fContextIdx(0U), fTitle(""),
     fTemplateSpecIter(0), fSingleDecl(FD)
{

}

TClingMethodInfo::~TClingMethodInfo()
{
   delete fTemplateSpecIter;
}

TDictionary::DeclId_t TClingMethodInfo::GetDeclId() const
{
   if (!IsValid()) {
      return TDictionary::DeclId_t();
   }
   return (const clang::Decl*)(GetMethodDecl()->getCanonicalDecl());
}

const clang::FunctionDecl *TClingMethodInfo::GetMethodDecl() const
{
   if (!IsValid()) {
      return 0;
   }

   if (fSingleDecl)
      return fSingleDecl;

   if (fTemplateSpecIter)
      return *(*fTemplateSpecIter);

   return llvm::dyn_cast<clang::FunctionDecl>(*fIter);
}

void TClingMethodInfo::CreateSignature(TString &signature) const
{
   signature = "(";
   if (!IsValid()) {
      signature += ")";
      return;
   }

   R__LOCKGUARD(gInterpreterMutex);
   TClingMethodArgInfo arg(fInterp, this);

   int idx = 0;
   while (arg.Next()) {
      if (idx) {
         signature += ", ";
      }
      signature += arg.Type()->Name();
      if (arg.Name() && strlen(arg.Name())) {
         signature += " ";
         signature += arg.Name();
      }
      if (arg.DefaultValue()) {
         signature += " = ";
         signature += arg.DefaultValue();
      }
      ++idx;
   }
   signature += ")";
}

void TClingMethodInfo::Init(const clang::FunctionDecl *decl)
{
   fContexts.clear();
   fFirstTime = true;
   fContextIdx = 0U;
   fIter = clang::DeclContext::decl_iterator();
   delete fTemplateSpecIter;
   fTemplateSpecIter = 0;
   fSingleDecl = decl;
}

void *TClingMethodInfo::InterfaceMethod(const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt) const
{
   if (!IsValid()) {
      return 0;
   }
   R__LOCKGUARD(gInterpreterMutex);
   TClingCallFunc cf(fInterp,normCtxt);
   cf.SetFunc(this);
   return cf.InterfaceMethod();
}

bool TClingMethodInfo::IsValid() const
{
   if (fSingleDecl) return fSingleDecl;
   else if (fTemplateSpecIter) {
      // Could trigger deserialization of decls.
      R__LOCKGUARD(gInterpreterMutex);
      cling::Interpreter::PushTransactionRAII RAII(fInterp);
      return *(*fTemplateSpecIter);
   }
   return *fIter;
}

int TClingMethodInfo::NArg() const
{
   if (!IsValid()) {
      return -1;
   }
   const clang::FunctionDecl *fd = GetMethodDecl();
   unsigned num_params = fd->getNumParams();
   // Truncate cast to fit cint interface.
   return static_cast<int>(num_params);
}

int TClingMethodInfo::NDefaultArg() const
{
   if (!IsValid()) {
      return -1;
   }
   const clang::FunctionDecl *fd = GetMethodDecl();
   unsigned num_params = fd->getNumParams();
   unsigned min_args = fd->getMinRequiredArguments();
   unsigned defaulted_params = num_params - min_args;
   // Truncate cast to fit cint interface.
   return static_cast<int>(defaulted_params);
}

int TClingMethodInfo::InternalNext()
{

   assert(!fSingleDecl && "This is not an iterator!");

   if (!fFirstTime && !*fIter) {
      // Iterator is already invalid.
      return 0;
   }
   while (true) {
      // Advance to the next decl.
      if (fFirstTime) {
         // The cint semantics are weird.
         fFirstTime = false;
      }
      else {
         if (fTemplateSpecIter) {
            ++(*fTemplateSpecIter);
            if ( !(*fTemplateSpecIter) ) {
               // We reached the end of the template specialization.
               delete fTemplateSpecIter;
               fTemplateSpecIter = 0;
               ++fIter;
            } else {
               return 1;
            }
         } else {
            ++fIter;
         }
      }
      // Fix it if we have gone past the end of the current decl context.
      while (!*fIter) {
         ++fContextIdx;
         if (fContextIdx >= fContexts.size()) {
            // Iterator is now invalid.
            return 0;
         }
         clang::DeclContext *dc = fContexts[fContextIdx];
         // Could trigger deserialization of decls.

         cling::Interpreter::PushTransactionRAII RAII(fInterp);
         fIter = dc->decls_begin();
         if (*fIter) {
            // Good, a non-empty context.
            break;
         }
      }

      clang::FunctionTemplateDecl *templateDecl =
         llvm::dyn_cast<clang::FunctionTemplateDecl>(*fIter);
      if ( templateDecl ) {
         // SpecIterator calls clang::FunctionTemplateDecl::spec_begin
         // which calls clang::FunctionTemplateDecl::LoadLazySpecializations
         cling::Interpreter::PushTransactionRAII RAII(fInterp);
         SpecIterator subiter(templateDecl);
         if (subiter) {
            delete fTemplateSpecIter;
            fTemplateSpecIter = new SpecIterator(templateDecl);
            return 1;
         }
      }

      // Return if this decl is a function or method.
      if (llvm::isa<clang::FunctionDecl>(*fIter)) {
         // Iterator is now valid.
         return 1;
      }
//      if (clang::FunctionDecl *fdecl = llvm::dyn_cast<clang::FunctionDecl>(*fIter)) {
//         if (fdecl->getAccess() == clang::AS_public || fdecl->getAccess() == clang::AS_none) {
//            // Iterator is now valid.
//            return 1;
//         }
//      }
   }
}

int TClingMethodInfo::Next()
{
   return InternalNext();
}

long TClingMethodInfo::Property() const
{
   if (!IsValid()) {
      return 0L;
   }
   long property = 0L;
   property |= kIsCompiled;
   const clang::FunctionDecl *fd = GetMethodDecl();
   switch (fd->getAccess()) {
      case clang::AS_public:
         property |= kIsPublic;
         break;
      case clang::AS_protected:
         property |= kIsProtected;
         break;
      case clang::AS_private:
         property |= kIsPrivate;
         break;
      case clang::AS_none:
         if (fd->getDeclContext()->isNamespace())
            property |= kIsPublic;
         break;
      default:
         // IMPOSSIBLE
         break;
   }
   if (fd->getStorageClass() == clang::SC_Static) {
      property |= kIsStatic;
   }
   clang::QualType qt = fd->getReturnType().getCanonicalType();
   if (qt.isConstQualified()) {
      property |= kIsConstant;
   }
   while (1) {
      if (qt->isArrayType()) {
         qt = llvm::cast<clang::ArrayType>(qt)->getElementType();
         continue;
      }
      else if (qt->isReferenceType()) {
         property |= kIsReference;
         qt = llvm::cast<clang::ReferenceType>(qt)->getPointeeType();
         continue;
      }
      else if (qt->isPointerType()) {
         property |= kIsPointer;
         if (qt.isConstQualified()) {
            property |= kIsConstPointer;
         }
         qt = llvm::cast<clang::PointerType>(qt)->getPointeeType();
         continue;
      }
      else if (qt->isMemberPointerType()) {
         qt = llvm::cast<clang::MemberPointerType>(qt)->getPointeeType();
         continue;
      }
      break;
   }
   if (qt.isConstQualified()) {
      property |= kIsConstant;
   }
   if (const clang::CXXMethodDecl *md =
            llvm::dyn_cast<clang::CXXMethodDecl>(fd)) {
      if (md->getTypeQualifiers() & clang::Qualifiers::Const) {
         property |= kIsConstant | kIsConstMethod;
      }
      if (md->isVirtual()) {
         property |= kIsVirtual;
      }
      if (md->isPure()) {
         property |= kIsPureVirtual;
      }
      if (const clang::CXXConstructorDecl *cd =
               llvm::dyn_cast<clang::CXXConstructorDecl>(md)) {
         if (cd->isExplicit()) {
            property |= kIsExplicit;
         }
      }
      else if (const clang::CXXConversionDecl *cd =
                  llvm::dyn_cast<clang::CXXConversionDecl>(md)) {
         if (cd->isExplicit()) {
            property |= kIsExplicit;
         }
      }
   }
   return property;
}

long TClingMethodInfo::ExtraProperty() const
{
   // Return the property not already defined in Property
   // See TDictionary's EFunctionProperty
   if (!IsValid()) {
      return 0L;
   }
   long property = 0;
   const clang::FunctionDecl *fd = GetMethodDecl();
   if (fd->isOverloadedOperator()) {
      property |= kIsOperator;
   }
   else if (llvm::isa<clang::CXXConversionDecl>(fd)) {
      property |= kIsConversion;
   } else if (llvm::isa<clang::CXXConstructorDecl>(fd)) {
      property |= kIsConstructor;
   } else if (llvm::isa<clang::CXXDestructorDecl>(fd)) {
      property |= kIsDestructor;
   }
   return property;
}

TClingTypeInfo *TClingMethodInfo::Type() const
{
   TTHREAD_TLS_DECL_ARG( TClingTypeInfo, ti, fInterp );
   if (!IsValid()) {
      ti.Init(clang::QualType());
      return &ti;
   }
   if (llvm::isa<clang::CXXConstructorDecl>(GetMethodDecl())) {
      // CINT claims that constructors return the class object.
      const clang::TypeDecl* ctorClass = llvm::dyn_cast_or_null<clang::TypeDecl>
         (GetMethodDecl()->getDeclContext());
      if (!ctorClass) {
         Error("TClingMethodInfo::Type", "Cannot find DeclContext for constructor!");
      } else {
         clang::QualType qt(ctorClass->getTypeForDecl(), 0);
         ti.Init(qt);
      }
   } else {
      clang::QualType qt = GetMethodDecl()->getReturnType();
      ti.Init(qt);
   }
   return &ti;
}

std::string TClingMethodInfo::GetMangledName() const
{
   if (!IsValid()) {
      return "";
   }
   std::string mangled_name;
   mangled_name.clear();
   const FunctionDecl* D = GetMethodDecl();

   R__LOCKGUARD(gInterpreterMutex);
   GlobalDecl GD;
   if (const CXXConstructorDecl* Ctor = dyn_cast<CXXConstructorDecl>(D))
     GD = GlobalDecl(Ctor, Ctor_Complete);
   else if (const CXXDestructorDecl* Dtor = dyn_cast<CXXDestructorDecl>(D))
     GD = GlobalDecl(Dtor, Dtor_Deleting);
   else
     GD = GlobalDecl(D);

   cling::utils::Analyze::maybeMangleDeclName(GD, mangled_name);
   return mangled_name;
}

const char *TClingMethodInfo::GetPrototype(const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt) const
{
   if (!IsValid()) {
      return 0;
   }
   TTHREAD_TLS_DECL( std::string, buf );
   buf.clear();
   buf += Type()->Name();
   buf += ' ';
   if (const clang::TypeDecl *td = llvm::dyn_cast<clang::TypeDecl>(GetMethodDecl()->getDeclContext())) {
      std::string name;
      clang::QualType qualType(td->getTypeForDecl(),0);
      ROOT::TMetaUtils::GetFullyQualifiedTypeName(name,qualType,*fInterp);
      buf += name;
      buf += "::";
   } else if (const clang::NamedDecl *nd = llvm::dyn_cast<clang::NamedDecl>(GetMethodDecl()->getDeclContext())) {
      std::string name;
      clang::PrintingPolicy policy(GetMethodDecl()->getASTContext().getPrintingPolicy());
      llvm::raw_string_ostream stream(name);
      nd->getNameForDiagnostic(stream, policy, /*Qualified=*/true);
      stream.flush();
      buf += name;
      buf += "::";
   }
   buf += Name(normCtxt);
   buf += '(';
   TClingMethodArgInfo arg(fInterp, this);
   int idx = 0;
   while (arg.Next()) {
      if (idx) {
         buf += ", ";
      }
      buf += arg.Type()->Name();
      if (arg.Name() && strlen(arg.Name())) {
         buf += ' ';
         buf += arg.Name();
      }
      if (arg.DefaultValue()) {
         buf += " = ";
         buf += arg.DefaultValue();
      }
      ++idx;
   }
   buf += ')';
   if (const clang::CXXMethodDecl *md =
       llvm::dyn_cast<clang::CXXMethodDecl>( GetMethodDecl())) {
      if (md->getTypeQualifiers() & clang::Qualifiers::Const) {
         buf += " const";
      }
   }
   return buf.c_str();
}

const char *TClingMethodInfo::Name(const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt) const
{
   if (!IsValid()) {
      return 0;
   }
   TTHREAD_TLS_DECL( std::string, buf );
   ((TCling*)gCling)->GetFunctionName(GetMethodDecl(),buf);
   return buf.c_str();
}

const char *TClingMethodInfo::TypeName() const
{
   if (!IsValid()) {
      // FIXME: Cint does not check!
      return 0;
   }
   return Type()->Name();
}

const char *TClingMethodInfo::Title()
{
   if (!IsValid()) {
      return 0;
   }

   //NOTE: We can't use it as a cache due to the "thoughtful" self iterator
   //if (fTitle.size())
   //   return fTitle.c_str();

   // Try to get the comment either from the annotation or the header file if present

   // Iterate over the redeclarations, we can have muliple definitions in the
   // redecl chain (came from merging of pcms).
   const FunctionDecl *FD = GetMethodDecl();

   R__LOCKGUARD(gInterpreterMutex);

   // Could trigger deserialization of decls.
   cling::Interpreter::PushTransactionRAII RAII(fInterp);
   if (const FunctionDecl *AnnotFD
       = ROOT::TMetaUtils::GetAnnotatedRedeclarable(FD)) {
      if (AnnotateAttr *A = AnnotFD->getAttr<AnnotateAttr>()) {
         fTitle = A->getAnnotation().str();
         return fTitle.c_str();
      }
   }
   if (!FD->isFromASTFile()) {
      // Try to get the comment from the header file if present
      // but not for decls from AST file, where rootcling would have
      // created an annotation
      fTitle = ROOT::TMetaUtils::GetComment(*FD).str();
   }

   return fTitle.c_str();
}

