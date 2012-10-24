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
#include "Property.h"
#include "TClingProperty.h"
#include "TClingTypeInfo.h"
#include "TMetaUtils.h"

#include "cling/Interpreter/Interpreter.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/Type.h"
#include "clang/Basic/IdentifierTable.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/OwningPtr.h"

#include <string>

using namespace clang;

TClingMethodInfo::TClingMethodInfo(cling::Interpreter *interp,
                                   TClingClassInfo *ci)
   : fInterp(interp), fFirstTime(true), fContextIdx(0U), fTitle(""), 
     fSingleDecl(0)
{
   if (!ci || !ci->IsValid()) {
      return;
   }
   clang::DeclContext *dc =
      llvm::cast<clang::DeclContext>(const_cast<clang::Decl*>(ci->GetDecl()));
   dc->collectAllContexts(fContexts);
   fIter = dc->decls_begin();
   InternalNext();
   fFirstTime = true;
}

TClingMethodInfo::TClingMethodInfo(cling::Interpreter *interp,
                                   const clang::FunctionDecl *FD) 
   : fInterp(interp), fFirstTime(true), fContextIdx(0U), fTitle(""), 
     fSingleDecl(FD)
{

}


const clang::FunctionDecl *TClingMethodInfo::GetMethodDecl() const
{
   if (fSingleDecl)
      return fSingleDecl;

   if (!IsValid()) {
      return 0;
   }
   return llvm::dyn_cast<clang::FunctionDecl>(*fIter);
}

void TClingMethodInfo::CreateSignature(TString &signature) const
{
   signature = "(";
   if (!IsValid()) {
      signature += ")";
      return;
   }
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
   if (!decl) {
      return;
   }
   clang::DeclContext *DC =
      const_cast<clang::DeclContext *>(decl->getDeclContext());
   DC = DC->getPrimaryContext();
   DC->collectAllContexts(fContexts);
   fIter = DC->decls_begin();
   while (InternalNext()) {
      if (*fIter == decl) {
         fFirstTime = true;
         break;
      }
   }
}

void *TClingMethodInfo::InterfaceMethod() const
{
   if (!IsValid()) {
      return 0;
   }
   TClingCallFunc cf(fInterp);
   cf.SetFunc(this);
   return cf.InterfaceMethod();
}

bool TClingMethodInfo::IsValid() const
{
   return *fIter;
}

int TClingMethodInfo::NArg() const
{
   if (!IsValid()) {
      return -1;
   }
   const clang::FunctionDecl *fd = llvm::cast<clang::FunctionDecl>(*fIter);
   unsigned num_params = fd->getNumParams();
   // Truncate cast to fit cint interface.
   return static_cast<int>(num_params);
}

int TClingMethodInfo::NDefaultArg() const
{
   if (!IsValid()) {
      return -1;
   }
   const clang::FunctionDecl *fd = llvm::cast<clang::FunctionDecl>(*fIter);
   unsigned num_params = fd->getNumParams();
   unsigned min_args = fd->getMinRequiredArguments();
   unsigned defaulted_params = num_params - min_args;
   // Truncate cast to fit cint interface.
   return static_cast<int>(defaulted_params);
}

int TClingMethodInfo::InternalNext()
{

   assert(!fSingleDecl && "This is not an iterator!");

   if (!*fIter) {
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
         ++fIter;
      }
      // Fix it if we have gone past the end of the current decl context.
      while (!*fIter) {
         ++fContextIdx;
         if (fContextIdx >= fContexts.size()) {
            // Iterator is now invalid.
            return 0;
         }
         clang::DeclContext *dc = fContexts[fContextIdx];
         fIter = dc->decls_begin();
         if (*fIter) {
            // Good, a non-empty context.
            break;
         }
      }
      // Return if this decl is a function or method.
      if (llvm::isa<clang::FunctionDecl>(*fIter)) {
         // Iterator is now valid.
         return 1;
      }
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
   property |= G__BIT_ISCOMPILED;
   const clang::FunctionDecl *fd =
      llvm::dyn_cast<clang::FunctionDecl>(*fIter);
   switch (fd->getAccess()) {
      case clang::AS_public:
         property |= G__BIT_ISPUBLIC;
         break;
      case clang::AS_protected:
         property |= G__BIT_ISPROTECTED;
         break;
      case clang::AS_private:
         property |= G__BIT_ISPRIVATE;
         break;
      case clang::AS_none:
         // IMPOSSIBLE
         break;
      default:
         // IMPOSSIBLE
         break;
   }
   if (fd->getStorageClass() == clang::SC_Static) {
      property |= G__BIT_ISSTATIC;
   }
   clang::QualType qt = fd->getResultType().getCanonicalType();
   if (qt.isConstQualified()) {
      property |= G__BIT_ISCONSTANT;
   }
   while (1) {
      if (qt->isArrayType()) {
         qt = llvm::cast<clang::ArrayType>(qt)->getElementType();
         continue;
      }
      else if (qt->isReferenceType()) {
         property |= G__BIT_ISREFERENCE;
         qt = llvm::cast<clang::ReferenceType>(qt)->getPointeeType();
         continue;
      }
      else if (qt->isPointerType()) {
         property |= G__BIT_ISPOINTER;
         if (qt.isConstQualified()) {
            property |= G__BIT_ISPCONSTANT;
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
      property |= G__BIT_ISCONSTANT;
   }
   if (const clang::CXXMethodDecl *md =
            llvm::dyn_cast<clang::CXXMethodDecl>(fd)) {
      if (md->getTypeQualifiers() & clang::Qualifiers::Const) {
         property |= G__BIT_ISCONSTANT | G__BIT_ISMETHCONSTANT;
      }
      if (md->isVirtual()) {
         property |= G__BIT_ISVIRTUAL;
      }
      if (md->isPure()) {
         property |= G__BIT_ISPUREVIRTUAL;
      }
      if (const clang::CXXConstructorDecl *cd =
               llvm::dyn_cast<clang::CXXConstructorDecl>(md)) {
         if (cd->isExplicit()) {
            property |= G__BIT_ISEXPLICIT;
         }
      }
      else if (const clang::CXXConversionDecl *cd =
                  llvm::dyn_cast<clang::CXXConversionDecl>(md)) {
         if (cd->isExplicit()) {
            property |= G__BIT_ISEXPLICIT;
         }
      }
   }
   return property;
}

TClingTypeInfo *TClingMethodInfo::Type() const
{
   static TClingTypeInfo ti(fInterp);
   ti.Init(clang::QualType());
   if (!IsValid()) {
      return &ti;
   }
   clang::QualType qt = llvm::cast<clang::FunctionDecl>(*fIter)->
                        getResultType();
   ti.Init(qt);
   return &ti;
}

const char *TClingMethodInfo::GetMangledName() const
{
   if (!IsValid()) {
      return 0;
   }
   const char *fname = 0;
   static std::string mangled_name;
   mangled_name.clear();
   llvm::raw_string_ostream os(mangled_name);
   llvm::OwningPtr<clang::MangleContext> mangle(fIter->getASTContext().
         createMangleContext());
   const clang::NamedDecl *nd = llvm::dyn_cast<clang::NamedDecl>(*fIter);
   if (!nd) {
      return 0;
   }
   if (!mangle->shouldMangleDeclName(nd)) {
      clang::IdentifierInfo *ii = nd->getIdentifier();
      fname = ii->getNameStart();
   }
   else {
      if (const clang::CXXConstructorDecl *d =
               llvm::dyn_cast<clang::CXXConstructorDecl>(nd)) {
         //Ctor_Complete,          // Complete object ctor
         //Ctor_Base,              // Base object ctor
         //Ctor_CompleteAllocating // Complete object allocating ctor (unused)
         mangle->mangleCXXCtor(d, clang::Ctor_Complete, os);
      }
      else if (const clang::CXXDestructorDecl *d =
                  llvm::dyn_cast<clang::CXXDestructorDecl>(nd)) {
         //Dtor_Deleting, // Deleting dtor
         //Dtor_Complete, // Complete object dtor
         //Dtor_Base      // Base object dtor
         mangle->mangleCXXDtor(d, clang::Dtor_Deleting, os);
      }
      else {
         mangle->mangleName(nd, os);
      }
      os.flush();
      fname = mangled_name.c_str();
   }
   return fname;
}

const char *TClingMethodInfo::GetPrototype() const
{
   if (!IsValid()) {
      return 0;
   }
   static std::string buf;
   buf.clear();
   buf += Type()->Name();
   buf += ' ';
   std::string name;
   clang::PrintingPolicy policy(fIter->getASTContext().getPrintingPolicy());
   const clang::NamedDecl *nd = llvm::cast<clang::NamedDecl>(*fIter);
   nd->getNameForDiagnostic(name, policy, /*Qualified=*/true);
   buf += name;
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
   return buf.c_str();
}

const char *TClingMethodInfo::Name() const
{
   if (!IsValid()) {
      return 0;
   }
   static std::string buf;
   buf.clear();
   clang::PrintingPolicy policy(fIter->getASTContext().getPrintingPolicy());
   llvm::dyn_cast<clang::NamedDecl>(*fIter)->
   getNameForDiagnostic(buf, policy, /*Qualified=*/false);
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
   if (const FunctionDecl *FD = llvm::dyn_cast<FunctionDecl>(GetMethodDecl())) {
      if ( (FD = ROOT::TMetaUtils::GetAnnotatedRedeclarable(FD)) ) {
         if (AnnotateAttr *A = FD->getAttr<AnnotateAttr>()) {
            fTitle = A->getAnnotation().str();
            return fTitle.c_str();
         }
      }
   }
   // Try to get the comment from the header file if present
   fTitle = ROOT::TMetaUtils::GetComment(*GetMethodDecl()).str();

   return fTitle.c_str();
}

