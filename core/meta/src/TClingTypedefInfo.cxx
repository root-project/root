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
// TClingTypedefInfo                                                    //
//                                                                      //
// Emulation of the CINT TypedefInfo class.                             //
//                                                                      //
// The CINT C++ interpreter provides an interface to metadata about     //
// a typedef through the TypedefInfo class.  This class provides the    //
// same functionality, using an interface as close as possible to       //
// TypedefInfo but the typedef metadata comes from the Clang C++        //
// compiler, not CINT.                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClingTypedefInfo.h"

#include "TDictionary.h"
#include "TError.h"
#include "TMetaUtils.h"
#include "Rtypes.h" // for gDebug
#include "ThreadLocalStorage.h"

#include "cling/Interpreter/LookupHelper.h"
#include "cling/Utils/AST.h"
#include "clang/AST/Attr.h"

using namespace clang;

//______________________________________________________________________________
TClingTypedefInfo::TClingTypedefInfo(cling::Interpreter *interp,
                                     const char *name)
   : fInterp(interp), fFirstTime(true), fDescend(false), fDecl(0), fTitle("")
{
   // Lookup named typedef and initialize the iterator to point to it.
   // Yields a non-iterable TClingTypedefInfo (fIter is invalid).
   Init(name);
}

TClingTypedefInfo::TClingTypedefInfo(cling::Interpreter *interp,
                                     const clang::TypedefNameDecl *TdefD)
   : fInterp(interp), fFirstTime(true), fDescend(false), fDecl(TdefD),
     fTitle("")
{
   // Initialize with a clang::TypedefDecl.
   // fIter is invalid; cannot call Next().
}

//______________________________________________________________________________
const clang::Decl *TClingTypedefInfo::GetDecl() const
{
   // Get the current typedef declaration.
   return fDecl;
}

//______________________________________________________________________________
void TClingTypedefInfo::Init(const char *name)
{
   // Lookup named typedef and reset the iterator to point to it.

   fDecl = 0;

   // Reset the iterator to invalid.
   fFirstTime = true;
   fDescend = false;
   fIter = clang::DeclContext::decl_iterator();
   fIterStack.clear();

   // Some trivial early exit, covering many cases in a cheap way.
   if (!name || !*name) return;
   const char lastChar = name[strlen(name) - 1];
   if (lastChar == '*' || lastChar == '&' || !strncmp(name, "const ", 6))
      return;

   // Ask the cling interpreter to lookup the name for us.
   const cling::LookupHelper& lh = fInterp->getLookupHelper();
   clang::QualType QT = lh.findType(name,
                                    gDebug > 5 ? cling::LookupHelper::WithDiagnostics
                                    : cling::LookupHelper::NoDiagnostics);
   if (QT.isNull()) {
      std::string buf = TClassEdit::InsertStd(name);
      if (buf != name) {
         QT = lh.findType(buf,
                          gDebug > 5 ? cling::LookupHelper::WithDiagnostics
                          : cling::LookupHelper::NoDiagnostics);
      }
      if (QT.isNull()) {
         return;
      }
   }
   const clang::TypedefType *td = QT->getAs<clang::TypedefType>();
   // if (fDecl && !llvm::isa<clang::TypedefDecl>(fDecl)) {
   if (!td) {
      // If what the lookup found is not a typedef, ignore it.
      return;
   }
   fDecl = td->getDecl();
}

//______________________________________________________________________________
bool TClingTypedefInfo::IsValid() const
{
   // Return true if the current iterator position is valid.
   return fDecl;
}

//______________________________________________________________________________
int TClingTypedefInfo::InternalNext()
{
   // Increment the iterator, return true if new position is valid.
   if (!*fIter) {
      // Iterator is already invalid.
      if (fFirstTime && fDecl) {
         std::string buf;
         clang::PrintingPolicy Policy(fDecl->getASTContext().getPrintingPolicy());
         llvm::raw_string_ostream stream(buf);
         llvm::dyn_cast<clang::NamedDecl>(fDecl)
            ->getNameForDiagnostic(stream, Policy, /*Qualified=*/false);
         stream.flush();
         Error("TClingTypedefInfo::InternalNext","Next called but iteration not prepared for %s!",buf.c_str());
      }
      return 0;
   }
   // Deserialization might happen during the iteration.
   cling::Interpreter::PushTransactionRAII pushedT(fInterp);
   while (true) {
      // Advance to next usable decl, or return if
      // there is no next usable decl.
      if (fFirstTime) {
         // The cint semantics are strange.
         fFirstTime = false;
      }
      else {
         // Advance the iterator one decl, descending into
         // the current decl context if necessary.
         if (!fDescend) {
            // Do not need to scan the decl context of the
            // current decl, move on to the next decl.
            ++fIter;
         }
         else {
            // Descend into the decl context of the current decl.
            fDescend = false;
            fIterStack.push_back(fIter);
            clang::DeclContext *dc = llvm::cast<clang::DeclContext>(*fIter);
            fIter = dc->decls_begin();
         }
         // Fix it if we went past the end.
         while (!*fIter && fIterStack.size()) {
            fIter = fIterStack.back();
            fIterStack.pop_back();
            ++fIter;
         }
         // Check for final termination.
         if (!*fIter) {
            // We have reached the end of the translation unit, all done.
            fDecl = 0;
            return 0;
         }
      }
      // Return if this decl is a typedef.
      if (llvm::isa<clang::TypedefNameDecl>(*fIter)) {
         fDecl = *fIter;
         return 1;
      }
      // Descend into namespaces and classes.
      clang::Decl::Kind dk = fIter->getKind();
      if ((dk == clang::Decl::Namespace) || (dk == clang::Decl::CXXRecord) ||
            (dk == clang::Decl::ClassTemplateSpecialization)) {
         fDescend = true;
      }
   }
}

//______________________________________________________________________________
int TClingTypedefInfo::Next()
{
   // Increment the iterator.
   return InternalNext();
}

//______________________________________________________________________________
long TClingTypedefInfo::Property() const
{
   // Return a bit mask of metadata about the current typedef.
   if (!IsValid()) {
      return 0L;
   }
   long property = 0L;
   property |= kIsTypedef;
   const clang::TypedefNameDecl *td = llvm::dyn_cast<clang::TypedefNameDecl>(fDecl);
   clang::QualType qt = td->getUnderlyingType().getCanonicalType();
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
   if (qt->isBuiltinType()) {
      property |= kIsFundamental;
   }
   if (qt.isConstQualified()) {
      property |= kIsConstant;
   }
   return property;
}

//______________________________________________________________________________
int TClingTypedefInfo::Size() const
{
   // Return the size in bytes of the underlying type of the current typedef.
   if (!IsValid()) {
      return 1;
   }
   clang::ASTContext &context = fDecl->getASTContext();
   const clang::TypedefNameDecl *td = llvm::dyn_cast<clang::TypedefNameDecl>(fDecl);
   clang::QualType qt = td->getUnderlyingType();
   if (qt->isDependentType()) {
      // The underlying type is dependent on a template parameter,
      // we have no idea what it is yet.
      return 0;
   }
   if (const clang::RecordType *rt = qt->getAs<clang::RecordType>()) {
      if (!rt->getDecl()->getDefinition()) {
         // This is a typedef to a forward-declared type.
         return 0;
      }
   }
   // Note: This is an int64_t.
   clang::CharUnits::QuantityType quantity =
      context.getTypeSizeInChars(qt).getQuantity();
   // Truncate cast to fit the CINT interface.
   return static_cast<int>(quantity);
}

//______________________________________________________________________________
const char *TClingTypedefInfo::TrueName(const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt) const
{
   // Get the name of the underlying type of the current typedef.
   if (!IsValid()) {
      return "(unknown)";
   }
   // Note: This must be static because we return a pointer to the internals.
   TTHREAD_TLS_DECL( std::string, truename);
   truename.clear();
   const clang::TypedefNameDecl *td = llvm::dyn_cast<clang::TypedefNameDecl>(fDecl);
   clang::QualType underlyingType = td->getUnderlyingType();
   if (underlyingType->isBooleanType()) {
      return "bool";
   }
   const clang::ASTContext &ctxt = fInterp->getCI()->getASTContext();
   ROOT::TMetaUtils::GetNormalizedName(truename, ctxt.getTypedefType(td), *fInterp, normCtxt);

   return truename.c_str();
}

//______________________________________________________________________________
const char *TClingTypedefInfo::Name() const
{
   // Get the name of the current typedef.
   if (!IsValid()) {
      return "(unknown)";
   }
   // Note: This must be static because we return a pointer to the internals.
   TTHREAD_TLS_DECL( std::string, fullname);
   fullname.clear();
   const clang::TypedefNameDecl *td = llvm::dyn_cast<clang::TypedefNameDecl>(fDecl);
   const clang::ASTContext &ctxt = fDecl->getASTContext();
   ROOT::TMetaUtils::GetFullyQualifiedTypeName(fullname,ctxt.getTypedefType(td),*fInterp);
   return fullname.c_str();
}

//______________________________________________________________________________
const char *TClingTypedefInfo::Title()
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
   if (const TypedefNameDecl *TND = llvm::dyn_cast<TypedefNameDecl>(GetDecl())) {
      if ( (TND = ROOT::TMetaUtils::GetAnnotatedRedeclarable(TND)) ) {
         if (AnnotateAttr *A = TND->getAttr<AnnotateAttr>()) {
            fTitle = A->getAnnotation().str();
            return fTitle.c_str();
         }
      }
   }
   else if (!GetDecl()->isFromASTFile()) {
      // Try to get the comment from the header file if present
      // but not for decls from AST file, where rootcling would have
      // created an annotation
      fTitle = ROOT::TMetaUtils::GetComment(*GetDecl()).str();
   }
   return fTitle.c_str();
}

