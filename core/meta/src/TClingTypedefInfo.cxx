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

#include "Property.h"
#include "TClingProperty.h"
#include "TError.h"
#include "TMetaUtils.h"
#include "Rtypes.h" // for gDebug

#include "cling/Interpreter/LookupHelper.h"

using namespace clang;

//______________________________________________________________________________
TClingTypedefInfo::TClingTypedefInfo(cling::Interpreter *interp,
                                     const char *name)
   : fInterp(interp), fFirstTime(true), fDescend(false), fDecl(0), fTitle("")
{
   // Lookup named typedef and initialize the iterator to point to it.
   if (gDebug > 0) {
      Info("TClingTypedefInfo::TClingTypedefInfo(interp,name)",
           "looking up typedef: %s\n", name);
   }
   fDecl = fInterp->getLookupHelper().findScope(name);
   if (fDecl && !llvm::isa<clang::TypedefDecl>(fDecl)) {
      // If what the lookup found is not a typedef, ignore it.
      fDecl = 0;
   }
   if (fDecl) {
      if (gDebug > 0) {
         Info("TClingTypedefInfo::TClingTypedefInfo(interp,name)",
              "found typedef name: %s  decl: 0x%lx\n", name, (long) fDecl);
      }
      // Initialize iterator to point at found typedef.
      AdvanceToDecl(fDecl);
   }
   else {
      if (gDebug > 0) {
         Info("TClingTypedefInfo::TClingTypedefInfo(interp,name)",
              "typedef not found name: %s\n", name);
      }
   }
}

TClingTypedefInfo::TClingTypedefInfo(cling::Interpreter *interp,
                                     const clang::TypedefDecl *TdefD)
   : fInterp(interp), fFirstTime(true), fDescend(false), fDecl(TdefD), 
     fTitle("")
{
   // Lookup named typedef and initialize the iterator to point to it.
   if (gDebug > 0) {
      Info("TClingTypedefInfo::TClingTypedefInfo(interp,name)",
           "looking up typedef: %s\n", TdefD->getNameAsString().c_str());
   }
   if (fDecl) {
      if (gDebug > 0) {
         Info("TClingTypedefInfo::TClingTypedefInfo(interp,name)",
              "found typedef name: %s  decl: 0x%lx\n", 
              TdefD->getNameAsString().c_str(), (long) fDecl);
      }
      // Initialize iterator to point at found typedef.
      AdvanceToDecl(fDecl);
   }
   else {
      if (gDebug > 0) {
         Info("TClingTypedefInfo::TClingTypedefInfo(interp,name)",
              "typedef not found name: %s\n", TdefD->getNameAsString().c_str());
      }
   }
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
   if (gDebug > 0) {
      Info("TClingTypedefInfo::Init(name)", "looking up typedef: %s\n", name);
   }
   // Reset the iterator to invalid.
   fFirstTime = true;
   fDescend = false;
   fIter = clang::DeclContext::decl_iterator();
   fIterStack.clear();
   // Ask the cling interpreter to lookup the name for us.
   fDecl = fInterp->getLookupHelper().findScope(name);
   if (!fDecl) {
      if (gDebug > 0) {
         Info("TClingTypedefInfo::Init(name)",
              "typedef not found name: %s\n", name);
      }
      return;
   }
   if (gDebug > 0) {
      Info("TClingTypedefInfo::Init(name)", "found typedef name: "
           "%s  decl: 0x%lx\n", name, (long) fDecl);
   }
   // Initialize iterator to point at found typedef.
   AdvanceToDecl(fDecl);
}

//______________________________________________________________________________
bool TClingTypedefInfo::IsValid() const
{
   // Return true if the current iterator position is valid.
   return fDecl;
}

//______________________________________________________________________________
int TClingTypedefInfo::AdvanceToDecl(const clang::Decl *target_decl)
{
   // Set the iterator to point at the given declaration.
   const clang::TranslationUnitDecl *tu = target_decl->getTranslationUnitDecl();
   const clang::DeclContext *dc = llvm::cast<clang::DeclContext>(tu);
   fFirstTime = true;
   fDescend = false;
   fIter = dc->decls_begin();
   fDecl = 0;
   fIterStack.clear();
   while (InternalNext()) {
      if (fDecl == target_decl) {
         fFirstTime = true;
         return 1;
      }
   }
   return 0;
}

//______________________________________________________________________________
int TClingTypedefInfo::InternalNext()
{
   // Increment the iterator, return true if new position is valid.
   if (!*fIter) {
      // Iterator is already invalid.
      return 0;
   }
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
      if (llvm::isa<clang::TypedefDecl>(*fIter)) {
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
   property |= G__BIT_ISTYPEDEF;
   const clang::TypedefDecl *td = llvm::dyn_cast<clang::TypedefDecl>(fDecl);
   clang::QualType qt = td->getUnderlyingType().getCanonicalType();
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
   if (qt->isBuiltinType()) {
      property |= G__BIT_ISFUNDAMENTAL;
   }
   if (qt.isConstQualified()) {
      property |= G__BIT_ISCONSTANT;
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
   const clang::TypedefDecl *td = llvm::dyn_cast<clang::TypedefDecl>(fDecl);
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
const char *TClingTypedefInfo::TrueName() const
{
   // Get the name of the underlying type of the current typedef.
   if (!IsValid()) {
      return "(unknown)";
   }
   // Note: This must be static because we return a pointer to the internals.
   static std::string truename;
   truename.clear();
   const clang::TypedefDecl *td = llvm::dyn_cast<clang::TypedefDecl>(fDecl);
   clang::QualType underlyingType = td->getUnderlyingType();
   if (underlyingType->isBooleanType()) {
      return "bool";
   }
   truename = td->getUnderlyingType().getAsString();
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
   static std::string fullname;
   fullname.clear();
   clang::PrintingPolicy Policy(fDecl->getASTContext().getPrintingPolicy());
   llvm::dyn_cast<clang::NamedDecl>(fDecl)->
   getNameForDiagnostic(fullname, Policy, /*Qualified=*/true);
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

   // Try to get the comment from the header file if present
   fTitle = ROOT::TMetaUtils::GetComment(*GetDecl()).str();
   return fTitle.c_str();
}

