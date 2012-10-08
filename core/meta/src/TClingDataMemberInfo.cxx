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
// TClingDataMemberInfo                                                 //
//                                                                      //
// Emulation of the CINT DataMemberInfo class.                          //
//                                                                      //
// The CINT C++ interpreter provides an interface to metadata about     //
// the data members of a class through the DataMemberInfo class.  This  //
// class provides the same functionality, using an interface as close   //
// as possible to DataMemberInfo but the data member metadata comes     //
// from the Clang C++ compiler, not CINT.                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClingDataMemberInfo.h"

#include "Property.h"
#include "TClingProperty.h"
#include "TClingTypeInfo.h"
#include "TMetaUtils.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

TClingDataMemberInfo::TClingDataMemberInfo(cling::Interpreter *interp,
                                           TClingClassInfo *ci)
   : fInterp(interp), fClassInfo(0), fFirstTime(true), fTitle(""), fSingleDecl(0)
{
   if (!ci) {
      // We are meant to iterate over the global namespace (well at least CINT did).
      fClassInfo = new TClingClassInfo(interp);      
   } else {
      fClassInfo = new TClingClassInfo(*ci);
   }
   if (fClassInfo->IsValid()) {
      const Decl *D = fClassInfo->GetDecl();
      fIter = llvm::cast<clang::DeclContext>(D)->decls_begin();
      const TagDecl *TD = ROOT::TMetaUtils::GetAnnotatedRedeclarable(llvm::dyn_cast<TagDecl>(D));
      if (TD)
         fIter = TD->decls_begin();
      
      // Move to first data member.
      InternalNext();
      fFirstTime = true;
   }
}
TClingDataMemberInfo::TClingDataMemberInfo(const clang::ValueDecl *ValD)
   : fInterp(0), fClassInfo(0), fFirstTime(true), fTitle(""), 
     fSingleDecl(ValD) {
   using namespace llvm;
   assert(isa<TranslationUnitDecl>(ValD->getDeclContext()) && "Not TU?");
   assert((isa<VarDecl>(ValD) || 
           isa<FieldDecl>(ValD) || 
           isa<EnumConstantDecl>(ValD)) &&
          "The decl should be either VarDecl or FieldDecl or EnumConstDecl");
}


int TClingDataMemberInfo::ArrayDim() const
{
   if (!IsValid()) {
      return -1;
   }
   // Sanity check the current data member.
   clang::Decl::Kind DK = fIter->getKind();
   if (
      (DK != clang::Decl::Field) &&
      (DK != clang::Decl::Var) &&
      (DK != clang::Decl::EnumConstant)
   ) {
      // Error, was not a data member, variable, or enumerator.
      return -1;
   }
   if (DK == clang::Decl::EnumConstant) {
      // We know that an enumerator value does not have array type.
      return 0;
   }
   // To get this information we must count the number
   // of arry type nodes in the canonical type chain.
   const clang::ValueDecl *VD = llvm::dyn_cast<clang::ValueDecl>(*fIter);
   clang::QualType QT = VD->getType().getCanonicalType();
   int cnt = 0;
   while (1) {
      if (QT->isArrayType()) {
         ++cnt;
         QT = llvm::cast<clang::ArrayType>(QT)->getElementType();
         continue;
      }
      else if (QT->isReferenceType()) {
         QT = llvm::cast<clang::ReferenceType>(QT)->getPointeeType();
         continue;
      }
      else if (QT->isPointerType()) {
         QT = llvm::cast<clang::PointerType>(QT)->getPointeeType();
         continue;
      }
      else if (QT->isMemberPointerType()) {
         QT = llvm::cast<clang::MemberPointerType>(QT)->getPointeeType();
         continue;
      }
      break;
   }
   return cnt;
}

int TClingDataMemberInfo::MaxIndex(int dim) const
{
   if (!IsValid()) {
      return -1;
   }
   // Sanity check the current data member.
   clang::Decl::Kind DK = fIter->getKind();
   if (
      (DK != clang::Decl::Field) &&
      (DK != clang::Decl::Var) &&
      (DK != clang::Decl::EnumConstant)
   ) {
      // Error, was not a data member, variable, or enumerator.
      return -1;
   }
   if (DK == clang::Decl::EnumConstant) {
      // We know that an enumerator value does not have array type.
      return 0;
   }
   // To get this information we must count the number
   // of arry type nodes in the canonical type chain.
   const clang::ValueDecl *VD = llvm::dyn_cast<clang::ValueDecl>(*fIter);
   clang::QualType QT = VD->getType().getCanonicalType();
   int paran = ArrayDim();
   if ((dim < 0) || (dim >= paran)) {
      // Passed dimension is out of bounds.
      return -1;
   }
   int cnt = dim;
   int max = 0;
   while (1) {
      if (QT->isArrayType()) {
         if (cnt == 0) {
            if (const clang::ConstantArrayType *CAT =
                     llvm::dyn_cast<clang::ConstantArrayType>(QT)
               ) {
               max = static_cast<int>(CAT->getSize().getZExtValue());
            }
            else if (llvm::dyn_cast<clang::IncompleteArrayType>(QT)) {
               max = INT_MAX;
            }
            else {
               max = -1;
            }
            break;
         }
         --cnt;
         QT = llvm::cast<clang::ArrayType>(QT)->getElementType();
         continue;
      }
      else if (QT->isReferenceType()) {
         QT = llvm::cast<clang::ReferenceType>(QT)->getPointeeType();
         continue;
      }
      else if (QT->isPointerType()) {
         QT = llvm::cast<clang::PointerType>(QT)->getPointeeType();
         continue;
      }
      else if (QT->isMemberPointerType()) {
         QT = llvm::cast<clang::MemberPointerType>(QT)->getPointeeType();
         continue;
      }
      break;
   }
   return max;
}

int TClingDataMemberInfo::InternalNext()
{
   assert(!fSingleDecl && "This is not an iterator!");

   // Move to next acceptable data member.
   while (*fIter) {
      // Move to next decl in context.
      if (fFirstTime) {
         fFirstTime = false;
      }
      else {
         ++fIter;
      }
      // Handle reaching end of current decl context.
      if (!*fIter && fIterStack.size()) {
         // End of current decl context, and we have more to go.
         fIter = fIterStack.back();
         fIterStack.pop_back();
         ++fIter;
         continue;
      }
      // Handle final termination.
      if (!*fIter) {
         return 0;
      }
      // Valid decl, recurse into it, accept it, or reject it.
      clang::Decl::Kind DK = fIter->getKind();
      if (DK == clang::Decl::Enum) {
         // We have an enum, recurse into these.
         // Note: For C++11 we will have to check for a transparent context.
         fIterStack.push_back(fIter);
         fIter = llvm::dyn_cast<clang::DeclContext>(*fIter)->decls_begin();
         continue;
      }
      if ((DK == clang::Decl::Field) || (DK == clang::Decl::EnumConstant) ||
            (DK == clang::Decl::Var)) {
         // Stop on class data members, enumerator values,
         // and namespace variable members.
         return 1;
      }
   }
   return 0;
}

long TClingDataMemberInfo::Offset() const
{
   if (!IsValid()) {
      return -1L;
   }
   // Sanity check the current data member.
   clang::Decl::Kind DK = fIter->getKind();
   if (
      (DK != clang::Decl::Field) &&
      (DK != clang::Decl::Var) &&
      (DK != clang::Decl::EnumConstant)
   ) {
      // Error, was not a data member, variable, or enumerator.
      return -1L;
   }
   if (DK == clang::Decl::Field) {
      // The current member is a non-static data member.
      const clang::FieldDecl *FD = llvm::dyn_cast<clang::FieldDecl>(*fIter);
      clang::ASTContext &Context = FD->getASTContext();
      const clang::RecordDecl *RD = FD->getParent();
      const clang::ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
      uint64_t bits = Layout.getFieldOffset(FD->getFieldIndex());
      int64_t offset = Context.toCharUnitsFromBits(bits).getQuantity();
      return static_cast<long>(offset);
   }
   // The current member is static data member, enumerator constant,
   // or a global variable.
   // FIXME: We are supposed to return the address of the storage
   //        for the member here, only the interpreter knows that.
   return -1L;
}

long TClingDataMemberInfo::Property() const
{
   if (!IsValid()) {
      return 0L;
   }
   long property = 0L;
   switch (fIter->getAccess()) {
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
   if (const clang::VarDecl *vard = llvm::dyn_cast<clang::VarDecl>(*fIter)) {
      if (vard->getStorageClass() == clang::SC_Static) {
         property |= G__BIT_ISSTATIC;
      }
   }
   const clang::ValueDecl *vd = llvm::dyn_cast<clang::ValueDecl>(*fIter);
   clang::QualType qt = vd->getType();
   if (llvm::isa<clang::TypedefType>(qt)) {
      property |= G__BIT_ISTYPEDEF;
   }
   qt = qt.getCanonicalType();
   if (qt.isConstQualified()) {
      property |= G__BIT_ISCONSTANT;
   }
   while (1) {
      if (qt->isArrayType()) {
         property |= G__BIT_ISARRAY;
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
   const clang::DeclContext *dc = fIter->getDeclContext();
   if (const clang::TagDecl *td = llvm::dyn_cast<clang::TagDecl>(dc)) {
      if (td->isClass()) {
         property |= G__BIT_ISCLASS;
      }
      else if (td->isStruct()) {
         property |= G__BIT_ISSTRUCT;
      }
      else if (td->isUnion()) {
         property |= G__BIT_ISUNION;
      }
      else if (td->isEnum()) {
         property |= G__BIT_ISENUM;
      }
   }
   if (dc->isNamespace() && !dc->isTranslationUnit()) {
      property |= G__BIT_ISNAMESPACE;
   }
   return property;
}

long TClingDataMemberInfo::TypeProperty() const
{
   if (!IsValid()) {
      return 0L;
   }
   const clang::ValueDecl *vd = llvm::dyn_cast<clang::ValueDecl>(*fIter);
   clang::QualType qt = vd->getType();
   return TClingTypeInfo(fInterp, qt).Property();
}

int TClingDataMemberInfo::TypeSize() const
{
   if (!IsValid()) {
      return -1;
   }
   // Sanity check the current data member.
   clang::Decl::Kind dk = fIter->getKind();
   if ((dk != clang::Decl::Field) && (dk != clang::Decl::Var) &&
         (dk != clang::Decl::EnumConstant)) {
      // Error, was not a data member, variable, or enumerator.
      return -1;
   }
   const clang::ValueDecl *vd = llvm::dyn_cast<clang::ValueDecl>(*fIter);
   clang::QualType qt = vd->getType();
   if (qt->isIncompleteType()) {
      // We cannot determine the size of forward-declared types.
      return -1;
   }
   clang::ASTContext &context = fIter->getASTContext();
   // Truncate cast to fit to cint interface.
   return static_cast<int>(context.getTypeSizeInChars(qt).getQuantity());
}

const char *TClingDataMemberInfo::TypeName() const
{
   if (!IsValid()) {
      return 0;
   }
   // Note: This must be static because we return a pointer inside it!
   static std::string buf;
   buf.clear();
   clang::PrintingPolicy policy(fIter->getASTContext().getPrintingPolicy());
   policy.AnonymousTagLocations = false;
   if (const clang::ValueDecl *vd = llvm::dyn_cast<clang::ValueDecl>(*fIter)) {
      buf = vd->getType().getAsString(policy);
      return buf.c_str();
   }
   return 0;
}

const char *TClingDataMemberInfo::TypeTrueName() const
{
   if (!IsValid()) {
      return 0;
   }
   // Note: This must be static because we return a pointer inside it!
   static std::string buf;
   buf.clear();
   clang::PrintingPolicy policy(fIter->getASTContext().getPrintingPolicy());
   policy.AnonymousTagLocations = false;
   if (const clang::ValueDecl *vd = llvm::dyn_cast<clang::ValueDecl>(*fIter)) {
      buf = vd->getType().getCanonicalType().getAsString(policy);
      return buf.c_str();
   }
   return 0;
}

const char *TClingDataMemberInfo::Name() const
{
   if (!IsValid()) {
      return 0;
   }
   // Note: This must be static because we return a pointer inside it!
   static std::string buf;
   buf.clear();
   if (const clang::NamedDecl *nd = llvm::dyn_cast<clang::NamedDecl>(*fIter)) {
      clang::PrintingPolicy policy(fIter->getASTContext().getPrintingPolicy());
      nd->getNameForDiagnostic(buf, policy, /*Qualified=*/false);
      return buf.c_str();
   }
   return 0;
}

const char *TClingDataMemberInfo::Title()
{
   if (!IsValid()) {
      return 0;
   }

   //NOTE: We can't use it as a cache due to the "thoughtful" self iterator
   //if (fTitle.size())
   //   return fTitle.c_str();

   // Try to get the comment either from the annotation or the header file if present
   if (AnnotateAttr *A = GetDecl()->getAttr<AnnotateAttr>())
      fTitle = A->getAnnotation().str();
   else
      // Try to get the comment from the header file if present
      fTitle = ROOT::TMetaUtils::GetComment(*GetDecl()).str();

   return fTitle.c_str();
}

const char *TClingDataMemberInfo::ValidArrayIndex() const
{
   if (!IsValid()) {
      return 0;
   }
   // FIXME: Implement when rootcint makes this available!
   return 0;
}

