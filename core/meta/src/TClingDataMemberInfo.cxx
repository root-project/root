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

tcling_DataMemberInfo::~tcling_DataMemberInfo()
{
   // CINT material.
   delete fDataMemberInfo;
   fDataMemberInfo = 0;
   delete fClassInfo;
   fClassInfo = 0;
   // Clang material.
   delete fTClingClassInfo;
   fTClingClassInfo = 0;
}

tcling_DataMemberInfo::tcling_DataMemberInfo(cling::Interpreter* interp)
   : fDataMemberInfo(0)
   , fClassInfo(0)
   , fInterp(interp)
   , fTClingClassInfo(0)
   , fFirstTime(true)
{
   fDataMemberInfo = new G__DataMemberInfo;
   fClassInfo = new G__ClassInfo();
   fTClingClassInfo = new tcling_ClassInfo(fInterp);
   fIter = fInterp->getCI()->getASTContext().getTranslationUnitDecl()->
           decls_begin();
   // Move to first global variable.
   InternalNext();
}

tcling_DataMemberInfo::tcling_DataMemberInfo(cling::Interpreter* interp,
      tcling_ClassInfo* tcling_class_info)
   : fDataMemberInfo(0)
   , fClassInfo(0)
   , fInterp(interp)
   , fTClingClassInfo(0)
   , fFirstTime(true)
{
   if (!tcling_class_info || !tcling_class_info->IsValid()) {
      fDataMemberInfo = new G__DataMemberInfo;
      fClassInfo = new G__ClassInfo();
      fTClingClassInfo = new tcling_ClassInfo(fInterp);
      fIter = fInterp->getCI()->getASTContext().getTranslationUnitDecl()->
              decls_begin();
      // Move to first global variable.
      InternalNext();
      return;
   }
   fDataMemberInfo = new G__DataMemberInfo(*tcling_class_info->GetClassInfo());
   fClassInfo = new G__ClassInfo(*tcling_class_info->GetClassInfo());
   fTClingClassInfo = new tcling_ClassInfo(*tcling_class_info);
   if (tcling_class_info->IsValidClang()) {
      fIter = llvm::cast<clang::DeclContext>(tcling_class_info->GetDecl())->
              decls_begin();
      // Move to first data member.
      InternalNext();
   }
}

tcling_DataMemberInfo::tcling_DataMemberInfo(const tcling_DataMemberInfo& rhs)
{
   fDataMemberInfo = new G__DataMemberInfo(*rhs.fDataMemberInfo);
   fClassInfo = new G__ClassInfo(*rhs.fClassInfo);
   fInterp = rhs.fInterp;
   fTClingClassInfo = new tcling_ClassInfo(*rhs.fTClingClassInfo);
   fFirstTime = rhs.fFirstTime;
   fIter = rhs.fIter;
   fIterStack = rhs.fIterStack;
}

tcling_DataMemberInfo& tcling_DataMemberInfo::operator=(
   const tcling_DataMemberInfo& rhs)
{
   if (this == &rhs) {
      return *this;
   }
   delete fDataMemberInfo;
   fDataMemberInfo = new G__DataMemberInfo(*rhs.fDataMemberInfo);
   fInterp = rhs.fInterp;
   delete fClassInfo;
   fClassInfo = new G__ClassInfo(*rhs.fClassInfo);
   delete fTClingClassInfo;
   fTClingClassInfo = new tcling_ClassInfo(*rhs.fTClingClassInfo);
   fFirstTime = rhs.fFirstTime;
   fIter = rhs.fIter;
   fIterStack = rhs.fIterStack;
   return *this;
}

G__DataMemberInfo* tcling_DataMemberInfo::GetDataMemberInfo() const
{
   return fDataMemberInfo;
}

G__ClassInfo* tcling_DataMemberInfo::GetClassInfo() const
{
   return fClassInfo;
}

tcling_ClassInfo* tcling_DataMemberInfo::GetTClingClassInfo() const
{
   return fTClingClassInfo;
}

clang::Decl* tcling_DataMemberInfo::GetDecl() const
{
   return *fIter;
}

int tcling_DataMemberInfo::ArrayDim() const
{
   if (!IsValid()) {
      return -1;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fDataMemberInfo->ArrayDim();
      }
      return -1;
   }
   if (!gAllowClang) {
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
   const clang::ValueDecl* VD = llvm::dyn_cast<clang::ValueDecl>(*fIter);
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

bool tcling_DataMemberInfo::IsValidCint() const
{
   if (gAllowCint) {
      return fDataMemberInfo->IsValid();
   }
   return false;
}

bool tcling_DataMemberInfo::IsValidClang() const
{
   if (gAllowClang) {
      return *fIter;
   }
   return false;
}

bool tcling_DataMemberInfo::IsValid() const
{
   return IsValidCint() || IsValidClang();
}

int tcling_DataMemberInfo::MaxIndex(int dim) const
{
   if (!IsValid()) {
      return -1;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fDataMemberInfo->MaxIndex(dim);
      }
      return -1;
   }
   if (!gAllowClang) {
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
   const clang::ValueDecl* VD = llvm::dyn_cast<clang::ValueDecl>(*fIter);
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
            if (const clang::ConstantArrayType* CAT =
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

int tcling_DataMemberInfo::InternalNext()
{
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

bool tcling_DataMemberInfo::Next()
{
   if (!gAllowClang) {
      if (gAllowCint) {
         return fDataMemberInfo->Next();
      }
      return false;
   }
   return InternalNext();
}

long tcling_DataMemberInfo::Offset() const
{
   if (!IsValid()) {
      return -1L;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fDataMemberInfo->Offset();
      }
      return -1L;
   }
   if (!gAllowClang) {
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
      const clang::FieldDecl* FD = llvm::dyn_cast<clang::FieldDecl>(*fIter);
      clang::ASTContext& Context = FD->getASTContext();
      const clang::RecordDecl* RD = FD->getParent();
      const clang::ASTRecordLayout& Layout = Context.getASTRecordLayout(RD);
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

long tcling_DataMemberInfo::Property() const
{
   if (!IsValid()) {
      return 0L;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fDataMemberInfo->Property();
      }
      return 0L;
   }
   if (!gAllowClang) {
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
   if (const clang::VarDecl* VarD = llvm::dyn_cast<clang::VarDecl>(*fIter)) {
      if (VarD->getStorageClass() == clang::SC_Static) {
         property |= G__BIT_ISSTATIC;
      }
   }
   const clang::ValueDecl* ValueD = llvm::dyn_cast<clang::ValueDecl>(*fIter);
   clang::QualType QT = ValueD->getType();
   if (llvm::isa<clang::TypedefType>(QT)) {
      property |= G__BIT_ISTYPEDEF;
   }
   QT = QT.getCanonicalType();
   if (QT.isConstQualified()) {
      property |= G__BIT_ISCONSTANT;
   }
   while (1) {
      if (QT->isArrayType()) {
         property |= G__BIT_ISARRAY;
         QT = llvm::cast<clang::ArrayType>(QT)->getElementType();
         continue;
      }
      else if (QT->isReferenceType()) {
         property |= G__BIT_ISREFERENCE;
         QT = llvm::cast<clang::ReferenceType>(QT)->getPointeeType();
         continue;
      }
      else if (QT->isPointerType()) {
         property |= G__BIT_ISPOINTER;
         if (QT.isConstQualified()) {
            property |= G__BIT_ISPCONSTANT;
         }
         QT = llvm::cast<clang::PointerType>(QT)->getPointeeType();
         continue;
      }
      else if (QT->isMemberPointerType()) {
         QT = llvm::cast<clang::MemberPointerType>(QT)->getPointeeType();
         continue;
      }
      break;
   }
   if (QT->isBuiltinType()) {
      property |= G__BIT_ISFUNDAMENTAL;
   }
   if (QT.isConstQualified()) {
      property |= G__BIT_ISCONSTANT;
   }
   const clang::DeclContext* DC = fIter->getDeclContext();
   if (const clang::TagDecl* TD = llvm::dyn_cast<clang::TagDecl>(DC)) {
      if (TD->isClass()) {
         property |= G__BIT_ISCLASS;
      }
      else if (TD->isStruct()) {
         property |= G__BIT_ISSTRUCT;
      }
      else if (TD->isUnion()) {
         property |= G__BIT_ISUNION;
      }
      else if (TD->isEnum()) {
         property |= G__BIT_ISENUM;
      }
   }
   if (DC->isNamespace() && !DC->isTranslationUnit()) {
      property |= G__BIT_ISNAMESPACE;
   }
   if (gAllowCint) {
      if (IsValidCint()) {
         long cint_property = fDataMemberInfo->Property();
         if (property != cint_property) {
            if (gDebug > 0) {
               fprintf(stderr,
                       "VALIDITY: tcling_DataMemberInfo::Property:  "
                       "cint: 0x%lx  clang: 0x%lx\n",
                       (unsigned long) cint_property,
                       (unsigned long) property);
            }
         }
      }
   }
   return property;
}

long tcling_DataMemberInfo::TypeProperty() const
{
   if (!IsValid()) {
      return 0L;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fDataMemberInfo->Type()->Property();
      }
      return 0L;
   }
   if (!gAllowClang) {
      return 0L;
   }
   const clang::ValueDecl* ValueD = llvm::dyn_cast<clang::ValueDecl>(*fIter);
   clang::QualType QT = ValueD->getType();
   return tcling_TypeInfo(fInterp, QT).Property();
}

int tcling_DataMemberInfo::TypeSize() const
{
   if (!IsValid()) {
      return -1;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fDataMemberInfo->Type()->Size();
      }
      return -1;
   }
   if (!gAllowClang) {
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
   const clang::ValueDecl* VD = llvm::dyn_cast<clang::ValueDecl>(*fIter);
   clang::QualType QT = VD->getType();
   if (QT->isIncompleteType()) {
      // We cannot determine the size of forward-declared types.
      return -1;
   }
   clang::ASTContext& Context = fIter->getASTContext();
   return static_cast<int>(Context.getTypeSizeInChars(QT).getQuantity());
}

const char* tcling_DataMemberInfo::TypeName() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fDataMemberInfo->Type()->Name();
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   // Note: This must be static because we return a pointer inside it!
   static std::string buf;
   buf.clear();
   clang::PrintingPolicy P(fIter->getASTContext().getPrintingPolicy());
   P.AnonymousTagLocations = false;
   if (const clang::ValueDecl* VD = llvm::dyn_cast<clang::ValueDecl>(*fIter)) {
      buf = VD->getType().getAsString(P);
      return buf.c_str();
   }
   return 0;
}

const char* tcling_DataMemberInfo::TypeTrueName() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fDataMemberInfo->Type()->TrueName();
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   // Note: This must be static because we return a pointer inside it!
   static std::string buf;
   buf.clear();
   clang::PrintingPolicy P(fIter->getASTContext().getPrintingPolicy());
   P.AnonymousTagLocations = false;
   if (const clang::ValueDecl* VD = llvm::dyn_cast<clang::ValueDecl>(*fIter)) {
      buf = VD->getType().getCanonicalType().getAsString(P);
      return buf.c_str();
   }
   return 0;
}

const char* tcling_DataMemberInfo::Name() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fDataMemberInfo->Name();
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   // Note: This must be static because we return a pointer inside it!
   static std::string buf;
   buf.clear();
   if (const clang::NamedDecl* ND = llvm::dyn_cast<clang::NamedDecl>(*fIter)) {
      clang::PrintingPolicy Policy(fIter->getASTContext().getPrintingPolicy());
      ND->getNameForDiagnostic(buf, Policy, /*Qualified=*/false);
      return buf.c_str();
   }
   return 0;
}

const char* tcling_DataMemberInfo::Title() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fDataMemberInfo->Title();
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   // FIXME: Implement when rootcint makes this available!
   return "";
}

const char* tcling_DataMemberInfo::ValidArrayIndex() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fDataMemberInfo->ValidArrayIndex();
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   // FIXME: Implement when rootcint makes this available!
   return 0;
}

