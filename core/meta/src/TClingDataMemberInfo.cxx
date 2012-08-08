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
   : fDataMemberInfo(new G__DataMemberInfo)
   , fClassInfo(0)
   , fInterp(interp)
   , fTClingClassInfo(0)
   , fFirstTime(true)
   , fIter(fInterp->getCI()->getASTContext().getTranslationUnitDecl()->decls_begin())
{
   fClassInfo = new G__ClassInfo();
   fTClingClassInfo = new tcling_ClassInfo(fInterp);
   // Move to first global variable.
   InternalNextValidMember();
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
      fIter = fInterp->getCI()->getASTContext().getTranslationUnitDecl()->decls_begin();
      // Move to first global variable.
      InternalNextValidMember();
      return;
   }
   fDataMemberInfo = new G__DataMemberInfo(*tcling_class_info->GetClassInfo());
   fClassInfo = new G__ClassInfo(*tcling_class_info->GetClassInfo());
   fTClingClassInfo = new tcling_ClassInfo(*tcling_class_info);
   fIter = llvm::dyn_cast<clang::DeclContext>(tcling_class_info->GetDecl())->
           decls_begin();
   // Move to first data member.
   InternalNextValidMember();
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

tcling_DataMemberInfo& tcling_DataMemberInfo::operator=(const tcling_DataMemberInfo& rhs)
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
   //return fDataMemberInfo->ArrayDim();
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
      return -1L;
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

bool tcling_DataMemberInfo::IsValid() const
{
   if (fFirstTime) {
      return false;
   }
   return *fIter;
}

int tcling_DataMemberInfo::MaxIndex(int dim) const
{
   //return fDataMemberInfo->MaxIndex(dim);
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
      return -1L;
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

void tcling_DataMemberInfo::InternalNextValidMember()
{
   //static std::string buf;
   // Move to next acceptable data member.
   while (1) {
      // Reject members we do not want, and recurse into
      // transparent contexts.
      while (*fIter) {
         // Valid decl, recurse into it, accept it, or reject it.
         clang::Decl::Kind DK = fIter->getKind();
         if (DK == clang::Decl::Enum) {
            // Recurse down into a transparent context.
            //buf = "recurse into: ";
            //buf += fIter->getDeclKindName();
            //if (llvm::dyn_cast<clang::NamedDecl>(*fIter)) {
            //   buf += " " + llvm::dyn_cast<clang::NamedDecl>(*fIter)->getNameAsString();
            //}
            //if (llvm::dyn_cast<clang::DeclContext>(*fIter)) {
            //   if (llvm::dyn_cast<clang::DeclContext>(*fIter)->
            //          isTransparentContext()
            //   ) {
            //      buf += " transparent";
            //   }
            //}
            //fprintf(stderr, "%s\n", buf.c_str());
            fIterStack.push_back(fIter);
            fIter = llvm::dyn_cast<clang::DeclContext>(*fIter)->decls_begin();
            continue;
         }
         else if (
            (DK == clang::Decl::Field) ||
            (DK == clang::Decl::EnumConstant) ||
            (DK == clang::Decl::Var)
         ) {
            // We will process these kinds of members.
            break;
         }
         // Rejected, next member.
         //buf = "rejected: ";
         //buf += fIter->getDeclKindName();
         //if (llvm::dyn_cast<clang::NamedDecl>(*fIter)) {
         //   buf += " " + llvm::dyn_cast<clang::NamedDecl>(*fIter)->getNameAsString();
         //}
         //if (llvm::dyn_cast<clang::DeclContext>(*fIter)) {
         //   if (llvm::dyn_cast<clang::DeclContext>(*fIter)->
         //          isTransparentContext()
         //   ) {
         //      buf += " transparent";
         //   }
         //}
         //fprintf(stderr, "%s\n", buf.c_str());
         ++fIter;
      }
      // Accepted member, or at end of decl context.
      if (!*fIter && fIterStack.size()) {
         // End of decl context, and we have more to go.
         //fprintf(stderr, "pop stack:\n");
         fIter = fIterStack.back();
         fIterStack.pop_back();
         ++fIter;
         continue;
      }
      // Accepted member, or at end of outermost decl context.
      //if (*fIter) {
      //buf = "accepted: ";
      //buf += fIter->getDeclKindName();
      //if (llvm::dyn_cast<clang::NamedDecl>(*fIter)) {
      //   buf += " " + llvm::dyn_cast<clang::NamedDecl>(*fIter)->getNameAsString();
      //}
      //if (llvm::dyn_cast<clang::DeclContext>(*fIter)) {
      //   if (llvm::dyn_cast<clang::DeclContext>(*fIter)->
      //          isTransparentContext()
      //   ) {
      //      buf += " transparent";
      //   }
      //}
      //fprintf(stderr, "%s\n", buf.c_str());
      //}
      //else {
      //fprintf(stderr, "end of outermost decl context:\n");
      //}
      break;
   }
}

bool tcling_DataMemberInfo::Next()
{
   fDataMemberInfo->Next();
   if (!fIterStack.size() && !*fIter) {
      // Terminate early if we are already invalid.
      //fprintf(stderr, "Next: early termination!\n");
      return false;
   }
   if (fFirstTime) {
      // No increment for first data member, the cint interface is awkward.
      //fprintf(stderr, "Next: first time!\n");
      fFirstTime = false;
   }
   else {
      // Move to next data member.
      ++fIter;
      InternalNextValidMember();
   }
   // Accepted member, or at end of outermost decl context.
   if (!*fIter) {
      // We are now invalid, return that.
      return false;
   }
   // We are now pointing at the next data member, return that we are valid.
   return true;
}

long tcling_DataMemberInfo::Offset() const
{
   //return fDataMemberInfo->Offset();
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
   //return fDataMemberInfo->Property();
   return 0L;
}

long tcling_DataMemberInfo::TypeProperty() const
{
   //return fDataMemberInfo->Type()->Property();
   return 0L;
}

int tcling_DataMemberInfo::TypeSize() const
{
   //return fDataMemberInfo->Type()->Size();
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
   const clang::ValueDecl* VD = llvm::dyn_cast<clang::ValueDecl>(*fIter);
   clang::QualType QT = VD->getType();
   if (QT->isIncompleteType()) {
      // We cannot determine the size of forward-declared types.
      return -1L;
   }
   clang::ASTContext& Context = fIter->getASTContext();
   return static_cast<int>(Context.getTypeSizeInChars(QT).getQuantity());
}

const char* tcling_DataMemberInfo::TypeName() const
{
   //return fDataMemberInfo->Type()->Name();
   static std::string buf;
   if (!IsValid()) {
      return 0;
   }
   buf.clear();
   clang::PrintingPolicy P(fIter->getASTContext().getPrintingPolicy());
   P.AnonymousTagLocations = false;
   if (llvm::dyn_cast<clang::ValueDecl>(*fIter)) {
      buf = llvm::dyn_cast<clang::ValueDecl>(*fIter)->getType().getAsString(P);
      //llvm::dyn_cast<clang::ValueDecl>(*fIter)->getType().dump();
   }
   else {
      return 0;
   }
   return buf.c_str();
}

const char* tcling_DataMemberInfo::TypeTrueName() const
{
   //return fDataMemberInfo->Type()->TrueName();
   static std::string buf;
   if (!IsValid()) {
      return 0;
   }
   buf.clear();
   clang::PrintingPolicy P(fIter->getASTContext().getPrintingPolicy());
   P.AnonymousTagLocations = false;
   if (clang::dyn_cast<clang::ValueDecl>(*fIter)) {
      buf = clang::dyn_cast<clang::ValueDecl>(*fIter)->
            getType().getCanonicalType().getAsString(P);
      //llvm::dyn_cast<clang::ValueDecl>(*fIter)->getType().
      //   getCanonicalType().dump();
   }
   else {
      return 0;
   }
   return buf.c_str();
}

const char* tcling_DataMemberInfo::Name() const
{
   //return fDataMemberInfo->Name();
   static std::string buf;
   if (!IsValid()) {
      return 0;
   }
   buf.clear();
   //buf = fIter->getDeclKindName();
   if (llvm::dyn_cast<clang::NamedDecl>(*fIter)) {
      clang::PrintingPolicy P((*fIter)->getASTContext().getPrintingPolicy());
      llvm::dyn_cast<clang::NamedDecl>(*fIter)->
      getNameForDiagnostic(buf, P, false);
   }
#if 0
   {
      clang::SourceLocation Loc = fIter->getLocation();
   }
   {
      clang::SourceLocation LocStart = fIter->getLocStart();
   }
   {
      clang::SourceLocation LocEnd = fIter->getLocEnd();
   }
   {
      clang::SourceRange LocRange = fIter->getSourceRange();
      {
         clang::SourceLocation Loc = LocRange.getBegin();
         std::string empty;
         llvm::raw_string_ostream OS(empty);
         clang::CompilerInstance* CI = tcling_Dict::GetCI();
         Loc.print(OS, CI->getSourceManager());
         buf += " " + OS.str();
      }
      {
         clang::SourceLocation Loc = LocRange.getEnd();
         std::string empty;
         llvm::raw_string_ostream OS(empty);
         clang::CompilerInstance* CI = tcling_Dict::GetCI();
         Loc.print(OS, CI->getSourceManager());
         buf += " " + OS.str();
      }
   }
#endif // 0
   //if (llvm::dyn_cast<clang::DeclContext>(*fIter)) {
   //   if (llvm::dyn_cast<clang::DeclContext>(*fIter)->
   //          isTransparentContext()
   //   ) {
   //      buf += " transparent";
   //   }
   //}
   return buf.c_str();
}

const char* tcling_DataMemberInfo::Title() const
{
   //return fDataMemberInfo->Title();
   return "";
}

const char* tcling_DataMemberInfo::ValidArrayIndex() const
{
   //return fDataMemberInfo->ValidArrayIndex();
   return 0;
}

