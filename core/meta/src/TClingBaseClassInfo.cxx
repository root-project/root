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
// TClingBaseClassInfo                                                  //
//                                                                      //
// Emulation of the CINT BaseClassInfo class.                           //
//                                                                      //
// The CINT C++ interpreter provides an interface to metadata about     //
// the base classes of a class through the BaseClassInfo class.  This   //
// class provides the same functionality, using an interface as close   //
// as possible to BaseClassInfo but the base class metadata comes from  //
// the Clang C++ compiler, not CINT.                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClingBaseClassInfo.h"

tcling_BaseClassInfo::~tcling_BaseClassInfo()
{
   delete fBaseClassInfo;
   fBaseClassInfo = 0;
   fInterp = 0;
   delete fDerivedClassInfo;
   fDerivedClassInfo = 0;
   fDecl = 0;
   fIter = 0;
   delete fClassInfo;
   fClassInfo = 0;
}

tcling_BaseClassInfo::tcling_BaseClassInfo(tcling_ClassInfo* tcling_class_info)
   : fBaseClassInfo(0)
   , fInterp(0)
   , fDerivedClassInfo(0)
   , fFirstTime(true)
   , fDescend(false)
   , fDecl(0)
   , fIter(0)
   , fClassInfo(0)
   , fOffset(0L)
{
   if (!tcling_class_info || !tcling_class_info->IsValid()) {
      G__ClassInfo cli;
      fBaseClassInfo = new G__BaseClassInfo(cli);
      fInterp = 0;
      fDerivedClassInfo = new tcling_ClassInfo(fInterp);
      return;
   }
   fInterp = tcling_class_info->GetInterpreter();
   fBaseClassInfo = new G__BaseClassInfo(*tcling_class_info->GetClassInfo());
   fDerivedClassInfo = new tcling_ClassInfo(*tcling_class_info);
}

tcling_BaseClassInfo::tcling_BaseClassInfo(const tcling_BaseClassInfo& rhs)
   : fBaseClassInfo(0)
   , fInterp(0)
   , fDerivedClassInfo(0)
   , fFirstTime(true)
   , fDescend(false)
   , fDecl(0)
   , fIter(0)
   , fClassInfo(0)
   , fOffset(0L)
{
   if (!rhs.IsValid()) {
      G__ClassInfo cli;
      fBaseClassInfo = new G__BaseClassInfo(cli);
      fInterp = 0;
      fDerivedClassInfo = new tcling_ClassInfo(fInterp);
      return;
   }
   fBaseClassInfo = new G__BaseClassInfo(*rhs.fBaseClassInfo);
   fInterp = rhs.fInterp;
   fDerivedClassInfo = new tcling_ClassInfo(*rhs.fDerivedClassInfo);
   fFirstTime = rhs.fFirstTime;
   fDescend = rhs.fDescend;
   fDecl = rhs.fDecl;
   fIter = rhs.fIter;
   fClassInfo = new tcling_ClassInfo(*rhs.fClassInfo);
   fIterStack = rhs.fIterStack;
   fOffset = rhs.fOffset;
}

tcling_BaseClassInfo& tcling_BaseClassInfo::operator=(
   const tcling_BaseClassInfo& rhs)
{
   if (this == &rhs) {
      return *this;
   }
   if (!rhs.IsValid()) {
      delete fBaseClassInfo;
      fBaseClassInfo = 0;
      G__ClassInfo cli;
      fBaseClassInfo = new G__BaseClassInfo(cli);
      fInterp = 0;
      delete fDerivedClassInfo;
      fDerivedClassInfo = 0;
      fDerivedClassInfo = new tcling_ClassInfo(fInterp);
      fFirstTime = true;
      fDescend = false;
      fDecl = 0;
      fIter = 0;
      delete fClassInfo;
      fClassInfo = 0;
      // FIXME: Change this to use the swap trick to free the memory.
      fIterStack.clear();
      fOffset = 0L;
   }
   else {
      delete fBaseClassInfo;
      fBaseClassInfo = new G__BaseClassInfo(*rhs.fBaseClassInfo);
      fInterp = rhs.fInterp;
      delete fDerivedClassInfo;
      fDerivedClassInfo = new tcling_ClassInfo(*rhs.fDerivedClassInfo);
      fFirstTime = rhs.fFirstTime;
      fDescend = rhs.fDescend;
      fDecl = rhs.fDecl;
      fIter = rhs.fIter;
      delete fClassInfo;
      fClassInfo = new tcling_ClassInfo(*rhs.fClassInfo);
      fIterStack = rhs.fIterStack;
      fOffset = rhs.fOffset;
   }
   return *this;
}

G__BaseClassInfo* tcling_BaseClassInfo::GetBaseClassInfo() const
{
   return fBaseClassInfo;
}

tcling_ClassInfo* tcling_BaseClassInfo::GetDerivedClassInfo() const
{
   return fDerivedClassInfo;
}

tcling_ClassInfo* tcling_BaseClassInfo::GetClassInfo() const
{
   return fClassInfo;
}

long tcling_BaseClassInfo::GetOffsetBase() const
{
   return fOffset;
}

int tcling_BaseClassInfo::InternalNext(int onlyDirect)
{
   // Exit early if the iterator is already invalid.
   if (fIter == llvm::dyn_cast<clang::CXXRecordDecl>(fDecl)->bases_end()) {
      return 0;
   }
   // Advance the iterator.
   if (fFirstTime) {
      // The cint semantics are strange.
      const clang::CXXRecordDecl* CRD =
         llvm::dyn_cast<clang::CXXRecordDecl>(fDerivedClassInfo->GetDecl());
      if (!CRD) {
         // We were initialized with something that is not a class.
         // FIXME: We should prevent this from happening!
         return 0;
      }
      fIter = CRD->bases_begin();
      fFirstTime = false;
   }
   else if (!onlyDirect && fDescend) {
      // We previous processed a base class which itself has bases,
      // now we process the bases of that base class.
      fDescend = false;
      const clang::RecordType* Ty =
         fIter->getType()->getAs<clang::RecordType>();
      clang::CXXRecordDecl* Base =
         llvm::cast_or_null<clang::CXXRecordDecl>(
            Ty->getDecl()->getDefinition());
      clang::ASTContext& Context = Base->getASTContext();
      const clang::RecordDecl* RD = llvm::dyn_cast<clang::RecordDecl>(fDecl);
      const clang::ASTRecordLayout& Layout = Context.getASTRecordLayout(RD);
      int64_t offset = Layout.getBaseClassOffset(Base).getQuantity();
      fOffset += static_cast<long>(offset);
      fIterStack.push_back(std::make_pair(
                              std::make_pair(fDecl, fIter), static_cast<long>(offset)));
      fDecl = Base;
      fIter = Base->bases_begin();
   }
   else {
      // Simple case, move on to the next base class specifier.
      ++fIter;
   }
   // Fix it if we went past the end.
   while (
      (fIter == llvm::dyn_cast<clang::CXXRecordDecl>(fDecl)->bases_end()) &&
      fIterStack.size()
   ) {
      // All done with this base class.
      fDecl = fIterStack.back().first.first;
      fIter = fIterStack.back().first.second;
      fOffset -= fIterStack.back().second;
      fIterStack.pop_back();
      ++fIter;
   }
   // Check for final termination.
   if (fIter == llvm::dyn_cast<clang::CXXRecordDecl>(fDecl)->bases_end()) {
      // We have reached the end of the direct bases, all done.
      return 0;
   }
   return 1;
}

int tcling_BaseClassInfo::Next()
{
   return Next(1);
}

int tcling_BaseClassInfo::Next(int onlyDirect)
{
   if (!IsValid()) {
      return 0;
   }
   if (!fDerivedClassInfo->GetDecl()) {
      return fBaseClassInfo->Next(onlyDirect);
   }
   while (1) {
      // Advance the iterator.
      int valid_flag = InternalNext(onlyDirect);
      // Check if we have reached the end of the direct bases.
      if (!valid_flag) {
         // We have, all done.
         delete fClassInfo;
         fClassInfo = 0;
         return 0;
      }
      // Check if current base class is a dependent type, that is, an
      // uninstantiated template class.
      const clang::RecordType* Ty =
         fIter->getType()->getAs<clang::RecordType>();
      if (!Ty) {
         // A dependent type (uninstantiated template), skip it.
         continue;
      }
      // Check if current base class has a definition.
      const clang::CXXRecordDecl* Base =
         llvm::cast_or_null<clang::CXXRecordDecl>(Ty->getDecl()->
               getDefinition());
      if (!Base) {
         // No definition yet (just forward declared), skip it.
         continue;
      }
      // Now that we are going to return this base, check to see if
      // we need to examine its bases next call.
      if (!onlyDirect && Base->getNumBases()) {
         fDescend = true;
      }
      // Update info for this base class.
      fClassInfo = new tcling_ClassInfo(fInterp, Base);
      return 1;
   }
}

long tcling_BaseClassInfo::Offset() const
{
   //return fBaseClassInfo->Offset();
   if (!IsValid()) {
      return -1;
   }
   if (!fDerivedClassInfo->GetDecl()) {
      return fBaseClassInfo->Offset();
   }
   const clang::RecordType* Ty = fIter->getType()->getAs<clang::RecordType>();
   if (!Ty) {
      // A dependent type (uninstantiated template), invalid.
      return -1;
   }
   // Check if current base class has a definition.
   const clang::CXXRecordDecl* Base =
      llvm::cast_or_null<clang::CXXRecordDecl>(Ty->getDecl()->
            getDefinition());
   if (!Base) {
      // No definition yet (just forward declared), invalid.
      return -1;
   }
   clang::ASTContext& Context = Base->getASTContext();
   const clang::RecordDecl* RD = llvm::dyn_cast<clang::RecordDecl>(fDecl);
   const clang::ASTRecordLayout& Layout = Context.getASTRecordLayout(RD);
   int64_t offset = Layout.getBaseClassOffset(Base).getQuantity();
   return fOffset + static_cast<long>(offset);
}

long tcling_BaseClassInfo::Property() const
{
   //return fBaseClassInfo->Property();
   if (!IsValid()) {
      return 0;
   }
   if (!fDerivedClassInfo->GetDecl()) {
      return fBaseClassInfo->Property();
   }
   long property = 0L;
   if (fIter->isVirtual()) {
      property |= G__BIT_ISVIRTUALBASE;
   }
   if (fDecl == fDerivedClassInfo->GetDecl()) {
      property |= G__BIT_ISDIRECTINHERIT;
   }
   switch (fIter->getAccessSpecifier()) {
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
   return property;
}

long tcling_BaseClassInfo::Tagnum() const
{
   //return fBaseClassInfo->Tagnum();
   if (!IsValid()) {
      return -1;
   }
   if (!fDerivedClassInfo->GetDecl()) {
      return fBaseClassInfo->Tagnum();
   }
   // Note: This *must* return a *cint* tagnum for now.
   return fClassInfo->Tagnum();
}

const char* tcling_BaseClassInfo::FullName() const
{
   //return fBaseClassInfo->Fullname();
   if (!IsValid()) {
      return 0;
   }
   if (!fDerivedClassInfo->GetDecl()) {
      return fBaseClassInfo->Fullname();
   }
   return fClassInfo->FullName();
}

const char* tcling_BaseClassInfo::Name() const
{
   //return fBaseClassInfo->Name();
   if (!IsValid()) {
      return 0;
   }
   if (!fDerivedClassInfo->GetDecl()) {
      return fBaseClassInfo->Name();
   }
   return fClassInfo->Name();
}

const char* tcling_BaseClassInfo::TmpltName() const
{
   //return fBaseClassInfo->TmpltName();
   if (!IsValid()) {
      return 0;
   }
   if (!fDerivedClassInfo->GetDecl()) {
      return fBaseClassInfo->TmpltName();
   }
   return fClassInfo->TmpltName();
}

bool tcling_BaseClassInfo::IsValid() const
{
   if (!fDerivedClassInfo->GetDecl()) {
      return fBaseClassInfo->IsValid();
   }
   if (
      fDecl && // the base class we are currently iterating over is valid, and
      // our internal iterator is currently valid, and
      fIter &&
      (fIter != llvm::dyn_cast<clang::CXXRecordDecl>(fDecl)->bases_end()) &&
      fClassInfo && // our current base has a tcling_ClassInfo, and
      fClassInfo->IsValid() // our current base is a valid class
   ) {
      return true;
   }
   return false;
}

