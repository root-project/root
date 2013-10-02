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

#include "TClingClassInfo.h"
#include "TDictionary.h"
#include "TMetaUtils.h"

#include "TError.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Transaction.h"


#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/Module.h"

#include <string>
#include <sstream>
#include <iostream>

using namespace llvm;
using namespace clang;
using namespace std;

static const string indent_string("   ");

TClingBaseClassInfo::TClingBaseClassInfo(cling::Interpreter* interp,
                                         const TClingClassInfo* ci)
   : fInterp(interp), fClassInfo(0), fFirstTime(true), fDescend(false),
     fDecl(0), fIter(0), fBaseInfo(0), fOffset(0L)
{
   if (!ci) {
      fClassInfo = new TClingClassInfo(interp);
      return;
   }
   fClassInfo = ci;
   if (!fClassInfo->GetDecl()) {
      return;
   }
   const clang::CXXRecordDecl* CRD =
      llvm::dyn_cast<clang::CXXRecordDecl>(fClassInfo->GetDecl());
   if (!CRD) {
      // We were initialized with something that is not a class.
      // FIXME: We should prevent this from happening!
      return;
   }
   fDecl = CRD;
   fIter = CRD->bases_begin();
}

TClingBaseClassInfo::TClingBaseClassInfo(cling::Interpreter* interp,
                                         const TClingClassInfo* derived,
                                         TClingClassInfo* base)
   : fInterp(interp), fClassInfo(0), fFirstTime(true), fDescend(false),
     fDecl(0), fIter(0), fBaseInfo(0), fOffset(0L)
{
   if (!derived->GetDecl()) {
      return;
   }
   const clang::CXXRecordDecl* CRD =
      llvm::dyn_cast<clang::CXXRecordDecl>(derived->GetDecl());
   if (!CRD) {
      // We were initialized with something that is not a class.
      // FIXME: We should prevent this from happening!
      return;
   }
   fClassInfo = derived;
   fDecl = CRD;
   fBaseInfo = new TClingClassInfo(*base);
   fIter = CRD->bases_begin();
}

TClingBaseClassInfo::TClingBaseClassInfo(const TClingBaseClassInfo& rhs)
   : fInterp(rhs.fInterp), fClassInfo(0), fFirstTime(rhs.fFirstTime),
     fDescend(rhs.fDescend), fDecl(rhs.fDecl), fIter(rhs.fIter), fBaseInfo(0),
     fIterStack(rhs.fIterStack), fOffset(rhs.fOffset)
{
   fClassInfo = new TClingClassInfo(*rhs.fClassInfo);
   fBaseInfo = new TClingClassInfo(*rhs.fBaseInfo);
}

TClingBaseClassInfo& TClingBaseClassInfo::operator=(
   const TClingBaseClassInfo& rhs)
{
   if (this != &rhs) {
      fInterp = rhs.fInterp;
      delete fClassInfo;
      fClassInfo = new TClingClassInfo(*rhs.fClassInfo);
      fFirstTime = rhs.fFirstTime;
      fDescend = rhs.fDescend;
      fDecl = rhs.fDecl;
      fIter = rhs.fIter;
      delete fBaseInfo;
      fBaseInfo = new TClingClassInfo(*rhs.fBaseInfo);
      fIterStack = rhs.fIterStack;
      fOffset = rhs.fOffset;
   }
   return *this;
}

TClingClassInfo *TClingBaseClassInfo::GetBase() const
{
   if (!IsValid()) {
      return 0;
   }
   return fBaseInfo;
}

OffsetPtrFunc_t TClingBaseClassInfo::GenerateBaseOffsetFunction(const TClingClassInfo * derivedClass, TClingClassInfo* targetClass, void* address) const
{
   // Generate a function at run-time that would calculate the offset 
   // from the parameter derived class to the parameter target class for the
   // address.

   //  Get the class or namespace name.
   string derived_class_name;
   if (derivedClass->GetType()) {
      // This is a class, struct, or union member.
      clang::QualType QTDerived(derivedClass->GetType(), 0);
      ROOT::TMetaUtils::GetFullyQualifiedTypeName(derived_class_name, QTDerived, *fInterp);
   }
   else if (const clang::NamedDecl* ND =
            dyn_cast<clang::NamedDecl>(derivedClass->GetDecl()->getDeclContext())) {
      // This is a namespace member.
      raw_string_ostream stream(derived_class_name);
      ND->getNameForDiagnostic(stream, ND->getASTContext().getPrintingPolicy(), /*Qualified=*/true);
      stream.flush();
   }
   string target_class_name;
   if (targetClass->GetType()) {
      // This is a class, struct, or union member.
      clang::QualType QTTarget(targetClass->GetType(), 0);
      ROOT::TMetaUtils::GetFullyQualifiedTypeName(target_class_name, QTTarget, *fInterp);
   }
   else if (const clang::NamedDecl* ND =
            dyn_cast<clang::NamedDecl>(targetClass->GetDecl()->getDeclContext())) {
      // This is a namespace member.
      raw_string_ostream stream(target_class_name);
      ND->getNameForDiagnostic(stream, ND->getASTContext().getPrintingPolicy(), /*Qualified=*/true);
      stream.flush();
   }
   //  Make the wrapper name.
   string wrapper_name;
   {
      ostringstream buf;
      buf << derived_class_name;
      buf << '_';
      buf << target_class_name;
      wrapper_name = buf.str();
   }
   //  Write the wrapper code.
   string wrapperFwdDecl = "long " + wrapper_name + "(void* address)";
   ostringstream buf;
   buf << wrapperFwdDecl << "{\n";
   buf << indent_string << derived_class_name << " *object = (" << derived_class_name << "*)address;";
   buf << "\n";
   buf << indent_string << target_class_name << " *target = object;";
   buf << "\n";
   buf << indent_string << "return ((long)target - (long)object);";
   buf << "\n";
   buf << "}\n";
   string wrapper = "extern \"C\" " + wrapperFwdDecl + ";\n"
                    + buf.str();

   //  Compile the wrapper code.
   const FunctionDecl* WFD = 0;
   {
      cling::Transaction* Tp = 0;
      cling::Interpreter::CompilationResult CR = fInterp->declare(wrapper, &Tp);
      if (CR != cling::Interpreter::kSuccess) {
         Error("TClingBaseClassInfo::GenerateBaseOffsetFunction", "Wrapper compile failed!");
         return 0;
      }
      for (cling::Transaction::const_iterator I = Tp->decls_begin(),
           E = Tp->decls_end(); !WFD && I != E; ++I) {
         if (I->m_Call == cling::Transaction::kCCIHandleTopLevelDecl) {
         const FunctionDecl* createWFD = dyn_cast<FunctionDecl>(*I->m_DGR.begin());
            if (createWFD && isa<TranslationUnitDecl>(createWFD->getDeclContext())) {
               DeclarationName FName = createWFD->getDeclName();
               if (const IdentifierInfo* FII = FName.getAsIdentifierInfo()) {
                  if (FII->getName() == wrapper_name) {
                     WFD = createWFD;
                  }
               }
            }
         }
      }
      if (!WFD) {
         Error("TClingBaseClassInfo::GenerateBaseOffsetFunction",
               "Wrapper compile did not return a function decl!");
         return 0;
      }
   }

   //  Get the wrapper function pointer
   //  from the ExecutionEngine (the JIT).
   const GlobalValue* GV = fInterp->getModule()->getNamedValue(wrapper_name);
   if (!GV) {
      Error("TClingBaseClassInfo::GenerateBaseOffsetFunction",
            "Wrapper function name not found in Module!");
      return 0;
   }
   ExecutionEngine* EE = fInterp->getExecutionEngine();
   OffsetPtrFunc_t f = (OffsetPtrFunc_t)EE->getPointerToGlobalIfAvailable(GV);
   if (!f) {
      //  Wrapper function not yet codegened by the JIT,
      //  force this to happen now.
      f = (OffsetPtrFunc_t)EE->getPointerToGlobal(GV);
      if (!f) {
         Error("TClingBaseClassInfo::GenerateBaseOffsetFunction", "Wrapper function has no "
               "mapping in Module after forced codegen!");
         return 0;
      }
   }

   return f;
}

bool TClingBaseClassInfo::IsValid() const
{
   return
      // inited with a valid class, and
      fClassInfo->IsValid() &&
      // the base class we are iterating over is valid, and
      fDecl &&
      // our internal iterator is currently valid, and
      fIter &&
      (fIter != llvm::dyn_cast<clang::CXXRecordDecl>(fDecl)->bases_end()) &&
      // our current base has a TClingClassInfo, and
      fBaseInfo &&
      // our current base is a valid class
      fBaseInfo->IsValid();
}

int TClingBaseClassInfo::InternalNext(int onlyDirect)
{
   // Exit early if the iterator is already invalid.
   if (!fDecl || !fIter ||
         (fIter == llvm::dyn_cast<clang::CXXRecordDecl>(fDecl)->bases_end())) {
      return 0;
   }
   // Advance to the next valid base.
   while (1) {
      // Advance the iterator.
      if (fFirstTime) {
         // The cint semantics are strange.
         fFirstTime = false;
      }
      else if (!onlyDirect && fDescend) {
         // We previously processed a base class which itself has bases,
         // now we process the bases of that base class.
         fDescend = false;
         const clang::RecordType *Ty = fIter->getType()->
                                       getAs<clang::RecordType>();
         // Note: We made sure this would work when we selected the
         //       base for processing.
         clang::CXXRecordDecl *Base = llvm::cast<clang::CXXRecordDecl>(
                                         Ty->getDecl()->getDefinition());
         clang::ASTContext &Context = Base->getASTContext();
         const clang::RecordDecl *RD = llvm::dyn_cast<clang::RecordDecl>(fDecl);
         const clang::ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
         int64_t offset = Layout.getBaseClassOffset(Base).getQuantity();
         fOffset += static_cast<long>(offset);
         fIterStack.push_back(std::make_pair(std::make_pair(fDecl, fIter),
                                             static_cast<long>(offset)));
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
         delete fBaseInfo;
         fBaseInfo = 0;
         // Iterator is now invalid.
         return 0;
      }
      // Check if current base class is a dependent type, that is, an
      // uninstantiated template class.
      const clang::TagType *Ty = fIter->getType()->getAs<clang::TagType>();
      if (!Ty) {
         // A dependent type (uninstantiated template), skip it.
         continue;
      }
      // Check if current base class has a definition.
      const clang::CXXRecordDecl *Base =
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
      delete fBaseInfo;
      clang::QualType bType = ROOT::TMetaUtils::ReSubstTemplateArg(fIter->getType(),fClassInfo->GetType());
      fBaseInfo = new TClingClassInfo(fInterp, *bType);
      // Iterator is now valid.
      return 1;
   }
}

int TClingBaseClassInfo::Next(int onlyDirect)
{
   return InternalNext(onlyDirect);
}

int TClingBaseClassInfo::Next()
{
   return Next(1);
}

long TClingBaseClassInfo::Offset(void * address) const
{
   // Compute the offset of the derived class to the base class.

   if (!IsValid()) {
      return -1;
   }
   // Offset to the class itself.
   if (fClassInfo->GetDecl() == fBaseInfo->GetDecl()) {
      return 0;
   }
   // The base class is not in the inheritance hierarchy of the class.
   if (!fClassInfo->IsBase(fBaseInfo->Name())) {
      return -1;
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
   // If the base class has no virtual inheritance.
   if (!(fBaseInfo->ClassProperty() & kClassHasVirtual)) {
      clang::ASTContext& Context = Base->getASTContext();
      const clang::RecordDecl* RD = llvm::dyn_cast<clang::RecordDecl>(fDecl);
      const clang::ASTRecordLayout& Layout = Context.getASTRecordLayout(RD);
      int64_t offset = Layout.getBaseClassOffset(Base).getQuantity();
      long clang_val = fOffset + static_cast<long>(offset);
      return clang_val;
   }
   // Virtual inheritance case
   OffsetPtrFunc_t executableFunc = fClassInfo->FindBaseOffsetFunction(fBaseInfo->GetDecl());
   if (!executableFunc) {
      // Error generated already by GenerateBaseOffsetFunction if executableFunc = 0.
      executableFunc = GenerateBaseOffsetFunction(fClassInfo, fBaseInfo, address);
      const_cast<TClingClassInfo*>(fClassInfo)->AddBaseOffsetFunction(fBaseInfo->GetDecl(), executableFunc);
   }
   if (address)
      return (*executableFunc)(address);
   
   return -1;   
}


long TClingBaseClassInfo::Property() const
{
   if (!IsValid()) {
      return 0L;
   }
   long property = 0L;
   if (fIter->isVirtual()) {
      property |= kIsVirtualBase;
   }
   if (fDecl == fClassInfo->GetDecl()) {
      property |= kIsDirectInherit;
   }
   switch (fIter->getAccessSpecifier()) {
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
         // IMPOSSIBLE
         break;
      default:
         // IMPOSSIBLE
         break;
   }
   return property;
}

long TClingBaseClassInfo::Tagnum() const
{
   if (!IsValid()) {
      return -1L;
   }
   return fBaseInfo->Tagnum();
}

const char* TClingBaseClassInfo::FullName(const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt) const
{
   if (!IsValid()) {
      return 0;
   }
   return fBaseInfo->FullName(normCtxt);
}

const char* TClingBaseClassInfo::Name() const
{
   if (!IsValid()) {
      return 0;
   }
   return fBaseInfo->Name();
}

const char* TClingBaseClassInfo::TmpltName() const
{
   if (!IsValid()) {
      return 0;
   }
   return fBaseInfo->TmpltName();
}

