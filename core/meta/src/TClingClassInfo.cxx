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
// TClingClassInfo                                                      //
//                                                                      //
// Emulation of the CINT ClassInfo class.                               //
//                                                                      //
// The CINT C++ interpreter provides an interface to metadata about     //
// a class through the ClassInfo class.  This class provides the same   //
// functionality, using an interface as close as possible to ClassInfo  //
// but the class metadata comes from the Clang C++ compiler, not CINT.  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClingClassInfo.h"

#include "TClassEdit.h"
#include "TClingBaseClassInfo.h"
#include "TClingMethodInfo.h"
#include "Property.h"
#include "TClingProperty.h"
#include "TClingTypeInfo.h"
#include "TError.h"
#include "TMetaUtils.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/LookupHelper.h"
#include "cling/Interpreter/StoredValueRef.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/Frontend/CompilerInstance.h"
 
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>
#include <string>

using namespace clang;

TClingClassInfo::TClingClassInfo(cling::Interpreter *interp)
   : fInterp(interp), fFirstTime(true), fDescend(false), fDecl(0), fType(0),
     fNMethods(0)
{
   clang::TranslationUnitDecl *TU =
      interp->getCI()->getASTContext().getTranslationUnitDecl();
   fIter = TU->decls_begin();
   InternalNext();
   fFirstTime = true;
   fDecl = 0;
   fType = 0;
}

TClingClassInfo::TClingClassInfo(cling::Interpreter *interp, const char *name)
   : fInterp(interp), fFirstTime(true), fDescend(false), fDecl(0), fType(0),
     fTitle(""), fNMethods(0)
{
   const cling::LookupHelper& lh = fInterp->getLookupHelper();
   const clang::Type *type = 0;
   const clang::Decl *decl = lh.findScope(name,&type);
   if (!decl) {
      std::string buf = TClassEdit::InsertStd(name);
      decl = lh.findScope(buf,&type);
   }
   fDecl = decl;
   fType = type;
}

TClingClassInfo::TClingClassInfo(cling::Interpreter *interp,
                                 const clang::Type &tag)
   : fInterp(interp), fFirstTime(true), fDescend(false), fDecl(0), fType(0), 
     fTitle(""), fNMethods(0)
{
   Init(tag);
}

long TClingClassInfo::ClassProperty() const
{
   if (!IsValid()) {
      return 0L;
   }
   const clang::RecordDecl *RD = llvm::dyn_cast<clang::RecordDecl>(fDecl);
   if (!RD) {
      // We are an enum or namespace.
      // The cint interface always returns 0L for these guys.
      return 0L;
   }
   if (RD->isUnion()) {
      // The cint interface always returns 0L for these guys.
      return 0L;
   }
   // We now have a class or a struct.
   const clang::CXXRecordDecl *CRD =
      llvm::dyn_cast<clang::CXXRecordDecl>(fDecl);
   long property = 0L;
   property |= G__CLS_VALID;
   if (CRD->isAbstract()) {
      property |= G__CLS_ISABSTRACT;
   }
   if (CRD->hasUserDeclaredConstructor()) {
      property |= G__CLS_HASEXPLICITCTOR;
   }
   if (
      !CRD->hasUserDeclaredConstructor() &&
      !CRD->hasTrivialDefaultConstructor()
   ) {
      property |= G__CLS_HASIMPLICITCTOR;
   }
   if (
      CRD->hasUserProvidedDefaultConstructor() ||
      !CRD->hasTrivialDefaultConstructor()
   ) {
      property |= G__CLS_HASDEFAULTCTOR;
   }
   if (CRD->hasUserDeclaredDestructor()) {
      property |= G__CLS_HASEXPLICITDTOR;
   }
   else if (!CRD->hasTrivialDestructor()) {
      property |= G__CLS_HASIMPLICITDTOR;
   }
   if (CRD->hasUserDeclaredCopyAssignment()) {
      property |= G__CLS_HASASSIGNOPR;
   }
   if (CRD->isPolymorphic()) {
      property |= G__CLS_HASVIRTUAL;
   }
   return property;
}

void TClingClassInfo::Delete(void *arena) const
{
   // Invoke operator delete on a pointer to an object
   // of this class type.
   if (!IsValid()) {
      return;
   }
   std::ostringstream os;
   os << "delete (" << Name() << "*)"
      << reinterpret_cast<unsigned long>(arena) << ";";
   cling::Interpreter::CompilationResult err =
      fInterp->execute(os.str());
   if (err != cling::Interpreter::kSuccess) {
      return;
   }
   return;
}

void TClingClassInfo::DeleteArray(void *arena, bool dtorOnly) const
{
   // Invoke operator delete[] on a pointer to an array object
   // of this class type.
   if (!IsValid()) {
      return;
   }
   if (dtorOnly) {
      // There is no syntax in C++ for invoking the placement delete array
      // operator, so we have to placement delete each element by hand.
      // Unfortunately we do not know how many elements to delete.
      Error("DeleteArray", "Placement delete of an array is unsupported!\n");
   }
   else {
      std::ostringstream os;
      os << "delete[] (" << Name() << "*)"
         << reinterpret_cast<unsigned long>(arena) << ";";
      cling::Interpreter::CompilationResult err =
         fInterp->execute(os.str());
      if (err != cling::Interpreter::kSuccess) {
         return;
      }
   }
   return;
}

void TClingClassInfo::Destruct(void *arena) const
{
   // Invoke placement operator delete on a pointer to an array object
   // of this class type.
   if (!IsValid()) {
      return;
   }
   const char *name = Name();
   std::ostringstream os;
   os << "((" << name << "*)" << reinterpret_cast<unsigned long>(arena)
      << ")->" << name << "::~" << name << "();";
   cling::Interpreter::CompilationResult err =
      fInterp->execute(os.str());
   if (err != cling::Interpreter::kSuccess) {
      return;
   }
   return;
}

TClingMethodInfo TClingClassInfo::GetMethod(const char *fname,
      const char *proto, long *poffset, MatchMode mode /*= ConversionMatch*/,
      InheritanceMode imode /*= WithInheritance*/) const
{
   if (poffset) {
      *poffset = 0L;
   }
   if (!IsValid()) {
      TClingMethodInfo tmi(fInterp);
      return tmi;
   }
   const cling::LookupHelper& lh = fInterp->getLookupHelper();
   const clang::FunctionDecl *fd = lh.findFunctionProto(fDecl, fname, proto);
   if (!fd) {
      // Function not found.
      TClingMethodInfo tmi(fInterp);
      return tmi;
   }
   if (poffset) {
     // We have been asked to return a this pointer adjustment.
     if (const clang::CXXMethodDecl *md =
           llvm::dyn_cast<clang::CXXMethodDecl>(fd)) {
        // This is a class member function.
        *poffset = GetOffset(md);
     }
   }
   TClingMethodInfo tmi(fInterp);
   tmi.Init(fd);
   return tmi;
}

int TClingClassInfo::GetMethodNArg(const char *method, const char *proto) const
{
   // Note: Used only by TQObject.cxx:170 and only for interpreted classes.
   if (!IsValid()) {
      return -1;
   }
   int clang_val = -1;
   const clang::FunctionDecl *decl =
     fInterp->getLookupHelper().findFunctionProto(fDecl, method, proto);
   if (decl) {
      unsigned num_params = decl->getNumParams();
      clang_val = static_cast<int>(num_params);
   }
   return clang_val;
}

long TClingClassInfo::GetOffset(const clang::CXXMethodDecl* md) const
{
   long offset = 0L;
   const clang::CXXRecordDecl* definer = md->getParent();
   const clang::CXXRecordDecl* accessor =
      llvm::cast<clang::CXXRecordDecl>(fDecl);
   if (definer != accessor) {
      // This function may not be accessible using a pointer
      // to the declaring class, get the adjustment necessary
      // to convert that to a pointer to the defining class.
      TClingBaseClassInfo bi(fInterp, const_cast<TClingClassInfo*>(this));
      while (bi.Next(0)) {
         TClingClassInfo* bci = bi.GetBase();
         if (bci->GetDecl() == definer) {
            // We have found the right base class, now get the
            // necessary adjustment.
            offset = bi.Offset();
            break;
         }
      }
   }
   return offset;
}

bool TClingClassInfo::HasDefaultConstructor() const
{
   // Return true if there a public constructor taking no argument
   // (including a constructor that has default for all its argument).

   // Note: This is could enhanced to also know about the ROOT ioctor
   // but this was not the case in CINT.

   if (!IsValid()) {
      return false;
   }
   
   const clang::CXXRecordDecl *CRD =
      llvm::dyn_cast<clang::CXXRecordDecl>(fDecl);

   if (!CRD) return true; 

   for(clang::CXXRecordDecl::ctor_iterator iter = CRD->ctor_begin(), end = CRD->ctor_end();
       iter != end;
       ++iter)
   {
      if (iter->getAccess() == clang::AS_public) {
         // We can reach this constructor.
         if (iter->getNumParams() == 0) {
            return true;
         }
         // Most likely just this test is needed.
         if (iter->getMinRequiredArguments() == 0) {
            return true;
         }
      }
   }

   return false;
}

bool TClingClassInfo::HasMethod(const char *name) const
{
   if (!IsValid()) {
      return false;
   }
   bool found = false;
   std::string given_name(name);
   if (!llvm::isa<clang::EnumDecl>(fDecl)) {
      // We are a class, struct, union, namespace, or translation unit.
      clang::DeclContext *DC = const_cast<clang::DeclContext*>(llvm::cast<clang::DeclContext>(fDecl));
      llvm::SmallVector<clang::DeclContext *, 2> fContexts;
      DC->collectAllContexts(fContexts);
      for (unsigned I = 0; !found && (I < fContexts.size()); ++I) {
         DC = fContexts[I];
         for (clang::DeclContext::decl_iterator iter = DC->decls_begin();
               *iter; ++iter) {
            if (const clang::FunctionDecl *FD =
                     llvm::dyn_cast<clang::FunctionDecl>(*iter)) {
               if (FD->getNameAsString() == given_name) {
                  found = true;
                  break;
               }
            }
         }
      }
   }
   return found;
}

void TClingClassInfo::Init(const char *name)
{
   fFirstTime = true;
   fDescend = false;
   fIter = clang::DeclContext::decl_iterator();
   fDecl = 0;
   fType = 0;
   fIterStack.clear();
   const cling::LookupHelper& lh = fInterp->getLookupHelper();
   const clang::Decl *decl = lh.findScope(name);
   if (!decl) {
      std::string buf = TClassEdit::InsertStd(name);
      decl = lh.findScope(buf);
   }
   fDecl = decl;
   if (decl) {
      const clang::RecordDecl *rdecl = llvm::dyn_cast<clang::RecordDecl>(decl);
      if (rdecl) fType = rdecl->getASTContext().getRecordType(rdecl)->getAs<clang::RecordType>();
   } 
}

void TClingClassInfo::Init(int tagnum)
{
   Fatal("TClingClassInfo::Init(tagnum)","Should no longer be called");
   return;
}

void TClingClassInfo::Init(const clang::Type &tag)
{
   fType = &tag;
   fDecl = fType->getAsCXXRecordDecl();
   if (!fDecl) {
      clang::QualType qType(fType,0);
      static clang::PrintingPolicy
         printPol(fInterp->getCI()->getLangOpts());
      printPol.SuppressScope = false;
      Error("TClingClassInfo::Init(const clang::Type&)","The given type %s does not point to a CXXRecordDecl",
            qType.getAsString(printPol).c_str());
   }
}

bool TClingClassInfo::IsBase(const char *name) const
{
   if (!IsValid()) {
      return false;
   }
   TClingClassInfo base(fInterp, name);
   if (!base.IsValid()) {
      return false;
   }
   const clang::CXXRecordDecl *CRD =
      llvm::dyn_cast<clang::CXXRecordDecl>(fDecl);
   if (!CRD) {
      // We are an enum, namespace, or translation unit,
      // we cannot be the base of anything.
      return false;
   }
   const clang::CXXRecordDecl *baseCRD =
      llvm::dyn_cast<clang::CXXRecordDecl>(base.GetDecl());
   return CRD->isDerivedFrom(baseCRD);
}

bool TClingClassInfo::IsEnum(cling::Interpreter *interp, const char *name)
{
   // Note: This is a static member function.
   TClingClassInfo info(interp, name);
   if (info.IsValid() && (info.Property() & G__BIT_ISENUM)) {
      return true;
   }
   return false;
}

bool TClingClassInfo::IsLoaded() const
{
   if (!IsValid()) {
      return false;
   }
   // All clang classes are considered loaded.
   return true;
}

bool TClingClassInfo::IsValid() const
{
   return fDecl;
}

bool TClingClassInfo::IsValidMethod(const char *method, const char *proto,
                                    long *offset) const
{
   // Check if the method with the given prototype exist.
   if (!IsValid()) {
      return false;
   }
   if (offset) {
      *offset = 0L; // humm suspicious.
   }
   const clang::FunctionDecl *decl =
      fInterp->getLookupHelper().findFunctionProto(fDecl, method, proto);
   return (decl != 0);
}

int TClingClassInfo::InternalNext()
{
   if (!*fIter) {
      // Iterator is already invalid.
      if (fFirstTime && fDecl) {
         std::string buf;
         clang::PrintingPolicy Policy(fDecl->getASTContext().getPrintingPolicy());
         llvm::dyn_cast<clang::NamedDecl>(fDecl)->getNameForDiagnostic(buf, Policy, /*Qualified=*/false);         
         Error("TClingClassInfo::InternalNext","Next called but iteration not prepared for %s!",buf.c_str());
      }
      return 0;
   }
   while (true) {
      // Advance to next usable decl, or return if there is no next usable decl.
      if (fFirstTime) {
         // The cint semantics are strange.
         fFirstTime = false;
      }
      else {
         // Advance the iterator one decl, descending into the current decl
         // context if necessary.
         if (!fDescend) {
            // Do not need to scan the decl context of the current decl,
            // move on to the next decl.
            ++fIter;
         }
         else {
            // Descend into the decl context of the current decl.
            fDescend = false;
            //fprintf(stderr,
            //   "TClingClassInfo::InternalNext:  "
            //   "pushing ...\n");
            fIterStack.push_back(fIter);
            clang::DeclContext *DC = llvm::cast<clang::DeclContext>(*fIter);
            fIter = DC->decls_begin();
         }
         // Fix it if we went past the end.
         while (!*fIter && fIterStack.size()) {
            //fprintf(stderr,
            //   "TClingClassInfo::InternalNext:  "
            //   "popping ...\n");
            fIter = fIterStack.back();
            fIterStack.pop_back();
            ++fIter;
         }
         // Check for final termination.
         if (!*fIter) {
            // We have reached the end of the translation unit, all done.
            fDecl = 0;
            fType = 0;
            return 0;
         }
      }
      // Return if this decl is a class, struct, union, enum, or namespace.
      clang::Decl::Kind DK = fIter->getKind();
      if ((DK == clang::Decl::Namespace) || (DK == clang::Decl::Enum) ||
            (DK == clang::Decl::CXXRecord) ||
            (DK == clang::Decl::ClassTemplateSpecialization)) {
         const clang::TagDecl *TD = llvm::dyn_cast<clang::TagDecl>(*fIter);
         if (TD && !TD->isCompleteDefinition()) {
            // For classes and enums, stop only on definitions.
            continue;
         }
         if (DK == clang::Decl::Namespace) {
            // For namespaces, stop only on the first definition.
            if (!fIter->isCanonicalDecl()) {
               // Not the first definition.
               fDescend = true;
               continue;
            }
         }
         if (DK != clang::Decl::Enum) {
            // We do not descend into enums.
            clang::DeclContext *DC = llvm::cast<clang::DeclContext>(*fIter);
            if (*DC->decls_begin()) {
               // Next iteration will begin scanning the decl context
               // contained by this decl.
               fDescend = true;
            }
         }
         // Iterator is now valid.
         fDecl = *fIter;
         fType = 0;
         if (fDecl) {
            const clang::RecordDecl *rdecl = llvm::dyn_cast<clang::RecordDecl>(fDecl);
            if (rdecl) fType = rdecl->getASTContext().getRecordType(rdecl).getTypePtr();
         }
         return 1;
      }
   }
}

int TClingClassInfo::Next()
{
   return InternalNext();
}

void *TClingClassInfo::New() const
{
   // Invoke a new expression to use the class constructor
   // that takes no arguments to create an object of this class type.

   if (!HasDefaultConstructor()) {
      return 0;
   }
   std::ostringstream os;
   os << "new " << Name() << ";";
   cling::StoredValueRef val;
   cling::Interpreter::CompilationResult err =
      fInterp->evaluate(os.str(), val);
   if (err != cling::Interpreter::kSuccess) {
      return 0;
   }
   // The ref-counted pointer will get destructed by StoredValueRef,
   // but not the allocation! I.e. the following is fine:
   return llvm::GVTOP(val.get().value);
}

void *TClingClassInfo::New(int n) const
{
   // Invoke a new expression to use the class constructor
   // that takes no arguments to create an array object
   // of this class type.
   if (!HasDefaultConstructor()) {
      return 0;
   }
   std::ostringstream os;
   os << "new " << Name() << "[" << n << "];";
   cling::StoredValueRef val;
   cling::Interpreter::CompilationResult err =
      fInterp->evaluate(os.str(), val);
   if (err != cling::Interpreter::kSuccess) {
      return 0;
   }
   // The ref-counted pointer will get destructed by StoredValueRef,
   // but not the allocation! I.e. the following is fine:
   return llvm::GVTOP(val.get().value);
}

void *TClingClassInfo::New(int n, void *arena) const
{
   // Invoke a placement new expression to use the class
   // constructor that takes no arguments to create an
   // array of objects of this class type in the given
   // memory arena.
   if (!HasDefaultConstructor()) {
      return 0;
   }
   std::ostringstream os;
   os << "new ((void*)" << reinterpret_cast<unsigned long>(arena) << ") "
      << Name() << "[" << n << "];";
   cling::StoredValueRef val;
   cling::Interpreter::CompilationResult err =
      fInterp->evaluate(os.str(), val);
   if (err != cling::Interpreter::kSuccess) {
      return 0;
   }
   // The ref-counted pointer will get destructed by StoredValueRef,
   // but not the allocation! I.e. the following is fine:
   return llvm::GVTOP(val.get().value);
}

void *TClingClassInfo::New(void *arena) const
{
   // Invoke a placement new expression to use the class
   // constructor that takes no arguments to create an
   // object of this class type in the given memory arena.
   if (!HasDefaultConstructor()) {
      return 0;
   }
   std::ostringstream os;
   os << "new ((void*)" << reinterpret_cast<unsigned long>(arena) << ") "
      << Name() << ";";
   cling::StoredValueRef val;
   cling::Interpreter::CompilationResult err =
      fInterp->evaluate(os.str(), val);
   if (err != cling::Interpreter::kSuccess) {
      return 0;
   }
   // The ref-counted pointer will get destructed by StoredValueRef,
   // but not the allocation! I.e. the following is fine:
   return llvm::GVTOP(val.get().value);
}

int TClingClassInfo::NMethods() const
{
   // Return the number of methods
   fNMethods = 0;
   clang::DeclContext *DC = const_cast<clang::DeclContext*>(llvm::cast<clang::DeclContext>(fDecl));
   llvm::SmallVector<clang::DeclContext *, 2> contexts;
   DC->collectAllContexts(contexts);

   bool noUpdate = fLastDeclForNMethods.size() == contexts.size();
   for (unsigned I = 0; noUpdate && I < contexts.size(); ++I) {
      noUpdate &= (fLastDeclForNMethods[I] && !fLastDeclForNMethods[I]->getNextDeclInContext());
   }
   if (noUpdate)
      return fNMethods;

   // We have a new decl; update the method count.
   for (unsigned I = 0; I < contexts.size(); ++I) {
      DC = contexts[I];
      clang::Decl* lastDecl = 0;
      for (clang::DeclContext::decl_iterator iter = DC->decls_begin();
           *iter; ++iter) {
         lastDecl = *iter;
         if (llvm::isa<clang::FunctionDecl>(lastDecl)) {
            ++fNMethods;
         }
      }
      fLastDeclForNMethods[I] = lastDecl;
   }
   return fNMethods;
}

long TClingClassInfo::Property() const
{
   if (!IsValid()) {
      return 0L;
   }
   long property = 0L;
   property |= G__BIT_ISCPPCOMPILED;
   clang::Decl::Kind DK = fDecl->getKind();
   if ((DK == clang::Decl::Namespace) || (DK == clang::Decl::TranslationUnit)) {
      property |= G__BIT_ISNAMESPACE;
      return property;
   }
   // Note: Now we have class, enum, struct, union only.
   const clang::TagDecl *TD = llvm::dyn_cast<clang::TagDecl>(fDecl);
   if (!TD) {
      return 0L;
   }
   if (TD->isEnum()) {
      property |= G__BIT_ISENUM;
      return property;
   }
   // Note: Now we have class, struct, union only.
   const clang::CXXRecordDecl *CRD =
      llvm::dyn_cast<clang::CXXRecordDecl>(fDecl);
   if (CRD->isClass()) {
      property |= G__BIT_ISCLASS;
   }
   else if (CRD->isStruct()) {
      property |= G__BIT_ISSTRUCT;
   }
   else if (CRD->isUnion()) {
      property |= G__BIT_ISUNION;
   }
   if (CRD->isAbstract()) {
      property |= G__BIT_ISABSTRACT;
   }
   return property;
}

int TClingClassInfo::RootFlag() const
{
   if (!IsValid()) {
      return 0;
   }
   // FIXME: Implement this when rootcling provides the value.
   return 0;
}

int TClingClassInfo::Size() const
{
   if (!IsValid()) {
      return -1;
   }
   clang::Decl::Kind DK = fDecl->getKind();
   if (DK == clang::Decl::Namespace) {
      // Namespaces are special for cint.
      return 1;
   }
   else if (DK == clang::Decl::Enum) {
      // Enums are special for cint.
      return 0;
   }
   const clang::RecordDecl *RD = llvm::dyn_cast<clang::RecordDecl>(fDecl);
   if (!RD) {
      // Should not happen.
      return -1;
   }
   if (!RD->getDefinition()) {
      // Forward-declared class.
      return 0;
   }
   clang::ASTContext &Context = fDecl->getASTContext();
   const clang::ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
   int64_t size = Layout.getSize().getQuantity();
   int clang_size = static_cast<int>(size);
   return clang_size;
}

long TClingClassInfo::Tagnum() const
{
   // Note: This *must* return a *cint* tagnum for now.
   if (!IsValid()) {
      return -1L;
   }
   return reinterpret_cast<long>(fDecl);
}

const char *TClingClassInfo::FileName() const
{
   if (!IsValid()) {
      return 0;
   }
   static std::string buf;
   buf = ROOT::TMetaUtils::GetFileName(GetDecl());
   return buf.c_str();
}

const char *TClingClassInfo::FullName(const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt) const
{
   // Return QualifiedName.
   
   if (!IsValid()) {
      return 0;
   }
   // Note: This *must* be static because we are returning a pointer inside it!
   static std::string buf;
   buf.clear();
   if (fType) {
      clang::QualType type(fType,0);
      ROOT::TMetaUtils::GetNormalizedName(buf, type, *fInterp, normCtxt);
   } else {
      clang::PrintingPolicy Policy(fDecl->getASTContext().getPrintingPolicy());
      llvm::dyn_cast<clang::NamedDecl>(fDecl)->getNameForDiagnostic(buf, Policy, /*Qualified=*/true);
   }
   return buf.c_str();
}

const char *TClingClassInfo::Name() const
{
   // Return unqualified name.

   if (!IsValid()) {
      return 0;
   }
   // Note: This *must* be static because we are returning a pointer inside it!
   static std::string buf;
   buf.clear();
   clang::PrintingPolicy Policy(fDecl->getASTContext().getPrintingPolicy());
   llvm::dyn_cast<clang::NamedDecl>(fDecl)->getNameForDiagnostic(buf, Policy, /*Qualified=*/false);
   return buf.c_str();
}

const char *TClingClassInfo::Title()
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
   if (const TagDecl *TD = llvm::dyn_cast<TagDecl>(GetDecl())) {
      if ( (TD = ROOT::TMetaUtils::GetAnnotatedRedeclarable(TD)) ) {
         if (AnnotateAttr *A = TD->getAttr<AnnotateAttr>()) {
            fTitle = A->getAnnotation().str();
            return fTitle.c_str();
         }
      }
   }

   // Try to get the comment from the header file if present
   const clang::CXXRecordDecl *CRD =
      llvm::dyn_cast<clang::CXXRecordDecl>(GetDecl());
   if (CRD) 
      fTitle = ROOT::TMetaUtils::GetClassComment(*CRD,0,*fInterp).str();

   return fTitle.c_str();
}

const char *TClingClassInfo::TmpltName() const
{
   if (!IsValid()) {
      return 0;
   }
   // Note: This *must* be static because we are returning a pointer inside it!
   static std::string buf;
   buf.clear();
   // Note: This does *not* include the template arguments!
   buf = llvm::dyn_cast<clang::NamedDecl>(fDecl)->getNameAsString();
   return buf.c_str();
}
