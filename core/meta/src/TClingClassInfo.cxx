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
   : fInterp(interp), fFirstTime(true), fDescend(false), fDecl(0)
{
   clang::TranslationUnitDecl *TU =
      interp->getCI()->getASTContext().getTranslationUnitDecl();
   fIter = TU->decls_begin();
   InternalNext();
   fFirstTime = true;
   fDecl = 0;
}

TClingClassInfo::TClingClassInfo(cling::Interpreter *interp, const char *name)
   : fInterp(interp), fFirstTime(true), fDescend(false), fDecl(0), fTitle("")
{
   if (gDebug > 0) {
      Info("TClingClassInfo(name)", "looking up class name: %s\n", name);
   }
   const cling::LookupHelper& lh = fInterp->getLookupHelper();
   const clang::Decl *decl = lh.findScope(name);
   if (!decl) {
      if (gDebug > 0) {
         Info("TClingClassInfo(name)", "cling class not found name: %s\n",
              name);
      }
      std::string buf = TClassEdit::InsertStd(name);
      decl = lh.findScope(buf);
      if (!decl) {
         if (gDebug > 0) {
            Info("TClingClassInfo(name)", "cling class not found name: %s\n",
                 buf.c_str());
         }
      }
      else {
         if (gDebug > 0) {
            Info("TClingClassInfo(name)", "found cling class name: %s  "
                 "decl: 0x%lx\n", buf.c_str(), (long) decl);
         }
      }
   }
   else {
      if (gDebug > 0) {
         Info("TClingClassInfo(name)", "found cling class name: %s  "
              "decl: 0x%lx\n", name, (long) decl);
      }
   }
   if (decl) {
      // Position our iterator on the found decl.
      /*AdvanceToDecl(decl);*/
      fFirstTime = false;
      //fDescend = false;
      fIter = clang::DeclContext::decl_iterator();
      //fTemplateDecl = 0;
      //fSpecIter = clang::ClassTemplateDecl::spec_iterator(0);
      fDecl = const_cast<clang::Decl*>(decl);
      //fIterStack.clear();
   }
}

TClingClassInfo::TClingClassInfo(cling::Interpreter *interp,
                                 const clang::Decl *decl)
   : fInterp(interp), fFirstTime(true), fDescend(false), fDecl(0), fTitle("")
{
   if (decl) {
      // Position our iterator on the given decl.
      AdvanceToDecl(decl);
      //fFirstTime = true;
      //fDescend = false;
      //fIter = clang::DeclContext::decl_iterator();
      //fTemplateDecl = 0;
      //fSpecIter = clang::ClassTemplateDecl::spec_iterator(0);
      //fDecl = const_cast<clang::Decl*>(decl);
      //fIterStack.clear();
   }
   else {
      // FIXME: Maybe initialize iterator to global namespace?
      fDecl = 0;
   }
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
      fInterp->evaluate(os.str());
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
         fInterp->evaluate(os.str());
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
      fInterp->evaluate(os.str());
   if (err != cling::Interpreter::kSuccess) {
      return;
   }
   return;
}

TClingMethodInfo TClingClassInfo::GetMethod(const char *fname,
      const char *arg, long *poffset, MatchMode mode /*= ConversionMatch*/,
      InheritanceMode imode /*= WithInheritance*/) const
{
   if (!IsValid()) {
      TClingMethodInfo tmi(fInterp);
      return tmi;
   }
   const cling::LookupHelper& lh = fInterp->getLookupHelper();
   const clang::FunctionDecl *FD = lh.findFunctionArgs(fDecl, fname, arg);
   if (poffset) {
      *poffset = 0L;
   }
   TClingMethodInfo tmi(fInterp);
   tmi.Init(FD);
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

bool TClingClassInfo::HasDefaultConstructor() const
{
   // Note: This is a ROOT special!  It actually test for the root ioctor.
   if (!IsValid()) {
      return false;
   }
   // FIXME: Look for root ioctor when we have function lookup, and
   //        rootcling can tell us what the name of the ioctor is.
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
      clang::DeclContext *DC = llvm::cast<clang::DeclContext>(fDecl);
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
   if (gDebug > 0) {
      Info("TClingClassInfo::Init(name)", "looking up class: %s\n", name);
   }
   fFirstTime = true;
   fDescend = false;
   fIter = clang::DeclContext::decl_iterator();
   fDecl = 0;
   fIterStack.clear();
   const cling::LookupHelper& lh = fInterp->getLookupHelper();
   const clang::Decl *decl = lh.findScope(name);
   if (!decl) {
      if (gDebug > 0) {
         Info("TClingClassInfo::Init(name)", "cling class not found "
              "name: %s\n", name);
      }
      std::string buf = TClassEdit::InsertStd(name);
      decl = lh.findScope(buf);
      if (!decl) {
         if (gDebug > 0) {
            Info("TClingClassInfo::Init(name)", "cling class not found "
                 "name: %s\n", buf.c_str());
         }
      }
      else {
         if (gDebug > 0) {
            Info("TClingClassInfo::Init(name)", "found cling class "
                 "name: %s  decl: 0x%lx\n", buf.c_str(), (long) decl);
         }
      }
   }
   else {
      if (gDebug > 0) {
         Info("TClingClassInfo::Init(name)", "found cling class "
              "name: %s  decl: 0x%lx\n", name, (long) decl);
      }
   }
   if (decl) {
      // Position our iterator on the given decl.
      AdvanceToDecl(decl);
      //fFirstTime = true;
      //fDescend = false;
      //fIter = clang::DeclContext::decl_iterator();
      //fTemplateDecl = 0;
      //fSpecIter = clang::ClassTemplateDecl::spec_iterator(0);
      //fDecl = const_cast<clang::Decl*>(decl);
      //fIterStack.clear();
   }
}

void TClingClassInfo::Init(int tagnum)
{
   return;
#if 0
   if (gDebug > 0) {
      Info("TClingClassInfo::Init(tagnum)", "looking up tagnum: %d\n", tagnum);
   }
   if (!gAllowCint) {
      delete fClassInfo;
      fClassInfo = new G__ClassInfo;
      fDecl = 0;
      return;
   }
   fFirstTime = true;
   fDescend = false;
   fIter = clang::DeclContext::decl_iterator();
   fDecl = 0;
   fIterStack.clear();
   fClassInfo->Init(tagnum);
   if (!fClassInfo->IsValid()) {
      if (gDebug > 0) {
         fprintf(stderr, "TClingClassInfo::Init(tagnum): could not find cint "
                 "class for tagnum: %d\n", tagnum);
      }
      return;
   }
   if (gAllowClang) {
      const char *name = fClassInfo->Fullname();
      if (gDebug > 0) {
         fprintf(stderr, "TClingClassInfo::Init(tagnum): found cint class "
                 "name: %s  tagnum: %d\n", name, tagnum);
      }
      if (!name || (name[0] == '\0')) {
         // No name, or name is blank, could be anonymous
         // class/struct/union or enum.  Cint does not give
         // us enough information to find the same decl in clang.
         return;
      }
      const clang::Decl *decl = fInterp->lookupScope(name);
      if (!decl) {
         if (gDebug > 0) {
            fprintf(stderr,
                    "TClingClassInfo::Init(tagnum): cling class not found "
                    "name: %s  tagnum: %d\n", name, tagnum);
         }
         std::string buf = TClassEdit::InsertStd(name);
         decl = const_cast<clang::Decl *>(fInterp->lookupScope(buf));
         if (!decl) {
            if (gDebug > 0) {
               fprintf(stderr,
                       "TClingClassInfo::Init(tagnum): cling class not found "
                       "name: %s  tagnum: %d\n", buf.c_str(), tagnum);
            }
         }
         else {
            if (gDebug > 0) {
               fprintf(stderr, "TClingClassInfo::Init(tagnum): "
                       "found cling class name: %s  decl: 0x%lx\n",
                       buf.c_str(), (long) decl);
            }
         }
      }
      else {
         if (gDebug > 0) {
            fprintf(stderr, "TClingClassInfo::Init(tagnum): found cling class "
                    "name: %s  decl: 0x%lx\n", name, (long) decl);
         }
      }
      if (decl) {
         // Position our iterator on the given decl.
         AdvanceToDecl(decl);
         //fFirstTime = true;
         //fDescend = false;
         //fIter = clang::DeclContext::decl_iterator();
         //fTemplateDecl = 0;
         //fSpecIter = clang::ClassTemplateDecl::spec_iterator(0);
         //fDecl = const_cast<clang::Decl*>(decl);
         //fIterStack.clear();
      }
   }
#endif // 0
   //--
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
   if (!IsValid()) {
      return false;
   }
   return GetMethod(method, proto, offset).IsValid();
}

int TClingClassInfo::AdvanceToDecl(const clang::Decl *target_decl)
{
   const clang::TranslationUnitDecl *TU = target_decl->getTranslationUnitDecl();
   const clang::DeclContext *DC = llvm::cast<clang::DeclContext>(TU);
   fFirstTime = true;
   fDescend = false;
   fIter = DC->decls_begin();
   fDecl = 0;
   fIterStack.clear();
   while (InternalNext()) {
      if (fDecl == target_decl) {
         return 1;
      }
   }
   return 0;
}

int TClingClassInfo::InternalNext()
{
   if (!*fIter) {
      // Iterator is already invalid.
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
   if (!IsValid()) {
      return 0;
   }
   std::ostringstream os;
   os << "new " << Name() << ";";
   cling::StoredValueRef val;
   cling::Interpreter::CompilationResult err =
      fInterp->evaluate(os.str(), &val);
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
   if (!IsValid()) {
      return 0;
   }
   std::ostringstream os;
   os << "new " << Name() << "[" << n << "];";
   cling::StoredValueRef val;
   cling::Interpreter::CompilationResult err =
      fInterp->evaluate(os.str(), &val);
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
   if (!IsValid()) {
      return 0;
   }
   std::ostringstream os;
   os << "new ((void*)" << reinterpret_cast<unsigned long>(arena) << ") "
      << Name() << "[" << n << "];";
   cling::StoredValueRef val;
   cling::Interpreter::CompilationResult err =
      fInterp->evaluate(os.str(), &val);
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
   if (!IsValid()) {
      return 0;
   }
   std::ostringstream os;
   os << "new ((void*)" << reinterpret_cast<unsigned long>(arena) << ") "
      << Name() << ";";
   cling::StoredValueRef val;
   cling::Interpreter::CompilationResult err =
      fInterp->evaluate(os.str(), &val);
   if (err != cling::Interpreter::kSuccess) {
      return 0;
   }
   // The ref-counted pointer will get destructed by StoredValueRef,
   // but not the allocation! I.e. the following is fine:
   return llvm::GVTOP(val.get().value);
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
   // FIXME: Implement this when rootcling provides the information.
   return 0;
}

const char *TClingClassInfo::FullName() const
{
   if (!IsValid()) {
      return 0;
   }
   // Note: This *must* be static because we are returning a pointer inside it!
   static std::string buf;
   buf.clear();
   clang::PrintingPolicy Policy(fDecl->getASTContext().getPrintingPolicy());
   llvm::dyn_cast<clang::NamedDecl>(fDecl)->
   getNameForDiagnostic(buf, Policy, /*Qualified=*/true);
   return buf.c_str();
}

const char *TClingClassInfo::Name() const
{
   if (!IsValid()) {
      return 0;
   }
   // Note: This *must* be static because we are returning a pointer inside it!
   static std::string buf;
   buf.clear();
   clang::PrintingPolicy Policy(fDecl->getASTContext().getPrintingPolicy());
   llvm::dyn_cast<clang::NamedDecl>(fDecl)->
   getNameForDiagnostic(buf, Policy, /*Qualified=*/false);
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
      if ( (TD = ROOT::TMetaUtils::GetAnnotatedRedeclarable(TD)) ) 
         if (AnnotateAttr *A = TD->getAttr<AnnotateAttr>())
            fTitle = A->getAnnotation().str();
   }
   else
      // Try to get the comment from the header file if present
      fTitle = ROOT::TMetaUtils::GetComment(*GetDecl()).str();

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

