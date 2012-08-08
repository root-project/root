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

TClingClassInfo::~TClingClassInfo()
{
   delete fClassInfo;
   fClassInfo = 0;
   fInterp = 0;
   //fFirstTime = true;
   //fDescend = false;
   //fIter = clang::DeclContext::decl_iterator();
   fDecl = 0;
   fIterStack.clear();
}

// NOT IMPLEMENTED
//TClingClassInfo::TClingClassInfo()
//   : fClassInfo(0), fInterp(0), fFirstTime(true), fDescend(false),
//     fDecl(0)
//{
//}

TClingClassInfo::TClingClassInfo(const TClingClassInfo& rhs)
   : fClassInfo(0), fInterp(rhs.fInterp), fFirstTime(rhs.fFirstTime),
     fDescend(rhs.fDescend), fIter(rhs.fIter), fDecl(rhs.fDecl),
     fIterStack(rhs.fIterStack)
{
   fClassInfo = new G__ClassInfo(*rhs.fClassInfo);
}

TClingClassInfo& TClingClassInfo::operator=(const TClingClassInfo& rhs)
{
   if (this != &rhs) {
      delete fClassInfo;
      fClassInfo = new G__ClassInfo(*rhs.fClassInfo);
      fInterp = rhs.fInterp;
      fFirstTime = rhs.fFirstTime;
      fDescend = rhs.fDescend;
      fIter = rhs.fIter;
      fDecl = rhs.fDecl;
      fIterStack.clear();
      fIterStack = rhs.fIterStack;
   }
   return *this;
}

TClingClassInfo::TClingClassInfo(cling::Interpreter* interp)
   : fClassInfo(0), fInterp(interp), fFirstTime(true), fDescend(false),
     fDecl(0)
{
   fClassInfo = new G__ClassInfo();
   clang::TranslationUnitDecl* TU =
      interp->getCI()->getASTContext().getTranslationUnitDecl();
   fIter = TU->decls_begin();
   InternalNext();
   fFirstTime = true;
   fDecl = 0;
}

TClingClassInfo::TClingClassInfo(cling::Interpreter* interp, const char* name)
   : fClassInfo(0), fInterp(interp), fFirstTime(true), fDescend(false),
     fDecl(0)
{
   if (gDebug > 0) {
      fprintf(stderr,
              "TClingClassInfo(name): looking up class name: %s\n", name);
   }
   if (gAllowCint) {
      fClassInfo = new G__ClassInfo(name);
      if (gDebug > 0) {
         if (!fClassInfo->IsValid()) {
            fprintf(stderr,
                    "TClingClassInfo(name): could not find cint class for name: %s\n",
                    name);
         }
         else {
            fprintf(stderr,
                    "TClingClassInfo(name): found cint class for name: %s  "
                    "tagnum: %d\n", name, fClassInfo->Tagnum());
         }
      }
   }
   else {
      fClassInfo = new G__ClassInfo;
   }
   if (gAllowClang) {
      const clang::Decl* decl = fInterp->lookupScope(name);
      if (!decl) {
         if (gDebug > 0) {
            fprintf(stderr, "TClingClassInfo(name): cling class not found "
                    "name: %s\n", name);
         }
         std::string buf = TClassEdit::InsertStd(name);
         decl = fInterp->lookupScope(buf);
         if (!decl) {
            if (gDebug > 0) {
               fprintf(stderr, "TClingClassInfo(name): cling class not found "
                       "name: %s\n", buf.c_str());
            }
         }
         else {
            if (gDebug > 0) {
               fprintf(stderr,
                       "TClingClassInfo(name): found cling class name: %s  "
                       "decl: 0x%lx\n", buf.c_str(), (long) decl);
            }
         }
      }
      else {
         if (gDebug > 0) {
            fprintf(stderr, "TClingClassInfo(name): found cling class name: %s  "
                    "decl: 0x%lx\n", name, (long) decl);
         }
      }
      if (decl) {
         // Position our iterator on the found decl.
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
}

TClingClassInfo::TClingClassInfo(cling::Interpreter* interp,
                                   const clang::Decl* decl)
   : fClassInfo(0), fInterp(interp), fFirstTime(true), fDescend(false),
     fDecl(0)
{
   if (gAllowCint) {
      std::string buf;
      clang::PrintingPolicy Policy(decl->getASTContext().getPrintingPolicy());
      llvm::dyn_cast<clang::NamedDecl>(decl)->
      getNameForDiagnostic(buf, Policy, /*Qualified=*/true);
      if (gDebug > 0) {
         fprintf(stderr, "TClingClassInfo(decl): looking up class name: %s  "
                 "decl: 0x%lx\n", buf.c_str(), (long) decl);
      }
      fClassInfo = new G__ClassInfo(buf.c_str());
      if (gDebug > 0) {
         if (!fClassInfo->IsValid()) {
            fprintf(stderr,
                    "TClingClassInfo(decl): could not find cint class for "
                    "name: %s  decl: 0x%lx\n", buf.c_str(), (long) decl);
         }
         else {
            fprintf(stderr, "TClingClassInfo(decl): found cint class for "
                    "name: %s  tagnum: %d\n", buf.c_str(),
                    fClassInfo->Tagnum());
         }
      }
   }
   else {
      fClassInfo = new G__ClassInfo();
   }
   if (gAllowClang) {
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
}

G__ClassInfo* TClingClassInfo::GetClassInfo() const
{
   return fClassInfo;
}

cling::Interpreter* TClingClassInfo::GetInterpreter()
{
   return fInterp;
}

const clang::Decl* TClingClassInfo::GetDecl() const
{
   return fDecl;
}

long TClingClassInfo::ClassProperty() const
{
   if (!IsValid()) {
      return 0L;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fClassInfo->ClassProperty();
      }
      return 0L;
   }
   if (!gAllowClang) {
      return 0L;
   }
   const clang::RecordDecl* RD = llvm::dyn_cast<clang::RecordDecl>(fDecl);
   if (!RD) {
      // We are an enum or namespace.
      // The cint interface always returns 0L for these guys.
      if (gAllowCint) {
         if (IsValidCint()) {
            long cint_property = fClassInfo->ClassProperty();
            if (cint_property != 0L) {
               if (gDebug > 0) {
                  fprintf(stderr,
                          "VALIDITY: TClingClassInfo::ClassProperty: %s  "
                          "cint: 0x%lx  clang: 0x%lx\n", fClassInfo->Fullname(),
                          cint_property, 0L);
               }
            }
         }
      }
      return 0L;
   }
   if (RD->isUnion()) {
      // The cint interface always returns 0L for these guys.
      if (gAllowCint) {
         if (IsValidCint()) {
            long cint_property = fClassInfo->ClassProperty();
            if (cint_property != 0L) {
               if (gDebug > 0) {
                  fprintf(stderr,
                          "VALIDITY: TClingClassInfo::ClassProperty: %s  "
                          "cint: 0x%lx  clang: 0x%lx\n", fClassInfo->Fullname(),
                          cint_property, 0L);
               }
            }
         }
      }
      return 0L;
   }
   // We now have a class or a struct.
   const clang::CXXRecordDecl* CRD =
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
   // Partially compensate for crazy cint bug.  Poor little cint tries
   // to make sure that it does not mark the class as having an implicit
   // constructor if there is an explicit one, but it does this in one
   // single pass over all the member functions, so if the programmer
   // did not put the constructors first, then the testing does not work.
   // And even worse, cint always puts the destructor in the first slot
   // of the member function list, so if the destructor is virtual the
   // class is marked as having an implicit constructor even if later
   // on we find an explicit constructor in the list.
   //if (clang::CXXDestructorDecl* DD = CRD->getDestructor()) {
   //   if (DD->isVirtual()) {
   //      property |= G__CLS_HASIMPLICITCTOR;
   //   }
   //}
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
   // This validity check can never work, the cint semantics are just
   // too complicated.  The properties come out different depending on
   // whether the class is compiled or interpreted and depending on
   // what options are selected in the linkdef file.
   //
   //if (IsValidCint()) {
   //   long cint_property = fClassInfo->ClassProperty();
   //   if (property != cint_property) {
   //      fprintf(stderr, "VALIDITY: TClingClassInfo::ClassProperty: %s  "
   //              "cint: 0x%lx  clang: 0x%lx\n", fClassInfo->Fullname(),
   //              cint_property, property);
   //   }
   //}
   return property;
}

void TClingClassInfo::Delete(void* arena) const
{
   // Note: This is an interpreter function.
   if (!IsValid()) {
      return;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         fClassInfo->Delete(arena);
      }
      return;
   }
   if (!gAllowClang) {
      return;
   }
   // TODO: Implement this when cling provides function call.
   return;
}

void TClingClassInfo::DeleteArray(void* arena, bool dtorOnly) const
{
   // Note: This is an interpreter function.
   if (!IsValid()) {
      return;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         fClassInfo->DeleteArray(arena, dtorOnly);
      }
      return;
   }
   if (!gAllowClang) {
      return;
   }
   // TODO: Implement this when cling provides function call.
   return;
}

void TClingClassInfo::Destruct(void* arena) const
{
   // Note: This is an interpreter function.
   if (!IsValid()) {
      return;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         fClassInfo->Destruct(arena);
      }
      return;
   }
   if (!gAllowClang) {
      return;
   }
   // TODO: Implement this when cling provides function call.
   return;
}

tcling_MethodInfo TClingClassInfo::GetMethod(const char* fname,
      const char* arg, long* poffset, MatchMode mode /*= ConversionMatch*/,
      InheritanceMode imode /*= WithInheritance*/) const
{
   if (!IsValid()) {
      tcling_MethodInfo tmi(fInterp);
      return tmi;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         G__MethodInfo mi = fClassInfo->GetMethod(fname, arg, poffset,
                            (Cint::G__ClassInfo::MatchMode) mode,
                            (Cint::G__ClassInfo::InheritanceMode) imode);
         tcling_MethodInfo tmi(fInterp, &mi);
         return tmi;
      }
      tcling_MethodInfo tmi(fInterp);
      return tmi;
   }
   if (!gAllowClang) {
      tcling_MethodInfo tmi(fInterp);
      return tmi;
   }
   const clang::FunctionDecl* FD =
      fInterp->lookupFunctionArgs(fDecl, fname, arg);
   if (poffset) {
      *poffset = 0L;
   }
   tcling_MethodInfo tmi(fInterp);
   tmi.Init(FD);
   return tmi;
}

int TClingClassInfo::GetMethodNArg(const char* method, const char* proto) const
{
   // Note: Used only by TQObject.cxx:170 and only for interpreted classes.
   if (!IsValid()) {
      return -1;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         G__MethodInfo meth;
         long offset = 0L;
         meth = fClassInfo->GetMethod(method, proto, &offset);
         if (meth.IsValid()) {
            return meth.NArg();
         }
         return -1;
      }
      return -1;
   }
   if (!gAllowClang) {
      return -1;
   }
   int clang_val = -1;
   const clang::FunctionDecl* decl =
      fInterp->lookupFunctionProto(fDecl, method, proto);
   if (decl) {
      unsigned num_params = decl->getNumParams();
      clang_val = static_cast<int>(num_params);
   }
   if (gAllowCint) {
      if (IsValidCint()) {
         G__MethodInfo meth;
         long offset = 0L;
         meth = fClassInfo->GetMethod(method, proto, &offset);
         int cint_val = -1;
         if (meth.IsValid()) {
            cint_val = meth.NArg();
         }
         if (clang_val != cint_val) {
            if (gDebug > 0) {
               fprintf(stderr,
                       "VALIDITY: TClingClassInfo::GetMethodNArg(method,proto): "
                       "%s(%s)  cint: %d  clang: %d\n",
                       method, proto, cint_val, clang_val);
            }
         }
      }
   }
   return clang_val;
}

bool TClingClassInfo::HasDefaultConstructor() const
{
   // Note: This is a ROOT special!  It actually test for the root ioctor.
   if (!IsValid()) {
      return false;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fClassInfo->HasDefaultConstructor();
      }
      return false;
   }
   if (!gAllowClang) {
      return false;
   }
   // FIXME: Look for root ioctor when we have function lookup, and
   //        rootcling can tell us what the name of the ioctor is.
   return false;
}

bool TClingClassInfo::HasMethod(const char* name) const
{
   if (!IsValid()) {
      return false;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fClassInfo->HasMethod(name);
      }
      return false;
   }
   if (!gAllowClang) {
      return false;
   }
   bool found = false;
   std::string given_name(name);
   if (!llvm::isa<clang::EnumDecl>(fDecl)) {
      // We are a class, struct, union, namespace, or translation unit.
      clang::DeclContext* DC = llvm::cast<clang::DeclContext>(fDecl);
      llvm::SmallVector<clang::DeclContext*, 2> fContexts;
      DC->collectAllContexts(fContexts);
      for (unsigned I = 0; !found && (I < fContexts.size()); ++I) {
         DC = fContexts[I];
         for (clang::DeclContext::decl_iterator iter = DC->decls_begin();
               *iter; ++iter) {
            if (const clang::FunctionDecl* FD =
                     llvm::dyn_cast<clang::FunctionDecl>(*iter)) {
               if (FD->getNameAsString() == given_name) {
                  found = true;
                  break;
               }
            }
         }
      }
   }
   if (gAllowCint) {
      if (IsValidCint()) {
         int cint_val = fClassInfo->HasMethod(name);
         int clang_val = found;
         if (clang_val != cint_val) {
            if (gDebug > 0) {
               fprintf(stderr, "VALIDITY: TClingClassInfo::HasMethod(name): "
                       "%s::%s  cint: %d  clang: %d\n", fClassInfo->Fullname(),
                       name, cint_val, clang_val);
            }
         }
      }
   }
   return found;
}

void TClingClassInfo::Init(const char* name)
{
   if (gDebug > 0) {
      fprintf(stderr, "TClingClassInfo::Init(name): looking up class: %s\n",
              name);
   }
   fFirstTime = true;
   fDescend = false;
   fIter = clang::DeclContext::decl_iterator();
   fDecl = 0;
   fIterStack.clear();
   if (gAllowCint) {
      fClassInfo->Init(name);
      if (gDebug > 0) {
         if (!fClassInfo->IsValid()) {
            fprintf(stderr, "TClingClassInfo::Init(name): "
                    "could not find cint class for name: %s\n", name);
         }
         else {
            fprintf(stderr, "TClingClassInfo::Init(name): "
                    "found cint class for name: %s  tagnum: %d\n",
                    name, fClassInfo->Tagnum());
         }
      }
   }
   else {
      delete fClassInfo;
      fClassInfo = new G__ClassInfo;
   }
   if (gAllowClang) {
      const clang::Decl* decl = fInterp->lookupScope(name);
      if (!decl) {
         if (gDebug > 0) {
            fprintf(stderr, "TClingClassInfo::Init(name): "
                    "cling class not found name: %s\n", name);
         }
         std::string buf = TClassEdit::InsertStd(name);
         decl = fInterp->lookupScope(buf);
         if (!decl) {
            if (gDebug > 0) {
               fprintf(stderr,
                       "TClingClassInfo::Init(name): cling class not found "
                       "name: %s\n", buf.c_str());
            }
         }
         else {
            if (gDebug > 0) {
               fprintf(stderr,
                       "TClingClassInfo::Init(name): found cling class "
                       "name: %s  decl: 0x%lx\n", buf.c_str(), (long) decl);
            }
         }
      }
      else {
         if (gDebug > 0) {
            fprintf(stderr, "TClingClassInfo::Init(name): found cling class "
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
}

void TClingClassInfo::Init(int tagnum)
{
   if (gDebug > 0) {
      fprintf(stderr, "TClingClassInfo::Init(tagnum): looking up tagnum: %d\n",
              tagnum);
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
      const char* name = fClassInfo->Fullname();
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
      const clang::Decl* decl = fInterp->lookupScope(name);
      if (!decl) {
         if (gDebug > 0) {
            fprintf(stderr,
                    "TClingClassInfo::Init(tagnum): cling class not found "
                    "name: %s  tagnum: %d\n", name, tagnum);
         }
         std::string buf = TClassEdit::InsertStd(name);
         decl = const_cast<clang::Decl*>(fInterp->lookupScope(buf));
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
}

bool TClingClassInfo::IsBase(const char* name) const
{
   if (!IsValid()) {
      return false;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fClassInfo->IsBase(name);
      }
      return false;
   }
   if (!gAllowClang) {
      return false;
   }
   TClingClassInfo base(fInterp, name);
   if (!base.IsValid()) {
      if (gAllowCint) {
         if (IsValidCint()) {
            int cint_val = fClassInfo->IsBase(name);
            int clang_val = 0;
            if (clang_val != cint_val) {
               if (gDebug > 0) {
                  fprintf(stderr,
                          "VALIDITY: TClingClassInfo::IsBase(name): "
                          "%s(%s)  cint: %d  clang: %d\n",
                          fClassInfo->Fullname(), name, cint_val, clang_val);
               }
            }
         }
      }
      return false;
   }
   if (!base.IsValidClang()) {
      if (IsValidCint()) {
         int cint_val = fClassInfo->IsBase(name);
         int clang_val = 0;
         if (clang_val != cint_val) {
            if (gDebug > 0) {
               fprintf(stderr, "VALIDITY: TClingClassInfo::IsBase(name): "
                       "%s(%s)  cint: %d  clang: %d\n", fClassInfo->Fullname(),
                       name, cint_val, clang_val);
            }
            return cint_val;
         }
      }
      return false;
   }
   const clang::CXXRecordDecl* CRD =
      llvm::dyn_cast<clang::CXXRecordDecl>(fDecl);
   if (!CRD) {
      // We are an enum, namespace, or translation unit,
      // we cannot be the base of anything.
      return false;
   }
   const clang::CXXRecordDecl* baseCRD =
      llvm::dyn_cast<clang::CXXRecordDecl>(base.GetDecl());
   if (gAllowCint) {
      if (IsValidCint()) {
         int cint_val = fClassInfo->IsBase(name);
         int clang_val = CRD->isDerivedFrom(baseCRD);
         if (clang_val != cint_val) {
            if (gDebug > 0) {
               fprintf(stderr, "VALIDITY: TClingClassInfo::IsBase(name): "
                       "%s(%s)  cint: %d  clang: %d\n", fClassInfo->Fullname(),
                       name, cint_val, clang_val);
            }
         }
      }
   }
   return CRD->isDerivedFrom(baseCRD);
}

bool TClingClassInfo::IsEnum(cling::Interpreter* interp, const char* name)
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
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fClassInfo->IsLoaded();
      }
      return false;
   }
   if (!gAllowClang) {
      return false;
   }
   // All clang classes are considered loaded.
   return true;
}

bool TClingClassInfo::IsValid() const
{
   return IsValidCint() || IsValidClang();
}

bool TClingClassInfo::IsValidCint() const
{
   if (gAllowCint) {
      return fClassInfo->IsValid();
   }
   return false;
}

bool TClingClassInfo::IsValidClang() const
{
   if (gAllowClang) {
      return fDecl;
   }
   return false;
}

bool TClingClassInfo::IsValidMethod(const char* method, const char* proto,
                                     long* offset) const
{
   if (!IsValid()) {
      return false;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fClassInfo->GetMethod(method, proto, offset).IsValid();
      }
      return false;
   }
   if (gAllowCint) {
      if (IsValidCint()) {
         bool cint_val = fClassInfo->GetMethod(method, proto, offset).IsValid();
         bool clang_val = GetMethod(method, proto, offset).IsValid();
         if (clang_val != cint_val) {
            if (gDebug > 0) {
               fprintf(stderr,
                       "VALIDITY: TClingClassInfo::IsValidMethod: %s(%s)  "
                       "cint: %d  clang: %d\n", method, proto, cint_val, clang_val);
            }
         }
      }
   }
   return GetMethod(method, proto, offset).IsValid();
}

int TClingClassInfo::AdvanceToDecl(const clang::Decl* target_decl)
{
   const clang::TranslationUnitDecl* TU = target_decl->getTranslationUnitDecl();
   const clang::DeclContext* DC = llvm::cast<clang::DeclContext>(TU);
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
            clang::DeclContext* DC = llvm::cast<clang::DeclContext>(*fIter);
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
#if 0
      if (clang::NamedDecl* ND =
               llvm::dyn_cast<clang::NamedDecl>(*fIter)) {
         clang::ASTContext& Context = ND->getASTContext();
         clang::PrintingPolicy Policy(Context.getPrintingPolicy());
         std::string tmp;
         ND->getNameForDiagnostic(tmp, Policy, /*Qualified=*/true);
         fprintf(stderr,
                 "TClingClassInfo::InternalNext:  "
                 "0x%08lx %s  %s\n",
                 (long) *fIter, fIter->getDeclKindName(), tmp.c_str());
      }
#endif // 0
      // Return if this decl is a class, struct, union, enum, or namespace.
      clang::Decl::Kind DK = fIter->getKind();
      if ((DK == clang::Decl::Namespace) || (DK == clang::Decl::Enum) ||
            (DK == clang::Decl::CXXRecord) ||
            (DK == clang::Decl::ClassTemplateSpecialization)) {
         const clang::TagDecl* TD = llvm::dyn_cast<clang::TagDecl>(*fIter);
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
            clang::DeclContext* DC = llvm::cast<clang::DeclContext>(*fIter);
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
   if (!gAllowClang) {
      if (gAllowCint) {
         return fClassInfo->Next();
      }
      return 0;
   }
   return InternalNext();
}

void* TClingClassInfo::New() const
{
   // Note: This is an interpreter function.
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fClassInfo->New();
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   // TODO: Fix this when cling implements function call.
   return 0;
}

void* TClingClassInfo::New(int n) const
{
   // Note: This is an interpreter function.
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fClassInfo->New(n);
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   // TODO: Fix this when cling implements function call.
   return 0;
}

void* TClingClassInfo::New(int n, void* arena) const
{
   // Note: This is an interpreter function.
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fClassInfo->New(n, arena);
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   // TODO: Fix this when cling implements function call.
   return 0;
}

void* TClingClassInfo::New(void* arena) const
{
   // Note: This is an interpreter function.
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fClassInfo->New(arena);
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   // TODO: Fix this when cling implements function call.
   return 0;
}

long TClingClassInfo::Property() const
{
   if (!IsValid()) {
      return 0L;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fClassInfo->Property();
      }
      return 0L;
   }
   if (!gAllowClang) {
      return 0L;
   }
   long property = 0L;
   property |= G__BIT_ISCPPCOMPILED;
   clang::Decl::Kind DK = fDecl->getKind();
   if ((DK == clang::Decl::Namespace) || (DK == clang::Decl::TranslationUnit)) {
      property |= G__BIT_ISNAMESPACE;
      if (gAllowCint) {
         if (IsValidCint()) {
            long cint_property = fClassInfo->Property();
            cint_property &= ~static_cast<long>(G__BIT_ISCPPCOMPILED);
            long clang_property = property;
            clang_property &= ~static_cast<long>(G__BIT_ISCPPCOMPILED);
            if (cint_property && (cint_property != clang_property)) {
               if (gDebug > 0) {
                  fprintf(stderr, "VALIDITY: TClingClassInfo::Property: %s  "
                          "cint: 0x%lx  clang: 0x%lx\n", fClassInfo->Fullname(),
                          cint_property, clang_property);
               }
            }
         }
      }
      return property;
   }
   // Note: Now we have class, enum, struct, union only.
   const clang::TagDecl* TD = llvm::dyn_cast<clang::TagDecl>(fDecl);
   if (!TD) {
      if (gAllowCint) {
         if (IsValidCint()) {
            long cint_property = fClassInfo->Property();
            cint_property &= ~static_cast<long>(G__BIT_ISCPPCOMPILED);
            if (cint_property != 0L) {
               if (gDebug > 0) {
                  fprintf(stderr,
                          "VALIDITY: TClingClassInfo::Property: %s  "
                          "cint: 0x%lx  clang: 0x%lx\n", fClassInfo->Fullname(),
                          cint_property, 0L);
               }
            }
         }
      }
      return 0L;
   }
   if (TD->isEnum()) {
      property |= G__BIT_ISENUM;
      if (gAllowCint) {
         if (IsValidCint()) {
            long cint_property = fClassInfo->Property();
            cint_property &= ~static_cast<long>(G__BIT_ISCPPCOMPILED);
            long clang_property = property;
            clang_property &= ~static_cast<long>(G__BIT_ISCPPCOMPILED);
            if (cint_property && (cint_property != clang_property)) {
               if (gDebug > 0) {
                  fprintf(stderr,
                          "VALIDITY: TClingClassInfo::Property: %s  "
                          "cint: 0x%lx  clang: 0x%lx\n", fClassInfo->Fullname(),
                          cint_property, clang_property);
               }
            }
         }
      }
      return property;
   }
   // Note: Now we have class, struct, union only.
   const clang::CXXRecordDecl* CRD =
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
   if (gAllowCint) {
      if (IsValidCint()) {
         long cint_property = fClassInfo->Property();
         cint_property &= ~static_cast<long>(G__BIT_ISCPPCOMPILED);
         cint_property &= ~static_cast<long>(G__BIT_ISABSTRACT);
         long clang_property = property;
         clang_property &= ~static_cast<long>(G__BIT_ISCPPCOMPILED);
         clang_property &= ~static_cast<long>(G__BIT_ISABSTRACT);
         if (cint_property && (cint_property != clang_property)) {
            if (gDebug > 0) {
               fprintf(stderr, "VALIDITY: TClingClassInfo::Property: %s  "
                       "cint: 0x%lx  clang: 0x%lx\n", fClassInfo->Fullname(),
                       cint_property, clang_property);
            }
         }
      }
   }
   return property;
}

int TClingClassInfo::RootFlag() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fClassInfo->RootFlag();
      }
      return 0;
   }
   if (!gAllowClang) {
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
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fClassInfo->Size();
      }
      return -1;
   }
   if (!gAllowClang) {
      return -1;
   }
   clang::Decl::Kind DK = fDecl->getKind();
   if (DK == clang::Decl::Namespace) {
      // Namespaces are special for cint.
      if (gAllowCint) {
         if (IsValidCint()) {
            int cint_size = fClassInfo->Size();
            if ((cint_size != 0) && (cint_size != 1)) {
               if (gDebug > 0) {
                  fprintf(stderr,
                          "VALIDITY: TClingClassInfo::Size: namespace %s  "
                          "cint: %d  clang: %d\n", fClassInfo->Fullname(),
                          cint_size, 1);
               }
            }
         }
      }
      return 1;
   }
   else if (DK == clang::Decl::Enum) {
      // Enums are special for cint.
      if (gAllowCint) {
         if (IsValidCint()) {
            int cint_size = fClassInfo->Size();
            if ((cint_size != 0) && (cint_size != 4)) {
               if (gDebug > 0) {
                  fprintf(stderr,
                          "VALIDITY: TClingClassInfo::Size: enum %s  cint: "
                          "%d  clang: %d\n", fClassInfo->Fullname(), cint_size, 0);
               }
            }
         }
      }
      return 0;
   }
   const clang::RecordDecl* RD = llvm::dyn_cast<clang::RecordDecl>(fDecl);
   if (!RD) {
      // Should not happen.
      if (gAllowCint) {
         if (IsValidCint()) {
            int cint_size = fClassInfo->Size();
            if (cint_size != -1) {
               if (gDebug > 0) {
                  fprintf(stderr,
                          "VALIDITY: TClingClassInfo::Size: %s  cint: %d  "
                          "clang: %d\n", fClassInfo->Fullname(), cint_size, -1);
               }
            }
         }
      }
      return -1;
   }
   if (!RD->getDefinition()) {
      // Forward-declared class.
      return 0;
   }
   clang::ASTContext& Context = fDecl->getASTContext();
   const clang::ASTRecordLayout& Layout = Context.getASTRecordLayout(RD);
   int64_t size = Layout.getSize().getQuantity();
   int clang_size = static_cast<int>(size);
   if (gAllowCint) {
      if (IsValidCint()) {
         int cint_size = fClassInfo->Size();
         if (cint_size && (cint_size != clang_size)) {
            if (gDebug > 0) {
               fprintf(stderr,
                       "VALIDITY: TClingClassInfo::Size: %s  cint: %d  "
                       "clang: %d\n", fClassInfo->Fullname(), cint_size, clang_size);
            }
         }
      }
   }
   return clang_size;
}

long TClingClassInfo::Tagnum() const
{
   // Note: This *must* return a *cint* tagnum for now.
   if (!IsValid()) {
      return -1L;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fClassInfo->Tagnum();
      }
      return -1L;
   }
   if (!gAllowClang) {
      return -1;
   }
   return reinterpret_cast<long>(fDecl);
}

const char* TClingClassInfo::FileName() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fClassInfo->FileName();
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   // FIXME: Implement this when rootcling provides the information.
   return 0;
}

const char* TClingClassInfo::FullName() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fClassInfo->Fullname();
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   // Note: This *must* be static because we are returning a pointer inside it!
   static std::string buf;
   buf.clear();
   clang::PrintingPolicy Policy(fDecl->getASTContext().getPrintingPolicy());
   llvm::dyn_cast<clang::NamedDecl>(fDecl)->
   getNameForDiagnostic(buf, Policy, /*Qualified=*/true);
   if (gAllowCint) {
      if (IsValidCint()) {
         const char* cint_fullname = fClassInfo->Fullname();
         if (buf != cint_fullname) {
            if (gDebug > 0) {
               fprintf(stderr,
                       "VALIDITY: TClingClassInfo::FullName:  "
                       "cint: %s  clang: %s\n", cint_fullname, buf.c_str());
            }
         }
      }
   }
   return buf.c_str();
}

const char* TClingClassInfo::Name() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fClassInfo->Name();
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   // Note: This *must* be static because we are returning a pointer inside it!
   static std::string buf;
   buf.clear();
   clang::PrintingPolicy Policy(fDecl->getASTContext().getPrintingPolicy());
   llvm::dyn_cast<clang::NamedDecl>(fDecl)->
   getNameForDiagnostic(buf, Policy, /*Qualified=*/false);
   if (gAllowCint) {
      if (IsValidCint()) {
         const char* cint_name = fClassInfo->Name();
         if (buf != cint_name) {
            if (gDebug > 0) {
               fprintf(stderr,
                       "VALIDITY: TClingClassInfo::Name:  "
                       "cint: %s  clang: %s\n", cint_name, buf.c_str());
            }
         }
      }
   }
   return buf.c_str();
}

const char* TClingClassInfo::Title() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fClassInfo->Title();
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   // FIXME: Implement this when rootcling provides the info.
   return 0;
}

const char* TClingClassInfo::TmpltName() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fClassInfo->TmpltName();
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   // Note: This *must* be static because we are returning a pointer inside it!
   static std::string buf;
   buf.clear();
   // Note: This does *not* include the template arguments!
   buf = llvm::dyn_cast<clang::NamedDecl>(fDecl)->getNameAsString();
   if (gAllowCint) {
      if (IsValidCint()) {
         const char* cint_tmpltname = fClassInfo->TmpltName();
         if (buf != cint_tmpltname) {
            if (gDebug > 0) {
               fprintf(stderr,
                       "VALIDITY: TClingClassInfo::TmpltName:  "
                       "cint: %s  clang: %s\n", cint_tmpltname, buf.c_str());
            }
         }
      }
   }
   return buf.c_str();
}

