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

tcling_ClassInfo::~tcling_ClassInfo()
{
   delete fClassInfo;
   fClassInfo = 0;
   fInterp = 0;
   fDecl = 0;
}

tcling_ClassInfo::tcling_ClassInfo(cling::Interpreter* interp)
   : fClassInfo(new G__ClassInfo), fInterp(interp), fDecl(0)
{
}

tcling_ClassInfo::tcling_ClassInfo(const tcling_ClassInfo& rhs)
{
   fClassInfo = new G__ClassInfo(*rhs.fClassInfo);
   fInterp = rhs.fInterp;
   fDecl = rhs.fDecl;
}

tcling_ClassInfo& tcling_ClassInfo::operator=(const tcling_ClassInfo& rhs)
{
   if (this != &rhs) {
      delete fClassInfo;
      fClassInfo = new G__ClassInfo(*rhs.fClassInfo);
      fInterp = rhs.fInterp;
      fDecl = rhs.fDecl;
   }
   return *this;
}

tcling_ClassInfo::tcling_ClassInfo(cling::Interpreter* interp, const char* name)
   : fClassInfo(0), fInterp(interp), fDecl(0)
{
   if (gDebug > 0) {
      fprintf(stderr,
         "tcling_ClassInfo(name): looking up class name: %s\n", name);
   }
   fClassInfo = new G__ClassInfo(name);
   if (gDebug > 0) {
      if (!fClassInfo->IsValid()) {
         fprintf(stderr,
            "tcling_ClassInfo(name): could not find cint class for name: %s\n",
            name);
      }
      else {
         fprintf(stderr,
            "tcling_ClassInfo(name): found cint class for name: %s  "
            "tagnum: %d\n", name, fClassInfo->Tagnum());
      }
   }
   clang::Decl* decl = const_cast<clang::Decl*>(fInterp->lookupScope(name));
   if (!decl) {
      if (gDebug > 0) {
         fprintf(stderr, "tcling_ClassInfo(name): cling class not found "
                 "name: %s\n", name);
      }
      std::string buf = TClassEdit::InsertStd(name);
      decl = const_cast<clang::Decl*>(fInterp->lookupScope(buf));
      if (!decl) {
         if (gDebug > 0) {
            fprintf(stderr, "tcling_ClassInfo(name): cling class not found "
                    "name: %s\n", buf.c_str());
         }
      }
      else {
         fDecl = decl;
         if (gDebug > 0) {
            fprintf(stderr,
               "tcling_ClassInfo(name): found cling class name: %s  "
               "decl: 0x%lx\n", buf.c_str(), (long) fDecl);
         }
      }
   }
   else {
      fDecl = decl;
      if (gDebug > 0) {
         fprintf(stderr, "tcling_ClassInfo(name): found cling class name: %s  "
                 "decl: 0x%lx\n", name, (long) fDecl);
      }
   }
}

tcling_ClassInfo::tcling_ClassInfo(cling::Interpreter* interp,
                                   const clang::Decl* decl)
   : fClassInfo(0), fInterp(interp), fDecl(decl)
{
   std::string buf;
   clang::PrintingPolicy P(fDecl->getASTContext().getPrintingPolicy());
   llvm::dyn_cast<clang::NamedDecl>(fDecl)->getNameForDiagnostic(buf, P, true);
   if (gDebug > 0) {
      fprintf(stderr, "tcling_ClassInfo(decl): looking up class name: %s  "
              "decl: 0x%lx\n", buf.c_str(), (long) fDecl);
   }
   fClassInfo = new G__ClassInfo(buf.c_str());
   if (gDebug > 0) {
      if (!fClassInfo->IsValid()) {
         fprintf(stderr,
            "tcling_ClassInfo(decl): could not find cint class for "
            "name: %s  decl: 0x%lx\n", buf.c_str(), (long) fDecl);
      }
      else {
         fprintf(stderr, "tcling_ClassInfo(decl): found cint class for "
                 "name: %s  tagnum: %d\n", buf.c_str(), fClassInfo->Tagnum());
      }
   }
}

G__ClassInfo* tcling_ClassInfo::GetClassInfo() const
{
   return fClassInfo;
}

cling::Interpreter* tcling_ClassInfo::GetInterpreter()
{
   return fInterp;
}

const clang::Decl* tcling_ClassInfo::GetDecl() const
{
   return fDecl;
}

long tcling_ClassInfo::ClassProperty() const
{
   if (!IsValid()) {
      return 0L;
   }
   if (!IsValidClang()) {
      return fClassInfo->ClassProperty();
   }
   const clang::RecordDecl* RD = llvm::dyn_cast<clang::RecordDecl>(fDecl);
   if (!RD) {
      // We are an enum or namespace.
      // The cint interface always returns 0L for these guys.
      if (IsValidCint()) {
         long cint_property = fClassInfo->ClassProperty();
         if (cint_property != 0L) {
            if (gDebug > 0) {
               fprintf(stderr,
                  "VALIDITY: tcling_ClassInfo::ClassProperty: %s  "
                  "cint: 0x%lx  clang: 0x%lx\n", fClassInfo->Fullname(),
                  cint_property, 0L);
            }
            return cint_property;
         }
      }
      return 0L;
   }
   if (RD->isUnion()) {
      // The cint interface always returns 0L for these guys.
      if (IsValidCint()) {
         long cint_property = fClassInfo->ClassProperty();
         if (cint_property != 0L) {
            if (gDebug > 0) {
               fprintf(stderr, "VALIDITY: tcling_ClassInfo::ClassProperty: %s  "
                       "cint: 0x%lx  clang: 0x%lx\n", fClassInfo->Fullname(),
                       cint_property, 0L);
            }
            return cint_property;
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
   //      fprintf(stderr, "VALIDITY: tcling_ClassInfo::ClassProperty: %s  "
   //              "cint: 0x%lx  clang: 0x%lx\n", fClassInfo->Fullname(),
   //              cint_property, property);
   //   }
   //}
   // FIXME: Remove this when we are ready to accept the differences.
   if (IsValidCint()) {
      return fClassInfo->ClassProperty();
   }
   return property;
}

void tcling_ClassInfo::Delete(void* arena) const
{
   // Note: This is an interpreter function.
   if (!IsValid()) {
      return;
   }
   if (!IsValidClang()) {
      fClassInfo->Delete(arena);
      return;
   }
   // TODO: Implement this when cling provides function call.
   if (IsValidCint()) {
      fClassInfo->Delete(arena);
      return;
   }
}

void tcling_ClassInfo::DeleteArray(void* arena, bool dtorOnly) const
{
   // Note: This is an interpreter function.
   if (!IsValid()) {
      return;
   }
   if (!IsValidClang()) {
      fClassInfo->DeleteArray(arena, dtorOnly);
      return;
   }
   // TODO: Implement this when cling provides function call.
   if (IsValidCint()) {
      fClassInfo->DeleteArray(arena, dtorOnly);
      return;
   }
}

void tcling_ClassInfo::Destruct(void* arena) const
{
   // Note: This is an interpreter function.
   return fClassInfo->Destruct(arena);
}

tcling_MethodInfo* tcling_ClassInfo::GetMethod(const char* fname,
      const char* arg, long* poffset, MatchMode mode /*= ConversionMatch*/,
      InheritanceMode imode /*= WithInheritance*/) const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      G__MethodInfo* mi = new G__MethodInfo(fClassInfo->GetMethod(
            fname, arg, poffset, (Cint::G__ClassInfo::MatchMode) mode,
            (Cint::G__ClassInfo::InheritanceMode) imode));
      tcling_MethodInfo* tmi = new tcling_MethodInfo(fInterp, mi);
      delete mi;
      mi = 0;
      return tmi;
   }
   // FIXME: Implement this with clang!
   if (IsValidCint()) {
      G__MethodInfo* mi = new G__MethodInfo(fClassInfo->GetMethod(
            fname, arg, poffset, (Cint::G__ClassInfo::MatchMode) mode,
            (Cint::G__ClassInfo::InheritanceMode) imode));
      tcling_MethodInfo* tmi = new tcling_MethodInfo(fInterp, mi);
      delete mi;
      mi = 0;
      return tmi;
   }
   return 0;
}

int tcling_ClassInfo::GetMethodNArg(const char* method, const char* proto) const
{
   // Note: Used only by TQObject.cxx:170 and only for interpreted classes.
   if (!IsValid()) {
      return false;
   }
   if (!IsValidClang()) {
      G__MethodInfo meth;
      long offset = 0L;
      meth = fClassInfo->GetMethod(method, proto, &offset);
      if (meth.IsValid()) {
         return meth.NArg();
      }
      return -1;
   }
   if (IsValidCint()) {
      G__MethodInfo meth;
      long offset = 0L;
      meth = fClassInfo->GetMethod(method, proto, &offset);
      int cint_val = -1;
      if (meth.IsValid()) {
         cint_val = meth.NArg();
      }
      const clang::FunctionDecl* decl =
         fInterp->lookupFunctionProto(fDecl, method, proto);
      int clang_val = -1;
      if (decl) {
         clang_val = static_cast<int>(decl->getNumParams());
      }
      if (clang_val != cint_val) {
         if (gDebug > 0) {
            fprintf(stderr,
               "VALIDITY: tcling_ClassInfo::GetMethodNArg(method,proto): "
               "%s(%s)  cint: %d  clang: %d\n",
               method, proto, cint_val, clang_val);
         }
      }
      return cint_val;
   }
   const clang::FunctionDecl* decl =
      fInterp->lookupFunctionProto(fDecl, method, proto);
   if (!decl) {
      return -1;
   }
   unsigned clang_val = decl->getNumParams();
   return static_cast<int>(clang_val);
}

bool tcling_ClassInfo::HasDefaultConstructor() const
{
   // Note: This is a ROOT special!  It actually test for the root ioctor.
   if (!IsValid()) {
      return false;
   }
   if (!IsValidClang()) {
      return fClassInfo->HasDefaultConstructor();
   }
   // FIXME: Look for root ioctor when we have function lookup, and
   //        rootcling can tell us what the name of the ioctor is.
   if (IsValidCint()) {
      return fClassInfo->HasDefaultConstructor();
   }
   return false;
}

bool tcling_ClassInfo::HasMethod(const char* name) const
{
   if (!IsValid()) {
      return false;
   }
   if (!IsValidClang()) {
      return fClassInfo->HasMethod(name);
   }
   const clang::CXXRecordDecl* CRD =
      llvm::dyn_cast<clang::CXXRecordDecl>(fDecl);
   if (!CRD) {
      // We are an enum or namespace.
      // FIXME: Make it work for a namespace!
      if (IsValidCint()) {
         int cint_val = fClassInfo->HasMethod(name);
         int clang_val = false;
         if (clang_val != cint_val) {
            if (gDebug > 0) {
               fprintf(stderr, "VALIDITY: tcling_ClassInfo::HasMethod(name): "
                       "%s(%s)  cint: %d  clang: %d\n", fClassInfo->Fullname(),
                       name, cint_val, clang_val);
            }
            return cint_val;
         }
      }
      return false;
   }
   bool result = false;
   std::string given_name(name);
   for (
      clang::CXXRecordDecl::method_iterator M = CRD->method_begin(),
      MEnd = CRD->method_end();
      M != MEnd;
      ++M
   ) {
      if (M->getNameAsString() == given_name) {
         result = true;
      }
   }
   if (IsValidCint()) {
      int cint_val = fClassInfo->HasMethod(name);
      int clang_val = result;
      if (clang_val != cint_val) {
         if (gDebug > 0) {
            fprintf(stderr, "VALIDITY: tcling_ClassInfo::HasMethod(name): "
                    "%s(%s)  cint: %d  clang: %d\n", fClassInfo->Fullname(),
                    name, cint_val, clang_val);
         }
      }
      return cint_val;
   }
   return result;
}

void tcling_ClassInfo::Init(const char* name)
{
   if (gDebug > 0) {
      fprintf(stderr, "tcling_ClassInfo::Init(name): looking up class: %s\n",
              name);
   }
   fDecl = 0;
   fClassInfo->Init(name);
   if (gDebug > 0) {
      if (!fClassInfo->IsValid()) {
         fprintf(stderr, "tcling_ClassInfo::Init(name): could not find cint "
                 "class for name: %s\n", name);
      }
      else {
         fprintf(stderr, "tcling_ClassInfo::Init(name): found cint class for "
                 "name: %s  tagnum: %d\n", name, fClassInfo->Tagnum());
      }
   }
   clang::Decl* decl = const_cast<clang::Decl*>(fInterp->lookupScope(name));
   if (!decl) {
      if (gDebug > 0) {
         fprintf(stderr, "tcling_ClassInfo::Init(name): cling class not found "
                 "name: %s\n", name);
      }
      // FIXME: Remove this call!
      std::string buf = TClassEdit::InsertStd(name);
      decl = const_cast<clang::Decl*>(fInterp->lookupScope(buf));
      if (!decl) {
         if (gDebug > 0) {
            fprintf(stderr,
               "tcling_ClassInfo::Init(name): cling class not found "
               "name: %s\n", buf.c_str());
         }
      }
      else {
         fDecl = decl;
         if (gDebug > 0) {
            fprintf(stderr,
               "tcling_ClassInfo::Init(name): found cling class "
               "name: %s  decl: 0x%lx\n", buf.c_str(), (long) fDecl);
         }
      }
   }
   else {
      fDecl = decl;
      if (gDebug > 0) {
         fprintf(stderr, "tcling_ClassInfo::Init(name): found cling class "
                 "name: %s  decl: 0x%lx\n", name, (long) fDecl);
      }
   }
}

void tcling_ClassInfo::Init(int tagnum)
{
   if (gDebug > 0) {
      fprintf(stderr, "tcling_ClassInfo::Init(tagnum): looking up tagnum: %d\n",
              tagnum);
   }
   fDecl = 0;
   fClassInfo->Init(tagnum);
   if (!fClassInfo->IsValid()) {
      if (gDebug > 0) {
         fprintf(stderr, "tcling_ClassInfo::Init(tagnum): could not find cint "
                 "class for tagnum: %d\n", tagnum);
      }
      return;
   }
   const char* name = fClassInfo->Fullname();
   if (gDebug > 0) {
      fprintf(stderr, "tcling_ClassInfo::Init(tagnum): found cint class "
              "name: %s  tagnum: %d\n", name, tagnum);
   }
   if (!name || (name[0] == '\0')) {
      // No name, or name is blank, could be anonymous
      // class/struct/union or enum.  Cint does not give
      // us enough information to find the same decl in clang.
      return;
   }
   clang::Decl* decl = const_cast<clang::Decl*>(fInterp->lookupScope(name));
   if (!decl) {
      if (gDebug > 0) {
         fprintf(stderr,
            "tcling_ClassInfo::Init(tagnum): cling class not found "
            "name: %s  tagnum: %d\n", name, tagnum);
      }
      std::string buf = TClassEdit::InsertStd(name);
      decl = const_cast<clang::Decl*>(fInterp->lookupScope(buf));
      if (!decl) {
         if (gDebug > 0) {
            fprintf(stderr,
               "tcling_ClassInfo::Init(tagnum): cling class not found "
               "name: %s  tagnum: %d\n", buf.c_str(), tagnum);
         }
      }
      else {
         fDecl = decl;
         if (gDebug > 0) {
            fprintf(stderr, "tcling_ClassInfo::Init(tagnum): found cling class "
                    "name: %s  decl: 0x%lx\n", buf.c_str(), (long) fDecl);
         }
      }
   }
   else {
      fDecl = decl;
      if (gDebug > 0) {
         fprintf(stderr, "tcling_ClassInfo::Init(tagnum): found cling class "
                 "name: %s  decl: 0x%lx\n", name, (long) fDecl);
      }
   }
}

bool tcling_ClassInfo::IsBase(const char* name) const
{
   if (!IsValid()) {
      return false;
   }
   if (!IsValidClang()) {
      return fClassInfo->IsBase(name);
   }
   tcling_ClassInfo base(fInterp, name);
   if (!base.IsValid()) {
      if (IsValidCint()) {
         int cint_val = fClassInfo->IsBase(name);
         int clang_val = 0;
         if (clang_val != cint_val) {
            if (gDebug > 0) {
               fprintf(stderr, "VALIDITY: tcling_ClassInfo::IsBase(name): "
                       "%s(%s)  cint: %d  clang: %d\n", fClassInfo->Fullname(),
                       name, cint_val, clang_val);
            }
            return cint_val;
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
               fprintf(stderr, "VALIDITY: tcling_ClassInfo::IsBase(name): "
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
      // We are an enum or namespace, we cannot be the base of anything.
      return false;
   }
   const clang::CXXRecordDecl* baseCRD =
      llvm::dyn_cast<clang::CXXRecordDecl>(base.GetDecl());
   if (IsValidCint()) {
      int cint_val = fClassInfo->IsBase(name);
      int clang_val = CRD->isDerivedFrom(baseCRD);
      if (clang_val != cint_val) {
         if (gDebug > 0) {
            fprintf(stderr, "VALIDITY: tcling_ClassInfo::IsBase(name): "
                    "%s(%s)  cint: %d  clang: %d\n", fClassInfo->Fullname(),
                    name, cint_val, clang_val);
         }
         return cint_val;
      }
   }
   return CRD->isDerivedFrom(baseCRD);
}

bool tcling_ClassInfo::IsEnum(cling::Interpreter* interp, const char* name)
{
   // Note: This is a static member function.
   tcling_ClassInfo info(interp, name);
   if (info.IsValid() && (info.Property() & G__BIT_ISENUM)) {
      return true;
   }
   return false;
}

bool tcling_ClassInfo::IsLoaded() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      return fClassInfo->IsLoaded();
   }
   // FIXME: What could this mean for clang?
   if (IsValidCint()) {
      return fClassInfo->IsLoaded();
   }
   return true;
}

bool tcling_ClassInfo::IsValid() const
{
   return IsValidCint() || IsValidClang();
}

bool tcling_ClassInfo::IsValidCint() const
{
   if (fClassInfo) {
      if (fClassInfo->IsValid()) {
         return true;
      }
   }
   return false;
}

bool tcling_ClassInfo::IsValidClang() const
{
   return fDecl;
}

bool tcling_ClassInfo::IsValidMethod(const char* method, const char* proto,
                                     long* offset) const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      return fClassInfo->GetMethod(method, proto, offset).IsValid();
   }
   // FIXME: Fix this when we have function lookup.
   if (IsValidCint()) {
      return fClassInfo->GetMethod(method, proto, offset).IsValid();
   }
   return false;
}

int tcling_ClassInfo::Next()
{
   int cint_val = fClassInfo->Next();
   if (cint_val) {
      Init(fClassInfo->Tagnum());
   }
   return cint_val;
}

void* tcling_ClassInfo::New() const
{
   // Note: This is an interpreter function.
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      return fClassInfo->New();
   }
   // TODO: Fix this when cling implements function call.
   if (IsValidCint()) {
      return fClassInfo->New();
   }
   return 0;
}

void* tcling_ClassInfo::New(int n) const
{
   // Note: This is an interpreter function.
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      return fClassInfo->New(n);
   }
   // TODO: Fix this when cling implements function call.
   if (IsValidCint()) {
      return fClassInfo->New(n);
   }
   return 0;
}

void* tcling_ClassInfo::New(int n, void* arena) const
{
   // Note: This is an interpreter function.
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      return fClassInfo->New(n, arena);
   }
   // TODO: Fix this when cling implements function call.
   if (IsValidCint()) {
      return fClassInfo->New(n, arena);
   }
   return 0;
}

void* tcling_ClassInfo::New(void* arena) const
{
   // Note: This is an interpreter function.
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      return fClassInfo->New(arena);
   }
   // TODO: Fix this when cling implements function call.
   if (IsValidCint()) {
      return fClassInfo->New(arena);
   }
   return 0;
}

long tcling_ClassInfo::Property() const
{
   if (!IsValid()) {
      return 0L;
   }
   if (!IsValidClang()) {
      return fClassInfo->Property();
   }
   long property = 0L;
   property |= G__BIT_ISCPPCOMPILED;
   clang::Decl::Kind DK = fDecl->getKind();
   if (DK == clang::Decl::Namespace) {
      property |= G__BIT_ISNAMESPACE;
      if (IsValidCint()) {
         long cint_property = fClassInfo->Property();
         cint_property &= ~static_cast<long>(G__BIT_ISCPPCOMPILED);
         long clang_property = property;
         clang_property &= ~static_cast<long>(G__BIT_ISCPPCOMPILED);
         if (cint_property && (cint_property != clang_property)) {
            if (gDebug > 0) {
               fprintf(stderr, "VALIDITY: tcling_ClassInfo::Property: %s  "
                       "cint: 0x%lx  clang: 0x%lx\n", fClassInfo->Fullname(),
                       cint_property, clang_property);
            }
            return fClassInfo->Property();
         }
      }
      return property;
   }
   // Note: Now we have class, enum, struct, union only.
   const clang::TagDecl* TD = llvm::dyn_cast<clang::TagDecl>(fDecl);
   if (!TD) {
      if (IsValidCint()) {
         long cint_property = fClassInfo->Property();
         cint_property &= ~static_cast<long>(G__BIT_ISCPPCOMPILED);
         if (cint_property != 0L) {
            if (gDebug > 0) {
               fprintf(stderr, "VALIDITY: tcling_ClassInfo::Property: %s  "
                       "cint: 0x%lx  clang: 0x%lx\n", fClassInfo->Fullname(),
                       cint_property, 0L);
            }
            return fClassInfo->Property();
         }
      }
      return 0L;
   }
   if (TD->isEnum()) {
      property |= G__BIT_ISENUM;
      if (IsValidCint()) {
         long cint_property = fClassInfo->Property();
         cint_property &= ~static_cast<long>(G__BIT_ISCPPCOMPILED);
         long clang_property = property;
         clang_property &= ~static_cast<long>(G__BIT_ISCPPCOMPILED);
         if (cint_property && (cint_property != clang_property)) {
            if (gDebug > 0) {
               fprintf(stderr, "VALIDITY: tcling_ClassInfo::Property: %s  "
                       "cint: 0x%lx  clang: 0x%lx\n", fClassInfo->Fullname(),
                       cint_property, clang_property);
            }
            return fClassInfo->Property();
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
   if (IsValidCint()) {
      long cint_property = fClassInfo->Property();
      cint_property &= ~static_cast<long>(G__BIT_ISCPPCOMPILED);
      cint_property &= ~static_cast<long>(G__BIT_ISABSTRACT);
      long clang_property = property;
      clang_property &= ~static_cast<long>(G__BIT_ISCPPCOMPILED);
      clang_property &= ~static_cast<long>(G__BIT_ISABSTRACT);
      if (cint_property && (cint_property != clang_property)) {
         if (gDebug > 0) {
            fprintf(stderr, "VALIDITY: tcling_ClassInfo::Property: %s  "
                    "cint: 0x%lx  clang: 0x%lx\n", fClassInfo->Fullname(),
                    cint_property, clang_property);
         }
      }
   }
   return property;
}

int tcling_ClassInfo::RootFlag() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      return fClassInfo->RootFlag();
   }
   // TODO: Fix this when rootcling provides the value.
   if (IsValidCint()) {
      return fClassInfo->RootFlag();
   }
   return 0;
}

int tcling_ClassInfo::Size() const
{
   if (!IsValid()) {
      return -1;
   }
   if (!IsValidClang()) {
      return fClassInfo->Size();
   }
   clang::Decl::Kind DK = fDecl->getKind();
   if (DK == clang::Decl::Namespace) {
      // Namespaces are special for cint.
      if (IsValidCint()) {
         int cint_size = fClassInfo->Size();
         if ((cint_size != 0) && (cint_size != 1)) {
            if (gDebug > 0) {
               fprintf(stderr,
                  "VALIDITY: tcling_ClassInfo::Size: namespace %s  "
                  "cint: %d  clang: %d\n", fClassInfo->Fullname(),
                  cint_size, 1);
            }
            return cint_size;
         }
      }
      return 1;
   }
   else if (DK == clang::Decl::Enum) {
      // Enums are special for cint.
      if (IsValidCint()) {
         int cint_size = fClassInfo->Size();
         if ((cint_size != 0) && (cint_size != 4)) {
            if (gDebug > 0) {
               fprintf(stderr,
                  "VALIDITY: tcling_ClassInfo::Size: enum %s  cint: "
                  "%d  clang: %d\n", fClassInfo->Fullname(), cint_size, 0);
            }
            return cint_size;
         }
      }
      return 0;
   }
   const clang::RecordDecl* RD = llvm::dyn_cast<clang::RecordDecl>(fDecl);
   if (!RD) {
      // Should not happen.
      if (IsValidCint()) {
         int cint_size = fClassInfo->Size();
         if (cint_size != -1) {
            if (gDebug > 0) {
               fprintf(stderr,
                  "VALIDITY: tcling_ClassInfo::Size: %s  cint: %d  "
                  "clang: %d\n", fClassInfo->Fullname(), cint_size, -1);
            }
            return cint_size;
         }
      }
      return -1;
   }
   clang::ASTContext& Context = fDecl->getASTContext();
   const clang::ASTRecordLayout& Layout = Context.getASTRecordLayout(RD);
   int64_t size = Layout.getSize().getQuantity();
   int clang_size = static_cast<int>(size);
   if (IsValidCint()) {
      int cint_size = fClassInfo->Size();
      if (cint_size && (cint_size != clang_size)) {
         if (gDebug > 0) {
            fprintf(stderr,
               "VALIDITY: tcling_ClassInfo::Size: %s  cint: %d  "
               "clang: %d\n", fClassInfo->Fullname(), cint_size, clang_size);
         }
         return cint_size;
      }
   }
   return clang_size;
}

long tcling_ClassInfo::Tagnum() const
{
   // Note: This *must* return a *cint* tagnum for now.
   if (!IsValid()) {
      return -1L;
   }
   if (!IsValidClang()) {
      return fClassInfo->Tagnum();
   }
   // TODO: What could this possibly mean for clang?
   if (IsValidCint()) {
      return fClassInfo->Tagnum();
   }
   return -1;
}

const char* tcling_ClassInfo::FileName() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      return fClassInfo->FileName();
   }
   // TODO: Fix this when rootcling provides the information.
   if (IsValidCint()) {
      return fClassInfo->FileName();
   }
   return 0;
}

const char* tcling_ClassInfo::FullName() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      return fClassInfo->Fullname();
   }
   // Note: This *must* be static because we are returning a pointer inside it!
   static std::string buf;
   buf.clear();
   clang::PrintingPolicy P(fDecl->getASTContext().getPrintingPolicy());
   llvm::dyn_cast<clang::NamedDecl>(fDecl)->
      getNameForDiagnostic(buf, P, /*Qualified=*/true);
   if (IsValidCint()) {
      const char* cint_fullname = fClassInfo->Fullname();
      if (buf != cint_fullname) {
         if (gDebug > 0) {
            fprintf(stderr, "VALIDITY: tcling_ClassInfo::FullName: cint: %s  "
                    "clang: %s\n", cint_fullname, buf.c_str());
         }
         return cint_fullname;
      }
   }
   return buf.c_str();
}

const char* tcling_ClassInfo::Name() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      return fClassInfo->Name();
   }
   // Note: This *must* be static because we are returning a pointer inside it!
   static std::string buf;
   buf.clear();
   clang::PrintingPolicy P(fDecl->getASTContext().getPrintingPolicy());
   llvm::dyn_cast<clang::NamedDecl>(fDecl)->
      getNameForDiagnostic(buf, P, /*Qualified=*/false);
   if (IsValidCint()) {
      const char* cint_name = fClassInfo->Name();
      if (buf != cint_name) {
         if (gDebug > 0) {
            fprintf(stderr, "VALIDITY: tcling_ClassInfo::Name: cint: %s  "
                    "clang: %s\n", cint_name, buf.c_str());
         }
         return cint_name;
      }
   }
   return buf.c_str();
}

const char* tcling_ClassInfo::Title() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      return fClassInfo->Title();
   }
   // TODO: We need to get this by enhancing the lexer
   //       to retain class comments.
   if (IsValidCint()) {
      return fClassInfo->Title();
   }
   return 0;
}

const char* tcling_ClassInfo::TmpltName() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      return fClassInfo->TmpltName();
   }
   // Note: This *must* be static because we are returning a pointer inside it!
   static std::string buf;
   buf.clear();
   // Note: This does *not* include the template arguments!
   buf = llvm::dyn_cast<clang::NamedDecl>(fDecl)->getNameAsString();
   if (IsValidCint()) {
      const char* cint_tmpltname = fClassInfo->TmpltName();
      if (buf != cint_tmpltname) {
         if (gDebug > 0) {
            fprintf(stderr, "VALIDITY: tcling_ClassInfo::TmpltName: cint: %s  "
                    "clang: %s\n", cint_tmpltname, buf.c_str());
         }
         return cint_tmpltname;
      }
   }
   return buf.c_str();
}

