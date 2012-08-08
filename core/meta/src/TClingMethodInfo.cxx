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
// TClingMethodInfo                                                     //
//                                                                      //
// Emulation of the CINT MethodInfo class.                              //
//                                                                      //
// The CINT C++ interpreter provides an interface to metadata about     //
// a function through the MethodInfo class.  This class provides the    //
// same functionality, using an interface as close as possible to       //
// MethodInfo but the typedef metadata comes from the Clang C++         //
// compiler, not CINT.                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClingMethodInfo.h"

tcling_MethodInfo::~tcling_MethodInfo()
{
   delete fMethodInfo;
   fMethodInfo = 0;
   fInterp = 0;
   //fContexts.clear();
   //fFirstTime = true;
   //fContextIdx = 0U;
   //fIter = clang::DeclContext::decl_iterator();
}

tcling_MethodInfo::tcling_MethodInfo(cling::Interpreter* interp)
   : fMethodInfo(0), fInterp(interp), fFirstTime(true), fContextIdx(0U)
{
   fMethodInfo = new G__MethodInfo();
}

tcling_MethodInfo::tcling_MethodInfo(cling::Interpreter* interp,
                                     G__MethodInfo* info)
   : fMethodInfo(0), fInterp(interp), fFirstTime(true), fContextIdx(0U)
{
   fMethodInfo = new G__MethodInfo(*info);
   // Note: We leave the clang part invalid, this routine can only
   //       be used when there is no clang decl for the containing class.
}

tcling_MethodInfo::tcling_MethodInfo(cling::Interpreter* interp,
                                     tcling_ClassInfo* tcling_class_info)
   : fMethodInfo(0), fInterp(interp), fFirstTime(true), fContextIdx(0U)
{
   if (!tcling_class_info || !tcling_class_info->IsValid()) {
      fMethodInfo = new G__MethodInfo();
      return;
   }
   fMethodInfo = new G__MethodInfo();
   if (gAllowCint) {
      fMethodInfo->Init(*tcling_class_info->GetClassInfo());
   }
   if (gAllowClang) {
      clang::DeclContext* DC = llvm::cast<clang::DeclContext>(
                                  const_cast<clang::Decl*>(tcling_class_info->GetDecl()));
      DC->collectAllContexts(fContexts);
      fIter = DC->decls_begin();
      InternalNext();
   }
}

tcling_MethodInfo::tcling_MethodInfo(const tcling_MethodInfo& rhs)
   : fMethodInfo(0), fInterp(rhs.fInterp), fContexts(rhs.fContexts),
     fFirstTime(rhs.fFirstTime), fContextIdx(rhs.fContextIdx),
     fIter(rhs.fIter)
{
   fMethodInfo = new G__MethodInfo(*rhs.fMethodInfo);
}

tcling_MethodInfo& tcling_MethodInfo::operator=(const tcling_MethodInfo& rhs)
{
   if (this != &rhs) {
      delete fMethodInfo;
      fMethodInfo = new G__MethodInfo(*rhs.fMethodInfo);
      fInterp = rhs.fInterp;
      fContexts = rhs.fContexts;
      fFirstTime = rhs.fFirstTime;
      fContextIdx = rhs.fContextIdx;
      fIter = rhs.fIter;
   }
   return *this;
}

G__MethodInfo* tcling_MethodInfo::GetMethodInfo() const
{
   return fMethodInfo;
}

const clang::FunctionDecl* tcling_MethodInfo::GetMethodDecl() const
{
   if (!gAllowClang) {
      return 0;
   }
   if (!IsValidClang()) {
      return 0;
   }
   const clang::FunctionDecl* FD = llvm::dyn_cast<clang::FunctionDecl>(*fIter);
   return FD;
}

void tcling_MethodInfo::CreateSignature(TString& signature) const
{
   signature = "(";
   if (!IsValid()) {
      signature += ")";
      return;
   }
   tcling_MethodArgInfo arg(fInterp, this);
   int idx = 0;
   while (arg.Next()) {
      if (idx) {
         signature += ", ";
      }
      signature += arg.Type()->Name();
      if (arg.Name() && strlen(arg.Name())) {
         signature += " ";
         signature += arg.Name();
      }
      if (arg.DefaultValue()) {
         signature += " = ";
         signature += arg.DefaultValue();
      }
      ++idx;
   }
   signature += ")";
}

void tcling_MethodInfo::Init(const clang::FunctionDecl* decl)
{
   delete fMethodInfo;
   fMethodInfo = new G__MethodInfo;
   fContexts.clear();
   fFirstTime = true;
   fContextIdx = 0U;
   fIter = clang::DeclContext::decl_iterator();
   if (!decl) {
      return;
   }
   if (!gAllowClang) {
      return;
   }
   clang::DeclContext* DC =
      const_cast<clang::DeclContext*>(decl->getDeclContext());
   DC = DC->getPrimaryContext();
   DC->collectAllContexts(fContexts);
   fIter = DC->decls_begin();
   while (InternalNext()) {
      if (*fIter == decl) {
         break;
      }
   }
   // FIXME: What about fMethodInfo?
}

void* tcling_MethodInfo::InterfaceMethod() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         G__InterfaceMethod p = fMethodInfo->InterfaceMethod();
         if (!p) {
            struct G__bytecodefunc* bytecode = fMethodInfo->GetBytecode();
            if (bytecode) {
               p = (G__InterfaceMethod) G__exec_bytecode;
            }
         }
         return (void*) p;
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   tcling_CallFunc cf(fInterp);
   cf.SetFunc(this);
   return cf.InterfaceMethod();
}

bool tcling_MethodInfo::IsValidCint() const
{
   if (gAllowCint) {
      return fMethodInfo->IsValid();
   }
   return false;
}

bool tcling_MethodInfo::IsValidClang() const
{
   if (gAllowClang) {
      return *fIter;
   }
   return false;
}

bool tcling_MethodInfo::IsValid() const
{
   return IsValidClang() || IsValidCint();
}

int tcling_MethodInfo::NArg() const
{
   if (!IsValid()) {
      return -1;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fMethodInfo->NArg();
      }
      return -1;
   }
   if (!gAllowClang) {
      return -1;
   }
   const clang::FunctionDecl* FD = llvm::cast<clang::FunctionDecl>(*fIter);
   unsigned num_params = FD->getNumParams();
   if (gAllowCint) {
      if (IsValidCint()) {
         int cint_val = fMethodInfo->NArg();
         int clang_val = static_cast<int>(num_params);
         if (clang_val != cint_val) {
            if (gDebug > 0) {
               fprintf(stderr,
                       "VALIDITY: tcling_MethodInfo::NArg: cint: %d  "
                       "clang: %d\n", cint_val, clang_val);
            }
         }
      }
   }
   return static_cast<int>(num_params);
}

int tcling_MethodInfo::NDefaultArg() const
{
   if (!IsValid()) {
      return -1;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fMethodInfo->NDefaultArg();
      }
      return -1;
   }
   if (!gAllowClang) {
      return -1;
   }
   const clang::FunctionDecl* FD = llvm::cast<clang::FunctionDecl>(*fIter);
   unsigned num_params = FD->getNumParams();
   unsigned min_args = FD->getMinRequiredArguments();
   unsigned defaulted_params = num_params - min_args;
   if (gAllowCint) {
      if (IsValidCint()) {
         int cint_val = fMethodInfo->NDefaultArg();
         int clang_val = static_cast<int>(defaulted_params);
         if (clang_val != cint_val) {
            if (gDebug > 0) {
               fprintf(stderr,
                       "VALIDITY: tcling_MethodInfo::NDefaultArg: cint: %d  "
                       "clang: %d\n", cint_val, clang_val);
            }
         }
      }
   }
   return static_cast<int>(defaulted_params);
}

int tcling_MethodInfo::InternalNext()
{
   if (!*fIter) {
      // Iterator is already invalid.
      return 0;
   }
   while (true) {
      // Advance to the next decl.
      if (fFirstTime) {
         // The cint semantics are weird.
         fFirstTime = false;
      }
      else {
         ++fIter;
      }
      // Fix it if we have gone past the end of the current decl context.
      while (!*fIter) {
         ++fContextIdx;
         if (fContextIdx >= fContexts.size()) {
            // Iterator is now invalid.
            return 0;
         }
         clang::DeclContext* DC = fContexts[fContextIdx];
         fIter = DC->decls_begin();
         if (*fIter) {
            // Good, a non-empty context.
            break;
         }
      }
      // Return if this decl is a function or method.
      if (llvm::isa<clang::FunctionDecl>(*fIter)) {
         // Iterator is now valid.
         return 1;
      }
   }
}

int tcling_MethodInfo::Next()
{
   if (!gAllowClang) {
      if (gAllowCint) {
         return fMethodInfo->Next();
      }
      return 0;
   }
   return InternalNext();
}

long tcling_MethodInfo::Property() const
{
   if (!IsValid()) {
      return 0L;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fMethodInfo->Property();
      }
      return 0L;
   }
   if (!gAllowClang) {
      return 0L;
   }
   long property = 0L;
   property |= G__BIT_ISCOMPILED;
   const clang::FunctionDecl* FD =
      llvm::dyn_cast<clang::FunctionDecl>(*fIter);
   switch (FD->getAccess()) {
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
   if (FD->getStorageClass() == clang::SC_Static) {
      property |= G__BIT_ISSTATIC;
   }
   clang::QualType QT = FD->getResultType().getCanonicalType();
   if (QT.isConstQualified()) {
      property |= G__BIT_ISCONSTANT;
   }
   while (1) {
      if (QT->isArrayType()) {
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
   if (QT.isConstQualified()) {
      property |= G__BIT_ISCONSTANT;
   }
   if (const clang::CXXMethodDecl* MD =
            llvm::dyn_cast<clang::CXXMethodDecl>(FD)) {
      if (MD->getTypeQualifiers() & clang::Qualifiers::Const) {
         property |= G__BIT_ISCONSTANT | G__BIT_ISMETHCONSTANT;
      }
      if (MD->isVirtual()) {
         property |= G__BIT_ISVIRTUAL;
      }
      if (MD->isPure()) {
         property |= G__BIT_ISPUREVIRTUAL;
      }
      if (const clang::CXXConstructorDecl* CD =
               llvm::dyn_cast<clang::CXXConstructorDecl>(MD)) {
         if (CD->isExplicit()) {
            property |= G__BIT_ISEXPLICIT;
         }
      }
      else if (const clang::CXXConversionDecl* CD =
                  llvm::dyn_cast<clang::CXXConversionDecl>(MD)) {
         if (CD->isExplicit()) {
            property |= G__BIT_ISEXPLICIT;
         }
      }
   }
   if (gAllowCint) {
      if (IsValidCint()) {
         long cint_property = fMethodInfo->Property();
         if (property != cint_property) {
            if (gDebug > 0) {
               fprintf(stderr,
                       "VALIDITY: tcling_MethodInfo::Property: "
                       "cint: 0x%lx  clang: 0x%lx\n",
                       (unsigned long) cint_property,
                       (unsigned long) property);
            }
         }
      }
   }
   return property;
}

tcling_TypeInfo* tcling_MethodInfo::Type() const
{
   static tcling_TypeInfo ti(fInterp);
   ti.Init(clang::QualType());
   if (!IsValidClang()) {
      if (gAllowCint) {
         ti.Init(fMethodInfo->Type()->Name());
      }
      return &ti;
   }
   if (!gAllowClang) {
      return &ti;
   }
   clang::QualType QT = llvm::cast<clang::FunctionDecl>(*fIter)->
                        getResultType();
   ti.Init(QT);
   if (gAllowCint) {
      if (IsValidCint()) {
         const char* cint_name = fMethodInfo->Type()->Name();
         const char* clang_name = ti.Name();
         if (clang_name != cint_name) {
            if (gDebug > 0) {
               fprintf(stderr,
                       "VALIDITY: tcling_MethodInfo::Type: cint: %s  "
                       "clang: %s\n", cint_name, clang_name);
            }
         }
      }
   }
   return &ti;
}

const char* tcling_MethodInfo::GetMangledName() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fMethodInfo->GetMangledName();
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   const char* fname = 0;
   static std::string mangled_name;
   mangled_name.clear();
   llvm::raw_string_ostream OS(mangled_name);
   llvm::OwningPtr<clang::MangleContext> Mangle(fIter->getASTContext().
         createMangleContext());
   const clang::NamedDecl* ND = llvm::dyn_cast<clang::NamedDecl>(*fIter);
   if (!ND) {
      return 0;
   }
   if (!Mangle->shouldMangleDeclName(ND)) {
      clang::IdentifierInfo* II = ND->getIdentifier();
      fname = II->getNameStart();
   }
   else {
      if (const clang::CXXConstructorDecl* D =
               llvm::dyn_cast<clang::CXXConstructorDecl>(ND)) {
         //Ctor_Complete,          // Complete object ctor
         //Ctor_Base,              // Base object ctor
         //Ctor_CompleteAllocating // Complete object allocating ctor (unused)
         Mangle->mangleCXXCtor(D, clang::Ctor_Complete, OS);
      }
      else if (const clang::CXXDestructorDecl* D =
                  llvm::dyn_cast<clang::CXXDestructorDecl>(ND)) {
         //Dtor_Deleting, // Deleting dtor
         //Dtor_Complete, // Complete object dtor
         //Dtor_Base      // Base object dtor
         Mangle->mangleCXXDtor(D, clang::Dtor_Deleting, OS);
      }
      else {
         Mangle->mangleName(ND, OS);
      }
      OS.flush();
      fname = mangled_name.c_str();
   }
   if (gAllowCint) {
      if (IsValidCint()) {
         const char* cint_val = fMethodInfo->GetMangledName();
         const char* clang_val = fname;
         if (clang_val != cint_val) {
            if (gDebug > 0) {
               fprintf(stderr,
                       "VALIDITY: tcling_MethodInfo::GetMangledName: "
                       "cint: %s  clang: %s\n", cint_val, clang_val);
            }
         }
      }
   }
   return fname;
}

const char* tcling_MethodInfo::GetPrototype() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fMethodInfo->GetPrototype();
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   static std::string buf;
   buf.clear();
   buf += Type()->Name();
   buf += ' ';
   std::string name;
   clang::PrintingPolicy Policy(fIter->getASTContext().getPrintingPolicy());
   const clang::NamedDecl* ND = llvm::cast<clang::NamedDecl>(*fIter);
   ND->getNameForDiagnostic(name, Policy, /*Qualified=*/true);
   buf += name;
   buf += '(';
   tcling_MethodArgInfo arg(fInterp, this);
   int idx = 0;
   while (arg.Next()) {
      if (idx) {
         buf += ", ";
      }
      buf += arg.Type()->Name();
      if (arg.Name() && strlen(arg.Name())) {
         buf += ' ';
         buf += arg.Name();
      }
      if (arg.DefaultValue()) {
         buf += " = ";
         buf += arg.DefaultValue();
      }
      ++idx;
   }
   buf += ')';
   if (gAllowCint) {
      if (IsValidCint()) {
         const char* cint_val  = fMethodInfo->GetPrototype();
         const char* clang_val = buf.c_str();
         if (clang_val != cint_val) {
            if (gDebug > 0) {
               fprintf(stderr,
                       "VALIDITY: tcling_MethodInfo::GetPrototype:  "
                       "cint: %s  clang: %s\n", cint_val, clang_val);
            }
         }
      }
   }
   return buf.c_str();
}

const char* tcling_MethodInfo::Name() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fMethodInfo->Name();
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   static std::string buf;
   buf.clear();
   clang::PrintingPolicy Policy(fIter->getASTContext().getPrintingPolicy());
   llvm::dyn_cast<clang::NamedDecl>(*fIter)->
   getNameForDiagnostic(buf, Policy, /*Qualified=*/true);
   if (gAllowCint) {
      if (IsValidCint()) {
         const char* cint_val = fMethodInfo->Name();
         const char* clang_val = buf.c_str();
         if (clang_val != cint_val) {
            if (gDebug > 0) {
               fprintf(stderr,
                       "VALIDITY: tcling_MethodInfo::Name: "
                       "cint: %s  clang: %s\n", cint_val, clang_val);
            }
         }
      }
   }
   return buf.c_str();
}

const char* tcling_MethodInfo::TypeName() const
{
   if (!IsValid()) {
      // FIXME: Cint does not check!
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fMethodInfo->Type()->Name();
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   return Type()->Name();
}

const char* tcling_MethodInfo::Title() const
{
   // FIXME: Implement this when we have comment parsing!
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fMethodInfo->Title();
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   return "";
}

