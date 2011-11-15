// @(#)root/meta:$Id$
// Author: Paul Russo, 2009-10-06

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// This class defines an interface to the CINT C/C++ interpreter made   //
// by Masaharu Goto from HP Japan, using cling as interpreter backend.  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TCintWithCling.h"
#include "TROOT.h"
#include "TApplication.h"
#include "TGlobal.h"
#include "TDataType.h"
#include "TClass.h"
#include "TClassEdit.h"
#include "TClassTable.h"
#include "TBaseClass.h"
#include "TDataMember.h"
#include "TMethod.h"
#include "TMethodArg.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TString.h"
#include "THashList.h"
#include "TOrdCollection.h"
#include "TVirtualPad.h"
#include "TSystem.h"
#include "TVirtualMutex.h"
#include "TError.h"
#include "TEnv.h"
#include "THashTable.h"
#include "RConfigure.h"
#include "compiledata.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/HeaderSearchOptions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Serialization/ASTReader.h"
#include "cling/Interpreter/Interpreter.h"
#include "cling/MetaProcessor/MetaProcessor.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <set>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

#include <cxxabi.h>
#include <limits.h>

#ifdef __APPLE__
#include <dlfcn.h>
#endif // __APPLE__

using namespace std;

R__EXTERN int optind;

//______________________________________________________________________________
//
//
//

void* autoloadCallback(const std::string& mangled_name)
{
   // Autoload a library. Given a mangled function name find the
   // library which provides the function and load it.
   //--
   //
   //  Use the C++ ABI provided function to demangle the function name.
   //
   int err = 0;
   char* demangled_name = abi::__cxa_demangle(mangled_name.c_str(), 0, 0, &err);
   if (err) {
      return 0;
   }
   //fprintf(stderr, "demangled name: '%s'\n", demangled_name);
   //
   //  Separate out the class or namespace part of the
   //  function name.
   //
   std::string name(demangled_name);
   // Remove the function arguments.
   std::string::size_type pos = name.rfind('(');
   if (pos != std::string::npos) {
      name.erase(pos);
   }
   // Remove the function name.
   pos = name.rfind(':');
   if (pos != std::string::npos) {
      if ((pos != 0) && (name[pos-1] == ':')) {
         name.erase(pos-1);
      }
   }
   //fprintf(stderr, "name: '%s'\n", name.c_str());
   // Now we have the class or namespace name, so do the lookup.
   TString libs = gCint->GetClassSharedLibs(name.c_str());
   if (libs.IsNull()) {
      // Not found in the map, all done.
      return 0;
   }
   //fprintf(stderr, "library: %s\n", iter->second.c_str());
   // Now we have the name of the libraries to load, so load them.
   
   TString lib;
   Ssiz_t posLib = 0;
   while (libs.Tokenize(lib, posLib)) {
      std::string errmsg;
      bool load_failed = llvm::sys::DynamicLibrary::LoadLibraryPermanently(lib, &errmsg);
      if (load_failed) {
         // The library load failed, all done.
         //fprintf(stderr, "load failed: %s\n", errmsg.c_str());
         return 0;
      }
   }
   //fprintf(stderr, "load succeeded.\n");
   // Get the address of the function being called.
   void* addr = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(mangled_name.c_str());
   //fprintf(stderr, "addr: %016lx\n", reinterpret_cast<unsigned long>(addr));
   return addr;
}

//______________________________________________________________________________
//
//
//

class tcling_MethodInfo;

class tcling_ClassInfo {
public: // Types
   enum MatchMode {
      ExactMatch = 0,
      ConversionMatch = 1,
      ConversionMatchBytecode = 2
   };
   enum InheritanceMode {
      InThisScope = 0,
      WithInheritance = 1
   }; 
public:
   ~tcling_ClassInfo();
   explicit tcling_ClassInfo();
   explicit tcling_ClassInfo(const char*);
   explicit tcling_ClassInfo(const clang::Decl*);
   tcling_ClassInfo(const tcling_ClassInfo&);
   tcling_ClassInfo& operator=(const tcling_ClassInfo&);
   G__ClassInfo* GetClassInfo() const;
   const clang::Decl* GetDecl() const;
   //int GetIdx() const;
   long ClassProperty() const;
   void Delete(void* arena) const;
   void DeleteArray(void* arena, bool dtorOnly) const;
   void Destruct(void* arena) const;
   int NMethods() const;
   tcling_MethodInfo* GetMethod(const char* fname, const char* arg,
      long* poffset, MatchMode mode = ConversionMatch,
      InheritanceMode imode = WithInheritance) const;
   int GetMethodNArg(const char* method, const char* proto) const;
   bool HasDefaultConstructor() const;
   bool HasMethod(const char* name) const;
   void Init(const char* name);
   void Init(int tagnum);
   bool IsBase(const char* name) const;
   static bool IsEnum(const char* name);
   bool IsLoaded() const;
   bool IsValid() const;
   bool IsValidCint() const;
   bool IsValidClang() const;
   bool IsValidMethod(const char* method, const char* proto,
                      long* offset) const;
   int Next() const;
   void* New() const;
   void* New(int n) const;
   void* New(int n, void* arena) const;
   void* New(void* arena) const;
   long Property() const;
   int RootFlag() const;
   int Size() const;
   long Tagnum() const;
   const char* FileName() const;
   const char* FullName() const;
   const char* Name() const;
   const char* Title() const;
   const char* TmpltName() const;
private:
   //
   //  CINT material
   //
   /// CINT class info for this class, we own.
   G__ClassInfo* fClassInfo;
   //
   //  Cling material
   //
   /// Clang AST Node for this class, we do *not* own.
   const clang::Decl* fDecl;
   /// Our iterator, index into tcling class table.
   //int fIdx;
};

class tcling_BaseClassInfo {
public:
   ~tcling_BaseClassInfo();
   explicit tcling_BaseClassInfo(); // NOT IMPLEMENTED.
   explicit tcling_BaseClassInfo(tcling_ClassInfo*);
   tcling_BaseClassInfo(const tcling_BaseClassInfo&);
   tcling_BaseClassInfo& operator=(const tcling_BaseClassInfo&);
   G__BaseClassInfo* GetBaseClassInfo() const;
   tcling_ClassInfo* GetDerivedClassInfo() const;
   tcling_ClassInfo* GetClassInfo() const;
   long GetOffsetBase() const;
   int InternalNext(int onlyDirect);
   int Next();
   int Next(int onlyDirect);
   long Offset() const;
   long Property() const;
   long Tagnum() const;
   const char* FullName() const;
   const char* Name() const;
   const char* TmpltName() const;
   bool IsValid() const;
private:
   //
   // CINT material.
   //
   /// CINT base class info, we own.
   G__BaseClassInfo* fBaseClassInfo;
   //
   // Cling material.
   //
   /// Class we were intialized with, we own.
   tcling_ClassInfo* fDerivedClassInfo;
   /// Flag to provide Cint semantics for iterator advancement (not first time)
   bool fFirstTime;
   /// Flag for signaling the need to descend on this advancement.
   bool fDescend;
   /// Current class whose bases we are iterating through, we do *not* own.
   const clang::Decl* fDecl;
   /// Current iterator.
   clang::CXXRecordDecl::base_class_const_iterator fIter;
   /// Class info of base class our iterator is currently pointing at, we own.
   tcling_ClassInfo* fClassInfo;
   /// Iterator stack.
   std::vector < std::pair < std::pair < const clang::Decl*,
       clang::CXXRecordDecl::base_class_const_iterator > , long > > fIterStack;
   /// Offset of the current base, fDecl, in the most-derived class.
   long fOffset;
};

class tcling_DataMemberInfo {
public:
   ~tcling_DataMemberInfo();
   explicit tcling_DataMemberInfo();
   explicit tcling_DataMemberInfo(tcling_ClassInfo*);
   tcling_DataMemberInfo(const tcling_DataMemberInfo&);
   tcling_DataMemberInfo& operator=(const tcling_DataMemberInfo&);
   G__DataMemberInfo* GetDataMemberInfo() const;
   G__ClassInfo* GetClassInfo() const;
   tcling_ClassInfo* GetTClingClassInfo() const;
   clang::Decl* GetDecl() const;
   int GetIdx() const;
   int ArrayDim() const;
   bool IsValid() const;
   int MaxIndex(int dim) const;
   bool Next();
   long Offset() const;
   long Property() const;
   long TypeProperty() const;
   int TypeSize() const;
   const char* TypeName() const;
   const char* TypeTrueName() const;
   const char* Name() const;
   const char* Title() const;
   const char* ValidArrayIndex() const;
private:
   void InternalNextValidMember();
private:
   //
   // CINT material.
   //
   /// CINT data member info, we own.
   G__DataMemberInfo* fDataMemberInfo;
   /// CINT class info, we own.
   G__ClassInfo* fClassInfo;
   //
   // Clang material.
   //
   /// Class we are iterating over, we own.
   tcling_ClassInfo* fTClingClassInfo;
   /// We need to skip the first increment to support the cint Next() semantics.
   bool fFirstTime;
   /// Current decl.
   clang::DeclContext::decl_iterator fIter;
   /// Recursion stack for traversing nested transparent scopes.
   std::vector<clang::DeclContext::decl_iterator> fIterStack;
};

class tcling_TypeInfo {
public:
   ~tcling_TypeInfo();
   explicit tcling_TypeInfo();
   explicit tcling_TypeInfo(const char* name);
   explicit tcling_TypeInfo(G__value* val);
   tcling_TypeInfo(const tcling_TypeInfo&);
   tcling_TypeInfo& operator=(const tcling_TypeInfo&);
   G__TypeInfo* GetTypeInfo() const;
   G__ClassInfo* GetClassInfo() const;
   clang::Decl* GetDecl() const;
   void Init(const char* name);
   bool IsValid() const;
   const char* Name() const;
   long Property() const;
   int RefType() const;
   int Size() const;
   const char* TrueName() const;
private:
   //
   //  CINT part
   //
   /// CINT type info, we own.
   G__TypeInfo* fTypeInfo;
   /// CINT class info, we own.
   G__ClassInfo* fClassInfo;
   //
   //  Cling part
   //
   /// Clang AST Node for the type, we do *not* own.
   clang::Decl* fDecl;
};

class tcling_TypedefInfo {
public:
   ~tcling_TypedefInfo();
   explicit tcling_TypedefInfo();
   explicit tcling_TypedefInfo(const char*);
   tcling_TypedefInfo(const tcling_TypedefInfo&);
   tcling_TypedefInfo& operator=(const tcling_TypedefInfo&);
   G__TypedefInfo* GetTypedefInfo() const;
   clang::Decl* GetDecl() const;
   int GetIdx() const;
   void Init(const char* name);
   bool IsValid() const;
   bool IsValidCint() const;
   bool IsValidClang() const;
   long Property() const;
   int Size() const;
   const char* TrueName() const;
   const char* Name() const;
   const char* Title() const;
   int Next();
private:
   //
   //  CINT info.
   //
   /// CINT typedef info for this class, we own.
   G__TypedefInfo* fTypedefInfo;
   //
   //  clang info.
   //
   /// Clang AST Node for this typedef, we do *not* own.
   clang::Decl* fDecl;
   /// Index in typedef table of fDecl.
   int fIdx;
};

class tcling_MethodInfo {
public:
   ~tcling_MethodInfo();
   explicit tcling_MethodInfo();
   explicit tcling_MethodInfo(G__MethodInfo* info); // FIXME
   explicit tcling_MethodInfo(tcling_ClassInfo*);
   tcling_MethodInfo(const tcling_MethodInfo&);
   tcling_MethodInfo& operator=(const tcling_MethodInfo&);
   G__MethodInfo* GetMethodInfo() const;
   void CreateSignature(TString& signature) const;
   G__InterfaceMethod InterfaceMethod() const;
   bool IsValid() const;
   int NArg() const;
   int NDefaultArg() const;
   int Next() const;
   long Property() const;
   void* Type() const;
   const char* GetMangledName() const;
   const char* GetPrototype() const;
   const char* Name() const;
   const char* TypeName() const;
   const char* Title() const;
private:
   //
   // CINT material.
   //
   /// cint method iterator, we own.
   G__MethodInfo* fMethodInfo;
   //
   // Cling material.
   //
   /// Class, namespace, or translation unit we were initialized with.
   tcling_ClassInfo* fInitialClassInfo;
   /// Class, namespace, or translation unit we are iterating over now.
   clang::Decl* fDecl;
   /// Our iterator.
   clang::DeclContext::decl_iterator fIter;
   /// Our iterator's current function.
   clang::Decl* fFunction;
};

class tcling_MethodArgInfo {
public:
   ~tcling_MethodArgInfo();
   explicit tcling_MethodArgInfo();
   explicit tcling_MethodArgInfo(tcling_MethodInfo*);
   tcling_MethodArgInfo(const tcling_MethodArgInfo&);
   tcling_MethodArgInfo& operator=(const tcling_MethodArgInfo&);
   G__MethodInfo* GetMethodArgInfo() const;
   bool IsValid() const;
   int Next() const;
   long Property() const;
   const char* DefaultValue() const;
   const char* Name() const;
   const char* TypeName() const;
private:
   //
   // CINT material.
   //
   /// cint method argument iterator, we own.
   G__MethodArgInfo* fMethodArgInfo;
   //
   // Cling material.
   //
};

class tcling_CallFunc {
public:
   ~tcling_CallFunc();
   explicit tcling_CallFunc();
   tcling_CallFunc(const tcling_CallFunc&);
   tcling_CallFunc& operator=(const tcling_CallFunc&);
   void Exec(void* address) const;
   long ExecInt(void* address) const;
   long ExecInt64(void* address) const;
   double ExecDouble(void* address) const;
   void* FactoryMethod() const;
   void Init() const;
   G__InterfaceMethod InterfaceMethod() const;
   bool IsValid() const;
   void ResetArg() const;
   void SetArg(long param) const;
   void SetArg(double param) const;
   void SetArg(long long param) const;
   void SetArg(unsigned long long param) const;
   void SetArgArray(long* paramArr, int nparam) const;
   void SetArgs(const char* param) const;
   void SetFunc(tcling_ClassInfo* info, const char* method, const char* params, long* offset) const;
   void SetFunc(tcling_MethodInfo* info) const;
   void SetFuncProto(tcling_ClassInfo* info, const char* method, const char* proto, long* offset) const;
private:
   //
   // CINT material.
   //
   /// cint method iterator, we own.
   G__CallFunc* fCallFunc;
   //
   // Cling material.
   //
};

//______________________________________________________________________________
//
//
//

tcling_ClassInfo::~tcling_ClassInfo()
{
   delete fClassInfo;
   fClassInfo = 0;
   fDecl = 0;
}

tcling_ClassInfo::tcling_ClassInfo()
   : fClassInfo(new G__ClassInfo), fDecl(0)//, fIdx(-1)
{
}

tcling_ClassInfo::tcling_ClassInfo(const tcling_ClassInfo& rhs)
{
   fClassInfo = new G__ClassInfo(rhs.fClassInfo->Tagnum());
   fDecl = rhs.fDecl;
   //fIdx = rhs.fIdx;
}

tcling_ClassInfo& tcling_ClassInfo::operator=(const tcling_ClassInfo& rhs)
{
   if (this != &rhs) {
      delete fClassInfo;
      fClassInfo = new G__ClassInfo(rhs.fClassInfo->Tagnum());
      fDecl = rhs.fDecl;
      //fIdx = rhs.fIdx;
   }
   return *this;
}

tcling_ClassInfo::tcling_ClassInfo(const char* name)
   : fClassInfo(0), fDecl(0)//, fIdx(-1)
{
   fprintf(stderr, "tcling_ClassInfo(name): looking up class name: %s\n", name);
   fClassInfo = new G__ClassInfo(name);
#if 0
   if (!fClassInfo->IsValid()) {
      fprintf(stderr, "tcling_ClassInfo(name): could not find cint class for "
              "name: %s\n", name);
   }
   else {
      fprintf(stderr, "tcling_ClassInfo(name): found cint class for "
              "name: %s  tagnum: %d\n", name, fClassInfo->Tagnum());
   }
   std::multimap<const std::string, const clang::Decl*>::iterator iter =
      tcling_Dict::ClassNameToDecl()->find(name);
   if (iter == tcling_Dict::ClassNameToDecl()->end()) {
      fprintf(stderr, "tcling_ClassInfo(name): cling class not found "
              "name: %s\n", name);
   }
   else {
      fDecl = (clang::Decl*) iter->second;
      fprintf(stderr, "tcling_ClassInfo(name): found cling class name: %s  "
              "decl: 0x%lx\n", name, (long) fDecl);
      std::map<const clang::Decl*, int>::iterator iter_idx =
         tcling_Dict::ClassDeclToIdx()->find(fDecl);
      if (iter_idx == tcling_Dict::ClassDeclToIdx()->end()) {
         fprintf(stderr, "tcling_ClassInfo(name): could not find idx "
                 "for name: %s  decl: 0x%lx\n", name, (long) fDecl);
      }
      else {
         fIdx = iter_idx->second;
         fprintf(stderr, "tcling_ClassInfo(name): found idx: %d for "
                 "name: %s  decl: 0x%lx\n", fIdx, name, (long) fDecl);
      }
   }
#endif // 0
   //--
}

tcling_ClassInfo::tcling_ClassInfo(const clang::Decl* decl)
   : fClassInfo(0), fDecl(decl)//, fIdx(-1)
{
   std::string buf;
   clang::PrintingPolicy P(fDecl->getASTContext().getPrintingPolicy());
   llvm::dyn_cast<clang::NamedDecl>(fDecl)->getNameForDiagnostic(buf, P, true);
   fClassInfo = new G__ClassInfo(buf.c_str());
#if 0
   if (!fClassInfo->IsValid()) {
      fprintf(stderr, "tcling_ClassInfo(decl): could not find cint class for "
              "name: %s  decl: 0x%lx\n", buf.c_str(), (long) fDecl);
   }
   else {
      fprintf(stderr, "tcling_ClassInfo(decl): found cint class for "
              "name: %s  tagnum: %d\n", buf.c_str(), fClassInfo->Tagnum());
   }
   fprintf(stderr, "tcling_ClassInfo(decl): looking up class name: %s  "
           "decl: 0x%lx\n", buf.c_str(), (long) fDecl);
   std::map<const clang::Decl*, int>::iterator iter_idx =
      tcling_Dict::ClassDeclToIdx()->find(fDecl);
   if (iter_idx == tcling_Dict::ClassDeclToIdx()->end()) {
      fprintf(stderr, "tcling_ClassInfo(decl): could not find idx for "
              "name: %s  decl: 0x%lx\n", buf.c_str(), (long) fDecl);
   }
   else {
      fIdx = iter_idx->second;
      fprintf(stderr, "tcling_ClassInfo(decl): found idx: %d for "
              "name: %s  decl: 0x%lx\n", fIdx, buf.c_str(), (long) fDecl);
   }
#endif // 0
   //--
}

G__ClassInfo* tcling_ClassInfo::GetClassInfo() const
{
   return fClassInfo;
}

const clang::Decl* tcling_ClassInfo::GetDecl() const
{
   return fDecl;
}

#if 0
int tcling_ClassInfo::GetIdx() const
{
   return fIdx;
}
#endif // 0

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
      // Enum or Namespace.
      // The cint interface always returns 0L for these guys.
      return 0L;
   }
   if (RD->isUnion()) {
      // The cint interface always returns 0L for these guys.
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
   else if (!CRD->hasTrivialDefaultConstructor()) {
      property |= G__CLS_HASIMPLICITCTOR;
   }
   if (CRD->hasDeclaredDefaultConstructor()) {
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

void tcling_ClassInfo::Delete(void* arena) const
{
   // Note: This is an interpreter function.
   return fClassInfo->Delete(arena);
}

void tcling_ClassInfo::DeleteArray(void* arena, bool dtorOnly) const
{
   // Note: This is an interpreter function.
   return fClassInfo->DeleteArray(arena, dtorOnly);
}

void tcling_ClassInfo::Destruct(void* arena) const
{
   // Note: This is an interpreter function.
   return fClassInfo->Destruct(arena);
}

int tcling_ClassInfo::NMethods() const
{
   // FIXME: Implement this with clang!
   return fClassInfo->NMethods();
}

tcling_MethodInfo* tcling_ClassInfo::GetMethod(const char* fname,
   const char* arg, long* poffset, MatchMode mode /*= ConversionMatch*/,
   InheritanceMode imode /*= WithInheritance*/) const
{
   // FIXME: Implement this with clang!
   G__MethodInfo* mi = new G__MethodInfo(fClassInfo->GetMethod(
      fname, arg, poffset, (Cint::G__ClassInfo::MatchMode) mode,
      (Cint::G__ClassInfo::InheritanceMode) imode));
   tcling_MethodInfo* tmi = new tcling_MethodInfo(mi);
   delete mi;
   mi = 0;
   return tmi;
}

int tcling_ClassInfo::GetMethodNArg(const char* method, const char* proto) const
{
   // Note: Used only by TQObject.cxx:170 and only for interpreted classes.
   G__MethodInfo meth;
   long offset = 0L;
   meth = fClassInfo->GetMethod(method, proto, &offset);
   if (meth.IsValid()) {
      return meth.NArg();
   }
   return -1;
}

bool tcling_ClassInfo::HasDefaultConstructor() const
{
   // Note: This is a ROOT special!
   return fClassInfo->HasDefaultConstructor();
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
      // Must be an enum or namespace.
      // FIXME: Make it work for a namespace!
      return false;
   }
   std::string given_name(name);
   for (
      clang::CXXRecordDecl::method_iterator M = CRD->method_begin(),
      MEnd = CRD->method_end();
      M != MEnd;
      ++M
   ) {
      if ((*M)->getNameAsString() == given_name) {
         return true;
      }
   }
   return false;
}

void tcling_ClassInfo::Init(const char* name)
{
   fprintf(stderr, "tcling_ClassInfo::Init(name): looking up class: %s\n",
           name);
   fClassInfo = 0;
   fDecl = 0;
   //fIdx = -1;
   fClassInfo->Init(name);
#if 0
   if (!fClassInfo->IsValid()) {
      fprintf(stderr, "tcling_ClassInfo::Init(name): could not find cint "
              "class for name: %s\n", name);
   }
   else {
      fprintf(stderr, "tcling_ClassInfo::Init(name): found cint class for "
              "name: %s  tagnum: %d\n", name, fClassInfo->Tagnum());
   }
   std::multimap<const std::string, const clang::Decl*>::iterator iter =
      tcling_Dict::ClassNameToDecl()->find(name);
   if (iter == tcling_Dict::ClassNameToDecl()->end()) {
      fprintf(stderr, "tcling_ClassInfo::Init(name): cling class not found "
              "name: %s\n", name);
   }
   else {
      fDecl = (clang::Decl*) iter->second;
      fprintf(stderr, "tcling_ClassInfo::Init(name): found cling class "
              "name: %s  decl: 0x%lx\n", name, (long) fDecl);
      std::map<const clang::Decl*, int>::iterator iter_idx =
         tcling_Dict::ClassDeclToIdx()->find(fDecl);
      if (iter_idx == tcling_Dict::ClassDeclToIdx()->end()) {
         fprintf(stderr, "tcling_ClassInfo::Init(name): could not find idx "
                 "for name: %s  decl: 0x%lx\n", name, (long) fDecl);
      }
      else {
         fIdx = iter_idx->second;
         fprintf(stderr, "tcling_ClassInfo::Init(name): found idx: %d for "
                 "name: %s  decl: 0x%lx\n", fIdx, name, (long) fDecl);
      }
   }
#endif // 0
   //--
}

void tcling_ClassInfo::Init(int tagnum)
{
   fprintf(stderr, "tcling_ClassInfo::Init(tagnum): looking up tagnum: %d\n",
           tagnum);
   fClassInfo = 0;
   fDecl = 0;
   //fIdx = -1;
   fClassInfo->Init(tagnum);
#if 0
   if (!fClassInfo->IsValid()) {
      fprintf(stderr, "tcling_ClassInfo::Init(tagnum): could not find cint "
              "class for tagnum: %d\n", tagnum);
      return;
   }
   const char* name = fClassInfo->Fullname();
   fprintf(stderr, "tcling_ClassInfo::Init(tagnum): found cint class "
           "name: %s  tagnum: %d\n", name, tagnum);
   std::multimap<const std::string, const clang::Decl*>::iterator iter =
      tcling_Dict::ClassNameToDecl()->find(name);
   if (iter == tcling_Dict::ClassNameToDecl()->end()) {
      fprintf(stderr, "tcling_ClassInfo::Init(tagnum): cling class not found "
              "name: %s  tagnum: %d\n", name, tagnum);
   }
   else {
      fDecl = (clang::Decl*) iter->second;
      fprintf(stderr, "tcling_ClassInfo::Init(tagnum): found cling class "
              "name: %s  decl: 0x%lx\n", name, (long) fDecl);
      std::map<const clang::Decl*, int>::iterator iter_idx =
         tcling_Dict::ClassDeclToIdx()->find(fDecl);
      if (iter_idx == tcling_Dict::ClassDeclToIdx()->end()) {
         fprintf(stderr, "tcling_ClassInfo::Init(tagnum): could not find idx "
                 "for name: %s  decl: 0x%lx  tagnum: %d\n", name, (long) fDecl,
                 tagnum);
      }
      else {
         fIdx = iter_idx->second;
         fprintf(stderr, "tcling_ClassInfo::Init(tagnum): found idx: %d for "
                 "name: %s  decl: 0x%lx  tagnum: %d\n", fIdx, name, (long) fDecl,
                 tagnum);
      }
   }
#endif // 0
   //--
}

bool tcling_ClassInfo::IsBase(const char* name) const
{
   if (!IsValid()) {
      return false;
   }
   if (!IsValidClang()) {
      return fClassInfo->IsBase(name);
   }
   tcling_ClassInfo base(name);
   if (!base.IsValid()) {
      return false;
   }
   if (!base.IsValidClang()) {
      return false;
   }
   const clang::CXXRecordDecl* CRD =
      llvm::dyn_cast<clang::CXXRecordDecl>(fDecl);
   if (!CRD) {
      // Must be an enum or namespace.
      return false;
   }
   const clang::CXXRecordDecl* baseCRD =
      llvm::dyn_cast<clang::CXXRecordDecl>(base.GetDecl());
   return CRD->isDerivedFrom(baseCRD);
}

bool tcling_ClassInfo::IsEnum(const char* name)
{
   // Note: This is a static member function.
   tcling_ClassInfo info(name);
   if (info.IsValid() && (info.Property() & G__BIT_ISENUM)) {
      return true;
   }
   return false;
}

bool tcling_ClassInfo::IsLoaded() const
{
   return fClassInfo->IsLoaded();
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
   // Note: Use this when we can get a one-to-one match between the
   //       cint class table and clang decls (otherwise we cannot
   //       trust the value of fIdx).
   //return fDecl && (fIdx > -1) &&
   //   (fIdx < (int) tcling_Dict::Classes()->size());
   return fDecl;
}

bool tcling_ClassInfo::IsValidMethod(const char* method, const char* proto,
                                     long* offset) const
{
   return fClassInfo->GetMethod(method, proto, offset).IsValid();
}

int tcling_ClassInfo::Next() const
{
   return fClassInfo->Next();
}

void* tcling_ClassInfo::New() const
{
   // Note: This is an interpreter function.
   return fClassInfo->New();
}

void* tcling_ClassInfo::New(int n) const
{
   // Note: This is an interpreter function.
   return fClassInfo->New(n);
}

void* tcling_ClassInfo::New(int n, void* arena) const
{
   // Note: This is an interpreter function.
   return fClassInfo->New(n, arena);
}

void* tcling_ClassInfo::New(void* arena) const
{
   // Note: This is an interpreter function.
   return fClassInfo->New(arena);
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
      return property;
   }
   // Note: Now we have class, enum, struct, union only.
   const clang::TagDecl* TD = llvm::dyn_cast<clang::TagDecl>(fDecl);
   if (!TD) {
      return 0L;
   }
   if (TD->isEnum()) {
      property |= G__BIT_ISENUM;
      return property;
   }
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
   return property;
}

int tcling_ClassInfo::RootFlag() const
{
   return fClassInfo->RootFlag();
}

int tcling_ClassInfo::Size() const
{
   if (!IsValid()) {
      return -1;
   }
   if (!IsValidClang()) {
      return fClassInfo->Size();
   }
   const clang::RecordDecl* RD = llvm::dyn_cast<clang::RecordDecl>(fDecl);
   if (!RD) {
      // Must be an enum or a namespace.
      return -1;
   }
   clang::ASTContext& Context = fDecl->getASTContext();
   const clang::ASTRecordLayout& Layout = Context.getASTRecordLayout(RD);
   int64_t size = Layout.getSize().getQuantity();
   return static_cast<int>(size);
}

long tcling_ClassInfo::Tagnum() const
{
   // Note: This *must* return a *cint* tagnum for now.
   return fClassInfo->Tagnum();
}

const char* tcling_ClassInfo::FileName() const
{
   return fClassInfo->FileName();
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
   //buf = tcling_Dict::get_fully_qualified_name(
   //   llvm::dyn_cast<clang::NamedDecl>(fDecl));
   clang::PrintingPolicy P(fDecl->getASTContext().getPrintingPolicy());
   llvm::dyn_cast<clang::NamedDecl>(fDecl)->getNameForDiagnostic(buf, P, true);
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
   //buf = llvm::dyn_cast<clang::NamedDecl>(fDecl)->getNameAsString();
   clang::PrintingPolicy P(fDecl->getASTContext().getPrintingPolicy());
   llvm::dyn_cast<clang::NamedDecl>(fDecl)->getNameForDiagnostic(buf, P, false);
   return buf.c_str();
}

const char* tcling_ClassInfo::Title() const
{
   return fClassInfo->Title();
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
   return buf.c_str();
}

//______________________________________________________________________________
//
//
//

tcling_BaseClassInfo::~tcling_BaseClassInfo()
{
   delete fBaseClassInfo;
   fBaseClassInfo = 0;
   delete fDerivedClassInfo;
   fDerivedClassInfo = 0;
   fDecl = 0;
   fIter = 0;
   delete fClassInfo;
   fClassInfo = 0;
}

tcling_BaseClassInfo::tcling_BaseClassInfo(tcling_ClassInfo* tcling_class_info)
   : fBaseClassInfo(0)
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
      fDerivedClassInfo = new tcling_ClassInfo;
      return;
   }
   fBaseClassInfo = new G__BaseClassInfo(*tcling_class_info->GetClassInfo());
   fDerivedClassInfo = new tcling_ClassInfo(*tcling_class_info);
}

tcling_BaseClassInfo::tcling_BaseClassInfo(const tcling_BaseClassInfo& rhs)
   : fBaseClassInfo(0)
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
      fDerivedClassInfo = new tcling_ClassInfo;
      return;
   }
   fBaseClassInfo = new G__BaseClassInfo(*rhs.fBaseClassInfo);
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
      delete fDerivedClassInfo;
      fDerivedClassInfo = 0;
      fDerivedClassInfo = new tcling_ClassInfo;
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
      fClassInfo = new tcling_ClassInfo(Base);
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

//______________________________________________________________________________
//
//
//

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

tcling_DataMemberInfo::tcling_DataMemberInfo()
   : fDataMemberInfo(new G__DataMemberInfo)
   , fClassInfo(0)
   , fTClingClassInfo(0)
   , fFirstTime(true)
{
   fClassInfo = new G__ClassInfo();
   fTClingClassInfo = new tcling_ClassInfo();
   //fIter = tcling_Dict::GetTranslationUnitDecl()->decls_begin();
   // Move to first global variable.
   //InternalNextValidMember();
}

tcling_DataMemberInfo::tcling_DataMemberInfo(tcling_ClassInfo* tcling_class_info)
   : fDataMemberInfo(0)
   , fClassInfo(0)
   , fTClingClassInfo(0)
   , fFirstTime(true)
{
   if (!tcling_class_info || !tcling_class_info->IsValid()) {
      fDataMemberInfo = new G__DataMemberInfo;
      fClassInfo = new G__ClassInfo();
      fTClingClassInfo = new tcling_ClassInfo();
      //fIter = tcling_Dict::GetTranslationUnitDecl()->decls_begin();
      // Move to first global variable.
      //InternalNextValidMember();
      return;
   }
   fDataMemberInfo = new G__DataMemberInfo(*tcling_class_info->GetClassInfo());
   fClassInfo = new G__ClassInfo(*tcling_class_info->GetClassInfo());
   fTClingClassInfo = new tcling_ClassInfo(*tcling_class_info);
   fIter = llvm::dyn_cast<clang::DeclContext>(tcling_class_info->GetDecl())->
           decls_begin();
   // Move to first data member.
   //InternalNextValidMember();
}

tcling_DataMemberInfo::tcling_DataMemberInfo(const tcling_DataMemberInfo& rhs)
{
   fDataMemberInfo = new G__DataMemberInfo(*rhs.fDataMemberInfo);
   fClassInfo = new G__ClassInfo(*rhs.fClassInfo);
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
   return fDataMemberInfo->ArrayDim();
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
   return fDataMemberInfo->IsValid();
   if (fFirstTime) {
      return false;
   }
   return *fIter;
}

int tcling_DataMemberInfo::MaxIndex(int dim) const
{
   return fDataMemberInfo->MaxIndex(dim);
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
   // Move to next acceptable data member.
   while (1) {
      // Reject members we do not want, and recurse into
      // transparent contexts.
      while (*fIter) {
         // Valid decl, recurse into it, accept it, or reject it.
         clang::Decl::Kind DK = fIter->getKind();
         if (DK == clang::Decl::Enum) {
            // Recurse down into a transparent context.
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
         ++fIter;
      }
      // Accepted member, or at end of decl context.
      if (!*fIter && fIterStack.size()) {
         // End of decl context, and we have more to go.
         fIter = fIterStack.back();
         fIterStack.pop_back();
         ++fIter;
         continue;
      }
      // Accepted member, or at end of outermost decl context.
      break;
   }
}

bool tcling_DataMemberInfo::Next()
{
   return fDataMemberInfo->Next();
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
   return fDataMemberInfo->Offset();
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
   return fDataMemberInfo->Property();
}

long tcling_DataMemberInfo::TypeProperty() const
{
   return fDataMemberInfo->Type()->Property();
}

int tcling_DataMemberInfo::TypeSize() const
{
   return fDataMemberInfo->Type()->Size();
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
   return fDataMemberInfo->Type()->Name();
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
   return fDataMemberInfo->Type()->TrueName();
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
   return fDataMemberInfo->Name();
   static std::string buf;
   if (!IsValid()) {
      return 0;
   }
   buf.clear();
   if (llvm::dyn_cast<clang::NamedDecl>(*fIter)) {
      clang::PrintingPolicy P((*fIter)->getASTContext().getPrintingPolicy());
      llvm::dyn_cast<clang::NamedDecl>(*fIter)->
      getNameForDiagnostic(buf, P, false);
   }
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
   return fDataMemberInfo->Title();
}

const char* tcling_DataMemberInfo::ValidArrayIndex() const
{
   return fDataMemberInfo->ValidArrayIndex();
}

//______________________________________________________________________________
//
//
//

tcling_TypeInfo::~tcling_TypeInfo()
{
   delete fTypeInfo;
   fTypeInfo = 0;
   delete fClassInfo;
   fClassInfo = 0;
   fDecl = 0;
}

tcling_TypeInfo::tcling_TypeInfo()
   : fTypeInfo(0), fClassInfo(0), fDecl(0)
{
   fTypeInfo = new G__TypeInfo;
   fClassInfo = new G__ClassInfo;
}

tcling_TypeInfo::tcling_TypeInfo(const char* name)
   : fTypeInfo(0), fClassInfo(0), fDecl(0)
{
   fTypeInfo = new G__TypeInfo(name);
   int tagnum = fTypeInfo->Tagnum();
   if (tagnum == -1) {
      fClassInfo = new G__ClassInfo;
      return;
   }
   fClassInfo = new G__ClassInfo(tagnum);
#if 0
   fprintf(stderr, "tcling_TypeInfo(name): looking up cling class: %s  "
           "tagnum: %d\n", name, tagnum);
   std::multimap<const std::string, const clang::Decl*>::iterator iter =
      tcling_Dict::ClassNameToDecl()->find(name);
   if (iter == tcling_Dict::ClassNameToDecl()->end()) {
      fprintf(stderr, "tcling_TypeInfo(name): cling class not found: %s  "
              "tagnum: %d\n", name, tagnum);
      return;
   }
   fDecl = (clang::Decl*) iter->second;
   fprintf(stderr, "tcling_TypeInfo(name): cling class found: %s  "
           "tagnum: %d  Decl: 0x%lx\n", name, tagnum, (long) fDecl);
#endif // 0
   //--
}

tcling_TypeInfo::tcling_TypeInfo(G__value* val)
   : fTypeInfo(0), fClassInfo(0), fDecl(0)
{
   fTypeInfo = new G__TypeInfo(*val);
   int tagnum = fTypeInfo->Tagnum();
   if (tagnum == -1) {
      fClassInfo = new G__ClassInfo;
      return;
   }
   fClassInfo = new G__ClassInfo(tagnum);
#if 0
   const char* name = fClassInfo->Fullname();
   fprintf(stderr, "tcling_TypeInfo(val): looking up cling class: %s  "
           "tagnum: %d\n", name, tagnum);
   std::multimap<const std::string, const clang::Decl*>::iterator iter =
      tcling_Dict::ClassNameToDecl()->find(name);
   if (iter == tcling_Dict::ClassNameToDecl()->end()) {
      fprintf(stderr, "tcling_TypeInfo(val): cling class not found: %s  "
              "tagnum: %d\n", name, tagnum);
      return;
   }
   fDecl = (clang::Decl*) iter->second;
   fprintf(stderr, "tcling_TypeInfo(val): cling class found: %s  "
           "tagnum: %d  Decl: 0x%lx\n", name, tagnum, (long) fDecl);
#endif // 0
   //--
}

tcling_TypeInfo::tcling_TypeInfo(const tcling_TypeInfo& rhs)
{
   fTypeInfo = new G__TypeInfo(*rhs.fTypeInfo);
   fClassInfo = new G__ClassInfo(rhs.fClassInfo->Tagnum());
   fDecl = rhs.fDecl;
}

tcling_TypeInfo& tcling_TypeInfo::operator=(const tcling_TypeInfo& rhs)
{
   if (this == &rhs) {
      return *this;
   }
   delete fTypeInfo;
   fTypeInfo = new G__TypeInfo(*rhs.fTypeInfo);
   delete fClassInfo;
   fClassInfo = new G__ClassInfo(rhs.fClassInfo->Tagnum());
   fDecl = rhs.fDecl;
   return *this;
}

G__TypeInfo* tcling_TypeInfo::GetTypeInfo() const
{
   return fTypeInfo;
}

G__ClassInfo* tcling_TypeInfo::GetClassInfo() const
{
   return fClassInfo;
}

clang::Decl* tcling_TypeInfo::GetDecl() const
{
   return fDecl;
}

void tcling_TypeInfo::Init(const char* name)
{
   fTypeInfo->Init(name);
   int tagnum = fTypeInfo->Tagnum();
   if (tagnum == -1) {
      fClassInfo = new G__ClassInfo;
      fDecl = 0;
      return;
   }
   fClassInfo  = new G__ClassInfo(tagnum);
#if 0
   const char* fullname = fClassInfo->Fullname();
   fprintf(stderr, "tcling_TypeInfo::Init(name): looking up cling class: %s  "
           "tagnum: %d\n", fullname, tagnum);
   std::multimap<const std::string, const clang::Decl*>::iterator iter =
      tcling_Dict::ClassNameToDecl()->find(fullname);
   if (iter == tcling_Dict::ClassNameToDecl()->end()) {
      fprintf(stderr, "tcling_TypeInfo::Init(name): cling class not found: %s  "
              "tagnum: %d\n", fullname, tagnum);
      return;
   }
   fDecl = (clang::Decl*) iter->second;
   fprintf(stderr, "tcling_TypeInfo::Init(name): cling class found: %s  "
           "tagnum: %d  Decl: 0x%lx\n", fullname, tagnum, (long) fDecl);
#endif // 0
   //--
}

bool tcling_TypeInfo::IsValid() const
{
   return fTypeInfo->IsValid();
}

const char* tcling_TypeInfo::Name() const
{
   return fTypeInfo->Name();
}

long tcling_TypeInfo::Property() const
{
   return fTypeInfo->Property();
}

int tcling_TypeInfo::RefType() const
{
   return fTypeInfo->Reftype();
}

int tcling_TypeInfo::Size() const
{
   return fTypeInfo->Size();
}

const char* tcling_TypeInfo::TrueName() const
{
   return fTypeInfo->TrueName();
}

//______________________________________________________________________________
//
//
//

tcling_TypedefInfo::~tcling_TypedefInfo()
{
   delete fTypedefInfo;
   fTypedefInfo = 0;
   fDecl = 0;
}

tcling_TypedefInfo::tcling_TypedefInfo()
   : fTypedefInfo(0), fDecl(0), fIdx(-1)
{
   fTypedefInfo = new G__TypedefInfo();
}

tcling_TypedefInfo::tcling_TypedefInfo(const char* name)
   : fTypedefInfo(0), fDecl(0), fIdx(-1)
{
   fTypedefInfo = new G__TypedefInfo(name);
}

tcling_TypedefInfo::tcling_TypedefInfo(const tcling_TypedefInfo& rhs)
{
   fTypedefInfo = new G__TypedefInfo(rhs.fTypedefInfo->Typenum());
#if 0
   fDecl = rhs.fDecl;
   fIdx = rhs.fIdx;
#endif // 0
   //--
}

tcling_TypedefInfo& tcling_TypedefInfo::operator=(const tcling_TypedefInfo& rhs)
{
   if (this == &rhs) {
      return *this;
   }
   delete fTypedefInfo;
   fTypedefInfo = new G__TypedefInfo(rhs.fTypedefInfo->Typenum());
   return *this;
   fDecl = rhs.fDecl;
   fIdx = rhs.fIdx;
   return *this;
}

G__TypedefInfo* tcling_TypedefInfo::GetTypedefInfo() const
{
   return fTypedefInfo;
}

clang::Decl* tcling_TypedefInfo::GetDecl() const
{
   return fDecl;
}

int tcling_TypedefInfo::GetIdx() const
{
   return fIdx;
}

void tcling_TypedefInfo::Init(const char* name)
{
   //fprintf(stderr, "tcling_TypedefInfo::Init(name): looking up typedef: %s\n",
   //        name);
   fDecl = 0;
   fIdx = -1;
   fTypedefInfo->Init(name);
#if 0
   if (!fTypedefInfo->IsValid()) {
      fprintf(stderr, "tcling_TypedefInfo::Init(name): could not find cint "
              "typedef for name: %s\n", name);
   }
   else {
      fprintf(stderr, "tcling_TypedefInfo::Init(name): found cint typedef for "
              "name: %s  tagnum: %d\n", name, fTypedefInfo->Tagnum());
   }
   std::multimap<const std::string, const clang::Decl*>::iterator iter =
      tcling_Dict::TypedefNameToDecl()->find(name);
   if (iter == tcling_Dict::TypedefNameToDecl()->end()) {
      fprintf(stderr, "tcling_TypedefInfo::Init(name): cling typedef not found "
              "name: %s\n", name);
   }
   else {
      fDecl = (clang::Decl*) iter->second;
      fprintf(stderr, "tcling_TypedefInfo::Init(name): found cling typedef "
              "name: %s  decl: 0x%lx\n", name, (long) fDecl);
      std::map<const clang::Decl*, int>::iterator iter_idx =
         tcling_Dict::TypedefDeclToIdx()->find(fDecl);
      if (iter_idx == tcling_Dict::TypedefDeclToIdx()->end()) {
         fprintf(stderr, "tcling_TypedefInfo::Init(name): could not find idx "
                 "for name: %s  decl: 0x%lx\n", name, (long) fDecl);
      }
      else {
         fIdx = iter_idx->second;
         fprintf(stderr, "tcling_TypedefInfo::Init(name): found idx: %d for "
                 "name: %s  decl: 0x%lx\n", fIdx, name, (long) fDecl);
      }
   }
#endif // 0
   //--
}

bool tcling_TypedefInfo::IsValid() const
{
   return IsValidCint();
#if 0
   return IsValidCint() || IsValidClang();
#endif // 0
   //--
}

bool tcling_TypedefInfo::IsValidCint() const
{
   return fTypedefInfo->IsValid();
}

bool tcling_TypedefInfo::IsValidClang() const
{
   // Note: Use this when we can get a one-to-one match between the
   //       cint typedef table and clang decls (otherwise we cannot
   //       trust the value of fIdx).
   //return fDecl && (fIdx > -1) &&
   //   (fIdx < (int) tcling_Dict::Typedefs()->size());
   return fDecl;
}

long tcling_TypedefInfo::Property() const
{
   return fTypedefInfo->Property();
   if (!IsValid()) {
      return 0L;
   }
   if (!IsValidClang()) {
      return fTypedefInfo->Property();
   }
   long property = 0L;
   property |= G__BIT_ISTYPEDEF;
   const clang::TypedefNameDecl* TD =
      llvm::dyn_cast<clang::TypedefNameDecl>(fDecl);
   clang::QualType QT = TD->getUnderlyingType().getCanonicalType();
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
   if (QT->isBuiltinType()) {
      property |= G__BIT_ISFUNDAMENTAL;
   }
   if (QT.isConstQualified()) {
      property |= G__BIT_ISCONSTANT;
   }
   return property;
}

int tcling_TypedefInfo::Size() const
{
   return fTypedefInfo->Size();
   if (!IsValid()) {
      return 1;
   }
   if (!IsValidClang()) {
      return fTypedefInfo->Size();
   }
   clang::ASTContext& Context = fDecl->getASTContext();
   const clang::TypedefNameDecl* TD =
      llvm::dyn_cast<clang::TypedefNameDecl>(fDecl);
   clang::QualType QT = TD->getUnderlyingType();
   // Note: This is an int64_t.
   clang::CharUnits::QuantityType Quantity =
      Context.getTypeSizeInChars(QT).getQuantity();
   return static_cast<int>(Quantity);
}

const char* tcling_TypedefInfo::TrueName() const
{
   return fTypedefInfo->TrueName();
   if (!IsValid()) {
      return "(unknown)";
   }
   if (!IsValidClang()) {
      return fTypedefInfo->TrueName();
   }
   // Note: This must be static because we return a pointer to the internals.
   static std::string truename;
   truename.clear();
   const clang::TypedefNameDecl* TD =
      llvm::dyn_cast<clang::TypedefNameDecl>(fDecl);
   truename = TD->getUnderlyingType().getAsString();
   return truename.c_str();
}

const char* tcling_TypedefInfo::Name() const
{
   return fTypedefInfo->Name();
   if (!IsValid()) {
      return "(unknown)";
   }
   if (!IsValidClang()) {
      return fTypedefInfo->Name();
   }
   // Note: This must be static because we return a pointer to the internals.
   static std::string fullname;
   fullname.clear();
   clang::PrintingPolicy P(fDecl->getASTContext().getPrintingPolicy());
   llvm::dyn_cast<clang::NamedDecl>(fDecl)->
   getNameForDiagnostic(fullname, P, true);
   return fullname.c_str();
}

const char* tcling_TypedefInfo::Title() const
{
   return fTypedefInfo->Title();
   if (!IsValid()) {
      return "";
   }
   if (!IsValidClang()) {
      return fTypedefInfo->Title();
   }
   // FIXME: This needs information from the comments in the header file.
   return fTypedefInfo->Title();
}

int tcling_TypedefInfo::Next()
{
   return fTypedefInfo->Next();
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      return fTypedefInfo->Next();
   }
   return fTypedefInfo->Next();
}

//______________________________________________________________________________
//
//
//

tcling_MethodInfo::~tcling_MethodInfo()
{
   delete fMethodInfo;
   fMethodInfo = 0;
   delete fInitialClassInfo;
   fInitialClassInfo = 0;
   fDecl = 0;
   fFunction = 0;
}

tcling_MethodInfo::tcling_MethodInfo()
   : fMethodInfo(0)
   , fInitialClassInfo(0)
   , fDecl(0)
   , fFunction(0)
{
   fMethodInfo = new G__MethodInfo();
   fInitialClassInfo = new tcling_ClassInfo;
}

tcling_MethodInfo::tcling_MethodInfo(G__MethodInfo* info)
   : fMethodInfo(0)
   , fInitialClassInfo(0)
   , fDecl(0)
   , fFunction(0)
{
   fMethodInfo = new G__MethodInfo(*info);
}

tcling_MethodInfo::tcling_MethodInfo(tcling_ClassInfo* tcling_class_info)
   : fMethodInfo(0)
   , fInitialClassInfo(0)
   , fDecl(0)
   , fFunction(0)
{
   if (!tcling_class_info || !tcling_class_info->IsValid()) {
      fMethodInfo = new G__MethodInfo();
      fInitialClassInfo = new tcling_ClassInfo;
      return;
   }
   fMethodInfo = new G__MethodInfo();
   fMethodInfo->Init(*tcling_class_info->GetClassInfo());
   fInitialClassInfo = new tcling_ClassInfo(*tcling_class_info);
}

tcling_MethodInfo::tcling_MethodInfo(const tcling_MethodInfo& rhs)
   : fMethodInfo(0)
   , fInitialClassInfo(0)
   , fDecl(0)
   , fFunction(0)
{
   if (!rhs.IsValid()) {
      fMethodInfo = new G__MethodInfo();
      fInitialClassInfo = new tcling_ClassInfo;
      return;
   }
   fMethodInfo = new G__MethodInfo(*rhs.fMethodInfo);
   fInitialClassInfo = new tcling_ClassInfo(*rhs.fInitialClassInfo);
   fDecl = rhs.fDecl;
   fIter = rhs.fIter;
   fFunction = rhs.fFunction;
}

tcling_MethodInfo& tcling_MethodInfo::operator=(const tcling_MethodInfo& rhs)
{
   if (this == &rhs) {
      return *this;
   }
   if (!rhs.IsValid()) {
      delete fMethodInfo;
      fMethodInfo = new G__MethodInfo();
      delete fInitialClassInfo;
      fInitialClassInfo = new tcling_ClassInfo;
      fDecl = 0;
      fFunction = 0;
   }
   else {
      delete fMethodInfo;
      fMethodInfo = new G__MethodInfo(*rhs.fMethodInfo);
      delete fInitialClassInfo;
      fInitialClassInfo = new tcling_ClassInfo(*rhs.fInitialClassInfo);
      fDecl = rhs.fDecl;
      fIter = rhs.fIter;
      fFunction = rhs.fFunction;
   }
   return *this;
}

G__MethodInfo* tcling_MethodInfo::GetMethodInfo() const
{
   return fMethodInfo;
}

void tcling_MethodInfo::CreateSignature(TString& signature) const
{
   G__MethodArgInfo arg(*fMethodInfo);
   int ifirst = 0;
   signature = "(";
   while (arg.Next()) {
      if (ifirst) {
         signature += ", ";
      }
      if (arg.Type() == 0) {
         break;
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
      ifirst++;
   }
   signature += ")";
}

G__InterfaceMethod tcling_MethodInfo::InterfaceMethod() const
{
   G__InterfaceMethod p = fMethodInfo->InterfaceMethod();
   if (!p) {
      struct G__bytecodefunc* bytecode = fMethodInfo->GetBytecode();
      if (bytecode) {
         p = (G__InterfaceMethod) G__exec_bytecode;
      }
   }
   return p;
}

bool tcling_MethodInfo::IsValid() const
{
   return fMethodInfo->IsValid();
}

int tcling_MethodInfo::NArg() const
{
   return fMethodInfo->NArg();
}

int tcling_MethodInfo::NDefaultArg() const
{
   return fMethodInfo->NDefaultArg();
}

int tcling_MethodInfo::Next() const
{
   return fMethodInfo->Next();
}

long tcling_MethodInfo::Property() const
{
   return fMethodInfo->Property();
}

void* tcling_MethodInfo::Type() const
{
   return fMethodInfo->Type();
}

const char* tcling_MethodInfo::GetMangledName() const
{
   return fMethodInfo->GetMangledName();
}

const char* tcling_MethodInfo::GetPrototype() const
{
   return fMethodInfo->GetPrototype();
}

const char* tcling_MethodInfo::Name() const
{
   return fMethodInfo->Name();
}

const char* tcling_MethodInfo::TypeName() const
{
   return fMethodInfo->Type()->Name();
}

const char* tcling_MethodInfo::Title() const
{
   return fMethodInfo->Title();
}

//______________________________________________________________________________
//
//
//

tcling_MethodArgInfo::~tcling_MethodArgInfo()
{
   delete fMethodArgInfo;
   fMethodArgInfo = 0;
}

tcling_MethodArgInfo::tcling_MethodArgInfo()
   : fMethodArgInfo(0)
{
   fMethodArgInfo = new G__MethodArgInfo();
}

tcling_MethodArgInfo::tcling_MethodArgInfo(tcling_MethodInfo* tcling_method_info)
   : fMethodArgInfo(0)
{
   if (!tcling_method_info || !tcling_method_info->IsValid()) {
      fMethodArgInfo = new G__MethodArgInfo();
      return;
   }
   fMethodArgInfo = new G__MethodArgInfo(*tcling_method_info->GetMethodInfo());
}

tcling_MethodArgInfo::tcling_MethodArgInfo(const tcling_MethodArgInfo& rhs)
   : fMethodArgInfo(0)
{
   if (!rhs.IsValid()) {
      fMethodArgInfo = new G__MethodArgInfo();
      return;
   }
   fMethodArgInfo = new G__MethodArgInfo(*rhs.fMethodArgInfo);
}

tcling_MethodArgInfo& tcling_MethodArgInfo::operator=(const tcling_MethodArgInfo& rhs)
{
   if (this == &rhs) {
      return *this;
   }
   if (!rhs.IsValid()) {
      delete fMethodArgInfo;
      fMethodArgInfo = new G__MethodArgInfo();
   }
   else {
      delete fMethodArgInfo;
      fMethodArgInfo = new G__MethodArgInfo(*rhs.fMethodArgInfo);
   }
   return *this;
}

bool tcling_MethodArgInfo::IsValid() const
{
   return fMethodArgInfo->IsValid();
}

int tcling_MethodArgInfo::Next() const
{
   return fMethodArgInfo->Next();
}

long tcling_MethodArgInfo::Property() const
{
   return fMethodArgInfo->Property();
}

const char* tcling_MethodArgInfo::DefaultValue() const
{
   return fMethodArgInfo->DefaultValue();
}

const char* tcling_MethodArgInfo::Name() const
{
   return fMethodArgInfo->Name();
}

const char* tcling_MethodArgInfo::TypeName() const
{
   return fMethodArgInfo->Type()->Name();
}

//______________________________________________________________________________
//
//
//

tcling_CallFunc::~tcling_CallFunc()
{
   delete fCallFunc;
   fCallFunc = 0;
}

tcling_CallFunc::tcling_CallFunc()
   : fCallFunc(0)
{
   fCallFunc = new G__CallFunc();
}

tcling_CallFunc::tcling_CallFunc(const tcling_CallFunc& rhs)
   : fCallFunc(0)
{
   if (!rhs.IsValid()) {
      fCallFunc = new G__CallFunc();
      return;
   }
   fCallFunc = new G__CallFunc(*rhs.fCallFunc);
}

tcling_CallFunc& tcling_CallFunc::operator=(const tcling_CallFunc& rhs)
{
   if (this == &rhs) {
      return *this;
   }
   if (!rhs.IsValid()) {
      delete fCallFunc;
      fCallFunc = new G__CallFunc();
   }
   else {
      delete fCallFunc;
      fCallFunc = new G__CallFunc(*rhs.fCallFunc);
   }
   return *this;
}

void tcling_CallFunc::Exec(void* address) const
{
   fCallFunc->Exec(address);
}

long tcling_CallFunc::ExecInt(void* address) const
{
   return fCallFunc->ExecInt(address);
}

long tcling_CallFunc::ExecInt64(void* address) const
{
   return fCallFunc->ExecInt64(address);
}

double tcling_CallFunc::ExecDouble(void* address) const
{
   return fCallFunc->ExecDouble(address);
}

void* tcling_CallFunc::FactoryMethod() const
{
   G__MethodInfo* info = new G__MethodInfo(fCallFunc->GetMethodInfo());
   tcling_MethodInfo* tcling_mi = new tcling_MethodInfo(info);
   delete info;
   info = 0;
   return (void*) tcling_mi; // FIXME
}

void tcling_CallFunc::Init() const
{
   fCallFunc->Init();
}

G__InterfaceMethod tcling_CallFunc::InterfaceMethod() const
{
   return fCallFunc->InterfaceMethod();
}

bool tcling_CallFunc::IsValid() const
{
   return fCallFunc->IsValid();
}

void tcling_CallFunc::ResetArg() const
{
   fCallFunc->ResetArg();
}

void tcling_CallFunc::SetArg(long param) const
{
   fCallFunc->SetArg(param);
}

void tcling_CallFunc::SetArg(double param) const
{
   fCallFunc->SetArg(param);
}

void tcling_CallFunc::SetArg(long long param) const
{
   fCallFunc->SetArg(param);
}

void tcling_CallFunc::SetArg(unsigned long long param) const
{
   fCallFunc->SetArg(param);
}

void tcling_CallFunc::SetArgArray(long* paramArr, int nparam) const
{
   fCallFunc->SetArgArray(paramArr, nparam);
}

void tcling_CallFunc::SetArgs(const char* param) const
{
   fCallFunc->SetArgs(param);
}

void tcling_CallFunc::SetFunc(tcling_ClassInfo* info, const char* method, const char* params, long* offset) const
{
   fCallFunc->SetFunc(info->GetClassInfo(), method, params, offset);
}

void tcling_CallFunc::SetFunc(tcling_MethodInfo* info) const
{
   fCallFunc->SetFunc(*info->GetMethodInfo());
}

void tcling_CallFunc::SetFuncProto(tcling_ClassInfo* info, const char* method, const char* proto, long* offset) const
{
   fCallFunc->SetFuncProto(info->GetClassInfo(), method, proto, offset);
}

//______________________________________________________________________________
//
//
//

extern "C" int ScriptCompiler(const char* filename, const char* opt)
{
   return gSystem->CompileMacro(filename, opt);
}

extern "C" int IgnoreInclude(const char* fname, const char* expandedfname)
{
   return gROOT->IgnoreInclude(fname, expandedfname);
}

extern "C" void TCint_UpdateClassInfo(char* c, Long_t l)
{
   TCintWithCling::UpdateClassInfo(c, l);
}

extern "C" int TCint_AutoLoadCallback(char* c, char* l)
{
   ULong_t varp = G__getgvp();
   G__setgvp((Long_t)G__PVOID);
   string cls(c);
   int result =  TCintWithCling::AutoLoadCallback(cls.c_str(), l);
   G__setgvp(varp);
   return result;
}

extern "C" void* TCint_FindSpecialObject(char* c, G__ClassInfo* ci, void** p1, void** p2)
{
   return TCintWithCling::FindSpecialObject(c, ci, p1, p2);
}

//______________________________________________________________________________
//
//
//

#if 0
//______________________________________________________________________________
static void collect_comment(Preprocessor& PP, ExpectedData& ED)
{
   // Create a raw lexer to pull all the comments out of the main file.
   // We don't want to look in #include'd headers for expected-error strings.
   SourceManager& SM = PP.getSourceManager();
   FileID FID = SM.getMainFileID();
   if (SM.getMainFileID().isInvalid()) {
      return;
   }
   // Create a lexer to lex all the tokens of the main file in raw mode.
   const llvm::MemoryBuffer* FromFile = SM.getBuffer(FID);
   Lexer RawLex(FID, FromFile, SM, PP.getLangOptions());
   // Return comments as tokens, this is how we find expected diagnostics.
   RawLex.SetCommentRetentionState(true);
   Token Tok;
   Tok.setKind(tok::comment);
   while (Tok.isNot(tok::eof)) {
      RawLex.Lex(Tok);
      if (!Tok.is(tok::comment)) {
         continue;
      }
      std::string Comment = PP.getSpelling(Tok);
      if (Comment.empty()) {
         continue;
      }
      // Find all expected errors/warnings/notes.
      ParseDirective(&Comment[0], Comment.size(), ED, PP, Tok.getLocation());
   };
}
#endif // 0

//______________________________________________________________________________
//
//
//

int TCint_GenerateDictionary(const std::vector<std::string> &classes,
                             const std::vector<std::string> &headers,
                             const std::vector<std::string> &fwdDecls,
                             const std::vector<std::string> &unknown)
{
   //This function automatically creates the "LinkDef.h" file for templated
   //classes then executes CompileMacro on it.
   //The name of the file depends on the class name, and it's not generated again
   //if the file exist.
   if (classes.empty()) {
      return 0;
   }
   // Use the name of the first class as the main name.
   const std::string& className = classes[0];
   //(0) prepare file name
   TString fileName = "AutoDict_";
   std::string::const_iterator sIt;
   for (sIt = className.begin(); sIt != className.end(); sIt++) {
      if (*sIt == '<' || *sIt == '>' ||
            *sIt == ' ' || *sIt == '*' ||
            *sIt == ',' || *sIt == '&' ||
            *sIt == ':') {
         fileName += '_';
      }
      else {
         fileName += *sIt;
      }
   }
   if (classes.size() > 1) {
      Int_t chk = 0;
      std::vector<std::string>::const_iterator it = classes.begin();
      while ((++it) != classes.end()) {
         for (UInt_t cursor = 0; cursor != it->length(); ++cursor) {
            chk = chk * 3 + it->at(cursor);
         }
      }
      fileName += TString::Format("_%u", chk);
   }
   fileName += ".cxx";
   if (gSystem->AccessPathName(fileName) != 0) {
      //file does not exist
      //(1) prepare file data
      // If STL, also request iterators' operators.
      // vector is special: we need to check whether
      // vector::iterator is a typedef to pointer or a
      // class.
      static std::set<std::string> sSTLTypes;
      if (sSTLTypes.empty()) {
         sSTLTypes.insert("vector");
         sSTLTypes.insert("list");
         sSTLTypes.insert("deque");
         sSTLTypes.insert("map");
         sSTLTypes.insert("multimap");
         sSTLTypes.insert("set");
         sSTLTypes.insert("multiset");
         sSTLTypes.insert("queue");
         sSTLTypes.insert("priority_queue");
         sSTLTypes.insert("stack");
         sSTLTypes.insert("iterator");
      }
      std::vector<std::string>::const_iterator it;
      std::string fileContent("");
      for (it = headers.begin(); it != headers.end(); ++it) {
         fileContent += "#include \"" + *it + "\"\n";
      }
      for (it = unknown.begin(); it != unknown.end(); ++it) {
         TClass* cl = TClass::GetClass(it->c_str());
         if (cl && cl->GetDeclFileName()) {
            TString header(gSystem->BaseName(cl->GetDeclFileName()));
            TString dir(gSystem->DirName(cl->GetDeclFileName()));
            TString dirbase(gSystem->BaseName(dir));
            while (dirbase.Length() && dirbase != "."
                   && dirbase != "include" && dirbase != "inc"
                   && dirbase != "prec_stl") {
               gSystem->PrependPathName(dirbase, header);
               dir = gSystem->DirName(dir);
            }
            fileContent += TString("#include \"") + header + "\"\n";
         }
      }
      for (it = fwdDecls.begin(); it != fwdDecls.end(); ++it) {
         fileContent += "class " + *it + ";\n";
      }
      fileContent += "#ifdef __CINT__ \n";
      fileContent += "#pragma link C++ nestedclasses;\n";
      fileContent += "#pragma link C++ nestedtypedefs;\n";
      for (it = classes.begin(); it != classes.end(); ++it) {
         std::string n(*it);
         size_t posTemplate = n.find('<');
         std::set<std::string>::const_iterator iSTLType = sSTLTypes.end();
         if (posTemplate != std::string::npos) {
            n.erase(posTemplate, std::string::npos);
            if (n.compare(0, 5, "std::") == 0) {
               n.erase(0, 5);
            }
            iSTLType = sSTLTypes.find(n);
         }
         fileContent += "#pragma link C++ class ";
         fileContent +=    *it + "+;\n" ;
         fileContent += "#pragma link C++ class ";
         if (iSTLType != sSTLTypes.end()) {
            // STL class; we cannot (and don't need to) store iterators;
            // their shadow and the compiler's version don't agree. So
            // don't ask for the '+'
            fileContent +=    *it + "::*;\n" ;
         }
         else {
            // Not an STL class; we need to allow the I/O of contained
            // classes (now that we have a dictionary for them).
            fileContent +=    *it + "::*+;\n" ;
         }
         std::string oprLink("#pragma link C++ operators ");
         oprLink += *it;
         // Don't! Requests e.g. op<(const vector<T>&, const vector<T>&):
         // fileContent += oprLink + ";\n";
         if (iSTLType != sSTLTypes.end()) {
            if (n == "vector") {
               fileContent += "#ifdef G__VECTOR_HAS_CLASS_ITERATOR\n";
            }
            fileContent += oprLink + "::iterator;\n";
            fileContent += oprLink + "::const_iterator;\n";
            fileContent += oprLink + "::reverse_iterator;\n";
            if (n == "vector") {
               fileContent += "#endif\n";
            }
         }
      }
      fileContent += "#endif\n";
      //end(1)
      //(2) prepare the file
      FILE* filePointer;
      filePointer = fopen(fileName, "w");
      if (filePointer == NULL) {
         //can't open a file
         return 1;
      }
      //end(2)
      //write data into the file
      fprintf(filePointer, "%s", fileContent.c_str());
      fclose(filePointer);
   }
   //(3) checking if we can compile a macro, if not then cleaning
   Int_t oldErrorIgnoreLevel = gErrorIgnoreLevel;
   gErrorIgnoreLevel = kWarning; // no "Info: creating library..."
   Int_t ret = gSystem->CompileMacro(fileName, "k");
   gErrorIgnoreLevel = oldErrorIgnoreLevel;
   if (ret == 0) { //can't compile a macro
      return 2;
   }
   //end(3)
   return 0;
}

int TCint_GenerateDictionary(const std::string& className,
                             const std::vector<std::string> &headers,
                             const std::vector<std::string> &fwdDecls,
                             const std::vector<std::string> &unknown)
{
   //This function automatically creates the "LinkDef.h" file for templated
   //classes then executes CompileMacro on it.
   //The name of the file depends on the class name, and it's not generated again
   //if the file exist.
   std::vector<std::string> classes;
   classes.push_back(className);
   return TCint_GenerateDictionary(classes, headers, fwdDecls, unknown);
}

//______________________________________________________________________________
//
//
//

// It is a "fantom" method to synchronize user keyboard input
// and ROOT prompt line (for WIN32)
const char* fantomline = "TRint::EndOfLineAction();";

//______________________________________________________________________________
//
//
//

void* TCintWithCling::fgSetOfSpecials = 0;

//______________________________________________________________________________
//
//
//

ClassImp(TCintWithCling)

//______________________________________________________________________________
TCintWithCling::TCintWithCling(const char *name, const char *title)
   : TInterpreter(name, title)
   , fSharedLibs("")
   , fSharedLibsSerial(-1)
   , fGlobalsListSerial(-1)
   , fInterpreter(0)
   , fMetaProcessor(0)
{
   // Initialize the CINT interpreter interface.
   fMore      = 0;
   fPrompt[0] = 0;
   fMapfile   = 0;
   fRootmapFiles = 0;
   fLockProcessLine = kTRUE;
   // Disable the autoloader until it is explicitly enabled.
   G__set_class_autoloading(0);
   G__RegisterScriptCompiler(&ScriptCompiler);
   G__set_ignoreinclude(&IgnoreInclude);
   G__InitUpdateClassInfo(&TCint_UpdateClassInfo);
   G__InitGetSpecialObject(&TCint_FindSpecialObject);
   // check whether the compiler is available:
   char* path = gSystem->Which(gSystem->Getenv("PATH"), gSystem->BaseName(COMPILER));
   if (path && path[0]) {
      G__InitGenerateDictionary(&TCint_GenerateDictionary);
   }
   delete[] path;
   ResetAll();
#ifndef R__WIN32
   optind = 1;  // make sure getopt() works in the main program
#endif // R__WIN32
   // Make sure that ALL macros are seen as C++.
   G__LockCpp();
   // Initialize for ROOT:
   // Disallow the interpretation of Rtypes.h, TError.h and TGenericClassInfo.h
   ProcessLine("#define ROOT_Rtypes 0");
   ProcessLine("#define ROOT_TError 0");
   ProcessLine("#define ROOT_TGenericClassInfo 0");
   TString include;
   // Add the root include directory to list searched by default
#ifndef ROOTINCDIR
   include = gSystem->Getenv("ROOTSYS");
   include.Append("/include");
#else // ROOTINCDIR
   include = ROOTINCDIR;
#endif // ROOTINCDIR
   TCintWithCling::AddIncludePath(include);
   // Allow the usage of ClassDef and ClassImp in interpreted macros
   // if RtypesCint.h can be found (think of static executable without include/)
   char* whichTypesCint = gSystem->Which(include, "RtypesCint.h");
   if (whichTypesCint) {
      ProcessLine("#include <RtypesCint.h>");
      delete[] whichTypesCint;
   }
   // Initialize the CINT+cling interpreter interface.

   TString interpInclude;
#ifndef ROOTINCDIR
   TString rootsys = gSystem->Getenv("ROOTSYS");
   interpInclude = rootsys + "/etc";
#else // ROOTINCDIR
   interpInclude = ROOTETCDIR;
#endif // ROOTINCDIR
   interpInclude.Prepend("-I");
   const char* interpArgs[] = {interpInclude.Data(), 0};

   TString llvmDir;
   if (gSystem->Getenv("$(LLVMDIR)")) {
      llvmDir = gSystem->ExpandPathName("$(LLVMDIR)");
   }
#ifdef R__LLVMDIR
   if (llvmDir.IsNull()) {
      llvmDir = R__LLVMDIR;
   }
#endif // R__LLVMDIR

   fInterpreter = new cling::Interpreter(1, interpArgs, llvmDir); 
   fInterpreter->installLazyFunctionCreator(autoloadCallback);

   // Add the root include directory and etc/ to list searched by default.
   // Use explicit TCintWithCling::AddIncludePath() to avoid vtable: we're in the c'tor!
#ifndef ROOTINCDIR
   TCintWithCling::AddIncludePath(rootsys + "/include");
   TString dictDir = rootsys + "/lib";
#else // ROOTINCDIR
   TCintWithCling::AddIncludePath(ROOTINCDIR);
   TString dictDir = ROOTLIBDIR;
#endif // ROOTINCDIR

   clang::CompilerInstance * CI = fInterpreter->getCI ();
   CI->getPreprocessor().getHeaderSearchInfo().configureModules(dictDir.Data(), "NONE");
   const char* dictEntry = 0;
   void* dictDirPtr = gSystem->OpenDirectory(dictDir);
   while ((dictEntry = gSystem->GetDirEntry(dictDirPtr))) {
      static const char dictExt[] = "_dict.pcm";
      size_t lenDictEntry = strlen(dictEntry);
      if (lenDictEntry <= 9 || strcmp(dictEntry + lenDictEntry - 9, dictExt)) {
         continue;
      }
      //TString dictFile = dictDir + "/" + dictEntry;
      Info("LoadDictionaries", "Loading PCH %s", dictEntry);
      TString module(dictEntry);
      module.Remove(module.Length() - 4, 4);
      fInterpreter->processLine(std::string("__import_module__ ") + module.Data() + ";",
                                true /*raw*/);
   }
   gSystem->FreeDirectory(dictDirPtr);

   fMetaProcessor = new cling::MetaProcessor(*fInterpreter);

   // to pull in gPluginManager
   fMetaProcessor->process("#include \"TPluginManager.h\"");
}

//______________________________________________________________________________
TCintWithCling::~TCintWithCling()
{
   // Destroy the CINT interpreter interface.
   if (fMore != -1) {
      // only close the opened files do not free memory:
      // G__scratch_all();
      G__close_inputfiles();
   }
   delete fMapfile;
   delete fRootmapFiles;
   gCint = 0;
#ifdef R__COMPLETE_MEM_TERMINATION
   G__scratch_all();
#endif // R__COMPLETE_MEM_TERMINATION
   //--
}

//______________________________________________________________________________
Long_t TCintWithCling::ProcessLine(const char *line, EErrorCode *error)
{
   // Let CINT process a command line.
   // If the command is executed and the result of G__process_cmd is 0,
   // the return value is the int value corresponding to the result of the command
   // (float and double return values will be truncated).

   Long_t ret = 0;
   // Store our line. line is a static buffer in TApplication
   // and we get called recursively through G__process_cmd.
   TString sLine(line);
   if ((sLine[0] == '#') || (sLine[0] == '.')) {
      // Preprocessor or special cmd, have cint process it too.
      // Do not actually run things, load theminstead:
      char haveX = sLine[1];
      if (haveX == 'x' || haveX == 'X')
         sLine[1] = 'L';
      TString sLineNoArgs(sLine);
      Ssiz_t posOpenParen = sLineNoArgs.Last('(');
      if (posOpenParen != kNPOS && sLineNoArgs.EndsWith(")")) {
         sLineNoArgs.Remove(posOpenParen, sLineNoArgs.Length() - posOpenParen);
      }
      ret = TCintWithCling::ProcessLine(sLineNoArgs, error);
      sLine[1] = haveX;
   }
   static const char *fantomline = "TRint::EndOfLineAction();";
   if (sLine == fantomline) {
      // end of line action, CINT-only.
      return TCintWithCling::ProcessLine(sLine, error);
   }
   TString aclicMode;
   TString arguments;
   TString io;
   TString fname;
   if (!strncmp(sLine.Data(), ".L", 2)) { // Load cmd, check for use of ACLiC.
      fname = gSystem->SplitAclicMode(sLine.Data()+3, aclicMode, arguments, io);
   }
   if (aclicMode.Length()) { // ACLiC, pass to cint and not to cling.
      ret = TCintWithCling::ProcessLine(sLine, error);
   }
   else {
      if (fMetaProcessor->process(sLine) > 0) {
         printf("...\n");
      }
   }

   return ret;
}

//______________________________________________________________________________
void TCintWithCling::PrintIntro()
{
   // Print CINT introduction and help message.

   Printf("\nCINT/ROOT C/C++ Interpreter version %s", G__cint_version());
   Printf(">>>>>>>>> cling-ified version <<<<<<<<<<");
   Printf("Type ? for help. Commands must be C++ statements.");
   Printf("Enclose multiple statements between { }.");
}

//______________________________________________________________________________
void TCintWithCling::AddIncludePath(const char *path)
{
   // Add the given path to the list of directories in which the interpreter
   // looks for include files. Only one path item can be specified at a
   // time, i.e. "path1:path2" is not supported.

   fInterpreter->AddIncludePath(path);
   TCintWithCling::AddIncludePath(path);
}

//______________________________________________________________________________
void TCintWithCling::InspectMembers(TMemberInspector&, void* obj, const char* clname)
{
   Printf("Inspecting class %s\n", clname);
}

//______________________________________________________________________________
void TCintWithCling::ClearFileBusy()
{
   // Reset CINT internal state in case a previous action was not correctly
   // terminated by G__init_cint() and G__dlmod().
   R__LOCKGUARD(gCINTMutex);
   G__clearfilebusy(0);
}

//______________________________________________________________________________
void TCintWithCling::ClearStack()
{
   // Delete existing temporary values
   R__LOCKGUARD(gCINTMutex);
   G__clearstack();
}

//______________________________________________________________________________
Int_t TCintWithCling::InitializeDictionaries()
{
   // Initialize all registered dictionaries. Normally this is already done
   // by G__init_cint() and G__dlmod().
   R__LOCKGUARD(gCINTMutex);
   return G__call_setup_funcs();
}

//______________________________________________________________________________
void TCintWithCling::EnableAutoLoading()
{
   // Enable the automatic loading of shared libraries when a class
   // is used that is stored in a not yet loaded library. Uses the
   // information stored in the class/library map (typically
   // $ROOTSYS/etc/system.rootmap).
   R__LOCKGUARD(gCINTMutex);
   G__set_class_autoloading_callback(&TCint_AutoLoadCallback);
   G__set_class_autoloading(1);
   LoadLibraryMap();
}

//______________________________________________________________________________
void TCintWithCling::EndOfLineAction()
{
   // It calls a "fantom" method to synchronize user keyboard input
   // and ROOT prompt line.
   ProcessLineSynch(fantomline);
}

//______________________________________________________________________________
Bool_t TCintWithCling::IsLoaded(const char* filename) const
{
   // Return true if the file has already been loaded by cint.
   // We will try in this order:
   //   actual filename
   //   filename as a path relative to
   //            the include path
   //            the shared library path
   R__LOCKGUARD(gCINTMutex);
   G__SourceFileInfo file(filename);
   if (file.IsValid()) {
      return kTRUE;
   };
   char* next = gSystem->Which(TROOT::GetMacroPath(), filename, kReadPermission);
   if (next) {
      file.Init(next);
      delete[] next;
      if (file.IsValid()) {
         return kTRUE;
      };
   }
   TString incPath = gSystem->GetIncludePath(); // of the form -Idir1  -Idir2 -Idir3
   incPath.Append(":").Prepend(" ");
   incPath.ReplaceAll(" -I", ":");      // of form :dir1 :dir2:dir3
   while (incPath.Index(" :") != -1) {
      incPath.ReplaceAll(" :", ":");
   }
   incPath.Prepend(".:");
# ifdef CINTINCDIR
   TString cintdir = CINTINCDIR;
# else
   TString cintdir = "$(ROOTSYS)/cint";
# endif
   incPath.Append(":");
   incPath.Append(cintdir);
   incPath.Append("/include:");
   incPath.Append(cintdir);
   incPath.Append("/stl");
   next = gSystem->Which(incPath, filename, kReadPermission);
   if (next) {
      file.Init(next);
      delete[] next;
      if (file.IsValid()) {
         return kTRUE;
      };
   }
   next = gSystem->DynamicPathName(filename, kTRUE);
   if (next) {
      file.Init(next);
      delete[] next;
      if (file.IsValid()) {
         return kTRUE;
      };
   }
   return kFALSE;
}

//______________________________________________________________________________
Int_t TCintWithCling::Load(const char* filename, Bool_t system)
{
   // Load a library file in CINT's memory.
   // if 'system' is true, the library is never unloaded.
   R__LOCKGUARD2(gCINTMutex);
   int i;
   if (!system) {
      i = G__loadfile(filename);
   }
   else {
      i = G__loadsystemfile(filename);
   }
   UpdateListOfTypes();
   return i;
}

//______________________________________________________________________________
void TCintWithCling::LoadMacro(const char* filename, EErrorCode* error)
{
   // Load a macro file in CINT's memory.
   ProcessLine(Form(".L %s", filename), error);
   UpdateListOfTypes();
   UpdateListOfGlobals();
   UpdateListOfGlobalFunctions();
}

//______________________________________________________________________________
Long_t TCintWithCling::ProcessLineAsynch(const char* line, EErrorCode* error)
{
   // Let CINT process a command line asynch.
   return ProcessLine(line, error);
}

//______________________________________________________________________________
Long_t TCintWithCling::ProcessLineSynch(const char* line, EErrorCode* error)
{
   // Let CINT process a command line synchronously, i.e we are waiting
   // it will be finished.
   R__LOCKGUARD(fLockProcessLine ? gCINTMutex : 0);
   if (gApplication) {
      if (gApplication->IsCmdThread()) {
         return ProcessLine(line, error);
      }
      return 0;
   }
   return ProcessLine(line, error);
}

//______________________________________________________________________________
Long_t TCintWithCling::Calc(const char* line, EErrorCode* error)
{
   // Directly execute an executable statement (e.g. "func()", "3+5", etc.
   // however not declarations, like "Int_t x;").
   Long_t result;
#ifdef R__WIN32
   // Test on ApplicationImp not being 0 is needed because only at end of
   // TApplication ctor the IsLineProcessing flag is set to 0, so before
   // we can not use it.
   if (gApplication && gApplication->GetApplicationImp()) {
      while (gROOT->IsLineProcessing() && !gApplication) {
         Warning("Calc", "waiting for CINT thread to free");
         gSystem->Sleep(500);
      }
      gROOT->SetLineIsProcessing();
   }
#endif
   R__LOCKGUARD2(gCINTMutex);
   result = (Long_t) G__int_cast(G__calc((char*)line));
   if (error) {
      *error = (EErrorCode)G__lasterror();
   }
#ifdef R__WIN32
   if (gApplication && gApplication->GetApplicationImp()) {
      gROOT->SetLineHasBeenProcessed();
   }
#endif
   return result;
}

//______________________________________________________________________________
void TCintWithCling::SetGetline(const char * (*getlineFunc)(const char* prompt),
                       void (*histaddFunc)(const char* line))
{
   // Set a getline function to call when input is needed.
   G__SetGetlineFunc(getlineFunc, histaddFunc);
}

//______________________________________________________________________________
void TCintWithCling::RecursiveRemove(TObject* obj)
{
   // Delete object from CINT symbol table so it can not be used anymore.
   // CINT objects are always on the heap.
   R__LOCKGUARD(gCINTMutex);
   if (obj->IsOnHeap() && fgSetOfSpecials && !((std::set<TObject*>*)fgSetOfSpecials)->empty()) {
      std::set<TObject*>::iterator iSpecial = ((std::set<TObject*>*)fgSetOfSpecials)->find(obj);
      if (iSpecial != ((std::set<TObject*>*)fgSetOfSpecials)->end()) {
         DeleteGlobal(obj);
         ((std::set<TObject*>*)fgSetOfSpecials)->erase(iSpecial);
      }
   }
}

//______________________________________________________________________________
void TCintWithCling::Reset()
{
   // Reset the CINT state to the state saved by the last call to
   // TCintWithCling::SaveContext().
   R__LOCKGUARD(gCINTMutex);
   G__scratch_upto(&fDictPos);
}

//______________________________________________________________________________
void TCintWithCling::ResetAll()
{
   // Reset the CINT state to its initial state.
   R__LOCKGUARD(gCINTMutex);
   G__init_cint("cint +V");
   G__init_process_cmd();
}

//______________________________________________________________________________
void TCintWithCling::ResetGlobals()
{
   // Reset the CINT global object state to the state saved by the last
   // call to TCintWithCling::SaveGlobalsContext().
   R__LOCKGUARD(gCINTMutex);
   G__scratch_globals_upto(&fDictPosGlobals);
}

//______________________________________________________________________________
void TCintWithCling::ResetGlobalVar(void* obj)
{
   // Reset the CINT global object state to the state saved by the last
   // call to TCintWithCling::SaveGlobalsContext().
   R__LOCKGUARD(gCINTMutex);
   G__resetglobalvar(obj);
}

//______________________________________________________________________________
void TCintWithCling::RewindDictionary()
{
   // Rewind CINT dictionary to the point where it was before executing
   // the current macro. This function is typically called after SEGV or
   // ctlr-C after doing a longjmp back to the prompt.
   R__LOCKGUARD(gCINTMutex);
   G__rewinddictionary();
}

//______________________________________________________________________________
Int_t TCintWithCling::DeleteGlobal(void* obj)
{
   // Delete obj from CINT symbol table so it cannot be accessed anymore.
   // Returns 1 in case of success and 0 in case object was not in table.
   R__LOCKGUARD(gCINTMutex);
   return G__deleteglobal(obj);
}

//______________________________________________________________________________
void TCintWithCling::SaveContext()
{
   // Save the current CINT state.
   R__LOCKGUARD(gCINTMutex);
   G__store_dictposition(&fDictPos);
}

//______________________________________________________________________________
void TCintWithCling::SaveGlobalsContext()
{
   // Save the current CINT state of global objects.
   R__LOCKGUARD(gCINTMutex);
   G__store_dictposition(&fDictPosGlobals);
}

//______________________________________________________________________________
void TCintWithCling::UpdateListOfGlobals()
{
   // Update the list of pointers to global variables. This function
   // is called by TROOT::GetListOfGlobals().
   if (!gROOT->fGlobals) {
      // No globals registered yet, trigger it:
      gROOT->GetListOfGlobals();
      // It already called us again.
      return;
   }
   if (fGlobalsListSerial == G__DataMemberInfo::SerialNumber()) {
      return;
   }
   fGlobalsListSerial = G__DataMemberInfo::SerialNumber();
   R__LOCKGUARD2(gCINTMutex);
   tcling_DataMemberInfo* t = (tcling_DataMemberInfo*) DataMemberInfo_Factory();
   // Loop over all global vars known to cint.
   while (t->Next()) {
      if (t->IsValid() && t->Name()) {
         // Remove any old version in the list.
         {
            TGlobal* g = (TGlobal*) gROOT->fGlobals->FindObject(t->Name());
            if (g) {
               gROOT->fGlobals->Remove(g);
               delete g;
            }
         }
         gROOT->fGlobals->Add(new TGlobal((DataMemberInfo_t*) new tcling_DataMemberInfo(*t)));
      }
   }
}

//______________________________________________________________________________
void TCintWithCling::UpdateListOfGlobalFunctions()
{
   // Update the list of pointers to global functions. This function
   // is called by TROOT::GetListOfGlobalFunctions().
   if (!gROOT->fGlobalFunctions) {
      // No global functions registered yet, trigger it:
      gROOT->GetListOfGlobalFunctions();
      // We were already called by TROOT::GetListOfGlobalFunctions()
      return;
   }
   R__LOCKGUARD2(gCINTMutex);
   tcling_MethodInfo t;
   while (t.Next()) {
      // if name cannot be obtained no use to put in list
      if (t.IsValid() && t.Name()) {
         Bool_t needToAdd = kTRUE;
         // first remove if already in list
         TList* listFuncs = ((THashTable*)(gROOT->fGlobalFunctions))->
            GetListForObject(t.Name());
         if (listFuncs && t.InterfaceMethod()) {
            Long_t prop = -1;
            TIter iFunc(listFuncs);
            Bool_t foundStart = kFALSE;
            TFunction* f = 0;
            while (needToAdd && (f = (TFunction*)iFunc())) {
               if (strcmp(f->GetName(), t.Name())) {
                  if (foundStart) {
                     break;
                  }
                  continue;
               }
               foundStart = kTRUE;
               if (f->InterfaceMethod()) {
                  if (prop == -1) {
                     prop = t.Property();
                  }
                  needToAdd = !(prop & G__BIT_ISCOMPILED) &&
                     (t.GetMangledName() != f->GetMangledName());
               }
            }
         }
         if (needToAdd) {
            gROOT->fGlobalFunctions->Add(new TFunction(new tcling_MethodInfo(t)));
         }
      }
   }
}

//______________________________________________________________________________
void TCintWithCling::UpdateListOfTypes()
{
   // Update the list of pointers to Datatype (typedef) definitions. This
   // function is called by TROOT::GetListOfTypes().
   R__LOCKGUARD2(gCINTMutex);
   //////// Remember the index of the last type that we looked at,
   //////// so that we don't keep reprocessing the same types.
   //////static int last_typenum = -1;
   //////// Also remember the count from the last time the dictionary
   //////// was rewound.  If it's been rewound since the last time we've
   //////// been called, then we recan everything.
   //////static int last_scratch_count = 0;
   //////int this_scratch_count = G__scratch_upto(0);
   //////if (this_scratch_count != last_scratch_count) {
   //////   last_scratch_count = this_scratch_count;
   //////   last_typenum = -1;
   //////}
   //////// Scan from where we left off last time.
   tcling_TypedefInfo* t = (tcling_TypedefInfo*) TypedefInfo_Factory();
   while (t->Next()) {
      const char* name = t->Name();
      if (gROOT && gROOT->fTypes && t->IsValid() && name) {
         TDataType* d = (TDataType*) gROOT->fTypes->FindObject(name);
         // only add new types, don't delete old ones with the same name
         // (as is done in UpdateListOfGlobals()),
         // this 'feature' is being used in TROOT::GetType().
         if (!d) {
            gROOT->fTypes->Add(new TDataType(new tcling_TypedefInfo(*t)));
         }
         //////last_typenum = t->Typenum();
      }
   }
}

//______________________________________________________________________________
void TCintWithCling::SetClassInfo(TClass* cl, Bool_t reload)
{
   // Set pointer to the tcling_ClassInfo in TClass.
   R__LOCKGUARD2(gCINTMutex);
   if (cl->fClassInfo && !reload) {
      return;
   }
   delete (tcling_ClassInfo*) cl->fClassInfo;
   cl->fClassInfo = 0;
   cl->fDecl = 0;
   std::string name(cl->GetName());
   tcling_ClassInfo* info = new tcling_ClassInfo(name.c_str());
   if (!info->IsValid()) {
      bool cint_class_exists = CheckClassInfo(name.c_str());
      if (!cint_class_exists) {
         // Try resolving all the typedefs (even Float_t and Long64_t).
         name = TClassEdit::ResolveTypedef(name.c_str(), kTRUE);
         if (name == cl->GetName()) {
            // No typedefs found, all done.
            return;
         }
         // Try the new name.
         cint_class_exists = CheckClassInfo(name.c_str());
         if (!cint_class_exists) {
            // Nothing found, nothing to do.
            return;
         }
      }
      info = new tcling_ClassInfo(name.c_str());
      if (!info->IsValid()) {
         // Failed, done.
         return;
      }
   }
   cl->fClassInfo = info; // Note: We are transfering ownership here.
   cl->fDecl = const_cast<clang::Decl*>(info->GetDecl());
   // In case a class contains an external enum, the enum will be seen as a
   // class. We must detect this special case and make the class a Zombie.
   // Here we assume that a class has at least one method.
   // We can NOT call TClass::Property from here, because this method
   // assumes that the TClass is well formed to do a lot of information
   // caching. The method SetClassInfo (i.e. here) is usually called during
   // the building phase of the TClass, hence it is NOT well formed yet.
   Bool_t zombieCandidate = kFALSE;
   if (
      info->IsValid() &&
      !(info->Property() & (kIsClass | kIsStruct | kIsNamespace))
   ) {
      zombieCandidate = kTRUE;
   }
   if (!info->IsLoaded()) {
      if (info->Property() & (kIsNamespace)) {
         // Namespaces can have info but no corresponding CINT dictionary
         // because they are auto-created if one of their contained
         // classes has a dictionary.
         zombieCandidate = kTRUE;
      }
      // this happens when no CINT dictionary is available
      delete info;
      cl->fClassInfo = 0;
      cl->fDecl = 0;
   }
   if (zombieCandidate && !TClassEdit::IsSTLCont(cl->GetName())) {
      cl->MakeZombie();
   }
}

//______________________________________________________________________________
Bool_t TCintWithCling::CheckClassInfo(const char* name, Bool_t autoload /*= kTRUE*/)
{
   // Checks if a class with the specified name is defined in CINT.
   // Returns kFALSE is class is not defined.
   // In the case where the class is not loaded and belongs to a namespace
   // or is nested, looking for the full class name is outputing a lots of
   // (expected) error messages.  Currently the only way to avoid this is to
   // specifically check that each level of nesting is already loaded.
   // In case of templates the idea is that everything between the outer
   // '<' and '>' has to be skipped, e.g.: aap<pipo<noot>::klaas>::a_class
   R__LOCKGUARD(gCINTMutex);
   Int_t nch = strlen(name) * 2;
   char* classname = new char[nch];
   strlcpy(classname, name, nch);
   char* current = classname;
   while (*current) {
      while (*current && *current != ':' && *current != '<') {
         current++;
      }
      if (!*current) {
         break;
      }
      if (*current == '<') {
         int level = 1;
         current++;
         while (*current && level > 0) {
            if (*current == '<') {
               level++;
            }
            if (*current == '>') {
               level--;
            }
            current++;
         }
         continue;
      }
      // *current == ':', must be a "::"
      if (*(current + 1) != ':') {
         Error("CheckClassInfo", "unexpected token : in %s", classname);
         delete[] classname;
         return kFALSE;
      }
      *current = '\0';
      tcling_ClassInfo info(classname);
      if (!info.IsValid()) {
         delete[] classname;
         return kFALSE;
      }
      *current = ':';
      current += 2;
   }
   strlcpy(classname, name, nch);
   int flag = 2;
   if (!autoload) {
      flag = 3;
   }
   Int_t tagnum = G__defined_tagname(classname, flag); // This function might modify the name (to add space between >>).
   if (tagnum >= 0) {
      G__ClassInfo info(tagnum);
      // If autoloading is off then Property() == 0 for autoload entries.
      if (!autoload && !info.Property()) {
         return kTRUE;
      }
      if (info.Property() & (G__BIT_ISENUM | G__BIT_ISCLASS | G__BIT_ISSTRUCT | G__BIT_ISUNION | G__BIT_ISNAMESPACE)) {
         // We are now sure that the entry is not in fact an autoload entry.
         delete[] classname;
         return kTRUE;
      }
   }
   tcling_TypedefInfo t(name);
   if (t.IsValid() && !(t.Property() & G__BIT_ISFUNDAMENTAL)) {
      delete[] classname;
      return kTRUE;
   }
   delete[] classname;
   return kFALSE;
}

//______________________________________________________________________________
void TCintWithCling::CreateListOfBaseClasses(TClass* cl)
{
   // Create list of pointers to base class(es) for TClass cl.
   R__LOCKGUARD2(gCINTMutex);
   if (cl->fBase) {
      return;
   }
   cl->fBase = new TList;
   tcling_ClassInfo tci(cl->GetName());
   tcling_BaseClassInfo t(&tci);
   while (t.Next()) {
      // if name cannot be obtained no use to put in list
      if (t.IsValid() && t.Name()) {
         tcling_BaseClassInfo* a = new tcling_BaseClassInfo(t);
         cl->fBase->Add(new TBaseClass(a, cl));
      }
   }
}

//______________________________________________________________________________
void TCintWithCling::CreateListOfDataMembers(TClass* cl)
{
   // Create list of pointers to data members for TClass cl.
   R__LOCKGUARD2(gCINTMutex);
   if (cl->fData) {
      return;
   }
   cl->fData = new TList;
   tcling_DataMemberInfo t((tcling_ClassInfo*)cl->GetClassInfo());
   while (t.Next()) {
      // if name cannot be obtained no use to put in list
      if (t.IsValid() && t.Name() && strcmp(t.Name(), "G__virtualinfo")) {
         tcling_DataMemberInfo* a = new tcling_DataMemberInfo(t);
         cl->fData->Add(new TDataMember(a, cl));
      }
   }
}

//______________________________________________________________________________
void TCintWithCling::CreateListOfMethods(TClass* cl)
{
   // Create list of pointers to methods for TClass cl.
   R__LOCKGUARD2(gCINTMutex);
   if (cl->fMethod) {
      return;
   }
   cl->fMethod = new THashList;
   tcling_MethodInfo t((tcling_ClassInfo*)cl->GetClassInfo());
   while (t.Next()) {
      // if name cannot be obtained no use to put in list
      if (t.IsValid() && t.Name()) {
         tcling_MethodInfo* a = new tcling_MethodInfo(t);
         cl->fMethod->Add(new TMethod(a, cl));
      }
   }
}

//______________________________________________________________________________
void TCintWithCling::UpdateListOfMethods(TClass* cl)
{
   // Update the list of pointers to method for TClass cl, if necessary
   if (cl->fMethod) {
      R__LOCKGUARD2(gCINTMutex);
      tcling_ClassInfo* info = (tcling_ClassInfo*) cl->GetClassInfo();
      if (!info || cl->fMethod->GetEntries() == info->NMethods()) {
         return;
      }
      delete cl->fMethod;
      cl->fMethod = 0;
   }
   CreateListOfMethods(cl);
}

//______________________________________________________________________________
void TCintWithCling::CreateListOfMethodArgs(TFunction* m)
{
   // Create list of pointers to method arguments for TMethod m.
   R__LOCKGUARD2(gCINTMutex);
   if (m->fMethodArgs) {
      return;
   }
   m->fMethodArgs = new TList;
   // FIXME, direct use of data member.
   tcling_MethodArgInfo t((tcling_MethodInfo*)m->fInfo);
   while (t.Next()) {
      if (t.IsValid()) {
         tcling_MethodArgInfo* a = new tcling_MethodArgInfo(t);
         m->fMethodArgs->Add(new TMethodArg(a, m));
      }
   }
}

//______________________________________________________________________________
Int_t TCintWithCling::GenerateDictionary(const char* classes, const char* includes /* = 0 */, const char* /* options  = 0 */)
{
   // Generate the dictionary for the C++ classes listed in the first
   // argmument (in a semi-colon separated list).
   // 'includes' contains a semi-colon separated list of file to
   // #include in the dictionary.
   // For example:
   //    gInterpreter->GenerateDictionary("vector<vector<float> >;list<vector<float> >","list;vector");
   // or
   //    gInterpreter->GenerateDictionary("myclass","myclass.h;myhelper.h");
   if (classes == 0 || classes[0] == 0) {
      return 0;
   }
   // Split the input list
   std::vector<std::string> listClasses;
   for (
      const char* current = classes, *prev = classes;
      *current != 0;
      ++current
   ) {
      if (*current == ';') {
         listClasses.push_back(std::string(prev, current - prev));
         prev = current + 1;
      }
      else if (*(current + 1) == 0) {
         listClasses.push_back(std::string(prev, current + 1 - prev));
         prev = current + 1;
      }
   }
   std::vector<std::string> listIncludes;
   for (
      const char* current = includes, *prev = includes;
      *current != 0;
      ++current
   ) {
      if (*current == ';') {
         listIncludes.push_back(std::string(prev, current - prev));
         prev = current + 1;
      }
      else if (*(current + 1) == 0) {
         listIncludes.push_back(std::string(prev, current + 1 - prev));
         prev = current + 1;
      }
   }
   // Generate the temporary dictionary file
   return TCint_GenerateDictionary(listClasses, listIncludes,
      std::vector<std::string>(), std::vector<std::string>());
}


//______________________________________________________________________________
TString TCintWithCling::GetMangledName(TClass* cl, const char* method,
                              const char* params)
{
   // Return the CINT mangled name for a method of a class with parameters
   // params (params is a string of actual arguments, not formal ones). If the
   // class is 0 the global function list will be searched.
   R__LOCKGUARD2(gCINTMutex);
   tcling_CallFunc func;
   if (cl) {
      Long_t offset;
      func.SetFunc((tcling_ClassInfo*)cl->GetClassInfo(), method, params,
         &offset);
   }
   else {
      tcling_ClassInfo gcl;
      Long_t offset;
      func.SetFunc(&gcl, method, params, &offset);
   }
   tcling_MethodInfo* mi = (tcling_MethodInfo*) func.FactoryMethod();
   const char* mangled_name = mi->GetMangledName();
   delete mi;
   mi = 0;
   return mangled_name;
}

//______________________________________________________________________________
TString TCintWithCling::GetMangledNameWithPrototype(TClass* cl, const char* method,
      const char* proto)
{
   // Return the CINT mangled name for a method of a class with a certain
   // prototype, i.e. "char*,int,float". If the class is 0 the global function
   // list will be searched.
   R__LOCKGUARD2(gCINTMutex);
   Long_t offset;
   if (cl) {
      return ((tcling_ClassInfo*)cl->GetClassInfo())->
             GetMethod(method, proto, &offset)->GetMangledName();
   }
   tcling_ClassInfo gcl;
   return gcl.GetMethod(method, proto, &offset)->GetMangledName();
}

//______________________________________________________________________________
void* TCintWithCling::GetInterfaceMethod(TClass* cl, const char* method,
                                const char* params)
{
   // Return pointer to CINT interface function for a method of a class with
   // parameters params (params is a string of actual arguments, not formal
   // ones). If the class is 0 the global function list will be searched.
   R__LOCKGUARD2(gCINTMutex);
   tcling_CallFunc func;
   if (cl) {
      Long_t offset;
      func.SetFunc((tcling_ClassInfo*)cl->GetClassInfo(), method, params,
         &offset);
   }
   else {
      tcling_ClassInfo gcl;
      Long_t offset;
      func.SetFunc(&gcl, method, params, &offset);
   }
   return (void*) func.InterfaceMethod();
}

//______________________________________________________________________________
void* TCintWithCling::GetInterfaceMethodWithPrototype(TClass* cl, const char* method,
      const char* proto)
{
   // Return pointer to CINT interface function for a method of a class with
   // a certain prototype, i.e. "char*,int,float". If the class is 0 the global
   // function list will be searched.
   R__LOCKGUARD2(gCINTMutex);
   G__InterfaceMethod f;
   if (cl) {
      Long_t offset;
      f = ((tcling_ClassInfo*)cl->GetClassInfo())->
          GetMethod(method, proto, &offset)->InterfaceMethod();
   }
   else {
      Long_t offset;
      tcling_ClassInfo gcl;
      f = gcl.GetMethod(method, proto, &offset)->InterfaceMethod();
   }
   return (void*) f;
}

//______________________________________________________________________________
const char* TCintWithCling::GetInterpreterTypeName(const char* name, Bool_t full)
{
   // The 'name' is known to the interpreter, this function returns
   // the internal version of this name (usually just resolving typedefs)
   // This is used in particular to synchronize between the name used
   // by rootcint and by the run-time enviroment (TClass)
   // Return 0 if the name is not known.
   R__LOCKGUARD(gCINTMutex);
   if (!gInterpreter->CheckClassInfo(name)) {
      return 0;
   }
   tcling_ClassInfo cl(name);
   if (!cl.IsValid()) {
      return 0;
   }
   if (full) {
      return cl.FullName();
   }
   return cl.Name();
}

//______________________________________________________________________________
void TCintWithCling::Execute(const char* function, const char* params, int* error)
{
   // Execute a global function with arguments params.
   R__LOCKGUARD2(gCINTMutex);
   tcling_ClassInfo cl;
   Long_t offset;
   tcling_CallFunc func;
   func.SetFunc(&cl, function, params, &offset);
   func.Exec(0);
   if (error) {
      *error = G__lasterror();
   }
}

//______________________________________________________________________________
void TCintWithCling::Execute(TObject* obj, TClass* cl, const char* method,
                    const char* params, int* error)
{
   // Execute a method from class cl with arguments params.
   R__LOCKGUARD2(gCINTMutex);
   // If the actual class of this object inherits 2nd (or more) from TObject,
   // 'obj' is unlikely to be the start of the object (as described by IsA()),
   // hence gInterpreter->Execute will improperly correct the offset.
   void* addr = cl->DynamicCast(TObject::Class(), obj, kFALSE);
   Long_t offset = 0L;
   tcling_CallFunc func;
   func.SetFunc((tcling_ClassInfo*)cl->GetClassInfo(), method, params, &offset);
   void* address = (void*)((Long_t)addr + offset);
   func.Exec(address);
   if (error) {
      *error = G__lasterror();
   }
}

//______________________________________________________________________________
void TCintWithCling::Execute(TObject* obj, TClass* cl, TMethod* method,
      TObjArray* params, int* error)
{
   // Execute a method from class cl with the arguments in array params
   // (params[0] ... params[n] = array of TObjString parameters).
   // Convert the TObjArray array of TObjString parameters to a character
   // string of comma separated parameters.
   // The parameters of type 'char' are enclosed in double quotes and all
   // internal quotes are escaped.
   if (!method) {
      Error("Execute", "No method was defined");
      return;
   }
   TList* argList = method->GetListOfMethodArgs();
   // Check number of actual parameters against of expected formal ones
   Int_t nparms = argList->LastIndex() + 1;
   Int_t argc   = params ? params->LastIndex() + 1 : 0;
   if (nparms != argc) {
      Error("Execute", "Wrong number of the parameters");
      return;
   }
   const char* listpar = "";
   TString complete(10);
   if (params) {
      // Create a character string of parameters from TObjArray
      TIter next(params);
      for (Int_t i = 0; i < argc; i ++) {
         TMethodArg* arg = (TMethodArg*) argList->At(i);
         tcling_TypeInfo type(arg->GetFullTypeName());
         TObjString* nxtpar = (TObjString*) next();
         if (i) {
            complete += ',';
         }
         if (strstr(type.TrueName(), "char")) {
            TString chpar('\"');
            chpar += (nxtpar->String()).ReplaceAll("\"", "\\\"");
            // At this point we have to check if string contains \\"
            // and apply some more sophisticated parser. Not implemented yet!
            complete += chpar;
            complete += '\"';
         }
         else {
            complete += nxtpar->String();
         }
      }
      listpar = complete.Data();
   }
   Execute(obj, cl, (char*)method->GetName(), (char*)listpar, error);
}

//______________________________________________________________________________
Long_t TCintWithCling::ExecuteMacro(const char* filename, EErrorCode* error)
{
   // Execute a CINT macro.
   R__LOCKGUARD(gCINTMutex);
   return TApplication::ExecuteFile(filename, (int*)error);
}

//______________________________________________________________________________
const char* TCintWithCling::GetTopLevelMacroName() const
{
   // Return the file name of the current un-included interpreted file.
   // See the documentation for GetCurrentMacroName().
   G__SourceFileInfo srcfile(G__get_ifile()->filenum);
   while (srcfile.IncludedFrom().IsValid()) {
      srcfile = srcfile.IncludedFrom();
   }
   return srcfile.Name();
}

//______________________________________________________________________________
const char* TCintWithCling::GetCurrentMacroName() const
{
   // Return the file name of the currently interpreted file,
   // included or not. Example to illustrate the difference between
   // GetCurrentMacroName() and GetTopLevelMacroName():
   // BEGIN_HTML <!--
   /* -->
      <span style="color:#ffffff;background-color:#7777ff;padding-left:0.3em;padding-right:0.3em">inclfile.h</span>
      <!--div style="border:solid 1px #ffff77;background-color: #ffffdd;float:left;padding:0.5em;margin-bottom:0.7em;"-->
      <div class="code">
      <pre style="margin:0pt">#include &lt;iostream&gt;
   void inclfunc() {
   std::cout &lt;&lt; "In inclfile.h" &lt;&lt; std::endl;
   std::cout &lt;&lt; "  TCintWithCling::GetCurrentMacroName() returns  " &lt;&lt;
      TCintWithCling::GetCurrentMacroName() &lt;&lt; std::endl;
   std::cout &lt;&lt; "  TCintWithCling::GetTopLevelMacroName() returns " &lt;&lt;
      TCintWithCling::GetTopLevelMacroName() &lt;&lt; std::endl;
   }</pre></div>
      <div style="clear:both"></div>
      <span style="color:#ffffff;background-color:#7777ff;padding-left:0.3em;padding-right:0.3em">mymacro.C</span>
      <div style="border:solid 1px #ffff77;background-color: #ffffdd;float:left;padding:0.5em;margin-bottom:0.7em;">
      <pre style="margin:0pt">#include &lt;iostream&gt;
   #include "inclfile.h"
   void mymacro() {
   std::cout &lt;&lt; "In mymacro.C" &lt;&lt; std::endl;
   std::cout &lt;&lt; "  TCintWithCling::GetCurrentMacroName() returns  " &lt;&lt;
      TCintWithCling::GetCurrentMacroName() &lt;&lt; std::endl;
   std::cout &lt;&lt; "  TCintWithCling::GetTopLevelMacroName() returns " &lt;&lt;
      TCintWithCling::GetTopLevelMacroName() &lt;&lt; std::endl;
   std::cout &lt;&lt; "  Now calling inclfunc..." &lt;&lt; std::endl;
   inclfunc();
   }</pre></div>
   <div style="clear:both"></div>
   <!-- */
   // --> END_HTML
   // Running mymacro.C will print:
   //
   // root [0] .x mymacro.C
   // In mymacro.C
   //   TCintWithCling::GetCurrentMacroName() returns  ./mymacro.C
   //   TCintWithCling::GetTopLevelMacroName() returns ./mymacro.C
   //   Now calling inclfunc...
   // In inclfile.h
   //   TCintWithCling::GetCurrentMacroName() returns  inclfile.h
   //   TCintWithCling::GetTopLevelMacroName() returns ./mymacro.C
   return G__get_ifile()->name;
}

//______________________________________________________________________________
const char* TCintWithCling::TypeName(const char* typeDesc)
{
   // Return the absolute type of typeDesc.
   // E.g.: typeDesc = "class TNamed**", returns "TNamed".
   // You need to use the result immediately before it is being overwritten.
   static char* t = 0;
   static unsigned int tlen = 0;
   R__LOCKGUARD(gCINTMutex); // Because of the static array.
   unsigned int dlen = strlen(typeDesc);
   if (dlen > tlen) {
      delete[] t;
      t = new char[dlen + 1];
      tlen = dlen;
   }
   char* s, *template_start;
   if (!strstr(typeDesc, "(*)(")) {
      s = (char*)strchr(typeDesc, ' ');
      template_start = (char*)strchr(typeDesc, '<');
      if (!strcmp(typeDesc, "long long")) {
         strlcpy(t, typeDesc, dlen + 1);
      }
      else if (!strncmp(typeDesc, "unsigned ", s + 1 - typeDesc)) {
         strlcpy(t, typeDesc, dlen + 1);
      }
      // s is the position of the second 'word' (if any)
      // except in the case of templates where there will be a space
      // just before any closing '>': eg.
      //    TObj<std::vector<UShort_t,__malloc_alloc_template<0> > >*
      else if (s && (template_start == 0 || (s < template_start))) {
         strlcpy(t, s + 1, dlen + 1);
      }
      else {
         strlcpy(t, typeDesc, dlen + 1);
      }
   }
   else {
      strlcpy(t, typeDesc, dlen + 1);
   }
   int l = strlen(t);
   while (l > 0 && (t[l - 1] == '*' || t[l - 1] == '&')) {
      t[--l] = 0;
   }
   return t;
}

//______________________________________________________________________________
Int_t TCintWithCling::LoadLibraryMap(const char* rootmapfile)
{
   // Load map between class and library. If rootmapfile is specified a
   // specific rootmap file can be added (typically used by ACLiC).
   // In case of error -1 is returned, 0 otherwise.
   // Cint uses this information to automatically load the shared library
   // for a class (autoload mechanism).
   // See also the AutoLoadCallback() method below.
   R__LOCKGUARD(gCINTMutex);
   // open the [system].rootmap files
   if (!fMapfile) {
      fMapfile = new TEnv(".rootmap");
      fMapfile->IgnoreDuplicates(kTRUE);
      fRootmapFiles = new TObjArray;
      fRootmapFiles->SetOwner();
      // Make sure that this information will be useable by inserting our
      // autoload call back!
      G__set_class_autoloading_callback(&TCint_AutoLoadCallback);
   }
   // Load all rootmap files in the dynamic load path ((DY)LD_LIBRARY_PATH, etc.).
   // A rootmap file must end with the string ".rootmap".
   TString ldpath = gSystem->GetDynamicPath();
   if (ldpath != fRootmapLoadPath) {
      fRootmapLoadPath = ldpath;
#ifdef WIN32
      TObjArray* paths = ldpath.Tokenize(";");
#else
      TObjArray* paths = ldpath.Tokenize(":");
#endif
      TString d;
      for (Int_t i = 0; i < paths->GetEntriesFast(); i++) {
         d = ((TObjString*)paths->At(i))->GetString();
         // check if directory already scanned
         Int_t skip = 0;
         for (Int_t j = 0; j < i; j++) {
            TString pd = ((TObjString*)paths->At(j))->GetString();
            if (pd == d) {
               skip++;
               break;
            }
         }
         if (!skip) {
            void* dirp = gSystem->OpenDirectory(d);
            if (dirp) {
               if (gDebug > 3) {
                  Info("LoadLibraryMap", "%s", d.Data());
               }
               const char* f1;
               while ((f1 = gSystem->GetDirEntry(dirp))) {
                  TString f = f1;
                  if (f.EndsWith(".rootmap")) {
                     TString p;
                     p = d + "/" + f;
                     if (!gSystem->AccessPathName(p, kReadPermission)) {
                        if (!fRootmapFiles->FindObject(f) && f != ".rootmap") {
                           if (gDebug > 4) {
                              Info("LoadLibraryMap", "   rootmap file: %s", p.Data());
                           }
                           fMapfile->ReadFile(p, kEnvGlobal);
                           fRootmapFiles->Add(new TNamed(f, p));
                        }
                        //                        else {
                        //                           fprintf(stderr,"Reject %s because %s is already there\n",p.Data(),f.Data());
                        //                           fRootmapFiles->FindObject(f)->ls();
                        //                        }
                     }
                  }
                  if (f.BeginsWith("rootmap")) {
                     TString p;
                     p = d + "/" + f;
                     FileStat_t stat;
                     if (gSystem->GetPathInfo(p, stat) == 0 && R_ISREG(stat.fMode)) {
                        Warning("LoadLibraryMap", "please rename %s to end with \".rootmap\"", p.Data());
                     }
                  }
               }
            }
            gSystem->FreeDirectory(dirp);
         }
      }
      delete paths;
      if (!fMapfile->GetTable()->GetEntries()) {
         return -1;
      }
   }
   if (rootmapfile && *rootmapfile) {
      // Add content of a specific rootmap file
      Bool_t ignore = fMapfile->IgnoreDuplicates(kFALSE);
      fMapfile->ReadFile(rootmapfile, kEnvGlobal);
      fRootmapFiles->Add(new TNamed(gSystem->BaseName(rootmapfile), rootmapfile));
      fMapfile->IgnoreDuplicates(ignore);
   }
   TEnvRec* rec;
   TIter next(fMapfile->GetTable());
   while ((rec = (TEnvRec*) next())) {
      TString cls = rec->GetName();
      if (!strncmp(cls.Data(), "Library.", 8) && cls.Length() > 8) {
         // get the first lib from the list of lib and dependent libs
         TString libs = rec->GetValue();
         if (libs == "") {
            continue;
         }
         TString delim(" ");
         TObjArray* tokens = libs.Tokenize(delim);
         const char* lib = ((TObjString*)tokens->At(0))->GetName();
         // convert "@@" to "::", we used "@@" because TEnv
         // considers "::" a terminator
         cls.Remove(0, 8);
         cls.ReplaceAll("@@", "::");
         // convert "-" to " ", since class names may have
         // blanks and TEnv considers a blank a terminator
         cls.ReplaceAll("-", " ");
         if (cls.Contains(":")) {
            // We have a namespace and we have to check it first
            int slen = cls.Length();
            for (int k = 0; k < slen; k++) {
               if (cls[k] == ':') {
                  if (k + 1 >= slen || cls[k + 1] != ':') {
                     // we expected another ':'
                     break;
                  }
                  if (k) {
                     TString base = cls(0, k);
                     if (base == "std") {
                        // std is not declared but is also ignored by CINT!
                        break;
                     }
                     else {
                        // Only declared the namespace do not specify any library because
                        // the namespace might be spread over several libraries and we do not
                        // know (yet?) which one the user will need!
                        // But what if it's not a namespace but a class?
                        // Does CINT already know it?
                        const char* baselib = G__get_class_autoloading_table((char*)base.Data());
                        if ((!baselib || !baselib[0]) && !rec->FindObject(base)) {
                           G__set_class_autoloading_table((char*)base.Data(), (char*)"");
                        }
                     }
                     ++k;
                  }
               }
               else if (cls[k] == '<') {
                  // We do not want to look at the namespace inside the template parameters!
                  break;
               }
            }
         }
         G__set_class_autoloading_table((char*)cls.Data(), (char*)lib);
         G__security_recover(stderr); // Ignore any error during this setting.
         if (gDebug > 6) {
            const char* wlib = gSystem->DynamicPathName(lib, kTRUE);
            if (wlib) {
               Info("LoadLibraryMap", "class %s in %s", cls.Data(), wlib);
            }
            else {
               Info("LoadLibraryMap", "class %s in %s (library does not exist)", cls.Data(), lib);
            }
            delete[] wlib;
         }
         delete tokens;
      }
   }
   return 0;
}

//______________________________________________________________________________
Int_t TCintWithCling::RescanLibraryMap()
{
   // Scan again along the dynamic path for library maps. Entries for the loaded
   // shared libraries are unloaded first. This can be useful after reseting
   // the dynamic path through TSystem::SetDynamicPath()
   // In case of error -1 is returned, 0 otherwise.
   UnloadAllSharedLibraryMaps();
   LoadLibraryMap();
   return 0;
}

//______________________________________________________________________________
Int_t TCintWithCling::ReloadAllSharedLibraryMaps()
{
   // Reload the library map entries coming from all the loaded shared libraries,
   // after first unloading the current ones.
   // In case of error -1 is returned, 0 otherwise.
   const TString sharedLibLStr = GetSharedLibs();
   const TObjArray* sharedLibL = sharedLibLStr.Tokenize(" ");
   const Int_t nrSharedLibs = sharedLibL->GetEntriesFast();
   for (Int_t ilib = 0; ilib < nrSharedLibs; ilib++) {
      const TString sharedLibStr = ((TObjString*)sharedLibL->At(ilib))->GetString();
      const  TString sharedLibBaseStr = gSystem->BaseName(sharedLibStr);
      const Int_t ret = UnloadLibraryMap(sharedLibBaseStr);
      if (ret < 0) {
         continue;
      }
      TString rootMapBaseStr = sharedLibBaseStr;
      if (sharedLibBaseStr.EndsWith(".dll")) {
         rootMapBaseStr.ReplaceAll(".dll", "");
      }
      else if (sharedLibBaseStr.EndsWith(".DLL")) {
         rootMapBaseStr.ReplaceAll(".DLL", "");
      }
      else if (sharedLibBaseStr.EndsWith(".so")) {
         rootMapBaseStr.ReplaceAll(".so", "");
      }
      else if (sharedLibBaseStr.EndsWith(".sl")) {
         rootMapBaseStr.ReplaceAll(".sl", "");
      }
      else if (sharedLibBaseStr.EndsWith(".dl")) {
         rootMapBaseStr.ReplaceAll(".dl", "");
      }
      else if (sharedLibBaseStr.EndsWith(".a")) {
         rootMapBaseStr.ReplaceAll(".a", "");
      }
      else {
         Error("ReloadAllSharedLibraryMaps", "Unknown library type %s", sharedLibBaseStr.Data());
         delete sharedLibL;
         return -1;
      }
      rootMapBaseStr += ".rootmap";
      const char* rootMap = gSystem->Which(gSystem->GetDynamicPath(), rootMapBaseStr);
      if (!rootMap) {
         Error("ReloadAllSharedLibraryMaps", "Could not find rootmap %s in path", rootMap);
         delete[] rootMap;
         delete sharedLibL;
         return -1;
      }
      const Int_t status = LoadLibraryMap(rootMap);
      if (status < 0) {
         Error("ReloadAllSharedLibraryMaps", "Error loading map %s", rootMap);
         delete[] rootMap;
         delete sharedLibL;
         return -1;
      }
      delete[] rootMap;
   }
   delete sharedLibL;
   return 0;
}

//______________________________________________________________________________
Int_t TCintWithCling::UnloadAllSharedLibraryMaps()
{
   // Unload the library map entries coming from all the loaded shared libraries.
   // Returns 0 if succesful
   const TString sharedLibLStr = GetSharedLibs();
   const TObjArray* sharedLibL = sharedLibLStr.Tokenize(" ");
   for (Int_t ilib = 0; ilib < sharedLibL->GetEntriesFast(); ilib++) {
      const TString sharedLibStr = ((TObjString*)sharedLibL->At(ilib))->GetString();
      const  TString sharedLibBaseStr = gSystem->BaseName(sharedLibStr);
      UnloadLibraryMap(sharedLibBaseStr);
   }
   delete sharedLibL;
   return 0;
}

//______________________________________________________________________________
Int_t TCintWithCling::UnloadLibraryMap(const char* library)
{
   // Unload library map entries coming from the specified library.
   // Returns -1 in case no entries for the specified library were found,
   // 0 otherwise.
   if (!fMapfile || !library || !*library) {
      return 0;
   }
   TEnvRec* rec;
   TIter next(fMapfile->GetTable());
   R__LOCKGUARD(gCINTMutex);
   Int_t ret = 0;
   while ((rec = (TEnvRec*) next())) {
      TString cls = rec->GetName();
      if (!strncmp(cls.Data(), "Library.", 8) && cls.Length() > 8) {
         // get the first lib from the list of lib and dependent libs
         TString libs = rec->GetValue();
         if (libs == "") {
            continue;
         }
         TString delim(" ");
         TObjArray* tokens = libs.Tokenize(delim);
         const char* lib = ((TObjString*)tokens->At(0))->GetName();
         // convert "@@" to "::", we used "@@" because TEnv
         // considers "::" a terminator
         cls.Remove(0, 8);
         cls.ReplaceAll("@@", "::");
         // convert "-" to " ", since class names may have
         // blanks and TEnv considers a blank a terminator
         cls.ReplaceAll("-", " ");
         if (cls.Contains(":")) {
            // We have a namespace and we have to check it first
            int slen = cls.Length();
            for (int k = 0; k < slen; k++) {
               if (cls[k] == ':') {
                  if (k + 1 >= slen || cls[k + 1] != ':') {
                     // we expected another ':'
                     break;
                  }
                  if (k) {
                     TString base = cls(0, k);
                     if (base == "std") {
                        // std is not declared but is also ignored by CINT!
                        break;
                     }
                     else {
                        // Only declared the namespace do not specify any library because
                        // the namespace might be spread over several libraries and we do not
                        // know (yet?) which one the user will need!
                        //G__remove_from_class_autoloading_table((char*)base.Data());
                     }
                     ++k;
                  }
               }
               else if (cls[k] == '<') {
                  // We do not want to look at the namespace inside the template parameters!
                  break;
               }
            }
         }
         if (!strcmp(library, lib)) {
            if (fMapfile->GetTable()->Remove(rec) == 0) {
               Error("UnloadLibraryMap", "entry for <%s,%s> not found in library map table", cls.Data(), lib);
               ret = -1;
            }
            G__set_class_autoloading_table((char*)cls.Data(), (char*) - 1);
            G__security_recover(stderr); // Ignore any error during this setting.
         }
         delete tokens;
      }
   }
   if (ret >= 0) {
      TString library_rootmap(library);
      library_rootmap.Append(".rootmap");
      TNamed* mfile = 0;
      while ((mfile = (TNamed*)fRootmapFiles->FindObject(library_rootmap))) {
         fRootmapFiles->Remove(mfile);
         delete mfile;
      }
      fRootmapFiles->Compress();
   }
   return ret;
}

//______________________________________________________________________________
Int_t TCintWithCling::AutoLoad(const char* cls)
{
   // Load library containing the specified class. Returns 0 in case of error
   // and 1 in case if success.
   R__LOCKGUARD(gCINTMutex);
   Int_t status = 0;
   if (!gROOT || !gInterpreter || gROOT->TestBit(TObject::kInvalidObject)) {
      return status;
   }
   // Prevent the recursion when the library dictionary are loaded.
   Int_t oldvalue = G__set_class_autoloading(0);
   // lookup class to find list of dependent libraries
   TString deplibs = GetClassSharedLibs(cls);
   if (!deplibs.IsNull()) {
      TString delim(" ");
      TObjArray* tokens = deplibs.Tokenize(delim);
      for (Int_t i = tokens->GetEntriesFast() - 1; i > 0; i--) {
         const char* deplib = ((TObjString*)tokens->At(i))->GetName();
         if (gROOT->LoadClass(cls, deplib) == 0) {
            if (gDebug > 0)
               ::Info("TCintWithCling::AutoLoad", "loaded dependent library %s for class %s",
                      deplib, cls);
         }
         else
            ::Error("TCintWithCling::AutoLoad", "failure loading dependent library %s for class %s",
                    deplib, cls);
      }
      const char* lib = ((TObjString*)tokens->At(0))->GetName();
      if (lib[0]) {
         if (gROOT->LoadClass(cls, lib) == 0) {
            if (gDebug > 0)
               ::Info("TCintWithCling::AutoLoad", "loaded library %s for class %s",
                      lib, cls);
            status = 1;
         }
         else
            ::Error("TCintWithCling::AutoLoad", "failure loading library %s for class %s",
                    lib, cls);
      }
      delete tokens;
   }
   G__set_class_autoloading(oldvalue);
   return status;
}

//______________________________________________________________________________
Int_t TCintWithCling::AutoLoadCallback(const char* cls, const char* lib)
{
   // Load library containing specified class. Returns 0 in case of error
   // and 1 in case if success.
   R__LOCKGUARD(gCINTMutex);
   if (!gROOT || !gInterpreter || !cls || !lib) {
      return 0;
   }
   // calls to load libCore might come in the very beginning when libCore
   // dictionary is not fully loaded yet, ignore it since libCore is always
   // loaded
   if (strstr(lib, "libCore")) {
      return 1;
   }
   // lookup class to find list of dependent libraries
   TString deplibs = gInterpreter->GetClassSharedLibs(cls);
   if (!deplibs.IsNull()) {
      if (gDebug > 0 && gDebug <= 4)
         ::Info("TCintWithCling::AutoLoadCallback", "loaded dependent library %s for class %s",
                deplibs.Data(), cls);
      TString delim(" ");
      TObjArray* tokens = deplibs.Tokenize(delim);
      for (Int_t i = tokens->GetEntriesFast() - 1; i > 0; i--) {
         const char* deplib = ((TObjString*)tokens->At(i))->GetName();
         if (gROOT->LoadClass(cls, deplib) == 0) {
            if (gDebug > 4)
               ::Info("TCintWithCling::AutoLoadCallback", "loaded dependent library %s for class %s",
                      deplib, cls);
         }
         else {
            ::Error("TCintWithCling::AutoLoadCallback", "failure loading dependent library %s for class %s",
                    deplib, cls);
         }
      }
      delete tokens;
   }
   if (lib[0]) {
      if (gROOT->LoadClass(cls, lib) == 0) {
         if (gDebug > 0)
            ::Info("TCintWithCling::AutoLoadCallback", "loaded library %s for class %s",
                   lib, cls);
         return 1;
      }
      else {
         ::Error("TCintWithCling::AutoLoadCallback", "failure loading library %s for class %s",
                 lib, cls);
      }
   }
   return 0;
}

//______________________________________________________________________________
//FIXME: Use of G__ClassInfo in the interface!
void* TCintWithCling::FindSpecialObject(const char* item, G__ClassInfo* type,
                               void** prevObj, void** assocPtr)
{
   // Static function called by CINT when it finds an un-indentified object.
   // This function tries to find the UO in the ROOT files, directories, etc.
   // This functions has been registered by the TCint ctor.
   if (!*prevObj || *assocPtr != gDirectory) {
      *prevObj = gROOT->FindSpecialObject(item, *assocPtr);
      if (!fgSetOfSpecials) {
         fgSetOfSpecials = new std::set<TObject*>;
      }
      if (*prevObj) {
         ((std::set<TObject*>*)fgSetOfSpecials)->insert((TObject*)*prevObj);
      }
   }
   if (*prevObj) {
      type->Init(((TObject*)*prevObj)->ClassName());
   }
   return *prevObj;
}

//______________________________________________________________________________
// Helper class for UpdateClassInfo
namespace {
class TInfoNode {
private:
   string fName;
   Long_t fTagnum;
public:
   TInfoNode(const char* item, Long_t tagnum)
      : fName(item), fTagnum(tagnum)
   {}
   void Update() {
      TCintWithCling::UpdateClassInfoWork(fName.c_str(), fTagnum);
   }
};
}

//______________________________________________________________________________
void TCintWithCling::UpdateClassInfo(char* item, Long_t tagnum)
{
   // Static function called by CINT when it changes the tagnum for
   // a class (e.g. after re-executing the setup function). In such
   // cases we have to update the tagnum in the G__ClassInfo used by
   // the TClass for class "item".
   R__LOCKGUARD(gCINTMutex);
   if (gROOT && gROOT->GetListOfClasses()) {
      static Bool_t entered = kFALSE;
      static vector<TInfoNode> updateList;
      Bool_t topLevel;
      if (entered) {
         topLevel = kFALSE;
      }
      else {
         entered = kTRUE;
         topLevel = kTRUE;
      }
      if (topLevel) {
         UpdateClassInfoWork(item, tagnum);
      }
      else {
         // If we are called indirectly from within another call to
         // TCintWithCling::UpdateClassInfo, we delay the update until the dictionary loading
         // is finished (i.e. when we return to the top level TCintWithCling::UpdateClassInfo).
         // This allows for the dictionary to be fully populated when we actually
         // update the TClass object.   The updating of the TClass sometimes
         // (STL containers and when there is an emulated class) forces the building
         // of the TClass object's real data (which needs the dictionary info).
         updateList.push_back(TInfoNode(item, tagnum));
      }
      if (topLevel) {
         while (!updateList.empty()) {
            TInfoNode current(updateList.back());
            updateList.pop_back();
            current.Update();
         }
         entered = kFALSE;
      }
   }
}

//______________________________________________________________________________
void TCintWithCling::UpdateClassInfoWork(const char* item, Long_t tagnum)
{
   // This does the actual work of UpdateClassInfo.
   Bool_t load = kFALSE;
   if (strchr(item, '<') && TClass::GetClassShortTypedefHash()) {
      // We have a template which may have duplicates.
      TString resolvedItem(
         TClassEdit::ResolveTypedef(TClassEdit::ShortType(item,
                                    TClassEdit::kDropStlDefault).c_str(), kTRUE));
      if (resolvedItem != item) {
         TClass* cl = (TClass*)gROOT->GetListOfClasses()->FindObject(resolvedItem);
         if (cl) {
            load = kTRUE;
         }
      }
      if (!load) {
         TIter next(TClass::GetClassShortTypedefHash()->GetListForObject(resolvedItem));
         while (TClass::TNameMapNode* htmp =
                   static_cast<TClass::TNameMapNode*>(next())) {
            if (resolvedItem == htmp->String()) {
               TClass* cl = gROOT->GetClass(htmp->fOrigName, kFALSE);
               if (cl) {
                  // we found at least one equivalent.
                  // let's force a reload
                  load = kTRUE;
                  break;
               }
            }
         }
      }
   }
   TClass* cl = gROOT->GetClass(item, load);
   if (cl) {
      cl->ResetClassInfo(tagnum);
   }
}

//______________________________________________________________________________
void TCintWithCling::UpdateAllCanvases()
{
   // Update all canvases at end the terminal input command.
   TIter next(gROOT->GetListOfCanvases());
   TVirtualPad* canvas;
   while ((canvas = (TVirtualPad*)next())) {
      canvas->Update();
   }
}

//______________________________________________________________________________
const char* TCintWithCling::GetSharedLibs()
{
   // Return the list of shared libraries known to CINT.
   if (fSharedLibsSerial == G__SourceFileInfo::SerialNumber()) {
      return fSharedLibs;
   }
   fSharedLibsSerial = G__SourceFileInfo::SerialNumber();
   fSharedLibs.Clear();
   G__SourceFileInfo cursor(0);
   while (cursor.IsValid()) {
      const char* filename = cursor.Name();
      if (filename == 0) {
         continue;
      }
      Int_t len = strlen(filename);
      const char* end = filename + len;
      Bool_t needToSkip = kFALSE;
      if (len > 5 && ((strcmp(end - 4, ".dll") == 0) || (strstr(filename, "Dict.") != 0)  || (strstr(filename, "MetaTCint") != 0))) {
         // Filter out the cintdlls
         static const char* excludelist [] = {
            "stdfunc.dll", "stdcxxfunc.dll", "posix.dll", "ipc.dll", "posix.dll"
            "string.dll", "vector.dll", "vectorbool.dll", "list.dll", "deque.dll",
            "map.dll", "map2.dll", "set.dll", "multimap.dll", "multimap2.dll",
            "multiset.dll", "stack.dll", "queue.dll", "valarray.dll",
            "exception.dll", "stdexcept.dll", "complex.dll", "climits.dll",
            "libvectorDict.", "libvectorboolDict.", "liblistDict.", "libdequeDict.",
            "libmapDict.", "libmap2Dict.", "libsetDict.", "libmultimapDict.", "libmultimap2Dict.",
            "libmultisetDict.", "libstackDict.", "libqueueDict.", "libvalarrayDict."
         };
         static const unsigned int excludelistsize = sizeof(excludelist) / sizeof(excludelist[0]);
         static int excludelen[excludelistsize] = { -1};
         if (excludelen[0] == -1) {
            for (unsigned int i = 0; i < excludelistsize; ++i) {
               excludelen[i] = strlen(excludelist[i]);
            }
         }
         const char* basename = gSystem->BaseName(filename);
         for (unsigned int i = 0; !needToSkip && i < excludelistsize; ++i) {
            needToSkip = (!strncmp(basename, excludelist[i], excludelen[i]));
         }
      }
      if (!needToSkip &&
            (
#if defined(R__MACOSX) && defined(MAC_OS_X_VERSION_10_5)
               (dlopen_preflight(filename)) ||
#endif
               (len > 2 && strcmp(end - 2, ".a") == 0)    ||
               (len > 3 && (strcmp(end - 3, ".sl") == 0   ||
                            strcmp(end - 3, ".dl") == 0   ||
                            strcmp(end - 3, ".so") == 0)) ||
               (len > 4 && (strcasecmp(end - 4, ".dll") == 0)) ||
               (len > 6 && (strcasecmp(end - 6, ".dylib") == 0)))) {
         if (!fSharedLibs.IsNull()) {
            fSharedLibs.Append(" ");
         }
         fSharedLibs.Append(filename);
      }
      cursor.Next();
   }
   return fSharedLibs;
}

//______________________________________________________________________________
const char* TCintWithCling::GetClassSharedLibs(const char* cls)
{
   // Get the list of shared libraries containing the code for class cls.
   // The first library in the list is the one containing the class, the
   // others are the libraries the first one depends on. Returns 0
   // in case the library is not found.
   if (!cls || !*cls) {
      return 0;
   }
   // lookup class to find list of libraries
   if (fMapfile) {
      TString c = TString("Library.") + cls;
      // convert "::" to "@@", we used "@@" because TEnv
      // considers "::" a terminator
      c.ReplaceAll("::", "@@");
      // convert "-" to " ", since class names may have
      // blanks and TEnv considers a blank a terminator
      c.ReplaceAll(" ", "-");
      // Use TEnv::Lookup here as the rootmap file must start with Library.
      // and do not support using any stars (so we do not need to waste time
      // with the search made by TEnv::GetValue).
      TEnvRec* libs_record = fMapfile->Lookup(c);
      if (libs_record) {
         const char* libs = libs_record->GetValue();
         return (*libs) ? libs : 0;
      }
   }
   return 0;
}

//______________________________________________________________________________
const char* TCintWithCling::GetSharedLibDeps(const char* lib)
{
   // Get the list a libraries on which the specified lib depends. The
   // returned string contains as first element the lib itself.
   // Returns 0 in case the lib does not exist or does not have
   // any dependencies.
   if (!fMapfile || !lib || !lib[0]) {
      return 0;
   }
   TString libname(lib);
   Ssiz_t idx = libname.Last('.');
   if (idx != kNPOS) {
      libname.Remove(idx);
   }
   TEnvRec* rec;
   TIter next(fMapfile->GetTable());
   size_t len = libname.Length();
   while ((rec = (TEnvRec*) next())) {
      const char* libs = rec->GetValue();
      if (!strncmp(libs, libname.Data(), len) && strlen(libs) >= len
            && (!libs[len] || libs[len] == ' ' || libs[len] == '.')) {
         return libs;
      }
   }
   return 0;
}

//______________________________________________________________________________
Bool_t TCintWithCling::IsErrorMessagesEnabled() const
{
   // If error messages are disabled, the interpreter should suppress its
   // failures and warning messages from stdout.
   return !G__const_whatnoerror();
}

//______________________________________________________________________________
Bool_t TCintWithCling::SetErrorMessages(Bool_t enable)
{
   // If error messages are disabled, the interpreter should suppress its
   // failures and warning messages from stdout. Return the previous state.
   if (enable) {
      G__const_resetnoerror();
   }
   else {
      G__const_setnoerror();
   }
   return !G__const_whatnoerror();
}

//______________________________________________________________________________
const char* TCintWithCling::GetIncludePath()
{
   // Refresh the list of include paths known to the interpreter and return it
   // with -I prepended.
   R__LOCKGUARD(gCINTMutex);
   fIncludePath = "";
   G__IncludePathInfo path;
   while (path.Next()) {
      const char* pathname = path.Name();
      fIncludePath.Append(" -I\"").Append(pathname).Append("\" ");
   }
   return fIncludePath;
}

//______________________________________________________________________________
const char* TCintWithCling::GetSTLIncludePath() const
{
   // Return the directory containing CINT's stl cintdlls.
   static TString stldir;
   if (!stldir.Length()) {
#ifdef CINTINCDIR
      stldir = CINTINCDIR;
#else
      stldir = gRootDir;
      stldir += "/cint";
#endif
      if (!stldir.EndsWith("/")) {
         stldir += '/';
      }
      stldir += "cint/stl";
   }
   return stldir;
}

//______________________________________________________________________________
//                      M I S C
//______________________________________________________________________________

int TCintWithCling::DisplayClass(FILE* fout, char* name, int base, int start) const
{
   // Interface to CINT function
   return G__display_class(fout, name, base, start);
}

//______________________________________________________________________________
int TCintWithCling::DisplayIncludePath(FILE* fout) const
{
   // Interface to CINT function
   return G__display_includepath(fout);
}

//______________________________________________________________________________
void* TCintWithCling::FindSym(const char* entry) const
{
   // Interface to CINT function
   return G__findsym(entry);
}

//______________________________________________________________________________
void TCintWithCling::GenericError(const char* error) const
{
   // Interface to CINT function
   G__genericerror(error);
}

//______________________________________________________________________________
Long_t TCintWithCling::GetExecByteCode() const
{
   // Interface to CINT function
   return (Long_t)G__exec_bytecode;
}

//______________________________________________________________________________
Long_t TCintWithCling::Getgvp() const
{
   // Interface to CINT function
   return (Long_t)G__getgvp();
}

//______________________________________________________________________________
const char* TCintWithCling::Getp2f2funcname(void* receiver) const
{
   // Interface to CINT function
   return G__p2f2funcname(receiver);
}

//______________________________________________________________________________
int TCintWithCling::GetSecurityError() const
{
   // Interface to CINT function
   return G__get_security_error();
}

//______________________________________________________________________________
int TCintWithCling::LoadFile(const char* path) const
{
   // Interface to CINT function
   return G__loadfile(path);
}

//______________________________________________________________________________
void TCintWithCling::LoadText(const char* text) const
{
   // Interface to CINT function
   G__load_text(text);
}

//______________________________________________________________________________
const char* TCintWithCling::MapCppName(const char* name) const
{
   // Interface to CINT function
   return G__map_cpp_name(name);
}

//______________________________________________________________________________
void TCintWithCling::SetAlloclockfunc(void (*p)()) const
{
   // Interface to CINT function
   G__set_alloclockfunc(p);
}

//______________________________________________________________________________
void TCintWithCling::SetAllocunlockfunc(void (*p)()) const
{
   // Interface to CINT function
   G__set_allocunlockfunc(p);
}

//______________________________________________________________________________
int TCintWithCling::SetClassAutoloading(int autoload) const
{
   // Interface to CINT function
   return G__set_class_autoloading(autoload);
}

//______________________________________________________________________________
void TCintWithCling::SetErrmsgcallback(void* p) const
{
   // Interface to CINT function
   G__set_errmsgcallback(p);
}

//______________________________________________________________________________
void TCintWithCling::Setgvp(Long_t gvp) const
{
   // Interface to CINT function
   G__setgvp(gvp);
}

//______________________________________________________________________________
void TCintWithCling::SetRTLD_NOW() const
{
   // Interface to CINT function
   G__Set_RTLD_NOW();
}

//______________________________________________________________________________
void TCintWithCling::SetRTLD_LAZY() const
{
   // Interface to CINT function
   G__Set_RTLD_LAZY();
}

//______________________________________________________________________________
void TCintWithCling::SetTempLevel(int val) const
{
   // Interface to CINT function
   G__settemplevel(val);
}

//______________________________________________________________________________
int TCintWithCling::UnloadFile(const char* path) const
{
   // Interface to CINT function
   return G__unloadfile(path);
}



//______________________________________________________________________________
//
//  G__CallFunc interface
//

//______________________________________________________________________________
void TCintWithCling::CallFunc_Delete(CallFunc_t* func) const
{
   delete(tcling_CallFunc*) func;
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_Exec(CallFunc_t* func, void* address) const
{
   tcling_CallFunc* f = (tcling_CallFunc*) func;
   f->Exec(address);
}

//______________________________________________________________________________
Long_t TCintWithCling::CallFunc_ExecInt(CallFunc_t* func, void* address) const
{
   tcling_CallFunc* f = (tcling_CallFunc*) func;
   return f->ExecInt(address);
}

//______________________________________________________________________________
Long_t TCintWithCling::CallFunc_ExecInt64(CallFunc_t* func, void* address) const
{
   tcling_CallFunc* f = (tcling_CallFunc*) func;
   return f->ExecInt64(address);
}

//______________________________________________________________________________
Double_t TCintWithCling::CallFunc_ExecDouble(CallFunc_t* func, void* address) const
{
   tcling_CallFunc* f = (tcling_CallFunc*) func;
   return f->ExecDouble(address);
}

//______________________________________________________________________________
CallFunc_t* TCintWithCling::CallFunc_Factory() const
{
   return (CallFunc_t*) new tcling_CallFunc();
}

//______________________________________________________________________________
CallFunc_t* TCintWithCling::CallFunc_FactoryCopy(CallFunc_t* func) const
{
   return (CallFunc_t*) new tcling_CallFunc(*(tcling_CallFunc*)func);
}

//______________________________________________________________________________
MethodInfo_t* TCintWithCling::CallFunc_FactoryMethod(CallFunc_t* func) const
{
   tcling_CallFunc* f = (tcling_CallFunc*) func;
   return (MethodInfo_t*) f->FactoryMethod();
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_Init(CallFunc_t* func) const
{
   tcling_CallFunc* f = (tcling_CallFunc*) func;
   f->Init();
}

//______________________________________________________________________________
bool TCintWithCling::CallFunc_IsValid(CallFunc_t* func) const
{
   tcling_CallFunc* f = (tcling_CallFunc*) func;
   return f->IsValid();
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_ResetArg(CallFunc_t* func) const
{
   tcling_CallFunc* f = (tcling_CallFunc*) func;
   f->ResetArg();
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_SetArg(CallFunc_t* func, Long_t param) const
{
   tcling_CallFunc* f = (tcling_CallFunc*) func;
   f->SetArg(param);
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_SetArg(CallFunc_t* func, Double_t param) const
{
   tcling_CallFunc* f = (tcling_CallFunc*) func;
   f->SetArg(param);
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_SetArg(CallFunc_t* func, Long64_t param) const
{
   tcling_CallFunc* f = (tcling_CallFunc*) func;
   f->SetArg(param);
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_SetArg(CallFunc_t* func, ULong64_t param) const
{
   tcling_CallFunc* f = (tcling_CallFunc*) func;
   f->SetArg(param);
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_SetArgArray(CallFunc_t* func, Long_t* paramArr, Int_t nparam) const
{
   tcling_CallFunc* f = (tcling_CallFunc*) func;
   f->SetArgArray(paramArr, nparam);
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_SetArgs(CallFunc_t* func, const char* param) const
{
   tcling_CallFunc* f = (tcling_CallFunc*) func;
   f->SetArgs(param);
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_SetFunc(CallFunc_t* func, ClassInfo_t* info, const char* method, const char* params, Long_t* offset) const
{
   tcling_CallFunc* f = (tcling_CallFunc*) func;
   tcling_ClassInfo* ci = (tcling_ClassInfo*) info;
   f->SetFunc(ci, method, params, offset);
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_SetFunc(CallFunc_t* func, MethodInfo_t* info) const
{
   tcling_CallFunc* f = (tcling_CallFunc*) func;
   tcling_MethodInfo* minfo = (tcling_MethodInfo*) info;
   f->SetFunc(minfo);
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_SetFuncProto(CallFunc_t* func, ClassInfo_t* info, const char* method, const char* proto, Long_t* offset) const
{
   // Interface to CINT function
   tcling_CallFunc* f = (tcling_CallFunc*) func;
   tcling_ClassInfo* ci = (tcling_ClassInfo*) info;
   f->SetFuncProto(ci, method, proto, offset);
}


//______________________________________________________________________________
//
//  G__ClassInfo interface
//

Long_t TCintWithCling::ClassInfo_ClassProperty(ClassInfo_t* cinfo) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return tcling_info->ClassProperty();
}

//______________________________________________________________________________
void TCintWithCling::ClassInfo_Delete(ClassInfo_t* cinfo) const
{
   delete(tcling_ClassInfo*) cinfo;
}

//______________________________________________________________________________
void TCintWithCling::ClassInfo_Delete(ClassInfo_t* cinfo, void* arena) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   tcling_info->Delete(arena);
}

//______________________________________________________________________________
void TCintWithCling::ClassInfo_DeleteArray(ClassInfo_t* cinfo, void* arena, bool dtorOnly) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   tcling_info->DeleteArray(arena, dtorOnly);
}

//______________________________________________________________________________
void TCintWithCling::ClassInfo_Destruct(ClassInfo_t* cinfo, void* arena) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   tcling_info->Destruct(arena);
}

//______________________________________________________________________________
ClassInfo_t* TCintWithCling::ClassInfo_Factory() const
{
   return (ClassInfo_t*) new tcling_ClassInfo();
}

//______________________________________________________________________________
ClassInfo_t* TCintWithCling::ClassInfo_Factory(ClassInfo_t* cinfo) const
{
   return (ClassInfo_t*) new tcling_ClassInfo(*(tcling_ClassInfo*)cinfo);
}

//______________________________________________________________________________
ClassInfo_t* TCintWithCling::ClassInfo_Factory(const char* name) const
{
   return (ClassInfo_t*) new tcling_ClassInfo(name);
}

//______________________________________________________________________________
int TCintWithCling::ClassInfo_GetMethodNArg(ClassInfo_t* cinfo, const char* method, const char* proto) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return tcling_info->GetMethodNArg(method, proto);
}

//______________________________________________________________________________
bool TCintWithCling::ClassInfo_HasDefaultConstructor(ClassInfo_t* cinfo) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return tcling_info->HasDefaultConstructor();
}

//______________________________________________________________________________
bool TCintWithCling::ClassInfo_HasMethod(ClassInfo_t* cinfo, const char* name) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return tcling_info->HasMethod(name);
}

//______________________________________________________________________________
void TCintWithCling::ClassInfo_Init(ClassInfo_t* cinfo, const char* name) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   tcling_info->Init(name);
}

//______________________________________________________________________________
void TCintWithCling::ClassInfo_Init(ClassInfo_t* cinfo, int tagnum) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   tcling_info->Init(tagnum);
}

//______________________________________________________________________________
bool TCintWithCling::ClassInfo_IsBase(ClassInfo_t* cinfo, const char* name) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return tcling_info->IsBase(name);
}

//______________________________________________________________________________
bool TCintWithCling::ClassInfo_IsEnum(const char* name) const
{
   return tcling_ClassInfo::IsEnum(name);
}

//______________________________________________________________________________
bool TCintWithCling::ClassInfo_IsLoaded(ClassInfo_t* cinfo) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return tcling_info->IsLoaded();
}

//______________________________________________________________________________
bool TCintWithCling::ClassInfo_IsValid(ClassInfo_t* cinfo) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return tcling_info->IsValid();
}

//______________________________________________________________________________
bool TCintWithCling::ClassInfo_IsValidMethod(ClassInfo_t* cinfo, const char* method, const char* proto, Long_t* offset) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return tcling_info->IsValidMethod(method, proto, offset);
}

//______________________________________________________________________________
int TCintWithCling::ClassInfo_Next(ClassInfo_t* cinfo) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return tcling_info->Next();
}

//______________________________________________________________________________
void* TCintWithCling::ClassInfo_New(ClassInfo_t* cinfo) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return tcling_info->New();
}

//______________________________________________________________________________
void* TCintWithCling::ClassInfo_New(ClassInfo_t* cinfo, int n) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return tcling_info->New(n);
}

//______________________________________________________________________________
void* TCintWithCling::ClassInfo_New(ClassInfo_t* cinfo, int n, void* arena) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return tcling_info->New(n, arena);
}

//______________________________________________________________________________
void* TCintWithCling::ClassInfo_New(ClassInfo_t* cinfo, void* arena) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return tcling_info->New(arena);
}

//______________________________________________________________________________
Long_t TCintWithCling::ClassInfo_Property(ClassInfo_t* cinfo) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return tcling_info->Property();
}

//______________________________________________________________________________
int TCintWithCling::ClassInfo_RootFlag(ClassInfo_t* cinfo) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return tcling_info->RootFlag();
}

//______________________________________________________________________________
int TCintWithCling::ClassInfo_Size(ClassInfo_t* cinfo) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return tcling_info->Size();
}

//______________________________________________________________________________
Long_t TCintWithCling::ClassInfo_Tagnum(ClassInfo_t* cinfo) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return tcling_info->Tagnum();
}

//______________________________________________________________________________
const char* TCintWithCling::ClassInfo_FileName(ClassInfo_t* cinfo) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return tcling_info->FileName();
}

//______________________________________________________________________________
const char* TCintWithCling::ClassInfo_FullName(ClassInfo_t* cinfo) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return tcling_info->FullName();
}

//______________________________________________________________________________
const char* TCintWithCling::ClassInfo_Name(ClassInfo_t* cinfo) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return tcling_info->Name();
}

//______________________________________________________________________________
const char* TCintWithCling::ClassInfo_Title(ClassInfo_t* cinfo) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return tcling_info->Title();
}

//______________________________________________________________________________
const char* TCintWithCling::ClassInfo_TmpltName(ClassInfo_t* cinfo) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return tcling_info->TmpltName();
}



//______________________________________________________________________________
//
//  G__BaseClassInfo interface
//

//______________________________________________________________________________
void TCintWithCling::BaseClassInfo_Delete(BaseClassInfo_t* bcinfo) const
{
   delete(tcling_BaseClassInfo*) bcinfo;
}

//______________________________________________________________________________
BaseClassInfo_t* TCintWithCling::BaseClassInfo_Factory(ClassInfo_t* cinfo) const
{
   tcling_ClassInfo* tcling_info = (tcling_ClassInfo*) cinfo;
   return (BaseClassInfo_t*) new tcling_BaseClassInfo(tcling_info);
}

//______________________________________________________________________________
int TCintWithCling::BaseClassInfo_Next(BaseClassInfo_t* bcinfo) const
{
   tcling_BaseClassInfo* tcling_info = (tcling_BaseClassInfo*) bcinfo;
   return tcling_info->Next();
}

//______________________________________________________________________________
int TCintWithCling::BaseClassInfo_Next(BaseClassInfo_t* bcinfo, int onlyDirect) const
{
   tcling_BaseClassInfo* tcling_info = (tcling_BaseClassInfo*) bcinfo;
   return tcling_info->Next(onlyDirect);
}

//______________________________________________________________________________
Long_t TCintWithCling::BaseClassInfo_Offset(BaseClassInfo_t* bcinfo) const
{
   tcling_BaseClassInfo* tcling_info = (tcling_BaseClassInfo*) bcinfo;
   return tcling_info->Offset();
}

//______________________________________________________________________________
Long_t TCintWithCling::BaseClassInfo_Property(BaseClassInfo_t* bcinfo) const
{
   tcling_BaseClassInfo* tcling_info = (tcling_BaseClassInfo*) bcinfo;
   return tcling_info->Property();
}

//______________________________________________________________________________
Long_t TCintWithCling::BaseClassInfo_Tagnum(BaseClassInfo_t* bcinfo) const
{
   tcling_BaseClassInfo* tcling_info = (tcling_BaseClassInfo*) bcinfo;
   return tcling_info->Tagnum();
}

//______________________________________________________________________________
const char* TCintWithCling::BaseClassInfo_FullName(BaseClassInfo_t* bcinfo) const
{
   tcling_BaseClassInfo* tcling_info = (tcling_BaseClassInfo*) bcinfo;
   return tcling_info->FullName();
}

//______________________________________________________________________________
const char* TCintWithCling::BaseClassInfo_Name(BaseClassInfo_t* bcinfo) const
{
   tcling_BaseClassInfo* tcling_info = (tcling_BaseClassInfo*) bcinfo;
   return tcling_info->Name();
}

//______________________________________________________________________________
const char* TCintWithCling::BaseClassInfo_TmpltName(BaseClassInfo_t* bcinfo) const
{
   tcling_BaseClassInfo* tcling_info = (tcling_BaseClassInfo*) bcinfo;
   return tcling_info->TmpltName();
}

//______________________________________________________________________________
//
//  G__DataMemberInfo interface
//

//______________________________________________________________________________
int TCintWithCling::DataMemberInfo_ArrayDim(DataMemberInfo_t* dminfo) const
{
   tcling_DataMemberInfo* tcling_info = (tcling_DataMemberInfo*) dminfo;
   return tcling_info->ArrayDim();
}

//______________________________________________________________________________
void TCintWithCling::DataMemberInfo_Delete(DataMemberInfo_t* dminfo) const
{
   delete(tcling_DataMemberInfo*) dminfo;
}

//______________________________________________________________________________
DataMemberInfo_t* TCintWithCling::DataMemberInfo_Factory(ClassInfo_t* clinfo /*= 0*/) const
{
   tcling_ClassInfo* tcling_class_info = (tcling_ClassInfo*) clinfo;
   return (DataMemberInfo_t*) new tcling_DataMemberInfo(tcling_class_info);
}

//______________________________________________________________________________
DataMemberInfo_t* TCintWithCling::DataMemberInfo_FactoryCopy(DataMemberInfo_t* dminfo) const
{
   tcling_DataMemberInfo* tcling_info = (tcling_DataMemberInfo*) dminfo;
   return (DataMemberInfo_t*) new tcling_DataMemberInfo(*tcling_info);
}

//______________________________________________________________________________
bool TCintWithCling::DataMemberInfo_IsValid(DataMemberInfo_t* dminfo) const
{
   tcling_DataMemberInfo* tcling_info = (tcling_DataMemberInfo*) dminfo;
   return tcling_info->IsValid();
}

//______________________________________________________________________________
int TCintWithCling::DataMemberInfo_MaxIndex(DataMemberInfo_t* dminfo, Int_t dim) const
{
   tcling_DataMemberInfo* tcling_info = (tcling_DataMemberInfo*) dminfo;
   return tcling_info->MaxIndex(dim);
}

//______________________________________________________________________________
int TCintWithCling::DataMemberInfo_Next(DataMemberInfo_t* dminfo) const
{
   tcling_DataMemberInfo* tcling_info = (tcling_DataMemberInfo*) dminfo;
   return tcling_info->Next();
}

//______________________________________________________________________________
Long_t TCintWithCling::DataMemberInfo_Offset(DataMemberInfo_t* dminfo) const
{
   tcling_DataMemberInfo* tcling_info = (tcling_DataMemberInfo*) dminfo;
   return tcling_info->Offset();
}

//______________________________________________________________________________
Long_t TCintWithCling::DataMemberInfo_Property(DataMemberInfo_t* dminfo) const
{
   tcling_DataMemberInfo* tcling_info = (tcling_DataMemberInfo*) dminfo;
   return tcling_info->Property();
}

//______________________________________________________________________________
Long_t TCintWithCling::DataMemberInfo_TypeProperty(DataMemberInfo_t* dminfo) const
{
   tcling_DataMemberInfo* tcling_info = (tcling_DataMemberInfo*) dminfo;
   return tcling_info->TypeProperty();
}

//______________________________________________________________________________
int TCintWithCling::DataMemberInfo_TypeSize(DataMemberInfo_t* dminfo) const
{
   tcling_DataMemberInfo* tcling_info = (tcling_DataMemberInfo*) dminfo;
   return tcling_info->TypeSize();
}

//______________________________________________________________________________
const char* TCintWithCling::DataMemberInfo_TypeName(DataMemberInfo_t* dminfo) const
{
   tcling_DataMemberInfo* tcling_info = (tcling_DataMemberInfo*) dminfo;
   return tcling_info->TypeName();
}

//______________________________________________________________________________
const char* TCintWithCling::DataMemberInfo_TypeTrueName(DataMemberInfo_t* dminfo) const
{
   tcling_DataMemberInfo* tcling_info = (tcling_DataMemberInfo*) dminfo;
   return tcling_info->TypeTrueName();
}

//______________________________________________________________________________
const char* TCintWithCling::DataMemberInfo_Name(DataMemberInfo_t* dminfo) const
{
   tcling_DataMemberInfo* tcling_info = (tcling_DataMemberInfo*) dminfo;
   return tcling_info->Name();
}

//______________________________________________________________________________
const char* TCintWithCling::DataMemberInfo_Title(DataMemberInfo_t* dminfo) const
{
   tcling_DataMemberInfo* tcling_info = (tcling_DataMemberInfo*) dminfo;
   return tcling_info->Title();
}

//______________________________________________________________________________
const char* TCintWithCling::DataMemberInfo_ValidArrayIndex(DataMemberInfo_t* dminfo) const
{
   tcling_DataMemberInfo* tcling_info = (tcling_DataMemberInfo*) dminfo;
   return tcling_info->ValidArrayIndex();
}



//______________________________________________________________________________
//
//  G__MethodInfo interface
//

//______________________________________________________________________________
void TCintWithCling::MethodInfo_Delete(MethodInfo_t* minfo) const
{
   // Interface to CINT function
   delete(tcling_MethodInfo*) minfo;
}

//______________________________________________________________________________
void TCintWithCling::MethodInfo_CreateSignature(MethodInfo_t* minfo, TString& signature) const
{
   tcling_MethodInfo* info = (tcling_MethodInfo*) minfo;
   info->CreateSignature(signature);
}

//______________________________________________________________________________
MethodInfo_t* TCintWithCling::MethodInfo_Factory() const
{
   return (MethodInfo_t*) new tcling_MethodInfo();
}

//______________________________________________________________________________
MethodInfo_t* TCintWithCling::MethodInfo_FactoryCopy(MethodInfo_t* minfo) const
{
   return (MethodInfo_t*) new tcling_MethodInfo(*(tcling_MethodInfo*)minfo);
}

//______________________________________________________________________________
void* TCintWithCling::MethodInfo_InterfaceMethod(MethodInfo_t* minfo) const
{
   tcling_MethodInfo* info = (tcling_MethodInfo*) minfo;
   return (void*) info->InterfaceMethod();
}

//______________________________________________________________________________
bool TCintWithCling::MethodInfo_IsValid(MethodInfo_t* minfo) const
{
   tcling_MethodInfo* info = (tcling_MethodInfo*) minfo;
   return info->IsValid();
}

//______________________________________________________________________________
int TCintWithCling::MethodInfo_NArg(MethodInfo_t* minfo) const
{
   tcling_MethodInfo* info = (tcling_MethodInfo*) minfo;
   return info->NArg();
}

//______________________________________________________________________________
int TCintWithCling::MethodInfo_NDefaultArg(MethodInfo_t* minfo) const
{
   tcling_MethodInfo* info = (tcling_MethodInfo*) minfo;
   return info->NDefaultArg();
}

//______________________________________________________________________________
int TCintWithCling::MethodInfo_Next(MethodInfo_t* minfo) const
{
   tcling_MethodInfo* info = (tcling_MethodInfo*) minfo;
   return info->Next();
}

//______________________________________________________________________________
Long_t TCintWithCling::MethodInfo_Property(MethodInfo_t* minfo) const
{
   tcling_MethodInfo* info = (tcling_MethodInfo*) minfo;
   return info->Property();
}

//______________________________________________________________________________
void* TCintWithCling::MethodInfo_Type(MethodInfo_t* minfo) const
{
   tcling_MethodInfo* info = (tcling_MethodInfo*) minfo;
   return info->Type();
}

//______________________________________________________________________________
const char* TCintWithCling::MethodInfo_GetMangledName(MethodInfo_t* minfo) const
{
   tcling_MethodInfo* info = (tcling_MethodInfo*) minfo;
   return info->GetMangledName();
}

//______________________________________________________________________________
const char* TCintWithCling::MethodInfo_GetPrototype(MethodInfo_t* minfo) const
{
   tcling_MethodInfo* info = (tcling_MethodInfo*) minfo;
   return info->GetPrototype();
}

//______________________________________________________________________________
const char* TCintWithCling::MethodInfo_Name(MethodInfo_t* minfo) const
{
   tcling_MethodInfo* info = (tcling_MethodInfo*) minfo;
   return info->Name();
}

//______________________________________________________________________________
const char* TCintWithCling::MethodInfo_TypeName(MethodInfo_t* minfo) const
{
   tcling_MethodInfo* info = (tcling_MethodInfo*) minfo;
   return info->Name();
}

//______________________________________________________________________________
const char* TCintWithCling::MethodInfo_Title(MethodInfo_t* minfo) const
{
   tcling_MethodInfo* info = (tcling_MethodInfo*) minfo;
   return info->Title();
}

//______________________________________________________________________________
//
//  G__MethodArgInfo interface
//

//______________________________________________________________________________
void TCintWithCling::MethodArgInfo_Delete(MethodArgInfo_t* marginfo) const
{
   delete (tcling_MethodArgInfo*) marginfo;
}

//______________________________________________________________________________
MethodArgInfo_t* TCintWithCling::MethodArgInfo_Factory() const
{
   return (MethodArgInfo_t*) new tcling_MethodArgInfo();
}

//______________________________________________________________________________
MethodArgInfo_t* TCintWithCling::MethodArgInfo_FactoryCopy(MethodArgInfo_t* marginfo) const
{
   return (MethodArgInfo_t*)
      new tcling_MethodArgInfo(*(tcling_MethodArgInfo*)marginfo);
}

//______________________________________________________________________________
bool TCintWithCling::MethodArgInfo_IsValid(MethodArgInfo_t* marginfo) const
{
   tcling_MethodArgInfo* info = (tcling_MethodArgInfo*) marginfo;
   return info->IsValid();
}

//______________________________________________________________________________
int TCintWithCling::MethodArgInfo_Next(MethodArgInfo_t* marginfo) const
{
   tcling_MethodArgInfo* info = (tcling_MethodArgInfo*) marginfo;
   return info->Next();
}

//______________________________________________________________________________
Long_t TCintWithCling::MethodArgInfo_Property(MethodArgInfo_t* marginfo) const
{
   tcling_MethodArgInfo* info = (tcling_MethodArgInfo*) marginfo;
   return info->Property();
}

//______________________________________________________________________________
const char* TCintWithCling::MethodArgInfo_DefaultValue(MethodArgInfo_t* marginfo) const
{
   tcling_MethodArgInfo* info = (tcling_MethodArgInfo*) marginfo;
   return info->DefaultValue();
}

//______________________________________________________________________________
const char* TCintWithCling::MethodArgInfo_Name(MethodArgInfo_t* marginfo) const
{
   tcling_MethodArgInfo* info = (tcling_MethodArgInfo*) marginfo;
   return info->Name();
}

//______________________________________________________________________________
const char* TCintWithCling::MethodArgInfo_TypeName(MethodArgInfo_t* marginfo) const
{
   tcling_MethodArgInfo* info = (tcling_MethodArgInfo*) marginfo;
   return info->TypeName();
}


//______________________________________________________________________________
//
//  G__TypeInfo interface
//

//______________________________________________________________________________
void TCintWithCling::TypeInfo_Delete(TypeInfo_t* tinfo) const
{
   delete (tcling_TypeInfo*) tinfo;
}

//______________________________________________________________________________
TypeInfo_t* TCintWithCling::TypeInfo_Factory() const
{
   return (TypeInfo_t*) new tcling_TypeInfo();
}

//______________________________________________________________________________
TypeInfo_t* TCintWithCling::TypeInfo_Factory(G__value* pvalue) const
{
   return (TypeInfo_t*) new tcling_TypeInfo(pvalue);
}

//______________________________________________________________________________
TypeInfo_t* TCintWithCling::TypeInfo_FactoryCopy(TypeInfo_t* tinfo) const
{
   return (TypeInfo_t*) new tcling_TypeInfo(*(tcling_TypeInfo*)tinfo);
}

//______________________________________________________________________________
void TCintWithCling::TypeInfo_Init(TypeInfo_t* tinfo, const char* name) const
{
   tcling_TypeInfo* tcling_info = (tcling_TypeInfo*) tinfo;
   tcling_info->Init(name);
}

//______________________________________________________________________________
bool TCintWithCling::TypeInfo_IsValid(TypeInfo_t* tinfo) const
{
   tcling_TypeInfo* tcling_info = (tcling_TypeInfo*) tinfo;
   return tcling_info->IsValid();
}

//______________________________________________________________________________
const char* TCintWithCling::TypeInfo_Name(TypeInfo_t* tinfo) const
{
   tcling_TypeInfo* tcling_info = (tcling_TypeInfo*) tinfo;
   return tcling_info->Name();
}

//______________________________________________________________________________
Long_t TCintWithCling::TypeInfo_Property(TypeInfo_t* tinfo) const
{
   tcling_TypeInfo* tcling_info = (tcling_TypeInfo*) tinfo;
   return tcling_info->Property();
}

//______________________________________________________________________________
int TCintWithCling::TypeInfo_RefType(TypeInfo_t* tinfo) const
{
   tcling_TypeInfo* tcling_info = (tcling_TypeInfo*) tinfo;
   return tcling_info->RefType();
}

//______________________________________________________________________________
int TCintWithCling::TypeInfo_Size(TypeInfo_t* tinfo) const
{
   tcling_TypeInfo* tcling_info = (tcling_TypeInfo*) tinfo;
   return tcling_info->Size();
}

//______________________________________________________________________________
const char* TCintWithCling::TypeInfo_TrueName(TypeInfo_t* tinfo) const
{
   tcling_TypeInfo* tcling_info = (tcling_TypeInfo*) tinfo;
   return tcling_info->TrueName();
}


//______________________________________________________________________________
//
//  G__TypedefInfo interface
//

//______________________________________________________________________________
void TCintWithCling::TypedefInfo_Delete(TypedefInfo_t* tinfo) const
{
   delete (tcling_TypedefInfo*) tinfo;
}

//______________________________________________________________________________
TypedefInfo_t* TCintWithCling::TypedefInfo_Factory() const
{
   return (TypedefInfo_t*) new tcling_TypedefInfo();
}

//______________________________________________________________________________
TypedefInfo_t* TCintWithCling::TypedefInfo_FactoryCopy(TypedefInfo_t* tinfo) const
{
   return (TypedefInfo_t*) new tcling_TypedefInfo(*(tcling_TypedefInfo*)tinfo);
}

//______________________________________________________________________________
TypedefInfo_t TCintWithCling::TypedefInfo_Init(TypedefInfo_t* tinfo,
                                      const char* name) const
{
   tcling_TypedefInfo* tcling_info = (tcling_TypedefInfo*) tinfo;
   tcling_info->Init(name);
}

//______________________________________________________________________________
bool TCintWithCling::TypedefInfo_IsValid(TypedefInfo_t* tinfo) const
{
   tcling_TypedefInfo* tcling_info = (tcling_TypedefInfo*) tinfo;
   return tcling_info->IsValid();
}

//______________________________________________________________________________
Long_t TCintWithCling::TypedefInfo_Property(TypedefInfo_t* tinfo) const
{
   tcling_TypedefInfo* tcling_info = (tcling_TypedefInfo*) tinfo;
   return tcling_info->Property();
}

//______________________________________________________________________________
int TCintWithCling::TypedefInfo_Size(TypedefInfo_t* tinfo) const
{
   tcling_TypedefInfo* tcling_info = (tcling_TypedefInfo*) tinfo;
   return tcling_info->Size();
}

//______________________________________________________________________________
const char* TCintWithCling::TypedefInfo_TrueName(TypedefInfo_t* tinfo) const
{
   tcling_TypedefInfo* tcling_info = (tcling_TypedefInfo*) tinfo;
   return tcling_info->TrueName();
}

//______________________________________________________________________________
const char* TCintWithCling::TypedefInfo_Name(TypedefInfo_t* tinfo) const
{
   tcling_TypedefInfo* tcling_info = (tcling_TypedefInfo*) tinfo;
   return tcling_info->Name();
}

//______________________________________________________________________________
const char* TCintWithCling::TypedefInfo_Title(TypedefInfo_t* tinfo) const
{
   tcling_TypedefInfo* tcling_info = (tcling_TypedefInfo*) tinfo;
   return tcling_info->Title();
}

