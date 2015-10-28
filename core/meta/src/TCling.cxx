// @(#)root/meta:$Id$
// vim: sw=3 ts=3 expandtab foldmethod=indent

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCling                                                               //
//                                                                      //
// This class defines an interface to the cling C++ interpreter.        //
//                                                                      //
// Cling is a full ANSI compliant C++-11 interpreter based on           //
// clang/LLVM technology.                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TCling.h"

#include "TClingBaseClassInfo.h"
#include "TClingCallFunc.h"
#include "TClingClassInfo.h"
#include "TClingDataMemberInfo.h"
#include "TClingMethodArgInfo.h"
#include "TClingMethodInfo.h"
#include "TClingTypedefInfo.h"
#include "TClingTypeInfo.h"
#include "TClingValue.h"

#include "TROOT.h"
#include "TApplication.h"
#include "TGlobal.h"
#include "TDataType.h"
#include "TClass.h"
#include "TClassEdit.h"
#include "TClassTable.h"
#include "TClingCallbacks.h"
#include "TBaseClass.h"
#include "TDataMember.h"
#include "TMemberInspector.h"
#include "TMethod.h"
#include "TMethodArg.h"
#include "TFunctionTemplate.h"
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
#include "TEnum.h"
#include "TEnumConstant.h"
#include "THashTable.h"
#include "RConfigure.h"
#include "compiledata.h"
#include "TMetaUtils.h"
#include "TVirtualCollectionProxy.h"
#include "TVirtualStreamerInfo.h"
#include "TListOfDataMembers.h"
#include "TListOfEnums.h"
#include "TListOfEnumsWithLock.h"
#include "TListOfFunctions.h"
#include "TListOfFunctionTemplates.h"
#include "TProtoClass.h"
#include "TStreamerInfo.h" // This is here to avoid to use the plugin manager
#include "ThreadLocalStorage.h"
#include "TFile.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clang/Parse/Parser.h"

#include "cling/Interpreter/ClangInternalState.h"
#include "cling/Interpreter/DynamicLibraryManager.h"
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/LookupHelper.h"
#include "cling/Interpreter/Value.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/MetaProcessor/MetaProcessor.h"
#include "cling/Utils/AST.h"

#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"

#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <iostream>
#include <cassert>
#include <map>
#include <set>
#include <stdexcept>
#include <stdint.h>
#include <fstream>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef R__WIN32
#include <cxxabi.h>
#define R__DLLEXPORT
#endif
#include <limits.h>
#include <stdio.h>

#ifdef __APPLE__
#include <dlfcn.h>
#include <mach-o/dyld.h>
#endif // __APPLE__

#ifdef R__LINUX
#include <dlfcn.h>
#endif

#if defined(__CYGWIN__)
#include <sys/cygwin.h>
#define HMODULE void *
extern "C" {
   __declspec(dllimport) void * __stdcall GetCurrentProcess();
   __declspec(dllimport) bool __stdcall EnumProcessModules(void *, void **, unsigned long, unsigned long *);
   __declspec(dllimport) unsigned long __stdcall GetModuleFileNameExW(void *, void *, wchar_t *, unsigned long);
}
#endif

// Fragment copied from LLVM's raw_ostream.cpp
#if defined(_MSC_VER)
#ifndef STDIN_FILENO
# define STDIN_FILENO 0
#endif
#ifndef STDOUT_FILENO
# define STDOUT_FILENO 1
#endif
#ifndef STDERR_FILENO
# define STDERR_FILENO 2
#endif
#ifndef R__WIN32
//#if defined(HAVE_UNISTD_H)
# include <unistd.h>
//#endif
#else
#include "Windows4Root.h"
#include <Psapi.h>
#undef GetModuleFileName
#define RTLD_DEFAULT ((void *)::GetModuleHandle(NULL))
#define dlsym(library, function_name) ::GetProcAddress((HMODULE)library, function_name)
#define dlopen(library_name, flags) ::LoadLibrary(library_name)
#define dlclose(library) ::FreeLibrary((HMODULE)library)
#define R__DLLEXPORT __declspec(dllexport)
#endif
#endif

//______________________________________________________________________________
// These functions are helpers for debugging issues with non-LLVMDEV builds.
//
clang::DeclContext* TCling__DEBUG__getDeclContext(clang::Decl* D) {
   return D->getDeclContext();
}
clang::NamespaceDecl* TCling__DEBUG__DCtoNamespace(clang::DeclContext* DC) {
   return llvm::dyn_cast<clang::NamespaceDecl>(DC);
}
clang::RecordDecl* TCling__DEBUG__DCtoRecordDecl(clang::DeclContext* DC) {
   return llvm::dyn_cast<clang::RecordDecl>(DC);
}
void TCling__DEBUG__dump(clang::DeclContext* DC) {
   return DC->dumpDeclContext();
}
void TCling__DEBUG__dump(clang::Decl* D) {
   return D->dump();
}
void TCling__DEBUG__dump(clang::FunctionDecl* FD) {
   return FD->dump();
}
void TCling__DEBUG__decl_dump(void* D) {
   return ((clang::Decl*)D)->dump();
}
void TCling__DEBUG__printName(clang::Decl* D) {
   if (clang::NamedDecl* ND = llvm::dyn_cast<clang::NamedDecl>(D)) {
      std::string name;
      {
         llvm::raw_string_ostream OS(name);
         ND->getNameForDiagnostic(OS, D->getASTContext().getPrintingPolicy(),
                                  true /*Qualified*/);
      }
      printf("%s\n", name.c_str());
   }
}


using namespace std;
using namespace clang;
using namespace ROOT;

namespace {
  static const std::string gInterpreterClassDef = R"ICF(
#undef ClassDef
#define ClassDef(name, id) \
_ClassDefInterp_(name,id,virtual,) \
static int DeclFileLine() { return __LINE__; }
#undef ClassDefNV
#define ClassDefNV(name, id) \
_ClassDefInterp_(name,id,,) \
static int DeclFileLine() { return __LINE__; }
#undef ClassDefOverride
#define ClassDefOverride(name, id) \
_ClassDefInterp_(name,id,,override) \
static int DeclFileLine() { return __LINE__; }
)ICF";

  static const std::string gNonInterpreterClassDef = R"ICF(
#define __ROOTCLING__ 1
#undef ClassDef
#define ClassDef(name,id) \
_ClassDef_(name,id,virtual,) \
static int DeclFileLine() { return __LINE__; }
#undef ClassDefNV
#define ClassDefNV(name, id)\
_ClassDef_(name,id,,)\
static int DeclFileLine() { return __LINE__; }
#undef ClassDefOverride
#define ClassDefOverride(name, id)\
_ClassDef_(name,id,,override)\
static int DeclFileLine() { return __LINE__; }
)ICF";

// The macros below use ::Error, so let's ensure it is included
  static const std::string gClassDefInterpMacro = R"ICF(
#ifndef ROOT_TError
#include "TError.h"
#endif

#define _ClassDefInterp_(name,id,virtual_keyword, overrd) \
private: \
public: \
   static TClass *Class() { static TClass* sIsA = 0; if (!sIsA) sIsA = TClass::GetClass(#name); return sIsA; } \
   static const char *Class_Name() { return #name; } \
   static Version_t Class_Version() { return id; } \
   static TClass *Dictionary() { return 0; } \
   virtual_keyword TClass *IsA() const overrd { return name::Class(); } \
   virtual_keyword void ShowMembers(TMemberInspector&insp) const overrd { ::ROOT::Class_ShowMembers(name::Class(), this, insp); } \
   virtual_keyword void Streamer(TBuffer&) overrd { ::Error("Streamer", "Cannot stream interpreted class."); } \
   void StreamerNVirtual(TBuffer&ClassDef_StreamerNVirtual_b) { name::Streamer(ClassDef_StreamerNVirtual_b); } \
   static const char *DeclFileName() { return __FILE__; } \
   static int ImplFileLine() { return 0; } \
   static const char *ImplFileName() { return __FILE__; }
)ICF";
}
R__EXTERN int optind;

// The functions are used to bridge cling/clang/llvm compiled with no-rtti and
// ROOT (which uses rtti)

//______________________________________________________________________________

// Class extracting recursively every Enum type defined for a class.
class EnumVisitor : public RecursiveASTVisitor<EnumVisitor> {
private:
   llvm::SmallVector<EnumDecl*,128> &fClassEnums;
public:
   EnumVisitor(llvm::SmallVector<EnumDecl*,128> &enums) : fClassEnums(enums)
   {}

   bool TraverseStmt(Stmt*) {
      // Don't descend into function bodies.
      return true;
   }

   bool shouldVisitTemplateInstantiations() const { return true; }

   bool TraverseClassTemplateDecl(ClassTemplateDecl*) {
      // Don't descend into templates (but only instances thereof).
      return true; // returning false will abort the in-depth traversal.
   }

   bool TraverseClassTemplatePartialSpecializationDecl(ClassTemplatePartialSpecializationDecl*) {
      // Don't descend into templates partial specialization (but only instances thereof).
      return true; // returning false will abort the in-depth traversal.
   }

   bool VisitEnumDecl(EnumDecl *TEnumD) {
      if (!TEnumD->getDeclContext()->isDependentContext())
         fClassEnums.push_back(TEnumD);
      return true; // returning false will abort the in-depth traversal.
   }
};

//______________________________________________________________________________
static void TCling__UpdateClassInfo(const NamedDecl* TD)
{
   // Update TClingClassInfo for a class (e.g. upon seeing a definition).
   static Bool_t entered = kFALSE;
   static vector<const NamedDecl*> updateList;
   Bool_t topLevel;

   if (entered) topLevel = kFALSE;
   else {
      entered = kTRUE;
      topLevel = kTRUE;
   }
   if (topLevel) {
      ((TCling*)gInterpreter)->UpdateClassInfoWithDecl(TD);
   } else {
      // If we are called indirectly from within another call to
      // TCling::UpdateClassInfo, we delay the update until the dictionary loading
      // is finished (i.e. when we return to the top level TCling::UpdateClassInfo).
      // This allows for the dictionary to be fully populated when we actually
      // update the TClass object.   The updating of the TClass sometimes
      // (STL containers and when there is an emulated class) forces the building
      // of the TClass object's real data (which needs the dictionary info).
      updateList.push_back(TD);
   }
   if (topLevel) {
      while (!updateList.empty()) {
         ((TCling*)gInterpreter)->UpdateClassInfoWithDecl(updateList.back());
         updateList.pop_back();
      }
      entered = kFALSE;
   }
}

void TCling::UpdateEnumConstants(TEnum* enumObj, TClass* cl) const {
   const clang::Decl* D = static_cast<const clang::Decl*>(enumObj->GetDeclId());
   if(const clang::EnumDecl* ED = dyn_cast<clang::EnumDecl>(D)) {
      // Add the constants to the enum type.
      for (EnumDecl::enumerator_iterator EDI = ED->enumerator_begin(),
                EDE = ED->enumerator_end(); EDI != EDE; ++EDI) {
         // Get name of the enum type.
         std::string constbuf;
         if (const NamedDecl* END = llvm::dyn_cast<NamedDecl>(*EDI)) {
            PrintingPolicy Policy((*EDI)->getASTContext().getPrintingPolicy());
            llvm::raw_string_ostream stream(constbuf);
            (END)->getNameForDiagnostic(stream, Policy, /*Qualified=*/false);
         }
         const char* constantName = constbuf.c_str();

         // Get value of the constant.
         Long64_t value;
         const llvm::APSInt valAPSInt = (*EDI)->getInitVal();
         if (valAPSInt.isSigned()) {
            value = valAPSInt.getSExtValue();
         } else {
            value = valAPSInt.getZExtValue();
         }

         // Create the TEnumConstant or update it if existing
         TEnumConstant* enumConstant = nullptr;
         TClingClassInfo* tcCInfo = (TClingClassInfo*)(cl ? cl->GetClassInfo() : 0);
         TClingDataMemberInfo* tcDmInfo = new TClingDataMemberInfo(fInterpreter, *EDI, tcCInfo);
         DataMemberInfo_t* dmInfo = (DataMemberInfo_t*) tcDmInfo;
         if (TObject* encAsTObj = enumObj->GetConstants()->FindObject(constantName)){
            ((TEnumConstant*)encAsTObj)->Update(dmInfo);
         } else {
            enumConstant = new TEnumConstant(dmInfo, constantName, value, enumObj);
         }

         // Add the global constants to the list of Globals.
         if (!cl) {
            TCollection* globals = gROOT->GetListOfGlobals(false);
            if (!globals->FindObject(constantName)) {
               globals->Add(enumConstant);
            }
         }
      }
   }
}

TEnum* TCling::CreateEnum(void *VD, TClass *cl) const
{
   // Handle new enum declaration for either global and nested enums.

   // Create the enum type.
   TEnum* enumType = 0;
   const clang::Decl* D = static_cast<const clang::Decl*>(VD);
   std::string buf;
   if (const EnumDecl* ED = llvm::dyn_cast<EnumDecl>(D)) {
      // Get name of the enum type.
      PrintingPolicy Policy(ED->getASTContext().getPrintingPolicy());
      llvm::raw_string_ostream stream(buf);
      ED->getNameForDiagnostic(stream, Policy, /*Qualified=*/false);
      // If the enum is unnamed we do not add it to the list of enums i.e unusable.
   }
   if (buf.empty()) {
      return 0;
   }
   const char* name = buf.c_str();
   if (cl) {
      enumType = new TEnum(name, VD, cl);
   } else {
      enumType = new TEnum(name, VD, cl);
   }
   UpdateEnumConstants(enumType, cl);

   return enumType;
}

void TCling::HandleNewDecl(const void* DV, bool isDeserialized, std::set<TClass*> &modifiedTClasses) {
   // Handle new declaration.
   // Record the modified class, struct and namespaces in 'modifiedTClasses'.

   const clang::Decl* D = static_cast<const clang::Decl*>(DV);

   if (!D->isCanonicalDecl() && !isa<clang::NamespaceDecl>(D)
       && !dyn_cast<clang::RecordDecl>(D)) return;

   if (isa<clang::FunctionDecl>(D->getDeclContext())
       || isa<clang::TagDecl>(D->getDeclContext()))
      return;

   // Don't list templates.
   if (const clang::CXXRecordDecl* RD = dyn_cast<clang::CXXRecordDecl>(D)) {
      if (RD->getDescribedClassTemplate())
         return;
   } else if (const clang::FunctionDecl* FD = dyn_cast<clang::FunctionDecl>(D)) {
      if (FD->getDescribedFunctionTemplate())
         return;
   }

   if (const RecordDecl *TD = dyn_cast<RecordDecl>(D)) {
      if (TD->isCanonicalDecl() || TD->isThisDeclarationADefinition())
         TCling__UpdateClassInfo(TD);
   }
   else if (const NamedDecl *ND = dyn_cast<NamedDecl>(D)) {

      if (const TagDecl *TD = dyn_cast<TagDecl>(D)) {
         // Mostly just for EnumDecl (the other TagDecl are handled
         // by the 'RecordDecl' if statement.
         TCling__UpdateClassInfo(TD);
      } else if (const NamespaceDecl* NSD = dyn_cast<NamespaceDecl>(D)) {
         TCling__UpdateClassInfo(NSD);
      }

      // While classes are read completely (except for function template instances,
      // enum, data (and functions) can be added to namespaces at any time.
      // [this section can be removed when both the list of data member and the list
      //  of enums has been update to be active list]
      if (const NamespaceDecl* NCtx = dyn_cast<NamespaceDecl>(ND->getDeclContext())) {
         if (NCtx->getIdentifier()) {
            // No need to load the TClass: if there is something to update then
            // it must already exist.
            std::string NCtxName;
            PrintingPolicy Policy(NCtx->getASTContext().getPrintingPolicy());
            llvm::raw_string_ostream stream(NCtxName);
            NCtx->getNameForDiagnostic(stream, Policy, /*Qualified=*/true);

            TClass* cl = (TClass*)gROOT->GetListOfClasses()->FindObject(NCtxName.c_str());
            if (cl) {
               modifiedTClasses.insert(cl);
            }
         }
         return;
      }

      // We care about declarations on the global scope.
      if (!isa<TranslationUnitDecl>(ND->getDeclContext()))
         return;

      // Enums are lazyly created, thus we don not need to handle them here.
      if (isa<EnumDecl>(ND))
         return;

      // ROOT says that global is enum(lazylycreated)/var/field declared on the global
      // scope.
      if (!(isa<VarDecl>(ND)))
         return;

      // Skip if already in the list.
      if (gROOT->GetListOfGlobals()->FindObject(ND->getNameAsString().c_str()))
         return;

      // Put the global constants and global enums in the coresponding lists.
      gROOT->GetListOfGlobals()->Add(new TGlobal((DataMemberInfo_t *)
                                                new TClingDataMemberInfo(fInterpreter,
                                                                        cast<ValueDecl>(ND), 0)));
   }
}

extern "C"
void TCling__GetNormalizedContext(const ROOT::TMetaUtils::TNormalizedCtxt*& normCtxt)
{
   // We are sure in this context of the type of the interpreter
   normCtxt = &( (TCling*) gInterpreter)->GetNormalizedContext();
}

extern "C"
void TCling__UpdateListsOnCommitted(const cling::Transaction &T, cling::Interpreter*) {

   ((TCling*)gCling)->UpdateListsOnCommitted(T);
}

extern "C"
void TCling__UpdateListsOnUnloaded(const cling::Transaction &T) {

   ((TCling*)gCling)->UpdateListsOnUnloaded(T);
}
void TCling__TransactionRollback(const cling::Transaction &T) {

   ((TCling*)gCling)->TransactionRollback(T);
}
extern "C" void TCling__LibraryLoadedRTTI(const void* dyLibHandle,
                                          const char* canonicalName) {

   ((TCling*)gCling)->LibraryLoaded(dyLibHandle, canonicalName);
}

extern "C" void TCling__LibraryUnloadedRTTI(const void* dyLibHandle,
                                            const char* canonicalName) {

   ((TCling*)gCling)->LibraryUnloaded(dyLibHandle, canonicalName);
}


extern "C"
TObject* TCling__GetObjectAddress(const char *Name, void *&LookupCtx) {
   // The call to FindSpecialObject might induces any kind of use
   // of the interpreter ... (library loading, function calling, etc.)
   // ... and we _know_ we are in the middle of parsing, so let's make
   // sure to save the state and then restore it.

   cling::Interpreter *interpreter = ((TCling*)gCling)->GetInterpreter();

   // Save state of the PP
   Sema &SemaR = interpreter->getSema();
   ASTContext& C = SemaR.getASTContext();
   Preprocessor &PP = SemaR.getPreprocessor();
   Parser& P = const_cast<Parser&>(interpreter->getParser());
   Preprocessor::CleanupAndRestoreCacheRAII cleanupRAII(PP);
   Parser::ParserCurTokRestoreRAII savedCurToken(P);
   // After we have saved the token reset the current one to something which
   // is safe (semi colon usually means empty decl)
   Token& Tok = const_cast<Token&>(P.getCurToken());
   Tok.setKind(tok::semi);

   // We can't PushDeclContext, because we go up and the routine that pops
   // the DeclContext assumes that we drill down always.
   // We have to be on the global context. At that point we are in a
   // wrapper function so the parent context must be the global.
   Sema::ContextAndScopeRAII pushedDCAndS(SemaR, C.getTranslationUnitDecl(),
                                          SemaR.TUScope);

   return gROOT->FindSpecialObject(Name, LookupCtx);
}

extern "C" const Decl* TCling__GetObjectDecl(TObject *obj) {
   return ((TClingClassInfo*)obj->IsA()->GetClassInfo())->GetDecl();
}

extern "C" R__DLLEXPORT TInterpreter *CreateInterpreter(void* interpLibHandle)
{
   cling::DynamicLibraryManager::ExposeHiddenSharedLibrarySymbols(interpLibHandle);
   return new TCling("C++", "cling C++ Interpreter");
}

extern "C" R__DLLEXPORT void DestroyInterpreter(TInterpreter *interp)
{
   delete interp;
}

// Load library containing specified class. Returns 0 in case of error
// and 1 in case if success.
extern "C" int TCling__AutoLoadCallback(const char* className)
{
   return ((TCling*)gCling)->AutoLoad(className);
}

extern "C" int TCling__AutoParseCallback(const char* className)
{
   return ((TCling*)gCling)->AutoParse(className);
}

extern "C" const char* TCling__GetClassSharedLibs(const char* className)
{
   return ((TCling*)gCling)->GetClassSharedLibs(className);
}

// // Returns 0 for failure 1 for success
// extern "C" int TCling__IsAutoLoadNamespaceCandidate(const char* name)
// {
//    return ((TCling*)gCling)->IsAutoLoadNamespaceCandidate(name);
// }

// Returns 0 for failure 1 for success
extern "C" int TCling__IsAutoLoadNamespaceCandidate(const clang::NamespaceDecl* nsDecl)
{
   return ((TCling*)gCling)->IsAutoLoadNamespaceCandidate(nsDecl);
}

extern "C" int TCling__CompileMacro(const char *fileName, const char *options)
{
   string file(fileName);
   string opt(options);
   return gSystem->CompileMacro(file.c_str(), opt.c_str());
}

extern "C" void TCling__SplitAclicMode(const char* fileName, string &mode,
                                       string &args, string &io, string &fname)
{
   string file(fileName);
   TString f, amode, arguments, aclicio;
   f = gSystem->SplitAclicMode(file.c_str(), amode, arguments, aclicio);
   mode = amode.Data(); args = arguments.Data();
   io = aclicio.Data(); fname = f.Data();
}

//______________________________________________________________________________
//
//
//

#ifdef R__WIN32
extern "C" {
   char *__unDName(char *demangled, const char *mangled, int out_len,
                   void * (* pAlloc )(size_t), void (* pFree )(void *),
                   unsigned short int flags);
}
#endif

//______________________________________________________________________________
static clang::ClassTemplateDecl* FindTemplateInNamespace(clang::Decl* decl)
{
   // Find a template decl within N nested namespaces, 0<=N<inf
   // Assumes 1 and only 1 template present and 1 and only 1 entity contained
   // by the namespace. Example: ns1::ns2::..::nsN::myTemplate
   // Returns nullptr in case of error
   using namespace clang;
   if (NamespaceDecl* nsd = llvm::dyn_cast<NamespaceDecl>(decl)){
      return FindTemplateInNamespace(*nsd->decls_begin());
   }

   if (ClassTemplateDecl* ctd = llvm::dyn_cast<ClassTemplateDecl>(decl)){
      return ctd;
   }

   return nullptr; // something went wrong.
}

//______________________________________________________________________________
void* llvmLazyFunctionCreator(const std::string& mangled_name)
{
   // Autoload a library provided the mangled name of a missing symbol.
   return ((TCling*)gCling)->LazyFunctionCreatorAutoload(mangled_name);
}

//______________________________________________________________________________
//
//
//

int TCling_GenerateDictionary(const std::vector<std::string> &classes,
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
      static const std::set<std::string> sSTLTypes {"vector","list","unordered_list","deque","map","multimap","set","unordered_set","multiset","queue","priority_queue","stack","iterator"};
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

int TCling_GenerateDictionary(const std::string& className,
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
   return TCling_GenerateDictionary(classes, headers, fwdDecls, unknown);
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

void* TCling::fgSetOfSpecials = 0;

//______________________________________________________________________________
//
// llvm error handler through exceptions; see also cling/UserInterface
//
namespace {
   // Handle fatal llvm errors by throwing an exception.
   // Yes, throwing exceptions in error handlers is bad.
   // Doing nothing is pretty terrible, too.
   void exceptionErrorHandler(void * /*user_data*/,
                              const std::string& reason,
                              bool /*gen_crash_diag*/) {
      throw std::runtime_error(std::string(">>> Interpreter compilation error:\n") + reason);
   }
}

//______________________________________________________________________________
//
//
//

//______________________________________________________________________________
namespace{

   // An instance of this class causes the diagnostics of clang to be suppressed
   // during its lifetime
   class clangDiagSuppr {
   public:
      clangDiagSuppr(clang::DiagnosticsEngine& diag): fDiagEngine(diag){
         fOldDiagValue = fDiagEngine.getIgnoreAllWarnings();
         fDiagEngine.setIgnoreAllWarnings(true);
      }

      ~clangDiagSuppr() {
         fDiagEngine.setIgnoreAllWarnings(fOldDiagValue);
      }
   private:
      clang::DiagnosticsEngine& fDiagEngine;
      bool fOldDiagValue;
   };

}

////////////////////////////////////////////////////////////////////////////////
/// Allow calling autoparsing from TMetaUtils
bool TClingLookupHelper__AutoParse(const char *cname)
{
   return gCling->AutoParse(cname);
}

//______________________________________________________________________________
bool TClingLookupHelper__ExistingTypeCheck(const std::string &tname,
                                           std::string &result)
{
   // Try hard to avoid looking up in the Cling database as this could enduce
   // an unwanted autoparsing.
   result.clear();

   unsigned long offset = 0;
   if (strncmp(tname.c_str(), "const ", 6) == 0) {
      offset = 6;
   }
   unsigned long end = tname.length();
   while( end && (tname[end-1]=='&' || tname[end-1]=='*' || tname[end-1]==']') ) {
      if ( tname[end-1]==']' ) {
         --end;
         while ( end && tname[end-1]!='[' ) --end;
      }
      --end;
   }
   std::string innerbuf;
   const char *inner;
   if (end != tname.length()) {
      innerbuf = tname.substr(offset,end-offset);
      inner = innerbuf.c_str();
   } else {
      inner = tname.c_str()+offset;
   }

   //if (strchr(tname.c_str(),'[')!=0) fprintf(stderr,"DEBUG: checking on %s vs %s %lu %lu\n",tname.c_str(),inner,offset,end);
   if (gROOT->GetListOfClasses()->FindObject(inner)
       || TClassTable::Check(inner,result) ) {
      // This is a known class.
      return true;
   }

   THashTable *typeTable = dynamic_cast<THashTable*>( gROOT->GetListOfTypes() );
   TDataType *type = (TDataType *)typeTable->THashTable::FindObject( inner );
   if (type) {
      // This is a raw type and an already loaded typedef.
      const char *newname = type->GetFullTypeName();
      if (type->GetType() == kLong64_t) {
         newname = "Long64_t";
      } else if (type->GetType() == kULong64_t) {
         newname = "ULong64_t";
      }
      if (strcmp(inner,newname) == 0) {
         return true;
      }
      if (offset) result = "const ";
      result += newname;
      if ( end != tname.length() ) {
         result += tname.substr(end,tname.length()-end);
      }
      if (result == tname) result.clear();
      return true;
   }

   // Check if the name is an enumerator
   const auto lastPos = TClassEdit::GetUnqualifiedName(inner);
   if (lastPos != inner)   // Main switch: case 1 - scoped enum, case 2 global enum
   {
      // We have a scope
      // All of this C gymnastic is to avoid allocations on the heap
      const auto enName = lastPos;
      const auto scopeNameSize = ((Long64_t)lastPos - (Long64_t)inner) / sizeof(decltype(*lastPos)) - 2;
#ifdef R__WIN32
      char *scopeName = new char[scopeNameSize + 1];
#else
      char scopeName[scopeNameSize + 1]; // on the stack, +1 for the terminating character '\0'
#endif
      strncpy(scopeName, inner, scopeNameSize);
      scopeName[scopeNameSize] = '\0';
      // Check if the scope is in the list of classes
      if (auto scope = static_cast<TClass *>(gROOT->GetListOfClasses()->FindObject(scopeName))) {
         auto enumTable = dynamic_cast<const THashList *>(scope->GetListOfEnums(false));
         if (enumTable && enumTable->THashList::FindObject(enName)) return true;
      }
      // It may still be in one of the loaded protoclasses
      else if (auto scope = static_cast<TProtoClass *>(gClassTable->GetProtoNorm(scopeName))) {
         auto listOfEnums = scope->GetListOfEnums();
         if (listOfEnums) { // it could be null: no enumerators in the protoclass
            auto enumTable = dynamic_cast<const THashList *>(listOfEnums);
            if (enumTable && enumTable->THashList::FindObject(enName)) return true;
         }
      }
#ifdef R__WIN32
      delete [] scopeName;
#endif
   } else
   {
      // We don't have any scope: this could only be a global enum
      auto enumTable = dynamic_cast<const THashList *>(gROOT->GetListOfEnums());
      if (enumTable && enumTable->THashList::FindObject(inner)) return true;
   }

   if (gCling->GetClassSharedLibs(inner))
   {
      // This is a class name.
      return true;
   }

   return false;
}

//______________________________________________________________________________
TCling::TUniqueString::TUniqueString(Long64_t size)
{
   fContent.reserve(size);
}

//______________________________________________________________________________
inline const char *TCling::TUniqueString::Data()
{
   return fContent.c_str();
}

//______________________________________________________________________________
inline bool TCling::TUniqueString::Append(const std::string& str)
{
   // Append string to the storage if not added already.
   bool notPresent = fLinesHashSet.emplace(fHashFunc(str)).second;
   if (notPresent){
      fContent+=str;
   }
   return notPresent;
}

//______________________________________________________________________________
TCling::TCling(const char *name, const char *title)
: TInterpreter(name, title), fGlobalsListSerial(-1), fInterpreter(0),
   fMetaProcessor(0), fNormalizedCtxt(0), fPrevLoadedDynLibInfo(0),
   fClingCallbacks(0), fAutoLoadCallBack(0),
   fTransactionCount(0), fHeaderParsingOnDemand(true), fIsAutoParsingSuspended(kFALSE)
{
   // Initialize the cling interpreter interface.

   // rootcling also uses TCling for generating the dictionary ROOT files.
   bool fromRootCling = dlsym(RTLD_DEFAULT, "usedToIdentifyRootClingByDlSym");

   llvm::install_fatal_error_handler(&exceptionErrorHandler);

   fTemporaries = new std::vector<cling::Value>();

   std::vector<std::string> clingArgsStorage;
   clingArgsStorage.push_back("cling4root");

   std::string interpInclude;
   // rootcling sets its arguments through TROOT::GetExtraInterpreterArgs().
   if (!fromRootCling) {
      ROOT::TMetaUtils::SetPathsForRelocatability(clingArgsStorage);

      interpInclude = ROOT::TMetaUtils::GetInterpreterExtraIncludePath(false);
      clingArgsStorage.push_back(interpInclude);

      std::string pchFilename = interpInclude.substr(2) + "/allDict.cxx.pch";
      if (gSystem->Getenv("ROOT_PCH")) {
         pchFilename = gSystem->Getenv("ROOT_PCH");
      }
      clingArgsStorage.push_back("-include-pch");
      clingArgsStorage.push_back(pchFilename);

      // clingArgsStorage.push_back("-Xclang");
      // clingArgsStorage.push_back("-fmodules");

      std::string include;
#ifndef ROOTINCDIR
      include = gSystem->Getenv("ROOTSYS");
      include += "/include";
#else // ROOTINCDIR
      include = ROOTINCDIR;
#endif // ROOTINCDIR
      clingArgsStorage.push_back("-I");
      clingArgsStorage.push_back(include);
      clingArgsStorage.push_back("-Wno-undefined-inline");
      clingArgsStorage.push_back("-fsigned-char");
   }

   std::vector<const char*> interpArgs;
   for (std::vector<std::string>::const_iterator iArg = clingArgsStorage.begin(),
           eArg = clingArgsStorage.end(); iArg != eArg; ++iArg)
      interpArgs.push_back(iArg->c_str());

   std::string llvmResourceDir = ROOT::TMetaUtils::GetLLVMResourceDir(false);
   // Add statically injected extra arguments, usually coming from rootcling.
   for (const char** extraArgs = TROOT::GetExtraInterpreterArgs();
        extraArgs && *extraArgs; ++extraArgs) {
      if (!strcmp(*extraArgs, "-resource-dir")) {
         // Take the next arg as the llvm resource directory.
         llvmResourceDir = *(++extraArgs);
      } else {
         interpArgs.push_back(*extraArgs);
      }
   }

   fInterpreter = new cling::Interpreter(interpArgs.size(),
                                         &(interpArgs[0]),
                                         llvmResourceDir.c_str());

   if (!fromRootCling) {
      fInterpreter->installLazyFunctionCreator(llvmLazyFunctionCreator);

      // Add include path to etc/cling. FIXME: This is a short term solution. The
      // llvm/clang header files shouldn't be there at all. We have to get rid of
      // that dependency and avoid copying the header files.
      // Use explicit TCling::AddIncludePath() to avoid vtable: we're in the c'tor!
      TCling::AddIncludePath((interpInclude.substr(2) + "/cling").c_str());

      // Add the current path to the include path
      TCling::AddIncludePath(".");

      // Add the root include directory and etc/ to list searched by default.
      TCling::AddIncludePath(ROOT::TMetaUtils::GetROOTIncludeDir(false).c_str());
   }

   // Don't check whether modules' files exist.
   fInterpreter->getCI()->getPreprocessorOpts().DisablePCHValidation = true;

   // Until we can disable autoloading during Sema::CorrectTypo() we have
   // to disable spell checking.
   fInterpreter->getCI()->getLangOpts().SpellChecking = false;


   // We need stream that doesn't close its file descriptor, thus we are not
   // using llvm::outs. Keeping file descriptor open we will be able to use
   // the results in pipes (Savannah #99234).
   static llvm::raw_fd_ostream fMPOuts (STDOUT_FILENO, /*ShouldClose*/false);
   fMetaProcessor = new cling::MetaProcessor(*fInterpreter, fMPOuts);

   // For the list to also include string, we have to include it now.
   // rootcling does parts already if needed, e.g. genreflex does not want using
   // namespace std.
   if (fromRootCling) {
      fInterpreter->declare("#include \"RtypesCore.h\"\n"
                            "#include <string>\n"
                            "using std::string;");
   } else {
      fInterpreter->declare("#include \"Rtypes.h\"\n"
                            + gClassDefInterpMacro + "\n"
                            + gInterpreterClassDef + "\n"
                            + "#undef ClassImp\n"
                            "#define ClassImp(X)\n"
                            "#include <string>\n"
                            "using namespace std;");
   }

   // We are now ready (enough is loaded) to init the list of opaque typedefs.
   fNormalizedCtxt = new ROOT::TMetaUtils::TNormalizedCtxt(fInterpreter->getLookupHelper());
   fLookupHelper = new ROOT::TMetaUtils::TClingLookupHelper(*fInterpreter, *fNormalizedCtxt, TClingLookupHelper__ExistingTypeCheck, TClingLookupHelper__AutoParse);
   TClassEdit::Init(fLookupHelper);

   // Initialize the cling interpreter interface.
   fMore      = 0;
   fPrompt[0] = 0;
   fMapfile   = 0;
//    fMapNamespaces   = 0;
   fRootmapFiles = 0;
   fLockProcessLine = kTRUE;

   fAllowLibLoad = !fromRootCling;
   // Disallow auto-parsing in rootcling
   fIsAutoParsingSuspended = fromRootCling;
   // Disable the autoloader until it is explicitly enabled.
   SetClassAutoloading(false);

   ResetAll();
#ifndef R__WIN32
   //optind = 1;  // make sure getopt() works in the main program
#endif // R__WIN32

   // Attach cling callbacks
   std::unique_ptr<TClingCallbacks>
      clingCallbacks(new TClingCallbacks(fInterpreter));
   fClingCallbacks = clingCallbacks.get();
   fClingCallbacks->SetAutoParsingSuspended(fIsAutoParsingSuspended);
   fInterpreter->setCallbacks(std::move(clingCallbacks));

   if (!fromRootCling) {
      fInterpreter->enableDynamicLookup();
   }
}


//______________________________________________________________________________
TCling::~TCling()
{
   // Destroy the interpreter interface.
   delete fMapfile;
//    delete fMapNamespaces;
   delete fRootmapFiles;
   delete fMetaProcessor;
   delete fTemporaries;
   delete fNormalizedCtxt;
   delete fInterpreter;
   delete fLookupHelper;
   gCling = 0;
#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   Warning("~TCling", "Interface not available yet.");
#ifdef R__COMPLETE_MEM_TERMINATION
   // remove all cling objects
#endif
#endif
#endif
   //--
}

//______________________________________________________________________________
void TCling::Initialize()
{
   // Initialize the interpreter, once TROOT::fInterpreter is set.
   fClingCallbacks->Initialize();
}

//______________________________________________________________________________
static const char *FindLibraryName(void (*func)())
{
   // Wrapper around dladdr (and friends)

#if defined(__CYGWIN__) && defined(__GNUC__)
   return 0;
#elif defined(G__WIN32)
   MEMORY_BASIC_INFORMATION mbi;
   if (!VirtualQuery (func, &mbi, sizeof (mbi)))
   {
      return 0;
   }

   HMODULE hMod = (HMODULE) mbi.AllocationBase;
   TTHREAD_TLS_ARRAY(char, MAX_PATH, moduleName);

   if (!GetModuleFileNameA (hMod, moduleName, sizeof (moduleName)))
   {
      return 0;
   }
   return moduleName;
#else
   Dl_info info;
   if (dladdr((void*)func,&info)==0) {
      // Not in a known share library, let's give up
      return 0;
   } else {
      //fprintf(stdout,"Found address in %s\n",info.dli_fname);
      return info.dli_fname;
   }
#endif

}

//______________________________________________________________________________
bool TCling::LoadPCM(TString pcmFileName,
                     const char** headers,
                     void (*triggerFunc)()) const {
   // Tries to load a PCM; returns true on success.

   // pcmFileName is an intentional copy; updated by FindFile() below.

   TString searchPath;

   if (triggerFunc) {
      const char *libraryName = FindLibraryName(triggerFunc);
      if (libraryName) {
         searchPath = llvm::sys::path::parent_path(libraryName);
#ifdef R__WIN32
         searchPath += ";";
#else
         searchPath += ":";
#endif
      }
   }
   // Note: if we know where the library is, we probably shouldn't even
   // look in other places.
   searchPath.Append( gSystem->GetDynamicPath() );

   if (!gSystem->FindFile(searchPath, pcmFileName))
      return kFALSE;

   // Prevent the ROOT-PCMs hitting this during auto-load during
   // JITting - which will cause recursive compilation.
   // Avoid to call the plugin manager at all.
   TVirtualStreamerInfo::SetFactory(new TStreamerInfo());

   if (gROOT->IsRootFile(pcmFileName)) {
      Int_t oldDebug = gDebug;
      if (gDebug > 5) {
         gDebug -= 5;
         ::Info("TCling::LoadPCM", "Loading ROOT PCM %s", pcmFileName.Data());
      } else {
         gDebug = 0;
      }

      TDirectory::TContext ctxt;

      TFile *pcmFile = new TFile(pcmFileName+"?filetype=pcm","READ");

      auto listOfKeys = pcmFile->GetListOfKeys();

      // This is an empty pcm
      if (listOfKeys && listOfKeys->GetSize() == 0) {
         delete pcmFile;
         gDebug = oldDebug;
         return kTRUE;
      }

      TObjArray *protoClasses;
      if (gDebug > 1)
            ::Info("TCling::LoadPCM","reading protoclasses for %s \n",pcmFileName.Data());

      pcmFile->GetObject("__ProtoClasses", protoClasses);

      if (protoClasses) {
         for (auto obj : *protoClasses) {
            TProtoClass * proto = (TProtoClass*)obj;
            TClassTable::Add(proto);
         }
         // Now that all TClass-es know how to set them up we can update
         // existing TClasses, which might cause the creation of e.g. TBaseClass
         // objects which in turn requires the creation of TClasses, that could
         // come from the PCH, but maybe later in the loop. Instead of resolving
         // a dependency graph the addition to the TClassTable above allows us
         // to create these dependent TClasses as needed below.
         for (auto proto : *protoClasses) {
            if (TClass* existingCl
                = (TClass*)gROOT->GetListOfClasses()->FindObject(proto->GetName())) {
               // We have an existing TClass object. It might be emulated
               // or interpreted; we now have more information available.
               // Make that available.
               if (existingCl->GetState() != TClass::kHasTClassInit) {
                  DictFuncPtr_t dict = gClassTable->GetDict(proto->GetName());
                  if (!dict) {
                     ::Error("TCling::LoadPCM", "Inconsistent TClassTable for %s",
                             proto->GetName());
                  } else {
                     // This will replace the existing TClass.
                     TClass *ncl = (*dict)();
                     if (ncl) ncl->PostLoadCheck();

                  }
               }
            }
         }

         protoClasses->Clear(); // Ownership was transfered to TClassTable.
         delete protoClasses;
      }

      TObjArray *dataTypes;
      pcmFile->GetObject("__Typedefs", dataTypes);
      if (dataTypes) {
         for (auto typedf: *dataTypes)
            gROOT->GetListOfTypes()->Add(typedf);
         dataTypes->Clear(); // Ownership was transfered to TListOfTypes.
         delete dataTypes;
      }

      TObjArray *enums;
      pcmFile->GetObject("__Enums", enums);
      if (enums) {
         // Cache the pointers
         auto listOfGlobals = gROOT->GetListOfGlobals();
         auto listOfEnums = dynamic_cast<THashList*>(gROOT->GetListOfEnums());
         // Loop on enums and then on enum constants
         for (auto selEnum: *enums){
            const char* enumScope = selEnum->GetTitle();
            const char* enumName = selEnum->GetName();
            if (strcmp(enumScope,"") == 0){
               // This is a global enum and is added to the
               // list of enums and its constants to the list of globals
               if (!listOfEnums->THashList::FindObject(enumName)){
                  ((TEnum*) selEnum)->SetClass(nullptr);
                  listOfEnums->Add(selEnum);
               }
               for (auto enumConstant: *static_cast<TEnum*>(selEnum)->GetConstants()){
                  if (!listOfGlobals->FindObject(enumConstant)){
                     listOfGlobals->Add(enumConstant);
                  }
               }
            }
            else {
               // This enum is in a namespace. A TClass entry is bootstrapped if
               // none exists yet and the enum is added to it
               TClass* nsTClassEntry = TClass::GetClass(enumScope);
               if (!nsTClassEntry){
                  nsTClassEntry = new TClass(enumScope,0,TClass::kNamespaceForMeta, true);
               }
               auto listOfEnums = nsTClassEntry->fEnums.load();
               if (!listOfEnums) {
                  if ( (kIsClass | kIsStruct | kIsUnion) & nsTClassEntry->Property() ) {
                     // For this case, the list will be immutable once constructed
                     // (i.e. in this case, by the end of this routine).
                     listOfEnums = nsTClassEntry->fEnums = new TListOfEnums(nsTClassEntry);
                  } else {
                     //namespaces can have enums added to them
                     listOfEnums = nsTClassEntry->fEnums = new TListOfEnumsWithLock(nsTClassEntry);
                  }
               }
               if (listOfEnums && !listOfEnums->THashList::FindObject(enumName)){
                  ((TEnum*) selEnum)->SetClass(nsTClassEntry);
                  listOfEnums->Add(selEnum);
               }
            }
         }
         enums->Clear();
         delete enums;
      }

      delete pcmFile;

      gDebug = oldDebug;
   } else {
      if (gDebug > 5)
         ::Info("TCling::LoadPCM", "Loading clang PCM %s", pcmFileName.Data());

   }
   // Note: Declaring the relationship between the module (pcm) and the header
   // probably does not yet make sense since the pcm is 'only' a root file.
   // We also have to review if we still need to do this with the delay loading.
   // clang::CompilerInstance* CI = fInterpreter->getCI();
   // ROOT::TMetaUtils::declareModuleMap(CI, pcmFileName, headers);
   return kTRUE;
}

//______________________________________________________________________________

namespace {
   using namespace clang;

   class ExtLexicalStorageAdder: public RecursiveASTVisitor<ExtLexicalStorageAdder>{
      // This class is to be considered an helper for autoparsing.
      // It visits the AST and marks all classes (in all of their redeclarations)
      // with the setHasExternalLexicalStorage method.
   public:
      bool VisitRecordDecl(clang::RecordDecl* rcd){
         if (gDebug > 2)
            Info("ExtLexicalStorageAdder",
                 "Adding external lexical storage to class %s",
                 rcd->getNameAsString().c_str());
         auto reDeclPtr = rcd->getMostRecentDecl();
         do {
            reDeclPtr->setHasExternalLexicalStorage();
         } while ((reDeclPtr = reDeclPtr->getPreviousDecl()));

         return false;
      }
   };


}

//______________________________________________________________________________
void TCling::RegisterModule(const char* modulename,
                            const char** headers,
                            const char** includePaths,
                            const char* payloadCode,
                            const char* fwdDeclsCode,
                            void (*triggerFunc)(),
                            const FwdDeclArgsToKeepCollection_t& fwdDeclsArgToSkip,
                            const char** classesHeaders)
{
   // Inject the module named "modulename" into cling; load all headers.
   // headers is a 0-terminated array of header files to #include after
   // loading the module. The module is searched for in all $LD_LIBRARY_PATH
   // entries (or %PATH% on Windows).
   // This function gets called by the static initialization of dictionary
   // libraries.
   // The payload code is injected "as is" in the interpreter.
   // The value of 'triggerFunc' is used to find the shared library location.

   // rootcling also uses TCling for generating the dictionary ROOT files.
   static const bool fromRootCling = dlsym(RTLD_DEFAULT, "usedToIdentifyRootClingByDlSym");
   // We need the dictionary initialization but we don't want to inject the
   // declarations into the interpreter, except for those we really need for
   // I/O; see rootcling.cxx after the call to TCling__GetInterpreter().
   if (fromRootCling) return;

   // Treat Aclic Libs in a special way. Do not delay the parsing.
   bool hasHeaderParsingOnDemand = fHeaderParsingOnDemand;
   bool isACLiC = false;
   if (hasHeaderParsingOnDemand &&
       strstr(modulename, "_ACLiC_dict") != nullptr){
      if (gDebug>1)
         Info("TCling::RegisterModule",
              "Header parsing on demand is active but this is an Aclic library. Disabling it for this library.");
      hasHeaderParsingOnDemand = false;
      isACLiC = true;
   }


   // Make sure we relookup symbols that were search for before we loaded
   // their autoparse information.  We could be more subtil and remove only
   // the failed one or only the one in this module, but for now this is
   // better than nothing.
   fLookedUpClasses.clear();

   // Make sure we do not set off autoloading or autoparsing during the
   // module registration!
   Int_t oldAutoloadValue = SetClassAutoloading(false);

   TString pcmFileName(ROOT::TMetaUtils::GetModuleFileName(modulename).c_str());

   for (const char** inclPath = includePaths; *inclPath; ++inclPath) {
      TCling::AddIncludePath(*inclPath);
   }
   cling::Transaction* T = 0;
   // Put the template decls and the number of arguments to skip in the TNormalizedCtxt
   for (auto& fwdDeclArgToSkipPair : fwdDeclsArgToSkip){
      const std::string& fwdDecl = fwdDeclArgToSkipPair.first;
      const int nArgsToSkip = fwdDeclArgToSkipPair.second;
      auto compRes = fInterpreter->declare(fwdDecl.c_str(), &T);
      assert(cling::Interpreter::kSuccess == compRes &&
            "A fwd declaration could not be compiled");
      if (compRes!=cling::Interpreter::kSuccess){
         Warning("TCling::RegisterModule",
               "Problems in declaring string '%s' were encountered.",
               fwdDecl.c_str()) ;
         continue;
      }

      // Drill through namespaces recursively until the template is found
      if(ClassTemplateDecl* TD = FindTemplateInNamespace(T->getFirstDecl().getSingleDecl())){
         fNormalizedCtxt->AddTemplAndNargsToKeep(TD->getCanonicalDecl(), nArgsToSkip);
      }

   }

   // FIXME: Remove #define __ROOTCLING__ once PCMs are there.
   // This is used to give Sema the same view on ACLiC'ed files (which
   // are then #included through the dictionary) as rootcling had.
   TString code = fromRootCling ? "" : gNonInterpreterClassDef ;
   code += payloadCode;

   // We need to open the dictionary shared library, to resolve sylbols
   // requested by the JIT from it: as the library is currently being dlopen'ed,
   // its symbols are not yet reachable from the process.
   // Recursive dlopen seems to work just fine.
   const char* dyLibName = FindLibraryName(triggerFunc);
   if (dyLibName) {
      // We were able to determine the library name.
      void* dyLibHandle = dlopen(dyLibName, RTLD_LAZY | RTLD_GLOBAL);
#ifdef R__WIN32
      if (!dyLibHandle) {
         char dyLibError[1000];
         FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, NULL, GetLastError(),
                       MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), dyLibError,
                       sizeof(dyLibError), NULL);
#else
      const char* dyLibError = dlerror();
      if (dyLibError) {
#endif
         if (gDebug > 0) {
            ::Info("TCling::RegisterModule",
                   "Cannot open shared library %s for dictionary %s:\n  %s",
                   dyLibName, modulename, dyLibError);
         }
         dyLibName = 0;
      } else {
         fRegisterModuleDyLibs.push_back(dyLibHandle);
      }
   }

   if (hasHeaderParsingOnDemand && fwdDeclsCode){
      // We now parse the forward declarations. All the classes are then modified
      // in order for them to have an external lexical storage.
      std::string fwdDeclsCodeLessEnums;
      {
         // Search for enum forward decls and only declare them if no
         // declaration exists yet.
         std::string fwdDeclsLine;
         std::istringstream fwdDeclsCodeStr(fwdDeclsCode);
         std::vector<std::string> scope;
         while (std::getline(fwdDeclsCodeStr, fwdDeclsLine)) {
            if (fwdDeclsLine.find("namespace ") == 0
                || fwdDeclsLine.find("inline namespace ") == 0) {
               // skip leading "namespace ", trailing " {"
               scope.push_back(fwdDeclsLine.substr(10,
                                                   fwdDeclsLine.length() - 10 - 2));
            } else if (fwdDeclsLine == "}") {
               scope.pop_back();
            } else if (fwdDeclsLine.find("enum  __attribute__((annotate(\"") == 0) {
               clang::DeclContext* DC = 0;
               for (auto &&aScope: scope) {
                  DC = cling::utils::Lookup::Namespace(&fInterpreter->getSema(), aScope.c_str(), DC);
                  if (!DC) {
                     // No decl context means we have to fwd declare the enum.
                     break;
                  }
               }
               if (scope.empty() || DC) {
                  // We know the scope; let's look for the enum.
                  size_t posEnumName = fwdDeclsLine.find("\"))) ", 32);
                  R__ASSERT(posEnumName != std::string::npos && "Inconsistent enum fwd decl!");
                  posEnumName += 5; // skip "\"))) "
                  while (isspace(fwdDeclsLine[posEnumName]))
                     ++posEnumName;
                  size_t posEnumNameEnd = fwdDeclsLine.find(" : ", posEnumName);
                  R__ASSERT(posEnumNameEnd  != std::string::npos && "Inconsistent enum fwd decl (end)!");
                  while (isspace(fwdDeclsLine[posEnumNameEnd]))
                     --posEnumNameEnd;
                  // posEnumNameEnd now points to the last character of the name.

                  std::string enumName = fwdDeclsLine.substr(posEnumName,
                                                             posEnumNameEnd - posEnumName + 1);

                  if (clang::NamedDecl* enumDecl
                      = cling::utils::Lookup::Named(&fInterpreter->getSema(),
                                                    enumName.c_str(), DC)) {
                     // We have an existing enum decl (forward or definition);
                     // skip this.
                     R__ASSERT(llvm::dyn_cast<clang::EnumDecl>(enumDecl) && "not an enum decl!");
                     (void)enumDecl;
                     continue;
                  }
               }
            }
            fwdDeclsCodeLessEnums += fwdDeclsLine + "\n";
         }
      }

      if (fwdDeclsCodeLessEnums.size() != 0){ // Avoid the overhead if nothing is to be declared
         auto compRes = fInterpreter->declare(fwdDeclsCodeLessEnums, &T);
         assert(cling::Interpreter::kSuccess == compRes &&
               "The forward declarations could not be compiled");
         if (compRes!=cling::Interpreter::kSuccess){
            Warning("TCling::RegisterModule",
                  "Problems in compiling forward declarations for module %s: '%s'",
                  modulename, fwdDeclsCodeLessEnums.c_str()) ;
         }
         else if (T){
            // Loop over all decls in the transaction and go through them all
            // to mark them properly.
            // In order to do that, we first iterate over all the DelayedCallInfos
            // within the transaction. Then we loop over all Decls in the DeclGroupRef
            // contained in the DelayedCallInfos. For each decl, we traverse.
            ExtLexicalStorageAdder elsa;
            for (auto dciIt = T->decls_begin();dciIt!=T->decls_end();dciIt++){
               cling::Transaction::DelayCallInfo& dci = *dciIt;
               for(auto dit = dci.m_DGR.begin(); dit != dci.m_DGR.end(); ++dit) {
                  clang::Decl* declPtr = *dit;
                  elsa.TraverseDecl(declPtr);
               }
            }
         }
      }

      // Now we register all the headers necessary for the class
      // Typical format of the array:
      //    {"A", "classes.h", "@",
      //     "vector<A>", "vector", "@",
      //     "myClass", payloadCode, "@",
      //    nullptr};

      std::string temp;
      for (const char** classesHeader = classesHeaders; *classesHeader; ++classesHeader) {
         temp=*classesHeader;

         size_t theTemplateHash = 0;
         bool addTemplate = false;
         size_t posTemplate = temp.find('<');
         if (posTemplate != std::string::npos) {
            // Add an entry for the template itself.
            std::string templateName = temp.substr(0, posTemplate);
            theTemplateHash = fStringHashFunction(templateName);
            addTemplate = true;
         }
         size_t theHash = fStringHashFunction(temp);
         classesHeader++;
         for (const char** classesHeader_inner = classesHeader; 0!=strcmp(*classesHeader_inner,"@"); ++classesHeader_inner,++classesHeader){
            // This is done in order to distinguish headers from files and from the payloadCode
            if (payloadCode == *classesHeader_inner ){
               fPayloads.insert(theHash);
               if (addTemplate) fPayloads.insert(theTemplateHash);
            }
            if (gDebug > 2)
               Info("TCling::RegisterModule",
                     "Adding a header for %s", temp.c_str());
            fClassesHeadersMap[theHash].push_back(*classesHeader_inner);
            if (addTemplate) {
               if (fClassesHeadersMap.find(theTemplateHash) == fClassesHeadersMap.end()) {
                  fClassesHeadersMap[theTemplateHash].push_back(*classesHeader_inner);
               }
               addTemplate = false;
            }
         }
      }
   }


   if (strcmp(modulename,"libCore")!=0 && strcmp(modulename,"libRint")!=0
       && strcmp(modulename,"libThread")!=0 && strcmp(modulename,"libRIO")!=0
       && strcmp(modulename,"libcomplexDict")!=0 && strcmp(modulename,"libdequeDict")!=0
       && strcmp(modulename,"liblistDict")!=0 && strcmp(modulename,"libforward_listDict")!=0
       && strcmp(modulename,"libvectorDict")!=0
       && strcmp(modulename,"libmapDict")!=0 && strcmp(modulename,"libmultimap2Dict")!=0
       && strcmp(modulename,"libmap2Dict")!=0 && strcmp(modulename,"libmultimapDict")!=0
       && strcmp(modulename,"libsetDict")!=0 && strcmp(modulename,"libmultisetDict")!=0
       && strcmp(modulename,"libunordered_setDict")!=0 && strcmp(modulename,"libunordered_multisetDict")!=0
       && strcmp(modulename,"libunordered_mapDict")!=0 && strcmp(modulename,"libunordered_multimapDict")!=0
       && strcmp(modulename,"libvalarrayDict")!=0
       && strcmp(modulename,"G__GenVector32")!=0 && strcmp(modulename,"G__Smatrix32")!=0

       ) {
      // No pcm for now for libCore or libRint, the info is in the pch.
      if (!LoadPCM(pcmFileName, headers, triggerFunc)) {
         ::Error("TCling::RegisterModule", "cannot find dictionary module %s",
                 ROOT::TMetaUtils::GetModuleFileName(modulename).c_str());
      }
   }

   bool oldValue = false;
   if (fClingCallbacks)
     oldValue = SetClassAutoloading(false);

   { // scope within which diagnostics are de-activated
   // For now we disable diagnostics because we saw them already at
   // dictionary generation time. That won't be an issue with the PCMs.

      clangDiagSuppr diagSuppr(fInterpreter->getSema().getDiagnostics());

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
      Warning("TCling::RegisterModule","Diagnostics suppression should be gone by now.");
#endif
#endif

      if (!hasHeaderParsingOnDemand){
         const cling::Transaction* watermark = fInterpreter->getLastTransaction();
         cling::Interpreter::CompilationResult compRes = fInterpreter->parseForModule(code.Data());
         if (isACLiC) {
            // Register an unload point.
            fMetaProcessor->registerUnloadPoint(watermark, headers[0]);
         }

         assert(cling::Interpreter::kSuccess == compRes &&
                        "Payload code of a dictionary could not be parsed correctly.");
         if (compRes!=cling::Interpreter::kSuccess) {
            Warning("TCling::RegisterModule",
                  "Problems declaring payload for module %s.", modulename) ;
         }
      }
   }

   // Now that all the header have been registered/compiled, let's
   // make sure to 'reset' the TClass that have a class init in this module
   // but already had their type information available (using information/header
   // loaded from other modules or from class rules).
   if (!hasHeaderParsingOnDemand) {
      // This code is likely to be superseded by the similar code in LoadPCM,
      // and have been disabled, (inadvertently or awkwardly) by
      // commit 7903f09f3beea69e82ffba29f59fb2d656a4fd54 (Refactor the routines used for header parsing on demand)
      // whereas it seems that a more semantically correct conditional would have
      // been 'if this module does not have a rootpcm'.
      // Note: this need to be review when the clang pcm are being installed.
      //       #if defined(R__MUST_REVISIT)
      while (!fClassesToUpdate.empty()) {
         TClass *oldcl = fClassesToUpdate.back().first;
         if (oldcl->GetState() != TClass::kHasTClassInit) {
            // if (gDebug > 2) Info("RegisterModule", "Forcing TClass init for %s", oldcl->GetName());
            DictFuncPtr_t dict = fClassesToUpdate.back().second;
            fClassesToUpdate.pop_back();
            // Calling func could manipulate the list so, let maintain the list
            // then call the dictionary function.
            TClass *ncl = dict();
            if (ncl) ncl->PostLoadCheck();
         } else {
            fClassesToUpdate.pop_back();
         }
      }
   }

   if (fClingCallbacks)
     SetClassAutoloading(oldValue);

   if (!fromRootCling && !hasHeaderParsingOnDemand) {
      // __ROOTCLING__ might be pulled in through PCH
      fInterpreter->declare("#ifdef __ROOTCLING__\n"
                            "#undef __ROOTCLING__\n"
                            + gInterpreterClassDef +
                            "#endif");
   }

   if (dyLibName) {
      void* dyLibHandle = fRegisterModuleDyLibs.back();
      fRegisterModuleDyLibs.pop_back();
      dlclose(dyLibHandle);
   }

   SetClassAutoloading(oldAutoloadValue);
}

//______________________________________________________________________________
void TCling::RegisterTClassUpdate(TClass *oldcl,DictFuncPtr_t dict)
{
   // Register classes that already existed prior to their dictionary loading
   // and that already had a ClassInfo (and thus would not be refresh via
   // UpdateClassInfo.

   fClassesToUpdate.push_back(std::make_pair(oldcl,dict));
}

//______________________________________________________________________________
void TCling::UnRegisterTClassUpdate(const TClass *oldcl)
{
   // If the dictionary is loaded, we can remove the class from the list
   // (otherwise the class might be loaded twice).

   typedef std::vector<std::pair<TClass*,DictFuncPtr_t> >::iterator iterator;
   iterator stop = fClassesToUpdate.end();
   for(iterator i = fClassesToUpdate.begin();
       i != stop;
       ++i)
   {
      if ( i->first == oldcl ) {
         fClassesToUpdate.erase(i);
         return;
      }
   }
}

//______________________________________________________________________________
Long_t TCling::ProcessLine(const char* line, EErrorCode* error/*=0*/)
{
   // Let cling process a command line.
   //
   // If the command is executed and the error is 0, then the return value
   // is the int value corresponding to the result of the executed command
   // (float and double return values will be truncated).
   //

   // Copy the passed line, it comes from a static buffer in TApplication
   // which can be reentered through the Cling evaluation routines,
   // which would overwrite the static buffer and we would forget what we
   // were doing.
   //
   TString sLine(line);
   if (strstr(line,fantomline)) {
      // End-Of-Line action
      // See the comment (copied from above):
      // It is a "fantom" method to synchronize user keyboard input
      // and ROOT prompt line (for WIN32)
      // and is implemented by
      if (gApplication) {
         if (gApplication->IsCmdThread()) {
            if (gGlobalMutex && !gInterpreterMutex && fLockProcessLine) {
               gGlobalMutex->Lock();
               if (!gInterpreterMutex)
                  gInterpreterMutex = gGlobalMutex->Factory(kTRUE);
               gGlobalMutex->UnLock();
            }
            R__LOCKGUARD(fLockProcessLine ? gInterpreterMutex : 0);
            gROOT->SetLineIsProcessing();

            UpdateAllCanvases();

            gROOT->SetLineHasBeenProcessed();
         }
      }
      return 0;
   }

   if (gGlobalMutex && !gInterpreterMutex && fLockProcessLine) {
      gGlobalMutex->Lock();
      if (!gInterpreterMutex)
         gInterpreterMutex = gGlobalMutex->Factory(kTRUE);
      gGlobalMutex->UnLock();
   }
   R__LOCKGUARD(fLockProcessLine ? gInterpreterMutex : 0);
   gROOT->SetLineIsProcessing();

   struct InterpreterFlagsRAII_t {
      cling::Interpreter* fInterpreter;
      bool fWasDynamicLookupEnabled;

      InterpreterFlagsRAII_t(cling::Interpreter* interp):
         fInterpreter(interp),
         fWasDynamicLookupEnabled(interp->isDynamicLookupEnabled())
      {
         fInterpreter->enableDynamicLookup(true);
      }
      ~InterpreterFlagsRAII_t() {
         fInterpreter->enableDynamicLookup(fWasDynamicLookupEnabled);
      }
   } interpreterFlagsRAII(fInterpreter);

   // A non-zero returned value means the given line was
   // not a complete statement.
   int indent = 0;
   // This will hold the resulting value of the evaluation the given line.
   cling::Value result;
   cling::Interpreter::CompilationResult compRes = cling::Interpreter::kSuccess;
   if (!strncmp(sLine.Data(), ".L", 2) || !strncmp(sLine.Data(), ".x", 2) ||
       !strncmp(sLine.Data(), ".X", 2)) {
      // If there was a trailing "+", then CINT compiled the code above,
      // and we will need to strip the "+" before passing the line to cling.
      TString mod_line(sLine);
      TString aclicMode;
      TString arguments;
      TString io;
      TString fname = gSystem->SplitAclicMode(sLine.Data() + 3,
         aclicMode, arguments, io);
      if (aclicMode.Length()) {
         // Remove the leading '+'
         R__ASSERT(aclicMode[0]=='+' && "ACLiC mode must start with a +");
         aclicMode[0]='k';    // We always want to keep the .so around.
         if (aclicMode[1]=='+') {
            // We have a 2nd +
            aclicMode[1]='f'; // We want to force the recompilation.
         }
         if (!gSystem->CompileMacro(fname,aclicMode)) {
            // ACLiC failed.
            compRes = cling::Interpreter::kFailure;
         } else {
            if (strncmp(sLine.Data(), ".L", 2) != 0) {
               // if execution was requested.

               if (arguments.Length()==0) {
                  arguments = "()";
               }
               // We need to remove the extension.
               Ssiz_t ext = fname.Last('.');
               if (ext != kNPOS) {
                  fname.Remove(ext);
               }
               const char *function = gSystem->BaseName(fname);
               mod_line = function + arguments + io;
               cling::MetaProcessor::MaybeRedirectOutputRAII RAII(fMetaProcessor);
               indent = fMetaProcessor->process(mod_line, compRes, &result);
            }
         }
      } else {
         // not ACLiC
         bool unnamedMacro = false;
         {
            std::string line;
            std::ifstream in(fname);
            static const char whitespace[] = " \t\r\n";
            while (in) {
               std::getline(in, line);
               std::string::size_type posNonWS = line.find_first_not_of(whitespace);
               if (posNonWS == std::string::npos) continue;
               if (line[posNonWS] == '/' && line[posNonWS + 1] == '/')
                  // Too bad, we only suppose C++ comments here.
                  continue;
               unnamedMacro = (line[posNonWS] == '{');
               break;
            }
         }

         fCurExecutingMacros.push_back(fname);
         cling::MetaProcessor::MaybeRedirectOutputRAII RAII(fMetaProcessor);
         if (unnamedMacro) {
            compRes = fMetaProcessor->readInputFromFile(fname.Data(), &result,
                                                        true /*ignoreOutmostBlock*/);
         } else {
            // No DynLookup for .x, .L of named macros.
            fInterpreter->enableDynamicLookup(false);
            indent = fMetaProcessor->process(mod_line, compRes, &result);
         }
         fCurExecutingMacros.pop_back();
      }
   } // .L / .X / .x
   else {
      if (0!=strncmp(sLine.Data(), ".autodict ",10) && sLine != ".autodict") {
         // explicitly ignore .autodict without having to support it
         // in cling.
         cling::MetaProcessor::MaybeRedirectOutputRAII RAII(fMetaProcessor);

         // Turn off autoparsing if this is an include directive
         bool isInclusionDirective = sLine.Contains("\n#include");
         if (isInclusionDirective) {
            SuspendAutoParsing autoParseRaii(this);
            indent = fMetaProcessor->process(sLine, compRes, &result);
         } else {
            indent = fMetaProcessor->process(sLine, compRes, &result);
         }
      }
   }
   if (result.isValid())
      RegisterTemporary(result);
   if (indent) {
      if (error)
         *error = kProcessing;
      return 0;
   }
   if (error) {
      switch (compRes) {
      case cling::Interpreter::kSuccess: *error = kNoError; break;
      case cling::Interpreter::kFailure: *error = kRecoverable; break;
      case cling::Interpreter::kMoreInputExpected: *error = kProcessing; break;
      }
   }
   if (compRes == cling::Interpreter::kSuccess
       && result.isValid()
       && !result.isVoid())
   {
      gROOT->SetLineHasBeenProcessed();
      return result.simplisticCastAs<long>();
   }
   gROOT->SetLineHasBeenProcessed();
   return 0;
}

//______________________________________________________________________________
void TCling::PrintIntro()
{
   // No-op; see TRint instead.
}

//______________________________________________________________________________
void TCling::AddIncludePath(const char *path)
{
   // Add the given path to the list of directories in which the interpreter
   // looks for include files. Only one path item can be specified at a
   // time, i.e. "path1:path2" is not supported.

   R__LOCKGUARD(gInterpreterMutex);
   fInterpreter->AddIncludePath(path);
}

//______________________________________________________________________________
void TCling::InspectMembers(TMemberInspector& insp, const void* obj,
                            const TClass* cl, Bool_t isTransient)
{
   // Visit all members over members, recursing over base classes.

   if (insp.GetObjectValidity() == TMemberInspector::kUnset) {
      insp.SetObjectValidity(obj ? TMemberInspector::kValidObjectGiven
                             : TMemberInspector::kNoObjectGiven);
   }

   if (!cl || cl->GetCollectionProxy()) {
      // We do not need to investigate the content of the STL
      // collection, they are opaque to us (and details are
      // uninteresting).
      return;
   }

   static const TClassRef clRefString("std::string");
   if (clRefString == cl) {
      // We stream std::string without going through members..
      return;
   }

   const char* cobj = (const char*) obj; // for ptr arithmetics

   // Treat the case of std::complex in a special manner. We want to enforce
   // the layout of a stl implementation independent class, which is the
   // complex as implmented in ROOT5.

   // A simple lambda to simplify the code
   auto inspInspect =  [&] (ptrdiff_t offset){
      insp.Inspect(const_cast<TClass*>(cl), insp.GetParent(), "_real", cobj, isTransient);
      insp.Inspect(const_cast<TClass*>(cl), insp.GetParent(), "_imag", cobj + offset, isTransient);
   };

   auto complexType = TClassEdit::GetComplexType(cl->GetName());
   switch(complexType) {
      case TClassEdit::EComplexType::kNone:
      {
        break;
      }
      case TClassEdit::EComplexType::kFloat:
      {
        inspInspect(sizeof(float));
        return;
      }
      case TClassEdit::EComplexType::kDouble:
      {
        inspInspect(sizeof(double));
        return;
      }
      case TClassEdit::EComplexType::kInt:
      {
        inspInspect(sizeof(int));
        return;
      }
      case TClassEdit::EComplexType::kLong:
      {
        inspInspect(sizeof(long));
        return;
      }
   }

   static clang::PrintingPolicy
      printPol(fInterpreter->getCI()->getLangOpts());
   if (printPol.Indentation) {
      // not yet inialized
      printPol.Indentation = 0;
      printPol.SuppressInitializers = true;
   }

   const char* clname = cl->GetName();
   // Printf("Inspecting class %s\n", clname);

   const clang::ASTContext& astContext = fInterpreter->getCI()->getASTContext();
   const clang::Decl *scopeDecl = 0;
   const clang::Type *recordType = 0;

   if (cl->GetClassInfo()) {
      TClingClassInfo * clingCI = (TClingClassInfo *)cl->GetClassInfo();
      scopeDecl = clingCI->GetDecl();
      recordType = clingCI->GetType();
   } else {
      const cling::LookupHelper& lh = fInterpreter->getLookupHelper();
      // Diags will complain about private classes:
      scopeDecl = lh.findScope(clname, cling::LookupHelper::NoDiagnostics,
                               &recordType);
   }
   if (!scopeDecl) {
      Error("InspectMembers", "Cannot find Decl for class %s", clname);
      return;
   }
   const clang::CXXRecordDecl* recordDecl
     = llvm::dyn_cast<const clang::CXXRecordDecl>(scopeDecl);
   if (!recordDecl) {
      Error("InspectMembers", "Cannot find Decl for class %s is not a CXXRecordDecl.", clname);
      return;
   }

   cling::Interpreter::PushTransactionRAII deserRAII(fInterpreter);

   const clang::ASTRecordLayout& recLayout
      = astContext.getASTRecordLayout(recordDecl);

   // TVirtualCollectionProxy *proxy = cl->GetCollectionProxy();
   // if (proxy && ( proxy->GetProperties() & TVirtualCollectionProxy::kIsEmulated ) ) {
   //    Error("InspectMembers","The TClass for %s has an emulated proxy but we are looking at a compiled version of the collection!\n",
   //          cl->GetName());
   // }
   if (cl->Size() != recLayout.getSize().getQuantity()) {
      Error("InspectMembers","TClass and cling disagree on the size of the class %s, respectively %d %lld\n",
            cl->GetName(),cl->Size(),(Long64_t)recLayout.getSize().getQuantity());
   }

   unsigned iNField = 0;
   // iterate over fields
   // FieldDecls are non-static, else it would be a VarDecl.
   for (clang::RecordDecl::field_iterator iField = recordDecl->field_begin(),
        eField = recordDecl->field_end(); iField != eField;
        ++iField, ++iNField) {


      clang::QualType memberQT = iField->getType();
      if (recordType) {
         // if (we_need_to_do_the_subst_because_the_class_is_a_template_instance_of_double32_t)
         memberQT = ROOT::TMetaUtils::ReSubstTemplateArg(memberQT, recordType);
      }
      memberQT = cling::utils::Transform::GetPartiallyDesugaredType(astContext, memberQT, fNormalizedCtxt->GetConfig(), false /* fully qualify */);
      if (memberQT.isNull()) {
         std::string memberName;
         llvm::raw_string_ostream stream(memberName);
         iField->getNameForDiagnostic(stream, printPol, true /*fqi*/);
         stream.flush();
         Error("InspectMembers",
               "Cannot retrieve QualType for member %s while inspecting class %s",
               memberName.c_str(), clname);
         continue; // skip member
      }
      const clang::Type* memType = memberQT.getTypePtr();
      if (!memType) {
         std::string memberName;
         llvm::raw_string_ostream stream(memberName);
         iField->getNameForDiagnostic(stream, printPol, true /*fqi*/);
         stream.flush();
         Error("InspectMembers",
               "Cannot retrieve Type for member %s while inspecting class %s",
               memberName.c_str(), clname);
         continue; // skip member
      }

      const clang::Type* memNonPtrType = memType;
      Bool_t ispointer = false;
      if (memNonPtrType->isPointerType()) {
         ispointer = true;
         clang::QualType ptrQT
            = memNonPtrType->getAs<clang::PointerType>()->getPointeeType();
         if (recordType) {
            // if (we_need_to_do_the_subst_because_the_class_is_a_template_instance_of_double32_t)
            ptrQT = ROOT::TMetaUtils::ReSubstTemplateArg(ptrQT, recordType);
         }
         ptrQT = cling::utils::Transform::GetPartiallyDesugaredType(astContext, ptrQT, fNormalizedCtxt->GetConfig(), false /* fully qualify */);
         if (ptrQT.isNull()) {
            std::string memberName;
            llvm::raw_string_ostream stream(memberName);
            iField->getNameForDiagnostic(stream, printPol, true /*fqi*/);
            stream.flush();
            Error("InspectMembers",
                  "Cannot retrieve pointee Type for member %s while inspecting class %s",
                  memberName.c_str(), clname);
            continue; // skip member
         }
         memNonPtrType = ptrQT.getTypePtr();
      }

      // assemble array size(s): "[12][4][]"
      llvm::SmallString<8> arraySize;
      const clang::ArrayType* arrType = memNonPtrType->getAsArrayTypeUnsafe();
      unsigned arrLevel = 0;
      bool haveErrorDueToArray = false;
      while (arrType) {
         ++arrLevel;
         arraySize += '[';
         const clang::ConstantArrayType* constArrType =
         clang::dyn_cast<clang::ConstantArrayType>(arrType);
         if (constArrType) {
            constArrType->getSize().toStringUnsigned(arraySize);
         }
         arraySize += ']';
         clang::QualType subArrQT = arrType->getElementType();
         if (subArrQT.isNull()) {
            std::string memberName;
            llvm::raw_string_ostream stream(memberName);
            iField->getNameForDiagnostic(stream, printPol, true /*fqi*/);
            stream.flush();
            Error("InspectMembers",
                  "Cannot retrieve QualType for array level %d (i.e. element type of %s) for member %s while inspecting class %s",
                  arrLevel, subArrQT.getAsString(printPol).c_str(),
                  memberName.c_str(), clname);
            haveErrorDueToArray = true;
            break;
         }
         arrType = subArrQT.getTypePtr()->getAsArrayTypeUnsafe();
      }
      if (haveErrorDueToArray) {
         continue; // skip member
      }

      // construct member name
      std::string fieldName;
      if (memType->isPointerType()) {
         fieldName = "*";
      }

      // Check if this field has a custom ioname, if not, just use the one of the decl
      std::string ioname(iField->getName());
      ROOT::TMetaUtils::ExtractAttrPropertyFromName(**iField,"ioname",ioname);
      fieldName += ioname;
      fieldName += arraySize;

      // get member offset
      // NOTE currently we do not support bitfield and do not support
      // member that are not aligned on 'bit' boundaries.
      clang::CharUnits offset(astContext.toCharUnitsFromBits(recLayout.getFieldOffset(iNField)));
      ptrdiff_t fieldOffset = offset.getQuantity();

      // R__insp.Inspect(R__cl, R__insp.GetParent(), "fBits[2]", fBits);
      // R__insp.Inspect(R__cl, R__insp.GetParent(), "fName", &fName);
      // R__insp.InspectMember(fName, "fName.");
      // R__insp.Inspect(R__cl, R__insp.GetParent(), "*fClass", &fClass);
      insp.Inspect(const_cast<TClass*>(cl), insp.GetParent(), fieldName.c_str(), cobj + fieldOffset, isTransient);

      if (!ispointer) {
         const clang::CXXRecordDecl* fieldRecDecl = memNonPtrType->getAsCXXRecordDecl();
         if (fieldRecDecl) {
            // nested objects get an extra call to InspectMember
            // R__insp.InspectMember("FileStat_t", (void*)&fFileStat, "fFileStat.", false);
            std::string sFieldRecName;
            if (!ROOT::TMetaUtils::ExtractAttrPropertyFromName(*fieldRecDecl,"iotype",sFieldRecName)){
               ROOT::TMetaUtils::GetNormalizedName(sFieldRecName,
                                                   clang::QualType(memNonPtrType,0),
                                                   *fInterpreter,
                                                   *fNormalizedCtxt);
            }

            TDataMember* mbr = cl->GetDataMember(ioname.c_str());
            // if we can not find the member (which should not really happen),
            // let's consider it transient.
            Bool_t transient = isTransient || !mbr || !mbr->IsPersistent();

            insp.InspectMember(sFieldRecName.c_str(), cobj + fieldOffset,
                               (fieldName + '.').c_str(), transient);

         }
      }
   } // loop over fields

   // inspect bases
   // TNamed::ShowMembers(R__insp);
   unsigned iNBase = 0;
   for (clang::CXXRecordDecl::base_class_const_iterator iBase
        = recordDecl->bases_begin(), eBase = recordDecl->bases_end();
        iBase != eBase; ++iBase, ++iNBase) {
      clang::QualType baseQT = iBase->getType();
      if (baseQT.isNull()) {
         Error("InspectMembers",
               "Cannot find QualType for base number %d while inspecting class %s",
               iNBase, clname);
         continue;
      }
      const clang::CXXRecordDecl* baseDecl
         = baseQT->getAsCXXRecordDecl();
      if (!baseDecl) {
         Error("InspectMembers",
               "Cannot find CXXRecordDecl for base number %d while inspecting class %s",
               iNBase, clname);
         continue;
      }
      TClass* baseCl=nullptr;
      std::string sBaseName;
      // Try with the DeclId
      std::vector<TClass*> foundClasses;
      TClass::GetClass(static_cast<DeclId_t>(baseDecl), foundClasses);
      if (foundClasses.size()==1){
         baseCl=foundClasses[0];
      } else {
         // Try with the normalised Name, as a fallback
         if (!baseCl){
            ROOT::TMetaUtils::GetNormalizedName(sBaseName,
                                                baseQT,
                                                *fInterpreter,
                                                *fNormalizedCtxt);
            baseCl = TClass::GetClass(sBaseName.c_str());
         }
      }

      if (!baseCl){
         std::string qualNameForDiag;
         ROOT::TMetaUtils::GetQualifiedName(qualNameForDiag, *baseDecl);
         Error("InspectMembers",
               "Cannot find TClass for base class %s", qualNameForDiag.c_str() );
         continue;
      }

      int64_t baseOffset;
      if (iBase->isVirtual()) {
         if (insp.GetObjectValidity() == TMemberInspector::kNoObjectGiven) {
            if (!isTransient) {
               Error("InspectMembers",
                     "Base %s of class %s is virtual but no object provided",
                     sBaseName.c_str(), clname);
            }
            baseOffset = TVirtualStreamerInfo::kNeedObjectForVirtualBaseClass;
         } else {
            // We have an object to determine the vbase offset.
            TClingClassInfo* ci = (TClingClassInfo*)cl->GetClassInfo();
            TClingClassInfo* baseCi = (TClingClassInfo*)baseCl->GetClassInfo();
            if (ci && baseCi) {
               baseOffset = ci->GetBaseOffset(baseCi, const_cast<void*>(obj),
                                              true /*isDerivedObj*/);
               if (baseOffset == -1) {
                  Error("InspectMembers",
                        "Error calculating offset of virtual base %s of class %s",
                        sBaseName.c_str(), clname);
               }
            } else {
               Error("InspectMembers",
                     "Cannot calculate offset of virtual base %s of class %s",
                     sBaseName.c_str(), clname);
               continue;
            }
         }
      } else {
         baseOffset = recLayout.getBaseClassOffset(baseDecl).getQuantity();
      }
      // TOFIX: baseCl can be null here!
      if (baseCl->IsLoaded()) {
         // For loaded class, CallShowMember will (especially for TObject)
         // call the virtual ShowMember rather than the class specific version
         // resulting in an infinite recursion.
         InspectMembers(insp, cobj + baseOffset, baseCl, isTransient);
      } else {
         baseCl->CallShowMembers(cobj + baseOffset,
                                 insp, isTransient);
      }
   } // loop over bases
}

//______________________________________________________________________________
void TCling::ClearFileBusy()
{
   // Reset the interpreter internal state in case a previous action was not correctly
   // terminated.

   // No-op there is not equivalent state (to be cleared) in Cling.
}

//______________________________________________________________________________
void TCling::ClearStack()
{
   // Delete existing temporary values.

   // No-op for cling due to cling::Value.
}

//______________________________________________________________________________
bool TCling::Declare(const char* code)
{
   // Declare code to the interpreter, without any of the interpreter actions
   // that could trigger a re-interpretation of the code. I.e. make cling
   // behave like a compiler: no dynamic lookup, no input wrapping for
   // subsequent execution, no automatic provision of declarations but just a
   // plain #include.
   // Returns true on success, false on failure.

   R__LOCKGUARD(gInterpreterMutex);

   int oldload = SetClassAutoloading(0);
   SuspendAutoParsing autoParseRaii(this);

   bool oldDynLookup = fInterpreter->isDynamicLookupEnabled();
   fInterpreter->enableDynamicLookup(false);
   bool oldRawInput = fInterpreter->isRawInputEnabled();
   fInterpreter->enableRawInput(true);

   Bool_t ret = LoadText(code);

   fInterpreter->enableRawInput(oldRawInput);
   fInterpreter->enableDynamicLookup(oldDynLookup);
   SetClassAutoloading(oldload);
   return ret;
}

//______________________________________________________________________________
void TCling::EnableAutoLoading()
{
   // Enable the automatic loading of shared libraries when a class
   // is used that is stored in a not yet loaded library. Uses the
   // information stored in the class/library map (typically
   // $ROOTSYS/etc/system.rootmap).
   if (fAllowLibLoad) {
      LoadLibraryMap();
      SetClassAutoloading(true);
   }
}

//______________________________________________________________________________
void TCling::EndOfLineAction()
{
   // It calls a "fantom" method to synchronize user keyboard input
   // and ROOT prompt line.
   ProcessLineSynch(fantomline);
}

//______________________________________________________________________________
Bool_t TCling::IsLoaded(const char* filename) const
{
   // Return true if the file has already been loaded by cint.
   // We will try in this order:
   //   actual filename
   //   filename as a path relative to
   //            the include path
   //            the shared library path
   R__LOCKGUARD(gInterpreterMutex);

   //FIXME: if we use llvm::sys::fs::make_absolute all this can go away. See
   // cling::DynamicLibraryManager.

   std::string file_name = filename;
   size_t at = std::string::npos;
   while ((at = file_name.find("/./")) != std::string::npos)
       file_name.replace(at, 3, "/");

   std::string filesStr = "";
   llvm::raw_string_ostream filesOS(filesStr);
   clang::SourceManager &SM = fInterpreter->getCI()->getSourceManager();
   cling::ClangInternalState::printIncludedFiles(filesOS, SM);
   filesOS.flush();

   llvm::SmallVector<llvm::StringRef, 100> files;
   llvm::StringRef(filesStr).split(files, "\n");

   std::set<std::string> fileMap;
   // Fill fileMap; return early on exact match.
   for (llvm::SmallVector<llvm::StringRef, 100>::const_iterator
           iF = files.begin(), iE = files.end(); iF != iE; ++iF) {
      if ((*iF) == file_name.c_str()) return kTRUE; // exact match
      fileMap.insert(*iF);
   }

   if (fileMap.empty()) return kFALSE;

   // Check MacroPath.
   TString sFilename(file_name.c_str());
   if (gSystem->FindFile(TROOT::GetMacroPath(), sFilename, kReadPermission)
       && fileMap.count(sFilename.Data())) {
      return kTRUE;
   }

   // Check IncludePath.
   TString incPath = gSystem->GetIncludePath(); // of the form -Idir1  -Idir2 -Idir3
   incPath.Append(":").Prepend(" "); // to match " -I" (note leading ' ')
   incPath.ReplaceAll(" -I", ":");      // of form :dir1 :dir2:dir3
   while (incPath.Index(" :") != -1) {
      incPath.ReplaceAll(" :", ":");
   }
   incPath.Prepend(".:");
   sFilename = file_name.c_str();
   if (gSystem->FindFile(incPath, sFilename, kReadPermission)
       && fileMap.count(sFilename.Data())) {
      return kTRUE;
   }

   // Check shared library.
   sFilename = file_name.c_str();
   const char *found = gSystem->FindDynamicLibrary(sFilename, kTRUE);
   cling::DynamicLibraryManager* dyLibManager
      = fInterpreter->getDynamicLibraryManager();
   if (found) {
      if (dyLibManager->isLibraryLoaded(found)) {
         return kTRUE;
      }
   }
   //FIXME: We must use the cling::Interpreter::lookupFileOrLibrary iface.
   const clang::DirectoryLookup *CurDir = 0;
   clang::Preprocessor &PP = fInterpreter->getCI()->getPreprocessor();
   clang::HeaderSearch &HS = PP.getHeaderSearchInfo();
   const clang::FileEntry *FE = HS.LookupFile(file_name.c_str(),
                                              clang::SourceLocation(),
                                              /*isAngled*/ false,
                                              /*FromDir*/ 0, CurDir,
                                              clang::ArrayRef<std::pair<const clang::FileEntry *,
                                                                        const clang::DirectoryEntry *>>(),
                                              /*SearchPath*/ 0,
                                              /*RelativePath*/ 0,
                                              /*SuggestedModule*/ 0,
                                              /*SkipCache*/ false,
                                              /*OpenFile*/ false,
                                              /*CacheFail*/ false);
   if (FE && FE->isValid()) {
      // check in the source manager if the file is actually loaded
      clang::SourceManager &SM = fInterpreter->getCI()->getSourceManager();
      // this works only with header (and source) files...
      clang::FileID FID = SM.translateFile(FE);
      if (!FID.isInvalid() && FID.getHashValue() == 0)
         return kFALSE;
      else {
         clang::SrcMgr::SLocEntry SLocE = SM.getSLocEntry(FID);
         if (SLocE.isFile() && SLocE.getFile().getContentCache()->getRawBuffer() == 0)
            return kFALSE;
         if (!FID.isInvalid())
            return kTRUE;
      }
      // ...then check shared library again, but with full path now
      sFilename = FE->getName();
      if (gSystem->FindDynamicLibrary(sFilename, kTRUE)
          && fileMap.count(sFilename.Data())) {
         return kTRUE;
      }
   }
   return kFALSE;
}

//______________________________________________________________________________
void TCling::UpdateListOfLoadedSharedLibraries()
{
#if defined(R__WIN32) || defined(__CYGWIN__)
   HMODULE hModules[1024];
   void *hProcess;
   unsigned long cbModules;
   unsigned int i;
   hProcess = (void *)::GetCurrentProcess();
   ::EnumProcessModules(hProcess, hModules, sizeof(hModules), &cbModules);
   // start at 1 to skip the executable itself
   for (i = 1; i < (cbModules / sizeof(void *)); i++) {
      static const int bufsize = 260;
      wchar_t winname[bufsize];
      char posixname[bufsize];
      ::GetModuleFileNameExW(hProcess, hModules[i], winname, bufsize);
#if defined(__CYGWIN__)
      cygwin_conv_path(CCP_WIN_W_TO_POSIX, winname, posixname, bufsize);
#else
      std::wstring wpath = winname;
      std::replace(wpath.begin(), wpath.end(), '\\', '/');
      string path(wpath.begin(), wpath.end());
      strncpy(posixname, path.c_str(), bufsize);
#endif
      if (!fSharedLibs.Contains(posixname)) {
         RegisterLoadedSharedLibrary(posixname);
      }
   }
#elif defined(R__MACOSX)
   // fPrevLoadedDynLibInfo stores the *next* image index to look at
   uint32_t imageIndex = (uint32_t) (size_t) fPrevLoadedDynLibInfo;
   const char* imageName = 0;
   while ((imageName = _dyld_get_image_name(imageIndex))) {
      // Skip binary
      if (imageIndex > 0)
         RegisterLoadedSharedLibrary(imageName);
      ++imageIndex;
   }
   fPrevLoadedDynLibInfo = (void*)(size_t)imageIndex;
#elif defined(R__LINUX)
   struct PointerNo4_t {
      void* fSkip[3];
      void* fPtr;
   };
   struct LinkMap_t {
      void* fAddr;
      const char* fName;
      void* fLd;
      LinkMap_t* fNext;
      LinkMap_t* fPrev;
   };
   if (!fPrevLoadedDynLibInfo || fPrevLoadedDynLibInfo == (void*)(size_t)-1) {
      PointerNo4_t* procLinkMap = (PointerNo4_t*)dlopen(0,  RTLD_LAZY | RTLD_GLOBAL);
      // 4th pointer of 4th pointer is the linkmap.
      // See http://syprog.blogspot.fr/2011/12/listing-loaded-shared-objects-in-linux.html
      LinkMap_t* linkMap = (LinkMap_t*) ((PointerNo4_t*)procLinkMap->fPtr)->fPtr;
      RegisterLoadedSharedLibrary(linkMap->fName);
      fPrevLoadedDynLibInfo = linkMap;
   }

   LinkMap_t* iDyLib = (LinkMap_t*)fPrevLoadedDynLibInfo;
   while (iDyLib->fNext) {
      iDyLib = iDyLib->fNext;
      RegisterLoadedSharedLibrary(iDyLib->fName);
   }
   fPrevLoadedDynLibInfo = iDyLib;
#else
   Error("TCling::UpdateListOfLoadedSharedLibraries",
         "Platform not supported!");
#endif
}

//______________________________________________________________________________
void TCling::RegisterLoadedSharedLibrary(const char* filename)
{
   // Register a new shared library name with the interpreter; add it to
   // fSharedLibs.

   // Ignore NULL filenames, aka "the process".
   if (!filename) return;

   // Tell the interpreter that this library is available; all libraries can be
   // used to resolve symbols.
   cling::DynamicLibraryManager* DLM = fInterpreter->getDynamicLibraryManager();
   if (!DLM->isLibraryLoaded(filename)) {
      DLM->loadLibrary(filename, true /*permanent*/);
   }

#if defined(R__MACOSX)
   // Check that this is not a system library
   if (!strncmp(filename, "/usr/lib/system/", 16)
       || !strncmp(filename, "/usr/lib/libc++", 15)
       || !strncmp(filename, "/System/Library/Frameworks/", 27)
       || !strncmp(filename, "/System/Library/PrivateFrameworks/", 34)
       || !strncmp(filename, "/System/Library/CoreServices/", 29)
       || !strcmp(filename, "cl_kernels") // yepp, no directory
       || strstr(filename, "/usr/lib/libSystem")
       || strstr(filename, "/usr/lib/libstdc++")
       || strstr(filename, "/usr/lib/libicucore")
       || strstr(filename, "/usr/lib/libbsm")
       || strstr(filename, "/usr/lib/libobjc")
       || strstr(filename, "/usr/lib/libresolv")
       || strstr(filename, "/usr/lib/libauto")
       || strstr(filename, "/usr/lib/libcups")
       || strstr(filename, "/usr/lib/libDiagnosticMessagesClient")
       || strstr(filename, "/usr/lib/liblangid")
       || strstr(filename, "/usr/lib/libCRFSuite")
       || strstr(filename, "/usr/lib/libpam")
       || strstr(filename, "/usr/lib/libOpenScriptingUtil")
       || strstr(filename, "/usr/lib/libextension"))
      return;
#elif defined(__CYGWIN__)
   // Check that this is not a system library
   static const int bufsize = 260;
   char posixwindir[bufsize];
   char *windir = getenv("WINDIR");
   if (windir)
      cygwin_conv_path(CCP_WIN_A_TO_POSIX, windir, posixwindir, bufsize);
   else
      snprintf(posixwindir, sizeof(posixwindir), "/Windows/");
   if (strstr(filename, posixwindir) ||
       strstr(filename, "/usr/bin/cyg"))
      return;
#elif defined(R__WIN32)
   if (strstr(filename, "/Windows/"))
      return;
#elif defined (R__LINUX)
   if (strstr(filename, "/ld-linux")
       || strstr(filename, "linux-gnu/")
       || strstr(filename, "/libstdc++.")
       || strstr(filename, "/libgcc")
       || strstr(filename, "/libc.")
       || strstr(filename, "/libdl.")
       || strstr(filename, "/libm."))
      return;
#endif
   // Update string of available libraries.
   if (!fSharedLibs.IsNull()) {
      fSharedLibs.Append(" ");
   }
   fSharedLibs.Append(filename);
}

//______________________________________________________________________________
Int_t TCling::Load(const char* filename, Bool_t system)
{
   // Load a library file in cling's memory.
   // if 'system' is true, the library is never unloaded.
   // Return 0 on success, -1 on failure.

   if (!fAllowLibLoad) {
      Error("Load","Trying to load library (%s) from rootcling.",filename);
      return -1;
   }

   // Used to return 0 on success, 1 on duplicate, -1 on failure, -2 on "fatal".
   R__LOCKGUARD2(gInterpreterMutex);
   cling::DynamicLibraryManager* DLM = fInterpreter->getDynamicLibraryManager();
   std::string canonLib = DLM->lookupLibrary(filename);
   cling::DynamicLibraryManager::LoadLibResult res
      = cling::DynamicLibraryManager::kLoadLibNotFound;
   if (!canonLib.empty()) {
      if (system)
         res = DLM->loadLibrary(filename, system);
      else {
         // For the non system libs, we'd like to be able to unload them.
         // FIXME: Here we lose the information about kLoadLibAlreadyLoaded case.
         cling::Interpreter::CompilationResult compRes;
         cling::MetaProcessor::MaybeRedirectOutputRAII RAII(fMetaProcessor);
         fMetaProcessor->process(Form(".L %s", canonLib.c_str()), compRes, /*cling::Value*/0);
         if (compRes == cling::Interpreter::kSuccess)
            res = cling::DynamicLibraryManager::kLoadLibSuccess;
      }
   }

   if (res == cling::DynamicLibraryManager::kLoadLibSuccess) {
      UpdateListOfLoadedSharedLibraries();
   }
   switch (res) {
   case cling::DynamicLibraryManager::kLoadLibSuccess: return 0;
   case cling::DynamicLibraryManager::kLoadLibAlreadyLoaded:  return 1;
   default: break;
   };
   return -1;
}

//______________________________________________________________________________
void TCling::LoadMacro(const char* filename, EErrorCode* error)
{
   // Load a macro file in cling's memory.
   ProcessLine(Form(".L %s", filename), error);
}

//______________________________________________________________________________
Long_t TCling::ProcessLineAsynch(const char* line, EErrorCode* error)
{
   // Let cling process a command line asynch.
   return ProcessLine(line, error);
}

//______________________________________________________________________________
Long_t TCling::ProcessLineSynch(const char* line, EErrorCode* error)
{
   // Let cling process a command line synchronously, i.e we are waiting
   // it will be finished.
   R__LOCKGUARD(fLockProcessLine ? gInterpreterMutex : 0);
   if (gApplication) {
      if (gApplication->IsCmdThread()) {
         return ProcessLine(line, error);
      }
      return 0;
   }
   return ProcessLine(line, error);
}

//______________________________________________________________________________
Long_t TCling::Calc(const char* line, EErrorCode* error)
{
   // Directly execute an executable statement (e.g. "func()", "3+5", etc.
   // however not declarations, like "Int_t x;").
#ifdef R__WIN32
   // Test on ApplicationImp not being 0 is needed because only at end of
   // TApplication ctor the IsLineProcessing flag is set to 0, so before
   // we can not use it.
   if (gApplication && gApplication->GetApplicationImp()) {
      while (gROOT->IsLineProcessing() && !gApplication) {
         Warning("Calc", "waiting for cling thread to free");
         gSystem->Sleep(500);
      }
      gROOT->SetLineIsProcessing();
   }
#endif // R__WIN32
   R__LOCKGUARD2(gInterpreterMutex);
   if (error) {
      *error = TInterpreter::kNoError;
   }
   cling::Value valRef;
   cling::Interpreter::CompilationResult cr = fInterpreter->evaluate(line, valRef);
   if (cr != cling::Interpreter::kSuccess) {
      // Failure in compilation.
      if (error) {
         // Note: Yes these codes are weird.
         *error = TInterpreter::kRecoverable;
      }
      return 0L;
   }
   if (!valRef.isValid()) {
      // Failure at runtime.
      if (error) {
         // Note: Yes these codes are weird.
         *error = TInterpreter::kDangerous;
      }
      return 0L;
   }

   if (valRef.isVoid()) {
      return 0;
   }

   RegisterTemporary(valRef);
#ifdef R__WIN32
   if (gApplication && gApplication->GetApplicationImp()) {
      gROOT->SetLineHasBeenProcessed();
   }
#endif // R__WIN32
   return valRef.simplisticCastAs<long>();
}

//______________________________________________________________________________
void TCling::SetGetline(const char * (*getlineFunc)(const char* prompt),
                                void (*histaddFunc)(const char* line))
{
   // Set a getline function to call when input is needed.

   // If cling offers a replacement for G__pause(), it would need to
   // also offer a way to customize at least the history recording.

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   Warning("SetGetline","Cling should support the equivalent of SetGetlineFunc(getlineFunc, histaddFunc)");
#endif
#endif
}

//______________________________________________________________________________
Bool_t TCling::HandleNewTransaction(const cling::Transaction &T)
{
   // Helper function to increase the internal Cling count of transactions
   // that change the AST.

   R__LOCKGUARD(gInterpreterMutex);

   if ((std::distance(T.decls_begin(), T.decls_end()) != 1)
      || T.deserialized_decls_begin() != T.deserialized_decls_end()
      || T.macros_begin() != T.macros_end()
      || ((!T.getFirstDecl().isNull()) && ((*T.getFirstDecl().begin()) != T.getWrapperFD()))) {
      fTransactionCount++;
      return true;
   }
   return false;
}

//______________________________________________________________________________
void TCling::RecursiveRemove(TObject* obj)
{
   // Delete object from cling symbol table so it can not be used anymore.
   // cling objects are always on the heap.
   R__LOCKGUARD(gInterpreterMutex);
   // Note that fgSetOfSpecials is supposed to be updated by TClingCallbacks::tryFindROOTSpecialInternal
   // (but isn't at the moment).
   if (obj->IsOnHeap() && fgSetOfSpecials && !((std::set<TObject*>*)fgSetOfSpecials)->empty()) {
      std::set<TObject*>::iterator iSpecial = ((std::set<TObject*>*)fgSetOfSpecials)->find(obj);
      if (iSpecial != ((std::set<TObject*>*)fgSetOfSpecials)->end()) {
         DeleteGlobal(obj);
         ((std::set<TObject*>*)fgSetOfSpecials)->erase(iSpecial);
      }
   }
}

//______________________________________________________________________________
void TCling::Reset()
{
   // Pressing Ctrl+C should forward here. In the case where we have had
   // continuation requested we must reset it.
   fMetaProcessor->cancelContinuation();
   // Reset the Cling state to the state saved by the last call to
   // TCling::SaveContext().
#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   R__LOCKGUARD(gInterpreterMutex);
   Warning("Reset","Cling should support the equivalent of scratch_upto(&fDictPos)");
#endif
#endif
}

//______________________________________________________________________________
void TCling::ResetAll()
{
   // Reset the Cling state to its initial state.

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   R__LOCKGUARD(gInterpreterMutex);
   Warning("ResetAll","Cling should support the equivalent of complete reset (unload everything but the startup decls.");
#endif
#endif
}

//______________________________________________________________________________
void TCling::ResetGlobals()
{
   // Reset in Cling the list of global variables to the state saved by the last
   // call to TCling::SaveGlobalsContext().
   //
   // Note: Right now, all we do is run the global destructors.
   //
   R__LOCKGUARD(gInterpreterMutex);
   // TODO:
   // Here we should iterate over the transactions (N-3) and revert.
   // N-3 because the first three internal to cling.

   fInterpreter->runAndRemoveStaticDestructors();
}

//______________________________________________________________________________
void TCling::ResetGlobalVar(void* obj)
{
   // Reset the Cling 'user' global objects/variables state to the state saved by the last
   // call to TCling::SaveGlobalsContext().

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   R__LOCKGUARD(gInterpreterMutex);
   Warning("ResetGlobalVar","Cling should support the equivalent of resetglobalvar(obj)");
#endif
#endif
}

//______________________________________________________________________________
void TCling::RewindDictionary()
{
   // Rewind Cling dictionary to the point where it was before executing
   // the current macro. This function is typically called after SEGV or
   // ctlr-C after doing a longjmp back to the prompt.

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   R__LOCKGUARD(gInterpreterMutex);
   Warning("RewindDictionary","Cling should provide a way to revert transaction similar to rewinddictionary()");
#endif
#endif
}

//______________________________________________________________________________
Int_t TCling::DeleteGlobal(void* obj)
{
   // Delete obj from Cling symbol table so it cannot be accessed anymore.
   // Returns 1 in case of success and 0 in case object was not in table.

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   R__LOCKGUARD(gInterpreterMutex);
   Warning("DeleteGlobal","Cling should provide the equivalent of deleteglobal(obj), see also DeleteVariable.");
#endif
#endif
   return 0;
}

//______________________________________________________________________________
Int_t TCling::DeleteVariable(const char* name)
{
   // Undeclare obj called name.
   // Returns 1 in case of success, 0 for failure.

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   Warning("DeleteVariable","should do more that just reseting the value to zero");
#endif
#endif

   R__LOCKGUARD(gInterpreterMutex);
   llvm::StringRef srName(name);
   const char* unscopedName = name;
   llvm::StringRef::size_type posScope = srName.rfind("::");
   const clang::DeclContext* declCtx = 0;
   if (posScope != llvm::StringRef::npos) {
      const cling::LookupHelper& lh = fInterpreter->getLookupHelper();
      const clang::Decl* scopeDecl
         = lh.findScope(srName.substr(0, posScope),
                        cling::LookupHelper::WithDiagnostics);
      if (!scopeDecl) {
         Error("DeleteVariable", "Cannot find enclosing scope for variable %s",
               name);
         return 0;
      }
      declCtx = llvm::dyn_cast<clang::DeclContext>(scopeDecl);
      if (!declCtx) {
         Error("DeleteVariable",
               "Enclosing scope for variable %s is not a declaration context",
               name);
         return 0;
      }
      unscopedName += posScope + 2;
   }
   clang::NamedDecl* nVarDecl
      = cling::utils::Lookup::Named(&fInterpreter->getSema(), unscopedName, declCtx);
   if (!nVarDecl) {
      Error("DeleteVariable", "Unknown variable %s", name);
      return 0;
   }
   clang::VarDecl* varDecl = llvm::dyn_cast<clang::VarDecl>(nVarDecl);
   if (!varDecl) {
      Error("DeleteVariable", "Entity %s is not a variable", name);
      return 0;
   }

   clang::QualType qType = varDecl->getType();
   const clang::Type* type = qType->getUnqualifiedDesugaredType();
   // Cannot set a reference's address to nullptr; the JIT can place it
   // into read-only memory (ROOT-7100).
   if (type->isPointerType()) {
      int** ppInt = (int**)fInterpreter->getAddressOfGlobal(GlobalDecl(varDecl));
      // set pointer to invalid.
      if (ppInt) *ppInt = 0;
   }
   return 1;
}

//______________________________________________________________________________
void TCling::SaveContext()
{
   // Save the current Cling state.

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   R__LOCKGUARD(gInterpreterMutex);
   Warning("SaveContext","Cling should provide a way to record a state watermark similar to store_dictposition(&fDictPos)");
#endif
#endif
}

//______________________________________________________________________________
void TCling::SaveGlobalsContext()
{
   // Save the current Cling state of global objects.

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   R__LOCKGUARD(gInterpreterMutex);
   Warning("SaveGlobalsContext","Cling should provide a way to record a watermark for the list of global variable similar to store_dictposition(&fDictPosGlobals)");
#endif
#endif
}

//______________________________________________________________________________
void TCling::UpdateListOfGlobals()
{
   // No op: see TClingCallbacks (used to update the list of globals)
}

//______________________________________________________________________________
void TCling::UpdateListOfGlobalFunctions()
{
   // No op: see TClingCallbacks (used to update the list of global functions)
}

//______________________________________________________________________________
void TCling::UpdateListOfTypes()
{
   // No op: see TClingCallbacks (used to update the list of types)
}

//______________________________________________________________________________
void TCling::SetClassInfo(TClass* cl, Bool_t reload)
{
   // Set pointer to the TClingClassInfo in TClass.
   // If 'reload' is true, (attempt to) generate a new ClassInfo even if we
   // already have one.

   R__LOCKGUARD2(gInterpreterMutex);
   if (cl->fClassInfo && !reload) {
      return;
   }
   //Remove the decl_id from the DeclIdToTClass map
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cl->fClassInfo;
   if (TClinginfo) {
      TClass::RemoveClassDeclId(TClinginfo->GetDeclId());
   }
   delete TClinginfo;
   cl->fClassInfo = 0;
   std::string name(cl->GetName());
   TClingClassInfo* info = new TClingClassInfo(fInterpreter, name.c_str());
   if (!info->IsValid()) {
      if (cl->fState != TClass::kHasTClassInit) {
         if (cl->fStreamerInfo->GetEntries() != 0) {
            cl->fState = TClass::kEmulated;
         } else {
            cl->fState = TClass::kForwardDeclared;
         }
      }
      return;
   }
   cl->fClassInfo = (ClassInfo_t*)info; // Note: We are transfering ownership here.
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
      // this happens when no dictionary is available
      delete info;
      cl->fClassInfo = 0;
   }
   if (zombieCandidate && !cl->GetCollectionType()) {
      cl->MakeZombie();
   }
   // If we reach here, the info was valid (See early returns).
   if (cl->fState != TClass::kHasTClassInit) {
      if (cl->fClassInfo) {
         cl->fState = TClass::kInterpreted;
         cl->ResetBit(TClass::kIsEmulation);
      } else {
//         if (TClassEdit::IsSTLCont(cl->GetName()) {
//            There will be an emulated collection proxy, is that the same?
//            cl->fState = TClass::kEmulated;
//         } else {
         if (cl->fStreamerInfo->GetEntries() != 0) {
            cl->fState = TClass::kEmulated;
         } else {
            cl->fState = TClass::kForwardDeclared;
         }
//         }
      }
   }
   if (cl->fClassInfo) {
      TClass::AddClassToDeclIdMap(((TClingClassInfo*)cl->fClassInfo)->GetDeclId(), cl);
   }
}

//______________________________________________________________________________
Bool_t TCling::CheckClassInfo(const char* name, Bool_t autoload, Bool_t isClassOrNamespaceOnly /* = kFALSE*/ )
{
   // Checks if an entity with the specified name is defined in Cling.
   // Returns kFALSE if the entity is not defined.
   // By default, structs, namespaces, classes, enums and unions are looked for.
   // If the flag isClassOrNamespaceOnly is true, classes, structs and
   // namespaces only are considered. I.e. if the name is an enum or a union,
   // the returned value is false.
   // In the case where the class is not loaded and belongs to a namespace
   // or is nested, looking for the full class name is outputing a lots of
   // (expected) error messages.  Currently the only way to avoid this is to
   // specifically check that each level of nesting is already loaded.
   // In case of templates the idea is that everything between the outer
   // '<' and '>' has to be skipped, e.g.: aap<pippo<noot>::klaas>::a_class

   R__LOCKGUARD(gInterpreterMutex);
   static const char *anonEnum = "anonymous enum ";
   static const int cmplen = strlen(anonEnum);

   if (0 == strncmp(name,anonEnum,cmplen)) {
      return kFALSE;
   }

   // Avoid the double search below in case the name is a fundamental type
   // or typedef to a fundamental type.
   THashTable *typeTable = dynamic_cast<THashTable*>( gROOT->GetListOfTypes() );
   TDataType *fundType = (TDataType *)typeTable->THashTable::FindObject( name );

   if (fundType && fundType->GetType() < TVirtualStreamerInfo::kObject
       && fundType->GetType() > 0) {
      // Fundamental type, no a class.
      return kFALSE;
   }

   // Migrated from within TClass::GetClass
   // If we want to know if a class or a namespace with this name exists in the
   // interpreter and this is an enum in the type system, before or after loading
   // according to the autoload function argument, return false.
   if (isClassOrNamespaceOnly &&
       TEnum::GetEnum(name, autoload ? TEnum::kAutoload : TEnum::kNone)) return false;

   const char *classname = name;

   int storeAutoload = SetClassAutoloading(autoload);

   // First we want to check whether the decl exist, but _without_
   // generating any template instantiation. However, the lookup
   // still will create a forward declaration of the class template instance
   // if it exist.  In this case, the return value of findScope will still
   // be zero but the type will be initialized.
   // Note in the corresponding code in ROOT 5, CINT was not instantiating
   // this forward declaration.
   const cling::LookupHelper& lh = fInterpreter->getLookupHelper();
   const clang::Type *type = 0;
   const clang::Decl *decl
      = lh.findScope(classname,
                     gDebug > 5 ? cling::LookupHelper::WithDiagnostics
                     : cling::LookupHelper::NoDiagnostics,
                     &type, /* intantiateTemplate= */ false );
   if (!decl) {
      std::string buf = TClassEdit::InsertStd(classname);
      decl = lh.findScope(buf,
                          gDebug > 5 ? cling::LookupHelper::WithDiagnostics
                          : cling::LookupHelper::NoDiagnostics,
                          &type,false);
   }

   if (type) {
      // If decl==0 and the type is valid, then we have a forward declaration.
      if (!decl) {
         // If we have a forward declaration for a class template instantiation,
         // we want to ignore it if it was produced/induced by the call to
         // findScope, however we can not distinguish those from the
         // instantiation induce by 'soft' use (and thus also induce by the
         // same underlying code paths)
         // ['soft' use = use not requiring a complete definition]
         // So to reduce the amount of disruption to the existing code we
         // would just ignore those for STL collection, for which we really
         // need to have the compiled collection proxy (and thus the TClass
         // bootstrap).
         clang::ClassTemplateSpecializationDecl *tmpltDecl =
            llvm::dyn_cast_or_null<clang::ClassTemplateSpecializationDecl>
               (type->getAsCXXRecordDecl());
         if (tmpltDecl && !tmpltDecl->getPointOfInstantiation().isValid()) {
            // Since the point of instantiation is invalid, we 'guess' that
            // the 'instantiation' of the forwarded type appended in
            // findscope.
            if (ROOT::TMetaUtils::IsSTLCont(*tmpltDecl)) {
               // For STL Collection we return false.
               SetClassAutoloading(storeAutoload);
               return kFALSE;
            }
         }
      }
      TClingClassInfo tci(fInterpreter, *type);
      if (!tci.IsValid()) {
         SetClassAutoloading(storeAutoload);
         return kFALSE;
      }
      auto propertiesMask = isClassOrNamespaceOnly ? kIsClass | kIsStruct | kIsNamespace :
                                                     kIsClass | kIsStruct | kIsNamespace | kIsEnum | kIsUnion;

      if (tci.Property() & propertiesMask) {
         // We are now sure that the entry is not in fact an autoload entry.
         SetClassAutoloading(storeAutoload);
         return kTRUE;
      }
   }

   SetClassAutoloading(storeAutoload);
   return (decl);

   // Setting up iterator part of TClingTypedefInfo is too slow.
   // Copy the lookup code instead:
   /*
   TClingTypedefInfo t(fInterpreter, name);
   if (t.IsValid() && !(t.Property() & kIsFundamental)) {
      delete[] classname;
      SetClassAutoloading(storeAutoload);
      return kTRUE;
   }
   */

//   const clang::Decl *decl = lh.findScope(name);
//   if (!decl) {
//      std::string buf = TClassEdit::InsertStd(name);
//      decl = lh.findScope(buf);
//   }

//   SetClassAutoloading(storeAutoload);
//   return (decl);
}

//______________________________________________________________________________
Bool_t TCling::CheckClassTemplate(const char *name)
{
   // Return true if there is a class template by the given name ...

   const cling::LookupHelper& lh = fInterpreter->getLookupHelper();
   const clang::Decl *decl
      = lh.findClassTemplate(name,
                             gDebug > 5 ? cling::LookupHelper::WithDiagnostics
                             : cling::LookupHelper::NoDiagnostics);
   if (!decl) {
      std::string strname = "std::";
      strname += name;
      decl = lh.findClassTemplate(strname,
                                  gDebug > 5 ? cling::LookupHelper::WithDiagnostics
                                  : cling::LookupHelper::NoDiagnostics);
   }
   return 0 != decl;
}

//______________________________________________________________________________
void TCling::CreateListOfBaseClasses(TClass *cl) const
{
   // Create list of pointers to base class(es) for TClass cl.
   R__LOCKGUARD2(gInterpreterMutex);
   if (cl->fBase) {
      return;
   }
   TClingClassInfo *tci = (TClingClassInfo *)cl->GetClassInfo();
   if (!tci) return;
   TClingBaseClassInfo t(fInterpreter, tci);
   // This is put here since TClingBaseClassInfo can trigger a
   // TClass::ResetCaches, which deallocates cl->fBase
   cl->fBase = new TList;
   while (t.Next()) {
      // if name cannot be obtained no use to put in list
      if (t.IsValid() && t.Name()) {
         TClingBaseClassInfo *a = new TClingBaseClassInfo(t);
         cl->fBase->Add(new TBaseClass((BaseClassInfo_t *)a, cl));
      }
   }
}

//______________________________________________________________________________
void TCling::LoadEnums(TListOfEnums& enumList) const
{
   // Create list of pointers to enums for TClass cl.
   R__LOCKGUARD2(gInterpreterMutex);

   const Decl * D;
   TClass* cl = enumList.GetClass();
   if (cl) {
      D = ((TClingClassInfo*)cl->GetClassInfo())->GetDecl();
   }
   else {
      D = fInterpreter->getCI()->getASTContext().getTranslationUnitDecl();
   }
   // Iterate on the decl of the class and get the enums.
   if (const clang::DeclContext* DC = dyn_cast<clang::DeclContext>(D)) {
      cling::Interpreter::PushTransactionRAII deserRAII(fInterpreter);
      // Collect all contexts of the namespace.
      llvm::SmallVector< DeclContext *, 4> allDeclContexts;
      const_cast< clang::DeclContext *>(DC)->collectAllContexts(allDeclContexts);
      for (llvm::SmallVector<DeclContext*, 4>::iterator declIter = allDeclContexts.begin(), declEnd = allDeclContexts.end();
           declIter != declEnd; ++declIter) {
         // Iterate on all decls for each context.
         for (clang::DeclContext::decl_iterator DI = (*declIter)->decls_begin(),
              DE = (*declIter)->decls_end(); DI != DE; ++DI) {
            if (const clang::EnumDecl* ED = dyn_cast<clang::EnumDecl>(*DI)) {
               // Get name of the enum type.
               std::string buf;
               PrintingPolicy Policy(ED->getASTContext().getPrintingPolicy());
               llvm::raw_string_ostream stream(buf);
               ED->getNameForDiagnostic(stream, Policy, /*Qualified=*/false);
               stream.flush();
               // If the enum is unnamed we do not add it to the list of enums i.e unusable.
               if (!buf.empty()) {
                  const char* name = buf.c_str();
                  // Add the enum to the list of loaded enums.
                  enumList.Get(ED, name);
               }
            }
         }
      }
   }
}

//______________________________________________________________________________
void TCling::LoadFunctionTemplates(TClass* cl) const
{
   // Create list of pointers to function templates for TClass cl.
   R__LOCKGUARD2(gInterpreterMutex);

   const Decl * D;
   TListOfFunctionTemplates* funcTempList;
   if (cl) {
      D = ((TClingClassInfo*)cl->GetClassInfo())->GetDecl();
      funcTempList = (TListOfFunctionTemplates*)cl->GetListOfFunctionTemplates(false);
   }
   else {
      D = fInterpreter->getCI()->getASTContext().getTranslationUnitDecl();
      funcTempList = (TListOfFunctionTemplates*)gROOT->GetListOfFunctionTemplates();
   }
   // Iterate on the decl of the class and get the enums.
   if (const clang::DeclContext* DC = dyn_cast<clang::DeclContext>(D)) {
      cling::Interpreter::PushTransactionRAII deserRAII(fInterpreter);
      // Collect all contexts of the namespace.
      llvm::SmallVector< DeclContext *, 4> allDeclContexts;
      const_cast< clang::DeclContext *>(DC)->collectAllContexts(allDeclContexts);
      for (llvm::SmallVector<DeclContext*, 4>::iterator declIter = allDeclContexts.begin(),
           declEnd = allDeclContexts.end(); declIter != declEnd; ++declIter) {
         // Iterate on all decls for each context.
         for (clang::DeclContext::decl_iterator DI = (*declIter)->decls_begin(),
              DE = (*declIter)->decls_end(); DI != DE; ++DI) {
            if (const clang::FunctionTemplateDecl* FTD = dyn_cast<clang::FunctionTemplateDecl>(*DI)) {
                  funcTempList->Get(FTD);
            }
         }
      }
   }
}
//______________________________________________________________________________
void TCling::CreateListOfDataMembers(TClass* cl) const
{
   // Create list of pointers to data members for TClass cl.
   // This is now a nop.  The creation and updating is handled in
   // TListOfDataMembers.

}

//______________________________________________________________________________
void TCling::CreateListOfMethods(TClass* cl) const
{
   // Create list of pointers to methods for TClass cl.
   // This is now a nop.  The creation and updating is handled in
   // TListOfFunctions.

}

//______________________________________________________________________________
void TCling::UpdateListOfMethods(TClass* cl) const
{
   // Update the list of pointers to method for TClass cl
   // This is now a nop.  The creation and updating is handled in
   // TListOfFunctions.
}

//______________________________________________________________________________
void TCling::UpdateListOfDataMembers(TClass* cl) const
{
   // Update the list of pointers to data members for TClass cl
   // This is now a nop.  The creation and updating is handled in
   // TListOfDataMembers.
}

//______________________________________________________________________________
void TCling::CreateListOfMethodArgs(TFunction* m) const
{
   // Create list of pointers to method arguments for TMethod m.
   R__LOCKGUARD2(gInterpreterMutex);
   if (m->fMethodArgs) {
      return;
   }
   m->fMethodArgs = new TList;
   TClingMethodArgInfo t(fInterpreter, (TClingMethodInfo*)m->fInfo);
   while (t.Next()) {
      if (t.IsValid()) {
         TClingMethodArgInfo* a = new TClingMethodArgInfo(t);
         m->fMethodArgs->Add(new TMethodArg((MethodArgInfo_t*)a, m));
      }
   }
}


//______________________________________________________________________________
TClass *TCling::GenerateTClass(const char *classname, Bool_t emulation, Bool_t silent /* = kFALSE */)
{
   // Generate a TClass for the given class.
   // Since the caller has already check the ClassInfo, let it give use the
   // result (via the value of emulation) rather than recalculate it.

// For now the following line would lead to the (unwanted) instantiation
// of class template.  This could/would need to be resurrected only if
// we re-introduce so sort of automatic instantiation.   However this would
// have to include carefull look at the template parameter to avoid
// creating instance we can not really use (if the parameter are only forward
// declaration or do not have all the necessary interfaces).

   //   TClingClassInfo tci(fInterpreter, classname);
   //   if (1 || !tci.IsValid()) {

   Version_t version = 1;
   if (TClassEdit::IsSTLCont(classname)) {
      version = TClass::GetClass("TVirtualStreamerInfo")->GetClassVersion();
   }
   TClass *cl = new TClass(classname, version, silent);
   if (emulation) {
      cl->SetBit(TClass::kIsEmulation);
   } else {
      // Set the class version if the class is versioned.
      // Note that we cannot just call CLASS::Class_Version() as we might not have
      // an execution engine (when invoked from rootcling).

      // Do not call cl->GetClassVersion(), it has side effects!
      Version_t oldvers = cl->fClassVersion;
      if (oldvers == version && cl->GetClassInfo()) {
         // We have a version and it might need an update.
         Version_t newvers = oldvers;
         TClingClassInfo* cli = (TClingClassInfo*)cl->GetClassInfo();
         if (llvm::isa<clang::NamespaceDecl>(cli->GetDecl())) {
            // Namespaces don't have class versions.
            return cl;
         }
         TClingMethodInfo mi = cli->GetMethod("Class_Version", "", 0 /*poffset*/,
                                              ROOT::kExactMatch,
                                              TClingClassInfo::kInThisScope);
         if (!mi.IsValid()) {
            if (cl->TestBit(TClass::kIsTObject)) {
               Error("GenerateTClass",
                     "Cannot find %s::Class_Version()! Class version might be wrong.",
                     cl->GetName());
            }
            return cl;
         }
         newvers = ROOT::TMetaUtils::GetClassVersion(llvm::dyn_cast<clang::RecordDecl>(cli->GetDecl()),
                                                     *fInterpreter);
         if (newvers == -1) {
            // Didn't manage to determine the class version from the AST.
            // Use runtime instead.
            if ((mi.Property() & kIsStatic)
                && !fInterpreter->isInSyntaxOnlyMode()) {
               // This better be a static function.
               TClingCallFunc callfunc(fInterpreter, *fNormalizedCtxt);
               callfunc.SetFunc(&mi);
               newvers = callfunc.ExecInt(0);
            } else {
               Error("GenerateTClass",
                     "Cannot invoke %s::Class_Version()! Class version might be wrong.",
                     cl->GetName());
            }
         }
         if (newvers != oldvers) {
            cl->fClassVersion = newvers;
            cl->fStreamerInfo->Expand(newvers + 2 + 10);
         }
      }
   }

   return cl;

//   } else {
//      return GenerateTClass(&tci,silent);
//   }
}

#if 0
//______________________________________________________________________________
static void GenerateTClass_GatherInnerIncludes(cling::Interpreter *interp, TString &includes,TClingClassInfo *info)
{
   includes += info->FileName();

   const clang::ClassTemplateSpecializationDecl *templateCl
      = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(info->GetDecl());
   if (templateCl) {
      for(unsigned int i=0; i <  templateCl->getTemplateArgs().size(); ++i) {
          const clang::TemplateArgument &arg( templateCl->getTemplateArgs().get(i) );
          if (arg.getKind() == clang::TemplateArgument::Type) {
             const clang::Type *uType = ROOT::TMetaUtils::GetUnderlyingType( arg.getAsType() );

            if (!uType->isFundamentalType() && !uType->isEnumeralType()) {
               // We really need a header file.
               const clang::CXXRecordDecl *argdecl = uType->getAsCXXRecordDecl();
               if (argdecl) {
                  includes += ";";
                  TClingClassInfo subinfo(interp,*(argdecl->getASTContext().getRecordType(argdecl).getTypePtr()));
                  GenerateTClass_GatherInnerIncludes(interp, includes, &subinfo);
               } else {
                  std::string Result;
                  llvm::raw_string_ostream OS(Result);
                  arg.print(argdecl->getASTContext().getPrintingPolicy(),OS);
                  Warning("TCling::GenerateTClass","Missing header file for %s",OS.str().c_str());
               }
            }
          }
      }
   }
}
#endif

//______________________________________________________________________________
TClass *TCling::GenerateTClass(ClassInfo_t *classinfo, Bool_t silent /* = kFALSE */)
{
   // Generate a TClass for the given class.

   TClingClassInfo *info = (TClingClassInfo*)classinfo;
   if (!info || !info->IsValid()) {
      Fatal("GenerateTClass","Requires a valid ClassInfo object");
      return 0;
   }
   // We are in the case where we have AST nodes for this class.
   TClass *cl = 0;
   std::string classname;
   info->FullName(classname,*fNormalizedCtxt); // Could we use Name()?
   if (TClassEdit::IsSTLCont(classname.c_str())) {
#if 0
      Info("GenerateTClass","Will (try to) generate the compiled TClass for %s.",classname.c_str());
      // We need to build up the list of required headers, by
      // looking at each template arguments.
      TString includes;
      GenerateTClass_GatherInnerIncludes(fInterpreter,includes,info);

      if (0 == GenerateDictionary(classname.c_str(),includes)) {
         // 0 means success.
         cl = TClass::LoadClass(classnam.c_str(), silent);
         if (cl == 0) {
            Error("GenerateTClass","Even though the dictionary generation for %s seemed successfull we can't find the TClass bootstrap!",classname.c_str());
         }
      }
#endif
      if (cl == 0) {
         int version = TClass::GetClass("TVirtualStreamerInfo")->GetClassVersion();
         cl = new TClass(classinfo, version, 0, 0, -1, -1, silent);
         cl->SetBit(TClass::kIsEmulation);
      }
   } else {
      // For regular class, just create a TClass on the fly ...
      // Not quite useful yet, but that what CINT used to do anyway.
      cl = new TClass(classinfo, 1, 0, 0, -1, -1, silent);
   }
   // Add the new TClass to the map of declid and TClass*.
   if (cl) {
      TClass::AddClassToDeclIdMap(((TClingClassInfo*)classinfo)->GetDeclId(), cl);
   }
   return cl;
}

//______________________________________________________________________________
Int_t TCling::GenerateDictionary(const char* classes, const char* includes /* = 0 */, const char* /* options  = 0 */)
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
   return TCling_GenerateDictionary(listClasses, listIncludes,
      std::vector<std::string>(), std::vector<std::string>());
}

//______________________________________________________________________________
TInterpreter::DeclId_t TCling::GetDataMember(ClassInfo_t *opaque_cl, const char *name) const
{
   // Return pointer to cling Decl of global/static variable that is located
   // at the address given by addr.

   R__LOCKGUARD2(gInterpreterMutex);
   DeclId_t d;
   TClingClassInfo *cl = (TClingClassInfo*)opaque_cl;

   if (cl) {
      d = cl->GetDataMember(name);
      // We check if the decl of the data member has an annotation which indicates
      // an ioname.
      // In case this is true, if the name requested is not the ioname, we
      // return 0, as if the member did not exist. In some sense we override
      // the information in the TClassInfo instance, isolating the typesystem in
      // TClass from the one in the AST.
      if (const ValueDecl* decl = (const ValueDecl*) d){
         std::string ioName;
         bool hasIoName = ROOT::TMetaUtils::ExtractAttrPropertyFromName(*decl,"ioname",ioName);
         if (hasIoName && ioName != name) return 0;
      }
   }
   else {
      TClingClassInfo gcl(fInterpreter);
      d = gcl.GetDataMember(name);
   }
   return d;
}

//______________________________________________________________________________
TInterpreter::DeclId_t TCling::GetEnum(TClass *cl, const char *name) const
{
   // Return pointer to cling Decl of global/static variable that is located
   // at the address given by addr.

   R__LOCKGUARD2(gInterpreterMutex);

   const clang::Decl* possibleEnum = 0;
   // FInd the context of the decl.
   if (cl) {
      TClingClassInfo *cci = (TClingClassInfo*)cl->GetClassInfo();
      if (cci) {
         const clang::DeclContext* dc = 0;
         if (const clang::Decl* D = cci->GetDecl()) {
            if (!(dc = dyn_cast<clang::NamespaceDecl>(D))) {
               dc = dyn_cast<clang::RecordDecl>(D);
            }
         }
         if (dc) {
            // If it is a data member enum.
            possibleEnum = cling::utils::Lookup::Named(&fInterpreter->getSema(), name, dc);
         } else {
            Error("TCling::GetEnum", "DeclContext not found for %s .\n", name);
         }
      }
   } else {
      // If it is a global enum.
      possibleEnum = cling::utils::Lookup::Named(&fInterpreter->getSema(), name);
   }
   if (possibleEnum && (possibleEnum != (clang::Decl*)-1)
       && isa<clang::EnumDecl>(possibleEnum)) {
      return possibleEnum;
   }
   return 0;
}

//______________________________________________________________________________
TInterpreter::DeclId_t TCling::GetDeclId( const llvm::GlobalValue *gv ) const
{
   // Return pointer to cling DeclId for a global value

   if (!gv) return 0;

   llvm::StringRef mangled_name = gv->getName();

   int err = 0;
   char* demangled_name_c = TClassEdit::DemangleName(mangled_name.str().c_str(), err);
   if (err) {
      if (err == -2) {
         // It might simply be an unmangled global name.
         DeclId_t d;
         TClingClassInfo gcl(fInterpreter);
         d = gcl.GetDataMember(mangled_name.str().c_str());
         return d;
      }
      return 0;
   }

   std::string scopename(demangled_name_c);
   free(demangled_name_c);

   //
   //  Separate out the class or namespace part of the
   //  function name.
   //
   std::string dataname;

   if (!strncmp(scopename.c_str(), "typeinfo for ", sizeof("typeinfo for ")-1)) {
      scopename.erase(0, sizeof("typeinfo for ")-1);
   } if (!strncmp(scopename.c_str(), "vtable for ", sizeof("vtable for ")-1)) {
      scopename.erase(0, sizeof("vtable for ")-1);
   } else {
      // See if it is a function
      std::string::size_type pos = scopename.rfind('(');
      if (pos != std::string::npos) {
         return 0;
      }
      // Separate the scope and member name
      pos = scopename.rfind(':');
      if (pos != std::string::npos) {
         if ((pos != 0) && (scopename[pos-1] == ':')) {
            dataname = scopename.substr(pos+1);
            scopename.erase(pos-1);
         }
      } else {
         scopename.clear();
         dataname = scopename;
      }
   }
   //fprintf(stderr, "name: '%s'\n", name.c_str());
   // Now we have the class or namespace name, so do the lookup.


   DeclId_t d;
   if (scopename.size()) {
      TClingClassInfo cl(fInterpreter,scopename.c_str());
      d = cl.GetDataMember(dataname.c_str());
   }
   else {
      TClingClassInfo gcl(fInterpreter);
      d = gcl.GetDataMember(dataname.c_str());
   }
   return d;
}

//______________________________________________________________________________
TInterpreter::DeclId_t TCling::GetDataMemberWithValue(const void *ptrvalue) const
{
   // NOT IMPLEMENTED.
   Error("GetDataMemberWithValue()", "not implemented");
   return 0;
}

//______________________________________________________________________________
TInterpreter::DeclId_t TCling::GetDataMemberAtAddr(const void *addr) const
{
   // Return pointer to cling DeclId for a data member with a given name.

   // NOT IMPLEMENTED.
   Error("GetDataMemberAtAddr()", "not implemented");
   return 0;
}

//______________________________________________________________________________
TString TCling::GetMangledName(TClass* cl, const char* method,
                               const char* params, Bool_t objectIsConst /* = kFALSE */)
{
   // Return the cling mangled name for a method of a class with parameters
   // params (params is a string of actual arguments, not formal ones). If the
   // class is 0 the global function list will be searched.
   R__LOCKGUARD2(gInterpreterMutex);
   TClingCallFunc func(fInterpreter,*fNormalizedCtxt);
   if (cl) {
      Long_t offset;
      func.SetFunc((TClingClassInfo*)cl->GetClassInfo(), method, params, objectIsConst,
         &offset);
   }
   else {
      TClingClassInfo gcl(fInterpreter);
      Long_t offset;
      func.SetFunc(&gcl, method, params, &offset);
   }
   TClingMethodInfo* mi = (TClingMethodInfo*) func.FactoryMethod();
   if (!mi) return "";
   TString mangled_name( mi->GetMangledName() );
   delete mi;
   return mangled_name;
}

//______________________________________________________________________________
TString TCling::GetMangledNameWithPrototype(TClass* cl, const char* method,
                                            const char* proto, Bool_t objectIsConst /* = kFALSE */,
                                            EFunctionMatchMode mode /* = kConversionMatch */)
{
   // Return the cling mangled name for a method of a class with a certain
   // prototype, i.e. "char*,int,float". If the class is 0 the global function
   // list will be searched.
   R__LOCKGUARD2(gInterpreterMutex);
   if (cl) {
      return ((TClingClassInfo*)cl->GetClassInfo())->
         GetMethod(method, proto, objectIsConst, 0 /*poffset*/, mode).GetMangledName();
   }
   TClingClassInfo gcl(fInterpreter);
   return gcl.GetMethod(method, proto, objectIsConst, 0 /*poffset*/, mode).GetMangledName();
}

//______________________________________________________________________________
void* TCling::GetInterfaceMethod(TClass* cl, const char* method,
                                 const char* params, Bool_t objectIsConst /* = kFALSE */)
{
   // Return pointer to cling interface function for a method of a class with
   // parameters params (params is a string of actual arguments, not formal
   // ones). If the class is 0 the global function list will be searched.
   R__LOCKGUARD2(gInterpreterMutex);
   TClingCallFunc func(fInterpreter,*fNormalizedCtxt);
   if (cl) {
      Long_t offset;
      func.SetFunc((TClingClassInfo*)cl->GetClassInfo(), method, params, objectIsConst,
                   &offset);
   }
   else {
      TClingClassInfo gcl(fInterpreter);
      Long_t offset;
      func.SetFunc(&gcl, method, params, &offset);
   }
   return (void*) func.InterfaceMethod();
}

//______________________________________________________________________________
TInterpreter::DeclId_t TCling::GetFunction(ClassInfo_t *opaque_cl, const char* method)
{
   // Return pointer to cling interface function for a method of a class with
   // a certain name.

   R__LOCKGUARD2(gInterpreterMutex);
   DeclId_t f;
   TClingClassInfo *cl = (TClingClassInfo*)opaque_cl;
   if (cl) {
      f = cl->GetMethod(method).GetDeclId();
   }
   else {
      TClingClassInfo gcl(fInterpreter);
      f = gcl.GetMethod(method).GetDeclId();
   }
   return f;

}

//______________________________________________________________________________
void TCling::GetFunctionOverloads(ClassInfo_t *cl, const char *funcname,
                                  std::vector<DeclId_t>& res) const
{
   // Insert overloads of name in cl to res.
   clang::Sema& S = fInterpreter->getSema();
   const clang::Decl* CtxDecl
      = cl ? (const clang::Decl*)((TClingClassInfo*)cl)->GetDeclId():
      S.Context.getTranslationUnitDecl();
   const clang::DeclContext*
      DeclCtx = llvm::dyn_cast<const clang::RecordDecl>(CtxDecl);
   if (!DeclCtx)
      DeclCtx = dyn_cast<clang::NamespaceDecl>(CtxDecl);
   if (!DeclCtx) return;
   clang::DeclarationName DName
      = &S.Context.Idents.get(funcname);
   clang::LookupResult R(S, DName, clang::SourceLocation(),
                         Sema::LookupOrdinaryName, clang::Sema::ForRedeclaration);
   S.LookupQualifiedName(R, const_cast<DeclContext*>(DeclCtx));
   if (R.empty()) return;
   R.resolveKind();
   res.reserve(res.size() + (R.end() - R.begin()));
   for (clang::LookupResult::iterator IR = R.begin(), ER = R.end();
        IR != ER; ++IR) {
      if (const clang::FunctionDecl* FD
          = llvm::dyn_cast<const clang::FunctionDecl>(*IR)) {
         if (!FD->getDescribedFunctionTemplate()) {
            res.push_back(FD);
         }
      }
   }
}

//______________________________________________________________________________
void* TCling::GetInterfaceMethodWithPrototype(TClass* cl, const char* method,
                                              const char* proto,
                                              Bool_t objectIsConst /* = kFALSE */,
                                              EFunctionMatchMode mode /* = kConversionMatch */)
{
   // Return pointer to cling interface function for a method of a class with
   // a certain prototype, i.e. "char*,int,float". If the class is 0 the global
   // function list will be searched.
   R__LOCKGUARD2(gInterpreterMutex);
   void* f;
   if (cl) {
      f = ((TClingClassInfo*)cl->GetClassInfo())->
         GetMethod(method, proto, objectIsConst, 0 /*poffset*/, mode).InterfaceMethod(*fNormalizedCtxt);
   }
   else {
      TClingClassInfo gcl(fInterpreter);
      f = gcl.GetMethod(method, proto, objectIsConst, 0 /*poffset*/, mode).InterfaceMethod(*fNormalizedCtxt);
   }
   return f;
}

//______________________________________________________________________________
TInterpreter::DeclId_t TCling::GetFunctionWithValues(ClassInfo_t *opaque_cl, const char* method,
                                                     const char* params,
                                                     Bool_t objectIsConst /* = kFALSE */)
{
   // Return pointer to cling DeclId for a method of a class with
   // a certain prototype, i.e. "char*,int,float". If the class is 0 the global
   // function list will be searched.
   R__LOCKGUARD2(gInterpreterMutex);
   DeclId_t f;
   TClingClassInfo *cl = (TClingClassInfo*)opaque_cl;
   if (cl) {
      f = cl->GetMethodWithArgs(method, params, objectIsConst, 0 /*poffset*/).GetDeclId();
   }
   else {
      TClingClassInfo gcl(fInterpreter);
      f = gcl.GetMethod(method, params, objectIsConst, 0 /*poffset*/).GetDeclId();
   }
   return f;
}

//______________________________________________________________________________
TInterpreter::DeclId_t TCling::GetFunctionWithPrototype(ClassInfo_t *opaque_cl, const char* method,
                                                        const char* proto,
                                                        Bool_t objectIsConst /* = kFALSE */,
                                                        EFunctionMatchMode mode /* = kConversionMatch */)
{
   // Return pointer to cling interface function for a method of a class with
   // a certain prototype, i.e. "char*,int,float". If the class is 0 the global
   // function list will be searched.
   R__LOCKGUARD2(gInterpreterMutex);
   DeclId_t f;
   TClingClassInfo *cl = (TClingClassInfo*)opaque_cl;
   if (cl) {
      f = cl->GetMethod(method, proto, objectIsConst, 0 /*poffset*/, mode).GetDeclId();
   }
   else {
      TClingClassInfo gcl(fInterpreter);
      f = gcl.GetMethod(method, proto, objectIsConst, 0 /*poffset*/, mode).GetDeclId();
   }
   return f;
}

//______________________________________________________________________________
TInterpreter::DeclId_t TCling::GetFunctionTemplate(ClassInfo_t *opaque_cl, const char* name)
{
   // Return pointer to cling interface function for a method of a class with
   // a certain name.

   R__LOCKGUARD2(gInterpreterMutex);
   DeclId_t f;
   TClingClassInfo *cl = (TClingClassInfo*)opaque_cl;
   if (cl) {
      f = cl->GetFunctionTemplate(name);
   }
   else {
      TClingClassInfo gcl(fInterpreter);
      f = gcl.GetFunctionTemplate(name);
   }
   return f;

}

//______________________________________________________________________________
void TCling::GetInterpreterTypeName(const char* name, std::string &output, Bool_t full)
{
   // The 'name' is known to the interpreter, this function returns
   // the internal version of this name (usually just resolving typedefs)
   // This is used in particular to synchronize between the name used
   // by rootcling and by the run-time enviroment (TClass)
   // Return 0 if the name is not known.

   output.clear();

   R__LOCKGUARD(gInterpreterMutex);

   TClingClassInfo cl(fInterpreter, name);
   if (!cl.IsValid()) {
      return ;
   }
   if (full) {
      cl.FullName(output,*fNormalizedCtxt);
      return;
   }
   // Well well well, for backward compatibility we need to act a bit too
   // much like CINT.
   TClassEdit::TSplitType splitname( cl.Name(), TClassEdit::kDropStd );
   splitname.ShortType(output, TClassEdit::kDropStd );

   return;
}

//______________________________________________________________________________
void TCling::Execute(const char* function, const char* params, int* error)
{
   // Execute a global function with arguments params.
   //
   // FIXME: The cint-based version of this code does not check if the
   //        SetFunc() call works, and does not do any real checking
   //        for errors from the Exec() call.  It did fetch the most
   //        recent cint security error and return that in error, but
   //        this does not really translate well to cling/clang.  We
   //        should enhance these interfaces so that we can report
   //        compilation and runtime errors properly.
   //
   R__LOCKGUARD2(gInterpreterMutex);
   if (error) {
      *error = TInterpreter::kNoError;
   }
   TClingClassInfo cl(fInterpreter);
   Long_t offset = 0L;
   TClingCallFunc func(fInterpreter,*fNormalizedCtxt);
   func.SetFunc(&cl, function, params, &offset);
   func.Exec(0);
}

//______________________________________________________________________________
void TCling::Execute(TObject* obj, TClass* cl, const char* method,
                     const char* params, Bool_t objectIsConst, int* error)
{
   // Execute a method from class cl with arguments params.
   //
   // FIXME: The cint-based version of this code does not check if the
   //        SetFunc() call works, and does not do any real checking
   //        for errors from the Exec() call.  It did fetch the most
   //        recent cint security error and return that in error, but
   //        this does not really translate well to cling/clang.  We
   //        should enhance these interfaces so that we can report
   //        compilation and runtime errors properly.
   //
   R__LOCKGUARD2(gInterpreterMutex);
   if (error) {
      *error = TInterpreter::kNoError;
   }
   // If the actual class of this object inherits 2nd (or more) from TObject,
   // 'obj' is unlikely to be the start of the object (as described by IsA()),
   // hence gInterpreter->Execute will improperly correct the offset.
   void* addr = cl->DynamicCast(TObject::Class(), obj, kFALSE);
   Long_t offset = 0L;
   TClingCallFunc func(fInterpreter,*fNormalizedCtxt);
   func.SetFunc((TClingClassInfo*)cl->GetClassInfo(), method, params, objectIsConst, &offset);
   void* address = (void*)((Long_t)addr + offset);
   func.Exec(address);
}

//______________________________________________________________________________
void TCling::Execute(TObject* obj, TClass* cl, const char* method,
                    const char* params, int* error)
{
   Execute(obj,cl,method,params,false,error);
}
//______________________________________________________________________________
void TCling::Execute(TObject* obj, TClass* cl, TMethod* method,
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
   Int_t argc   = params ? params->GetEntries() : 0;

   if (argc > nparms) {
      Error("Execute","Too many parameters to call %s, got %d but expected at most %d.",method->GetName(),argc,nparms);
      return;
   }
   if (nparms != argc) {
     // Let's see if the 'missing' argument are all defaulted.
     // if nparms==0 then either we stopped earlier either argc is also zero and we can't reach here.
     assert(nparms > 0);

     TMethodArg *arg = (TMethodArg *) argList->At( 0 );
     if (arg && arg->GetDefault() && arg->GetDefault()[0]) {
        // There is a default value for the first missing
        // argument, so we are fine.
     } else {
        Int_t firstDefault = -1;
        for (Int_t i = 0; i < nparms; i ++) {
           arg = (TMethodArg *) argList->At( i );
           if (arg && arg->GetDefault() && arg->GetDefault()[0]) {
              firstDefault = i;
              break;
           }
        }
        if (firstDefault >= 0) {
           Error("Execute","Too few arguments to call %s, got only %d but expected at least %d and at most %d.",method->GetName(),argc,firstDefault,nparms);
        } else {
           Error("Execute","Too few arguments to call %s, got only %d but expected %d.",method->GetName(),argc,nparms);
        }
        return;
     }
   }

   const char* listpar = "";
   TString complete(10);
   if (params) {
      // Create a character string of parameters from TObjArray
      TIter next(params);
      for (Int_t i = 0; i < argc; i ++) {
         TMethodArg* arg = (TMethodArg*) argList->At(i);
         TClingTypeInfo type(fInterpreter, arg->GetFullTypeName());
         TObjString* nxtpar = (TObjString*) next();
         if (i) {
            complete += ',';
         }
         if (strstr(type.TrueName(*fNormalizedCtxt), "char")) {
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

   // And now execute it.
   R__LOCKGUARD2(gInterpreterMutex);
   if (error) {
      *error = TInterpreter::kNoError;
   }
   // If the actual class of this object inherits 2nd (or more) from TObject,
   // 'obj' is unlikely to be the start of the object (as described by IsA()),
   // hence gInterpreter->Execute will improperly correct the offset.
   void* addr = cl->DynamicCast(TObject::Class(), obj, kFALSE);
   TClingCallFunc func(fInterpreter,*fNormalizedCtxt);
   TClingMethodInfo *minfo = (TClingMethodInfo*)method->fInfo;
   func.Init(minfo);
   func.SetArgs(listpar);
   // Now calculate the 'this' pointer offset for the method
   // when starting from the class described by cl.
   const CXXMethodDecl * mdecl = dyn_cast<CXXMethodDecl>(minfo->GetMethodDecl());
   Long_t offset = ((TClingClassInfo*)cl->GetClassInfo())->GetOffset(mdecl);
   void* address = (void*)((Long_t)addr + offset);
   func.Exec(address);
}

//______________________________________________________________________________
void TCling::ExecuteWithArgsAndReturn(TMethod* method, void* address,
                                      const void* args[] /*=0*/,
                                      int nargs /*=0*/,
                                      void* ret/*= 0*/) const
{
   if (!method) {
      Error("ExecuteWithArgsAndReturn", "No method was defined");
      return;
   }

   TClingMethodInfo* minfo = (TClingMethodInfo*) method->fInfo;
   TClingCallFunc func(*minfo,*fNormalizedCtxt);
   func.ExecWithArgsAndReturn(address, args, nargs, ret);
}

//______________________________________________________________________________
Long_t TCling::ExecuteMacro(const char* filename, EErrorCode* error)
{
   // Execute a cling macro.
   R__LOCKGUARD(fLockProcessLine ? gInterpreterMutex : 0);
   fCurExecutingMacros.push_back(filename);
   Long_t result = TApplication::ExecuteFile(filename, (int*)error);
   fCurExecutingMacros.pop_back();
   return result;
}

//______________________________________________________________________________
const char* TCling::GetTopLevelMacroName() const
{
   // Return the file name of the current un-included interpreted file.
   // See the documentation for GetCurrentMacroName().

   Warning("GetTopLevelMacroName", "Must change return type!");
   return fCurExecutingMacros.back();
}

//______________________________________________________________________________
const char* TCling::GetCurrentMacroName() const
{
   // Return the file name of the currently interpreted file,
   // included or not. Example to illustrate the difference between
   // GetCurrentMacroName() and GetTopLevelMacroName():
   // BEGIN_HTML <!--
   /* -->
      <span style="color:#ffffff;background-color:#7777ff;padding-left:0.3em;padding-right:0.3em">inclfile.C</span>
      <!--div style="border:solid 1px #ffff77;background-color: #ffffdd;float:left;padding:0.5em;margin-bottom:0.7em;"-->
      <div class="code">
      <pre style="margin:0pt">#include &lt;iostream&gt;
   void inclfile() {
   std::cout &lt;&lt; "In inclfile.C" &lt;&lt; std::endl;
   std::cout &lt;&lt; "  TCling::GetCurrentMacroName() returns  " &lt;&lt;
      TCling::GetCurrentMacroName() &lt;&lt; std::endl;
   std::cout &lt;&lt; "  TCling::GetTopLevelMacroName() returns " &lt;&lt;
      TCling::GetTopLevelMacroName() &lt;&lt; std::endl;
   }</pre></div>
      <div style="clear:both"></div>
      <span style="color:#ffffff;background-color:#7777ff;padding-left:0.3em;padding-right:0.3em">mymacro.C</span>
      <div style="border:solid 1px #ffff77;background-color: #ffffdd;float:left;padding:0.5em;margin-bottom:0.7em;">
      <pre style="margin:0pt">#include &lt;iostream&gt;
   void mymacro() {
   std::cout &lt;&lt; "In mymacro.C" &lt;&lt; std::endl;
   std::cout &lt;&lt; "  TCling::GetCurrentMacroName() returns  " &lt;&lt;
      TCling::GetCurrentMacroName() &lt;&lt; std::endl;
   std::cout &lt;&lt; "  TCling::GetTopLevelMacroName() returns " &lt;&lt;
      TCling::GetTopLevelMacroName() &lt;&lt; std::endl;
   std::cout &lt;&lt; "  Now calling inclfile..." &lt;&lt; std::endl;
   gInterpreter->ProcessLine(".x inclfile.C");;
   }</pre></div>
   <div style="clear:both"></div>
   <!-- */
   // --> END_HTML
   // Running mymacro.C will print:
   //
   // root [0] .x mymacro.C
   // In mymacro.C
   //   TCling::GetCurrentMacroName() returns  ./mymacro.C
   //   TCling::GetTopLevelMacroName() returns ./mymacro.C
   //   Now calling inclfile...
   // In inclfile.h
   //   TCling::GetCurrentMacroName() returns  inclfile.C
   //   TCling::GetTopLevelMacroName() returns ./mymacro.C

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,0)
   Warning("GetCurrentMacroName", "Must change return type!");
#endif
#endif
   return fCurExecutingMacros.back();
}

//______________________________________________________________________________
const char* TCling::TypeName(const char* typeDesc)
{
   // Return the absolute type of typeDesc.
   // E.g.: typeDesc = "class TNamed**", returns "TNamed".
   // You need to use the result immediately before it is being overwritten.
   TTHREAD_TLS(char*) t = 0;
   TTHREAD_TLS(unsigned int) tlen = 0;

   unsigned int dlen = strlen(typeDesc);
   if (dlen > tlen) {
      delete[] t;
      t = new char[dlen + 1];
      tlen = dlen;
   }
   const char* s, *template_start;
   if (!strstr(typeDesc, "(*)(")) {
      s = strchr(typeDesc, ' ');
      template_start = strchr(typeDesc, '<');
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
int TCling::ReadRootmapFile(const char *rootmapfile, TUniqueString *uniqueString)
{
   // Read and parse a rootmapfile in its new format, and return 0 in case of
   // success, -1 if the file has already been read, and -3 in case its format
   // is the old one (e.g. containing "Library.ClassName")

   // For "class ", "namespace ", "typedef ", "header ", "enum " respectively
   const std::map<char, unsigned int> keyLenMap = {{'c',6},{'n',10},{'t',8},{'h',7},{'e',5}};

   if (rootmapfile && *rootmapfile) {

      // Add content of a specific rootmap file
      if (fRootmapFiles->FindObject(rootmapfile)) return -1;
      std::ifstream file(rootmapfile);
      std::string line; line.reserve(200);
      std::string lib_name; line.reserve(100);
      bool newFormat=false;
      while (getline(file, line, '\n')) {
         if (!newFormat &&
             (strstr(line.c_str(),"Library.")!=nullptr || strstr(line.c_str(),"Declare.")!=nullptr)) {
            file.close();
            return -3; // old format
         }
         newFormat=true;

         if (line.compare(0, 9, "{ decls }") == 0) {
            // forward declarations

            while (getline(file, line, '\n')) {
               if (line[0] == '[') break;
               uniqueString->Append(line);
            }
         }
         const char firstChar=line[0];
         if (firstChar == '[') {
            // new section (library)
            auto brpos = line.find(']');
            if (brpos == string::npos) continue;
            lib_name = line.substr(1, brpos-1);
            size_t nspaces = 0;
            while( lib_name[nspaces] == ' ' ) ++nspaces;
            if (nspaces) lib_name.replace(0, nspaces, "");
            if (gDebug > 3) {
               TString lib_nameTstr(lib_name.c_str());
               TObjArray* tokens = lib_nameTstr.Tokenize(" ");
               const char* lib = ((TObjString *)tokens->At(0))->GetName();
               const char* wlib = gSystem->DynamicPathName(lib, kTRUE);
               if (wlib) {
                  Info("ReadRootmapFile", "new section for %s", lib_nameTstr.Data());
               }
               else {
                  Info("ReadRootmapFile", "section for %s (library does not exist)", lib_nameTstr.Data());
               }
               delete[] wlib;
               delete tokens;
            }
         }
         else {
            auto keyLenIt = keyLenMap.find(firstChar);
            if (keyLenIt == keyLenMap.end()) continue;
            unsigned int keyLen = keyLenIt->second;
            // Do not make a copy, just start after the key
            const char *keyname = line.c_str()+keyLen;
            if (gDebug > 6)
               Info("ReadRootmapFile", "class %s in %s", keyname, lib_name.c_str());
            TEnvRec* isThere = fMapfile->Lookup(keyname);
            if (isThere){
               if(lib_name != isThere->GetValue()){ // the same key for two different libs
                  if (firstChar == 'n') {
                     if (gDebug > 3)
                        Info("ReadRootmapFile", "namespace %s found in %s is already in %s",
                           keyname, lib_name.c_str(), isThere->GetValue());
                  } else if (firstChar == 'h'){ // it is a header: add the libname to the list of libs to be loaded.
                     lib_name+=" ";
                     lib_name+=isThere->GetValue();
                     fMapfile->SetValue(keyname, lib_name.c_str());
                  }
                  else if (!TClassEdit::IsSTLCont(keyname)) {
                     Warning("ReadRootmapFile", "%s %s found in %s is already in %s", line.substr(0, keyLen).c_str(),
                           keyname, lib_name.c_str(), isThere->GetValue());
                  }
               } else { // the same key for the same lib
                  if (gDebug > 3)
                        Info("ReadRootmapFile","Key %s was already defined for %s", keyname, lib_name.c_str());
               }

            } else {
               fMapfile->SetValue(keyname, lib_name.c_str());
            }
         }
      }
      file.close();
   }

   return 0;
}

//______________________________________________________________________________
void TCling::InitRootmapFile(const char *name)
{
   // Create a resource table and read the (possibly) three resource files, i.e
   // $ROOTSYS/etc/system<name> (or ROOTETCDIR/system<name>), $HOME/<name> and
   // ./<name>. ROOT always reads ".rootrc" (in TROOT::InitSystem()). You can
   // read additional user defined resource files by creating addtional TEnv
   // objects. By setting the shell variable ROOTENV_NO_HOME=1 the reading of
   // the $HOME/<name> resource file will be skipped. This might be useful in
   // case the home directory resides on an automounted remote file system
   // and one wants to avoid the file system from being mounted.

   Bool_t ignore = fMapfile->IgnoreDuplicates(kFALSE);

   fMapfile->SetRcName(name);

   TString sname = "system";
   sname += name;
#ifdef ROOTETCDIR
   char *s = gSystem->ConcatFileName(ROOTETCDIR, sname);
#else
   TString etc = gRootDir;
#ifdef WIN32
   etc += "\\etc";
#else
   etc += "/etc";
#endif
#if defined(R__MACOSX) && (TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR)
   // on iOS etc does not exist and system<name> resides in $ROOTSYS
   etc = gRootDir;
#endif
   char *s = gSystem->ConcatFileName(etc, sname);
#endif

   Int_t ret = ReadRootmapFile(s);
   if (ret == -3) // old format
      fMapfile->ReadFile(s, kEnvGlobal);
   delete [] s;
   if (!gSystem->Getenv("ROOTENV_NO_HOME")) {
      s = gSystem->ConcatFileName(gSystem->HomeDirectory(), name);
      ret = ReadRootmapFile(s);
      if (ret == -3) // old format
         fMapfile->ReadFile(s, kEnvUser);
      delete [] s;
      if (strcmp(gSystem->HomeDirectory(), gSystem->WorkingDirectory())) {
         ret = ReadRootmapFile(name);
         if (ret == -3) // old format
            fMapfile->ReadFile(name, kEnvLocal);
      }
   } else {
      ret = ReadRootmapFile(name);
      if (ret == -3) // old format
         fMapfile->ReadFile(name, kEnvLocal);
   }
   fMapfile->IgnoreDuplicates(ignore);
}


namespace {
   using namespace clang;

   class ExtVisibleStorageAdder: public RecursiveASTVisitor<ExtVisibleStorageAdder>{
      // This class is to be considered an helper for autoloading.
      // It is a recursive visitor is used to inspect namespaces coming from
      // forward declarations in rootmaps and to set the external visible
      // storage flag for them.
   public:
      ExtVisibleStorageAdder(std::unordered_set<const NamespaceDecl*>& nsSet): fNSSet(nsSet) {};
      bool VisitNamespaceDecl(NamespaceDecl* nsDecl) {
         // We want to enable the external lookup for this namespace
         // because it may shadow the lookup of other names contained
         // in that namespace
         nsDecl->setHasExternalVisibleStorage();
         fNSSet.insert(nsDecl);
         return true;
      }
   private:
      std::unordered_set<const NamespaceDecl*>& fNSSet;

   };
}

//______________________________________________________________________________
Int_t TCling::LoadLibraryMap(const char* rootmapfile)
{
   // Load map between class and library. If rootmapfile is specified a
   // specific rootmap file can be added (typically used by ACLiC).
   // In case of error -1 is returned, 0 otherwise.
   // The interpreter uses this information to automatically load the shared
   // library for a class (autoload mechanism), see the AutoLoad() methods below.
   R__LOCKGUARD(gInterpreterMutex);
   // open the [system].rootmap files
   if (!fMapfile) {
      fMapfile = new TEnv();
      fMapfile->IgnoreDuplicates(kTRUE);
//       fMapNamespaces = new THashTable();
//       fMapNamespaces->SetOwner();
      fRootmapFiles = new TObjArray;
      fRootmapFiles->SetOwner();
      InitRootmapFile(".rootmap");
   }

   // Prepare a list of all forward declarations for cling
   // For some experiments it is easily as big as 500k characters. To be on the
   // safe side, we go for 1M.
   TUniqueString uniqueString(1048576);

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
         d = ((TObjString *)paths->At(i))->GetString();
         // check if directory already scanned
         Int_t skip = 0;
         for (Int_t j = 0; j < i; j++) {
            TString pd = ((TObjString *)paths->At(j))->GetString();
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
                           Int_t ret = ReadRootmapFile(p,&uniqueString);
                           if (ret == 0)
                              fRootmapFiles->Add(new TNamed(gSystem->BaseName(f), p.Data()));
                           if (ret == -3) {
                              // old format
                              fMapfile->ReadFile(p, kEnvGlobal);
                              fRootmapFiles->Add(new TNamed(f, p));
                           }
                        }
                        // else {
                        //    fprintf(stderr,"Reject %s because %s is already there\n",p.Data(),f.Data());
                        //    fRootmapFiles->FindObject(f)->ls();
                        // }
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
      Int_t res = ReadRootmapFile(rootmapfile, &uniqueString);
      if (res == 0) {
         //TString p = gSystem->ConcatFileName(gSystem->pwd(), rootmapfile);
         //fRootmapFiles->Add(new TNamed(gSystem->BaseName(rootmapfile), p.Data()));
         fRootmapFiles->Add(new TNamed(gSystem->BaseName(rootmapfile), rootmapfile));
      }
      else if (res == -3) {
         // old format
         Bool_t ignore = fMapfile->IgnoreDuplicates(kFALSE);
         fMapfile->ReadFile(rootmapfile, kEnvGlobal);
         fRootmapFiles->Add(new TNamed(gSystem->BaseName(rootmapfile), rootmapfile));
         fMapfile->IgnoreDuplicates(ignore);
      }
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
         // Fill in the namespace candidate list
//          Ssiz_t last = cls.Last(':');
//          if (last != kNPOS) {
//             // Please note that the funny op overlaod does substring.
//             TString namespaceCand = cls(0, last - 1);
//             // This is a reference to a substring that lives in fMapfile
//             if (!fMapNamespaces->FindObject(namespaceCand.Data()))
//                fMapNamespaces->Add(new TNamed(namespaceCand.Data(), ""));
//          }
         delete tokens;
      }
      else if (!strncmp(cls.Data(), "Declare.", 8) && cls.Length() > 8) {
         cls.Remove(0, 8);
         // convert "-" to " ", since class names may have
         // blanks and TEnv considers a blank a terminator
         cls.ReplaceAll("-", " ");
         fInterpreter->declare(cls.Data());
      }
   }

   // Process the forward declarations collected
   cling::Transaction* T = nullptr;
   auto compRes= fInterpreter->declare(uniqueString.Data(), &T);
   assert(cling::Interpreter::kSuccess == compRes && "A declaration in a rootmap could not be compiled");

   if (compRes!=cling::Interpreter::kSuccess){
      Warning("LoadLibraryMap",
               "Problems in %s declaring '%s' were encountered.", rootmapfile, uniqueString.Data()) ;
   }

   if (T){
      ExtVisibleStorageAdder evsAdder(fNSFromRootmaps);
      for (auto declIt = T->decls_begin(); declIt < T->decls_end(); ++declIt) {
         if (declIt->m_DGR.isSingleDecl()) {
            if (Decl* D = declIt->m_DGR.getSingleDecl()) {
               if (NamespaceDecl* NSD = dyn_cast<NamespaceDecl>(D)) {
                  evsAdder.TraverseDecl(NSD);
               }
            }
         }
      }
   }

   // clear duplicates

   return 0;
}

//______________________________________________________________________________
Int_t TCling::RescanLibraryMap()
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
Int_t TCling::ReloadAllSharedLibraryMaps()
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
Int_t TCling::UnloadAllSharedLibraryMaps()
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
Int_t TCling::UnloadLibraryMap(const char* library)
{
   // Unload library map entries coming from the specified library.
   // Returns -1 in case no entries for the specified library were found,
   // 0 otherwise.
   if (!fMapfile || !library || !*library) {
      return 0;
   }
   TString libname(library);
   Ssiz_t idx = libname.Last('.');
   if (idx != kNPOS) {
      libname.Remove(idx);
   }
   size_t len = libname.Length();
   TEnvRec *rec;
   TIter next(fMapfile->GetTable());
   R__LOCKGUARD(gInterpreterMutex);
   Int_t ret = 0;
   while ((rec = (TEnvRec *) next())) {
      TString cls = rec->GetName();
      if (cls.Length() > 2) {
         // get the first lib from the list of lib and dependent libs
         TString libs = rec->GetValue();
         if (libs == "") {
            continue;
         }
         TString delim(" ");
         TObjArray* tokens = libs.Tokenize(delim);
         const char* lib = ((TObjString *)tokens->At(0))->GetName();
         if (!strncmp(cls.Data(), "Library.", 8) && cls.Length() > 8) {
            // convert "@@" to "::", we used "@@" because TEnv
            // considers "::" a terminator
            cls.Remove(0, 8);
            cls.ReplaceAll("@@", "::");
            // convert "-" to " ", since class names may have
            // blanks and TEnv considers a blank a terminator
            cls.ReplaceAll("-", " ");
         }
         if (!strncmp(lib, libname.Data(), len)) {
            if (fMapfile->GetTable()->Remove(rec) == 0) {
               Error("UnloadLibraryMap", "entry for <%s, %s> not found in library map table", cls.Data(), lib);
               ret = -1;
            }
         }
         delete tokens;
      }
   }
   if (ret >= 0) {
      TString library_rootmap(library);
      if (!library_rootmap.EndsWith(".rootmap"))
         library_rootmap.Append(".rootmap");
      TNamed* mfile = 0;
      while ((mfile = (TNamed *)fRootmapFiles->FindObject(library_rootmap))) {
         fRootmapFiles->Remove(mfile);
         delete mfile;
      }
      fRootmapFiles->Compress();
   }
   return ret;
}

//______________________________________________________________________________
Int_t TCling::SetClassSharedLibs(const char *cls, const char *libs)
{
   // Register the autoloading information for a class.
   // libs is a space separated list of libraries.

   if (!cls || !*cls)
      return 0;

   TString key = TString("Library.") + cls;
   // convert "::" to "@@", we used "@@" because TEnv
   // considers "::" a terminator
   key.ReplaceAll("::", "@@");
   // convert "-" to " ", since class names may have
   // blanks and TEnv considers a blank a terminator
   key.ReplaceAll(" ", "-");

   R__LOCKGUARD(gInterpreterMutex);
   if (!fMapfile) {
      fMapfile = new TEnv();
      fMapfile->IgnoreDuplicates(kTRUE);
//       fMapNamespaces = new THashTable();
//       fMapNamespaces->SetOwner();

      fRootmapFiles = new TObjArray;
      fRootmapFiles->SetOwner();

      InitRootmapFile(".rootmap");
   }
   //fMapfile->SetValue(key, libs);
   fMapfile->SetValue(cls, libs);
   return 1;
}

//______________________________________________________________________________
TClass *TCling::GetClass(const std::type_info& typeinfo, Bool_t load) const
{
   // Demangle the name (from the typeinfo) and then request the class
   // via the usual name based interface (TClass::GetClass).

   int err = 0;
   char* demangled_name = TClassEdit::DemangleTypeIdName(typeinfo, err);
   if (err) return 0;
   TClass* theClass = TClass::GetClass(demangled_name, load, kTRUE);
   free(demangled_name);
   return theClass;
}

//______________________________________________________________________________
Int_t TCling::AutoLoad(const std::type_info& typeinfo, Bool_t knowDictNotLoaded /* = kFALSE */)
{
   // Load library containing the specified class. Returns 0 in case of error
   // and 1 in case if success.

   int err = 0;
   char* demangled_name_c = TClassEdit::DemangleTypeIdName(typeinfo, err);
   if (err) {
      return 0;
   }

   std::string demangled_name(demangled_name_c);
   free(demangled_name_c);

   // AutoLoad expects (because TClass::GetClass already prepares it that way) a
   // shortened name.
   TClassEdit::TSplitType splitname( demangled_name.c_str(), (TClassEdit::EModType)(TClassEdit::kLong64 | TClassEdit::kDropStd) );
   splitname.ShortType(demangled_name, TClassEdit::kDropStlDefault | TClassEdit::kDropStd);

   // No need to worry about typedef, they aren't any ... but there are
   // inlined namespaces ...

   Int_t result = AutoLoad(demangled_name.c_str());
   if (result == 0) {
      demangled_name = TClassEdit::GetLong64_Name(demangled_name);
      result = AutoLoad(demangled_name.c_str(), knowDictNotLoaded);
   }

   return result;
}

//______________________________________________________________________________
Int_t TCling::AutoLoad(const char *cls, Bool_t knowDictNotLoaded /* = kFALSE */)
{
   // Load library containing the specified class. Returns 0 in case of error
   // and 1 in case if success.

   R__LOCKGUARD(gInterpreterMutex);

   if (!knowDictNotLoaded && gClassTable->GetDictNorm(cls)) {
      // The library is alreday loaded as the class's dictionary is known.
      // Return success.
      // Note: the name (cls) is expected to be normalized as it comes either
      // from a callbacks (that can/should calculate the normalized name from the
      // decl) or from TClass::GetClass (which does also calculate the normalized
      // name).
      return 1;
   }

   if (gDebug > 2) {
      Info("TCling::AutoLoad",
           "Trying to autoload for %s", cls);
   }

   Int_t status = 0;
   if (!gROOT || !gInterpreter || gROOT->TestBit(TObject::kInvalidObject)) {
      if (gDebug > 2) {
         Info("TCling::AutoLoad",
              "Disabled due to gROOT or gInterpreter being invalid/not ready (the class name is %s)", cls);
      }
      return status;
   }
   if (!fAllowLibLoad) {
      // Never load any library from rootcling/genreflex.
      if (gDebug > 2) {
         Info("TCling::AutoLoad",
              "Explicitly disabled (the class name is %s)", cls);
      }
      return 0;
   }
   // Prevent the recursion when the library dictionary are loaded.
   Int_t oldvalue = SetClassAutoloading(false);
   // Try using externally provided callback first.
   if (fAutoLoadCallBack) {
      int success = (*(AutoLoadCallBack_t)fAutoLoadCallBack)(cls);
      if (success) {
         SetClassAutoloading(oldvalue);
         return success;
      }
   }
   // lookup class to find list of dependent libraries
   TString deplibs = GetClassSharedLibs(cls);
   if (!deplibs.IsNull()) {
      TString delim(" ");
      TObjArray* tokens = deplibs.Tokenize(delim);
      for (Int_t i = (tokens->GetEntriesFast() - 1); i > 0; --i) {
         const char* deplib = ((TObjString*)tokens->At(i))->GetName();
         if (gROOT->LoadClass(cls, deplib) == 0) {
            if (gDebug > 0) {
               Info("TCling::AutoLoad",
                    "loaded dependent library %s for %s", deplib, cls);
            }
         }
         else {
            Error("TCling::AutoLoad",
                  "failure loading dependent library %s for %s",
                  deplib, cls);
         }
      }
      const char* lib = ((TObjString*)tokens->At(0))->GetName();
      if (lib && lib[0]) {
         if (gROOT->LoadClass(cls, lib) == 0) {
            if (gDebug > 0) {
               Info("TCling::AutoLoad",
                    "loaded library %s for %s", lib, cls);
            }
            status = 1;
         }
         else {
            Error("TCling::AutoLoad",
                  "failure loading library %s for %s", lib, cls);
         }
      }
      delete tokens;
   }

   SetClassAutoloading(oldvalue);
   return status;
}

//______________________________________________________________________________
static cling::Interpreter::CompilationResult ExecAutoParse(const char *what,
                                                           Bool_t header,
                                                           cling::Interpreter *interpreter)
{
   // Parse the payload or header.

   // Save state of the PP
   Sema &SemaR = interpreter->getSema();
   ASTContext& C = SemaR.getASTContext();
   Preprocessor &PP = SemaR.getPreprocessor();
   Parser& P = const_cast<Parser&>(interpreter->getParser());
   Preprocessor::CleanupAndRestoreCacheRAII cleanupRAII(PP);
   Parser::ParserCurTokRestoreRAII savedCurToken(P);
   // After we have saved the token reset the current one to something which
   // is safe (semi colon usually means empty decl)
   Token& Tok = const_cast<Token&>(P.getCurToken());
   Tok.setKind(tok::semi);

   // We can't PushDeclContext, because we go up and the routine that pops
   // the DeclContext assumes that we drill down always.
   // We have to be on the global context. At that point we are in a
   // wrapper function so the parent context must be the global.
   Sema::ContextAndScopeRAII pushedDCAndS(SemaR, C.getTranslationUnitDecl(),
                                          SemaR.TUScope);
   std::string code = gNonInterpreterClassDef ;
   if (!header) {
      // This is the complete header file content and not the
      // name of a header.
      code += what;

   } else {
      code += ("#include \"");
      code += what;
      code += "\"\n";
   }
   code += ("#ifdef __ROOTCLING__\n"
            "#undef __ROOTCLING__\n"
            + gInterpreterClassDef +
            "#endif");

   cling::Interpreter::CompilationResult cr;
   {
      // scope within which diagnostics are de-activated
      // For now we disable diagnostics because we saw them already at
      // dictionary generation time. That won't be an issue with the PCMs.

      clangDiagSuppr diagSuppr(SemaR.getDiagnostics());

      #if defined(R__MUST_REVISIT)
      #if R__MUST_REVISIT(6,2)
      Warning("TCling::RegisterModule","Diagnostics suppression should be gone by now.");
      #endif
      #endif

      cr = interpreter->parseForModule(code);
   }
   return cr;
}

////////////////////////////////////////////////////////////////////////////////
/// Helper routine for TCling::AutoParse implementing the actual call to the
/// parser and looping over template parameters (if
/// any) and when they don't have a registered header to autoparse,
/// recurse over their template parameters.
///
/// Returns the number of header parsed.

UInt_t TCling::AutoParseImplRecurse(const char *cls, bool topLevel)
{
   // We assume the lock has already been taken.
   //    R__LOCKGUARD(gInterpreterMutex);

   Int_t nHheadersParsed = 0;

   // Loop on the possible autoparse keys
   bool skipFirstEntry = false;
   std::vector<std::string> autoparseKeys;
   if (strchr(cls, '<')) {
      int nestedLoc = 0;
      TClassEdit::GetSplit(cls, autoparseKeys, nestedLoc, TClassEdit::kDropTrailStar);
      // Check if we can skip the name of the template in the autoparses
      // Take all the scopes one by one. If all of them are in the AST, we do not
      // need to autoparse for that particular template.
      if (!autoparseKeys.empty()){
         TString templateName(autoparseKeys[0]);
         auto tokens = templateName.Tokenize("::");
         clang::NamedDecl* previousScopeAsNamedDecl = nullptr;
         clang::DeclContext* previousScopeAsContext = nullptr;
         for (auto const & scopeObj : *tokens){
            auto scopeName = ((TObjString*) scopeObj)->String().Data();
            previousScopeAsNamedDecl = cling::utils::Lookup::Named(&fInterpreter->getSema(), scopeName, previousScopeAsContext);
            // Check if we have multipple nodes in the AST with this name
            if ((clang::NamedDecl*)-1 == previousScopeAsNamedDecl) break;
            previousScopeAsContext = llvm::dyn_cast_or_null<clang::DeclContext>(previousScopeAsNamedDecl);
            if (!previousScopeAsContext) break; // this is not a context
         }
         delete tokens;
         // Now, let's check if the last scope, the template, has a definition, i.e. it's not a fwd decl
         if (auto templateDecl = llvm::dyn_cast_or_null<clang::ClassTemplateDecl>(previousScopeAsNamedDecl)) {
            if (auto templatedDecl = templateDecl->getTemplatedDecl()) {
               skipFirstEntry = nullptr != templatedDecl->getDefinition();
            }
         }

      }
   }
   if (topLevel) autoparseKeys.emplace_back(cls);

   for (const auto & apKeyStr : autoparseKeys) {
      if (skipFirstEntry) {
         skipFirstEntry=false;
         continue;
      }
      if (apKeyStr.empty()) continue;
      const char *apKey = apKeyStr.c_str();
      std::size_t normNameHash(fStringHashFunction(apKey));
      // If the class was not looked up
      if (gDebug > 1) {
         Info("TCling::AutoParse",
              "Starting autoparse for %s\n", apKey);
      }
      if (fLookedUpClasses.insert(normNameHash).second) {
         auto const &iter = fClassesHeadersMap.find(normNameHash);
         if (iter != fClassesHeadersMap.end()) {
            const cling::Transaction *T = fInterpreter->getCurrentTransaction();
            fTransactionHeadersMap.insert({T,normNameHash});
            auto const &hNamesPtrs = iter->second;
            if (gDebug > 1) {
               Info("TCling::AutoParse",
                    "We can proceed for %s. We have %s headers.", apKey, std::to_string(hNamesPtrs.size()).c_str());
            }
            for (auto & hName : hNamesPtrs) {
               if (fParsedPayloadsAddresses.count(hName) == 1) continue;
               if (0 != fPayloads.count(normNameHash)) {
                  float initRSSval=0.f, initVSIZEval=0.f;
                  (void) initRSSval; // Avoid unused var warning
                  (void) initVSIZEval;
                  if (gDebug > 0) {
                     Info("AutoParse",
                          "Parsing full payload for %s", apKey);
                     ProcInfo_t info;
                     gSystem->GetProcInfo(&info);
                     initRSSval = 1e-3*info.fMemResident;
                     initVSIZEval = 1e-3*info.fMemVirtual;
                  }
                  auto cRes = ExecAutoParse(hName, kFALSE, fInterpreter);
                  if (cRes != cling::Interpreter::kSuccess) {
                     if (hName[0] == '\n')
                        Error("AutoParse", "Error parsing payload code for class %s with content:\n%s", apKey, hName);
                  } else {
                     fParsedPayloadsAddresses.insert(hName);
                     nHheadersParsed++;
                     if (gDebug > 0){
                        ProcInfo_t info;
                        gSystem->GetProcInfo(&info);
                        float endRSSval = 1e-3*info.fMemResident;
                        float endVSIZEval = 1e-3*info.fMemVirtual;
                        Info("Autoparse", ">>> RSS key %s - before %.3f MB - after %.3f MB - delta %.3f MB", apKey, initRSSval, endRSSval, endRSSval-initRSSval);
                        Info("Autoparse", ">>> VSIZE key %s - before %.3f MB - after %.3f MB - delta %.3f MB", apKey, initVSIZEval, endVSIZEval, endVSIZEval-initVSIZEval);
                     }
                  }
               } else if (!IsLoaded(hName)) {
                  if (gDebug > 0) {
                     Info("AutoParse",
                          "Parsing single header %s", hName);
                  }
                  auto cRes = ExecAutoParse(hName, kTRUE, fInterpreter);
                  if (cRes != cling::Interpreter::kSuccess) {
                     Error("AutoParse", "Error parsing headerfile %s for class %s.", hName, apKey);
                  } else {
                     nHheadersParsed++;
                  }
               }
            }
         }
         else {
            // There is no header registered for this class, if this a
            // template, it will be instantiated if/when it is requested
            // and if we do no load/parse its components we might end up
            // not using an eventual specialization.
            if (strchr(apKey, '<')) {
               nHheadersParsed += AutoParseImplRecurse(apKey, false);
            }
         }
      }
   }

   return nHheadersParsed;

}

////////////////////////////////////////////////////////////////////////////////
/// Parse the headers relative to the class
/// Returns 1 in case of success, 0 in case of failure

Int_t TCling::AutoParse(const char *cls)
{
   R__LOCKGUARD(gInterpreterMutex);

   if (!fHeaderParsingOnDemand || fIsAutoParsingSuspended) {
      if (fClingCallbacks->IsAutoloadingEnabled()) {
         return AutoLoad(cls);
      } else {
         return 0;
      }
   }

   if (gDebug > 1) {
      Info("TCling::AutoParse",
           "Trying to autoparse for %s", cls);
   }

   // The catalogue of headers is in the dictionary
   if (fClingCallbacks->IsAutoloadingEnabled()) {
      AutoLoad(cls);
   }

   // Prevent the recursion when the library dictionary are loaded.
   Int_t oldAutoloadValue = SetClassAutoloading(false);

   // No recursive header parsing on demand; we require headers to be standalone.
   SuspendAutoParsing autoParseRAII(this);

   Int_t nHheadersParsed = AutoParseImplRecurse(cls,/*topLevel=*/ true);

   if (nHheadersParsed != 0) {
      while (!fClassesToUpdate.empty()) {
         TClass *oldcl = fClassesToUpdate.back().first;
         if (oldcl->GetState() != TClass::kHasTClassInit) {
            // if (gDebug > 2) Info("RegisterModule", "Forcing TClass init for %s", oldcl->GetName());
            DictFuncPtr_t dict = fClassesToUpdate.back().second;
            fClassesToUpdate.pop_back();
            // Calling func could manipulate the list so, let maintain the list
            // then call the dictionary function.
            TClass *ncl = dict();
            if (ncl) ncl->PostLoadCheck();
         } else {
            fClassesToUpdate.pop_back();
         }
      }
   }

   SetClassAutoloading(oldAutoloadValue);

   return nHheadersParsed > 0 ? 1 : 0;
}


//______________________________________________________________________________
void* TCling::LazyFunctionCreatorAutoload(const std::string& mangled_name) {
   // Autoload a library based on a missing symbol.

   // First see whether the symbol is in the library that we are currently
   // loading. It will have access to the symbols of its dependent libraries,
   // thus checking "back()" is sufficient.
   if (!fRegisterModuleDyLibs.empty()) {
      if (void* addr = dlsym(fRegisterModuleDyLibs.back(),
                             mangled_name.c_str())) {
         return addr;
      }
   }

   int err = 0;
   char* demangled_name_c = TClassEdit::DemangleName(mangled_name.c_str(), err);
   if (err) {
      return 0;
   }

   std::string name(demangled_name_c);
   free(demangled_name_c);

   //fprintf(stderr, "demangled name: '%s'\n", demangled_name);
   //
   //  Separate out the class or namespace part of the
   //  function name.
   //

   if (!strncmp(name.c_str(), "typeinfo for ", sizeof("typeinfo for ")-1)) {
      name.erase(0, sizeof("typeinfo for ")-1);
   } else if (!strncmp(name.c_str(), "vtable for ", sizeof("vtable for ")-1)) {
      name.erase(0, sizeof("vtable for ")-1);
   } else if (!strncmp(name.c_str(), "operator", sizeof("operator")-1)
              && !isalnum(name[sizeof("operator")])) {
     // operator...(A, B) - let's try with A!
     name.erase(0, sizeof("operator")-1);
     std::string::size_type pos = name.rfind('(');
     if (pos != std::string::npos) {
       name.erase(0, pos + 1);
       pos = name.find(",");
       if (pos != std::string::npos) {
         // remove next arg up to end, leaving only the first argument type.
         name.erase(pos);
       }
       pos = name.rfind(" const");
       if (pos != std::string::npos) {
         name.erase(pos, strlen(" const"));
       }
       while (!name.empty() && strchr("&*", name.back()))
         name.erase(name.length() - 1);
     }
   } else {
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
   }
   //fprintf(stderr, "name: '%s'\n", name.c_str());
   // Now we have the class or namespace name, so do the lookup.
   TString libs = GetClassSharedLibs(name.c_str());
   if (libs.IsNull()) {
      // Not found in the map, all done.
      return 0;
   }
   //fprintf(stderr, "library: %s\n", iter->second.c_str());
   // Now we have the name of the libraries to load, so load them.

   TString lib;
   Ssiz_t posLib = 0;
   while (libs.Tokenize(lib, posLib)) {
      if (gSystem->Load(lib, "", kFALSE /*system*/) < 0) {
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
Bool_t TCling::IsAutoLoadNamespaceCandidate(const char* name)
{
//    if (fMapNamespaces){
//       return fMapNamespaces->FindObject(name);
//    }
   return false;
}

//______________________________________________________________________________
Bool_t TCling::IsAutoLoadNamespaceCandidate(const clang::NamespaceDecl* nsDecl)
{
   return fNSFromRootmaps.count(nsDecl) != 0;
}

//______________________________________________________________________________
void TCling::UpdateClassInfoWithDecl(const void* vTD)
{
   // Internal function. Inform a TClass about its new TagDecl or NamespaceDecl.
   const NamedDecl* ND = static_cast<const NamedDecl*>(vTD);
   const TagDecl* td = dyn_cast<TagDecl>(ND);
   std::string name;
   TagDecl* tdDef = 0;
   if (td) {
      tdDef = td->getDefinition();
      // Let's pass the decl to the TClass only if it has a definition.
      if (!tdDef) return;
      td = tdDef;
      ND = td;

      if (llvm::isa<clang::FunctionDecl>(td->getDeclContext())) {
         // Ignore declaration within a function.
         return;
      }
      clang::QualType type( td->getTypeForDecl(), 0 );

      auto declName=ND->getNameAsString();
      if (!TClass::HasNoInfoOrEmuOrFwdDeclaredDecl(declName.c_str())){
//          printf ("Impossible to find a TClassEntry in kNoInfo or kEmulated the decl of which would be called %s. Skip w/o building the normalized name.\n",declName );
         return;
      }

      ROOT::TMetaUtils::GetNormalizedName(name, type, *fInterpreter, *fNormalizedCtxt);
   } else {
      name = ND->getNameAsString();
   }

   // Supposedly we are being called while something is being
   // loaded ... let's now tell the autoloader to do the work
   // yet another time.
   int storedAutoloading = SetClassAutoloading(false);
   // FIXME: There can be more than one TClass for a single decl.
   // for example vector<double> and vector<Double32_t>
   TClass* cl = (TClass*)gROOT->GetListOfClasses()->FindObject(name.c_str());
   if (cl && GetModTClasses().find(cl) == GetModTClasses().end()) {
      TClingClassInfo* cci = ((TClingClassInfo*)cl->fClassInfo);
      if (cci) {
         // If we only had a forward declaration then update the
         // TClingClassInfo with the definition if we have it now.
         const TagDecl* tdOld = llvm::dyn_cast_or_null<TagDecl>(cci->GetDecl());
         if (!tdOld || (tdDef && tdDef != tdOld)) {
            cl->ResetCaches();
            TClass::RemoveClassDeclId(cci->GetDeclId());
            if (td) {
               // It's a tag decl, not a namespace decl.
               cci->Init(*cci->GetType());
               TClass::AddClassToDeclIdMap(cci->GetDeclId(), cl);
            }
         }
      } else if (!cl->TestBit(TClass::kLoading) && !cl->fHasRootPcmInfo) {
         cl->ResetCaches();
         // yes, this is almost a waste of time, but we do need to lookup
         // the 'type' corresponding to the TClass anyway in order to
         // preserve the opaque typedefs (Double32_t)
         cl->fClassInfo = (ClassInfo_t *)new TClingClassInfo(fInterpreter, cl->GetName());
         // We now need to update the state and bits.
         if (cl->fState != TClass::kHasTClassInit) {
            // if (!cl->fClassInfo->IsValid()) cl->fState = TClass::kForwardDeclared; else
            cl->fState = TClass::kInterpreted;
            cl->ResetBit(TClass::kIsEmulation);
         }
         TClass::AddClassToDeclIdMap(((TClingClassInfo*)(cl->fClassInfo))->GetDeclId(), cl);
      }
   }
   SetClassAutoloading(storedAutoloading);
}

//______________________________________________________________________________
void TCling::UpdateClassInfo(char* item, Long_t tagnum)
{
   // No op: see TClingCallbacks
}

//______________________________________________________________________________
//FIXME: Factor out that function in TClass, because TClass does it already twice
void TCling::UpdateClassInfoWork(const char* item)
{
   // This is a no-op as part of the API.
   // TCling uses UpdateClassInfoWithDecl() instead.
}

//______________________________________________________________________________
void TCling::UpdateAllCanvases()
{
   // Update all canvases at end the terminal input command.
   TIter next(gROOT->GetListOfCanvases());
   TVirtualPad* canvas;
   while ((canvas = (TVirtualPad*)next())) {
      canvas->Update();
   }
}

//______________________________________________________________________________
void TCling::UpdateListsOnCommitted(const cling::Transaction &T) {

   std::set<TClass*> modifiedTClasses; // TClasses that require update after this transaction

   // If the transaction does not contain anything we can return earlier.
   if (!HandleNewTransaction(T)) return;

   bool isTUTransaction = false;
   if (T.decls_end()-T.decls_begin() == 1 && !T.hasNestedTransactions()) {
      clang::Decl* FirstDecl = *(T.decls_begin()->m_DGR.begin());
      if (clang::TranslationUnitDecl* TU
          = dyn_cast<clang::TranslationUnitDecl>(FirstDecl)) {
         // The is the first transaction, we have to expose to meta
         // what's already in the AST.
         isTUTransaction = true;

         // FIXME: don't load the world. Really, don't. Maybe
         // instead smarten TROOT::GetListOfWhateveros() which
         // currently is a THashList but could be a
         // TInterpreterLookupCollection, one that reimplements
         // TCollection::FindObject(name) and performs a lookup
         // if not found in its T(Hash)List.
         cling::Interpreter::PushTransactionRAII RAII(fInterpreter);
         for (clang::DeclContext::decl_iterator TUI = TU->decls_begin(),
                 TUE = TU->decls_end(); TUI != TUE; ++TUI)
            ((TCling*)gCling)->HandleNewDecl(*TUI, (*TUI)->isFromASTFile(),modifiedTClasses);
      }
   }

   std::set<const void*> TransactionDeclSet;
   if (!isTUTransaction && T.decls_end() - T.decls_begin()) {
      const clang::Decl* WrapperFD = T.getWrapperFD();
      for (cling::Transaction::const_iterator I = T.decls_begin(), E = T.decls_end();
          I != E; ++I) {
         if (I->m_Call != cling::Transaction::kCCIHandleTopLevelDecl
             && I->m_Call != cling::Transaction::kCCIHandleTagDeclDefinition)
            continue;

         for (DeclGroupRef::const_iterator DI = I->m_DGR.begin(),
                 DE = I->m_DGR.end(); DI != DE; ++DI) {
            if (*DI == WrapperFD)
               continue;
            TransactionDeclSet.insert(*DI);
            ((TCling*)gCling)->HandleNewDecl(*DI, false, modifiedTClasses);
         }
      }
   }

   // The above might trigger more decls to be deserialized.
   // Thus the iteration over the deserialized decls must be last.
   for (cling::Transaction::const_iterator I = T.deserialized_decls_begin(),
           E = T.deserialized_decls_end(); I != E; ++I) {
      for (DeclGroupRef::const_iterator DI = I->m_DGR.begin(),
              DE = I->m_DGR.end(); DI != DE; ++DI)
         if (TransactionDeclSet.find(*DI) == TransactionDeclSet.end()) {
            //FIXME: HandleNewDecl should take DeclGroupRef
            ((TCling*)gCling)->HandleNewDecl(*DI, /*isDeserialized*/true,
                                             modifiedTClasses);
         }
   }


   // When fully building the reflection info in TClass, a deserialization
   // could be triggered, which may result in request for building the
   // reflection info for the same TClass. This in turn will clear the caches
   // for the TClass in-flight and cause null ptr derefs.
   // FIXME: This is a quick fix, solving most of the issues. The actual
   // question is: Shouldn't TClass provide a lock mechanism on update or lock
   // itself until the update is done.
   //
   std::vector<TClass*> modifiedTClassesDiff(modifiedTClasses.size());
   std::vector<TClass*>::iterator it;
   it = set_difference(modifiedTClasses.begin(), modifiedTClasses.end(),
                       ((TCling*)gCling)->GetModTClasses().begin(),
                       ((TCling*)gCling)->GetModTClasses().end(),
                       modifiedTClassesDiff.begin());
   modifiedTClassesDiff.resize(it - modifiedTClassesDiff.begin());

   // Lock the TClass for updates
   ((TCling*)gCling)->GetModTClasses().insert(modifiedTClassesDiff.begin(),
                                              modifiedTClassesDiff.end());
   for (std::vector<TClass*>::const_iterator I = modifiedTClassesDiff.begin(),
           E = modifiedTClassesDiff.end(); I != E; ++I) {
      // Make sure the TClass has not been deleted.
      if (!gROOT->GetListOfClasses()->FindObject(*I)) {
         continue;
      }
      // Could trigger deserialization of decls.
      cling::Interpreter::PushTransactionRAII RAII(fInterpreter);
      // Unlock the TClass for updates
      ((TCling*)gCling)->GetModTClasses().erase(*I);

   }
}

//______________________________________________________________________________
void TCling::UnloadClassMembers(TClass* cl, const clang::DeclContext* DC) {

   // Helper function to go through the members of a class or namespace and unload them.

   TDataMember* var = 0;
   TFunction* function = 0;
   TEnum* e = 0;
   TFunctionTemplate* functiontemplate = 0;
   TListOfDataMembers* datamembers = (TListOfDataMembers*)cl->GetListOfDataMembers();
   TListOfFunctions* functions = (TListOfFunctions*)cl->GetListOfMethods();
   TListOfEnums* enums = (TListOfEnums*)cl->GetListOfEnums();
   TListOfFunctionTemplates* functiontemplates = (TListOfFunctionTemplates*)cl->GetListOfFunctionTemplates();
   for (DeclContext::decl_iterator RI = DC->decls_begin(), RE = DC->decls_end(); RI != RE; ++RI) {
      if (isa<VarDecl>(*RI) || isa<FieldDecl>(*RI)) {
         const clang::ValueDecl* VD = dyn_cast<ValueDecl>(*RI);
         var = (TDataMember*)datamembers->FindObject(VD->getNameAsString().c_str());
         if (var) {
            // Unload the global by setting the DataMemberInfo_t to 0
            datamembers->Unload(var);
            var->Update(0);
         }
      } else if (const FunctionDecl* FD = dyn_cast<FunctionDecl>(*RI)) {
         function = (TFunction*)functions->FindObject(FD->getNameAsString().c_str());
         if (function) {
            functions->Unload(function);
            function->Update(0);
         }
      } else if (const EnumDecl* ED = dyn_cast<EnumDecl>(*RI)) {
         e = (TEnum*)enums->FindObject(ED->getNameAsString().c_str());
         if (e) {
            TIter iEnumConst(e->GetConstants());
            while (TEnumConstant* enumConst = (TEnumConstant*)iEnumConst()) {
               // Since the enum is already created and valid that ensures us that
               // we have the enum constants created as well.
               enumConst = (TEnumConstant*)datamembers->FindObject(enumConst->GetName());
               if (enumConst && enumConst->IsValid()) {
                  datamembers->Unload(enumConst);
                  enumConst->Update(0);
               }
            }
            enums->Unload(e);
            e->Update(0);
         }
      } else if (const FunctionTemplateDecl* FTD = dyn_cast<FunctionTemplateDecl>(*RI)) {
         functiontemplate = (TFunctionTemplate*)functiontemplates->FindObject(FTD->getNameAsString().c_str());
         if (functiontemplate) {
            functiontemplates->Unload(functiontemplate);
            functiontemplate->Update(0);
         }
      }
   }
}

//______________________________________________________________________________
void TCling::UpdateListsOnUnloaded(const cling::Transaction &T)
{
   HandleNewTransaction(T);

   // Unload the objects from the lists and update the objects' state.
   TListOfFunctions* functions = (TListOfFunctions*)gROOT->GetListOfGlobalFunctions();
   TListOfFunctionTemplates* functiontemplates = (TListOfFunctionTemplates*)gROOT->GetListOfFunctionTemplates();
   TListOfEnums* enums = (TListOfEnums*)gROOT->GetListOfEnums();
   TListOfDataMembers* globals = (TListOfDataMembers*)gROOT->GetListOfGlobals();
  cling::Transaction::const_nested_iterator iNested = T.nested_begin();
   for(cling::Transaction::const_iterator I = T.decls_begin(), E = T.decls_end();
       I != E; ++I) {
      if (I->m_Call == cling::Transaction::kCCIHandleVTable)
         continue;

     if (I->m_Call == cling::Transaction::kCCINone) {
         UpdateListsOnUnloaded(**iNested);
         ++iNested;
         continue;
      }

      for (DeclGroupRef::const_iterator DI = I->m_DGR.begin(),
              DE = I->m_DGR.end(); DI != DE; ++DI) {

         // Do not mark a decl as unloaded if we are going to keep it
         // (because it comes from the pch) ...
         if ( (*DI)->isFromASTFile() )
            continue;

         // Deal with global variables and global enum constants.
         if (isa<VarDecl>(*DI) || isa<EnumConstantDecl>(*DI)) {
            TObject *obj = globals->Find((TListOfDataMembers::DeclId_t)*DI);
            if (globals->GetClass()) {
               TDataMember* var = dynamic_cast<TDataMember*>(obj);
               if (var && var->IsValid()) {
                  // Unload the global by setting the DataMemberInfo_t to 0
                  globals->Unload(var);
                  var->Update(0);
               }
            } else {
               TGlobal *g = dynamic_cast<TGlobal*>(obj);
               if (g && g->IsValid()) {
                  // Unload the global by setting the DataMemberInfo_t to 0
                  globals->Unload(g);
                  g->Update(0);
               }
            }
         // Deal with global functions.
         } else if (const FunctionDecl* FD = dyn_cast<FunctionDecl>(*DI)) {
            TFunction* function = (TFunction*)functions->Find((TListOfFunctions::DeclId_t)FD);
            if (function && function->IsValid()) {
               functions->Unload(function);
               function->Update(0);
            }
         // Deal with global function templates.
         } else if (const FunctionTemplateDecl* FTD = dyn_cast<FunctionTemplateDecl>(*DI)) {
            TFunctionTemplate* functiontemplate = (TFunctionTemplate*)functiontemplates->FindObject(FTD->getNameAsString().c_str());
            if (functiontemplate) {
               functiontemplates->Unload(functiontemplate);
               functiontemplate->Update(0);
            }
         // Deal with global enums.
         } else if (const EnumDecl* ED = dyn_cast<EnumDecl>(*DI)) {
            if (TEnum* e = (TEnum*)enums->Find((TListOfEnums::DeclId_t)ED)) {
               globals = (TListOfDataMembers*)gROOT->GetListOfGlobals();
               TIter iEnumConst(e->GetConstants());
               while (TEnumConstant* enumConst = (TEnumConstant*)iEnumConst()) {
                  // Since the enum is already created and valid that ensures us that
                  // we have the enum constants created as well.
                  enumConst = (TEnumConstant*)globals->FindObject(enumConst->GetName());
                  if (enumConst) {
                     globals->Unload(enumConst);
                     enumConst->Update(0);
                  }
               }
               enums->Unload(e);
               e->Update(0);
            }
         // Deal with classes. Unload the class and the data members will be not accessible anymore
         // Cannot declare the members in a different declaration like redeclarable namespaces.
         } else if (const clang::RecordDecl* RD = dyn_cast<RecordDecl>(*DI)) {
            std::vector<TClass*> vectTClass;
            // Only update the TClass if the definition is being unloaded.
            if (RD->isCompleteDefinition()) {
               if (TClass::GetClass(RD, vectTClass)) {
                  for (std::vector<TClass*>::iterator CI = vectTClass.begin(), CE = vectTClass.end();
                       CI != CE; ++CI) {
                     UnloadClassMembers((*CI), RD);
                     (*CI)->ResetClassInfo();
                  }
               }
            }
         // Deal with namespaces. Unload the members of the current redeclaration only.
         } else if (const clang::NamespaceDecl* ND = dyn_cast<NamespaceDecl>(*DI)) {
            std::vector<TClass*> vectTClass;
            if (TClass::GetClass(ND->getCanonicalDecl(), vectTClass)) {
               for (std::vector<TClass*>::iterator CI = vectTClass.begin(), CE = vectTClass.end();
                    CI != CE; ++CI) {
                  UnloadClassMembers((*CI), ND);
                  if (ND->isOriginalNamespace()) {
                     (*CI)->ResetClassInfo();
                  }
               }
            }
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
// If an autoparse was done during a transaction and that it is rolled back,
// we need to make sure the next request for the same autoparse will be
// honored.
void TCling::TransactionRollback(const cling::Transaction &T) {
   auto const &triter = fTransactionHeadersMap.find(&T);
   if (triter != fTransactionHeadersMap.end()) {
      std::size_t normNameHash = triter->second;

      fLookedUpClasses.erase(normNameHash);

      auto const &iter = fClassesHeadersMap.find(normNameHash);
      if (iter != fClassesHeadersMap.end()) {
         auto const &hNamesPtrs = iter->second;
         for (auto &hName : hNamesPtrs) {
            if (gDebug > 0) {
               Info("TransactionRollback",
                    "Restoring ability to autoaparse: %s", hName);
            }
            fParsedPayloadsAddresses.erase(hName);
         }
      }
   }
}

void TCling::LibraryLoaded(const void* dyLibHandle, const char* canonicalName) {
   //UpdateListOfLoadedSharedLibraries();
}

//______________________________________________________________________________
void TCling::LibraryUnloaded(const void* dyLibHandle, const char* canonicalName) {
   fPrevLoadedDynLibInfo = 0;
   fSharedLibs = "";
}

//______________________________________________________________________________
const char* TCling::GetSharedLibs()
{
   // Return the list of shared libraries loaded into the process.
   if (!fPrevLoadedDynLibInfo && fSharedLibs.IsNull())
      UpdateListOfLoadedSharedLibraries();
   return fSharedLibs;
}

//______________________________________________________________________________
const char* TCling::GetClassSharedLibs(const char* cls)
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
      TEnvRec* libs_record = 0;
      libs_record = fMapfile->Lookup(cls);
      if (libs_record) {
         const char* libs = libs_record->GetValue();
         return (*libs) ? libs : 0;
      }
      else {
         // Try the old format...
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
         TEnvRec* libs_record = 0;
         libs_record = fMapfile->Lookup(c);
         if (libs_record) {
            const char* libs = libs_record->GetValue();
            return (*libs) ? libs : 0;
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
const char* TCling::GetSharedLibDeps(const char* lib)
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
Bool_t TCling::IsErrorMessagesEnabled() const
{
   // If error messages are disabled, the interpreter should suppress its
   // failures and warning messages from stdout.
#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   Warning("IsErrorMessagesEnabled", "Interface not available yet.");
#endif
#endif
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TCling::SetErrorMessages(Bool_t enable)
{
   // If error messages are disabled, the interpreter should suppress its
   // failures and warning messages from stdout. Return the previous state.
#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   Warning("SetErrorMessages", "Interface not available yet.");
#endif
#endif
   return TCling::IsErrorMessagesEnabled();
}

//______________________________________________________________________________
const char* TCling::GetIncludePath()
{
   // Refresh the list of include paths known to the interpreter and return it
   // with -I prepended.

   R__LOCKGUARD(gInterpreterMutex);

   fIncludePath = "";

   llvm::SmallVector<std::string, 10> includePaths;//Why 10? Hell if I know.
   //false - no system header, true - with flags.
   fInterpreter->GetIncludePaths(includePaths, false, true);
   if (const size_t nPaths = includePaths.size()) {
      assert(!(nPaths & 1) && "GetIncludePath, number of paths and options is not equal");

      for (size_t i = 0; i < nPaths; i += 2) {
         if (i)
            fIncludePath.Append(' ');
         fIncludePath.Append(includePaths[i].c_str());

         if (includePaths[i] != "-I")
            fIncludePath.Append(' ');
         fIncludePath.Append('"');
         fIncludePath.Append(includePaths[i + 1], includePaths[i + 1].length());
         fIncludePath.Append('"');
      }
   }

   return fIncludePath;
}

//______________________________________________________________________________
const char* TCling::GetSTLIncludePath() const
{
   // Return the directory containing CINT's stl cintdlls.
   return "";
}

//______________________________________________________________________________
//                      M I S C
//______________________________________________________________________________

int TCling::DisplayClass(FILE* /*fout*/, const char* /*name*/, int /*base*/, int /*start*/) const
{
   // Interface to cling function
   return 0;
}

//______________________________________________________________________________
int TCling::DisplayIncludePath(FILE *fout) const
{
   // Interface to cling function
   assert(fout != 0 && "DisplayIncludePath, 'fout' parameter is null");

   llvm::SmallVector<std::string, 10> includePaths;//Why 10? Hell if I know.
   //false - no system header, true - with flags.
   fInterpreter->GetIncludePaths(includePaths, false, true);
   if (const size_t nPaths = includePaths.size()) {
      assert(!(nPaths & 1) && "DisplayIncludePath, number of paths and options is not equal");

      std::string allIncludes("include path:");
      for (size_t i = 0; i < nPaths; i += 2) {
         allIncludes += ' ';
         allIncludes += includePaths[i];

         if (includePaths[i] != "-I")
            allIncludes += ' ';
         allIncludes += includePaths[i + 1];
      }

      fprintf(fout, "%s\n", allIncludes.c_str());
   }

   return 0;
}

//______________________________________________________________________________
void* TCling::FindSym(const char* entry) const
{
   // Interface to cling function
   return fInterpreter->getAddressOfGlobal(entry);
}

//______________________________________________________________________________
void TCling::GenericError(const char* error) const
{
   // Let the interpreter issue a generic error, and set its error state.
#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   Warning("GenericError","Interface not available yet.");
#endif
#endif
}

//______________________________________________________________________________
Long_t TCling::GetExecByteCode() const
{
   // This routines used to return the address of the internal wrapper
   // function (of the interpreter) that was used to call *all* the
   // interpreted functions that were bytecode compiled (no longer
   // interpreted line by line).  In Cling, there is no such
   // wrapper function.
   // In practice this routines was use to decipher whether the
   // pointer returns by InterfaceMethod could be used to uniquely
   // represent the function.  In Cling if the function is in a
   // useable state (its compiled version is available), this is
   // always the case.
   // See TClass::GetMethod.

   return 0;
}

//______________________________________________________________________________
Long_t TCling::Getgvp() const
{
   // Interface to the CINT global object pointer which was controlling the
   // behavior of the wrapper around the calls to operator new and the constructor
   // and operator delete and the destructor.

   Error("Getgvp","This was controlling the behavior of the wrappers for object construction and destruction.\nThis is now a nop and likely change the behavior of the calling routines.");
   return 0;
}

//______________________________________________________________________________
const char* TCling::Getp2f2funcname(void*) const
{
   Error("Getp2f2funcname", "Will not be implemented: "
         "all function pointers are compiled!");
   return NULL;
}

//______________________________________________________________________________
int TCling::GetSecurityError() const
{
   // Interface to cling function
#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   Warning("GetSecurityError", "Interface not available yet.");
#endif
#endif
   return 0;
}

//______________________________________________________________________________
int TCling::LoadFile(const char* path) const
{
   // Load a source file or library called path into the interpreter.
   cling::Interpreter::CompilationResult compRes;
   cling::MetaProcessor::MaybeRedirectOutputRAII RAII(fMetaProcessor);
   fMetaProcessor->process(TString::Format(".L %s", path), compRes, /*cling::Value*/0);
   return compRes == cling::Interpreter::kFailure;
}

//______________________________________________________________________________
Bool_t TCling::LoadText(const char* text) const
{
   // Load the declarations from text into the interpreter.
   // Note that this cannot be (top level) statements; text must contain
   // top level declarations.
   // Returns true on success, false on failure.
   return (fInterpreter->declare(text) == cling::Interpreter::kSuccess);
}

//______________________________________________________________________________
const char* TCling::MapCppName(const char* name) const
{
   // Interface to cling function
   TTHREAD_TLS_DECL(std::string,buffer);
   ROOT::TMetaUtils::GetCppName(buffer,name);
   return buffer.c_str();
}

//______________________________________________________________________________
void TCling::SetAlloclockfunc(void (* /* p */ )()) const
{
   // [Place holder for Mutex Lock]
   // Provide the interpreter with a way to
   // acquire a lock used to protect critical section
   // of its code (non-thread safe parts).

   // nothing to do for now.
}

//______________________________________________________________________________
void TCling::SetAllocunlockfunc(void (* /* p */ )()) const
{
   // [Place holder for Mutex Unlock] Provide the interpreter with a way to
   // release a lock used to protect critical section
   // of its code (non-thread safe parts).

   // nothing to do for now.
}

//______________________________________________________________________________
int TCling::SetClassAutoloading(int autoload) const
{
   // Enable/Disable the Autoloading of libraries.
   // Returns the old value, i.e whether it was enabled or not.
   if (!autoload && !fClingCallbacks) return false;
   if (!fAllowLibLoad) return false;

   assert(fClingCallbacks && "We must have callbacks!");
   bool oldVal =  fClingCallbacks->IsAutoloadingEnabled();
   fClingCallbacks->SetAutoloadingEnabled(autoload);
   return oldVal;
}

//______________________________________________________________________________
int TCling::SetClassAutoparsing(int autoparse)
{
   // Enable/Disable the Autoparsing of headers.
   // Returns the old value, i.e whether it was enabled or not.
   bool oldVal = fHeaderParsingOnDemand;
   fHeaderParsingOnDemand = autoparse;
   return oldVal;
}

////////////////////////////////////////////////////////////////////////////////
/// Suspend the Autoparsing of headers.
/// Returns the old value, i.e whether it was suspended or not.

Bool_t TCling::SetSuspendAutoParsing(Bool_t value) {
   Bool_t old = fIsAutoParsingSuspended;
   fIsAutoParsingSuspended = value;
   if (fClingCallbacks) fClingCallbacks->SetAutoParsingSuspended(value);
   return old;
}

//______________________________________________________________________________
void TCling::SetErrmsgcallback(void* p) const
{
   // Set a callback to receive error messages.
#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   Warning("SetErrmsgcallback", "Interface not available yet.");
#endif
#endif
}

//______________________________________________________________________________
void TCling::Setgvp(Long_t gvp) const
{
   // Interface to the cling global object pointer which was controlling the
   // behavior of the wrapper around the calls to operator new and the constructor
   // and operator delete and the destructor.

   Error("Setgvp","This was controlling the behavior of the wrappers for object construction and destruction.\nThis is now a nop and likely change the behavior of the calling routines.");

}

//______________________________________________________________________________
void TCling::SetRTLD_NOW() const
{
   Error("SetRTLD_NOW()", "Will never be implemented! Don't use!");
}

//______________________________________________________________________________
void TCling::SetRTLD_LAZY() const
{
   Error("SetRTLD_LAZY()", "Will never be implemented! Don't use!");
}

//______________________________________________________________________________
void TCling::SetTempLevel(int val) const
{
   // Create / close a scope for temporaries. No-op for cling; use
   // cling::Value instead.
}

//______________________________________________________________________________
int TCling::UnloadFile(const char* path) const
{
   cling::DynamicLibraryManager* DLM = fInterpreter->getDynamicLibraryManager();
   std::string canonical = DLM->lookupLibrary(path);
   if (canonical.empty()) {
      canonical = path;
   }
   // Unload a shared library or a source file.
   cling::Interpreter::CompilationResult compRes;
   cling::MetaProcessor::MaybeRedirectOutputRAII RAII(fMetaProcessor);
   fMetaProcessor->process(Form(".U %s", canonical.c_str()), compRes, /*cling::Value*/0);
   return compRes == cling::Interpreter::kFailure;
}

//______________________________________________________________________________
TInterpreterValue *TCling::CreateTemporary()
{
   // The created temporary must be deleted by the caller.

   TClingValue *val = new TClingValue;
   return val;
}

//______________________________________________________________________________
void TCling::RegisterTemporary(const TInterpreterValue& value)
{
   using namespace cling;
   const Value* V = reinterpret_cast<const Value*>(value.GetValAddr());
   RegisterTemporary(*V);
}

//______________________________________________________________________________
void TCling::RegisterTemporary(const cling::Value& value)
{
   // Register value as a temporary, extending its lifetime to that of the
   // interpreter. This is needed for TCling's compatibility interfaces
   // returning long - the address of the temporary objects.
   // As such, "simple" types don't need to be stored; they are returned by
   // value; only pointers / references / objects need to be stored.

   if (value.isValid() && value.needsManagedAllocation()) {
      R__LOCKGUARD(gInterpreterMutex);
      fTemporaries->push_back(value);
   }
}

//______________________________________________________________________________
void TCling::AddFriendToClass(clang::FunctionDecl* function,
                              clang::CXXRecordDecl* klass) const
{
   // Inject function as a friend into klass.
   // With function being f in void f() {new N::PrivKlass(); } this enables
   // I/O of non-public classes.
   using namespace clang;
   ASTContext& Ctx = klass->getASTContext();
   FriendDecl::FriendUnion friendUnion(function);
   // one dummy object for the source location
   SourceLocation sl;
   FriendDecl* friendDecl = FriendDecl::Create(Ctx, klass, sl, friendUnion, sl);
   klass->pushFriendDecl(friendDecl);
}

//______________________________________________________________________________
//
//  DeclId getter.
//

//______________________________________________________________________________
TInterpreter::DeclId_t TCling::GetDeclId(CallFunc_t* func) const
{
   // Return a unique identifier of the declaration represented by the
   // CallFunc

   if (func) return ((TClingCallFunc*)func)->GetDecl()->getCanonicalDecl();
   return 0;
}

//______________________________________________________________________________
TInterpreter::DeclId_t TCling::GetDeclId(ClassInfo_t* cinfo) const
{
   // Return a (almost) unique identifier of the declaration represented by the
   // ClassInfo.  In ROOT, this identifier can point to more than one TClass
   // when the underlying class is a template instance involving one of the
   // opaque typedef.

   if (cinfo) return ((TClingClassInfo*)cinfo)->GetDeclId();
   return 0;
}

//______________________________________________________________________________
TInterpreter::DeclId_t TCling::GetDeclId(DataMemberInfo_t* data) const
{
   // Return a unique identifier of the declaration represented by the
   // MethodInfo

   if (data) return ((TClingDataMemberInfo*)data)->GetDeclId();
   return 0;
}

//______________________________________________________________________________
TInterpreter::DeclId_t TCling::GetDeclId(MethodInfo_t* method) const
{
   // Return a unique identifier of the declaration represented by the
   // MethodInfo

   if (method) return ((TClingMethodInfo*)method)->GetDeclId();
   return 0;
}

//______________________________________________________________________________
TInterpreter::DeclId_t TCling::GetDeclId(TypedefInfo_t* tinfo) const
{
   // Return a unique identifier of the declaration represented by the
   // TypedefInfo

   if (tinfo) return ((TClingTypedefInfo*)tinfo)->GetDecl()->getCanonicalDecl();
   return 0;
}

//______________________________________________________________________________
//
//  CallFunc interface
//

//______________________________________________________________________________
void TCling::CallFunc_Delete(CallFunc_t* func) const
{
   delete (TClingCallFunc*) func;
}

//______________________________________________________________________________
void TCling::CallFunc_Exec(CallFunc_t* func, void* address) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->Exec(address);
}

//______________________________________________________________________________
void TCling::CallFunc_Exec(CallFunc_t* func, void* address, TInterpreterValue& val) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->Exec(address, &val);
}

//______________________________________________________________________________
void TCling::CallFunc_ExecWithReturn(CallFunc_t* func, void* address, void* ret) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->ExecWithReturn(address, ret);
}

//______________________________________________________________________________
void TCling::CallFunc_ExecWithArgsAndReturn(CallFunc_t* func, void* address,
                                            const void* args[] /*=0*/,
                                            int nargs /*=0*/,
                                            void* ret/*=0*/) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->ExecWithArgsAndReturn(address, args, nargs, ret);
}

//______________________________________________________________________________
Long_t TCling::CallFunc_ExecInt(CallFunc_t* func, void* address) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   return f->ExecInt(address);
}

//______________________________________________________________________________
Long64_t TCling::CallFunc_ExecInt64(CallFunc_t* func, void* address) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   return f->ExecInt64(address);
}

//______________________________________________________________________________
Double_t TCling::CallFunc_ExecDouble(CallFunc_t* func, void* address) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   return f->ExecDouble(address);
}

//______________________________________________________________________________
CallFunc_t* TCling::CallFunc_Factory() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return (CallFunc_t*) new TClingCallFunc(fInterpreter,*fNormalizedCtxt);
}

//______________________________________________________________________________
CallFunc_t* TCling::CallFunc_FactoryCopy(CallFunc_t* func) const
{
   return (CallFunc_t*) new TClingCallFunc(*(TClingCallFunc*)func);
}

//______________________________________________________________________________
MethodInfo_t* TCling::CallFunc_FactoryMethod(CallFunc_t* func) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   return (MethodInfo_t*) f->FactoryMethod();
}

//______________________________________________________________________________
void TCling::CallFunc_IgnoreExtraArgs(CallFunc_t* func, bool ignore) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->IgnoreExtraArgs(ignore);
}

//______________________________________________________________________________
void TCling::CallFunc_Init(CallFunc_t* func) const
{
   R__LOCKGUARD(gInterpreterMutex);
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->Init();
}

//______________________________________________________________________________
bool TCling::CallFunc_IsValid(CallFunc_t* func) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   return f->IsValid();
}

//______________________________________________________________________________
TInterpreter::CallFuncIFacePtr_t
TCling::CallFunc_IFacePtr(CallFunc_t * func) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   return f->IFacePtr();
}

//______________________________________________________________________________
void TCling::CallFunc_ResetArg(CallFunc_t* func) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->ResetArg();
}

//______________________________________________________________________________
void TCling::CallFunc_SetArg(CallFunc_t* func, Long_t param) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->SetArg(param);
}

//______________________________________________________________________________
void TCling::CallFunc_SetArg(CallFunc_t* func, ULong_t param) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->SetArg(param);
}

//______________________________________________________________________________
void TCling::CallFunc_SetArg(CallFunc_t* func, Float_t param) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->SetArg(param);
}

//______________________________________________________________________________
void TCling::CallFunc_SetArg(CallFunc_t* func, Double_t param) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->SetArg(param);
}

//______________________________________________________________________________
void TCling::CallFunc_SetArg(CallFunc_t* func, Long64_t param) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->SetArg(param);
}

//______________________________________________________________________________
void TCling::CallFunc_SetArg(CallFunc_t* func, ULong64_t param) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->SetArg(param);
}

//______________________________________________________________________________
void TCling::CallFunc_SetArgArray(CallFunc_t* func, Long_t* paramArr, Int_t nparam) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->SetArgArray(paramArr, nparam);
}

//______________________________________________________________________________
void TCling::CallFunc_SetArgs(CallFunc_t* func, const char* param) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->SetArgs(param);
}

//______________________________________________________________________________
void TCling::CallFunc_SetFunc(CallFunc_t* func, ClassInfo_t* info, const char* method, const char* params, Long_t* offset) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   TClingClassInfo* ci = (TClingClassInfo*) info;
   f->SetFunc(ci, method, params, offset);
}

//______________________________________________________________________________
void TCling::CallFunc_SetFunc(CallFunc_t* func, ClassInfo_t* info, const char* method, const char* params, bool objectIsConst, Long_t* offset) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   TClingClassInfo* ci = (TClingClassInfo*) info;
   f->SetFunc(ci, method, params, objectIsConst, offset);
}
//______________________________________________________________________________
void TCling::CallFunc_SetFunc(CallFunc_t* func, MethodInfo_t* info) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   TClingMethodInfo* minfo = (TClingMethodInfo*) info;
   f->SetFunc(minfo);
}

//______________________________________________________________________________
void TCling::CallFunc_SetFuncProto(CallFunc_t* func, ClassInfo_t* info, const char* method, const char* proto, Long_t* offset, EFunctionMatchMode mode /* = kConversionMatch */) const
{
   // Interface to cling function
   TClingCallFunc* f = (TClingCallFunc*) func;
   TClingClassInfo* ci = (TClingClassInfo*) info;
   f->SetFuncProto(ci, method, proto, offset, mode);
}

//______________________________________________________________________________
void TCling::CallFunc_SetFuncProto(CallFunc_t* func, ClassInfo_t* info, const char* method, const char* proto, bool objectIsConst, Long_t* offset, EFunctionMatchMode mode /* = kConversionMatch */) const
{
   // Interface to cling function
   TClingCallFunc* f = (TClingCallFunc*) func;
   TClingClassInfo* ci = (TClingClassInfo*) info;
   f->SetFuncProto(ci, method, proto, objectIsConst, offset, mode);
}

//______________________________________________________________________________
void TCling::CallFunc_SetFuncProto(CallFunc_t* func, ClassInfo_t* info, const char* method, const std::vector<TypeInfo_t*> &proto, Long_t* offset, EFunctionMatchMode mode /* = kConversionMatch */) const
{
   // Interface to cling function
   TClingCallFunc* f = (TClingCallFunc*) func;
   TClingClassInfo* ci = (TClingClassInfo*) info;
   llvm::SmallVector<clang::QualType, 4> funcProto;
   for (std::vector<TypeInfo_t*>::const_iterator iter = proto.begin(), end = proto.end();
        iter != end; ++iter) {
      funcProto.push_back( ((TClingTypeInfo*)(*iter))->GetQualType() );
   }
   f->SetFuncProto(ci, method, funcProto, offset, mode);
}

//______________________________________________________________________________
void TCling::CallFunc_SetFuncProto(CallFunc_t* func, ClassInfo_t* info, const char* method, const std::vector<TypeInfo_t*> &proto, bool objectIsConst, Long_t* offset, EFunctionMatchMode mode /* = kConversionMatch */) const
{
   // Interface to cling function
   TClingCallFunc* f = (TClingCallFunc*) func;
   TClingClassInfo* ci = (TClingClassInfo*) info;
   llvm::SmallVector<clang::QualType, 4> funcProto;
   for (std::vector<TypeInfo_t*>::const_iterator iter = proto.begin(), end = proto.end();
        iter != end; ++iter) {
      funcProto.push_back( ((TClingTypeInfo*)(*iter))->GetQualType() );
   }
   f->SetFuncProto(ci, method, funcProto, objectIsConst, offset, mode);
}

//______________________________________________________________________________
//
//  ClassInfo interface
//

//______________________________________________________________________________
Bool_t TCling::ClassInfo_Contains(ClassInfo_t *info, DeclId_t declid) const
{
   // Return true if the entity pointed to by 'declid' is declared in
   // the context described by 'info'.  If info is null, look into the
   // global scope (translation unit scope).

   if (!declid) return kFALSE;

   const clang::Decl *scope;
   if (info) scope = ((TClingClassInfo*)info)->GetDecl();
   else scope = fInterpreter->getCI()->getASTContext().getTranslationUnitDecl();

   const clang::Decl *decl = reinterpret_cast<const clang::Decl*>(declid);
   const clang::DeclContext *ctxt = clang::Decl::castToDeclContext(scope);
   if (!decl || !ctxt) return kFALSE;
   if (decl->getDeclContext()->Equals(ctxt))
      return kTRUE;
   else if (decl->getDeclContext()->isTransparentContext() &&
            decl->getDeclContext()->getParent()->Equals(ctxt))
      return kTRUE;
   return kFALSE;
}

//______________________________________________________________________________
Long_t TCling::ClassInfo_ClassProperty(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->ClassProperty();
}

//______________________________________________________________________________
void TCling::ClassInfo_Delete(ClassInfo_t* cinfo) const
{
   delete (TClingClassInfo*) cinfo;
}

//______________________________________________________________________________
void TCling::ClassInfo_Delete(ClassInfo_t* cinfo, void* arena) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   TClinginfo->Delete(arena,*fNormalizedCtxt);
}

//______________________________________________________________________________
void TCling::ClassInfo_DeleteArray(ClassInfo_t* cinfo, void* arena, bool dtorOnly) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   TClinginfo->DeleteArray(arena, dtorOnly,*fNormalizedCtxt);
}

//______________________________________________________________________________
void TCling::ClassInfo_Destruct(ClassInfo_t* cinfo, void* arena) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   TClinginfo->Destruct(arena,*fNormalizedCtxt);
}

//______________________________________________________________________________
ClassInfo_t* TCling::ClassInfo_Factory(Bool_t all) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return (ClassInfo_t*) new TClingClassInfo(fInterpreter, all);
}

//______________________________________________________________________________
ClassInfo_t* TCling::ClassInfo_Factory(ClassInfo_t* cinfo) const
{
   return (ClassInfo_t*) new TClingClassInfo(*(TClingClassInfo*)cinfo);
}

//______________________________________________________________________________
ClassInfo_t* TCling::ClassInfo_Factory(const char* name) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return (ClassInfo_t*) new TClingClassInfo(fInterpreter, name);
}

//______________________________________________________________________________
int TCling::ClassInfo_GetMethodNArg(ClassInfo_t* cinfo, const char* method, const char* proto, Bool_t objectIsConst /* = false */, EFunctionMatchMode mode /* = kConversionMatch */) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->GetMethodNArg(method, proto, objectIsConst, mode);
}

//______________________________________________________________________________
bool TCling::ClassInfo_HasDefaultConstructor(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->HasDefaultConstructor();
}

//______________________________________________________________________________
bool TCling::ClassInfo_HasMethod(ClassInfo_t* cinfo, const char* name) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->HasMethod(name);
}

//______________________________________________________________________________
void TCling::ClassInfo_Init(ClassInfo_t* cinfo, const char* name) const
{
   R__LOCKGUARD(gInterpreterMutex);
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   TClinginfo->Init(name);
}

//______________________________________________________________________________
void TCling::ClassInfo_Init(ClassInfo_t* cinfo, int tagnum) const
{
   R__LOCKGUARD(gInterpreterMutex);
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   TClinginfo->Init(tagnum);
}

//______________________________________________________________________________
bool TCling::ClassInfo_IsBase(ClassInfo_t* cinfo, const char* name) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->IsBase(name);
}

//______________________________________________________________________________
bool TCling::ClassInfo_IsEnum(const char* name) const
{
   return TClingClassInfo::IsEnum(fInterpreter, name);
}

//______________________________________________________________________________
bool TCling::ClassInfo_IsLoaded(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->IsLoaded();
}

//______________________________________________________________________________
bool TCling::ClassInfo_IsValid(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->IsValid();
}

//______________________________________________________________________________
bool TCling::ClassInfo_IsValidMethod(ClassInfo_t* cinfo, const char* method, const char* proto, Long_t* offset, EFunctionMatchMode mode /* = kConversionMatch */) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->IsValidMethod(method, proto, false, offset, mode);
}

//______________________________________________________________________________
bool TCling::ClassInfo_IsValidMethod(ClassInfo_t* cinfo, const char* method, const char* proto, Bool_t objectIsConst, Long_t* offset, EFunctionMatchMode mode /* = kConversionMatch */) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->IsValidMethod(method, proto, objectIsConst, offset, mode);
}

//______________________________________________________________________________
int TCling::ClassInfo_Next(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->Next();
}

//______________________________________________________________________________
void* TCling::ClassInfo_New(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->New(*fNormalizedCtxt);
}

//______________________________________________________________________________
void* TCling::ClassInfo_New(ClassInfo_t* cinfo, int n) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->New(n,*fNormalizedCtxt);
}

//______________________________________________________________________________
void* TCling::ClassInfo_New(ClassInfo_t* cinfo, int n, void* arena) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->New(n, arena,*fNormalizedCtxt);
}

//______________________________________________________________________________
void* TCling::ClassInfo_New(ClassInfo_t* cinfo, void* arena) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->New(arena,*fNormalizedCtxt);
}

//______________________________________________________________________________
Long_t TCling::ClassInfo_Property(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->Property();
}

//______________________________________________________________________________
int TCling::ClassInfo_Size(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->Size();
}

//______________________________________________________________________________
Long_t TCling::ClassInfo_Tagnum(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->Tagnum();
}

//______________________________________________________________________________
const char* TCling::ClassInfo_FileName(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->FileName();
}

//______________________________________________________________________________
const char* TCling::ClassInfo_FullName(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   TTHREAD_TLS_DECL(std::string,output);
   TClinginfo->FullName(output,*fNormalizedCtxt);
   return output.c_str();
}

//______________________________________________________________________________
const char* TCling::ClassInfo_Name(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->Name();
}

//______________________________________________________________________________
const char* TCling::ClassInfo_Title(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->Title();
}

//______________________________________________________________________________
const char* TCling::ClassInfo_TmpltName(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->TmpltName();
}



//______________________________________________________________________________
//
//  BaseClassInfo interface
//

//______________________________________________________________________________
void TCling::BaseClassInfo_Delete(BaseClassInfo_t* bcinfo) const
{
   delete(TClingBaseClassInfo*) bcinfo;
}

//______________________________________________________________________________
BaseClassInfo_t* TCling::BaseClassInfo_Factory(ClassInfo_t* cinfo) const
{
   R__LOCKGUARD(gInterpreterMutex);
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return (BaseClassInfo_t*) new TClingBaseClassInfo(fInterpreter, TClinginfo);
}

//______________________________________________________________________________
BaseClassInfo_t* TCling::BaseClassInfo_Factory(ClassInfo_t* derived,
   ClassInfo_t* base) const
{
   R__LOCKGUARD(gInterpreterMutex);
   TClingClassInfo* TClinginfo = (TClingClassInfo*) derived;
   TClingClassInfo* TClinginfoBase = (TClingClassInfo*) base;
   return (BaseClassInfo_t*) new TClingBaseClassInfo(fInterpreter, TClinginfo, TClinginfoBase);
}

//______________________________________________________________________________
int TCling::BaseClassInfo_Next(BaseClassInfo_t* bcinfo) const
{
   TClingBaseClassInfo* TClinginfo = (TClingBaseClassInfo*) bcinfo;
   return TClinginfo->Next();
}

//______________________________________________________________________________
int TCling::BaseClassInfo_Next(BaseClassInfo_t* bcinfo, int onlyDirect) const
{
   TClingBaseClassInfo* TClinginfo = (TClingBaseClassInfo*) bcinfo;
   return TClinginfo->Next(onlyDirect);
}

//______________________________________________________________________________
Long_t TCling::BaseClassInfo_Offset(BaseClassInfo_t* toBaseClassInfo, void * address, bool isDerivedObject) const
{
   TClingBaseClassInfo* TClinginfo = (TClingBaseClassInfo*) toBaseClassInfo;
   return TClinginfo->Offset(address, isDerivedObject);
}

//______________________________________________________________________________
Long_t TCling::ClassInfo_GetBaseOffset(ClassInfo_t* fromDerived, ClassInfo_t* toBase, void * address, bool isDerivedObject) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) fromDerived;
   TClingClassInfo* TClinginfoBase = (TClingClassInfo*) toBase;
   // Offset to the class itself.
   if (TClinginfo->GetDecl() == TClinginfoBase->GetDecl()) {
      return 0;
   }
   return TClinginfo->GetBaseOffset(TClinginfoBase, address, isDerivedObject);
}

//______________________________________________________________________________
Long_t TCling::BaseClassInfo_Property(BaseClassInfo_t* bcinfo) const
{
   TClingBaseClassInfo* TClinginfo = (TClingBaseClassInfo*) bcinfo;
   return TClinginfo->Property();
}

//______________________________________________________________________________
ClassInfo_t *TCling::BaseClassInfo_ClassInfo(BaseClassInfo_t *bcinfo) const
{
   TClingBaseClassInfo* TClinginfo = (TClingBaseClassInfo*) bcinfo;
   return (ClassInfo_t *)TClinginfo->GetBase();
}

//______________________________________________________________________________
Long_t TCling::BaseClassInfo_Tagnum(BaseClassInfo_t* bcinfo) const
{
   TClingBaseClassInfo* TClinginfo = (TClingBaseClassInfo*) bcinfo;
   return TClinginfo->Tagnum();
}

//______________________________________________________________________________
const char* TCling::BaseClassInfo_FullName(BaseClassInfo_t* bcinfo) const
{
   TClingBaseClassInfo* TClinginfo = (TClingBaseClassInfo*) bcinfo;
   TTHREAD_TLS_DECL(std::string,output);
   TClinginfo->FullName(output,*fNormalizedCtxt);
   return output.c_str();
}

//______________________________________________________________________________
const char* TCling::BaseClassInfo_Name(BaseClassInfo_t* bcinfo) const
{
   TClingBaseClassInfo* TClinginfo = (TClingBaseClassInfo*) bcinfo;
   return TClinginfo->Name();
}

//______________________________________________________________________________
const char* TCling::BaseClassInfo_TmpltName(BaseClassInfo_t* bcinfo) const
{
   TClingBaseClassInfo* TClinginfo = (TClingBaseClassInfo*) bcinfo;
   return TClinginfo->TmpltName();
}

//______________________________________________________________________________
//
//  DataMemberInfo interface
//

//______________________________________________________________________________
int TCling::DataMemberInfo_ArrayDim(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->ArrayDim();
}

//______________________________________________________________________________
void TCling::DataMemberInfo_Delete(DataMemberInfo_t* dminfo) const
{
   delete(TClingDataMemberInfo*) dminfo;
}

//______________________________________________________________________________
DataMemberInfo_t* TCling::DataMemberInfo_Factory(ClassInfo_t* clinfo /*= 0*/) const
{
   R__LOCKGUARD(gInterpreterMutex);
   TClingClassInfo* TClingclass_info = (TClingClassInfo*) clinfo;
   return (DataMemberInfo_t*) new TClingDataMemberInfo(fInterpreter, TClingclass_info);
}

//______________________________________________________________________________
DataMemberInfo_t* TCling::DataMemberInfo_Factory(DeclId_t declid, ClassInfo_t* clinfo) const
{
   R__LOCKGUARD(gInterpreterMutex);
   const clang::Decl* decl = reinterpret_cast<const clang::Decl*>(declid);
   const clang::ValueDecl* vd = llvm::dyn_cast_or_null<clang::ValueDecl>(decl);
   return (DataMemberInfo_t*) new TClingDataMemberInfo(fInterpreter, vd, (TClingClassInfo*)clinfo);
}

//______________________________________________________________________________
DataMemberInfo_t* TCling::DataMemberInfo_FactoryCopy(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return (DataMemberInfo_t*) new TClingDataMemberInfo(*TClinginfo);
}

//______________________________________________________________________________
bool TCling::DataMemberInfo_IsValid(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->IsValid();
}

//______________________________________________________________________________
int TCling::DataMemberInfo_MaxIndex(DataMemberInfo_t* dminfo, Int_t dim) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->MaxIndex(dim);
}

//______________________________________________________________________________
int TCling::DataMemberInfo_Next(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->Next();
}

//______________________________________________________________________________
Long_t TCling::DataMemberInfo_Offset(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->Offset();
}

//______________________________________________________________________________
Long_t TCling::DataMemberInfo_Property(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->Property();
}

//______________________________________________________________________________
Long_t TCling::DataMemberInfo_TypeProperty(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->TypeProperty();
}

//______________________________________________________________________________
int TCling::DataMemberInfo_TypeSize(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->TypeSize();
}

//______________________________________________________________________________
const char* TCling::DataMemberInfo_TypeName(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->TypeName();
}

//______________________________________________________________________________
const char* TCling::DataMemberInfo_TypeTrueName(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->TypeTrueName(*fNormalizedCtxt);
}

//______________________________________________________________________________
const char* TCling::DataMemberInfo_Name(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->Name();
}

//______________________________________________________________________________
const char* TCling::DataMemberInfo_Title(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->Title();
}

//______________________________________________________________________________
const char* TCling::DataMemberInfo_ValidArrayIndex(DataMemberInfo_t* dminfo) const
{
   TTHREAD_TLS_DECL(std::string,result);

   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   result = TClinginfo->ValidArrayIndex().str();
   return result.c_str();
}

//______________________________________________________________________________
void TCling::SetDeclAttr(DeclId_t declId, const char* attribute)
{
   Decl* decl = static_cast<Decl*>(const_cast<void*>(declId));
   ASTContext &C = decl->getASTContext();
   SourceRange commentRange; // this is a fake comment range
   decl->addAttr( new (C) AnnotateAttr( commentRange, C, attribute, 0 ) );
}

//______________________________________________________________________________
//
// Function Template interface
//

//______________________________________________________________________________
static void ConstructorName(std::string &name, const clang::NamedDecl *decl,
                            cling::Interpreter &interp,
                            const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt)
{
   const clang::TypeDecl* td = llvm::dyn_cast<clang::TypeDecl>(decl->getDeclContext());
   if (!td) return;

   clang::QualType qualType(td->getTypeForDecl(),0);
   ROOT::TMetaUtils::GetNormalizedName(name, qualType, interp, normCtxt);
   unsigned int level = 0;
   for(size_t cursor = name.length()-1; cursor != 0; --cursor) {
      if (name[cursor] == '>') ++level;
      else if (name[cursor] == '<' && level) --level;
      else if (level == 0 && name[cursor] == ':') {
         name.erase(0,cursor+1);
         break;
      }
   }
}

//______________________________________________________________________________
void TCling::GetFunctionName(const clang::FunctionDecl *decl, std::string &output) const
{
   output.clear();
   if (llvm::isa<clang::CXXConstructorDecl>(decl))
   {
      ConstructorName(output, decl, *fInterpreter, *fNormalizedCtxt);

   } else if (llvm::isa<clang::CXXDestructorDecl>(decl))
   {
      ConstructorName(output, decl, *fInterpreter, *fNormalizedCtxt);
      output.insert(output.begin(), '~');
   } else {
      llvm::raw_string_ostream stream(output);
      decl->getNameForDiagnostic(stream, decl->getASTContext().getPrintingPolicy(), /*Qualified=*/false);
   }
}

//______________________________________________________________________________
TInterpreter::DeclId_t TCling::GetDeclId(FuncTempInfo_t *info) const
{
   // Return a unique identifier of the declaration represented by the
   // FuncTempInfo

   return (DeclId_t)info;
}

//______________________________________________________________________________
void   TCling::FuncTempInfo_Delete(FuncTempInfo_t * /* ft_info */) const
{
   // Delete the FuncTempInfo_t

   // Currently the address of ft_info is actually the decl itself,
   // so we have nothing to do.
}

//______________________________________________________________________________
FuncTempInfo_t *TCling::FuncTempInfo_Factory(DeclId_t declid) const
{
   // Construct a FuncTempInfo_t

   // Currently the address of ft_info is actually the decl itself,
   // so we have nothing to do.

   return (FuncTempInfo_t*)const_cast<void*>(declid);
}

//______________________________________________________________________________
FuncTempInfo_t *TCling::FuncTempInfo_FactoryCopy(FuncTempInfo_t *ft_info) const
{
   // Construct a FuncTempInfo_t

   // Currently the address of ft_info is actually the decl itself,
   // so we have nothing to do.

   return (FuncTempInfo_t*)ft_info;
}

//______________________________________________________________________________
Bool_t TCling::FuncTempInfo_IsValid(FuncTempInfo_t *t_info) const
{
   // Check validity of a FuncTempInfo_t

   // Currently the address of ft_info is actually the decl itself,
   // so we have nothing to do.

   return t_info != 0;
}

//______________________________________________________________________________
UInt_t TCling::FuncTempInfo_TemplateNargs(FuncTempInfo_t *ft_info) const
{
   // Return the maximum number of template arguments of the
   // function template described by ft_info.

   if (!ft_info) return 0;
   const clang::FunctionTemplateDecl *ft = (const clang::FunctionTemplateDecl*)ft_info;
   return ft->getTemplateParameters()->size();
}

//______________________________________________________________________________
UInt_t TCling::FuncTempInfo_TemplateMinReqArgs(FuncTempInfo_t *ft_info) const
{
   // Return the number of required template arguments of the
   // function template described by ft_info.

   if (!ft_info) return 0;
   const clang::FunctionTemplateDecl *ft = (clang::FunctionTemplateDecl*)ft_info;
   return ft->getTemplateParameters()->getMinRequiredArguments();
}

//______________________________________________________________________________
Long_t TCling::FuncTempInfo_Property(FuncTempInfo_t *ft_info) const
{
   // Return the property of the function template.

   if (!ft_info) return 0;

   long property = 0L;
   property |= kIsCompiled;

   const clang::FunctionTemplateDecl *ft = (clang::FunctionTemplateDecl*)ft_info;

   switch (ft->getAccess()) {
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
         if (ft->getDeclContext()->isNamespace())
            property |= kIsPublic;
         break;
      default:
         // IMPOSSIBLE
         break;
   }

   const clang::FunctionDecl *fd = ft->getTemplatedDecl();
   if (const clang::CXXMethodDecl *md =
       llvm::dyn_cast<clang::CXXMethodDecl>(fd)) {
      if (md->getTypeQualifiers() & clang::Qualifiers::Const) {
         property |= kIsConstant | kIsConstMethod;
      }
      if (md->isVirtual()) {
         property |= kIsVirtual;
      }
      if (md->isPure()) {
         property |= kIsPureVirtual;
      }
      if (const clang::CXXConstructorDecl *cd =
          llvm::dyn_cast<clang::CXXConstructorDecl>(md)) {
         if (cd->isExplicit()) {
            property |= kIsExplicit;
         }
      }
      else if (const clang::CXXConversionDecl *cd =
               llvm::dyn_cast<clang::CXXConversionDecl>(md)) {
         if (cd->isExplicit()) {
            property |= kIsExplicit;
         }
      }
   }
   return property;
}

//______________________________________________________________________________
void TCling::FuncTempInfo_Name(FuncTempInfo_t *ft_info, TString &output) const
{
   // Return the name of this function template.

   output.Clear();
   if (!ft_info) return;
   const clang::FunctionTemplateDecl *ft = (clang::FunctionTemplateDecl*)ft_info;
   std::string buf;
   GetFunctionName(ft->getTemplatedDecl(), buf);
   output = buf;
}

//______________________________________________________________________________
void TCling::FuncTempInfo_Title(FuncTempInfo_t *ft_info, TString &output) const
{
   // Return the comments associates with this function template.

   output.Clear();
   if (!ft_info) return;
   const clang::FunctionTemplateDecl *ft = (const clang::FunctionTemplateDecl*)ft_info;

   // Iterate over the redeclarations, we can have muliple definitions in the
   // redecl chain (came from merging of pcms).
   if (const RedeclarableTemplateDecl *AnnotFD
       = ROOT::TMetaUtils::GetAnnotatedRedeclarable((const RedeclarableTemplateDecl*)ft)) {
      if (AnnotateAttr *A = AnnotFD->getAttr<AnnotateAttr>()) {
         output = A->getAnnotation().str();
         return;
      }
   }
   if (!ft->isFromASTFile()) {
      // Try to get the comment from the header file if present
      // but not for decls from AST file, where rootcling would have
      // created an annotation
      output = ROOT::TMetaUtils::GetComment(*ft).str();
   }
}


//______________________________________________________________________________
//
//  MethodInfo interface
//

//______________________________________________________________________________
void TCling::MethodInfo_Delete(MethodInfo_t* minfo) const
{
   // Interface to cling function
   delete(TClingMethodInfo*) minfo;
}

//______________________________________________________________________________
void TCling::MethodInfo_CreateSignature(MethodInfo_t* minfo, TString& signature) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   info->CreateSignature(signature);
}

//______________________________________________________________________________
MethodInfo_t* TCling::MethodInfo_Factory() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return (MethodInfo_t*) new TClingMethodInfo(fInterpreter);
}

//______________________________________________________________________________
MethodInfo_t* TCling::MethodInfo_Factory(ClassInfo_t* clinfo) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return (MethodInfo_t*) new TClingMethodInfo(fInterpreter, (TClingClassInfo*)clinfo);
}

//______________________________________________________________________________
MethodInfo_t* TCling::MethodInfo_Factory(DeclId_t declid) const
{
   const clang::Decl* decl = reinterpret_cast<const clang::Decl*>(declid);
   R__LOCKGUARD(gInterpreterMutex);
   const clang::FunctionDecl* fd = llvm::dyn_cast_or_null<clang::FunctionDecl>(decl);
   return (MethodInfo_t*) new TClingMethodInfo(fInterpreter, fd);
}

//______________________________________________________________________________
MethodInfo_t* TCling::MethodInfo_FactoryCopy(MethodInfo_t* minfo) const
{
   return (MethodInfo_t*) new TClingMethodInfo(*(TClingMethodInfo*)minfo);
}

//______________________________________________________________________________
void* TCling::MethodInfo_InterfaceMethod(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return info->InterfaceMethod(*fNormalizedCtxt);
}

//______________________________________________________________________________
bool TCling::MethodInfo_IsValid(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return info->IsValid();
}

//______________________________________________________________________________
int TCling::MethodInfo_NArg(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return info->NArg();
}

//______________________________________________________________________________
int TCling::MethodInfo_NDefaultArg(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return info->NDefaultArg();
}

//______________________________________________________________________________
int TCling::MethodInfo_Next(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return info->Next();
}

//______________________________________________________________________________
Long_t TCling::MethodInfo_Property(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return info->Property();
}

//______________________________________________________________________________
Long_t TCling::MethodInfo_ExtraProperty(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return info->ExtraProperty();
}

//______________________________________________________________________________
TypeInfo_t* TCling::MethodInfo_Type(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return (TypeInfo_t*)info->Type();
}

//______________________________________________________________________________
const char* TCling::MethodInfo_GetMangledName(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   TTHREAD_TLS_DECL(TString, mangled_name);
   mangled_name = info->GetMangledName();
   return mangled_name;
}

//______________________________________________________________________________
const char* TCling::MethodInfo_GetPrototype(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return info->GetPrototype(*fNormalizedCtxt);
}

//______________________________________________________________________________
const char* TCling::MethodInfo_Name(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return info->Name(*fNormalizedCtxt);
}

//______________________________________________________________________________
const char* TCling::MethodInfo_TypeName(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return info->TypeName();
}

//______________________________________________________________________________
std::string TCling::MethodInfo_TypeNormalizedName(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   if (info && info->IsValid())
      return info->Type()->NormalizedName(*fNormalizedCtxt);
   else
      return "";
}

//______________________________________________________________________________
const char* TCling::MethodInfo_Title(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return info->Title();
}

//______________________________________________________________________________
auto TCling::MethodCallReturnType(TFunction *func) const -> EReturnType
{
   if (func) {
      return MethodInfo_MethodCallReturnType(func->fInfo);
   } else {
      return EReturnType::kOther;
   }
}

//______________________________________________________________________________
auto TCling::MethodInfo_MethodCallReturnType(MethodInfo_t* minfo) const -> EReturnType
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   if (info && info->IsValid()) {
      TClingTypeInfo *typeinfo = info->Type();
      clang::QualType QT( typeinfo->GetQualType().getCanonicalType() );
      if (QT->isEnumeralType()) {
         return EReturnType::kLong;
      } else if (QT->isPointerType()) {
         // Look for char*
         QT = llvm::cast<clang::PointerType>(QT)->getPointeeType();
         if ( QT->isCharType() ) {
            return EReturnType::kString;
         } else {
            return EReturnType::kOther;
         }
      } else if ( QT->isFloatingType() ) {
         int sz = typeinfo->Size();
         if (sz == 4 || sz == 8) {
            // Support only float and double.
            return EReturnType::kDouble;
         } else {
            return EReturnType::kOther;
         }
      } else if ( QT->isIntegerType() ) {
         int sz = typeinfo->Size();
         if (sz <= 8) {
            // Support only up to long long ... but
            // FIXME the TMethodCall::Execute only
            // return long (4 bytes) ...
            // The v5 implementation of TMethodCall::ReturnType
            // was not making the distinction so we let it go
            // as is for now, but we really need to upgrade
            // TMethodCall::Execute ...
            return EReturnType::kLong;
         } else {
            return EReturnType::kOther;
         }
      } else {
         return EReturnType::kOther;
      }
   } else {
      return EReturnType::kOther;
   }
}

//______________________________________________________________________________
//
//  MethodArgInfo interface
//

//______________________________________________________________________________
void TCling::MethodArgInfo_Delete(MethodArgInfo_t* marginfo) const
{
   delete(TClingMethodArgInfo*) marginfo;
}

//______________________________________________________________________________
MethodArgInfo_t* TCling::MethodArgInfo_Factory() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return (MethodArgInfo_t*) new TClingMethodArgInfo(fInterpreter);
}

//______________________________________________________________________________
MethodArgInfo_t* TCling::MethodArgInfo_Factory(MethodInfo_t *minfo) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return (MethodArgInfo_t*) new TClingMethodArgInfo(fInterpreter, (TClingMethodInfo*)minfo);
}

//______________________________________________________________________________
MethodArgInfo_t* TCling::MethodArgInfo_FactoryCopy(MethodArgInfo_t* marginfo) const
{
   return (MethodArgInfo_t*)
          new TClingMethodArgInfo(*(TClingMethodArgInfo*)marginfo);
}

//______________________________________________________________________________
bool TCling::MethodArgInfo_IsValid(MethodArgInfo_t* marginfo) const
{
   TClingMethodArgInfo* info = (TClingMethodArgInfo*) marginfo;
   return info->IsValid();
}

//______________________________________________________________________________
int TCling::MethodArgInfo_Next(MethodArgInfo_t* marginfo) const
{
   TClingMethodArgInfo* info = (TClingMethodArgInfo*) marginfo;
   return info->Next();
}

//______________________________________________________________________________
Long_t TCling::MethodArgInfo_Property(MethodArgInfo_t* marginfo) const
{
   TClingMethodArgInfo* info = (TClingMethodArgInfo*) marginfo;
   return info->Property();
}

//______________________________________________________________________________
const char* TCling::MethodArgInfo_DefaultValue(MethodArgInfo_t* marginfo) const
{
   TClingMethodArgInfo* info = (TClingMethodArgInfo*) marginfo;
   return info->DefaultValue();
}

//______________________________________________________________________________
const char* TCling::MethodArgInfo_Name(MethodArgInfo_t* marginfo) const
{
   TClingMethodArgInfo* info = (TClingMethodArgInfo*) marginfo;
   return info->Name();
}

//______________________________________________________________________________
const char* TCling::MethodArgInfo_TypeName(MethodArgInfo_t* marginfo) const
{
   TClingMethodArgInfo* info = (TClingMethodArgInfo*) marginfo;
   return info->TypeName();
}

//______________________________________________________________________________
std::string TCling::MethodArgInfo_TypeNormalizedName(MethodArgInfo_t* marginfo) const
{
   TClingMethodArgInfo* info = (TClingMethodArgInfo*) marginfo;
   return info->Type()->NormalizedName(*fNormalizedCtxt);
}

//______________________________________________________________________________
//
//  TypeInfo interface
//

//______________________________________________________________________________
void TCling::TypeInfo_Delete(TypeInfo_t* tinfo) const
{
   delete (TClingTypeInfo*) tinfo;
}

//______________________________________________________________________________
TypeInfo_t* TCling::TypeInfo_Factory() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return (TypeInfo_t*) new TClingTypeInfo(fInterpreter);
}

//______________________________________________________________________________
TypeInfo_t* TCling::TypeInfo_Factory(const char *name) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return (TypeInfo_t*) new TClingTypeInfo(fInterpreter, name);
}

//______________________________________________________________________________
TypeInfo_t* TCling::TypeInfo_FactoryCopy(TypeInfo_t* tinfo) const
{
   return (TypeInfo_t*) new TClingTypeInfo(*(TClingTypeInfo*)tinfo);
}

//______________________________________________________________________________
void TCling::TypeInfo_Init(TypeInfo_t* tinfo, const char* name) const
{
   R__LOCKGUARD(gInterpreterMutex);
   TClingTypeInfo* TClinginfo = (TClingTypeInfo*) tinfo;
   TClinginfo->Init(name);
}

//______________________________________________________________________________
bool TCling::TypeInfo_IsValid(TypeInfo_t* tinfo) const
{
   TClingTypeInfo* TClinginfo = (TClingTypeInfo*) tinfo;
   return TClinginfo->IsValid();
}

//______________________________________________________________________________
const char* TCling::TypeInfo_Name(TypeInfo_t* tinfo) const
{
   TClingTypeInfo* TClinginfo = (TClingTypeInfo*) tinfo;
   return TClinginfo->Name();
}

//______________________________________________________________________________
Long_t TCling::TypeInfo_Property(TypeInfo_t* tinfo) const
{
   TClingTypeInfo* TClinginfo = (TClingTypeInfo*) tinfo;
   return TClinginfo->Property();
}

//______________________________________________________________________________
int TCling::TypeInfo_RefType(TypeInfo_t* tinfo) const
{
   TClingTypeInfo* TClinginfo = (TClingTypeInfo*) tinfo;
   return TClinginfo->RefType();
}

//______________________________________________________________________________
int TCling::TypeInfo_Size(TypeInfo_t* tinfo) const
{
   TClingTypeInfo* TClinginfo = (TClingTypeInfo*) tinfo;
   return TClinginfo->Size();
}

//______________________________________________________________________________
const char* TCling::TypeInfo_TrueName(TypeInfo_t* tinfo) const
{
   TClingTypeInfo* TClinginfo = (TClingTypeInfo*) tinfo;
   return TClinginfo->TrueName(*fNormalizedCtxt);
}


//______________________________________________________________________________
//
//  TypedefInfo interface
//

//______________________________________________________________________________
void TCling::TypedefInfo_Delete(TypedefInfo_t* tinfo) const
{
   delete(TClingTypedefInfo*) tinfo;
}

//______________________________________________________________________________
TypedefInfo_t* TCling::TypedefInfo_Factory() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return (TypedefInfo_t*) new TClingTypedefInfo(fInterpreter);
}

//______________________________________________________________________________
TypedefInfo_t* TCling::TypedefInfo_Factory(const char *name) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return (TypedefInfo_t*) new TClingTypedefInfo(fInterpreter, name);
}

//______________________________________________________________________________
TypedefInfo_t* TCling::TypedefInfo_FactoryCopy(TypedefInfo_t* tinfo) const
{
   return (TypedefInfo_t*) new TClingTypedefInfo(*(TClingTypedefInfo*)tinfo);
}

//______________________________________________________________________________
void TCling::TypedefInfo_Init(TypedefInfo_t* tinfo,
                              const char* name) const
{
   R__LOCKGUARD(gInterpreterMutex);
   TClingTypedefInfo* TClinginfo = (TClingTypedefInfo*) tinfo;
   TClinginfo->Init(name);
}

//______________________________________________________________________________
bool TCling::TypedefInfo_IsValid(TypedefInfo_t* tinfo) const
{
   TClingTypedefInfo* TClinginfo = (TClingTypedefInfo*) tinfo;
   return TClinginfo->IsValid();
}

//______________________________________________________________________________
Int_t TCling::TypedefInfo_Next(TypedefInfo_t* tinfo) const
{
   TClingTypedefInfo* TClinginfo = (TClingTypedefInfo*) tinfo;
   return TClinginfo->Next();
}

//______________________________________________________________________________
Long_t TCling::TypedefInfo_Property(TypedefInfo_t* tinfo) const
{
   TClingTypedefInfo* TClinginfo = (TClingTypedefInfo*) tinfo;
   return TClinginfo->Property();
}

//______________________________________________________________________________
int TCling::TypedefInfo_Size(TypedefInfo_t* tinfo) const
{
   TClingTypedefInfo* TClinginfo = (TClingTypedefInfo*) tinfo;
   return TClinginfo->Size();
}

//______________________________________________________________________________
const char* TCling::TypedefInfo_TrueName(TypedefInfo_t* tinfo) const
{
   TClingTypedefInfo* TClinginfo = (TClingTypedefInfo*) tinfo;
   return TClinginfo->TrueName(*fNormalizedCtxt);
}

//______________________________________________________________________________
const char* TCling::TypedefInfo_Name(TypedefInfo_t* tinfo) const
{
   TClingTypedefInfo* TClinginfo = (TClingTypedefInfo*) tinfo;
   return TClinginfo->Name();
}

//______________________________________________________________________________
const char* TCling::TypedefInfo_Title(TypedefInfo_t* tinfo) const
{
   TClingTypedefInfo* TClinginfo = (TClingTypedefInfo*) tinfo;
   return TClinginfo->Title();
}
