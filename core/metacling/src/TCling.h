// @(#)root/meta:$Id$
// Author: Axel Naumann, 2011-10-19

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TCling
#define ROOT_TCling

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCling                                                               //
//                                                                      //
// This class defines an interface to the cling C++ interpreter.        //
//                                                                      //
// Cling is a full ANSI compliant C++ interpreter based on              //
// clang/LLVM technology.                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TInterpreter.h"

#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <utility>

#ifndef WIN32
#define TWin32SendClass char
#endif

namespace llvm {
   class GlobalValue;
   class StringRef;
}

namespace clang {
   class CXXRecordDecl;
   class Decl;
   class DeclContext;
   class EnumDecl;
   class FunctionDecl;
   class NamedDecl;
   class NamespaceDecl;
   class TagDecl;
   class Type;
   class QualType;
}
namespace cling {
   class Interpreter;
   class MetaProcessor;
   class Transaction;
   class Value;
}

class TClingCallbacks;
class TEnv;
class TFile;
class THashTable;
class TInterpreterValue;
class TMethod;
class TObjArray;
class TListOfDataMembers;
class TListOfFunctions;
class TListOfFunctionTemplates;
class TListOfEnums;

namespace ROOT {
   namespace TMetaUtils {
      class TNormalizedCtxt;
      class TClingLookupHelper;
   }
}

extern "C" {
   void TCling__UpdateListsOnCommitted(const cling::Transaction&,
                                       cling::Interpreter*);
   void TCling__UpdateListsOnUnloaded(const cling::Transaction&);
   void TCling__InvalidateGlobal(const clang::Decl*);
   void TCling__TransactionRollback(const cling::Transaction&);
   TObject* TCling__GetObjectAddress(const char *Name, void *&LookupCtx);
   const clang::Decl* TCling__GetObjectDecl(TObject *obj);
   void TCling__LibraryLoaded(const void* dyLibHandle,
                              const char* canonicalName);
   void TCling__LibraryUnloaded(const void* dyLibHandle,
                                const char* canonicalName);
   void TCling__RegisterRdictForLoadPCM(const std::string &pcmFileNameFullPath, llvm::StringRef *pcmContent);
}

class TCling final : public TInterpreter {
private: // Static Data Members

   static void* fgSetOfSpecials; // set of TObjects used in CINT variables

private: // Data Members

   Int_t           fMore;             // The brace indent level for the cint command line processor.
   Int_t           fExitCode;         // Value passed to exit() in interpreter.
   char            fPrompt[64];       // Command line prompt string.
   //cling::DictPosition fDictPos;          // dictionary context after initialization is complete.
   //cling::DictPosition fDictPosGlobals;   // dictionary context after ResetGlobals().
   TString         fSharedLibs;       // Shared libraries loaded by G__loadfile().
   Int_t           fGlobalsListSerial;// Last time we refreshed the ROOT list of globals.
   TString         fIncludePath;      // Interpreter include path.
   TString         fRootmapLoadPath;  // Dynamic load path for rootmap files.
   TEnv*           fMapfile;          // Association of classes to libraries.
   std::vector<std::string> fAutoLoadLibStorage; // A storage to return a const char* from GetClassSharedLibsForModule.
   std::map<size_t,std::vector<const char*>> fClassesHeadersMap; // Map of classes hashes and headers associated
   std::map<const cling::Transaction*,size_t> fTransactionHeadersMap; // Map which transaction contains which autoparse.
   std::set<size_t> fLookedUpClasses; // Set of classes for which headers were looked up already
   std::set<size_t> fPayloads; // Set of payloads
   std::set<const char*> fParsedPayloadsAddresses; // Set of payloads which were parsed
   std::hash<std::string> fStringHashFunction; // A simple hashing function
   std::unordered_set<const clang::NamespaceDecl*> fNSFromRootmaps;   // Collection of namespaces fwd declared in the rootmaps
   TObjArray*      fRootmapFiles;     // Loaded rootmap files.
   Bool_t          fLockProcessLine;  // True if ProcessLine should lock gInterpreterMutex.
   Bool_t          fCxxModulesEnabled;// True if C++ modules was enabled

   std::unique_ptr<cling::Interpreter>   fInterpreter;   // The interpreter.
   std::unique_ptr<cling::MetaProcessor> fMetaProcessor; // The metaprocessor.

   std::vector<cling::Value> *fTemporaries;    // Stack of temporaries
   ROOT::TMetaUtils::TNormalizedCtxt  *fNormalizedCtxt; // Which typedef to avoid stripping.
   ROOT::TMetaUtils::TClingLookupHelper *fLookupHelper; // lookup helper used by TClassEdit

   void*           fPrevLoadedDynLibInfo; // Internal info to mark the last loaded libray.
   std::vector<void*> fRegisterModuleDyLibs; // Stack of libraries currently running RegisterModule
   TClingCallbacks* fClingCallbacks; // cling::Interpreter owns it.
   struct CharPtrCmp_t {
      bool operator()(const char* a, const char *b) const {
         return strcmp(a, b) < 0;
      }
   };
   std::set<TClass*> fModTClasses;
   std::vector<std::pair<TClass*,DictFuncPtr_t> > fClassesToUpdate;
   void* fAutoLoadCallBack;
   ULong64_t fTransactionCount; // Cling counter for commited or unloaded transactions which changed the AST.
   std::vector<const char*> fCurExecutingMacros;

   typedef void* SpecialObjectLookupCtx_t;
   typedef std::unordered_map<std::string, TObject*> SpecialObjectMap_t;
   std::map<SpecialObjectLookupCtx_t, SpecialObjectMap_t> fSpecialObjectMaps;

   struct MutexStateAndRecurseCount {
      /// State of gCoreMutex when the first interpreter-related function was invoked.
      std::unique_ptr<ROOT::TVirtualRWMutex::State> fState;

      /// Interpreter-related functions will push the "entry" lock state to *this.
      /// Recursive calls will do that, too - but we must only forget about the lock
      /// state once this recursion count went to 0.
      Int_t fRecurseCount = 0;

      operator bool() const { return (bool)fState; }
   };

   std::vector<MutexStateAndRecurseCount> fInitialMutex{1};

   DeclId_t GetDeclId(const llvm::GlobalValue *gv) const;

   static Int_t DeepAutoLoadImpl(const char *cls);
   static Int_t ShallowAutoLoadImpl(const char *cls);

   Bool_t fHeaderParsingOnDemand;
   Bool_t fIsAutoParsingSuspended;

   UInt_t AutoParseImplRecurse(const char *cls, bool topLevel);
   constexpr static const char* kNullArgv[] = {nullptr};

   bool fIsShuttingDown = false;

protected:
   Bool_t SetSuspendAutoParsing(Bool_t value) override;

public: // Public Interface

   ~TCling() override;
   TCling(const char* name, const char* title, const char* const argv[]);
   TCling(const char* name, const char* title): TCling(name, title, kNullArgv) {}

   void    AddIncludePath(const char* path) override;
   void   *GetAutoLoadCallBack() const override { return fAutoLoadCallBack; }
   void   *SetAutoLoadCallBack(void* cb) override { void* prev = fAutoLoadCallBack; fAutoLoadCallBack = cb; return prev; }
   Int_t   AutoLoad(const char *classname, Bool_t knowDictNotLoaded = kFALSE) override;
   Int_t   AutoLoad(const std::type_info& typeinfo, Bool_t knowDictNotLoaded = kFALSE) override;
   Int_t   AutoParse(const char* cls) override;
   void*   LazyFunctionCreatorAutoload(const std::string& mangled_name);
   bool   LibraryLoadingFailed(const std::string&, const std::string&, bool, bool);
   Bool_t  IsAutoLoadNamespaceCandidate(const clang::NamespaceDecl* nsDecl);
   void    ClearFileBusy() override;
   void    ClearStack() override; // Delete existing temporary values
   Bool_t  Declare(const char* code) override;
   void    EndOfLineAction() override;
   TClass *GetClass(const std::type_info& typeinfo, Bool_t load) const override;
   Int_t   GetExitCode() const override { return fExitCode; }
   TEnv*   GetMapfile() const override { return fMapfile; }
   Int_t   GetMore() const override { return fMore; }
   TClass *GenerateTClass(const char *classname, Bool_t emulation, Bool_t silent = kFALSE) override;
   TClass *GenerateTClass(ClassInfo_t *classinfo, Bool_t silent = kFALSE) override;
   Int_t   GenerateDictionary(const char* classes, const char* includes = "", const char* options = 0) override;
   char*   GetPrompt() override { return fPrompt; }
   const char* GetSharedLibs() override;
   const char* GetClassSharedLibs(const char* cls) override;
   const char* GetSharedLibDeps(const char* lib, bool tryDyld = false) override;
   const char* GetIncludePath() override;
   const char* GetSTLIncludePath() const override;
   TObjArray*  GetRootMapFiles() const override { return fRootmapFiles; }
   unsigned long long GetInterpreterStateMarker() const override { return fTransactionCount;}
   void Initialize() override;
   void ShutDown() override;
   void    InspectMembers(TMemberInspector&, const void* obj, const TClass* cl, Bool_t isTransient) override;
   Bool_t  IsLoaded(const char* filename) const override;
   Bool_t  IsLibraryLoaded(const char* libname) const override;
   Bool_t  HasPCMForLibrary(const char *libname) const override;
   Int_t   Load(const char* filenam, Bool_t system = kFALSE) override;
   void    LoadMacro(const char* filename, EErrorCode* error = 0) override;
   Int_t   LoadLibraryMap(const char* rootmapfile = 0) override;
   Int_t   RescanLibraryMap() override;
   Int_t   ReloadAllSharedLibraryMaps() override;
   Int_t   UnloadAllSharedLibraryMaps() override;
   Int_t   UnloadLibraryMap(const char* library) override;
   Long_t  ProcessLine(const char* line, EErrorCode* error = 0) override;
   Long_t  ProcessLineAsynch(const char* line, EErrorCode* error = 0);
   Long_t  ProcessLineSynch(const char* line, EErrorCode* error = 0) override;
   void    PrintIntro() override;
   bool    RegisterPrebuiltModulePath(const std::string& FullPath,
                                      const std::string& ModuleMapName = "module.modulemap") const override;
   void    RegisterModule(const char* modulename,
                          const char** headers,
                          const char** includePaths,
                          const char* payloadCode,
                          const char* fwdDeclsCode,
                          void (*triggerFunc)(),
                          const FwdDeclArgsToKeepCollection_t& fwdDeclsArgToSkip,
                          const char** classesHeaders,
                          Bool_t lateRegistration = false,
                          Bool_t hasCxxModule = false) override;
   void    RegisterTClassUpdate(TClass *oldcl,DictFuncPtr_t dict) override;
   void    UnRegisterTClassUpdate(const TClass *oldcl) override;

   Int_t   SetClassSharedLibs(const char *cls, const char *libs) override;
   void    SetGetline(const char * (*getlineFunc)(const char* prompt),
                      void (*histaddFunc)(const char* line)) override;
   void    Reset() override;
   void    ResetAll() override;
   void    ResetGlobals() override;
   void    ResetGlobalVar(void* obj) override;
   void    RewindDictionary() override;
   Int_t   DeleteGlobal(void* obj) override;
   Int_t   DeleteVariable(const char *name) override;
   void    SaveContext() override;
   void    SaveGlobalsContext() override;
   void    UpdateListOfGlobals() override;
   void    UpdateListOfGlobalFunctions() override;
   void    UpdateListOfTypes() override;
   void    SetClassInfo(TClass* cl, Bool_t reload = kFALSE) override;

   ECheckClassInfo CheckClassInfo(const char *name, Bool_t autoload, Bool_t isClassOrNamespaceOnly = kFALSE) override;

   Bool_t  CheckClassTemplate(const char *name) override;
   Long_t  Calc(const char* line, EErrorCode* error = 0) override;
   void    CreateListOfBaseClasses(TClass* cl) const override;
   void    CreateListOfDataMembers(TClass* cl) const override;
   void    CreateListOfMethods(TClass* cl) const override;
   void    CreateListOfMethodArgs(TFunction* m) const override;
   void    UpdateListOfMethods(TClass* cl) const override;
   void    UpdateListOfDataMembers(TClass* cl) const;

   DeclId_t GetDataMember(ClassInfo_t *cl, const char *name) const override;
   DeclId_t GetDataMemberAtAddr(const void *addr) const override;
   DeclId_t GetDataMemberWithValue(const void *ptrvalue) const override;
   DeclId_t GetEnum(TClass *cl, const char *name) const override;
   TEnum*   CreateEnum(void *VD, TClass *cl) const override;
   void     UpdateEnumConstants(TEnum* enumObj, TClass* cl) const override;
   void     LoadEnums(TListOfEnums& cl) const override;
   std::string ToString(const char* type, void *obj) override;
   TString GetMangledName(TClass* cl, const char* method, const char* params, Bool_t objectIsConst = kFALSE) override;
   TString GetMangledNameWithPrototype(TClass* cl, const char* method, const char* proto, Bool_t objectIsConst = kFALSE, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch) override;
   void*   GetInterfaceMethod(TClass* cl, const char* method, const char* params, Bool_t objectIsConst = kFALSE) override;
   void*   GetInterfaceMethodWithPrototype(TClass* cl, const char* method, const char* proto, Bool_t objectIsConst = kFALSE, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch) override;
   DeclId_t GetFunction(ClassInfo_t *cl, const char *funcname) override;
   DeclId_t GetFunctionWithPrototype(ClassInfo_t *cl, const char* method, const char* proto, Bool_t objectIsConst = kFALSE, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch) override;
   DeclId_t GetFunctionWithValues(ClassInfo_t *cl, const char* method, const char* params, Bool_t objectIsConst = kFALSE) override;
   DeclId_t GetFunctionTemplate(ClassInfo_t *cl, const char *funcname) override;
   void     GetFunctionOverloads(ClassInfo_t *cl, const char *funcname, std::vector<DeclId_t>& res) const override;
   void     LoadFunctionTemplates(TClass* cl) const override;

   std::vector<std::string> GetUsingNamespaces(ClassInfo_t *cl) const override;

   void    GetInterpreterTypeName(const char* name, std::string &output, Bool_t full = kFALSE) override;
   void    Execute(const char* function, const char* params, int* error = 0) override;
   void    Execute(TObject* obj, TClass* cl, const char* method, const char* params, int* error = 0) override;
   void    Execute(TObject* obj, TClass* cl, const char* method, const char* params, Bool_t objectIsConst, int* error = 0);
   void    Execute(TObject* obj, TClass* cl, TMethod* method, TObjArray* params, int* error = 0) override;
   void    ExecuteWithArgsAndReturn(TMethod* method, void* address, const void* args[] = 0, int nargs = 0, void* ret= 0) const override;
   Long_t  ExecuteMacro(const char* filename, EErrorCode* error = 0) override;
   void    RecursiveRemove(TObject* obj) override;
   Bool_t  IsErrorMessagesEnabled() const override;
   Bool_t  SetErrorMessages(Bool_t enable = kTRUE) override;
   Bool_t  IsProcessLineLocked() const override {
      return fLockProcessLine;
   }
   void    SetProcessLineLock(Bool_t lock = kTRUE) override {
      fLockProcessLine = lock;
   }
   const char* TypeName(const char* typeDesc) override;

   void     SnapshotMutexState(ROOT::TVirtualRWMutex* mtx) override;
   void     ForgetMutexState() override;

   void     ApplyToInterpreterMutex(void* delta);
   void    *RewindInterpreterMutex();

   static void  UpdateClassInfo(char* name, Long_t tagnum);
   static void  UpdateClassInfoWork(const char* name);
          void  RefreshClassInfo(TClass *cl, const clang::NamedDecl *def, bool alias);
          void  UpdateClassInfoWithDecl(const clang::NamedDecl* ND);
   static void  UpdateAllCanvases();

   // Misc
   int    DisplayClass(FILE* fout, const char* name, int base, int start) const override;
   int    DisplayIncludePath(FILE* fout) const override;
   void*  FindSym(const char* entry) const override;
   void   GenericError(const char* error) const override;
   Long_t GetExecByteCode() const override;
   const char* GetTopLevelMacroName() const override;
   const char* GetCurrentMacroName() const override;
   int    GetSecurityError() const override;
   int    LoadFile(const char* path) const override;
   Bool_t LoadText(const char* text) const override;
   const char* MapCppName(const char*) const override;
   void   SetAlloclockfunc(void (*)()) const override;
   void   SetAllocunlockfunc(void (*)()) const override;
   int    SetClassAutoLoading(int) const override;
   int    SetClassAutoparsing(int) override ;
           Bool_t IsAutoParsingSuspended() const override { return fIsAutoParsingSuspended; }
   void   SetErrmsgcallback(void* p) const override;
   void   SetTempLevel(int val) const override;
   int    UnloadFile(const char* path) const override;

   void               CodeComplete(const std::string&, size_t&,
                                   std::vector<std::string>&) override;
   int Evaluate(const char*, TInterpreterValue&) override;
   std::unique_ptr<TInterpreterValue> MakeInterpreterValue() const override;
   void               RegisterTemporary(const TInterpreterValue& value);
   void               RegisterTemporary(const cling::Value& value);
   const ROOT::TMetaUtils::TNormalizedCtxt& GetNormalizedContext() const {return *fNormalizedCtxt;};
   TObject* GetObjectAddress(const char *Name, void *&LookupCtx);


   // core/meta helper functions.
   EReturnType MethodCallReturnType(TFunction *func) const override;
   virtual void GetFunctionName(const clang::FunctionDecl *decl, std::string &name) const;
   bool DiagnoseIfInterpreterException(const std::exception &e) const override;

   // CallFunc interface
   DeclId_t GetDeclId(CallFunc_t *info) const override;
   void   CallFunc_Delete(CallFunc_t* func) const override;
   void   CallFunc_Exec(CallFunc_t* func, void* address) const override;
   void   CallFunc_Exec(CallFunc_t* func, void* address, TInterpreterValue& val) const override;
   void   CallFunc_ExecWithReturn(CallFunc_t* func, void* address, void* ret) const override;
   void   CallFunc_ExecWithArgsAndReturn(CallFunc_t* func, void* address, const void* args[] = 0, int nargs = 0, void* ret = 0) const override;
   Long_t    CallFunc_ExecInt(CallFunc_t* func, void* address) const override;
   Long64_t  CallFunc_ExecInt64(CallFunc_t* func, void* address) const override;
   Double_t  CallFunc_ExecDouble(CallFunc_t* func, void* address) const override;
   CallFunc_t*   CallFunc_Factory() const override;
   CallFunc_t*   CallFunc_FactoryCopy(CallFunc_t* func) const override;
   MethodInfo_t* CallFunc_FactoryMethod(CallFunc_t* func) const override;
   void   CallFunc_IgnoreExtraArgs(CallFunc_t* func, bool ignore) const override;
   void   CallFunc_Init(CallFunc_t* func) const override;
   bool   CallFunc_IsValid(CallFunc_t* func) const override;
   CallFuncIFacePtr_t CallFunc_IFacePtr(CallFunc_t * func) const override;
   void   CallFunc_ResetArg(CallFunc_t* func) const override;
   void   CallFunc_SetArg(CallFunc_t* func, Long_t param) const override;
   void   CallFunc_SetArg(CallFunc_t* func, ULong_t param) const override;
   void   CallFunc_SetArg(CallFunc_t* func, Float_t param) const override;
   void   CallFunc_SetArg(CallFunc_t* func, Double_t param) const override;
   void   CallFunc_SetArg(CallFunc_t* func, Long64_t param) const override;
   void   CallFunc_SetArg(CallFunc_t* func, ULong64_t param) const override;
   void   CallFunc_SetArgArray(CallFunc_t* func, Long_t* paramArr, Int_t nparam) const override;
   void   CallFunc_SetArgs(CallFunc_t* func, const char* param) const override;
   void   CallFunc_SetFunc(CallFunc_t* func, ClassInfo_t* info, const char* method, const char* params, Long_t* Offset) const override;
   void   CallFunc_SetFunc(CallFunc_t* func, ClassInfo_t* info, const char* method, const char* params, bool objectIsConst, Long_t* Offset) const override;
   void   CallFunc_SetFunc(CallFunc_t* func, MethodInfo_t* info) const override;
   void   CallFunc_SetFuncProto(CallFunc_t* func, ClassInfo_t* info, const char* method, const char* proto, Long_t* Offset, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch) const override;
   void   CallFunc_SetFuncProto(CallFunc_t* func, ClassInfo_t* info, const char* method, const char* proto, bool objectIsConst, Long_t* Offset, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch) const override;
   void   CallFunc_SetFuncProto(CallFunc_t* func, ClassInfo_t* info, const char* method, const std::vector<TypeInfo_t*> &proto, Long_t* Offset, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch) const override;
   void   CallFunc_SetFuncProto(CallFunc_t* func, ClassInfo_t* info, const char* method, const std::vector<TypeInfo_t*> &proto, bool objectIsConst, Long_t* Offset, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch) const override;

   std::string CallFunc_GetWrapperCode(CallFunc_t *func) const override;

   // ClassInfo interface
   DeclId_t GetDeclId(ClassInfo_t *info) const override;
   Bool_t ClassInfo_Contains(ClassInfo_t *info, DeclId_t declid) const override;
   Long_t ClassInfo_ClassProperty(ClassInfo_t* info) const override;
   void   ClassInfo_Delete(ClassInfo_t* info) const override;
   void   ClassInfo_Delete(ClassInfo_t* info, void* arena) const override;
   void   ClassInfo_DeleteArray(ClassInfo_t* info, void* arena, bool dtorOnly) const override;
   void   ClassInfo_Destruct(ClassInfo_t* info, void* arena) const override;
   ClassInfo_t*  ClassInfo_Factory(Bool_t all = kTRUE) const override;
   ClassInfo_t*  ClassInfo_Factory(ClassInfo_t* cl) const override;
   ClassInfo_t*  ClassInfo_Factory(const char* name) const override;
   ClassInfo_t*  ClassInfo_Factory(DeclId_t declid) const override;
   Long_t   ClassInfo_GetBaseOffset(ClassInfo_t* fromDerived, ClassInfo_t* toBase, void * address, bool isDerivedObject) const override;
   int    ClassInfo_GetMethodNArg(ClassInfo_t* info, const char* method, const char* proto, Bool_t objectIsConst = false, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch) const override;
   bool   ClassInfo_HasDefaultConstructor(ClassInfo_t* info, Bool_t testio = kFALSE) const override;
   bool   ClassInfo_HasMethod(ClassInfo_t* info, const char* name) const override;
   void   ClassInfo_Init(ClassInfo_t* info, const char* funcname) const override;
   void   ClassInfo_Init(ClassInfo_t* info, int tagnum) const override;
   bool   ClassInfo_IsBase(ClassInfo_t* info, const char* name) const override;
   bool   ClassInfo_IsEnum(const char* name) const override;
   bool   ClassInfo_IsScopedEnum(ClassInfo_t* info) const override;
   EDataType ClassInfo_GetUnderlyingType(ClassInfo_t* info) const override;
   bool   ClassInfo_IsLoaded(ClassInfo_t* info) const override;
   bool   ClassInfo_IsValid(ClassInfo_t* info) const override;
   bool   ClassInfo_IsValidMethod(ClassInfo_t* info, const char* method, const char* proto, Long_t* offset, ROOT::EFunctionMatchMode /* mode */ = ROOT::kConversionMatch) const override;
   bool   ClassInfo_IsValidMethod(ClassInfo_t* info, const char* method, const char* proto, Bool_t objectIsConst, Long_t* offset, ROOT::EFunctionMatchMode /* mode */ = ROOT::kConversionMatch) const override;
   int    ClassInfo_Next(ClassInfo_t* info) const override;
   void*  ClassInfo_New(ClassInfo_t* info) const override;
   void*  ClassInfo_New(ClassInfo_t* info, int n) const override;
   void*  ClassInfo_New(ClassInfo_t* info, int n, void* arena) const override;
   void*  ClassInfo_New(ClassInfo_t* info, void* arena) const override;
   Long_t ClassInfo_Property(ClassInfo_t* info) const override;
   int    ClassInfo_Size(ClassInfo_t* info) const override;
   Long_t ClassInfo_Tagnum(ClassInfo_t* info) const override;
   const char* ClassInfo_FileName(ClassInfo_t* info) const override;
   const char* ClassInfo_FullName(ClassInfo_t* info) const override;
   const char* ClassInfo_Name(ClassInfo_t* info) const override;
   const char* ClassInfo_Title(ClassInfo_t* info) const override;
   const char* ClassInfo_TmpltName(ClassInfo_t* info) const override;

   // BaseClassInfo interface
   void   BaseClassInfo_Delete(BaseClassInfo_t* bcinfo) const override;
   BaseClassInfo_t*  BaseClassInfo_Factory(ClassInfo_t* info) const override;
   BaseClassInfo_t*  BaseClassInfo_Factory(ClassInfo_t* derived,
                                                   ClassInfo_t* base) const override;
   int    BaseClassInfo_Next(BaseClassInfo_t* bcinfo) const override;
   int    BaseClassInfo_Next(BaseClassInfo_t* bcinfo, int onlyDirect) const override;
   Long_t BaseClassInfo_Offset(BaseClassInfo_t* toBaseClassInfo, void * address, bool isDerivedObject) const override;
   Long_t BaseClassInfo_Property(BaseClassInfo_t* bcinfo) const override;
   Long_t BaseClassInfo_Tagnum(BaseClassInfo_t* bcinfo) const override;
   ClassInfo_t*BaseClassInfo_ClassInfo(BaseClassInfo_t * /* bcinfo */) const override;
   const char* BaseClassInfo_FullName(BaseClassInfo_t* bcinfo) const override;
   const char* BaseClassInfo_Name(BaseClassInfo_t* bcinfo) const override;
   const char* BaseClassInfo_TmpltName(BaseClassInfo_t* bcinfo) const override;

   // DataMemberInfo interface
   DeclId_t GetDeclId(DataMemberInfo_t *info) const override;
   int    DataMemberInfo_ArrayDim(DataMemberInfo_t* dminfo) const override;
   void   DataMemberInfo_Delete(DataMemberInfo_t* dminfo) const override;
   DataMemberInfo_t*  DataMemberInfo_Factory(ClassInfo_t* clinfo = 0) const override;
   DataMemberInfo_t  *DataMemberInfo_Factory(DeclId_t declid, ClassInfo_t* clinfo) const override;
   DataMemberInfo_t*  DataMemberInfo_FactoryCopy(DataMemberInfo_t* dminfo) const override;
   bool   DataMemberInfo_IsValid(DataMemberInfo_t* dminfo) const override;
   int    DataMemberInfo_MaxIndex(DataMemberInfo_t* dminfo, Int_t dim) const override;
   int    DataMemberInfo_Next(DataMemberInfo_t* dminfo) const override;
   Long_t DataMemberInfo_Offset(DataMemberInfo_t* dminfo) const override;
   Long_t DataMemberInfo_Property(DataMemberInfo_t* dminfo) const override;
   Long_t DataMemberInfo_TypeProperty(DataMemberInfo_t* dminfo) const override;
   int    DataMemberInfo_TypeSize(DataMemberInfo_t* dminfo) const override;
   const char* DataMemberInfo_TypeName(DataMemberInfo_t* dminfo) const override;
   const char* DataMemberInfo_TypeTrueName(DataMemberInfo_t* dminfo) const override;
   const char* DataMemberInfo_Name(DataMemberInfo_t* dminfo) const override;
   const char* DataMemberInfo_Title(DataMemberInfo_t* dminfo) const override;
   const char* DataMemberInfo_ValidArrayIndex(DataMemberInfo_t* dminfo) const override;
   void SetDeclAttr(DeclId_t, const char* /* attribute */) override;


   // Function Template interface
   DeclId_t GetDeclId(FuncTempInfo_t *info) const override;
   void   FuncTempInfo_Delete(FuncTempInfo_t * /* ft_info */) const override;
   FuncTempInfo_t  *FuncTempInfo_Factory(DeclId_t declid) const override;
   FuncTempInfo_t  *FuncTempInfo_FactoryCopy(FuncTempInfo_t * /* ft_info */) const override;
   Bool_t FuncTempInfo_IsValid(FuncTempInfo_t * /* ft_info */) const override;
   UInt_t FuncTempInfo_TemplateNargs(FuncTempInfo_t * /* ft_info */) const override;
   UInt_t FuncTempInfo_TemplateMinReqArgs(FuncTempInfo_t * /* ft_info */) const override;
   Long_t FuncTempInfo_Property(FuncTempInfo_t * /* ft_info */) const override;
   Long_t FuncTempInfo_ExtraProperty(FuncTempInfo_t * /* ft_info */) const override;
   void FuncTempInfo_Name(FuncTempInfo_t * /* ft_info */, TString& name) const override;
   void FuncTempInfo_Title(FuncTempInfo_t * /* ft_info */, TString& name) const override;

   // MethodInfo interface
   DeclId_t GetDeclId(MethodInfo_t *info) const override;
   void   MethodInfo_CreateSignature(MethodInfo_t* minfo, TString& signature) const override;
   void   MethodInfo_Delete(MethodInfo_t* minfo) const override;
   MethodInfo_t*  MethodInfo_Factory() const override;
   MethodInfo_t*  MethodInfo_Factory(ClassInfo_t *clinfo) const override;
   MethodInfo_t  *MethodInfo_Factory(DeclId_t declid) const override;
   MethodInfo_t*  MethodInfo_FactoryCopy(MethodInfo_t* minfo) const override;
   void*  MethodInfo_InterfaceMethod(MethodInfo_t* minfo) const override;
   bool   MethodInfo_IsValid(MethodInfo_t* minfo) const override;
   int    MethodInfo_NArg(MethodInfo_t* minfo) const override;
   int    MethodInfo_NDefaultArg(MethodInfo_t* minfo) const override;
   int    MethodInfo_Next(MethodInfo_t* minfo) const override;
   Long_t MethodInfo_Property(MethodInfo_t* minfo) const override;
   Long_t MethodInfo_ExtraProperty(MethodInfo_t* minfo) const override;
   TypeInfo_t*  MethodInfo_Type(MethodInfo_t* minfo) const override;
   EReturnType MethodInfo_MethodCallReturnType(MethodInfo_t* minfo) const override;
   const char* MethodInfo_GetMangledName(MethodInfo_t* minfo) const override;
   const char* MethodInfo_GetPrototype(MethodInfo_t* minfo) const override;
   const char* MethodInfo_Name(MethodInfo_t* minfo) const override;
   const char* MethodInfo_TypeName(MethodInfo_t* minfo) const override;
   std::string MethodInfo_TypeNormalizedName(MethodInfo_t* minfo) const override;
   const char* MethodInfo_Title(MethodInfo_t* minfo) const override;

   // MethodArgInfo interface
   void   MethodArgInfo_Delete(MethodArgInfo_t* marginfo) const override;
   MethodArgInfo_t*  MethodArgInfo_Factory() const override;
   MethodArgInfo_t*  MethodArgInfo_Factory(MethodInfo_t *minfo) const override;
   MethodArgInfo_t*  MethodArgInfo_FactoryCopy(MethodArgInfo_t* marginfo) const override;
   bool   MethodArgInfo_IsValid(MethodArgInfo_t* marginfo) const override;
   int    MethodArgInfo_Next(MethodArgInfo_t* marginfo) const override;
   Long_t MethodArgInfo_Property(MethodArgInfo_t* marginfo) const override;
   const char* MethodArgInfo_DefaultValue(MethodArgInfo_t* marginfo) const override;
   const char* MethodArgInfo_Name(MethodArgInfo_t* marginfo) const override;
   const char* MethodArgInfo_TypeName(MethodArgInfo_t* marginfo) const override;
   std::string MethodArgInfo_TypeNormalizedName(MethodArgInfo_t *marginfo) const override;

   // TypeInfo interface
   void   TypeInfo_Delete(TypeInfo_t* tinfo) const override;
   TypeInfo_t* TypeInfo_Factory() const override;
   TypeInfo_t *TypeInfo_Factory(const char* name) const override;
   TypeInfo_t* TypeInfo_FactoryCopy(TypeInfo_t* /* tinfo */) const override;
   void   TypeInfo_Init(TypeInfo_t* tinfo, const char* funcname) const override;
   bool   TypeInfo_IsValid(TypeInfo_t* tinfo) const override;
   const char* TypeInfo_Name(TypeInfo_t* /* info */) const override;
   Long_t TypeInfo_Property(TypeInfo_t* tinfo) const override;
   int    TypeInfo_RefType(TypeInfo_t* /* tinfo */) const override;
   int    TypeInfo_Size(TypeInfo_t* tinfo) const override;
   const char* TypeInfo_TrueName(TypeInfo_t* tinfo) const override;

   // TypedefInfo interface
   DeclId_t GetDeclId(TypedefInfo_t *info) const override;
   void   TypedefInfo_Delete(TypedefInfo_t* tinfo) const override;
   TypedefInfo_t*  TypedefInfo_Factory() const override;
   TypedefInfo_t*  TypedefInfo_Factory(const char* name) const override;
   TypedefInfo_t*  TypedefInfo_FactoryCopy(TypedefInfo_t* tinfo) const override;
   void   TypedefInfo_Init(TypedefInfo_t* tinfo, const char* name) const override;
   bool   TypedefInfo_IsValid(TypedefInfo_t* tinfo) const override;
   int    TypedefInfo_Next(TypedefInfo_t* tinfo) const override;
   Long_t TypedefInfo_Property(TypedefInfo_t* tinfo) const override;
   int    TypedefInfo_Size(TypedefInfo_t* tinfo) const override;
   const char* TypedefInfo_TrueName(TypedefInfo_t* tinfo) const override;
   const char* TypedefInfo_Name(TypedefInfo_t* tinfo) const override;
   const char* TypedefInfo_Title(TypedefInfo_t* tinfo) const override;

   std::set<TClass*>& GetModTClasses() { return fModTClasses; }

   void HandleNewDecl(const void* DV, bool isDeserialized, std::set<TClass*>& modifiedClasses);
   void UpdateListsOnCommitted(const cling::Transaction &T);
   void UpdateListsOnUnloaded(const cling::Transaction &T);
   void InvalidateGlobal(const clang::Decl *D);
   void TransactionRollback(const cling::Transaction &T);
   void LibraryLoaded(const void* dyLibHandle, const char* canonicalName);
   void LibraryUnloaded(const void* dyLibHandle, const char* canonicalName);

private: // Private Utility Functions and Classes
   template <typename List, typename Object>
   static void RemoveAndInvalidateObject(List &L, Object *O) {
      // Invalidate stored information by setting the `xxxInfo_t' to nullptr.
      if (O && O->IsValid())
         L.Unload(O), O->Update(nullptr);
   }

   void InvalidateCachedDecl(const std::tuple<TListOfDataMembers*,
                                        TListOfFunctions*,
                                        TListOfFunctionTemplates*,
                                        TListOfEnums*> &Lists, const clang::Decl *D);

   class SuspendAutoLoadingRAII {
      TCling *fTCling = nullptr;
      bool fOldValue;

   public:
      SuspendAutoLoadingRAII(TCling *tcling) : fTCling(tcling) { fOldValue = fTCling->SetClassAutoLoading(false); }
      ~SuspendAutoLoadingRAII() { fTCling->SetClassAutoLoading(fOldValue); }
   };

   class TUniqueString {
   public:
      TUniqueString() = delete;
      TUniqueString(const TUniqueString &) = delete;
      TUniqueString(Long64_t size);
      const char *Data();
      bool Append(const std::string &str);
   private:
      std::string fContent;
      std::set<size_t> fLinesHashSet;
      std::hash<std::string> fHashFunc;
   };

   TCling();
   TCling(const TCling&); // NOT IMPLEMENTED
   TCling& operator=(const TCling&); // NOT IMPLEMENTED

   void Execute(TMethod*, TObjArray*, int* /*error*/ = 0) override
   {
   }

   void UpdateListOfLoadedSharedLibraries();
   void RegisterLoadedSharedLibrary(const char* name);
   void AddFriendToClass(clang::FunctionDecl*, clang::CXXRecordDecl*) const;

   std::map<std::string, llvm::StringRef> fPendingRdicts;
   void RegisterRdictForLoadPCM(const std::string &pcmFileNameFullPath, llvm::StringRef *pcmContent);
   void LoadPCM(std::string pcmFileNameFullPath);
   void LoadPCMImpl(TFile &pcmFile);

   void InitRootmapFile(const char *name);
   int  ReadRootmapFile(const char *rootmapfile, TUniqueString* uniqueString = nullptr);
   Bool_t HandleNewTransaction(const cling::Transaction &T);
   bool IsClassAutoLoadingEnabled() const;
   void ProcessClassesToUpdate();
   cling::Interpreter *GetInterpreterImpl() const { return fInterpreter.get(); }
   cling::MetaProcessor *GetMetaProcessorImpl() const { return fMetaProcessor.get(); }

   friend void TCling__RegisterRdictForLoadPCM(const std::string &pcmFileNameFullPath, llvm::StringRef *pcmContent);
   friend cling::Interpreter* TCling__GetInterpreter();
};

#endif
