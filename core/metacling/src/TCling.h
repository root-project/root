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
   class IdentifierIterator;
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

   struct MutexStateAndRecurseCountDelta {
      using StateDelta = ROOT::TVirtualRWMutex::StateDelta;

      MutexStateAndRecurseCount   fInitialState;
      std::unique_ptr<StateDelta> fDelta;
   };

   MutexStateAndRecurseCount fInitialMutex;

   DeclId_t GetDeclId(const llvm::GlobalValue *gv) const;

   static Int_t DeepAutoLoadImpl(const char *cls, std::unordered_set<std::string> &visited, bool nameIsNormalized);
   static Int_t ShallowAutoLoadImpl(const char *cls);

   Bool_t fHeaderParsingOnDemand;
   Bool_t fIsAutoParsingSuspended;

   UInt_t AutoParseImplRecurse(const char *cls, bool topLevel);
   constexpr static const char* kNullArgv[] = {nullptr};

   bool fIsShuttingDown = false;

protected:
   Bool_t SetSuspendAutoParsing(Bool_t value) final;

public: // Public Interface

   virtual ~TCling();
   TCling(const char* name, const char* title, const char* const argv[]);
   TCling(const char* name, const char* title): TCling(name, title, kNullArgv) {}

   void    AddIncludePath(const char* path) final;
   void   *GetAutoLoadCallBack() const final { return fAutoLoadCallBack; }
   void   *SetAutoLoadCallBack(void* cb) final { void* prev = fAutoLoadCallBack; fAutoLoadCallBack = cb; return prev; }
   Int_t   AutoLoad(const char *classname, Bool_t knowDictNotLoaded = kFALSE) final;
   Int_t   AutoLoad(const std::type_info& typeinfo, Bool_t knowDictNotLoaded = kFALSE) final;
   Int_t   AutoParse(const char* cls) final;
   void*   LazyFunctionCreatorAutoload(const std::string& mangled_name);
   bool   LibraryLoadingFailed(const std::string&, const std::string&, bool, bool);
   Bool_t  IsAutoLoadNamespaceCandidate(const clang::NamespaceDecl* nsDecl);
   void    ClearFileBusy() final;
   void    ClearStack() final; // Delete existing temporary values
   Bool_t  Declare(const char* code) final;
   void    EndOfLineAction() final;
   TClass *GetClass(const std::type_info& typeinfo, Bool_t load) const final;
   Int_t   GetExitCode() const final { return fExitCode; }
   TEnv*   GetMapfile() const final { return fMapfile; }
   Int_t   GetMore() const final;
   TClass *GenerateTClass(const char *classname, Bool_t emulation, Bool_t silent = kFALSE) final;
   TClass *GenerateTClass(ClassInfo_t *classinfo, Bool_t silent = kFALSE) final;
   Int_t   GenerateDictionary(const char* classes, const char* includes = "", const char* options = nullptr) final;
   char*   GetPrompt() final { return fPrompt; }
   const char* GetSharedLibs() final;
   const char* GetClassSharedLibs(const char* cls) final;
   const char* GetSharedLibDeps(const char* lib, bool tryDyld = false) final;
   const char* GetIncludePath() final;
   virtual const char* GetSTLIncludePath() const final;
   TObjArray*  GetRootMapFiles() const final { return fRootmapFiles; }
   unsigned long long GetInterpreterStateMarker() const final { return fTransactionCount;}
   virtual void Initialize() final;
   virtual void ShutDown() final;
   void    InspectMembers(TMemberInspector&, const void* obj, const TClass* cl, Bool_t isTransient) final;
   Bool_t  IsLoaded(const char* filename) const final;
   Bool_t  IsLibraryLoaded(const char* libname) const final;
   Bool_t  HasPCMForLibrary(const char *libname) const final;
   Int_t   Load(const char* filenam, Bool_t system = kFALSE) final;
   void    LoadMacro(const char* filename, EErrorCode* error = nullptr) final;
   Int_t   LoadLibraryMap(const char* rootmapfile = nullptr) final;
   Int_t   RescanLibraryMap() final;
   Int_t   ReloadAllSharedLibraryMaps() final;
   Int_t   UnloadAllSharedLibraryMaps() final;
   Int_t   UnloadLibraryMap(const char* library) final;
   Longptr_t ProcessLine(const char* line, EErrorCode* error = nullptr) final;
   Longptr_t ProcessLineAsynch(const char* line, EErrorCode* error = nullptr);
   Longptr_t ProcessLineSynch(const char* line, EErrorCode* error = nullptr) final;
   void    PrintIntro() final;
   bool    RegisterPrebuiltModulePath(const std::string& FullPath,
                                      const std::string& ModuleMapName = "module.modulemap") const final;
   void    RegisterModule(const char* modulename,
                          const char** headers,
                          const char** includePaths,
                          const char* payloadCode,
                          const char* fwdDeclsCode,
                          void (*triggerFunc)(),
                          const FwdDeclArgsToKeepCollection_t& fwdDeclsArgToSkip,
                          const char** classesHeaders,
                          Bool_t lateRegistration = false,
                          Bool_t hasCxxModule = false) final;
   virtual void AddAvailableIndentifiers(TSeqCollection& Idents) final;
   void    RegisterTClassUpdate(TClass *oldcl,DictFuncPtr_t dict) final;
   void    UnRegisterTClassUpdate(const TClass *oldcl) final;

   Int_t   SetClassSharedLibs(const char *cls, const char *libs) final;
   void    SetGetline(const char * (*getlineFunc)(const char* prompt),
                      void (*histaddFunc)(const char* line)) final;
   void    Reset() final;
   void    ResetAll() final;
   void    ResetGlobals() final;
   void    ResetGlobalVar(void* obj) final;
   void    RewindDictionary() final;
   Int_t   DeleteGlobal(void* obj) final;
   Int_t   DeleteVariable(const char *name) final;
   void    SaveContext() final;
   void    SaveGlobalsContext() final;
   void    UpdateListOfGlobals() final;
   void    UpdateListOfGlobalFunctions() final;
   void    UpdateListOfTypes() final;
   void    SetClassInfo(TClass* cl, Bool_t reload = kFALSE) final;

   ECheckClassInfo CheckClassInfo(const char *name, Bool_t autoload, Bool_t isClassOrNamespaceOnly = kFALSE) final;

   Bool_t  CheckClassTemplate(const char *name) final;
   Longptr_t Calc(const char* line, EErrorCode* error = nullptr) final;
   void    CreateListOfBaseClasses(TClass* cl) const final;
   void    CreateListOfDataMembers(TClass* cl) const final;
   void    CreateListOfMethods(TClass* cl) const final;
   void    CreateListOfMethodArgs(TFunction* m) const final;
   void    UpdateListOfMethods(TClass* cl) const final;
   void    UpdateListOfDataMembers(TClass* cl) const;

   DeclId_t GetDataMember(ClassInfo_t *cl, const char *name) const final;
   DeclId_t GetDataMemberAtAddr(const void *addr) const final;
   DeclId_t GetDataMemberWithValue(const void *ptrvalue) const final;
   DeclId_t GetEnum(TClass *cl, const char *name) const final;
   TEnum*   CreateEnum(void *VD, TClass *cl) const final;
   void     UpdateEnumConstants(TEnum* enumObj, TClass* cl) const final;
   void     LoadEnums(TListOfEnums& cl) const final;
   std::string ToString(const char* type, void *obj) final;
   TString GetMangledName(TClass* cl, const char* method, const char* params, Bool_t objectIsConst = kFALSE) final;
   TString GetMangledNameWithPrototype(TClass* cl, const char* method, const char* proto, Bool_t objectIsConst = kFALSE, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch) final;
   void*   GetInterfaceMethod(TClass* cl, const char* method, const char* params, Bool_t objectIsConst = kFALSE) final;
   void*   GetInterfaceMethodWithPrototype(TClass* cl, const char* method, const char* proto, Bool_t objectIsConst = kFALSE, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch) final;
   DeclId_t GetFunction(ClassInfo_t *cl, const char *funcname) final;
   DeclId_t GetFunctionWithPrototype(ClassInfo_t *cl, const char* method, const char* proto, Bool_t objectIsConst = kFALSE, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch) final;
   DeclId_t GetFunctionWithValues(ClassInfo_t *cl, const char* method, const char* params, Bool_t objectIsConst = kFALSE) final;
   DeclId_t GetFunctionTemplate(ClassInfo_t *cl, const char *funcname) final;
   void     GetFunctionOverloads(ClassInfo_t *cl, const char *funcname, std::vector<DeclId_t>& res) const final;
   virtual void     LoadFunctionTemplates(TClass* cl) const final;

   std::vector<std::string> GetUsingNamespaces(ClassInfo_t *cl) const final;

   void    GetInterpreterTypeName(const char* name, std::string &output, Bool_t full = kFALSE) final;
   void    Execute(const char* function, const char* params, int* error = nullptr) final;
   void    Execute(TObject* obj, TClass* cl, const char* method, const char* params, int* error = nullptr) final;
   void    Execute(TObject* obj, TClass* cl, const char* method, const char* params, Bool_t objectIsConst, int* error = nullptr);
   void    Execute(TObject* obj, TClass* cl, TMethod* method, TObjArray* params, int* error = nullptr) final;
   void    ExecuteWithArgsAndReturn(TMethod* method, void* address, const void* args[] = nullptr, int nargs = 0, void* ret= nullptr) const final;
   Longptr_t ExecuteMacro(const char* filename, EErrorCode* error = nullptr) final;
   void    RecursiveRemove(TObject* obj) final;
   Bool_t  IsErrorMessagesEnabled() const final;
   Bool_t  SetErrorMessages(Bool_t enable = kTRUE) final;
   Bool_t  IsProcessLineLocked() const final {
      return fLockProcessLine;
   }
   void    SetProcessLineLock(Bool_t lock = kTRUE) final {
      fLockProcessLine = lock;
   }
   const char* TypeName(const char* typeDesc) final;

   void     SnapshotMutexState(ROOT::TVirtualRWMutex* mtx) final;
   void     ForgetMutexState() final;

   void     ApplyToInterpreterMutex(void* delta);
   void    *RewindInterpreterMutex();

   static void  UpdateClassInfo(char* name, Long_t tagnum);
   static void  UpdateClassInfoWork(const char* name);
          void  RefreshClassInfo(TClass *cl, const clang::NamedDecl *def, bool alias);
          void  UpdateClassInfoWithDecl(const clang::NamedDecl* ND);
   static void  UpdateAllCanvases();

   // Misc
   int    DisplayClass(FILE* fout, const char* name, int base, int start) const final;
   int    DisplayIncludePath(FILE* fout) const final;
   void*  FindSym(const char* entry) const final;
   void   GenericError(const char* error) const final;
   Long_t GetExecByteCode() const final;
   const char* GetTopLevelMacroName() const final;
   const char* GetCurrentMacroName() const final;
   int    GetSecurityError() const final;
   int    LoadFile(const char* path) const final;
   Bool_t LoadText(const char* text) const final;
   const char* MapCppName(const char*) const final;
   void   SetAlloclockfunc(void (*)()) const final;
   void   SetAllocunlockfunc(void (*)()) const final;
   int    SetClassAutoLoading(int) const final;
   int    SetClassAutoparsing(int) final;
   Bool_t IsAutoParsingSuspended() const final { return fIsAutoParsingSuspended; }
   void   SetErrmsgcallback(void* p) const final;
   void   ReportDiagnosticsToErrorHandler(bool enable = true) final;
   void   SetTempLevel(int val) const final;
   int    UnloadFile(const char* path) const final;

   void   CodeComplete(const std::string&, size_t&,
                       std::vector<std::string>&) final;
   int Evaluate(const char*, TInterpreterValue&) final;
   std::unique_ptr<TInterpreterValue> MakeInterpreterValue() const final;
   void               RegisterTemporary(const TInterpreterValue& value);
   void               RegisterTemporary(const cling::Value& value);
   const ROOT::TMetaUtils::TNormalizedCtxt& GetNormalizedContext() const {return *fNormalizedCtxt;};
   TObject* GetObjectAddress(const char *Name, void *&LookupCtx);


   // core/meta helper functions.
   EReturnType MethodCallReturnType(TFunction *func) const final;
   virtual void GetFunctionName(const clang::Decl *decl, std::string &name) const;
   bool DiagnoseIfInterpreterException(const std::exception &e) const final;

   // CallFunc interface
   DeclId_t GetDeclId(CallFunc_t *info) const final;
   void   CallFunc_Delete(CallFunc_t* func) const final;
   void   CallFunc_Exec(CallFunc_t* func, void* address) const final;
   void   CallFunc_Exec(CallFunc_t* func, void* address, TInterpreterValue& val) const final;
   void   CallFunc_ExecWithReturn(CallFunc_t* func, void* address, void* ret) const final;
   void   CallFunc_ExecWithArgsAndReturn(CallFunc_t* func, void* address, const void* args[] = nullptr, int nargs = 0, void *ret = nullptr) const final;
   Longptr_t CallFunc_ExecInt(CallFunc_t* func, void* address) const final;
   Long64_t  CallFunc_ExecInt64(CallFunc_t* func, void* address) const final;
   Double_t  CallFunc_ExecDouble(CallFunc_t* func, void* address) const final;
   CallFunc_t*   CallFunc_Factory() const final;
   CallFunc_t*   CallFunc_FactoryCopy(CallFunc_t* func) const final;
   MethodInfo_t* CallFunc_FactoryMethod(CallFunc_t* func) const final;
   void   CallFunc_IgnoreExtraArgs(CallFunc_t* func, bool ignore) const final;
   void   CallFunc_Init(CallFunc_t* func) const final;
   bool   CallFunc_IsValid(CallFunc_t* func) const final;
   CallFuncIFacePtr_t CallFunc_IFacePtr(CallFunc_t * func) const final;
   void   CallFunc_ResetArg(CallFunc_t* func) const final;
   void   CallFunc_SetArg(CallFunc_t* func, Long_t param) const final;
   void   CallFunc_SetArg(CallFunc_t* func, ULong_t param) const final;
   void   CallFunc_SetArg(CallFunc_t* func, Float_t param) const final;
   void   CallFunc_SetArg(CallFunc_t* func, Double_t param) const final;
   void   CallFunc_SetArg(CallFunc_t* func, Long64_t param) const final;
   void   CallFunc_SetArg(CallFunc_t* func, ULong64_t param) const final;
   void   CallFunc_SetArgArray(CallFunc_t* func, Longptr_t* paramArr, Int_t nparam) const final;
   void   CallFunc_SetArgs(CallFunc_t* func, const char* param) const final;
   void   CallFunc_SetFunc(CallFunc_t* func, ClassInfo_t* info, const char* method, const char* params, Longptr_t* Offset) const final;
   void   CallFunc_SetFunc(CallFunc_t* func, ClassInfo_t* info, const char* method, const char* params, bool objectIsConst, Longptr_t* Offset) const final;
   void   CallFunc_SetFunc(CallFunc_t* func, MethodInfo_t* info) const final;
   void   CallFunc_SetFuncProto(CallFunc_t* func, ClassInfo_t* info, const char* method, const char* proto, Longptr_t* Offset, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch) const final;
   void   CallFunc_SetFuncProto(CallFunc_t* func, ClassInfo_t* info, const char* method, const char* proto, bool objectIsConst, Longptr_t* Offset, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch) const final;
   void   CallFunc_SetFuncProto(CallFunc_t* func, ClassInfo_t* info, const char* method, const std::vector<TypeInfo_t*> &proto, Longptr_t* Offset, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch) const final;
   void   CallFunc_SetFuncProto(CallFunc_t* func, ClassInfo_t* info, const char* method, const std::vector<TypeInfo_t*> &proto, bool objectIsConst, Longptr_t* Offset, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch) const final;

   std::string CallFunc_GetWrapperCode(CallFunc_t *func) const final;

   // ClassInfo interface
   DeclId_t GetDeclId(ClassInfo_t *info) const final;
   Bool_t ClassInfo_Contains(ClassInfo_t *info, DeclId_t declid) const final;
   Long_t ClassInfo_ClassProperty(ClassInfo_t* info) const final;
   void   ClassInfo_Delete(ClassInfo_t* info) const final;
   void   ClassInfo_Delete(ClassInfo_t* info, void* arena) const final;
   void   ClassInfo_DeleteArray(ClassInfo_t* info, void* arena, bool dtorOnly) const final;
   void   ClassInfo_Destruct(ClassInfo_t* info, void* arena) const final;
   ClassInfo_t*  ClassInfo_Factory(Bool_t all = kTRUE) const final;
   ClassInfo_t*  ClassInfo_Factory(ClassInfo_t* cl) const final;
   ClassInfo_t*  ClassInfo_Factory(const char* name) const final;
   ClassInfo_t*  ClassInfo_Factory(DeclId_t declid) const final;
   Longptr_t ClassInfo_GetBaseOffset(ClassInfo_t* fromDerived, ClassInfo_t* toBase, void * address, bool isDerivedObject) const final;
   int    ClassInfo_GetMethodNArg(ClassInfo_t* info, const char* method, const char* proto, Bool_t objectIsConst = false, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch) const final;
   bool   ClassInfo_HasDefaultConstructor(ClassInfo_t* info, Bool_t testio = kFALSE) const final;
   bool   ClassInfo_HasMethod(ClassInfo_t* info, const char* name) const final;
   void   ClassInfo_Init(ClassInfo_t* info, const char* funcname) const final;
   void   ClassInfo_Init(ClassInfo_t* info, int tagnum) const final;
   bool   ClassInfo_IsBase(ClassInfo_t* info, const char* name) const final;
   bool   ClassInfo_IsEnum(const char* name) const final;
   bool   ClassInfo_IsScopedEnum(ClassInfo_t* info) const final;
   EDataType ClassInfo_GetUnderlyingType(ClassInfo_t* info) const final;
   bool   ClassInfo_IsLoaded(ClassInfo_t* info) const final;
   bool   ClassInfo_IsValid(ClassInfo_t* info) const final;
   bool   ClassInfo_IsValidMethod(ClassInfo_t* info, const char* method, const char* proto, Longptr_t* offset, ROOT::EFunctionMatchMode /* mode */ = ROOT::kConversionMatch) const final;
   bool   ClassInfo_IsValidMethod(ClassInfo_t* info, const char* method, const char* proto, Bool_t objectIsConst, Longptr_t* offset, ROOT::EFunctionMatchMode /* mode */ = ROOT::kConversionMatch) const final;
   int    ClassInfo_Next(ClassInfo_t* info) const final;
   void*  ClassInfo_New(ClassInfo_t* info) const final;
   void*  ClassInfo_New(ClassInfo_t* info, int n) const final;
   void*  ClassInfo_New(ClassInfo_t* info, int n, void* arena) const final;
   void*  ClassInfo_New(ClassInfo_t* info, void* arena) const final;
   Long_t ClassInfo_Property(ClassInfo_t* info) const final;
   int    ClassInfo_Size(ClassInfo_t* info) const final;
   Longptr_t ClassInfo_Tagnum(ClassInfo_t* info) const final;
   const char* ClassInfo_FileName(ClassInfo_t* info) const final;
   const char* ClassInfo_FullName(ClassInfo_t* info) const final;
   const char* ClassInfo_Name(ClassInfo_t* info) const final;
   const char* ClassInfo_Title(ClassInfo_t* info) const final;
   const char* ClassInfo_TmpltName(ClassInfo_t* info) const final;

   // BaseClassInfo interface
   void   BaseClassInfo_Delete(BaseClassInfo_t* bcinfo) const final;
   BaseClassInfo_t*  BaseClassInfo_Factory(ClassInfo_t* info) const final;
   BaseClassInfo_t*  BaseClassInfo_Factory(ClassInfo_t* derived,
                                           ClassInfo_t* base) const final;
   int    BaseClassInfo_Next(BaseClassInfo_t* bcinfo) const final;
   int    BaseClassInfo_Next(BaseClassInfo_t* bcinfo, int onlyDirect) const final;
   Longptr_t BaseClassInfo_Offset(BaseClassInfo_t* toBaseClassInfo, void * address, bool isDerivedObject) const final;
   Long_t BaseClassInfo_Property(BaseClassInfo_t* bcinfo) const final;
   Longptr_t BaseClassInfo_Tagnum(BaseClassInfo_t* bcinfo) const final;
   ClassInfo_t*BaseClassInfo_ClassInfo(BaseClassInfo_t * /* bcinfo */) const final;
   const char* BaseClassInfo_FullName(BaseClassInfo_t* bcinfo) const final;
   const char* BaseClassInfo_Name(BaseClassInfo_t* bcinfo) const final;
   const char* BaseClassInfo_TmpltName(BaseClassInfo_t* bcinfo) const final;

   // DataMemberInfo interface
   DeclId_t GetDeclId(DataMemberInfo_t *info) const final;
   int    DataMemberInfo_ArrayDim(DataMemberInfo_t* dminfo) const final;
   void   DataMemberInfo_Delete(DataMemberInfo_t* dminfo) const final;
   DataMemberInfo_t*  DataMemberInfo_Factory(ClassInfo_t* clinfo, TDictionary::EMemberSelection selection) const final;
   DataMemberInfo_t  *DataMemberInfo_Factory(DeclId_t declid, ClassInfo_t* clinfo) const final;
   DataMemberInfo_t*  DataMemberInfo_FactoryCopy(DataMemberInfo_t* dminfo) const final;
   bool   DataMemberInfo_IsValid(DataMemberInfo_t* dminfo) const final;
   int    DataMemberInfo_MaxIndex(DataMemberInfo_t* dminfo, Int_t dim) const final;
   int    DataMemberInfo_Next(DataMemberInfo_t* dminfo) const final;
   Longptr_t DataMemberInfo_Offset(DataMemberInfo_t* dminfo) const final;
   Long_t DataMemberInfo_Property(DataMemberInfo_t* dminfo) const final;
   Long_t DataMemberInfo_TypeProperty(DataMemberInfo_t* dminfo) const final;
   int    DataMemberInfo_TypeSize(DataMemberInfo_t* dminfo) const final;
   const char* DataMemberInfo_TypeName(DataMemberInfo_t* dminfo) const final;
   const char* DataMemberInfo_TypeTrueName(DataMemberInfo_t* dminfo) const final;
   const char* DataMemberInfo_Name(DataMemberInfo_t* dminfo) const final;
   const char* DataMemberInfo_Title(DataMemberInfo_t* dminfo) const final;
   const char* DataMemberInfo_ValidArrayIndex(DataMemberInfo_t* dminfo) const final;
   void SetDeclAttr(DeclId_t, const char* /* attribute */) final;


   // Function Template interface
   DeclId_t GetDeclId(FuncTempInfo_t *info) const final;
   void   FuncTempInfo_Delete(FuncTempInfo_t * /* ft_info */) const final;
   FuncTempInfo_t  *FuncTempInfo_Factory(DeclId_t declid) const final;
   FuncTempInfo_t  *FuncTempInfo_FactoryCopy(FuncTempInfo_t * /* ft_info */) const final;
   Bool_t FuncTempInfo_IsValid(FuncTempInfo_t * /* ft_info */) const final;
   UInt_t FuncTempInfo_TemplateNargs(FuncTempInfo_t * /* ft_info */) const final;
   UInt_t FuncTempInfo_TemplateMinReqArgs(FuncTempInfo_t * /* ft_info */) const final;
   Long_t FuncTempInfo_Property(FuncTempInfo_t * /* ft_info */) const final;
   Long_t FuncTempInfo_ExtraProperty(FuncTempInfo_t * /* ft_info */) const final;
   void FuncTempInfo_Name(FuncTempInfo_t * /* ft_info */, TString& name) const final;
   void FuncTempInfo_Title(FuncTempInfo_t * /* ft_info */, TString& name) const final;

   // MethodInfo interface
   DeclId_t GetDeclId(MethodInfo_t *info) const final;
   void   MethodInfo_CreateSignature(MethodInfo_t* minfo, TString& signature) const final;
   void   MethodInfo_Delete(MethodInfo_t* minfo) const final;
   MethodInfo_t*  MethodInfo_Factory() const final;
   MethodInfo_t*  MethodInfo_Factory(ClassInfo_t *clinfo) const final;
   MethodInfo_t  *MethodInfo_Factory(DeclId_t declid) const final;
   MethodInfo_t*  MethodInfo_FactoryCopy(MethodInfo_t* minfo) const final;
   void*  MethodInfo_InterfaceMethod(MethodInfo_t* minfo) const final;
   bool   MethodInfo_IsValid(MethodInfo_t* minfo) const final;
   int    MethodInfo_NArg(MethodInfo_t* minfo) const final;
   int    MethodInfo_NDefaultArg(MethodInfo_t* minfo) const final;
   int    MethodInfo_Next(MethodInfo_t* minfo) const final;
   Long_t MethodInfo_Property(MethodInfo_t* minfo) const final;
   Long_t MethodInfo_ExtraProperty(MethodInfo_t* minfo) const final;
   TypeInfo_t*  MethodInfo_Type(MethodInfo_t* minfo) const final;
   EReturnType MethodInfo_MethodCallReturnType(MethodInfo_t* minfo) const final;
   const char* MethodInfo_GetMangledName(MethodInfo_t* minfo) const final;
   const char* MethodInfo_GetPrototype(MethodInfo_t* minfo) const final;
   const char* MethodInfo_Name(MethodInfo_t* minfo) const final;
   const char* MethodInfo_TypeName(MethodInfo_t* minfo) const final;
   std::string MethodInfo_TypeNormalizedName(MethodInfo_t* minfo) const final;
   const char* MethodInfo_Title(MethodInfo_t* minfo) const final;

   // MethodArgInfo interface
   void   MethodArgInfo_Delete(MethodArgInfo_t* marginfo) const final;
   MethodArgInfo_t*  MethodArgInfo_Factory() const final;
   MethodArgInfo_t*  MethodArgInfo_Factory(MethodInfo_t *minfo) const final;
   MethodArgInfo_t*  MethodArgInfo_FactoryCopy(MethodArgInfo_t* marginfo) const final;
   bool   MethodArgInfo_IsValid(MethodArgInfo_t* marginfo) const final;
   int    MethodArgInfo_Next(MethodArgInfo_t* marginfo) const final;
   Long_t MethodArgInfo_Property(MethodArgInfo_t* marginfo) const final;
   const char* MethodArgInfo_DefaultValue(MethodArgInfo_t* marginfo) const final;
   const char* MethodArgInfo_Name(MethodArgInfo_t* marginfo) const final;
   const char* MethodArgInfo_TypeName(MethodArgInfo_t* marginfo) const final;
   std::string MethodArgInfo_TypeNormalizedName(MethodArgInfo_t *marginfo) const final;

   // TypeInfo interface
   void   TypeInfo_Delete(TypeInfo_t* tinfo) const final;
   TypeInfo_t* TypeInfo_Factory() const final;
   TypeInfo_t *TypeInfo_Factory(const char* name) const final;
   TypeInfo_t* TypeInfo_FactoryCopy(TypeInfo_t* /* tinfo */) const final;
   void   TypeInfo_Init(TypeInfo_t* tinfo, const char* funcname) const final;
   bool   TypeInfo_IsValid(TypeInfo_t* tinfo) const final;
   const char* TypeInfo_Name(TypeInfo_t* /* info */) const final;
   Long_t TypeInfo_Property(TypeInfo_t* tinfo) const final;
   int    TypeInfo_RefType(TypeInfo_t* /* tinfo */) const final;
   int    TypeInfo_Size(TypeInfo_t* tinfo) const final;
   const char* TypeInfo_TrueName(TypeInfo_t* tinfo) const final;

   // TypedefInfo interface
   DeclId_t GetDeclId(TypedefInfo_t *info) const final;
   void   TypedefInfo_Delete(TypedefInfo_t* tinfo) const final;
   TypedefInfo_t*  TypedefInfo_Factory() const final;
   TypedefInfo_t*  TypedefInfo_Factory(const char* name) const final;
   TypedefInfo_t*  TypedefInfo_FactoryCopy(TypedefInfo_t* tinfo) const final;
   void   TypedefInfo_Init(TypedefInfo_t* tinfo, const char* name) const final;
   bool   TypedefInfo_IsValid(TypedefInfo_t* tinfo) const final;
   int    TypedefInfo_Next(TypedefInfo_t* tinfo) const final;
   Long_t TypedefInfo_Property(TypedefInfo_t* tinfo) const final;
   int    TypedefInfo_Size(TypedefInfo_t* tinfo) const final;
   const char* TypedefInfo_TrueName(TypedefInfo_t* tinfo) const final;
   const char* TypedefInfo_Name(TypedefInfo_t* tinfo) const final;
   const char* TypedefInfo_Title(TypedefInfo_t* tinfo) const final;

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
   TCling(const TCling&) = delete;
   TCling& operator=(const TCling&) = delete;

   void Execute(TMethod*, TObjArray*, int* /*error*/ = nullptr) final {}

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
