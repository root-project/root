// @(#)root/meta:$Id$
// Author: Axel Naumann, 2011-10-19

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
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
// Cling is a full ANSI compliant C++-11 interpreter based on           //
// clang/LLVM technology.                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TInterpreter
#include "TInterpreter.h"
#endif

#include <set>
#include <map>
#include <vector>

#ifndef WIN32
#define TWin32SendClass char
#endif

namespace clang {
   class CXXRecordDecl;
   class Decl;
   class EnumDecl;
   class FunctionDecl;
}
namespace cling {
   class Interpreter;
   class MetaProcessor;
   class StoredValueRef;
   class Transaction;
}

class TClingCallbacks;
class TEnv;
class THashTable;
class TInterpreterValue;
class TMethod;
class TObjArray;

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
   TObject* TCling__GetObjectAddress(const char *Name, void *&LookupCtx);
   const clang::Decl* TCling__GetObjectDecl(TObject *obj);

}

class TCling : public TInterpreter {
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
   THashTable*     fMapNamespaces;    // Entries for the namespaces, that we need to signal to clang.
   TObjArray*      fRootmapFiles;     // Loaded rootmap files.
   Bool_t          fLockProcessLine;  // True if ProcessLine should lock gClingMutex.

   cling::Interpreter*   fInterpreter;   // The interpreter.
   cling::MetaProcessor* fMetaProcessor; // The metaprocessor.

   std::vector<cling::StoredValueRef> *fTemporaries;    // Stack of temporaries
   ROOT::TMetaUtils::TNormalizedCtxt  *fNormalizedCtxt; // Which typedef to avoid striping.
   ROOT::TMetaUtils::TClingLookupHelper *fLookupHelper; // lookup helper used by TClassEdit

   void*           fPrevLoadedDynLibInfo; // Internal info to mark the last loaded libray.
   TClingCallbacks* fClingCallbacks; // cling::Interpreter owns it.
   Bool_t          fHaveSinglePCM; // Whether a single ROOT PCM was provided
   std::vector<const void*> fDeserializedDecls; // Decls read from the AST
   struct CharPtrCmp_t {
      bool operator()(const char* a, const char *b) const {
         return strcmp(a, b) < 0;
      }
   };
   typedef std::map<const char* /*hdr*/, const char* /*mod*/, CharPtrCmp_t>
      ModuleForHeader_t;
   ModuleForHeader_t fModuleForHeader; // Which module a header is in. Assumes string storage in dictionary.
   std::set<TClass*> fModTClasses;

public: // Public Interface
   virtual ~TCling();
   TCling(const char* name, const char* title);

   cling::Interpreter *GetInterpreter() { return fInterpreter; }

   void    AddIncludePath(const char* path);
   Int_t   AutoLoad(const char* cls);
   Bool_t  IsAutoLoadNamespaceCandidate(const char* name);
   void    ClearFileBusy();
   void    ClearStack(); // Delete existing temporary values
   void    EnableAutoLoading();
   void    EndOfLineAction();
   Int_t   GetExitCode() const { return fExitCode; }
   TEnv*   GetMapfile() const { return fMapfile; }
   Int_t   GetMore() const { return fMore; }
   TClass *GenerateTClass(const char *classname, Bool_t emulation, Bool_t silent = kFALSE);
   TClass *GenerateTClass(ClassInfo_t *classinfo, Bool_t silent = kFALSE);
   Int_t   GenerateDictionary(const char* classes, const char* includes = 0, const char* options = 0);
   char*   GetPrompt() { return fPrompt; }
   const char* GetSharedLibs();
   const char* GetClassSharedLibs(const char* cls);
   const char* GetSharedLibDeps(const char* lib);
   const char* GetIncludePath();
   virtual const char* GetSTLIncludePath() const;
   TObjArray*  GetRootMapFiles() const { return fRootmapFiles; }
   virtual void Initialize();
   void    InspectMembers(TMemberInspector&, void* obj, const TClass* cl);
   Bool_t  IsLoaded(const char* filename) const;
   Int_t   Load(const char* filenam, Bool_t system = kFALSE);
   void    LoadMacro(const char* filename, EErrorCode* error = 0);
   Int_t   LoadLibraryMap(const char* rootmapfile = 0);
   Int_t   RescanLibraryMap();
   Int_t   ReloadAllSharedLibraryMaps();
   Int_t   UnloadAllSharedLibraryMaps();
   Int_t   UnloadLibraryMap(const char* library);
   Long_t  ProcessLine(const char* line, EErrorCode* error = 0);
   Long_t  ProcessLineAsynch(const char* line, EErrorCode* error = 0);
   Long_t  ProcessLineSynch(const char* line, EErrorCode* error = 0);
   void    PrintIntro();
   void    RegisterModule(const char* modulename,
                          const char** headers,
                          const char** allHeaders,
                          const char** includePaths,
                          const char** macroDefines,
                          const char** macroUndefines,
                          void (*triggerFunc)());
   Int_t   SetClassSharedLibs(const char *cls, const char *libs);
   void    SetGetline(const char * (*getlineFunc)(const char* prompt),
                      void (*histaddFunc)(const char* line));
   void    Reset();
   void    ResetAll();
   void    ResetGlobals();
   void    ResetGlobalVar(void* obj);
   void    RewindDictionary();
   Int_t   DeleteGlobal(void* obj);
   Int_t   DeleteVariable(const char *name);
   void    SaveContext();
   void    SaveGlobalsContext();
   void    UpdateListOfGlobals();
   void    UpdateListOfGlobalFunctions();
   void    UpdateListOfTypes();
   void    SetClassInfo(TClass* cl, Bool_t reload = kFALSE);
   Bool_t  CheckClassInfo(const char* name, Bool_t autoload = kTRUE);
   Bool_t  CheckClassTemplate(const char *name);
   Long_t  Calc(const char* line, EErrorCode* error = 0);
   void    CreateListOfBaseClasses(TClass* cl) const;
   void    CreateListOfDataMembers(TClass* cl) const;
   void    CreateListOfEnums(TClass* cl) const;
   void    CreateListOfMethods(TClass* cl) const;
   void    CreateListOfMethodArgs(TFunction* m) const;
   void    UpdateListOfMethods(TClass* cl) const;
   void    UpdateListOfDataMembers(TClass* cl) const;
   void    UpdateListOfEnums(TClass* cl) const;

   TString GetMangledName(TClass* cl, const char* method, const char* params, Bool_t objectIsConst = kFALSE);
   TString GetMangledNameWithPrototype(TClass* cl, const char* method, const char* proto, Bool_t objectIsConst = kFALSE, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch);
   void*   GetInterfaceMethod(TClass* cl, const char* method, const char* params, Bool_t objectIsConst = kFALSE);
   void*   GetInterfaceMethodWithPrototype(TClass* cl, const char* method, const char* proto, Bool_t objectIsConst = kFALSE, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch);
   const char* GetInterpreterTypeName(const char* name, Bool_t full = kFALSE);
   void    Execute(const char* function, const char* params, int* error = 0);
   void    Execute(TObject* obj, TClass* cl, const char* method, const char* params, int* error = 0);
   void    Execute(TObject* obj, TClass* cl, const char* method, const char* params, Bool_t objectIsConst, int* error = 0);
   void    Execute(TObject* obj, TClass* cl, TMethod* method, TObjArray* params, int* error = 0);
   Long_t  ExecuteMacro(const char* filename, EErrorCode* error = 0);
   void    RecursiveRemove(TObject* obj);
   Bool_t  IsErrorMessagesEnabled() const;
   Bool_t  SetErrorMessages(Bool_t enable = kTRUE);
   Bool_t  IsProcessLineLocked() const {
      return fLockProcessLine;
   }
   void    SetProcessLineLock(Bool_t lock = kTRUE) {
      fLockProcessLine = lock;
   }
   const char* TypeName(const char* typeDesc);

   static int   AutoLoadCallback(const char* cls, const char* lib);
   static void  UpdateClassInfo(char* name, Long_t tagnum);
   static void  UpdateClassInfoWork(const char* name);
          void  UpdateClassInfoWithDecl(const void* vTD);
   static void  UpdateAllCanvases();

   // Misc
   virtual int    DisplayClass(FILE* fout, const char* name, int base, int start) const;
   virtual int    DisplayIncludePath(FILE* fout) const;
   virtual void*  FindSym(const char* entry) const;
   virtual void   GenericError(const char* error) const;
   virtual Long_t GetExecByteCode() const;
   virtual Long_t Getgvp() const;
   virtual const char* Getp2f2funcname(void* receiver) const;
   virtual const char* GetTopLevelMacroName() const;
   virtual const char* GetCurrentMacroName() const;
   virtual int    GetSecurityError() const;
   virtual int    LoadFile(const char* path) const;
   virtual void   LoadText(const char* text) const;
   virtual const char* MapCppName(const char*) const;
   virtual void   SetAlloclockfunc(void (*)()) const;
   virtual void   SetAllocunlockfunc(void (*)()) const;
   virtual int    SetClassAutoloading(int) const;
   virtual void   SetErrmsgcallback(void* p) const;
   virtual void   Setgvp(Long_t) const;
   virtual void   SetRTLD_NOW() const;
   virtual void   SetRTLD_LAZY() const;
   virtual void   SetTempLevel(int val) const;
   virtual int    UnloadFile(const char* path) const;

   TInterpreterValue *CreateTemporary();
   void               RegisterTemporary(const TInterpreterValue& value);
   void               RegisterTemporary(const cling::StoredValueRef& value);

   // core/meta helper functions.
   virtual TMethodCall::EReturnType MethodCallReturnType(TFunction *func) const;

   // CallFunc interface
   virtual void   CallFunc_Delete(CallFunc_t* func) const;
   virtual void   CallFunc_Exec(CallFunc_t* func, void* address) const;
   virtual void   CallFunc_Exec(CallFunc_t* func, void* address, TInterpreterValue& val) const;
   virtual Long_t    CallFunc_ExecInt(CallFunc_t* func, void* address) const;
   virtual Long64_t  CallFunc_ExecInt64(CallFunc_t* func, void* address) const;
   virtual Double_t  CallFunc_ExecDouble(CallFunc_t* func, void* address) const;
   virtual CallFunc_t*   CallFunc_Factory() const;
   virtual CallFunc_t*   CallFunc_FactoryCopy(CallFunc_t* func) const;
   virtual MethodInfo_t* CallFunc_FactoryMethod(CallFunc_t* func) const;
   virtual void   CallFunc_IgnoreExtraArgs(CallFunc_t* func, bool ignore) const;
   virtual void   CallFunc_Init(CallFunc_t* func) const;
   virtual bool   CallFunc_IsValid(CallFunc_t* func) const;
   virtual void   CallFunc_ResetArg(CallFunc_t* func) const;
   virtual void   CallFunc_SetArg(CallFunc_t* func, Long_t param) const;
   virtual void   CallFunc_SetArg(CallFunc_t* func, Double_t param) const;
   virtual void   CallFunc_SetArg(CallFunc_t* func, Long64_t param) const;
   virtual void   CallFunc_SetArg(CallFunc_t* func, ULong64_t param) const;
   virtual void   CallFunc_SetArgArray(CallFunc_t* func, Long_t* paramArr, Int_t nparam) const;
   virtual void   CallFunc_SetArgs(CallFunc_t* func, const char* param) const;
   virtual void   CallFunc_SetFunc(CallFunc_t* func, ClassInfo_t* info, const char* method, const char* params, Long_t* Offset) const;
   virtual void   CallFunc_SetFunc(CallFunc_t* func, ClassInfo_t* info, const char* method, const char* params, bool objectIsConst, Long_t* Offset) const;
   virtual void   CallFunc_SetFunc(CallFunc_t* func, MethodInfo_t* info) const;
   virtual void   CallFunc_SetFuncProto(CallFunc_t* func, ClassInfo_t* info, const char* method, const char* proto, Long_t* Offset, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch) const;
   virtual void   CallFunc_SetFuncProto(CallFunc_t* func, ClassInfo_t* info, const char* method, const char* proto, bool objectIsConst, Long_t* Offset, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch) const;
   virtual void   CallFunc_SetFuncProto(CallFunc_t* func, ClassInfo_t* info, const char* method, const std::vector<TypeInfo_t*> &proto, Long_t* Offset, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch) const;
   virtual void   CallFunc_SetFuncProto(CallFunc_t* func, ClassInfo_t* info, const char* method, const std::vector<TypeInfo_t*> &proto, bool objectIsConst, Long_t* Offset, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch) const;

   // ClassInfo interface
   virtual Long_t ClassInfo_ClassProperty(ClassInfo_t* info) const;
   virtual void   ClassInfo_Delete(ClassInfo_t* info) const;
   virtual void   ClassInfo_Delete(ClassInfo_t* info, void* arena) const;
   virtual void   ClassInfo_DeleteArray(ClassInfo_t* info, void* arena, bool dtorOnly) const;
   virtual void   ClassInfo_Destruct(ClassInfo_t* info, void* arena) const;
   virtual ClassInfo_t*  ClassInfo_Factory() const;
   virtual ClassInfo_t*  ClassInfo_Factory(ClassInfo_t* cl) const;
   virtual ClassInfo_t*  ClassInfo_Factory(const char* name) const;
   virtual int    ClassInfo_GetMethodNArg(ClassInfo_t* info, const char* method, const char* proto, Bool_t objectIsConst = false, ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch) const;
   virtual bool   ClassInfo_HasDefaultConstructor(ClassInfo_t* info) const;
   virtual bool   ClassInfo_HasMethod(ClassInfo_t* info, const char* name) const;
   virtual void   ClassInfo_Init(ClassInfo_t* info, const char* funcname) const;
   virtual void   ClassInfo_Init(ClassInfo_t* info, int tagnum) const;
   virtual bool   ClassInfo_IsBase(ClassInfo_t* info, const char* name) const;
   virtual bool   ClassInfo_IsEnum(const char* name) const;
   virtual bool   ClassInfo_IsLoaded(ClassInfo_t* info) const;
   virtual bool   ClassInfo_IsValid(ClassInfo_t* info) const;
   virtual bool   ClassInfo_IsValidMethod(ClassInfo_t* info, const char* method, const char* proto, Long_t* offset, ROOT::EFunctionMatchMode /* mode */ = ROOT::kConversionMatch) const;
   virtual bool   ClassInfo_IsValidMethod(ClassInfo_t* info, const char* method, const char* proto, Bool_t objectIsConst, Long_t* offset, ROOT::EFunctionMatchMode /* mode */ = ROOT::kConversionMatch) const;
   virtual int    ClassInfo_Next(ClassInfo_t* info) const;
   virtual void*  ClassInfo_New(ClassInfo_t* info) const;
   virtual void*  ClassInfo_New(ClassInfo_t* info, int n) const;
   virtual void*  ClassInfo_New(ClassInfo_t* info, int n, void* arena) const;
   virtual void*  ClassInfo_New(ClassInfo_t* info, void* arena) const;
   virtual Long_t ClassInfo_Property(ClassInfo_t* info) const;
   virtual int    ClassInfo_Size(ClassInfo_t* info) const;
   virtual Long_t ClassInfo_Tagnum(ClassInfo_t* info) const;
   virtual const char* ClassInfo_FileName(ClassInfo_t* info) const;
   virtual const char* ClassInfo_FullName(ClassInfo_t* info) const;
   virtual const char* ClassInfo_Name(ClassInfo_t* info) const;
   virtual const char* ClassInfo_Title(ClassInfo_t* info) const;
   virtual const char* ClassInfo_TmpltName(ClassInfo_t* info) const;

   // BaseClassInfo interface
   virtual void   BaseClassInfo_Delete(BaseClassInfo_t* bcinfo) const;
   virtual BaseClassInfo_t*  BaseClassInfo_Factory(ClassInfo_t* info) const;
   virtual int    BaseClassInfo_Next(BaseClassInfo_t* bcinfo) const;
   virtual int    BaseClassInfo_Next(BaseClassInfo_t* bcinfo, int onlyDirect) const;
   virtual Long_t BaseClassInfo_Offset(BaseClassInfo_t* bcinfo) const;
   virtual Long_t BaseClassInfo_Property(BaseClassInfo_t* bcinfo) const;
   virtual Long_t BaseClassInfo_Tagnum(BaseClassInfo_t* bcinfo) const;
   virtual const char* BaseClassInfo_FullName(BaseClassInfo_t* bcinfo) const;
   virtual const char* BaseClassInfo_Name(BaseClassInfo_t* bcinfo) const;
   virtual const char* BaseClassInfo_TmpltName(BaseClassInfo_t* bcinfo) const;

   // DataMemberInfo interface
   virtual int    DataMemberInfo_ArrayDim(DataMemberInfo_t* dminfo) const;
   virtual void   DataMemberInfo_Delete(DataMemberInfo_t* dminfo) const;
   virtual DataMemberInfo_t*  DataMemberInfo_Factory(ClassInfo_t* clinfo = 0) const;
   virtual DataMemberInfo_t*  DataMemberInfo_FactoryCopy(DataMemberInfo_t* dminfo) const;
   virtual bool   DataMemberInfo_IsValid(DataMemberInfo_t* dminfo) const;
   virtual int    DataMemberInfo_MaxIndex(DataMemberInfo_t* dminfo, Int_t dim) const;
   virtual int    DataMemberInfo_Next(DataMemberInfo_t* dminfo) const;
   virtual Long_t DataMemberInfo_Offset(DataMemberInfo_t* dminfo) const;
   virtual Long_t DataMemberInfo_Property(DataMemberInfo_t* dminfo) const;
   virtual Long_t DataMemberInfo_TypeProperty(DataMemberInfo_t* dminfo) const;
   virtual int    DataMemberInfo_TypeSize(DataMemberInfo_t* dminfo) const;
   virtual const char* DataMemberInfo_TypeName(DataMemberInfo_t* dminfo) const;
   virtual const char* DataMemberInfo_TypeTrueName(DataMemberInfo_t* dminfo) const;
   virtual const char* DataMemberInfo_Name(DataMemberInfo_t* dminfo) const;
   virtual const char* DataMemberInfo_Title(DataMemberInfo_t* dminfo) const;
   virtual const char* DataMemberInfo_ValidArrayIndex(DataMemberInfo_t* dminfo) const;

   // MethodInfo interface
   virtual void   MethodInfo_CreateSignature(MethodInfo_t* minfo, TString& signature) const;
   virtual void   MethodInfo_Delete(MethodInfo_t* minfo) const;
   virtual MethodInfo_t*  MethodInfo_Factory() const;
   virtual MethodInfo_t*  MethodInfo_Factory(ClassInfo_t *clinfo) const;
   virtual MethodInfo_t*  MethodInfo_FactoryCopy(MethodInfo_t* minfo) const;
   virtual void*  MethodInfo_InterfaceMethod(MethodInfo_t* minfo) const;
   virtual bool   MethodInfo_IsValid(MethodInfo_t* minfo) const;
   virtual int    MethodInfo_NArg(MethodInfo_t* minfo) const;
   virtual int    MethodInfo_NDefaultArg(MethodInfo_t* minfo) const;
   virtual int    MethodInfo_Next(MethodInfo_t* minfo) const;
   virtual Long_t MethodInfo_Property(MethodInfo_t* minfo) const;
   virtual TypeInfo_t*  MethodInfo_Type(MethodInfo_t* minfo) const;
   virtual TMethodCall::EReturnType MethodInfo_MethodCallReturnType(MethodInfo_t* minfo) const;
   virtual const char* MethodInfo_GetMangledName(MethodInfo_t* minfo) const;
   virtual const char* MethodInfo_GetPrototype(MethodInfo_t* minfo) const;
   virtual const char* MethodInfo_Name(MethodInfo_t* minfo) const;
   virtual const char* MethodInfo_TypeName(MethodInfo_t* minfo) const;
   virtual std::string MethodInfo_TypeNormalizedName(MethodInfo_t* minfo) const;
   virtual const char* MethodInfo_Title(MethodInfo_t* minfo) const;

   // MethodArgInfo interface
   virtual void   MethodArgInfo_Delete(MethodArgInfo_t* marginfo) const;
   virtual MethodArgInfo_t*  MethodArgInfo_Factory() const;
   virtual MethodArgInfo_t*  MethodArgInfo_Factory(MethodInfo_t *minfo) const;
   virtual MethodArgInfo_t*  MethodArgInfo_FactoryCopy(MethodArgInfo_t* marginfo) const;
   virtual bool   MethodArgInfo_IsValid(MethodArgInfo_t* marginfo) const;
   virtual int    MethodArgInfo_Next(MethodArgInfo_t* marginfo) const;
   virtual Long_t MethodArgInfo_Property(MethodArgInfo_t* marginfo) const;
   virtual const char* MethodArgInfo_DefaultValue(MethodArgInfo_t* marginfo) const;
   virtual const char* MethodArgInfo_Name(MethodArgInfo_t* marginfo) const;
   virtual const char* MethodArgInfo_TypeName(MethodArgInfo_t* marginfo) const;
   virtual std::string MethodArgInfo_TypeNormalizedName(MethodArgInfo_t *marginfo) const;

   // TypeInfo interface
   virtual void   TypeInfo_Delete(TypeInfo_t* tinfo) const;
   virtual TypeInfo_t* TypeInfo_Factory() const;
   virtual TypeInfo_t *TypeInfo_Factory(const char* name) const;
   virtual TypeInfo_t* TypeInfo_FactoryCopy(TypeInfo_t* /* tinfo */) const;
   virtual void   TypeInfo_Init(TypeInfo_t* tinfo, const char* funcname) const;
   virtual bool   TypeInfo_IsValid(TypeInfo_t* tinfo) const;
   virtual const char* TypeInfo_Name(TypeInfo_t* /* info */) const;
   virtual Long_t TypeInfo_Property(TypeInfo_t* tinfo) const;
   virtual int    TypeInfo_RefType(TypeInfo_t* /* tinfo */) const;
   virtual int    TypeInfo_Size(TypeInfo_t* tinfo) const;
   virtual const char* TypeInfo_TrueName(TypeInfo_t* tinfo) const;

   // TypedefInfo interface
   virtual void   TypedefInfo_Delete(TypedefInfo_t* tinfo) const;
   virtual TypedefInfo_t*  TypedefInfo_Factory() const;
   virtual TypedefInfo_t*  TypedefInfo_Factory(const char* name) const;
   virtual TypedefInfo_t*  TypedefInfo_FactoryCopy(TypedefInfo_t* tinfo) const;
   virtual void   TypedefInfo_Init(TypedefInfo_t* tinfo, const char* name) const;
   virtual bool   TypedefInfo_IsValid(TypedefInfo_t* tinfo) const;
   virtual int    TypedefInfo_Next(TypedefInfo_t* tinfo) const;
   virtual Long_t TypedefInfo_Property(TypedefInfo_t* tinfo) const;
   virtual int    TypedefInfo_Size(TypedefInfo_t* tinfo) const;
   virtual const char* TypedefInfo_TrueName(TypedefInfo_t* tinfo) const;
   virtual const char* TypedefInfo_Name(TypedefInfo_t* tinfo) const;
   virtual const char* TypedefInfo_Title(TypedefInfo_t* tinfo) const;

   void AddDeserializedDecl(const void* D) { fDeserializedDecls.push_back(D); }
   std::vector<const void*>& GetDeserializedDecls() { return fDeserializedDecls; }
   std::set<TClass*>& GetModTClasses() { return fModTClasses; }

   void HandleNewDecl(const void* DV, bool isDeserialized, std::set<TClass*>& modifiedClasses);


private: // Private Utility Functions
   TCling();
   TCling(const TCling&); // NOT IMPLEMENTED
   TCling& operator=(const TCling&); // NOT IMPLEMENTED

   void Execute(TMethod*, TObjArray*, int* /*error*/ = 0)
   {
   }

   void UpdateListOfLoadedSharedLibraries();
   void RegisterLoadedSharedLibrary(const char* name);
   void AddFriendToClass(clang::FunctionDecl*, clang::CXXRecordDecl*) const;

   bool LoadPCM(TString pcmFileName, const char** headers,
                void (*triggerFunc)()) const;
   void HandleEnumDecl(const clang::Decl* D, bool isGlobal, TClass *cl = 0) const;

   ClassDef(TCling, 0) //Interface to cling C++ interpreter
};

#endif
