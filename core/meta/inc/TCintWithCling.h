// @(#)root/meta:$Id$
// Author: Axel Naumann, 2011-10-19

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TCintWithCling
#define ROOT_TCintWithCling

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCintWithCling                                                       //
//                                                                      //
// This class defines an interface to the CINT C/C++ interpreter made   //
// by Masaharu Goto of HP Japan.                                        //
//                                                                      //
// CINT is an almost full ANSI compliant C/C++ interpreter.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TInterpreter
#include "TInterpreter.h"
#endif // ROOT_TInterpreter

#ifndef __CINT__
#include "G__ci.h"
#include "Api.h"
#else // __CINT__
struct G__dictposition;
#endif // __CINT__

#ifndef WIN32
#define TWin32SendClass char
#endif // WIN32

namespace Cint {
class G__ClassInfo;
}

using namespace Cint;

class TMethod;
class TObjArray;
class TEnv;

namespace cling {
class Interpreter;
class MetaProcessor;
}

class TCintWithCling : public TInterpreter {

private: // Static Data Members

   static void* fgSetOfSpecials; // set of TObjects used in CINT variables

private: // Data Members

   Int_t fMore; // 1 if more input is required
   Int_t fExitCode; // value passed to exit() in interpreter
   char fPrompt[64]; // proposed prompt string
   G__dictposition fDictPos; // CINT dictionary context after init
   G__dictposition fDictPosGlobals; // CINT dict context after ResetGlobals()
   TString fSharedLibs; // list of shared libraries loaded by G__loadfile
   Int_t fSharedLibsSerial; // last time we set fSharedLibs
   Int_t fGlobalsListSerial; // last time we refresh the ROOT list of globals.
   TString fIncludePath; // list of CINT include paths
   TString fRootmapLoadPath; // dynamic load path used for rootmap files
   TEnv* fMapfile; // map of classes and libraries
   TObjArray* fRootmapFiles; // list of non-default rootmap files loaded
   Bool_t fLockProcessLine; // true if ProcessLine should lock gCINTMutex
   cling::Interpreter* fInterpreter; // cling
   cling::MetaProcessor* fMetaProcessor; // cling's command processor

public: // Public Interface

   virtual ~TCintWithCling();
   TCintWithCling(const char* name, const char* title);

   void    AddIncludePath(const char* path);
   Int_t   AutoLoad(const char* classname);
   void    ClearFileBusy();
   void    ClearStack(); // Delete existing temporary values
   void    EnableAutoLoading();
   void    EndOfLineAction();
   Int_t   GetExitCode() const { return fExitCode; }
   TEnv*   GetMapfile() const { return fMapfile; }
   Int_t   GetMore() const { return fMore; }
   Int_t   GenerateDictionary(const char* classes, const char* includes = 0, const char* options = 0);
   char*   GetPrompt() { return fPrompt; }
   const char* GetSharedLibs();
   const char* GetClassSharedLibs(const char* cls);
   const char* GetSharedLibDeps(const char* lib);
   const char* GetIncludePath();
   virtual const char* GetSTLIncludePath() const;
   TObjArray*  GetRootMapFiles() const { return fRootmapFiles; }
   Int_t   InitializeDictionaries();
   void    InspectMembers(TMemberInspector&, void* obj, const char* clname);
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
   void    SetGetline(const char * (*getlineFunc)(const char* prompt),
                      void (*histaddFunc)(const char* line));
   void    Reset();
   void    ResetAll();
   void    ResetGlobals();
   void    ResetGlobalVar(void* obj);
   void    RewindDictionary();
   Int_t   DeleteGlobal(void* obj);
   void    SaveContext();
   void    SaveGlobalsContext();
   void    UpdateListOfGlobals();
   void    UpdateListOfGlobalFunctions();
   void    UpdateListOfTypes();
   void    SetClassInfo(TClass* cl, Bool_t reload = kFALSE);
   Bool_t  CheckClassInfo(const char* name, Bool_t autoload = kTRUE);
   Long_t  Calc(const char* line, EErrorCode* error = 0);
   void    CreateListOfBaseClasses(TClass* cl);
   void    CreateListOfDataMembers(TClass* cl);
   void    CreateListOfMethods(TClass* cl);
   void    CreateListOfMethodArgs(TFunction* m);
   void    UpdateListOfMethods(TClass* cl);

   TString GetMangledName(TClass* cl, const char* method, const char* params);
   TString GetMangledNameWithPrototype(TClass* cl, const char* method, const char* proto);
   void*   GetInterfaceMethod(TClass* cl, const char* method, const char* params);
   void*   GetInterfaceMethodWithPrototype(TClass* cl, const char* method, const char* proto);
   const char* GetInterpreterTypeName(const char* name, Bool_t full = kFALSE);
   void    Execute(const char* function, const char* params, int* error = 0);
   void    Execute(TObject* obj, TClass* cl, const char* method, const char* params, int* error = 0);
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

   static void* FindSpecialObject(const char* name, G__ClassInfo* type, void** prevObj, void** assocPtr);
   static int   AutoLoadCallback(const char* cls, const char* lib);
   static void  UpdateClassInfo(char* name, Long_t tagnum);
   static void  UpdateClassInfoWork(const char* name, Long_t tagnum);
   static void  UpdateAllCanvases();

   // Misc
   virtual int    DisplayClass(FILE* fout, char* name, int base, int start) const;
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


   // G__CallFunc interface
   virtual void   CallFunc_Delete(void* func) const;
   virtual void   CallFunc_Exec(CallFunc_t* func, void* address) const;
   virtual Long_t    CallFunc_ExecInt(CallFunc_t* func, void* address) const;
   virtual Long_t    CallFunc_ExecInt64(CallFunc_t* func, void* address) const;
   virtual Double_t  CallFunc_ExecDouble(CallFunc_t* func, void* address) const;
   virtual CallFunc_t*   CallFunc_Factory() const;
   virtual CallFunc_t*   CallFunc_FactoryCopy(CallFunc_t* func) const;
   virtual MethodInfo_t* CallFunc_FactoryMethod(CallFunc_t* func) const;
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
   virtual void   CallFunc_SetFunc(CallFunc_t* func, MethodInfo_t* info) const;
   virtual void   CallFunc_SetFuncProto(CallFunc_t* func, ClassInfo_t* info, const char* method, const char* proto, Long_t* Offset) const;


   // G__ClassInfo interface
   virtual Long_t ClassInfo_ClassProperty(ClassInfo_t* info) const;
   virtual void   ClassInfo_Delete(ClassInfo_t* info) const;
   virtual void   ClassInfo_Delete(ClassInfo_t* info, void* arena) const;
   virtual void   ClassInfo_DeleteArray(ClassInfo_t* info, void* arena, bool dtorOnly) const;
   virtual void   ClassInfo_Destruct(ClassInfo_t* info, void* arena) const;
   virtual ClassInfo_t*  ClassInfo_Factory() const;
   //virtual ClassInfo_t  *ClassInfo_Factory(G__value * /* value */) const;
   virtual ClassInfo_t*  ClassInfo_Factory(ClassInfo_t* cl) const;
   virtual ClassInfo_t*  ClassInfo_Factory(const char* name) const;
   virtual ClassInfo_t*  ClassInfo_Factory(G__value*) const;
   virtual int    ClassInfo_GetMethodNArg(ClassInfo_t* info, const char* method, const char* proto) const;
   virtual bool   ClassInfo_HasDefaultConstructor(ClassInfo_t* info) const;
   virtual bool   ClassInfo_HasMethod(ClassInfo_t* info, const char* name) const;
   virtual void   ClassInfo_Init(ClassInfo_t* info, const char* funcname) const;
   virtual void   ClassInfo_Init(ClassInfo_t* info, int tagnum) const;
   virtual bool   ClassInfo_IsBase(ClassInfo_t* info, const char* name) const;
   virtual bool   ClassInfo_IsEnum(const char* name) const;
   virtual bool   ClassInfo_IsLoaded(ClassInfo_t* info) const;
   virtual bool   ClassInfo_IsValid(ClassInfo_t* info) const;
   virtual bool   ClassInfo_IsValidMethod(ClassInfo_t* info, const char* method, const char* proto, Long_t* offset) const;
   virtual int    ClassInfo_Next(ClassInfo_t* info) const;
   virtual void*  ClassInfo_New(ClassInfo_t* info) const;
   virtual void*  ClassInfo_New(ClassInfo_t* info, int n) const;
   virtual void*  ClassInfo_New(ClassInfo_t* info, int n, void* arena) const;
   virtual void*  ClassInfo_New(ClassInfo_t* info, void* arena) const;
   virtual Long_t ClassInfo_Property(ClassInfo_t* info) const;
   virtual int    ClassInfo_RootFlag(ClassInfo_t* info) const;
   virtual int    ClassInfo_Size(ClassInfo_t* info) const;
   virtual Long_t ClassInfo_Tagnum(ClassInfo_t* info) const;
   virtual const char* ClassInfo_FileName(ClassInfo_t* info) const;
   virtual const char* ClassInfo_FullName(ClassInfo_t* info) const;
   virtual const char* ClassInfo_Name(ClassInfo_t* info) const;
   virtual const char* ClassInfo_Title(ClassInfo_t* info) const;
   virtual const char* ClassInfo_TmpltName(ClassInfo_t* info) const;


   // G__BaseClassInfo interface
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

   // G__DataMemberInfo interface
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

   // G__MethodInfo interface
   virtual void   MethodInfo_CreateSignature(MethodInfo_t* minfo, TString& signature) const;
   virtual void   MethodInfo_Delete(MethodInfo_t* minfo) const;
   virtual MethodInfo_t*  MethodInfo_Factory() const;
   virtual MethodInfo_t*  MethodInfo_FactoryCopy(MethodInfo_t* minfo) const;
   virtual MethodInfo_t*  MethodInfo_InterfaceMethod(MethodInfo_t* minfo) const;
   virtual bool   MethodInfo_IsValid(MethodInfo_t* minfo) const;
   virtual int    MethodInfo_NArg(MethodInfo_t* minfo) const;
   virtual int    MethodInfo_NDefaultArg(MethodInfo_t* minfo) const;
   virtual int    MethodInfo_Next(MethodInfo_t* minfo) const;
   virtual Long_t MethodInfo_Property(MethodInfo_t* minfo) const;
   virtual TypeInfo_t*  MethodInfo_Type(MethodInfo_t* minfo) const;
   virtual const char* MethodInfo_GetMangledName(MethodInfo_t* minfo) const;
   virtual const char* MethodInfo_GetPrototype(MethodInfo_t* minfo) const;
   virtual const char* MethodInfo_Name(MethodInfo_t* minfo) const;
   virtual const char* MethodInfo_TypeName(MethodInfo_t* minfo) const;
   virtual const char* MethodInfo_Title(MethodInfo_t* minfo) const;

   // G__MethodArgInfo interface
   virtual void   MethodArgInfo_Delete(MethodArgInfo_t* marginfo) const;
   virtual MethodArgInfo_t*  MethodArgInfo_Factory() const;
   virtual MethodArgInfo_t*  MethodArgInfo_FactoryCopy(MethodArgInfo_t* marginfo) const;
   virtual bool   MethodArgInfo_IsValid(MethodArgInfo_t* marginfo) const;
   virtual int    MethodArgInfo_Next(MethodArgInfo_t* marginfo) const;
   virtual Long_t MethodArgInfo_Property(MethodArgInfo_t* marginfo) const;
   virtual const char* MethodArgInfo_DefaultValue(MethodArgInfo_t* marginfo) const;
   virtual const char* MethodArgInfo_Name(MethodArgInfo_t* marginfo) const;
   virtual const char* MethodArgInfo_TypeName(MethodArgInfo_t* marginfo) const;


   // G__TypeInfo interface
   virtual void   TypeInfo_Delete(TypeInfo_t* tinfo) const;
   virtual TypeInfo_t* TypeInfo_Factory() const;
   virtual TypeInfo_t* TypeInfo_Factory(G__value* /* value */) const;
   virtual TypeInfo_t* TypeInfo_FactoryCopy(TypeInfo_t* /* tinfo */) const;
   virtual void   TypeInfo_Init(TypeInfo_t* tinfo, const char* funcname) const;
   virtual bool   TypeInfo_IsValid(TypeInfo_t* tinfo) const;
   virtual const char* TypeInfo_Name(TypeInfo_t* /* info */) const;
   virtual Long_t TypeInfo_Property(TypeInfo_t* tinfo) const;
   virtual int    TypeInfo_RefType(TypeInfo_t* /* tinfo */) const;
   virtual int    TypeInfo_Size(TypeInfo_t* tinfo) const;
   virtual const char* TypeInfo_TrueName(TypeInfo_t* tinfo) const;


   // G__TypedefInfo interface
   virtual void   TypedefInfo_Delete(TypedefInfo_t* tinfo) const;
   virtual TypedefInfo_t*  TypedefInfo_Factory() const;
   virtual TypedefInfo_t*  TypedefInfo_FactoryCopy(TypedefInfo_t* tinfo) const;
   virtual void   TypedefInfo_Init(TypedefInfo_t* tinfo, const char* funcname) const;
   virtual bool   TypedefInfo_IsValid(TypedefInfo_t* tinfo) const;
   virtual Long_t TypedefInfo_Property(TypedefInfo_t* tinfo) const;
   virtual int    TypedefInfo_Size(TypedefInfo_t* tinfo) const;
   virtual const char* TypedefInfo_TrueName(TypedefInfo_t* tinfo) const;
   virtual const char* TypedefInfo_Name(TypedefInfo_t* tinfo) const;
   virtual const char* TypedefInfo_Title(TypedefInfo_t* tinfo) const;

private: // Private Utility Functions

   TCintWithCling()
      : fMore(-1)
      , fExitCode(0)
      , fSharedLibsSerial(0)
      , fGlobalsListSerial(0)
      , fIncludePath()
      , fRootmapLoadPath()
      , fMapfile(0)
      , fRootmapFiles(0)
      , fLockProcessLine(kFALSE)
      , fInterpreter(0)
      , fMetaProcessor()
   {
      // For Dictionary() only.
   }

   TCintWithCling(const TCintWithCling&); // NOT IMPLEMENTED
   TCintWithCling& operator=(const TCintWithCling&); // NOT IMPLEMENTED

   void Execute(TMethod*, TObjArray*, int* /*error*/ = 0)
   {
   }

public: // ROOT Metadata

   ClassDef(TCintWithCling, 0) //Interface to CINT C/C++ interpreter
};

#endif
