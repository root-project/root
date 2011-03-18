// @(#)root/meta:$Id$
// Author: Fons Rademakers   01/03/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TInterpreter
#define ROOT_TInterpreter

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TInterpreter                                                         //
//                                                                      //
// This class defines an abstract interface to a generic command line   //
// interpreter.                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TDictionary
#include "TDictionary.h"
#endif

class TClass;
class TEnv;
class TFunction;
class TMethod;
class TObjArray;
class TVirtualMutex;
extern "C" { struct G__value; }

R__EXTERN TVirtualMutex *gCINTMutex;

class TInterpreter : public TNamed {

protected:
   virtual void Execute(TMethod *method, TObjArray *params, int *error = 0) = 0;

public:
   enum EErrorCode {
      kNoError     = 0,
      kRecoverable = 1,
      kDangerous   = 2,
      kFatal       = 3,
      kProcessing  = 99
   };

   TInterpreter() { }   // for Dictionary
   TInterpreter(const char *name, const char *title = "Generic Interpreter");
   virtual ~TInterpreter() { }

   virtual void     AddIncludePath(const char *path) = 0;
   virtual Int_t    AutoLoad(const char *classname) = 0;
   virtual void     ClearFileBusy() = 0;
   virtual void     ClearStack() = 0; // Delete existing temporary values
   virtual void     EnableAutoLoading() = 0;
   virtual void     EndOfLineAction() = 0;
   virtual Int_t    GetExitCode() const = 0;
   virtual TEnv    *GetMapfile() const { return 0; }
   virtual Int_t    GetMore() const = 0;
   virtual Int_t    GenerateDictionary(const char *classes, const char *includes = 0, const char *options = 0) = 0; 
   virtual char    *GetPrompt() = 0;
   virtual const char *GetSharedLibs() = 0;
   virtual const char *GetClassSharedLibs(const char *cls) = 0;
   virtual const char *GetSharedLibDeps(const char *lib) = 0;
   virtual const char *GetIncludePath() = 0;
   virtual const char *GetSTLIncludePath() const { return ""; }
   virtual TObjArray  *GetRootMapFiles() const = 0;
   virtual Int_t    InitializeDictionaries() = 0;
   virtual Bool_t   IsLoaded(const char *filename) const = 0;
   virtual Int_t    Load(const char *filenam, Bool_t system = kFALSE) = 0;
   virtual void     LoadMacro(const char *filename, EErrorCode *error = 0) = 0;
   virtual Int_t    LoadLibraryMap(const char *rootmapfile = 0) = 0;
   virtual Int_t    RescanLibraryMap() = 0;
   virtual Int_t    ReloadAllSharedLibraryMaps() = 0;
   virtual Int_t    UnloadAllSharedLibraryMaps() = 0;
   virtual Int_t    UnloadLibraryMap(const char *library) = 0;
   virtual Long_t   ProcessLine(const char *line, EErrorCode *error = 0) = 0;
   virtual Long_t   ProcessLineSynch(const char *line, EErrorCode *error = 0) = 0;
   virtual void     PrintIntro() = 0;
   virtual void     SetGetline(char*(*getlineFunc)(const char* prompt),
                               void (*histaddFunc)(char* line)) = 0;
   virtual void     Reset() = 0;
   virtual void     ResetAll() = 0;
   virtual void     ResetGlobals() = 0;
   virtual void     ResetGlobalVar(void *obj) = 0;
   virtual void     RewindDictionary() = 0;
   virtual Int_t    DeleteGlobal(void *obj) = 0;
   virtual void     SaveContext() = 0;
   virtual void     SaveGlobalsContext() = 0;
   virtual void     UpdateListOfGlobals() = 0;
   virtual void     UpdateListOfGlobalFunctions() = 0;
   virtual void     UpdateListOfTypes() = 0;
   virtual void     SetClassInfo(TClass *cl, Bool_t reload = kFALSE) = 0;
   virtual Bool_t   CheckClassInfo(const char *name, Bool_t autoload = kTRUE) = 0;
   virtual Long_t   Calc(const char *line, EErrorCode* error = 0) = 0;
   virtual void     CreateListOfBaseClasses(TClass *cl) = 0;
   virtual void     CreateListOfDataMembers(TClass *cl) = 0;
   virtual void     CreateListOfMethods(TClass *cl) = 0;
   virtual void     CreateListOfMethodArgs(TFunction *m) = 0;
   virtual void     UpdateListOfMethods(TClass *cl) = 0;
   virtual TString  GetMangledName(TClass *cl, const char *method, const char *params) = 0;
   virtual TString  GetMangledNameWithPrototype(TClass *cl, const char *method, const char *proto) = 0;
   virtual const char *GetInterpreterTypeName(const char *name,Bool_t full = kFALSE) = 0;
   virtual void    *GetInterfaceMethod(TClass *cl, const char *method, const char *params) = 0;
   virtual void    *GetInterfaceMethodWithPrototype(TClass *cl, const char *method, const char *proto) = 0;
   virtual void     Execute(const char *function, const char *params, int *error = 0) = 0;
   virtual void     Execute(TObject *obj, TClass *cl, const char *method, const char *params, int *error = 0) = 0;
   virtual void     Execute(TObject *obj, TClass *cl, TMethod *method, TObjArray *params, int *error = 0) = 0;
   virtual Long_t   ExecuteMacro(const char *filename, EErrorCode *error = 0) = 0;
   virtual Bool_t   IsErrorMessagesEnabled() const = 0;
   virtual Bool_t   SetErrorMessages(Bool_t enable = kTRUE) = 0;
   virtual Bool_t   IsProcessLineLocked() const = 0;
   virtual void     SetProcessLineLock(Bool_t lock = kTRUE) = 0;
   virtual const char *TypeName(const char *s) = 0;

   // All the functions below must be virtual with a dummy implementation
   // These functions are redefined in TCint.
   //The dummy implementation avoids an implementation in TGWin32InterpreterProxy

   // Misc
   virtual int    DisplayClass(FILE * /* fout */,char * /* name */,int /* base */,int /* start */) const {return 0;}
   virtual int    DisplayIncludePath(FILE * /* fout */) const {return 0;}
   virtual void  *FindSym(const char * /* entry */) const {return 0;}
   virtual void   GenericError(const char * /* error */) const {;}
   virtual Long_t GetExecByteCode() const {return 0;}
   virtual Long_t Getgvp() const {return 0;}
   virtual const char *Getp2f2funcname(void * /* receiver */) const {return 0;}
   virtual const char *GetTopLevelMacroName() const {return 0;};
   virtual const char *GetCurrentMacroName()  const {return 0;};
   virtual int    GetSecurityError() const{return 0;}
   virtual int    LoadFile(const char * /* path */) const {return 0;}
   virtual void   LoadText(const char * /* text */) const {;}
   virtual const char *MapCppName(const char*) const {return 0;}
   virtual void   SetAlloclockfunc(void (*)()) const {;}  
   virtual void   SetAllocunlockfunc(void (*)()) const {;}  
   virtual int    SetClassAutoloading(int) const {return 0;}
   virtual void   SetErrmsgcallback(void * /* p */) const {;}
   virtual void   Setgvp(Long_t) const {;}
   virtual void   SetRTLD_NOW() const {;}
   virtual void   SetRTLD_LAZY() const {;}
   virtual void   SetTempLevel(int /* val */) const {;}
   virtual int    UnloadFile(const char * /* path */) const {return 0;}
   
   
   // G__CallFunc interface
   virtual void   CallFunc_Delete(void * /* func */) const {;}
   virtual void   CallFunc_Exec(CallFunc_t * /* func */, void * /* address */) const {;}
   virtual Long_t    CallFunc_ExecInt(CallFunc_t * /* func */, void * /* address */) const {return 0;}
   virtual Long_t    CallFunc_ExecInt64(CallFunc_t * /* func */, void * /* address */) const {return 0;}
   virtual Double_t  CallFunc_ExecDouble(CallFunc_t * /* func */, void * /* address */) const {return 0;}
   virtual CallFunc_t   *CallFunc_Factory() const {return 0;}
   virtual CallFunc_t   *CallFunc_FactoryCopy(CallFunc_t * /* func */) const {return 0;}
   virtual MethodInfo_t *CallFunc_FactoryMethod(CallFunc_t * /* func */) const {return 0;}
   virtual void   CallFunc_Init(CallFunc_t * /* func */) const {;}
   virtual bool   CallFunc_IsValid(CallFunc_t * /* func */) const {return 0;}
   virtual void   CallFunc_ResetArg(CallFunc_t * /* func */) const {;}
   virtual void   CallFunc_SetArg(CallFunc_t * /*func */, Long_t /* param */) const {;}
   virtual void   CallFunc_SetArg(CallFunc_t * /* func */, Double_t /* param */) const {;}
   virtual void   CallFunc_SetArg(CallFunc_t * /* func */, Long64_t /* param */) const {;}
   virtual void   CallFunc_SetArg(CallFunc_t * /* func */, ULong64_t /* param */) const {;}
   virtual void   CallFunc_SetArgArray(CallFunc_t * /* func */, Long_t * /* paramArr */, Int_t /* nparam */) const {;}
   virtual void   CallFunc_SetArgs(CallFunc_t * /* func */, const char * /* param */) const {;}
   virtual void   CallFunc_SetFunc(CallFunc_t * /* func */, ClassInfo_t * /* info */, const char * /* method */, const char * /* params */, Long_t * /* Offset */) const {;}
   virtual void   CallFunc_SetFunc(CallFunc_t * /* func */, MethodInfo_t * /* info */) const {;}
   virtual void   CallFunc_SetFuncProto(CallFunc_t * /* func */, ClassInfo_t * /* info */, const char * /* method */, const char * /* proto */, Long_t * /* Offset */) const {;}

               
   // G__ClassInfo interface            
   virtual Long_t ClassInfo_ClassProperty(ClassInfo_t * /* info */) const {return 0;}
   virtual void   ClassInfo_Delete(ClassInfo_t * /* info */) const {;}
   virtual void   ClassInfo_Delete(ClassInfo_t * /* info */, void * /* arena */) const {;}
   virtual void   ClassInfo_DeleteArray(ClassInfo_t * /* info */, void * /* arena */, bool /* dtorOnly */) const {;}
   virtual void   ClassInfo_Destruct(ClassInfo_t * /* info */, void * /* arena */) const {;}
   virtual ClassInfo_t  *ClassInfo_Factory() const {return 0;}
   virtual ClassInfo_t  *ClassInfo_Factory(G__value * /* value */) const {return 0;}
   virtual ClassInfo_t  *ClassInfo_Factory(ClassInfo_t * /* cl */) const {return 0;}
   virtual ClassInfo_t  *ClassInfo_Factory(const char * /* name */) const {return 0;}
   virtual int    ClassInfo_GetMethodNArg(ClassInfo_t * /* info */, const char * /* method */,const char * /* proto */) const {return 0;}
   virtual bool   ClassInfo_HasDefaultConstructor(ClassInfo_t * /* info */) const {return 0;}             
   virtual bool   ClassInfo_HasMethod(ClassInfo_t * /* info */, const char * /* name */) const {return 0;}             
   virtual void   ClassInfo_Init(ClassInfo_t * /* info */, const char * /* funcname */) const {;}
   virtual void   ClassInfo_Init(ClassInfo_t * /* info */, int /* tagnum */) const {;}
   virtual bool   ClassInfo_IsBase(ClassInfo_t * /* info */, const char * /* name */) const {return 0;}
   virtual bool   ClassInfo_IsEnum(const char * /* name */) const {return 0;}
   virtual bool   ClassInfo_IsLoaded(ClassInfo_t * /* info */) const {return 0;}             
   virtual bool   ClassInfo_IsValid(ClassInfo_t * /* info */) const {return 0;}             
   virtual bool   ClassInfo_IsValidMethod(ClassInfo_t * /* info */, const char * /* method */,const char * /* proto */, Long_t * /* offset */) const {return 0;}             
   virtual int    ClassInfo_Next(ClassInfo_t * /* info */) const {return 0;}
   virtual void  *ClassInfo_New(ClassInfo_t * /* info */) const {return 0;}
   virtual void  *ClassInfo_New(ClassInfo_t * /* info */, int /* n */) const {return 0;}
   virtual void  *ClassInfo_New(ClassInfo_t * /* info */, int /* n */, void * /* arena */) const {return 0;}
   virtual void  *ClassInfo_New(ClassInfo_t * /* info */, void * /* arena */) const {return 0;}
   virtual Long_t ClassInfo_Property(ClassInfo_t * /* info */) const {return 0;}
   virtual int    ClassInfo_RootFlag(ClassInfo_t * /* info */) const {return 0;}
   virtual int    ClassInfo_Size(ClassInfo_t * /* info */) const {return 0;}
   virtual Long_t ClassInfo_Tagnum(ClassInfo_t * /* info */) const {return 0;}
   virtual const char *ClassInfo_FileName(ClassInfo_t * /* info */) const {return 0;}
   virtual const char *ClassInfo_FullName(ClassInfo_t * /* info */) const {return 0;}
   virtual const char *ClassInfo_Name(ClassInfo_t * /* info */) const {return 0;}
   virtual const char *ClassInfo_Title(ClassInfo_t * /* info */) const {return 0;}
   virtual const char *ClassInfo_TmpltName(ClassInfo_t * /* info */) const {return 0;}
   
                  
   // G__BaseClassInfo interface            
   virtual void   BaseClassInfo_Delete(BaseClassInfo_t * /* bcinfo */) const {;}
   virtual BaseClassInfo_t  *BaseClassInfo_Factory(ClassInfo_t * /* info */) const {return 0;}
   virtual int    BaseClassInfo_Next(BaseClassInfo_t * /* bcinfo */) const {return 0;}
   virtual int    BaseClassInfo_Next(BaseClassInfo_t * /* bcinfo */, int  /* onlyDirect */) const {return 0;}
   virtual Long_t BaseClassInfo_Offset(BaseClassInfo_t * /* bcinfo */) const {return 0;}
   virtual Long_t BaseClassInfo_Property(BaseClassInfo_t * /* bcinfo */) const {return 0;}
   virtual Long_t BaseClassInfo_Tagnum(BaseClassInfo_t * /* bcinfo */) const {return 0;}
   virtual const char *BaseClassInfo_FullName(BaseClassInfo_t * /* bcinfo */) const {return 0;}
   virtual const char *BaseClassInfo_Name(BaseClassInfo_t * /* bcinfo */) const {return 0;}
   virtual const char *BaseClassInfo_TmpltName(BaseClassInfo_t * /* bcinfo */) const {return 0;}
               
   // G__DataMemberInfo interface            
   virtual int    DataMemberInfo_ArrayDim(DataMemberInfo_t * /* dminfo */) const {return 0;}
   virtual void   DataMemberInfo_Delete(DataMemberInfo_t * /* dminfo */) const {;}
   virtual DataMemberInfo_t  *DataMemberInfo_Factory(ClassInfo_t * /* clinfo */ = 0) const {return 0;}
   virtual DataMemberInfo_t  *DataMemberInfo_FactoryCopy(DataMemberInfo_t * /* dminfo */) const {return 0;}
   virtual bool   DataMemberInfo_IsValid(DataMemberInfo_t * /* dminfo */) const {return 0;}
   virtual int    DataMemberInfo_MaxIndex(DataMemberInfo_t * /* dminfo */, Int_t  /* dim */) const {return 0;}
   virtual int    DataMemberInfo_Next(DataMemberInfo_t * /* dminfo */) const {return 0;}
   virtual Long_t DataMemberInfo_Offset(DataMemberInfo_t * /* dminfo */) const {return 0;}
   virtual Long_t DataMemberInfo_Property(DataMemberInfo_t * /* dminfo */) const {return 0;}
   virtual Long_t DataMemberInfo_TypeProperty(DataMemberInfo_t * /* dminfo */) const {return 0;}
   virtual int    DataMemberInfo_TypeSize(DataMemberInfo_t * /* dminfo */) const {return 0;}
   virtual const char *DataMemberInfo_TypeName(DataMemberInfo_t * /* dminfo */) const {return 0;}
   virtual const char *DataMemberInfo_TypeTrueName(DataMemberInfo_t * /* dminfo */) const {return 0;}
   virtual const char *DataMemberInfo_Name(DataMemberInfo_t * /* dminfo */) const {return 0;}
   virtual const char *DataMemberInfo_Title(DataMemberInfo_t * /* dminfo */) const {return 0;}
   virtual const char *DataMemberInfo_ValidArrayIndex(DataMemberInfo_t * /* dminfo */) const {return 0;}
               
   // G__MethodInfo interface            
   virtual void   MethodInfo_CreateSignature(MethodInfo_t * /* minfo */, TString & /* signature */) const {;}
   virtual void   MethodInfo_Delete(MethodInfo_t * /* minfo */) const {;}
   virtual MethodInfo_t  *MethodInfo_Factory() const {return 0;}
   virtual MethodInfo_t  *MethodInfo_FactoryCopy(MethodInfo_t * /* minfo */) const {return 0;}
   virtual MethodInfo_t  *MethodInfo_InterfaceMethod(MethodInfo_t * /* minfo */) const {return 0;}
   virtual bool   MethodInfo_IsValid(MethodInfo_t * /* minfo */) const {return 0;}
   virtual int    MethodInfo_NArg(MethodInfo_t * /* minfo */) const {return 0;}
   virtual int    MethodInfo_NDefaultArg(MethodInfo_t * /* minfo */) const {return 0;}
   virtual int    MethodInfo_Next(MethodInfo_t * /* minfo */) const {return 0;}
   virtual Long_t MethodInfo_Property(MethodInfo_t * /* minfo */) const {return 0;}
   virtual TypeInfo_t  *MethodInfo_Type(MethodInfo_t * /* minfo */) const {return 0;}
   virtual const char *MethodInfo_GetMangledName(MethodInfo_t * /* minfo */) const {return 0;}
   virtual const char *MethodInfo_GetPrototype(MethodInfo_t * /* minfo */) const {return 0;}
   virtual const char *MethodInfo_Name(MethodInfo_t * /* minfo */) const {return 0;}
   virtual const char *MethodInfo_TypeName(MethodInfo_t * /* minfo */) const {return 0;}
   virtual const char *MethodInfo_Title(MethodInfo_t * /* minfo */) const {return 0;}
               
   // G__MethodArgInfo interface            
   virtual void   MethodArgInfo_Delete(MethodArgInfo_t * /* marginfo */) const {;}
   virtual MethodArgInfo_t  *MethodArgInfo_Factory() const {return 0;}
   virtual MethodArgInfo_t  *MethodArgInfo_FactoryCopy(MethodArgInfo_t * /* marginfo */) const {return 0;}
   virtual bool   MethodArgInfo_IsValid(MethodArgInfo_t * /* marginfo */) const {return 0;}
   virtual int    MethodArgInfo_Next(MethodArgInfo_t * /* marginfo */) const {return 0;}
   virtual Long_t MethodArgInfo_Property(MethodArgInfo_t * /* marginfo */) const {return 0;}
   virtual const char *MethodArgInfo_DefaultValue(MethodArgInfo_t * /* marginfo */) const {return 0;}
   virtual const char *MethodArgInfo_Name(MethodArgInfo_t * /* marginfo */) const {return 0;}
   virtual const char *MethodArgInfo_TypeName(MethodArgInfo_t * /* marginfo */) const {return 0;}

                  
   // G__TypeInfo interface            
   virtual void    TypeInfo_Delete(TypeInfo_t * /* tinfo */) const {;}
   virtual TypeInfo_t *TypeInfo_Factory() const {return 0;}
   virtual TypeInfo_t *TypeInfo_Factory(G__value * /* value */) const {return 0;}
   virtual TypeInfo_t *TypeInfo_FactoryCopy(TypeInfo_t * /* tinfo */) const {return 0;}
   virtual void   TypeInfo_Init(TypeInfo_t * /* tinfo */, const char * /* funcname */) const {;}
   virtual bool   TypeInfo_IsValid(TypeInfo_t * /* tinfo */) const {return 0;}
   virtual const char *TypeInfo_Name(TypeInfo_t * /* info */) const {return 0;}
   virtual Long_t TypeInfo_Property(TypeInfo_t * /* tinfo */) const {return 0;}
   virtual int    TypeInfo_RefType(TypeInfo_t * /* tinfo */) const {return 0;}
   virtual int    TypeInfo_Size(TypeInfo_t * /* tinfo */) const {return 0;}
   virtual const char *TypeInfo_TrueName(TypeInfo_t * /* tinfo */) const {return 0;}
   
                  
   // G__TypedefInfo interface            
   virtual void   TypedefInfo_Delete(TypedefInfo_t * /* tinfo */) const {;}
   virtual TypedefInfo_t  *TypedefInfo_Factory() const {return 0;}
   virtual TypedefInfo_t  *TypedefInfo_FactoryCopy(TypedefInfo_t * /* tinfo */) const {return 0;}
   virtual void   TypedefInfo_Init(TypedefInfo_t * /* tinfo */, const char * /* funcname */) const {;}
   virtual bool   TypedefInfo_IsValid(TypedefInfo_t * /* tinfo */) const {return 0;}
   virtual Long_t TypedefInfo_Property(TypedefInfo_t * /* tinfo */) const {return 0;}
   virtual int    TypedefInfo_Size(TypedefInfo_t * /* tinfo */) const {return 0;}
   virtual const char *TypedefInfo_TrueName(TypedefInfo_t * /* tinfo */) const {return 0;}
   virtual const char *TypedefInfo_Name(TypedefInfo_t * /* tinfo */) const {return 0;}
   virtual const char *TypedefInfo_Title(TypedefInfo_t * /* tinfo */) const {return 0;}

   static TInterpreter  *&Instance();

   ClassDef(TInterpreter,0)  //ABC defining interface to generic interpreter
};

#ifndef __CINT__
#define gInterpreter (TInterpreter::Instance())
R__EXTERN TInterpreter* (*gPtr2Interpreter)();
R__EXTERN TInterpreter* gCint;
#endif

#endif
