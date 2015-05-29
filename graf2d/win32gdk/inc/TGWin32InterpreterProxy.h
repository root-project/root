// $Id: TGWin32InterpreterProxy.h,v 1.15 2007/03/08 15:52:17 rdm Exp $
// Author: Valeriy Onuchin  15/11/03

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TGWin32InterpreterProxy
#define ROOT_TGWin32InterpreterProxy

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGWin32InterpreterProxy                                              //
//                                                                      //
// This class defines thread-safe interface to a command line           //
// interpreter.                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TInterpreter
#include "TInterpreter.h"
#endif

#if ROOT_VERSION_CODE < ROOT_VERSION(6,00,00)

#ifndef ROOT_TGWin32ProxyBase
#include "TGWin32ProxyBase.h"
#endif


class TGWin32InterpreterProxy : public TInterpreter , public TGWin32ProxyBase {

protected:
   void Execute(TMethod *method, TObjArray *params, int *error = 0) {}

public:

   TGWin32InterpreterProxy() { fMaxResponseTime = 1000000; fIsVirtualX = kFALSE; }
   TGWin32InterpreterProxy(const char *name, const char *title = "Generic Interpreter") {}
   virtual ~TGWin32InterpreterProxy() {}

   void     AddIncludePath(const char *path);
   Int_t    AutoLoad(const char *classname);
   void     ClearFileBusy();
   void     ClearStack();
   Bool_t   Declare(const char* code);
   void     EnableAutoLoading();
   void     EndOfLineAction();
   void     Initialize();
   void     InspectMembers(TMemberInspector&, void* obj, const TClass* cl);
   Int_t    Load(const char *filenam, Bool_t system = kFALSE);
   void     LoadMacro(const char *filename, EErrorCode *error = 0);
   Int_t    LoadLibraryMap(const char *rootmapfile = 0);
   Int_t    RescanLibraryMap();
   Int_t    ReloadAllSharedLibraryMaps();
   Int_t    UnloadAllSharedLibraryMaps();
   Int_t    UnloadLibraryMap(const char *library);
   Long_t   ProcessLine(const char *line, EErrorCode *error = 0);
   Long_t   ProcessLineSynch(const char *line, EErrorCode *error = 0);
   void     PrintIntro();
   Int_t    SetClassSharedLibs(const char *cls, const char *libs);
   void     SetGetline(const char*(*getlineFunc)(const char* prompt),
                       void (*histaddFunc)(const char* line));
   void     Reset();
   void     ResetAll();
   void     ResetGlobals();
   void     ResetGlobalVar(void *obj);
   void     RewindDictionary();
   Int_t    DeleteGlobal(void *obj);
   Int_t    DeleteVariable(const char *name);
   void     SaveContext();
   void     SaveGlobalsContext();
   void     UpdateListOfGlobals();
   void     UpdateListOfGlobalFunctions();
   void     UpdateListOfTypes();
   void     SetClassInfo(TClass *cl, Bool_t reload = kFALSE);
   Bool_t   CheckClassInfo(const char *name, Bool_t autoload, Bool_t isClassOrNamespaceOnly = kFALSE);
   Bool_t   CheckClassTemplate(const char *name);
   Long_t   Calc(const char *line, EErrorCode* error = 0);
   void     CreateListOfBaseClasses(TClass *cl);
   void     CreateListOfDataMembers(TClass *cl);
   void     CreateListOfMethods(TClass *cl);
   void     UpdateListOfMethods(TClass *cl);
   void     CreateListOfMethodArgs(TFunction *m);
   TString  GetMangledName(TClass *cl, const char *method, const char *params);
   TString  GetMangledNameWithPrototype(TClass *cl, const char *method, const char *proto);
   Long_t   ExecuteMacro(const char *filename, EErrorCode *error = 0);
   Bool_t   IsErrorMessagesEnabled() const { return RealObject()->IsErrorMessagesEnabled(); }
   Bool_t   SetErrorMessages(Bool_t enable = kTRUE);
   Bool_t   IsProcessLineLocked() const { return RealObject()->IsProcessLineLocked(); }
   void     SetProcessLineLock(Bool_t lock = kTRUE);
   Int_t    GetExitCode() const { return RealObject()->GetExitCode(); }
   TClass  *GenerateTClass(const char *classname, Bool_t emulation, Bool_t silent = kFALSE);
   TClass  *GenerateTClass(ClassInfo_t *classinfo, Bool_t silent = kFALSE); 
   Int_t    GenerateDictionary(const char *classes, const char *includes = 0, const char *options = 0); 
   Int_t    GetMore() const {  return RealObject()->GetMore(); }
   Bool_t   IsLoaded(const char *filename) const {  return RealObject()->IsLoaded(filename); }
   char    *GetPrompt();
   void    *GetInterfaceMethod(TClass *cl, const char *method, const char *params);
   void    *GetInterfaceMethodWithPrototype(TClass *cl, const char *method, const char *proto);
   void     GetInterpreterTypeName(const char*,std::string &output,Bool_t=kFALSE);
   void     Execute(const char *function, const char *params, int *error = 0);
   void     Execute(TObject *obj, TClass *cl, const char *method, const char *params, int *error = 0);
   void     Execute(TObject *obj, TClass *cl, TMethod *method, TObjArray *params, int *error = 0);
   const char *GetSharedLibs();
   const char *GetClassSharedLibs(const char *cls);
   const char *GetSharedLibDeps(const char *lib);
   const char *GetIncludePath();
   TObjArray  *GetRootMapFiles() const { return RealObject()->GetRootMapFiles(); }
   const char *TypeName(const char *s);

   static TInterpreter *RealObject();
   static TInterpreter *ProxyObject();
};

#endif

#endif
