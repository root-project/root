// @(#)root/meta:$Name:  $:$Id: TInterpreter.h,v 1.26 2006/12/01 15:05:58 rdm Exp $
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

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TClass;
class TFunction;
class TMethod;
class TObjArray;


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
   virtual Int_t    GetMore() const = 0;
   virtual char    *GetPrompt() = 0;
   virtual const char *GetSharedLibs() = 0;
   virtual const char *GetClassSharedLibs(const char *cls) = 0;
   virtual const char *GetSharedLibDeps(const char *lib) = 0;
   virtual const char *GetIncludePath() = 0;
   virtual TObjArray  *GetRootMapFiles() const = 0;
   virtual Int_t    InitializeDictionaries() = 0;
   virtual Bool_t   IsLoaded(const char *filename) const = 0;
   virtual Int_t    Load(const char *filenam, Bool_t system = kFALSE) = 0;
   virtual void     LoadMacro(const char *filename, EErrorCode *error = 0) = 0;
   virtual Long_t   ProcessLine(const char *line, EErrorCode *error = 0) = 0;
   virtual Long_t   ProcessLineSynch(const char *line, EErrorCode *error = 0) = 0;
   virtual void     PrintIntro() = 0;
   virtual void     Reset() = 0;
   virtual void     ResetAll() = 0;
   virtual void     ResetGlobals() = 0;
   virtual void     RewindDictionary() = 0;
   virtual Int_t    DeleteGlobal(void *obj) = 0;
   virtual void     SaveContext() = 0;
   virtual void     SaveGlobalsContext() = 0;
   virtual void     UpdateListOfGlobals() = 0;
   virtual void     UpdateListOfGlobalFunctions() = 0;
   virtual void     UpdateListOfTypes() = 0;
   virtual void     SetClassInfo(TClass *cl, Bool_t reload = kFALSE) = 0;
   virtual Bool_t   CheckClassInfo(const char *name) = 0;
   virtual Long_t   Calc(const char *line, EErrorCode* error = 0) = 0;
   virtual void     CreateListOfBaseClasses(TClass *cl) = 0;
   virtual void     CreateListOfDataMembers(TClass *cl) = 0;
   virtual void     CreateListOfMethods(TClass *cl) = 0;
   virtual void     CreateListOfMethodArgs(TFunction *m) = 0;
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

   static TInterpreter  *&Instance();

   ClassDef(TInterpreter,0)  //ABC defining interface to generic interpreter
};

#ifndef __CINT__
#define gInterpreter (TInterpreter::Instance())
R__EXTERN TInterpreter* (*gPtr2Interpreter)();
#endif

#endif
