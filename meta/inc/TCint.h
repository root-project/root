// @(#)root/meta:$Name:  $:$Id: TCint.h,v 1.27 2006/05/23 04:47:40 brun Exp $
// Author: Fons Rademakers   01/03/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TCint
#define ROOT_TCint

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCint                                                                //
//                                                                      //
// This class defines an interface to the CINT C/C++ interpreter made   //
// by Masaharu Goto of HP Japan.                                        //
//                                                                      //
// CINT is an almost full ANSI compliant C/C++ interpreter.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TInterpreter
#include "TInterpreter.h"
#endif

#ifndef __CINT__
#include "G__ci.h"
#else
struct G__dictposition;
#endif

#ifndef WIN32
# define  TWin32SendClass char
#endif

namespace Cint {
class G__ClassInfo;
}
using namespace Cint;
class TMethod;
class TObjArray;
class TEnv;
class TVirtualMutex;

R__EXTERN TVirtualMutex *gCINTMutex; 

class TCint : public TInterpreter {

private:
   Int_t           fMore;           //1 if more input is required
   Int_t           fExitCode;       //value passed to exit() in interpreter
   char            fPrompt[64];     //proposed prompt string
   G__dictposition fDictPos;        //CINT dictionary context after init
   G__dictposition fDictPosGlobals; //CINT dictionary context after ResetGlobals()
   TString         fSharedLibs;     //hold a list of lib loaded by G__loadfile
   TString         fIncludePath;    //hold a list of lib include path
   TEnv           *fMapfile;        //map of classes and libraries

   TCint() : fMore(-1), fExitCode(0), fDictPos(), fDictPosGlobals(), 
     fSharedLibs(), fIncludePath(), fMapfile(0) { }  //for Dictionary() only
   virtual void Execute(TMethod *, TObjArray *, int * /*error*/ = 0) { }

protected:
   TCint(const TCint&);
   TCint& operator=(const TCint&);

   virtual void ExecThreadCB(TWin32SendClass *command);
   virtual Int_t LoadLibraryMap();

public:
   TCint(const char *name, const char *title);
   virtual ~TCint();

   void    AddIncludePath(const char *path);
   Int_t   AutoLoad(const char *classname);
   void    ClearFileBusy();
   void    ClearStack(); // Delete existing temporary values
   void    EnableAutoLoading();
   void    EndOfLineAction();
   Int_t   GetExitCode() const { return fExitCode; }
   Int_t   GetMore() const { return fMore; }
   char   *GetPrompt() { return fPrompt; }
   const char *GetSharedLibs();
   const char *GetClassSharedLibs(const char *cls);
   const char *GetSharedLibDeps(const char *lib);
   const char *GetIncludePath();
   Int_t   InitializeDictionaries();
   Bool_t  IsLoaded(const char *filename) const;
   Int_t   Load(const char *filenam, Bool_t system = kFALSE);
   void    LoadMacro(const char *filename, EErrorCode *error = 0);
   Long_t  ProcessLine(const char *line, EErrorCode *error = 0);
   Long_t  ProcessLineAsynch(const char *line, EErrorCode *error = 0);
   Long_t  ProcessLineSynch(const char *line, EErrorCode *error = 0);
   void    PrintIntro();
   void    Reset();
   void    ResetAll();
   void    ResetGlobals();
   void    RewindDictionary();
   Int_t   DeleteGlobal(void *obj);
   void    SaveContext();
   void    SaveGlobalsContext();
   void    UpdateListOfGlobals();
   void    UpdateListOfGlobalFunctions();
   void    UpdateListOfTypes();
   void    SetClassInfo(TClass *cl, Bool_t reload = kFALSE);
   Bool_t  CheckClassInfo(const char *name);
   Long_t  Calc(const char *line, EErrorCode *error = 0);
   void    CreateListOfBaseClasses(TClass *cl);
   void    CreateListOfDataMembers(TClass *cl);
   void    CreateListOfMethods(TClass *cl);
   void    CreateListOfMethodArgs(TFunction *m);
   TString GetMangledName(TClass *cl, const char *method, const char *params);
   TString GetMangledNameWithPrototype(TClass *cl, const char *method, const char *proto);
   void   *GetInterfaceMethod(TClass *cl, const char *method, const char *params);
   void   *GetInterfaceMethodWithPrototype(TClass *cl, const char *method, const char *proto);
   const char *GetInterpreterTypeName(const char*name, Bool_t full = kFALSE);
   void    Execute(const char *function, const char *params, int *error = 0);
   void    Execute(TObject *obj, TClass *cl, const char *method, const char *params, int *error = 0);
   void    Execute(TObject *obj, TClass *cl, TMethod *method, TObjArray *params, int *error = 0);
   Long_t  ExecuteMacro(const char *filename, EErrorCode *error = 0);
   void    RecursiveRemove(TObject *obj);
   Bool_t  IsErrorMessagesEnabled();
   Bool_t  SetErrorMessages(Bool_t enable = kTRUE);
   const char *TypeName(const char *typeDesc);

   static void *FindSpecialObject(const char *name, G__ClassInfo *type, void **prevObj, void **assocPtr);
   static int   AutoLoadCallback(const char *cls, const char *lib);
   static void  UpdateClassInfo(char *name, Long_t tagnum);
   static void  UpdateAllCanvases();

   ClassDef(TCint,0)  //Interface to CINT C/C++ interpreter
};

#endif

