// @(#)root/meta:$Name:  $:$Id: TCint.h,v 1.6 2001/12/02 16:50:08 brun Exp $
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

// do not want to load fproto.h since it will cause problems in G__Rint
// (double declared functions)
#define G__FPROTO_H
#ifndef __CINT__
#include "common.h"
#else
struct G__dictposition;
#endif

#ifndef WIN32
# define  TWin32SendClass char
#endif

class G__ClassInfo;
class TMethod;
class TObjArray;

class TCint : public TInterpreter {

private:
   Int_t           fMore;           //1 if more input is required
   char            fPrompt[64];     //proposed prompt string
   G__dictposition fDictPos;        //CINT dictionary context after init
   G__dictposition fDictPosGlobals; //CINT dictionary context after ResetGlobals()
   TString         fSharedLibs;     //Hold a list of lib loaded by G__loadfile
   TString         fIncludePath;    //Hold a list of lib include path

   TCint() : fMore(-1) { }  //for Dictionary() only
   virtual void Execute(TMethod *, TObjArray *, int *error =0) { }

protected:
   virtual void ExecThreadCB(TWin32SendClass *command);

public:
   TCint(const char *name, const char *title);
   virtual ~TCint();

   void    AddIncludePath(const char *path);
   void    ClearFileBusy();
   void    ClearStack(); // Delete existing temporary values
   void    EndOfLineAction();
   Int_t   GetMore() const { return fMore; }
   char   *GetPrompt() { return fPrompt; }
   const char *GetSharedLibs();
   const char *GetIncludePath();
   Int_t   InitializeDictionaries();
   Bool_t  IsLoaded(const char *filename) const;
   void    LoadMacro(const char *filename, EErrorCode *error =0);
   Int_t   ProcessLine(const char *line, EErrorCode *error =0);
   Int_t   ProcessLineAsynch(const char *line, EErrorCode *error =0);
   Int_t   ProcessLineSynch(const char *line, EErrorCode *error =0);
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
   void   *GetInterfaceMethod(TClass *cl, char *method, char *params);
   void   *GetInterfaceMethodWithPrototype(TClass *cl, char *method, char *proto);
   void    Execute(const char *function, const char *params, int *error =0);
   void    Execute(TObject *obj, TClass *cl, const char *method, const char *params, int *error =0);
   void    Execute(TObject *obj, TClass *cl, TMethod *method, TObjArray *params, int *error =0);
   Int_t   ExecuteMacro(const char *filename, EErrorCode *error);
   Bool_t  IsErrorMessagesEnabled();
   Bool_t  SetErrorMessages(Bool_t enable = kTRUE);
   const char *TypeName(const char *typeDesc);

   static void *FindSpecialObject(char *name, G__ClassInfo *type, void **prevObj, void **assocPtr);
   static void  UpdateClassInfo(char *name, Long_t tagnum);
   static void  UpdateAllCanvases();

   ClassDef(TCint,0)  //Interface to CINT C/C++ interpreter
};

#endif

