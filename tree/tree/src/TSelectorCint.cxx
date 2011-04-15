// @(#)root/tree:$Id$
// Author: Rene Brun   05/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// This class is a special version of TSelector for user interpreted    //
// classes.                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TTree.h"
#include "THashList.h"
#include "TSelectorCint.h"
#include "TError.h"

ClassImp(TSelectorCint)

//______________________________________________________________________________
TSelectorCint::TSelectorCint() : TSelector(),
   fClass(0),
   fFuncVersion  (0),
   fFuncInit     (0),
   fFuncBegin    (0),
   fFuncSlBegin  (0),
   fFuncNotif    (0),
   fFuncSlTerm   (0),
   fFuncTerm     (0),
   fFuncCut      (0),
   fFuncFill     (0),
   fFuncProc     (0),
   fFuncOption   (0),
   fFuncObj      (0),
   fFuncInp      (0),
   fFuncOut      (0),
   fFuncAbort    (0),
   fFuncGetAbort (0),
   fFuncResetAbort (0),
   fFuncGetStat  (0),
   fIntSelector(0),fIsOwner(kFALSE)


{
   // Default constructor for a Selector.

}

//______________________________________________________________________________
TSelectorCint::~TSelectorCint()
{
   // destructor for a Selector.

   gCint->CallFunc_Delete(fFuncVersion);
   gCint->CallFunc_Delete(fFuncInit);
   gCint->CallFunc_Delete(fFuncBegin);
   gCint->CallFunc_Delete(fFuncSlBegin);
   gCint->CallFunc_Delete(fFuncNotif);
   gCint->CallFunc_Delete(fFuncSlTerm);
   gCint->CallFunc_Delete(fFuncTerm);
   gCint->CallFunc_Delete(fFuncCut);
   gCint->CallFunc_Delete(fFuncFill);
   gCint->CallFunc_Delete(fFuncProc);
   gCint->CallFunc_Delete(fFuncOption);
   gCint->CallFunc_Delete(fFuncObj);
   gCint->CallFunc_Delete(fFuncInp);
   gCint->CallFunc_Delete(fFuncOut);
   gCint->CallFunc_Delete(fFuncAbort);
   gCint->CallFunc_Delete(fFuncGetAbort);
   gCint->CallFunc_Delete(fFuncResetAbort);
   gCint->CallFunc_Delete(fFuncGetStat);

   if (fIsOwner && fIntSelector) gCint->ClassInfo_Delete(fClass,fIntSelector);
   gCint->ClassInfo_Delete(fClass);
}

//______________________________________________________________________________
void TSelectorCint::SetFuncProto(CallFunc_t *cf, ClassInfo_t *cl,
                                 const char* fname, const char* argtype,
                                 Bool_t required)
{
   // Set the function prototype.

   Long_t offset = 0;

   gCint->CallFunc_SetFuncProto(cf, cl,fname,argtype,&offset);

   if (gDebug > 2)
      Info("SetFuncProto","set %s(%s) offset = %ld",fname,argtype,offset);

   if (!gCint->CallFunc_IsValid(cf) && required)
      Error("SetFuncProto","cannot set %s(%s)",fname,argtype);
}

//______________________________________________________________________________
void TSelectorCint::Build(TSelector *iselector, ClassInfo_t *cl, Bool_t isowner)
{
   // Initialize the CallFunc objects when selector is interpreted.

   gCint->CallFunc_Delete(fFuncVersion);
   gCint->CallFunc_Delete(fFuncInit);
   gCint->CallFunc_Delete(fFuncBegin);
   gCint->CallFunc_Delete(fFuncSlBegin);
   gCint->CallFunc_Delete(fFuncNotif);
   gCint->CallFunc_Delete(fFuncSlTerm);
   gCint->CallFunc_Delete(fFuncTerm);
   gCint->CallFunc_Delete(fFuncCut);
   gCint->CallFunc_Delete(fFuncFill);
   gCint->CallFunc_Delete(fFuncProc);
   gCint->CallFunc_Delete(fFuncOption);
   gCint->CallFunc_Delete(fFuncObj);
   gCint->CallFunc_Delete(fFuncInp);
   gCint->CallFunc_Delete(fFuncOut);
   gCint->CallFunc_Delete(fFuncAbort);
   gCint->CallFunc_Delete(fFuncGetAbort);
   gCint->CallFunc_Delete(fFuncResetAbort);
   gCint->CallFunc_Delete(fFuncGetStat);

   if (fIsOwner && fIntSelector) gCint->ClassInfo_Delete(fClass, fIntSelector);
   gCint->ClassInfo_Delete(fClass);

   R__ASSERT(cl);

   // The CINT MethodInfo created by SetFuncProto will remember the address
   // of cl, so we need to keep it around.
   fClass        = gCint->ClassInfo_Factory(cl);

   fIntSelector  = iselector;
   fIsOwner      = isowner;
   fFuncVersion  = gCint->CallFunc_Factory();
   fFuncInit     = gCint->CallFunc_Factory();
   fFuncBegin    = gCint->CallFunc_Factory();
   fFuncSlBegin  = gCint->CallFunc_Factory();
   fFuncNotif    = gCint->CallFunc_Factory();
   fFuncSlTerm   = gCint->CallFunc_Factory();
   fFuncTerm     = gCint->CallFunc_Factory();
   fFuncCut      = gCint->CallFunc_Factory();
   fFuncFill     = gCint->CallFunc_Factory();
   fFuncProc     = gCint->CallFunc_Factory();
   fFuncOption   = gCint->CallFunc_Factory();
   fFuncObj      = gCint->CallFunc_Factory();
   fFuncInp      = gCint->CallFunc_Factory();
   fFuncOut      = gCint->CallFunc_Factory();
   fFuncAbort    = gCint->CallFunc_Factory();
   fFuncGetAbort = gCint->CallFunc_Factory();
   fFuncResetAbort = gCint->CallFunc_Factory();
   fFuncGetStat  = gCint->CallFunc_Factory();

   SetFuncProto(fFuncVersion,fClass,"Version","",kFALSE);
   SetFuncProto(fFuncInit,fClass,"Init","TTree*");
   SetFuncProto(fFuncBegin,fClass,"Begin","TTree*");
   SetFuncProto(fFuncSlBegin,fClass,"SlaveBegin","TTree*",kFALSE);
   SetFuncProto(fFuncNotif,fClass,"Notify","");
   SetFuncProto(fFuncSlTerm,fClass,"SlaveTerminate","",kFALSE);
   SetFuncProto(fFuncTerm,fClass,"Terminate","");
   SetFuncProto(fFuncCut,fClass,"ProcessCut","Long64_t",kFALSE);
   SetFuncProto(fFuncFill,fClass,"ProcessFill","Long64_t",kFALSE);
   SetFuncProto(fFuncProc,fClass,"Process","Long64_t",kFALSE);
   SetFuncProto(fFuncOption,fClass,"SetOption","const char*");
   SetFuncProto(fFuncObj,fClass,"SetObject","TObject*");
   SetFuncProto(fFuncInp,fClass,"SetInputList","TList*");
   SetFuncProto(fFuncOut,fClass,"GetOutputList","");
   SetFuncProto(fFuncAbort,fClass,"Abort","const char *,TSelector::EAbort",kFALSE);
   SetFuncProto(fFuncGetAbort,fClass,"GetAbort","",kFALSE);
   SetFuncProto(fFuncResetAbort,fClass,"ResetAbort","",kFALSE);
   SetFuncProto(fFuncGetStat,fClass,"GetStatus","");
}

//______________________________________________________________________________
int TSelectorCint::Version() const
{
   // Invoke the Version function via the interpreter.

   if (gDebug > 2)
      Info("Version","Call Version");

   if (gCint->CallFunc_IsValid(fFuncVersion)) {
      gCint->CallFunc_ResetArg(fFuncVersion);
      return gCint->CallFunc_ExecInt(fFuncVersion, fIntSelector);
   } else {
      return 0; // emulate for old version
   }
}

//______________________________________________________________________________
void TSelectorCint::Init(TTree *tree)
{
   // Invoke the Init function via the interpreter.

   if (gDebug > 2)
      Info("Init","Call Init tree = %p", tree);

   gCint->CallFunc_ResetArg(fFuncInit);
   gCint->CallFunc_SetArg(fFuncInit, (Long_t)tree);
   gCint->CallFunc_Exec(fFuncInit, fIntSelector);
}

//______________________________________________________________________________
void TSelectorCint::Begin(TTree *tree)
{
   // Invoke the Begin function via the interpreter.

   if (gDebug > 2)
      Info("Begin","Call Begin tree = %p", tree);
   gCint->CallFunc_ResetArg(fFuncBegin);
   gCint->CallFunc_SetArg(fFuncBegin, (Long_t)tree);
   gCint->CallFunc_ExecInt(fFuncBegin, fIntSelector);
}

//______________________________________________________________________________
void TSelectorCint::SlaveBegin(TTree *tree)
{
   // Invoke the SlaveBegin function via the interpreter if available.

   if (gDebug > 2)
      Info("SlaveBegin","Call SlaveBegin tree = %p", tree);

   if (gCint->CallFunc_IsValid(fFuncSlBegin)) {
      gCint->CallFunc_ResetArg(fFuncSlBegin);
      gCint->CallFunc_SetArg(fFuncSlBegin, (Long_t)tree);
      gCint->CallFunc_ExecInt(fFuncSlBegin, fIntSelector);
   } else {
      if (gDebug > 1)
         Info("SlaveBegin","SlaveBegin unavailable");
   }
}

//______________________________________________________________________________
Bool_t TSelectorCint::Notify()
{
   // Invoke the Notify function via the interpreter.

   if (gDebug > 2)
      Info("Notify","Call Notify");
   Long64_t sel = gCint->CallFunc_ExecInt(fFuncNotif, fIntSelector);
   return (Bool_t)sel;
}

//______________________________________________________________________________
Bool_t TSelectorCint::ProcessCut(Long64_t entry)
{
   // Invoke the ProcessCut function via the interpreter.

   if (gDebug > 3)
      Info("ProcessCut","Call ProcessCut entry = %lld", entry);

   if(gCint->CallFunc_IsValid(fFuncCut)) {
      gCint->CallFunc_ResetArg(fFuncCut);
      gCint->CallFunc_SetArg(fFuncCut, (Long_t)entry);
      Int_t sel = gCint->CallFunc_ExecInt(fFuncCut, fIntSelector);
      return (Bool_t)sel;
   } else {
      Error("ProcessCut","ProcessCut unavailable");
      return kFALSE;
   }
}

//______________________________________________________________________________
void TSelectorCint::ProcessFill(Long64_t entry)
{
   // Invoke the ProcessFill function via the interpreter.

   if (gDebug > 3)
      Info("ProcessFill","Call ProcessFill entry = %lld", entry);

   if(gCint->CallFunc_IsValid(fFuncFill)) {
      gCint->CallFunc_ResetArg(fFuncFill);
      gCint->CallFunc_SetArg(fFuncFill, (Long_t)entry);
      gCint->CallFunc_Exec(fFuncFill, fIntSelector);
   } else {
      Error("ProcessFill","ProcessFill unavailable");
   }
}

//______________________________________________________________________________
Bool_t TSelectorCint::Process(Long64_t entry)
{
   // Invoke the ProcessCut function via the interpreter.

   if (gDebug > 3)
      Info("Process","Call Process entry = %lld", entry);

   if(gCint->CallFunc_IsValid(fFuncProc)) {
      gCint->CallFunc_ResetArg(fFuncProc);
      gCint->CallFunc_SetArg(fFuncProc, (Long_t)entry);
      Int_t sel = gCint->CallFunc_ExecInt(fFuncProc, fIntSelector);
      return (Bool_t)sel;
   } else {
      Error("Process","Process unavailable");
      return kFALSE;
   }
}

//______________________________________________________________________________
void TSelectorCint::SetOption(const char *option)
{
   // Set the selector option.

   if (gDebug > 2)
      Info("SetOption","Option = %s", option);
   gCint->CallFunc_ResetArg(fFuncOption);
   gCint->CallFunc_SetArg(fFuncOption, (Long_t)option);
   gCint->CallFunc_Exec(fFuncOption, fIntSelector);
}

//______________________________________________________________________________
void TSelectorCint::SetObject(TObject *obj)
{
   // Set the current object.

   if (gDebug > 3)
      Info("SetObject","Object = %p", obj);
   gCint->CallFunc_ResetArg(fFuncObj);
   gCint->CallFunc_SetArg(fFuncObj, (Long_t)obj);
   gCint->CallFunc_Exec(fFuncObj, fIntSelector);
}

//______________________________________________________________________________
void TSelectorCint::SetInputList(TList *input)
{
   // Set the selector list of input objects.

   if (gDebug > 2)
      Info("SetInputList","Object = %p", input);
   gCint->CallFunc_ResetArg(fFuncInp);
   gCint->CallFunc_SetArg(fFuncInp,(Long_t)input);
   gCint->CallFunc_Exec(fFuncInp,fIntSelector);
}

//______________________________________________________________________________
TList *TSelectorCint::GetOutputList() const
{
   // Return the list of output object.

   TList *out = (TList *) gCint->CallFunc_ExecInt(fFuncOut, fIntSelector);

   if (gDebug > 2)
      Info("GetOutputList","List = %p", out);

   return out;
}

//______________________________________________________________________________
void TSelectorCint::SlaveTerminate()
{
   // Invoke the SlaveTerminate function via the interpreter if available.

   if (gDebug > 2)
      Info("SlaveTerminate","Call SlaveTerminate");

   if(gCint->CallFunc_IsValid(fFuncSlTerm)) {
      gCint->CallFunc_Exec(fFuncSlTerm, fIntSelector);
   } else {
      if (gDebug > 1)
         Info("SlaveTerminate","SlaveTerminate unavailable");
   }
}

//______________________________________________________________________________
void TSelectorCint::Terminate()
{
   // Invoke the Terminate function via the interpreter.

   if (gDebug > 2)
      Info("Terminate","Call Terminate");
   gCint->CallFunc_Exec(fFuncTerm,fIntSelector);
}

//______________________________________________________________________________
void TSelectorCint::Abort(const char *mesg, EAbort what)
{
   // Invoke the GetAbort function via the interpreter.
   
   if (gDebug > 2)
      Info("Abort","Call Abort");
   
   if (gCint->CallFunc_IsValid(fFuncAbort)) {
      gCint->CallFunc_ResetArg(fFuncAbort);
      gCint->CallFunc_SetArg(fFuncAbort, (Long_t)mesg);
      gCint->CallFunc_SetArg(fFuncAbort, (Long_t)what);
      gCint->CallFunc_ExecInt(fFuncAbort, fIntSelector);
   }
}

//______________________________________________________________________________
TSelector::EAbort TSelectorCint::GetAbort() const
{
   // Invoke the GetAbort function via the interpreter.
   
   if (gDebug > 2)
      Info("GetAbort","Call GetAbort");
   
   if (gCint->CallFunc_IsValid(fFuncGetAbort)) {
      gCint->CallFunc_ResetArg(fFuncGetAbort);
      return (EAbort)gCint->CallFunc_ExecInt(fFuncGetAbort, fIntSelector);
   } else {
      return kContinue; // emulate for old version
   }
}

//______________________________________________________________________________
void TSelectorCint::ResetAbort()
{
   // Invoke the GetAbort function via the interpreter.
   
   if (gDebug > 2)
      Info("ResetAbort","Call ResetAbort");
   
   if (gCint->CallFunc_IsValid(fFuncResetAbort)) {
      gCint->CallFunc_ResetArg(fFuncResetAbort);
      gCint->CallFunc_ExecInt(fFuncResetAbort, fIntSelector);
   }
}

//______________________________________________________________________________
Long64_t TSelectorCint::GetStatus() const
{
   // Invoke the GetStatus function via the interpreter.

   if (gDebug > 2)
      Info("GetStatus","Call GetStatus");

   if (gCint->CallFunc_IsValid(fFuncGetStat)) {
      gCint->CallFunc_ResetArg(fFuncGetStat);
      return gCint->CallFunc_ExecInt64(fFuncGetStat, fIntSelector);
   } else {
      return 0; // emulate for old version
   }
}

//______________________________________________________________________________
TClass *TSelectorCint::GetInterpretedClass() const
{
   // Retrieve the TClass object for the interpreted class.

   if (!fClass) return 0;
   return TClass::GetClass(gCint->ClassInfo_FullName(fClass), kTRUE);
}
