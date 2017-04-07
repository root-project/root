// @(#)root/tree:$Id$
// Author: Rene Brun   05/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TSelectorCint
\ingroup tree

This class is a special version of TSelector for user interpreted classes.
*/

#include "TROOT.h"
#include "TTree.h"
#include "THashList.h"
#define ROOT_TSelectorCint_SRC_FILE
#include "TSelectorCint.h"
#undef ROOT_TSelectorCint_SRC_FILE
#include "TError.h"

ClassImp(TSelectorCint)

////////////////////////////////////////////////////////////////////////////////

TSelectorCint::TSelectorCint() : TSelector(),
   fClass(0),
   fFuncVersion    (0),
   fFuncInit       (0),
   fFuncBegin      (0),
   fFuncSlBegin    (0),
   fFuncNotif      (0),
   fFuncSlTerm     (0),
   fFuncTerm       (0),
   fFuncCut        (0),
   fFuncFill       (0),
   fFuncProc       (0),
   fFuncOption     (0),
   fFuncObj        (0),
   fFuncInp        (0),
   fFuncOut        (0),
   fFuncAbort      (0),
   fFuncGetAbort   (0),
   fFuncResetAbort (0),
   fFuncGetStat    (0),
   fIntSelector(0),fIsOwner(kFALSE)

{
   // Default constructor for a Selector.

}

////////////////////////////////////////////////////////////////////////////////
/// Destructor for a Selector.

TSelectorCint::~TSelectorCint()
{
   gCling->CallFunc_Delete(fFuncVersion);
   gCling->CallFunc_Delete(fFuncInit);
   gCling->CallFunc_Delete(fFuncBegin);
   gCling->CallFunc_Delete(fFuncSlBegin);
   gCling->CallFunc_Delete(fFuncNotif);
   gCling->CallFunc_Delete(fFuncSlTerm);
   gCling->CallFunc_Delete(fFuncTerm);
   gCling->CallFunc_Delete(fFuncCut);
   gCling->CallFunc_Delete(fFuncFill);
   gCling->CallFunc_Delete(fFuncProc);
   gCling->CallFunc_Delete(fFuncOption);
   gCling->CallFunc_Delete(fFuncObj);
   gCling->CallFunc_Delete(fFuncInp);
   gCling->CallFunc_Delete(fFuncOut);
   gCling->CallFunc_Delete(fFuncAbort);
   gCling->CallFunc_Delete(fFuncGetAbort);
   gCling->CallFunc_Delete(fFuncResetAbort);
   gCling->CallFunc_Delete(fFuncGetStat);

   if (fIsOwner && fIntSelector) gCling->ClassInfo_Delete(fClass,fIntSelector);
   gCling->ClassInfo_Delete(fClass);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the function prototype.

void TSelectorCint::SetFuncProto(CallFunc_t *cf, ClassInfo_t *cl,
                                 const char* fname, const char* argtype,
                                 Bool_t required)
{
   Long_t offset = 0;

   gCling->CallFunc_SetFuncProto(cf, cl,fname,argtype,&offset);

   if (gDebug > 2)
      Info("SetFuncProto","set %s(%s) offset = %ld",fname,argtype,offset);

   if (!gCling->CallFunc_IsValid(cf) && required)
      Error("SetFuncProto","cannot set %s(%s)",fname,argtype);
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize the CallFunc objects when selector is interpreted.

void TSelectorCint::Build(TSelector *iselector, ClassInfo_t *cl, Bool_t isowner)
{
   gCling->CallFunc_Delete(fFuncVersion);
   gCling->CallFunc_Delete(fFuncInit);
   gCling->CallFunc_Delete(fFuncBegin);
   gCling->CallFunc_Delete(fFuncSlBegin);
   gCling->CallFunc_Delete(fFuncNotif);
   gCling->CallFunc_Delete(fFuncSlTerm);
   gCling->CallFunc_Delete(fFuncTerm);
   gCling->CallFunc_Delete(fFuncCut);
   gCling->CallFunc_Delete(fFuncFill);
   gCling->CallFunc_Delete(fFuncProc);
   gCling->CallFunc_Delete(fFuncOption);
   gCling->CallFunc_Delete(fFuncObj);
   gCling->CallFunc_Delete(fFuncInp);
   gCling->CallFunc_Delete(fFuncOut);
   gCling->CallFunc_Delete(fFuncAbort);
   gCling->CallFunc_Delete(fFuncGetAbort);
   gCling->CallFunc_Delete(fFuncResetAbort);
   gCling->CallFunc_Delete(fFuncGetStat);

   if (fIsOwner && fIntSelector) gCling->ClassInfo_Delete(fClass, fIntSelector);
   gCling->ClassInfo_Delete(fClass);

   R__ASSERT(cl);

   // The CINT MethodInfo created by SetFuncProto will remember the address
   // of cl, so we need to keep it around.
   fClass        = gCling->ClassInfo_Factory(cl);

   fIntSelector    = iselector;
   fIsOwner        = isowner;
   fFuncVersion    = gCling->CallFunc_Factory();
   fFuncInit       = gCling->CallFunc_Factory();
   fFuncBegin      = gCling->CallFunc_Factory();
   fFuncSlBegin    = gCling->CallFunc_Factory();
   fFuncNotif      = gCling->CallFunc_Factory();
   fFuncSlTerm     = gCling->CallFunc_Factory();
   fFuncTerm       = gCling->CallFunc_Factory();
   fFuncCut        = gCling->CallFunc_Factory();
   fFuncFill       = gCling->CallFunc_Factory();
   fFuncProc       = gCling->CallFunc_Factory();
   fFuncOption     = gCling->CallFunc_Factory();
   fFuncObj        = gCling->CallFunc_Factory();
   fFuncInp        = gCling->CallFunc_Factory();
   fFuncOut        = gCling->CallFunc_Factory();
   fFuncAbort      = gCling->CallFunc_Factory();
   fFuncGetAbort   = gCling->CallFunc_Factory();
   fFuncResetAbort = gCling->CallFunc_Factory();
   fFuncGetStat    = gCling->CallFunc_Factory();

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

////////////////////////////////////////////////////////////////////////////////
/// Invoke the Version function via the interpreter.

int TSelectorCint::Version() const
{
   if (gDebug > 2)
      Info("Version","Call Version");

   if (gCling->CallFunc_IsValid(fFuncVersion)) {
      gCling->CallFunc_ResetArg(fFuncVersion);
      return gCling->CallFunc_ExecInt(fFuncVersion, fIntSelector);
   } else {
      return 0; // emulate for old version
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Invoke the Init function via the interpreter.

void TSelectorCint::Init(TTree *tree)
{
   if (gDebug > 2)
      Info("Init","Call Init tree = %p", tree);

   gCling->CallFunc_ResetArg(fFuncInit);
   gCling->CallFunc_SetArg(fFuncInit, (Long_t)tree);
   gCling->CallFunc_Exec(fFuncInit, fIntSelector);
}

////////////////////////////////////////////////////////////////////////////////
/// Invoke the Begin function via the interpreter.

void TSelectorCint::Begin(TTree *tree)
{
   if (gDebug > 2)
      Info("Begin","Call Begin tree = %p", tree);
   gCling->CallFunc_ResetArg(fFuncBegin);
   gCling->CallFunc_SetArg(fFuncBegin, (Long_t)tree);
   gCling->CallFunc_ExecInt(fFuncBegin, fIntSelector);
}

////////////////////////////////////////////////////////////////////////////////
/// Invoke the SlaveBegin function via the interpreter if available.

void TSelectorCint::SlaveBegin(TTree *tree)
{
   if (gDebug > 2)
      Info("SlaveBegin","Call SlaveBegin tree = %p", tree);

   if (gCling->CallFunc_IsValid(fFuncSlBegin)) {
      gCling->CallFunc_ResetArg(fFuncSlBegin);
      gCling->CallFunc_SetArg(fFuncSlBegin, (Long_t)tree);
      gCling->CallFunc_ExecInt(fFuncSlBegin, fIntSelector);
   } else {
      if (gDebug > 1)
         Info("SlaveBegin","SlaveBegin unavailable");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Invoke the Notify function via the interpreter.

Bool_t TSelectorCint::Notify()
{
   if (gDebug > 2)
      Info("Notify","Call Notify");
   Long64_t sel = gCling->CallFunc_ExecInt(fFuncNotif, fIntSelector);
   return (Bool_t)sel;
}

////////////////////////////////////////////////////////////////////////////////
/// Invoke the ProcessCut function via the interpreter.

Bool_t TSelectorCint::ProcessCut(Long64_t entry)
{
   if (gDebug > 3)
      Info("ProcessCut","Call ProcessCut entry = %lld", entry);

   if (gCling->CallFunc_IsValid(fFuncCut)) {
      gCling->CallFunc_ResetArg(fFuncCut);
      gCling->CallFunc_SetArg(fFuncCut, (Long_t)entry);
      Int_t sel = gCling->CallFunc_ExecInt(fFuncCut, fIntSelector);
      return (Bool_t)sel;
   } else {
      Error("ProcessCut","ProcessCut unavailable");
      return kFALSE;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Invoke the ProcessFill function via the interpreter.

void TSelectorCint::ProcessFill(Long64_t entry)
{
   if (gDebug > 3)
      Info("ProcessFill","Call ProcessFill entry = %lld", entry);

   if (gCling->CallFunc_IsValid(fFuncFill)) {
      gCling->CallFunc_ResetArg(fFuncFill);
      gCling->CallFunc_SetArg(fFuncFill, (Long_t)entry);
      gCling->CallFunc_Exec(fFuncFill, fIntSelector);
   } else {
      Error("ProcessFill","ProcessFill unavailable");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Invoke the ProcessCut function via the interpreter.

Bool_t TSelectorCint::Process(Long64_t entry)
{
   if (gDebug > 3)
      Info("Process","Call Process entry = %lld", entry);

   if (gCling->CallFunc_IsValid(fFuncProc)) {
      gCling->CallFunc_ResetArg(fFuncProc);
      gCling->CallFunc_SetArg(fFuncProc, (Long_t)entry);
      Int_t sel = gCling->CallFunc_ExecInt(fFuncProc, fIntSelector);
      return (Bool_t)sel;
   } else {
      Error("Process","Process unavailable");
      return kFALSE;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set the selector option.

void TSelectorCint::SetOption(const char *option)
{
   if (gDebug > 2)
      Info("SetOption","Option = %s", option);
   gCling->CallFunc_ResetArg(fFuncOption);
   gCling->CallFunc_SetArg(fFuncOption, (Long_t)option);
   gCling->CallFunc_Exec(fFuncOption, fIntSelector);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the current object.

void TSelectorCint::SetObject(TObject *obj)
{
   if (gDebug > 3)
      Info("SetObject","Object = %p", obj);
   gCling->CallFunc_ResetArg(fFuncObj);
   gCling->CallFunc_SetArg(fFuncObj, (Long_t)obj);
   gCling->CallFunc_Exec(fFuncObj, fIntSelector);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the selector list of input objects.

void TSelectorCint::SetInputList(TList *input)
{
   if (gDebug > 2)
      Info("SetInputList","Object = %p", input);
   gCling->CallFunc_ResetArg(fFuncInp);
   gCling->CallFunc_SetArg(fFuncInp,(Long_t)input);
   gCling->CallFunc_Exec(fFuncInp,fIntSelector);
}

////////////////////////////////////////////////////////////////////////////////
/// Return the list of output object.

TList *TSelectorCint::GetOutputList() const
{
   TList *out = (TList *) gCling->CallFunc_ExecInt(fFuncOut, fIntSelector);

   if (gDebug > 2)
      Info("GetOutputList","List = %p", out);

   return out;
}

////////////////////////////////////////////////////////////////////////////////
/// Invoke the SlaveTerminate function via the interpreter if available.

void TSelectorCint::SlaveTerminate()
{
   if (gDebug > 2)
      Info("SlaveTerminate","Call SlaveTerminate");

   if (gCling->CallFunc_IsValid(fFuncSlTerm)) {
      gCling->CallFunc_Exec(fFuncSlTerm, fIntSelector);
   } else {
      if (gDebug > 1)
         Info("SlaveTerminate","SlaveTerminate unavailable");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Invoke the Terminate function via the interpreter.

void TSelectorCint::Terminate()
{
   if (gDebug > 2)
      Info("Terminate","Call Terminate");
   gCling->CallFunc_Exec(fFuncTerm,fIntSelector);
}

////////////////////////////////////////////////////////////////////////////////
/// Invoke the GetAbort function via the interpreter.

void TSelectorCint::Abort(const char *mesg, EAbort what)
{
   if (gDebug > 2)
      Info("Abort","Call Abort");

   if (gCling->CallFunc_IsValid(fFuncAbort)) {
      gCling->CallFunc_ResetArg(fFuncAbort);
      gCling->CallFunc_SetArg(fFuncAbort, (Long_t)mesg);
      gCling->CallFunc_SetArg(fFuncAbort, (Long_t)what);
      gCling->CallFunc_ExecInt(fFuncAbort, fIntSelector);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Invoke the GetAbort function via the interpreter.

TSelector::EAbort TSelectorCint::GetAbort() const
{
   if (gDebug > 2)
      Info("GetAbort","Call GetAbort");

   if (gCling->CallFunc_IsValid(fFuncGetAbort)) {
      gCling->CallFunc_ResetArg(fFuncGetAbort);
      return (EAbort)gCling->CallFunc_ExecInt(fFuncGetAbort, fIntSelector);
   } else {
      return kContinue; // emulate for old version
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Invoke the GetAbort function via the interpreter.

void TSelectorCint::ResetAbort()
{
   if (gDebug > 2)
      Info("ResetAbort","Call ResetAbort");

   if (gCling->CallFunc_IsValid(fFuncResetAbort)) {
      gCling->CallFunc_ResetArg(fFuncResetAbort);
      gCling->CallFunc_ExecInt(fFuncResetAbort, fIntSelector);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Invoke the GetStatus function via the interpreter.

Long64_t TSelectorCint::GetStatus() const
{
   if (gDebug > 2)
      Info("GetStatus","Call GetStatus");

   if (gCling->CallFunc_IsValid(fFuncGetStat)) {
      gCling->CallFunc_ResetArg(fFuncGetStat);
      return gCling->CallFunc_ExecInt64(fFuncGetStat, fIntSelector);
   } else {
      return 0; // emulate for old version
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve the TClass object for the interpreted class.

TClass *TSelectorCint::GetInterpretedClass() const
{
   if (!fClass) return 0;
   return TClass::GetClass(gCling->ClassInfo_FullName(fClass), kTRUE);
}
