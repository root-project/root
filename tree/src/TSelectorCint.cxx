// @(#)root/tree:$Name:  $:$Id: TSelectorCint.cxx,v 1.20 2005/11/14 22:36:48 rdm Exp $
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
#include "Api.h"
#include "TError.h"

ClassImp(TSelectorCint)

//______________________________________________________________________________
TSelectorCint::TSelectorCint() : TSelector()
{
   // Default constructor for a Selector.

   fFuncVersion = 0;
   fFuncInit    = 0;
   fFuncBegin   = 0;
   fFuncSlBegin = 0;
   fFuncNotif   = 0;
   fFuncSlTerm  = 0;
   fFuncTerm    = 0;
   fFuncCut     = 0;
   fFuncFill    = 0;
   fFuncProc    = 0;
   fFuncOption  = 0;
   fFuncObj     = 0;
   fFuncInp     = 0;
   fFuncOut     = 0;
   fIntSelector = 0;

}

//______________________________________________________________________________
TSelectorCint::~TSelectorCint()
{
   // destructor for a Selector.

   delete fFuncVersion;
   delete fFuncInit;
   delete fFuncBegin;
   delete fFuncSlBegin;
   delete fFuncNotif;
   delete fFuncSlTerm;
   delete fFuncTerm;
   delete fFuncCut;
   delete fFuncFill;
   delete fFuncProc;
   delete fFuncOption;
   delete fFuncObj;
   delete fFuncInp;
   delete fFuncOut;

   if (fIntSelector) fClass->Delete(fIntSelector);
   delete fClass;
}

//______________________________________________________________________________
void TSelectorCint::SetFuncProto(G__CallFunc *cf, G__ClassInfo* cl,
                                 const char* fname, const char* argtype,
                                 Bool_t required)
{
   // Set the function prototype.

   Long_t offset = 0;

   cf->SetFuncProto(cl,fname,argtype,&offset);

   if (gDebug > 2)
      Info("SetFuncProto","set %s(%s) offset = %ld",fname,argtype,offset);

   if (!cf->IsValid() && required)
      Error("SetFuncProto","cannot set %s(%s)",fname,argtype);
}

//______________________________________________________________________________
void TSelectorCint::Build(TSelector *iselector, G__ClassInfo *cl)
{
   // Initialize the CallFunc objects when selector is interpreted.

   R__ASSERT(cl);

   // The G__MethodInfo created by SetFuncProto will remember the address
   // of cl, so we need to keep it around.
   fClass       = new G__ClassInfo(*cl);

   fIntSelector = iselector;
   fFuncVersion = new G__CallFunc();
   fFuncInit    = new G__CallFunc();
   fFuncBegin   = new G__CallFunc();
   fFuncSlBegin = new G__CallFunc();
   fFuncNotif   = new G__CallFunc();
   fFuncSlTerm  = new G__CallFunc();
   fFuncTerm    = new G__CallFunc();
   fFuncCut     = new G__CallFunc();
   fFuncFill    = new G__CallFunc();
   fFuncProc    = new G__CallFunc();
   fFuncOption  = new G__CallFunc();
   fFuncObj     = new G__CallFunc();
   fFuncInp     = new G__CallFunc();
   fFuncOut     = new G__CallFunc();

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
}

//______________________________________________________________________________
int TSelectorCint::Version() const
{
   // Invoke the Version function via the interpreter.

   if (gDebug > 2)
      Info("Version","Call Version");

   if (fFuncVersion->IsValid()) {
      fFuncVersion->ResetArg();
      return fFuncVersion->ExecInt(fIntSelector);
   } else {
      return 0; // emulate for old version
   }
}

//______________________________________________________________________________
void TSelectorCint::Init(TTree *tree)
{
   // Invoke the Init function via the interpreter.

   if ( gDebug > 2 )
      Info("Init","Call Init tree = %p", tree);

   fFuncInit->ResetArg();
   fFuncInit->SetArg((Long_t)tree);
   fFuncInit->Exec(fIntSelector);
}

//______________________________________________________________________________
void TSelectorCint::Begin(TTree *tree)
{
   // Invoke the Begin function via the interpreter.

   if ( gDebug > 2 )
      Info("Begin","Call Begin tree = %p", tree);
   fFuncBegin->ResetArg();
   fFuncBegin->SetArg((Long_t)tree);
   fFuncBegin->ExecInt(fIntSelector);
}

//______________________________________________________________________________
void TSelectorCint::SlaveBegin(TTree *tree)
{
   // Invoke the SlaveBegin function via the interpreter if available.

   if (gDebug > 2)
      Info("SlaveBegin","Call SlaveBegin tree = %p", tree);

   if (fFuncSlBegin->IsValid()) {
      fFuncSlBegin->ResetArg();
      fFuncSlBegin->SetArg((Long_t)tree);
      fFuncSlBegin->ExecInt(fIntSelector);
   } else {
      if (gDebug > 1)
         Info("SlaveBegin","SlaveBegin unavailable");
   }
}

//______________________________________________________________________________
Bool_t TSelectorCint::Notify()
{
   // Invoke the Notify function via the interpreter.

   if ( gDebug > 2 )
      Info("Notify","Call Notify");
   Long64_t sel = fFuncNotif->ExecInt(fIntSelector);
   return (Bool_t)sel;
}

//______________________________________________________________________________
Bool_t TSelectorCint::ProcessCut(Long64_t entry)
{
   // Invoke the ProcessCut function via the interpreter.

   if (gDebug > 3)
      Info("ProcessCut","Call ProcessCut entry = %d", entry);

   if(fFuncCut->IsValid()) {
      fFuncCut->ResetArg();
      fFuncCut->SetArg((Long_t)entry);
      Int_t sel = fFuncCut->ExecInt(fIntSelector);
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
      Info("ProcessFill","Call ProcessFill entry = %d", entry);

   if(fFuncFill->IsValid()) {
      fFuncFill->ResetArg();
      fFuncFill->SetArg((Long_t)entry);
      fFuncFill->Exec(fIntSelector);
   } else {
      Error("ProcessFill","ProcessFill unavailable");
   }
}

//______________________________________________________________________________
Bool_t TSelectorCint::Process(Long64_t entry)
{
   // Invoke the ProcessCut function via the interpreter.

   if ( gDebug > 3 )
      Info("Process","Call Process entry = %d", entry);

   if(fFuncProc->IsValid()) {
      fFuncProc->ResetArg();
      fFuncProc->SetArg((Long_t)entry);
      Int_t sel = fFuncProc->ExecInt(fIntSelector);
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

   if ( gDebug > 2 )
      Info("SetOption","Option = %s", option);
   fFuncOption->ResetArg();
   fFuncOption->SetArg((Long_t)option);
   fFuncOption->Exec(fIntSelector);
}

//______________________________________________________________________________
void TSelectorCint::SetObject(TObject *obj)
{
   // Set the current object.

   if ( gDebug > 3 )
      Info("SetObject","Object = %p", obj);
   fFuncObj->ResetArg();
   fFuncObj->SetArg((Long_t)obj);
   fFuncObj->Exec(fIntSelector);
}

//______________________________________________________________________________
void TSelectorCint::SetInputList(TList *input)
{
   // Set the selector list of input objects.

   if ( gDebug > 2 )
      Info("SetInputList","Object = %p", input);
   fFuncInp->ResetArg();
   fFuncInp->SetArg((Long_t)input);
   fFuncInp->Exec(fIntSelector);
}

//______________________________________________________________________________
TList *TSelectorCint::GetOutputList() const
{
   // Return the list of output object.

   TList *out = (TList *) fFuncOut->ExecInt(fIntSelector);

   if ( gDebug > 2 )
      Info("GetOutputList","List = %p", out);

   return out;
}

//______________________________________________________________________________
void TSelectorCint::SlaveTerminate()
{
   // Invoke the SlaveTerminate function via the interpreter if available.

   if ( gDebug > 2 )
      Info("SlaveTerminate","Call SlaveTerminate");

   if(fFuncSlTerm->IsValid()) {
      fFuncSlTerm->Exec(fIntSelector);
   } else {
      if (gDebug > 1)
         Info("SlaveTerminate","SlaveTerminate unavailable");
   }
}

//______________________________________________________________________________
void TSelectorCint::Terminate()
{
   // Invoke the Terminate function via the interpreter.

   if ( gDebug > 2 )
      Info("Terminate","Call Terminate");
   fFuncTerm->Exec(fIntSelector);
}
