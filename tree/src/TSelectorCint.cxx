// @(#)root/treeplayer:$Name:  $:$Id: TSelectorCint.cxx,v 1.6 2002/04/19 18:24:02 rdm Exp $
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

ClassImp(TSelectorCint)

//______________________________________________________________________________
TSelectorCint::TSelectorCint() : TSelector()
{
   // Default constructor for a Selector.

   fFuncInit    = 0;
   fFuncBegin   = 0;
   fFuncNotif   = 0;
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

   delete fFuncInit;
   delete fFuncBegin;
   delete fFuncNotif;
   delete fFuncTerm;
   delete fFuncCut;
   delete fFuncFill;
   delete fFuncProc;
   delete fFuncOption;
   delete fFuncObj;
   delete fFuncInp;
   delete fFuncOut;
   delete fIntSelector;
}



//______________________________________________________________________________
void TSelectorCint::SetFuncProto(G__CallFunc *cf, G__ClassInfo* cl, const char* fname, const char* argtype)
{
   Long_t offset = 0;

   cf->SetFuncProto(cl,fname,argtype,&offset);

   if ( gDebug > 2 )
      Info("SetFuncProto","Set %s(%s)  offset = %ld",fname,argtype,offset);

   // TODO: this condition seems inverted ???
   if ( cf->IsValid() )
      Error("SetFuncProto","Cannot Set %s(%s)",fname,argtype);
}


//______________________________________________________________________________
void TSelectorCint::Build(TSelector *iselector, G__ClassInfo *cl)
{
   // Initialize the CallFunc objects when selector is interpreted

   fIntSelector = iselector;
   fFuncInit    = new G__CallFunc();
   fFuncBegin   = new G__CallFunc();
   fFuncNotif   = new G__CallFunc();
   fFuncTerm    = new G__CallFunc();
   fFuncCut     = new G__CallFunc();
   fFuncFill    = new G__CallFunc();
   fFuncProc    = new G__CallFunc();
   fFuncOption  = new G__CallFunc();
   fFuncObj     = new G__CallFunc();
   fFuncInp     = new G__CallFunc();
   fFuncOut     = new G__CallFunc();

   SetFuncProto(fFuncInit,cl,"Init","TTree*");
   SetFuncProto(fFuncBegin,cl,"Begin","TTree*");
   SetFuncProto(fFuncNotif,cl,"Notify","");
   SetFuncProto(fFuncTerm,cl,"Terminate","");
   SetFuncProto(fFuncCut,cl,"ProcessCut","int");
   SetFuncProto(fFuncFill,cl,"ProcessFill","int");
   SetFuncProto(fFuncProc,cl,"Process","int");
   SetFuncProto(fFuncOption,cl,"SetOption","const char*");
   SetFuncProto(fFuncObj,cl,"SetObject","TObject*");
   SetFuncProto(fFuncInp,cl,"SetInputList","TList*");
   SetFuncProto(fFuncOut,cl,"GetOutputList","");
}


//______________________________________________________________________________
void TSelectorCint::Init(TTree *tree)
{
   // Invoke the Init function via the interpreter

   if ( gDebug > 2 )
      Info("Init","Call Init tree = %p", tree);

   fFuncInit->ResetArg();
   fFuncInit->SetArg((Long_t)tree);
   fFuncInit->Exec(fIntSelector);
}


//______________________________________________________________________________
void TSelectorCint::Begin(TTree *tree)
{
   // Invoke the Begin function via the interpreter
   if ( gDebug > 2 )
      Info("Begin","Call Begin tree = %p", tree);
   fFuncBegin->ResetArg();
   fFuncBegin->SetArg((Long_t)tree);
   fFuncBegin->ExecInt(fIntSelector);
}


//______________________________________________________________________________
Bool_t TSelectorCint::Notify()
{
   // Invoke the Notify function via the interpreter
   if ( gDebug > 2 )
      Info("Notify","Call Notify");
   Int_t sel = fFuncNotif->ExecInt(fIntSelector);
   return (Bool_t)sel;
}


//______________________________________________________________________________
Bool_t TSelectorCint::ProcessCut(Int_t entry)
{
   // Invoke the ProcessCut function via the interpreter
   if ( gDebug > 3 )
      Info("ProcessCut","Call ProcessCut entry = %d", entry);
   fFuncCut->ResetArg();
   fFuncCut->SetArg((Long_t)entry);
   Int_t sel = fFuncCut->ExecInt(fIntSelector);
   return (Bool_t)sel;
}


//______________________________________________________________________________
void TSelectorCint::ProcessFill(Int_t entry)
{
   // Invoke the ProcessFill function via the interpreter
   if ( gDebug > 3 )
      Info("ProcessFill","Call ProcessFill entry = %d", entry);
   fFuncFill->ResetArg();
   fFuncFill->SetArg((Long_t)entry);
   fFuncFill->Exec(fIntSelector);
}


//______________________________________________________________________________
Bool_t TSelectorCint::Process(Int_t entry)
{
   // Invoke the ProcessCut function via the interpreter
   if ( gDebug > 3 )
      Info("Process","Call Process entry = %d", entry);
   fFuncProc->ResetArg();
   fFuncProc->SetArg((Long_t)entry);
   Int_t sel = fFuncProc->ExecInt(fIntSelector);
   return (Bool_t)sel;
}


//______________________________________________________________________________
void TSelectorCint::SetOption(const char *option)
{
   // Set the selector option
   if ( gDebug > 2 )
      Info("SetOption","Option = %s", option);
   fFuncOption->ResetArg();
   fFuncOption->SetArg((Long_t)option);
   fFuncOption->Exec(fIntSelector);
}


//______________________________________________________________________________
void TSelectorCint::SetObject(TObject *obj)
{
   // Set the current object
   if ( gDebug > 3 )
      Info("SetObject","Object = %p", obj);
   fFuncObj->ResetArg();
   fFuncObj->SetArg((Long_t)obj);
   fFuncObj->Exec(fIntSelector);
}


//______________________________________________________________________________
void TSelectorCint::SetInputList(TList *input)
{
   // Set the selector list of input objects
   if ( gDebug > 2 )
      Info("SetInputList","Object = %p", input);
   fFuncInp->ResetArg();
   fFuncInp->SetArg((Long_t)input);
   fFuncInp->Exec(fIntSelector);
}


//______________________________________________________________________________
TList *TSelectorCint::GetOutputList() const
{
   // Return the list of output object

   TList *out = (TList *) fFuncOut->ExecInt(fIntSelector);

   if ( gDebug > 2 )
      Info("GetOutputList","List = %p", out);

   return out;
}


//______________________________________________________________________________
void TSelectorCint::Terminate()
{
   // Invoke the Terminate function via the interpreter
   if ( gDebug > 2 )
      Info("Terminate","Call Terminate");
   fFuncTerm->Exec(fIntSelector);
}
