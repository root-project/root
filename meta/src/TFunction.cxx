// @(#)root/meta:$Name:  $:$Id: TFunction.cxx,v 1.14 2005/11/16 20:10:45 pcanal Exp $
// Author: Fons Rademakers   07/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Global functions class (global functions are obtaine from CINT).     //
// This class describes one single global function.                     //
// The TROOT class contains a list of all currently defined global      //
// functions (accessible via TROOT::GetListOfGlobalFunctions()).        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TFunction.h"
#include "TMethodArg.h"
#include "TROOT.h"
#include "TInterpreter.h"
#include "Strlen.h"
#include "Api.h"


ClassImp(TFunction)

//______________________________________________________________________________
TFunction::TFunction(G__MethodInfo *info) : TDictionary()
{
   // Default TFunction ctor. TFunctions are constructed in TROOT via
   // a call to TCint::UpdateListOfGlobalFunctions().

   fInfo       = info;
   fMethodArgs = 0;
   if (fInfo) {
      SetName(fInfo->Name());
      SetTitle(fInfo->Title());
      fMangledName = fInfo->GetMangledName();
   }
}

//______________________________________________________________________________
TFunction::TFunction(const TFunction &orig) : TDictionary(orig)
{
   // Copy operator.

   if (orig.fInfo) {
      fInfo = new G__MethodInfo(*orig.fInfo);
      fMangledName = fInfo->GetMangledName();
   } else
      fInfo = 0;
   fMethodArgs = 0;
}

//______________________________________________________________________________
TFunction& TFunction::operator=(const TFunction &rhs)
{
   // Assignment operator.

   if (this != &rhs) {
      delete fInfo;
      if (fMethodArgs) fMethodArgs->Delete();
      delete fMethodArgs;
      if (rhs.fInfo) {
         fInfo = new G__MethodInfo(*rhs.fInfo);
         SetName(fInfo->Name());
         SetTitle(fInfo->Title());
         fMangledName = fInfo->GetMangledName();
      } else
         fInfo = 0;
      fMethodArgs = 0;
   }
   return *this;
}

//______________________________________________________________________________
TFunction::~TFunction()
{
   // TFunction dtor deletes adopted G__MethodInfo.

   delete fInfo;

   if (fMethodArgs) fMethodArgs->Delete();
   delete fMethodArgs;
}

//______________________________________________________________________________
TObject *TFunction::Clone(const char *newname) const
{
   // Clone method.

   TNamed *newobj = new TFunction(*this);
   if (newname && strlen(newname)) newobj->SetName(newname);
   return newobj;
}

//______________________________________________________________________________
void TFunction::CreateSignature()
{
   // Using the CINT method arg information to create a complete signature string.

   G__MethodArgInfo arg(*fInfo);

   int ifirst = 0;
   fSignature = "(";
   while (arg.Next()) {
      if (ifirst) fSignature += ", ";
      if (arg.Type() == 0) break;
      //if (arg.Property() & G__BIT_ISCONSTANT)   // is returned as part of the name
      //   fSignature += "const ";
      fSignature += arg.Type()->Name();
      if (arg.Name() && strlen(arg.Name())) {
         fSignature += " ";
         fSignature += arg.Name();
      }
      if (arg.DefaultValue()) {
         fSignature += " = ";
         fSignature += arg.DefaultValue();
      }
      ifirst++;
   }
   fSignature += ")";
}

//______________________________________________________________________________
const char *TFunction::GetSignature()
{
   // Return signature of function.

   if (fInfo && fSignature.IsNull())
      CreateSignature();

   return fSignature.Data();
}

//______________________________________________________________________________
TList *TFunction::GetListOfMethodArgs()
{
   // Return list containing the TMethodArgs of a TFunction.

   if (!fMethodArgs) {
      if (!gInterpreter)
         Fatal("GetListOfMethodArgs", "gInterpreter not initialized");

      gInterpreter->CreateListOfMethodArgs(this);
   }
   return fMethodArgs;
}

//______________________________________________________________________________
const char *TFunction::GetReturnTypeName() const
{
   // Get full type description of function return type, e,g.: "class TDirectory*".

   if (fInfo->Type() == 0) return "Unknown";
   return fInfo->Type()->Name();
}

//______________________________________________________________________________
Int_t TFunction::GetNargs() const
{
   // Number of function arguments.

   return fInfo->NArg();
}

//______________________________________________________________________________
Int_t TFunction::GetNargsOpt() const
{
   // Number of function optional (default) arguments.

   return fInfo->NDefaultArg();
}

//______________________________________________________________________________
Long_t TFunction::Property() const
{
   // Get property description word. For meaning of bits see EProperty.

   return fInfo->Property();
}

//______________________________________________________________________________
void *TFunction::InterfaceMethod() const
{
   // Return pointer to the interface method. Using this pointer we
   // can find which TFunction belongs to a G__MethodInfo object.
   // Both need to have the same InterfaceMethod pointer.

   G__InterfaceMethod pfunc = fInfo->InterfaceMethod();
   if (!pfunc) {
      struct G__bytecodefunc *bytecode = fInfo->GetBytecode();

      if(bytecode) pfunc = (G__InterfaceMethod)G__exec_bytecode;
      else {
         pfunc = (G__InterfaceMethod)NULL;
      }
   }
   return (void*)pfunc;
}

//______________________________________________________________________________
const char *TFunction::GetMangledName() const
{
   // Returns the mangled name as defined by CINT, or 0 in case of error.

   // This function is being used by TROOT to determine the full identity of
   // of the function.  It has to work even if the function has been
   // unloaded by cint (in which case fInfo is actually hold reference to
   // memory that is (likely) not valid anymore.  So we cache the information.
   // Maybe we should also cache the rest of the informations .. but this might
   // be too much duplication of information.
   if (fInfo)
      return fMangledName;
   else
      return 0;
}

//______________________________________________________________________________
const char *TFunction::GetPrototype() const
{
   // Returns the prototype of a function as defined by CINT, or 0 in
   // case of error.

   if (fInfo)
      return fInfo->GetPrototype();
   else
      return 0;
}
