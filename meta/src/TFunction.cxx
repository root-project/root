// @(#)root/meta:$Name:  $:$Id: TFunction.cxx,v 1.4 2002/01/17 12:14:09 rdm Exp $
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
      if (arg.DefaultValue() && strlen(arg.DefaultValue())) {
         fSignature += " = ";
         char *charstar = strstr(arg.Type()->TrueName(),"char*");
         if (charstar) fSignature += "\"";
         fSignature += arg.DefaultValue();
         if (charstar) fSignature += "\"";
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
const char *TFunction::GetName() const
{
   // Get function name.

   return fInfo->Name();
}

//______________________________________________________________________________
const char *TFunction::GetTitle() const
{
   // Get function description string (comment).

   return fInfo->Title();
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
Int_t TFunction::Compare(const TObject *obj) const
{
   // Compare to other object. Returns 0<, 0 or >0 depending on
   // whether "this" is lexicographically less than, equal to, or
   // greater than obj.

   return strcmp(fInfo->Name(), obj->GetName());
}

//______________________________________________________________________________
ULong_t TFunction::Hash() const
{
   // Return hash value for TFunction based on its name.

   TString s = fInfo->Name();
   return s.Hash();
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
