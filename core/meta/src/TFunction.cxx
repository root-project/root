// @(#)root/meta:$Id$
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


ClassImp(TFunction)

//______________________________________________________________________________
TFunction::TFunction(MethodInfo_t *info) : TDictionary()
{
   // Default TFunction ctor. TFunctions are constructed in TROOT via
   // a call to TCint::UpdateListOfGlobalFunctions().

   fInfo       = info;
   fMethodArgs = 0;
   if (fInfo) {
      SetName(gCint->MethodInfo_Name(fInfo));
      SetTitle(gCint->MethodInfo_Title(fInfo));
      fMangledName = gCint->MethodInfo_GetMangledName(fInfo);
   }
}

//______________________________________________________________________________
TFunction::TFunction(const TFunction &orig) : TDictionary(orig)
{
   // Copy operator.

   if (orig.fInfo) {
      fInfo = gCint->MethodInfo_FactoryCopy(orig.fInfo);
      fMangledName = gCint->MethodInfo_GetMangledName(fInfo);
   } else
      fInfo = 0;
   fMethodArgs = 0;
}

//______________________________________________________________________________
TFunction& TFunction::operator=(const TFunction &rhs)
{
   // Assignment operator.

   if (this != &rhs) {
      gCint->MethodInfo_Delete(fInfo);
      if (fMethodArgs) fMethodArgs->Delete();
      delete fMethodArgs;
      if (rhs.fInfo) {
         fInfo = gCint->MethodInfo_FactoryCopy(rhs.fInfo);
         SetName(gCint->MethodInfo_Name(fInfo));
         SetTitle(gCint->MethodInfo_Title(fInfo));
         fMangledName = gCint->MethodInfo_GetMangledName(fInfo);
      } else
         fInfo = 0;
      fMethodArgs = 0;
   }
   return *this;
}

//______________________________________________________________________________
TFunction::~TFunction()
{
   // TFunction dtor deletes adopted CINT MethodInfo.

   gCint->MethodInfo_Delete(fInfo);

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

   gCint->MethodInfo_CreateSignature(fInfo, fSignature);
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

   if (gCint->MethodInfo_Type(fInfo) == 0) return "Unknown";
   return gCint->MethodInfo_TypeName(fInfo);
}

//______________________________________________________________________________
Int_t TFunction::GetNargs() const
{
   // Number of function arguments.

   return gCint->MethodInfo_NArg(fInfo);
}

//______________________________________________________________________________
Int_t TFunction::GetNargsOpt() const
{
   // Number of function optional (default) arguments.

   return gCint->MethodInfo_NDefaultArg(fInfo);
}

//______________________________________________________________________________
Long_t TFunction::Property() const
{
   // Get property description word. For meaning of bits see EProperty.

   return gCint->MethodInfo_Property(fInfo);
}

//______________________________________________________________________________
void *TFunction::InterfaceMethod() const
{
   // Return pointer to the interface method. Using this pointer we
   // can find which TFunction belongs to a CINT MethodInfo object.
   // Both need to have the same InterfaceMethod pointer.

   return gCint->MethodInfo_InterfaceMethod(fInfo);
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
      return gCint->MethodInfo_GetPrototype(fInfo);
   else
      return 0;
}
