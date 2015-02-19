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

#include <iostream>
#include "TVirtualMutex.h"

ClassImp(TFunction)

//______________________________________________________________________________
TFunction::TFunction(MethodInfo_t *info) : TDictionary()
{
   // Default TFunction ctor. TFunctions are constructed in TROOT via
   // a call to TCling::UpdateListOfGlobalFunctions().

   fInfo       = info;
   fMethodArgs = 0;
   if (fInfo) {
      SetName(gCling->MethodInfo_Name(fInfo));
      SetTitle(gCling->MethodInfo_Title(fInfo));
      fMangledName = gCling->MethodInfo_GetMangledName(fInfo);
   }
}

//______________________________________________________________________________
TFunction::TFunction(const TFunction &orig) : TDictionary(orig)
{
   // Copy operator.

   if (orig.fInfo) {
      R__LOCKGUARD(gInterpreterMutex);
      fInfo = gCling->MethodInfo_FactoryCopy(orig.fInfo);
      fMangledName = orig.fMangledName;
   } else
      fInfo = 0;
   fMethodArgs = 0;
}

//______________________________________________________________________________
TFunction& TFunction::operator=(const TFunction &rhs)
{
   // Assignment operator.

   if (this != &rhs) {
      R__LOCKGUARD(gInterpreterMutex);
      gCling->MethodInfo_Delete(fInfo);
      if (fMethodArgs) fMethodArgs->Delete();
      delete fMethodArgs;
      if (rhs.fInfo) {
         fInfo = gCling->MethodInfo_FactoryCopy(rhs.fInfo);
         SetName(gCling->MethodInfo_Name(fInfo));
         SetTitle(gCling->MethodInfo_Title(fInfo));
         fMangledName = gCling->MethodInfo_GetMangledName(fInfo);
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

   R__LOCKGUARD(gInterpreterMutex);
   gCling->MethodInfo_Delete(fInfo);

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

   R__LOCKGUARD(gInterpreterMutex);
   gCling->MethodInfo_CreateSignature(fInfo, fSignature);
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

   if (!fMethodArgs && fInfo) {
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

   R__LOCKGUARD2(gInterpreterMutex);
   if (fInfo == 0 || gCling->MethodInfo_Type(fInfo) == 0) return "Unknown";
   return gCling->MethodInfo_TypeName(fInfo);
}

//______________________________________________________________________________
std::string TFunction::GetReturnTypeNormalizedName() const
{
   // Get the normalized name of the return type.  A normalized name is fully
   // qualified and has all typedef desugared except for the 'special' typedef
   // which include Double32_t, Float16_t, [U]Long64_t and std::string.  It
   // also has std:: removed [This is subject to change].
   //

   R__LOCKGUARD2(gInterpreterMutex);
   if (fInfo == 0 || gCling->MethodInfo_Type(fInfo) == 0) return "Unknown";
   return gCling->MethodInfo_TypeNormalizedName(fInfo);
}

//______________________________________________________________________________
Int_t TFunction::GetNargs() const
{
   // Number of function arguments.

   if (fInfo) return gCling->MethodInfo_NArg(fInfo);
   else if (fMethodArgs) return fMethodArgs->GetEntries();
   else return 0;
}

//______________________________________________________________________________
Int_t TFunction::GetNargsOpt() const
{
   // Number of function optional (default) arguments.

   // FIXME: when unload this is an over-estimate.
   return fInfo ? gCling->MethodInfo_NDefaultArg(fInfo) : GetNargs();
}

//______________________________________________________________________________
Long_t TFunction::Property() const
{
   // Get property description word. For meaning of bits see EProperty.

   return fInfo ? gCling->MethodInfo_Property(fInfo) : 0;
}

//______________________________________________________________________________
Long_t TFunction::ExtraProperty() const
{
   // Get property description word. For meaning of bits see EProperty.

   return fInfo ? gCling->MethodInfo_ExtraProperty(fInfo) : 0;
}

//______________________________________________________________________________
TDictionary::DeclId_t TFunction::GetDeclId() const
{
   return gInterpreter->GetDeclId(fInfo);
}

//______________________________________________________________________________
void *TFunction::InterfaceMethod() const
{
   // Return pointer to the interface method. Using this pointer we
   // can find which TFunction belongs to a CINT MethodInfo object.
   // Both need to have the same InterfaceMethod pointer.

   return fInfo ? gCling->MethodInfo_InterfaceMethod(fInfo) : 0;
}

//______________________________________________________________________________
Bool_t TFunction::IsValid()
{
   // Return true if this function object is pointing to a currently
   // loaded function.  If a function is unloaded after the TFunction
   // is created, the TFunction will be set to be invalid.

   // Register the transaction when checking the validity of the object.
   if (!fInfo && UpdateInterpreterStateMarker()) {
      // Only for global functions. For data member functions TMethod does it.
      DeclId_t newId = gInterpreter->GetFunction(0, fName);
      if (newId) {
         MethodInfo_t *info = gInterpreter->MethodInfo_Factory(newId);
         Update(info);
      }
      return newId != 0;
   }
   return fInfo != 0;
}

//______________________________________________________________________________
const char *TFunction::GetMangledName() const
{
   // Returns the mangled name as defined by CINT, or 0 in case of error.

   return fMangledName;
}

//______________________________________________________________________________
const char *TFunction::GetPrototype() const
{
   // Returns the prototype of a function as defined by CINT, or 0 in
   // case of error.

   if (fInfo) {
      R__LOCKGUARD2(gInterpreterMutex);
      return gCling->MethodInfo_GetPrototype(fInfo);
   } else
      return 0;
}

//______________________________________________________________________________
void TFunction::ls(Option_t *options /* ="" */) const
{
   // List TFunction name and title.

   TDictionary::ls(options);
   TROOT::IndentLevel();
   std::cout << "     " << GetPrototype() << '\n';
}

//______________________________________________________________________________
void TFunction::Print(Option_t *options /* ="" */) const
{
   // Print TFunction name and title.

   TDictionary::Print(options);
}

//______________________________________________________________________________
Bool_t TFunction::Update(MethodInfo_t *info)
{
   // Update the TFunction to reflect the new info.
   //
   // This can be used to implement unloading (info == 0) and then reloading
   // (info being the 'new' decl address).

   if (info == 0) {

      if (fInfo) {
         R__LOCKGUARD(gInterpreterMutex);
         gCling->MethodInfo_Delete(fInfo);
      }
      fInfo = 0;
      if (fMethodArgs) {
        for (Int_t i = 0; i < fMethodArgs->LastIndex() + 1; i ++) {
           TMethodArg *arg = (TMethodArg *) fMethodArgs->At( i );
           arg->Update(0);
        }
      }
      return kTRUE;
   } else {
      if (fInfo) {
         R__LOCKGUARD(gInterpreterMutex);
         gCling->MethodInfo_Delete(fInfo);
      }
      fInfo = info;
      TString newMangledName = gCling->MethodInfo_GetMangledName(fInfo);
      if (newMangledName != fMangledName) {
         Error("Update","TFunction object updated with the 'wrong' MethodInfo (%s vs %s).",
               fMangledName.Data(),newMangledName.Data());
         fInfo = 0;
         return false;
      }
      SetTitle(gCling->MethodInfo_Title(fInfo));
      if (fMethodArgs) {
         MethodArgInfo_t *arg = gCling->MethodArgInfo_Factory(fInfo);
         Int_t i = 0;
         R__LOCKGUARD(gInterpreterMutex);
         while (gCling->MethodArgInfo_Next(arg)) {
            if (gCling->MethodArgInfo_IsValid(arg)) {
               MethodArgInfo_t *new_arg = gCling->MethodArgInfo_FactoryCopy(arg);
               ((TMethodArg *) fMethodArgs->At( i ))->Update(new_arg);
               ++i;
            }
         }
      }
      return kTRUE;
   }
}
