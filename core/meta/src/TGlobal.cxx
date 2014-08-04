// @(#)root/meta:$Id$
// Author: Rene Brun   13/11/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Global variables class (global variables are obtained from CINT).    //
// This class describes the attributes of a global variable.            //
// The TROOT class contains a list of all currently defined global      //
// variables (accessible via TROOT::GetListOfGlobals()).                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGlobal.h"
#include "TInterpreter.h"
#include "TList.h"
#include "TROOT.h"


ClassImp(TGlobal)

//______________________________________________________________________________
TGlobal::TGlobal(DataMemberInfo_t *info) : TDictionary(), fInfo(info)
{
   // Default TGlobal ctor.

   if (fInfo) {
      SetName(gCling->DataMemberInfo_Name(fInfo));
      SetTitle(gCling->DataMemberInfo_Title(fInfo));
   }
}

//______________________________________________________________________________
TGlobal::TGlobal(const TGlobal &rhs) : TDictionary( ), fInfo(0)
{
   // Copy constructor

   if (rhs.fInfo) {
      fInfo = gCling->DataMemberInfo_FactoryCopy(rhs.fInfo);
      SetName(gCling->DataMemberInfo_Name(fInfo));
      SetTitle(gCling->DataMemberInfo_Title(fInfo));
   }
}

//______________________________________________________________________________
TGlobal &TGlobal::operator=(const TGlobal &rhs)
{
   // Assignment operator.

   if (this != &rhs) {
      gCling->DataMemberInfo_Delete(fInfo);
      if (rhs.fInfo) {
         fInfo = gCling->DataMemberInfo_FactoryCopy(rhs.fInfo);
         SetName(gCling->DataMemberInfo_Name(fInfo));
         SetTitle(gCling->DataMemberInfo_Title(fInfo));
      }
   }
   return *this;
}

//______________________________________________________________________________
TGlobal::~TGlobal()
{
   // TGlobal dtor deletes adopted CINT DataMemberInfo object.

   gCling->DataMemberInfo_Delete(fInfo);
}

//______________________________________________________________________________
void *TGlobal::GetAddress() const
{
   // Return address of global.

   return (void *)gCling->DataMemberInfo_Offset(fInfo);
}

//______________________________________________________________________________
Int_t TGlobal::GetArrayDim() const
{
   // Return number of array dimensions.

   if (!fInfo) return 0;
   return gCling->DataMemberInfo_ArrayDim(fInfo);
}

//______________________________________________________________________________
TDictionary::DeclId_t TGlobal::GetDeclId() const
{
   return gInterpreter->GetDeclId(fInfo);
}

//______________________________________________________________________________
Int_t TGlobal::GetMaxIndex(Int_t dim) const
{
   // Return maximum index for array dimension "dim".

   if (!fInfo) return 0;
   return gCling->DataMemberInfo_MaxIndex(fInfo,dim);
}

//______________________________________________________________________________
const char *TGlobal::GetTypeName() const
{
   // Get type of global variable, e,g.: "class TDirectory*" -> "TDirectory".
   // Result needs to be used or copied immediately.

   if (!fInfo) return 0;
   return gCling->TypeName(gCling->DataMemberInfo_TypeName(fInfo));
}

//______________________________________________________________________________
const char *TGlobal::GetFullTypeName() const
{
   // Get full type description of global variable, e,g.: "class TDirectory*".

   if (!fInfo) return 0;
   return gCling->DataMemberInfo_TypeName(fInfo);
}

//______________________________________________________________________________
Bool_t TGlobal::IsValid()
{
   // Return true if this global object is pointing to a currently
   // loaded global.  If a global is unloaded after the TGlobal
   // is created, the TGlobal will be set to be invalid.

   // Register the transaction when checking the validity of the object.
   if (!fInfo && UpdateInterpreterStateMarker()) {
      DeclId_t newId = gInterpreter->GetDataMember(0, fName);
      if (newId) {
         DataMemberInfo_t *info = gInterpreter->DataMemberInfo_Factory(newId, 0);
         Update(info);
      }
      return newId != 0;
   }
   return fInfo != 0;
}

//______________________________________________________________________________
Long_t TGlobal::Property() const
{
   // Get property description word. For meaning of bits see EProperty.

   if (!fInfo) return 0;
   return gCling->DataMemberInfo_Property(fInfo);
}

//______________________________________________________________________________
Bool_t TGlobal::Update(DataMemberInfo_t *info)
{
   // Update the TFunction to reflect the new info.
   //
   // This can be used to implement unloading (info == 0) and then reloading
   // (info being the 'new' decl address).

   if (fInfo) gCling->DataMemberInfo_Delete(fInfo);
   fInfo = info;
   if (fInfo) {
      SetName(gCling->DataMemberInfo_Name(fInfo));
      SetTitle(gCling->DataMemberInfo_Title(fInfo));
   }
   return kTRUE;
}

TList& TGlobalMappedFunction::GetEarlyRegisteredGlobals()
{
   // Used to storeTGlobalMappedFunctions from other libs, before gROOT was inistialized
   static TList fEarlyRegisteredGlobals;
   return fEarlyRegisteredGlobals;
}

void TGlobalMappedFunction::Add(TGlobalMappedFunction* gmf)
{
   // Add to GetEarlyRegisteredGlobals() if gROOT is not yet initialized; add to
   // gROOT->GetListOfGlobals() otherwise.

   // Use "gCling" as synonym for "gROOT is initialized"
   if (gCling) {
      gROOT->GetListOfGlobals()->Add(gmf);
   } else {
      GetEarlyRegisteredGlobals().Add(gmf);
   }
}
