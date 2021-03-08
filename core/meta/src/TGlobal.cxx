// @(#)root/meta:$Id$
// Author: Rene Brun   13/11/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TGlobal
Global variables class (global variables are obtained from CINT).
This class describes the attributes of a global variable.
The TROOT class contains a list of all currently defined global
variables (accessible via TROOT::GetListOfGlobals()).
*/

#include "TGlobal.h"
#include "TInterpreter.h"
#include "TList.h"
#include "TROOT.h"


ClassImp(TGlobal);

////////////////////////////////////////////////////////////////////////////////
/// Default TGlobal ctor.

TGlobal::TGlobal(DataMemberInfo_t *info) : TDictionary(), fInfo(info)
{
   if (fInfo) {
      SetName(gCling->DataMemberInfo_Name(fInfo));
      SetTitle(gCling->DataMemberInfo_Title(fInfo));
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

TGlobal::TGlobal(const TGlobal &rhs) : TDictionary( ), fInfo(nullptr)
{
   if (rhs.fInfo) {
      fInfo = gCling->DataMemberInfo_FactoryCopy(rhs.fInfo);
      SetName(gCling->DataMemberInfo_Name(fInfo));
      SetTitle(gCling->DataMemberInfo_Title(fInfo));
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator.

TGlobal &TGlobal::operator=(const TGlobal &rhs)
{
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

////////////////////////////////////////////////////////////////////////////////
/// TGlobal dtor deletes adopted CINT DataMemberInfo object.

TGlobal::~TGlobal()
{
   if (fInfo && gCling) gCling->DataMemberInfo_Delete(fInfo);
}

////////////////////////////////////////////////////////////////////////////////
/// Return address of global.

void *TGlobal::GetAddress() const
{
   return (void *)gCling->DataMemberInfo_Offset(fInfo);
}

////////////////////////////////////////////////////////////////////////////////
/// Return number of array dimensions.

Int_t TGlobal::GetArrayDim() const
{
   if (!fInfo) return 0;
   return gCling->DataMemberInfo_ArrayDim(fInfo);
}

////////////////////////////////////////////////////////////////////////////////

TDictionary::DeclId_t TGlobal::GetDeclId() const
{
   return gInterpreter->GetDeclId(fInfo);
}

////////////////////////////////////////////////////////////////////////////////
/// Return maximum index for array dimension "dim".

Int_t TGlobal::GetMaxIndex(Int_t dim) const
{
   if (!fInfo) return 0;
   return gCling->DataMemberInfo_MaxIndex(fInfo,dim);
}

////////////////////////////////////////////////////////////////////////////////
/// Get type of global variable, e,g.: "class TDirectory*" -> "TDirectory".
/// Result needs to be used or copied immediately.

const char *TGlobal::GetTypeName() const
{
   if (!fInfo) return nullptr;
   return gCling->TypeName(gCling->DataMemberInfo_TypeName(fInfo));
}

////////////////////////////////////////////////////////////////////////////////
/// Get full type description of global variable, e,g.: "class TDirectory*".

const char *TGlobal::GetFullTypeName() const
{
   if (!fInfo) return nullptr;
   return gCling->DataMemberInfo_TypeName(fInfo);
}

////////////////////////////////////////////////////////////////////////////////
/// Return true if this global object is pointing to a currently
/// loaded global.  If a global is unloaded after the TGlobal
/// is created, the TGlobal will be set to be invalid.

Bool_t TGlobal::IsValid()
{
   // Register the transaction when checking the validity of the object.
   if (!fInfo && UpdateInterpreterStateMarker()) {
      DeclId_t newId = gInterpreter->GetDataMember(nullptr, fName);
      if (newId) {
         DataMemberInfo_t *info = gInterpreter->DataMemberInfo_Factory(newId, nullptr);
         Update(info);
      }
      return newId != nullptr;
   }
   return fInfo != nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Get property description word. For meaning of bits see EProperty.

Long_t TGlobal::Property() const
{
   if (!fInfo) return 0;
   return gCling->DataMemberInfo_Property(fInfo);
}

////////////////////////////////////////////////////////////////////////////////
/// Update the TFunction to reflect the new info.
///
/// This can be used to implement unloading (info == 0) and then reloading
/// (info being the 'new' decl address).

Bool_t TGlobal::Update(DataMemberInfo_t *info)
{
   if (fInfo) gCling->DataMemberInfo_Delete(fInfo);
   fInfo = info;
   if (fInfo) {
      SetName(gCling->DataMemberInfo_Name(fInfo));
      SetTitle(gCling->DataMemberInfo_Title(fInfo));
   }
   return kTRUE;
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor

TGlobalMappedFunction::TGlobalMappedFunction(const char *name, const char *type, GlobalFunc_t funcPtr)
   : fFuncPtr(funcPtr)
{
   SetNameTitle(name, type);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns list collected globals
/// Used to storeTGlobalMappedFunctions from other libs, before gROOT was initialized

TList& TGlobalMappedFunction::GetEarlyRegisteredGlobals()
{
   // Used to storeTGlobalMappedFunctions from other libs, before gROOT was initialized
   static TList fEarlyRegisteredGlobals;
   // For thread-safe setting of SetOwner(kTRUE).
   static bool earlyRegisteredGlobalsSetOwner
      = (fEarlyRegisteredGlobals.SetOwner(kTRUE), true);
   (void) earlyRegisteredGlobalsSetOwner; // silence unused var

   return fEarlyRegisteredGlobals;
}

////////////////////////////////////////////////////////////////////////////////
/// Add to GetEarlyRegisteredGlobals() if gROOT is not yet initialized; add to
/// gROOT->GetListOfGlobals() otherwise.

void TGlobalMappedFunction::Add(TGlobalMappedFunction* gmf)
{
   // Use "gCling" as synonym for "gROOT is initialized"
   if (gCling) {
      gROOT->GetListOfGlobals()->Add(gmf);
   } else {
      GetEarlyRegisteredGlobals().Add(gmf);
   }
}
