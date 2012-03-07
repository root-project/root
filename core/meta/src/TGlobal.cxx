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


ClassImp(TGlobal)

//______________________________________________________________________________
TGlobal::TGlobal(DataMemberInfo_t *info) : TDictionary(), fInfo(info)
{
   // Default TGlobal ctor. TGlobals are constructed in TROOT via
   // a call to TCint::UpdateListOfGlobals().

   if (fInfo) {
      SetName(gCint->DataMemberInfo_Name(fInfo));
      SetTitle(gCint->DataMemberInfo_Title(fInfo));
   }
}

//______________________________________________________________________________
TGlobal::TGlobal(const TGlobal &rhs) : TDictionary( ), fInfo(0)
{
   // Copy constructor
   
   if (rhs.fInfo) {
      fInfo = gCint->DataMemberInfo_FactoryCopy(rhs.fInfo);
      SetName(gCint->DataMemberInfo_Name(fInfo));
      SetTitle(gCint->DataMemberInfo_Title(fInfo));
   }
}

//______________________________________________________________________________
TGlobal &TGlobal::operator=(const TGlobal &rhs)
{
   // Assignment operator.
   
   if (this != &rhs) {
      gCint->DataMemberInfo_Delete(fInfo);
      if (rhs.fInfo) {
         fInfo = gCint->DataMemberInfo_FactoryCopy(rhs.fInfo);
         SetName(gCint->DataMemberInfo_Name(fInfo));
         SetTitle(gCint->DataMemberInfo_Title(fInfo));
      }
   }
   return *this;   
}

//______________________________________________________________________________
TGlobal::~TGlobal()
{
   // TGlobal dtor deletes adopted CINT DataMemberInfo object.

   gCint->DataMemberInfo_Delete(fInfo);
}

//______________________________________________________________________________
void *TGlobal::GetAddress() const
{
   // Return address of global.

   return (void *)gCint->DataMemberInfo_Offset(fInfo);
}

//______________________________________________________________________________
Int_t TGlobal::GetArrayDim() const
{
   // Return number of array dimensions.

   if (!fInfo) return 0;
   return gCint->DataMemberInfo_ArrayDim(fInfo);
}

//______________________________________________________________________________
Int_t TGlobal::GetMaxIndex(Int_t dim) const
{
   // Return maximum index for array dimension "dim".

   if (!fInfo) return 0;
   return gCint->DataMemberInfo_MaxIndex(fInfo,dim);
}

//______________________________________________________________________________
const char *TGlobal::GetTypeName() const
{
   // Get type of global variable, e,g.: "class TDirectory*" -> "TDirectory".
   // Result needs to be used or copied immediately.

   if (!fInfo) return 0;
   return gCint->TypeName(gCint->DataMemberInfo_TypeName(fInfo));
}

//______________________________________________________________________________
const char *TGlobal::GetFullTypeName() const
{
   // Get full type description of global variable, e,g.: "class TDirectory*".

   if (!fInfo) return 0;
   return gCint->DataMemberInfo_TypeName(fInfo);
}

//______________________________________________________________________________
Long_t TGlobal::Property() const
{
   // Get property description word. For meaning of bits see EProperty.

   if (!fInfo) return 0;
   return gCint->DataMemberInfo_Property(fInfo);
}
