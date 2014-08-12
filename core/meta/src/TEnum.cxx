// @(#)root/meta:$Id$
// Author: Bianca-Cristina Cristescu   10/07/13

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// The TEnum class implements the enum type.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TEnum.h"
#include "TEnumConstant.h"
#include "TInterpreter.h"


ClassImp(TEnum)

//______________________________________________________________________________
TEnum::TEnum(const char* name, void* info, TClass* cls)
   :fInfo(info), fClass(cls)
{
   //Constructor for TEnum class.
   //It take the name of the TEnum type, specification if it is global
   //and interpreter info.
   //Constant List is owner if enum not on global scope (thus constants not
   //in TROOT::GetListOfGlobals).
   SetNameTitle(name, "An enum type");
   if (cls) {
      fConstantList.SetOwner(kTRUE);
   }
}

//______________________________________________________________________________
TEnum::~TEnum()
{
   //Destructor
}

//______________________________________________________________________________
void TEnum::AddConstant(TEnumConstant* constant)
{
   //Add a EnumConstant to the list of constants of the Enum Type.
   fConstantList.Add(constant);
}

//______________________________________________________________________________
Bool_t TEnum::IsValid()
{
   // Return true if this enum object is pointing to a currently
   // loaded enum.  If a enum is unloaded after the TEnum
   // is created, the TEnum will be set to be invalid.

   // Register the transaction when checking the validity of the object.
   if (!fInfo && UpdateInterpreterStateMarker()) {
      DeclId_t newId = gInterpreter->GetEnum(fClass, fName);
      if (newId) {
         Update(newId);
      }
      return newId != 0;
   }
   return fInfo != 0;
}

//______________________________________________________________________________
Long_t TEnum::Property() const
{
   // Get property description word. For meaning of bits see EProperty.

   return kIsEnum;
}

//______________________________________________________________________________
void TEnum::Update(DeclId_t id)
{
   fInfo = (void*)id;
}
