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

ClassImp(TEnum)

//______________________________________________________________________________
TEnum::TEnum(const char* name, void* info, TClass* cls)
   : TNamed(name, "An enum type"), fInfo(info), fClass(cls)
{
   //Constructor for TEnum class.
   //It take the name of the TEnum type, specification if it is global
   //and interpreter info.
   //Constant List is owner if enum not on global scope (thus constants not
   //in TROOT::GetListOfGlobals).

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
void TEnum::Update(DeclId_t id)
{
   fInfo = (void*)id;
}
