// @(#)root/meta:$Id$
// Author: Bianca-Cristina Cristescu   10/07/13

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TEnumConstant
The TEnumConstant class implements the constants of the enum type.
*/

#include "TEnumConstant.h"
#include "TEnum.h"


ClassImp(TEnumConstant);

////////////////////////////////////////////////////////////////////////////////
/// Constructor of the TEnumConstant.
/// Takes as parameters DataMemberInfo, value, and enum type.

TEnumConstant::TEnumConstant(DataMemberInfo_t *info, const char* name, Long64_t value, TEnum* type)
   : TGlobal(info), fEnum(type), fValue(value) {
   //Set name of constant
   this->SetName(name);

   // Add the constant to the enum type.
   type->AddConstant(this);
}

////////////////////////////////////////////////////////////////////////////////
///Destructor

TEnumConstant::~TEnumConstant()
{
}
