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
// THe TEnumConstnt class implements the constants of the enum type.    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TEnumConstant.h"
#include "TEnum.h"


ClassImp(TEnumConstant)
//______________________________________________________________________________
TEnumConstant::TEnumConstant(DataMemberInfo_t *info, Long64_t value, TEnum* type)
{
	//Constructor of the TEnumConstant.
	//Takes as parameters DataMemeberInfo, value, and enum type.

	fValue = value;
	fEnum = type;
	fDataMemberInfo_t = info;

	//add teh constant to the enum type
	type->AddConstant(this);
}

//______________________________________________________________________________
TEnumConstant::~TEnumConstant()
{
	//Destructor
}

//______________________________________________________________________________
