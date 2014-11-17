// @(#)root/meta:$Id$
// Author: Bianca-Cristina Cristescu   09/07/13

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEnumConstant
#define ROOT_TEnumConstant


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TEnumConstant                                                        //
//                                                                      //
// TEnumConstant class defines a constant in the TEnum type.            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGlobal
#include "TGlobal.h"
#endif
#ifndef ROOT_TEnum
#include "TEnum.h"
#endif

class TEnum;

class TEnumConstant : public TGlobal {
private:
   const TEnum       *fEnum;  //the enum type
   Long64_t           fValue; //the value for the constant

public:
   TEnumConstant(): fEnum(0), fValue(-1) {}
   TEnumConstant(DataMemberInfo_t *info, const char* name, Long64_t value, TEnum* type);
   virtual ~TEnumConstant();

   void *GetAddress() const override { auto valPtr = &fValue; return (void*) const_cast<Long64_t*>(valPtr); }
   Long64_t      GetValue() const { return fValue; }
   const TEnum  *GetType() const { return fEnum; }

   const char *GetTypeName() const override { return fEnum->GetQualifiedName(); }
   const char *GetFullTypeName() const override { return GetTypeName(); }

   ClassDefOverride(TEnumConstant,2)  //Enum type constant
};

#endif
