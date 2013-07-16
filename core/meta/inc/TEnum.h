// @(#)root/meta:$Id$
// Author: Bianca-Cristina Cristescu   09/07/13

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEnum
#define ROOT_TEnum

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TEnum                                                                //
//                                                                      //
// TEnum class defines enum type.                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_THashList
#include "THashList.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TEnumConstant;

class TEnum : public TNamed {

private:
   THashList fConstantList;     //list of constants the enum type
   void*     fInfo;             //interpreter implementation provided declaration

public:

   TEnum(): fInfo(0) {}
   TEnum(const char* name, bool isGlobal, void* info);
   virtual ~TEnum();

   void AddConstant(TEnumConstant* constant);
   const TSeqCollection* GetConstants() const { return &fConstantList; }
   const TEnumConstant* GetConstant(const char* name) const {
      return (TEnumConstant*) fConstantList.FindObject(name);
   }

   ClassDef(TEnum,1)  //Enum type class
};

#endif
