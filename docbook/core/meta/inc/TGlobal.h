// @(#)root/meta:$Id$
// Author: Rene Brun   13/11/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGlobal
#define ROOT_TGlobal


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGlobal                                                              //
//                                                                      //
// Global variables class (global variables are obtained from CINT).    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TDictionary
#include "TDictionary.h"
#endif


class TGlobal : public TDictionary {

private:
   DataMemberInfo_t  *fInfo;      //pointer to CINT data member info

public:
   TGlobal(DataMemberInfo_t *info = 0);
   TGlobal (const TGlobal &);
   TGlobal &operator=(const TGlobal &);
   
   virtual       ~TGlobal();
   Int_t          GetArrayDim() const;
   Int_t          GetMaxIndex(Int_t dim) const;
   void          *GetAddress() const;
   const char    *GetTypeName() const;
   const char    *GetFullTypeName() const;
   Long_t         Property() const;

   ClassDef(TGlobal,0)  //Global variable class
};

#endif
