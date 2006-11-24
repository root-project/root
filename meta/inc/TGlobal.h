// @(#)root/meta:$Name:  $:$Id: TGlobal.h,v 1.3 2002/11/26 10:24:09 brun Exp $
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

namespace Cint {
class G__DataMemberInfo;
}
using namespace Cint;


class TGlobal : public TDictionary {

private:
   G__DataMemberInfo  *fInfo;      //pointer to CINT data member info

public:
   TGlobal(G__DataMemberInfo *info = 0);
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
