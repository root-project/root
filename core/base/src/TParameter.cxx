// @(#)root/base:$Id$
// Author: Maarten Ballintijn   21/06/2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TParameter<AParamType>                                               //
//                                                                      //
// Named parameter, streamable and storable.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TParameter.h"


templateClassImp(TParameter)

// Specialization of Merge for Bool_t to make windows happy  
template <>
Int_t TParameter<Bool_t>::Merge(TCollection *in)
{
   // Merge objects in the list.
   // Returns the number of objects that were in the list.
   TIter nxo(in);
   Int_t n = 0;
   while (TObject *o = nxo()) {
      TParameter<Bool_t> *c = dynamic_cast<TParameter<Bool_t> *>(o);
      if (c) {
         if (TestBit(TParameter::kMultiply))
            fVal *= c->GetVal();
         else
            fVal += c->GetVal();
         n++;
      }
   }

   return n;
}
