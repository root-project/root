// @(#)root/core/meta:$Id$e
// Author: Vassil Vassilev   13/03/2013

/*******************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.                     *
 * All rights reserved.                                                        *
 *                                                                             *
 * For the licensing terms see $ROOTSYS/LICENSE.                               *
 * For the list of contributors see $ROOTSYS/README/CREDITS.                   *
 ******************************************************************************/

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  Class representing a value came from the interpreter. Its main use case   //
//  is to TCallFunc. When TCallFunc returns by-value, ie a temporary          //
//  variable, its lifetime has to be extended. TInterpreterValue provides a   //
//  way to extend the temporaries lifetime and gives the user to control it.  //
//                                                                            //
//  The class is used to hide the implementation details of                   //
//  cling::StoredValueRef.                                                    //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TInterpreterValue
#define ROOT_TInterpreterValue

#include "Rtypes.h"

class TInterpreterValue {
private:
   void* fValue;
public:
   TInterpreterValue();
   TInterpreterValue(const TInterpreterValue& Other);
   TInterpreterValue& operator=(TInterpreterValue Other);
   ~TInterpreterValue();

   Bool_t   IsValid() const;
   Double_t GetAsDouble() const;
   Long_t   GetAsLong() const;
   ULong_t  GetAsUnsignedLong() const;
   void*    GetAsPointer() const;

   friend class TClingCallFunc;
};

#endif // ROOT_TInterpreterValue
