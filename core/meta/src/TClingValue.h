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
//  Class representing a value coming from cling. Its main use case           //
//  is to TCallFunc. When TCallFunc returns by-value, i.e. a temporary        //
//  variable, its lifetime has to be extended. TClingValue provides a         //
//  way to extend the temporaries lifetime and gives the user to control it.  //
//                                                                            //
//  The class is used to hide the implementation details of                   //
//  cling::StoredValueRef.                                                    //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TClingValue
#define ROOT_TClingValue

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_TInterpreterValue
#include "TInterpreterValue.h"
#endif

class TClingValue : public TInterpreterValue {
private:
   void* fValue;
public:
   TClingValue();
   TClingValue(const TClingValue& Other);
   TClingValue& operator=(TClingValue &Other);
   ~TClingValue();

   void* const& Get() const { return fValue; }
   void*& Get() { return fValue; }

   Bool_t   IsValid() const;
   Double_t GetAsDouble() const;
   Long_t   GetAsLong() const;
   ULong_t  GetAsUnsignedLong() const;
   void*    GetAsPointer() const;
};

#endif // ROOT_TClingValue
