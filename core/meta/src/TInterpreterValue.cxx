// @(#)root/core/meta:$Id$
// Author: Vassil Vassilev   14/03/2013

/*******************************************************************************
* Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.                     *
 * All rights reserved.                                                        *
 *                                                                             *
 * For the licensing terms see $ROOTSYS/LICENSE.                               *
 * For the list of contributors see $ROOTSYS/README/CREDITS.                   *
 ******************************************************************************/

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// TInterpreterValue                                                          //
//                                                                            //
// Bridge between cling::StoredValueRef and ROOT.                             //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "TInterpreterValue.h"

#include "cling/Interpreter/StoredValueRef.h"

static cling::StoredValueRef& GetAsStoredValueRef(void* value) {
   return reinterpret_cast<cling::StoredValueRef&>(value);
}

TInterpreterValue::TInterpreterValue() {
   // We default initialize to invalid value so that we could keep a "sane" 
   // state.
   new (&fValue) cling::StoredValueRef();
}

TInterpreterValue::TInterpreterValue(const TInterpreterValue& Other) {
   using namespace cling;
   new (&fValue) StoredValueRef(GetAsStoredValueRef(Other.fValue));
}

TInterpreterValue::~TInterpreterValue() {
   GetAsStoredValueRef(fValue).~StoredValueRef();
}

TInterpreterValue& TInterpreterValue::operator=(TInterpreterValue Other) {
   using namespace cling;
   StoredValueRef& That = reinterpret_cast<StoredValueRef&>(Other.fValue);
   GetAsStoredValueRef(fValue) = (GetAsStoredValueRef(fValue).operator=(That));
   return *this;
}

Bool_t TInterpreterValue::IsValid() const {
   return GetAsStoredValueRef(fValue).isValid();
}

Double_t TInterpreterValue::GetAsDouble() const {
   return GetAsStoredValueRef(fValue).get().getAs<double>();
}

Long_t TInterpreterValue::GetAsLong() const {
   return GetAsStoredValueRef(fValue).get().getAs<long>();
}

 ULong_t TInterpreterValue::GetAsUnsignedLong() const {
   return GetAsStoredValueRef(fValue).get().getAs<unsigned long>();
}

void* TInterpreterValue::GetAsPointer() const {
   return GetAsStoredValueRef(fValue).get().getAs<void*>();
}
