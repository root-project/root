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
// TClingValue                                                                //
//                                                                            //
// Bridge between cling::StoredValueRef and ROOT.                             //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "TClingValue.h"

#include "cling/Interpreter/StoredValueRef.h"

static const cling::StoredValueRef& GetAsStoredValueRef(void* const& value) {
   return reinterpret_cast<const cling::StoredValueRef&>(value);
}

static cling::StoredValueRef& GetAsStoredValueRef(void*& value) {
   return reinterpret_cast<cling::StoredValueRef&>(value);
}

TClingValue::TClingValue() : fValue(0) {
   // We default initialize to invalid value so that we could keep a "sane" state.
   new (&fValue) cling::StoredValueRef();
}

TClingValue::TClingValue(const TClingValue& Other) : TInterpreterValue(), fValue(0) {
   using namespace cling;
   new (&fValue) StoredValueRef(GetAsStoredValueRef(Other.fValue));
}

TClingValue::~TClingValue() {
   GetAsStoredValueRef(fValue).~StoredValueRef();
}

TClingValue& TClingValue::operator=(TClingValue &Other) {
   using namespace cling;
   StoredValueRef& That = reinterpret_cast<StoredValueRef&>(Other.fValue);
   GetAsStoredValueRef(fValue) = (GetAsStoredValueRef(fValue).operator=(That));
   return *this;
}

Bool_t TClingValue::IsValid() const {
   return GetAsStoredValueRef(fValue).isValid();
}

Double_t TClingValue::GetAsDouble() const {
   return GetAsStoredValueRef(fValue).get().getAs<double>();
}

Long_t TClingValue::GetAsLong() const {
   return GetAsStoredValueRef(fValue).get().getAs<long>();
}

 ULong_t TClingValue::GetAsUnsignedLong() const {
   return GetAsStoredValueRef(fValue).get().getAs<unsigned long>();
}

void* TClingValue::GetAsPointer() const {
   return GetAsStoredValueRef(fValue).get().getAs<void*>();
}
