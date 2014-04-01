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
// Bridge between cling::Value and ROOT.                                      //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "TClingValue.h"

#include "cling/Interpreter/Value.h"
#include <cassert>

TClingValue::TClingValue() {
   // We default initialize to invalid value to keep a "sane" state.
   assert(sizeof(fValue) >= sizeof(cling::Value)
          && "sizeof(fValue) too small!");
   new (&fValue) cling::Value();
}

TClingValue::TClingValue(const TClingValue& Other):
   TInterpreterValue() {
   assert(sizeof(fValue) >= sizeof(cling::Value)
          && "sizeof(fValue) too small!");
   new (&fValue) cling::Value(Other.ToCV());
}

TClingValue::~TClingValue() {
   ToCV().~Value();
}

TClingValue& TClingValue::operator=(TClingValue &Other) {
   using namespace cling;
   ToCV() = Other.ToCV();
   return *this;
}

Bool_t TClingValue::IsValid() const {
   return ToCV().isValid();
}

Double_t TClingValue::GetAsDouble() const {
   return ToCV().getDouble();
}

Long_t TClingValue::GetAsLong() const {
   return ToCV().getLL();
}

ULong_t TClingValue::GetAsUnsignedLong() const {
   return ToCV().getULL();
}

void* TClingValue::GetAsPointer() const {
   return ToCV().getPtr();
}
