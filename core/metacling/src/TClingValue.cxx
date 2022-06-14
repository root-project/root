// @(#)root/core/meta:$Id$
// Author: Vassil Vassilev   14/03/2013

/*******************************************************************************
* Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.                     *
 * All rights reserved.                                                        *
 *                                                                             *
 * For the licensing terms see $ROOTSYS/LICENSE.                               *
 * For the list of contributors see $ROOTSYS/README/CREDITS.                   *
 ******************************************************************************/

/** \class TClingValue
Bridge between cling::Value and ROOT.
*/

#include "TClingValue.h"

#include "cling/Interpreter/Value.h"
#include "llvm/Support/raw_ostream.h"

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

std::pair<std::string, std::string> TClingValue::ToTypeAndValueString() const {
  std::string output = ToString();
  int paren_level = 0;

  for (size_t pos = 0; pos < output.size(); ++pos) {
    if (output[pos] == '(')
      ++paren_level;
    else if (output[pos] == ')') {
      --paren_level;
      if (!paren_level)
        return std::make_pair(output.substr(0, pos + 1), output.substr(pos + 2));
    }
  }

  return std::make_pair("", output);
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

std::string TClingValue::ToString() const {
   std::string retVal;
   llvm::raw_string_ostream ost(retVal);
   ToCV().print(ost);
   return ost.str();
}
