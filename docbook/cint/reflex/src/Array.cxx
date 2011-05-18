// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
# define REFLEX_BUILD
#endif

#include "Array.h"

#include "Reflex/Type.h"
#include "Reflex/internal/OwnedMember.h"

#include <sstream>

//-------------------------------------------------------------------------------
Reflex::Array::Array(const Type& arrayType,
                     size_t len,
                     const std::type_info& typeinfo)
//-------------------------------------------------------------------------------
// Constructs an array type.
   : TypeBase(BuildTypeName(arrayType, len).c_str(), arrayType.SizeOf() * len, ARRAY, typeinfo, Type(), arrayType.RepresType()),
   fArrayType(arrayType),
   fLength(len) {
}


//-------------------------------------------------------------------------------
std::string
Reflex::Array::Name(unsigned int mod) const {
//-------------------------------------------------------------------------------
// Return the name of the array type.
   return BuildTypeName(fArrayType, fLength, mod);
}


//-------------------------------------------------------------------------------
std::string
Reflex::Array::BuildTypeName(const Type& typ,
                             size_t len,
                             unsigned int mod) {
//-------------------------------------------------------------------------------
// Build an array type name.
   std::ostringstream ost;
   Type t = typ;
   ost << "[" << len << "]";

   while (t.IsArray()) {
      ost << "[" << t.ArrayLength() << "]";
      t = t.ToType();
   }
   return t.Name(mod) + ost.str();
}
