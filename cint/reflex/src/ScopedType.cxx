// @(#)root/reflex:$Id$
// Author: Axel Naumannm 2009

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

#include "ScopedType.h"

//-------------------------------------------------------------------------------
Reflex::ScopedType::ScopedType(const char* name, size_t size,
                               TYPE typeType, const std::type_info& ti,
                               const Type& finalType, unsigned int modifiers,
                               REPRESTYPE represType)
//-------------------------------------------------------------------------------
// Construct the dictionary information for an enum
   : TypeBase(name, size, typeType, ti, finalType, represType),
   ScopeBase(name, typeType),
   fModifiers(modifiers) {
}
