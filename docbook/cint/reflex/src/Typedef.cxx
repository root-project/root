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

#include "Typedef.h"

#include "Reflex/Tools.h"
#include "Reflex/internal/OwnedMember.h"

//-------------------------------------------------------------------------------
Reflex::Typedef::Typedef(const char* typ,
                         const Type& typedefType,
                         TYPE typeTyp,
                         const Type& finalType,
                         REPRESTYPE represType)
//-------------------------------------------------------------------------------
   : TypeBase(typ, typedefType.SizeOf(), typeTyp, typeid(UnknownType), finalType,
              represType != REPRES_NOTYPE
              ? represType
              : (REPRESTYPE)(((typedefType.RepresType() == 'y') && strchr(typ, '('))
                             ? (REPRESTYPE) '1'
                             : typedefType.RepresType())),
     fTypedefType(typedefType) {
   // Construct typedef info.

   Type current = typedefType;

   while (current.IsTypedef())
      current = current.ToType();

   if (current.TypeInfo() != typeid(UnknownType)) {
      fTypeInfo = &current.TypeInfo();
   }
}


//-------------------------------------------------------------------------------
Reflex::Typedef::~Typedef() {
//-------------------------------------------------------------------------------
// Destructor.
}
