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

#include "PointerToMember.h"

#include "Reflex/internal/OwnedMember.h"

//-------------------------------------------------------------------------------
Reflex::PointerToMember::PointerToMember(const Type& pointerToMemberType,
                                         const Scope& pointerToMemberScope,
                                         const std::type_info& ti)
//-------------------------------------------------------------------------------
   : TypeBase(BuildTypeName(pointerToMemberType, pointerToMemberScope).c_str(), sizeof(void*), POINTERTOMEMBER, ti, Type(), (REPRESTYPE) 'a'),
   fPointerToMemberType(pointerToMemberType),
   fPointerToMemberScope(pointerToMemberScope) {
   // Construct dictionary info for a pointer to member type.
}


//-------------------------------------------------------------------------------
std::string
Reflex::PointerToMember::Name(unsigned int mod) const {
//-------------------------------------------------------------------------------
// Return the name of the pointer to member type.
   return BuildTypeName(fPointerToMemberType, fPointerToMemberScope, mod);
}


//-------------------------------------------------------------------------------
std::string
Reflex::PointerToMember::BuildTypeName(const Type& pointerToMemberType,
                                       const Scope& pointerToMemberScope,
                                       unsigned int mod) {
//-------------------------------------------------------------------------------
// Build the pointer to member type name.
   if (pointerToMemberType.TypeType() == FUNCTION) {
      std::string nam = pointerToMemberType.ReturnType().Name(mod) + " (" +
                        pointerToMemberScope.Name(mod) + "::*)(";

      Type_Iterator lastbutone = pointerToMemberType.FunctionParameter_End() - 1;

      for (Type_Iterator ti = pointerToMemberType.FunctionParameter_Begin();
           ti != pointerToMemberType.FunctionParameter_End(); ++ti) {
         nam += (*ti).Name(mod);

         if (ti != lastbutone) {
            nam += ", ";
         }
      }
      nam += ")";
      return nam;

   }
   return pointerToMemberType.Name(mod) + " " + pointerToMemberScope.Name(mod) + " ::*";
} // BuildTypeName
