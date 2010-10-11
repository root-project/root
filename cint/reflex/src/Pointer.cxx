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

#include "Pointer.h"

#include "Reflex/internal/OwnedMember.h"

#include <cctype>

using namespace std;

//-------------------------------------------------------------------------------
Reflex::Pointer::Pointer(const Type& pointerType,
                         const std::type_info& ti)
//-------------------------------------------------------------------------------
   : TypeBase(BuildTypeName(pointerType).c_str(), sizeof(void*), POINTER, ti, Type(),
              (REPRESTYPE) toupper(pointerType.RepresType() != REPRES_NOTYPE
                                   ? pointerType.RepresType()
                                   : Reflex::REPRES_CLASS)),
   fPointerType(pointerType) {
   // Construct the dictionary info for a pointer type.
}


//-------------------------------------------------------------------------------
std::string
Reflex::Pointer::Name(unsigned int mod) const {
//-------------------------------------------------------------------------------
// Return the name of the pointer type.
   return BuildTypeName(fPointerType, mod);
}


//-------------------------------------------------------------------------------
std::string
Reflex::Pointer::BuildTypeName(const Type& pointerType,
                               unsigned int mod) {
//-------------------------------------------------------------------------------
// Build the pointer type name.
   if (!pointerType.IsFunction()) {
      return pointerType.Name(mod) + "*";
   }
   // function pointer and pointer to function members
   else {
      std::string s = pointerType.ReturnType().Name(mod);
      s += " (";
      const Scope& decl = pointerType.DeclaringScope();

      if (decl) {
         s += decl.Name(SCOPED) + ":: ";
      }
      s += "*)(";

      if (pointerType.FunctionParameterSize()) {
         Type_Iterator pend = pointerType.FunctionParameter_End();

         for (Type_Iterator ti = pointerType.FunctionParameter_Begin();
              ti != pend;) {
            s += ti->Name(mod);

            if (++ti != pend) {
               s += ",";
            }
         }
      } else { s += "void"; }
      s += ")";
      return s;
   }
} // BuildTypeName
