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

#include "Function.h"

#include "Reflex/Tools.h"
#include "Reflex/internal/OwnedMember.h"

//-------------------------------------------------------------------------------
Reflex::Function::Function(const Type& retType,
                           const std::vector<Type>& parameters,
                           const std::type_info& ti,
                           TYPE functionType)
//-------------------------------------------------------------------------------
// Default constructor for a function type.
   : TypeBase(BuildTypeName(retType, parameters, QUALIFIED | SCOPED).c_str(), 0, functionType, ti, Type(), (REPRESTYPE) '1'),
   fParameters(parameters),
   fReturnType(retType),
   fModifiers(0) {
}


//-------------------------------------------------------------------------------
std::string
Reflex::Function::Name(unsigned int mod) const {
//-------------------------------------------------------------------------------
// Return the name of the function type.
   return BuildTypeName(fReturnType, fParameters, mod);
}


//-------------------------------------------------------------------------------
std::string
Reflex::Function::BuildTypeName(const Type& ret,
                                const std::vector<Type>& pars,
                                unsigned int mod) {
//-------------------------------------------------------------------------------
// Build the name of the function type in the form <returntype><space>(<param>*)
   std::string tyname = ret.Name(mod) + " (";

   if (pars.size() > 0) {
      std::vector<Type>::const_iterator it;

      for (it = pars.begin(); it != pars.end();) {
         tyname += it->Name(mod);

         if (++it != pars.end()) {
            tyname += ", ";
         }
      }
   } else {
      tyname += "void";
   }
   tyname += ")";
   return tyname;
} // BuildTypeName
