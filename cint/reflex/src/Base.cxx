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

#include "Reflex/Base.h"
#include "Class.h"

//-------------------------------------------------------------------------------
Reflex::Base::Base(const Type& baseType,
                   OffsetFunction offsetfp,
                   unsigned int modifiers):
   fOffsetFP(offsetfp),
   fModifiers(modifiers),
   fBaseType(Type()),
   fBaseClass(0) {
//-------------------------------------------------------------------------------
// Construct the information for a base. The pointer to the base class (type Class)
// is set to 0 initially and set on first access.
   fBaseType = baseType;
}


//-------------------------------------------------------------------------------
const Reflex::Class*
Reflex::Base::BaseClass() const {
//-------------------------------------------------------------------------------
// Return the pointer to the base class. Set on first access.
   if (fBaseClass) {
      return fBaseClass;
   }

   if (fBaseType) {
      fBaseClass = dynamic_cast<const Class*>(fBaseType.ToTypeBase());
      return fBaseClass;
   }
   return 0;
}


//-------------------------------------------------------------------------------
std::string
Reflex::Base::Name(unsigned int mod) const {
//-------------------------------------------------------------------------------
// Construct the name of the base. Qualify if requested.
   std::string s;

   if (0 != (mod & (QUALIFIED | Q))) {
      if (IsPublic()) {
         s += "public ";
      }

      if (IsProtected()) {
         s += "protected ";
      }

      if (IsPrivate()) {
         s += "private ";
      }

      if (IsVirtual()) {
         s += "virtual ";
      }
   }
   s += fBaseType.Name(mod);
   return s;
} // Name
