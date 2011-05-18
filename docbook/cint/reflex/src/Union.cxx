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

#include "Union.h"

#include "Reflex/Tools.h"

//______________________________________________________________________________
Reflex::Union::Union(const char* typ, size_t size, const std::type_info& ti, unsigned int modifiers, TYPE unionType /*=UNION*/):
   ScopedType(typ, size, unionType, ti, Type(), modifiers, REPRES_STRUCT)
   ,
   fCompleteType(false) {
// Construct union info.
}


//______________________________________________________________________________
Reflex::Union::~Union() {
// Destructor.
}


//______________________________________________________________________________
bool
Reflex::Union::IsComplete() const {
   // Return true if this union is complete. I.e. all dictionary information for all
   // data and function member types is available.
   if (!fCompleteType) {
      fCompleteType = true;
   }
   return fCompleteType;
}


//______________________________________________________________________________
void
Reflex::Union::AddFunctionMember(const Member& fm) const {
// Add function member fm to this union
   ScopeBase::AddFunctionMember(fm);

   if (fm.IsConstructor()) {
      fConstructors.push_back(fm);
   } else if (fm.IsDestructor()) {
      fDestructor = fm;
   }
}


//______________________________________________________________________________
Reflex::Member
Reflex::Union::AddFunctionMember(const char* nam,
                                 const Type& typ,
                                 StubFunction stubFP,
                                 void* stubCtx,
                                 const char* params,
                                 unsigned int modifiers) const {
   // Add function member to this union.
   Member fm(ScopeBase::AddFunctionMember(nam, typ, stubFP, stubCtx, params, modifiers));

   if (fm.IsConstructor()) {
      fConstructors.push_back(fm);
   } else if (fm.IsDestructor()) {
      fDestructor = fm;
   }
   return fm;
}
