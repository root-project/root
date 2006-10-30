// @(#)root/reflex:$Name:  $:$Id: Union.cxx,v 1.9 2006/08/01 09:14:33 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
#define REFLEX_BUILD
#endif

#include "Union.h"

#include "Reflex/Tools.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::Union::Union( const char * unionType,
                            size_t size,
                            const std::type_info & ti,
                            unsigned int modifiers ) 
//-------------------------------------------------------------------------------
   : TypeBase( unionType, size, UNION, ti ),
     ScopeBase( unionType, UNION),
     fModifiers( modifiers ) {
   // Construct union info.
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Union::~Union() {
//-------------------------------------------------------------------------------
// Destructor.
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member ROOT::Reflex::Union::MemberAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return nth member of this union.
   return ScopeBase::MemberAt( nth );
}
