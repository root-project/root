// @(#)root/reflex:$Name:  $:$Id: Pointer.cxx,v 1.10 2006/08/02 13:25:33 roiser Exp $
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

#include "Pointer.h"

#include "Reflex/internal/OwnedMember.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::Pointer::Pointer( const Type & pointerType,
                                const std::type_info & ti )
//-------------------------------------------------------------------------------
   : TypeBase( BuildTypeName(pointerType).c_str(), sizeof(void*), POINTER, ti ), 
     fPointerType( pointerType ) { 
   // Construct the dictionary info for a pointer type.
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::Pointer::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
// Return the name of the pointer type.
   return BuildTypeName( fPointerType, mod );
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::Pointer::BuildTypeName( const Type & pointerType,
                                                  unsigned int mod ) {
//-------------------------------------------------------------------------------
// Build the pointer type name.
   return pointerType.Name( mod ) + "*";
}
