// @(#)root/reflex:$Name: HEAD $:$Id: PointerToMember.cxx,v 1.8 2006/07/04 15:02:55 roiser Exp $
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

#include "PointerToMember.h"
#include "Reflex/Member.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::PointerToMember::PointerToMember( const Type & pointerToMemberType,
                                                const std::type_info & ti ) 
//------------------------------------------------------------------------------- 
   : TypeBase( BuildTypeName( pointerToMemberType ).c_str(), sizeof(void*), POINTERTOMEMBER, ti ),
     fPointerToMemberType( pointerToMemberType ) {
   // Construct dictionary info for a pointer to member type.
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::PointerToMember::Name( unsigned int mod ) const { 
//-------------------------------------------------------------------------------
// Return the name of the pointer to member type.
   return BuildTypeName( fPointerToMemberType, mod );
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::PointerToMember::BuildTypeName( const Type & pointerToMemberType,
                                                          unsigned int mod ) {
//-------------------------------------------------------------------------------
// Build the pointer to member type name.
   return pointerToMemberType.Name( mod ) + " ::*";
}
                                                          
