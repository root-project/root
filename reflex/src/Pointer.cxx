// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Pointer.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::Pointer::Pointer( const Type &           pointerType,
                                const std::type_info & ti )
//-------------------------------------------------------------------------------
  : TypeBase( BuildTypeName(pointerType).c_str(), sizeof(void*), POINTER, ti ), 
    fPointerType( pointerType ) { }


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::Pointer::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
  return BuildTypeName( fPointerType, mod );
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::Pointer::BuildTypeName( const Type & pointerType,
                                                  unsigned int mod ) {
//-------------------------------------------------------------------------------
  return pointerType.Name( mod ) + "*";
}
