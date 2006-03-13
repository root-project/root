// @(#)root/reflex:$Name:  $:$Id: Array.cxx,v 1.4 2006/03/06 12:51:46 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#define REFLEX_BUILD

#include "Array.h"
#include "Reflex/Type.h"
#include <sstream>

//-------------------------------------------------------------------------------
ROOT::Reflex::Array::Array( const Type & arrayType,
                            size_t len,
                            const std::type_info & typeinfo ) 
//-------------------------------------------------------------------------------
   : TypeBase( BuildTypeName(arrayType, len ).c_str(), 
               len*(arrayType.SizeOf()), ARRAY, typeinfo ), 
     fArrayType( arrayType ), 
     fLength( len ) { }


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::Array::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
   return BuildTypeName( fArrayType, fLength, mod );
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::Array::BuildTypeName( const Type & typ, 
                                                size_t len,
                                                unsigned int mod ) {
//-------------------------------------------------------------------------------
   std::ostringstream ost;
   ost << typ.Name( mod ) << "[" << len << "]";
   return ost.str();
}
