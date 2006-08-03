// @(#)root/reflex:$Name:  $:$Id: Typedef.cxx,v 1.9 2006/07/13 14:45:59 roiser Exp $
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

#include "Typedef.h"

#include "Reflex/Tools.h"
#include "Reflex/internal/OwnedMember.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::Typedef::Typedef( const char * typ,
                                const Type & typedefType,
                                TYPE typeTyp )
//-------------------------------------------------------------------------------
   : TypeBase(typ, typedefType.SizeOf() , typeTyp, typeid(UnknownType)), //typedefType.TypeInfo()),
     fTypedefType(typedefType) { 
   // Construct typedef info.

   Type current = typedefType;
   while ( current.IsTypedef() ) current = current.ToType();
   if ( current.TypeInfo() != typeid(UnknownType)) fTypeInfo = & current.TypeInfo();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Typedef::~Typedef() {
//-------------------------------------------------------------------------------
// Destructor.
}
