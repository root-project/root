// @(#)root/reflex:$Name:  $:$Id: TypedefBuilder.cxx,v 1.6 2006/03/13 15:49:51 roiser Exp $
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

#include "Reflex/Builder/TypedefBuilder.h"

#include "Typedef.h"


//-------------------------------------------------------------------------------
ROOT::Reflex::TypedefBuilderImpl::TypedefBuilderImpl( const char * typ,
                                                      const Type & typedefType ) {
//-------------------------------------------------------------------------------
   fTypedef = new Typedef( typ, typedefType );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::TypedefBuilderImpl::AddProperty( const char * key,
                                                    Any value ) {
//-------------------------------------------------------------------------------
   fTypedef->Properties().AddProperty( key, value );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::TypedefBuilderImpl::AddProperty( const char * key,
                                                    const char * value ) {
//-------------------------------------------------------------------------------
   AddProperty( key, Any(value));
}

