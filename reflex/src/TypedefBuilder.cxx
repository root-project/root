// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/Builder/TypedefBuilder.h"

#include "Typedef.h"


//-------------------------------------------------------------------------------
ROOT::Reflex::TypedefBuilderImpl::TypedefBuilderImpl( const char * TypeNth,
                                                      const Type & typedefType ) {
//-------------------------------------------------------------------------------
  fTypedef = new Typedef( TypeNth, typedefType );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::TypedefBuilderImpl::AddProperty( const char * key,
                                                    Any value ) {
//-------------------------------------------------------------------------------
  fTypedef->PropertyListGet().AddProperty( key, value );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::TypedefBuilderImpl::AddProperty( const char * key,
                                                    const char * value ) {
//-------------------------------------------------------------------------------
  AddProperty( key, Any(value));
}

