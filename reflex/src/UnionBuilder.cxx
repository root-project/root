// @(#)root/reflex:$Name:  $:$Id: UnionBuilder.cxx,v 1.4 2005/11/23 16:08:08 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/Builder/UnionBuilder.h"

#include "Reflex/Member.h"
#include "Reflex/Any.h"

#include "DataMember.h"
#include "Union.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::UnionBuilderImpl::UnionBuilderImpl( const char * nam,
                                                  size_t size,
                                                  const std::type_info & ti ) {
//-------------------------------------------------------------------------------
   fUnion = new Union( nam, size, ti );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::UnionBuilderImpl::AddItem( const char * nam,
                                              const Type & typ ) {
//-------------------------------------------------------------------------------
   fLastMember = Member(new DataMember( nam,
                                        typ,
                                        0,
                                        0 ));
   fUnion->AddDataMember( fLastMember );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::UnionBuilderImpl::AddProperty( const char * key,
                                                  Any value ) {
//-------------------------------------------------------------------------------
   if ( fLastMember ) fLastMember.Properties().AddProperty( key, value );
   else                fUnion->Properties().AddProperty(key, value );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::UnionBuilderImpl::AddProperty( const char * key,
                                                  const char * value ) {
//-------------------------------------------------------------------------------
   AddProperty( key, Any(value));
}
