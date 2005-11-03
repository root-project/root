// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/Builder/EnumBuilder.h"

#include "Reflex/Member.h"

#include "DataMember.h"
#include "Enum.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::EnumBuilderImpl::EnumBuilderImpl( const char * nam,
                                                const std::type_info & ti ) {
//-------------------------------------------------------------------------------
  fEnum = new Enum( nam, ti );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::EnumBuilderImpl::AddItem( const char * nam,
                                             long value ) {  
//-------------------------------------------------------------------------------
  fEnum->AddDataMember( Member(new DataMember( nam, 
                                               Type::ByName("int"), 
                                               value, 
                                               0 )));
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::EnumBuilderImpl::AddProperty( const char * key,
                                                 Any value ) {
//-------------------------------------------------------------------------------
  if ( fLastMember ) fLastMember.PropertyListGet().AddProperty( key , value );
  else                fEnum->PropertyListGet().AddProperty( key, value );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::EnumBuilderImpl::AddProperty( const char * key,
                                                 const char * value ) {
//-------------------------------------------------------------------------------
  AddProperty( key, Any(value));
}

