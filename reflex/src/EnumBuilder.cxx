// @(#)root/reflex:$Name:  $:$Id: EnumBuilder.cxx,v 1.6 2006/03/06 12:51:46 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#define REFLEX_BUILD

#include "Reflex/Builder/EnumBuilder.h"
#include "Reflex/Member.h"
#include "Reflex/Callback.h"

#include "DataMember.h"
#include "Enum.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::EnumBuilder::EnumBuilder( const char * nam,
                                        const std::type_info & ti ) {
//-------------------------------------------------------------------------------
   fEnum = new Enum( nam, ti );
}

//-------------------------------------------------------------------------------
ROOT::Reflex::EnumBuilder::~EnumBuilder() {
//-------------------------------------------------------------------------------
   FireClassCallback( *fEnum );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::EnumBuilder & ROOT::Reflex::EnumBuilder::AddItem( const char * nam,
                                                                long value ) {  
//-------------------------------------------------------------------------------
   fEnum->AddDataMember( Member(new DataMember( nam, 
                                                Type::ByName("int"), 
                                                value, 
                                                0 )));
   return *this;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::EnumBuilder & ROOT::Reflex::EnumBuilder::AddProperty( const char * key,
                                                                    Any value ) {
//-------------------------------------------------------------------------------
   if ( fLastMember ) fLastMember.Properties().AddProperty( key , value );
   else                fEnum->Properties().AddProperty( key, value );
   return *this;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::EnumBuilder &  ROOT::Reflex::EnumBuilder::AddProperty( const char * key,
                                                                     const char * value ) {
//-------------------------------------------------------------------------------
   AddProperty( key, Any(value));
   return *this;
}

