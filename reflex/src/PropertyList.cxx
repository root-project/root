// @(#)root/reflex:$Name: HEAD $:$Id: PropertyList.cxx,v 1.8 2006/07/04 15:02:55 roiser Exp $
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

#include "Reflex/PropertyList.h"

#include "Reflex/PropertyListImpl.h"
#include "Reflex/Any.h"

//-------------------------------------------------------------------------------
static ROOT::Reflex::Any & sEmptyAny() {
//-------------------------------------------------------------------------------
// Wrapper around static any.
   static ROOT::Reflex::Any a;
   return a;
}


//-------------------------------------------------------------------------------
std::ostream & ROOT::Reflex::operator<<( std::ostream & s,
                                                const PropertyList & p ) {
//-------------------------------------------------------------------------------
// Operator to put a property list on the ostream.
   if ( p.fPropertyListImpl ) s << *(p.fPropertyListImpl); 
   return s;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Any &
ROOT::Reflex::PropertyList::PropertyValue(const std::string & key) const {
//-------------------------------------------------------------------------------
// Get the value of a property as Any object.
   if ( fPropertyListImpl ) return fPropertyListImpl->PropertyValue( key );
   return sEmptyAny();
}

