// @(#)root/reflex:$Name:  $:$Id: PropertyList.cxx,v 1.10 2006/08/03 16:49:21 roiser Exp $
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

#include "Reflex/internal/PropertyListImpl.h"
#include "Reflex/Any.h"

//-------------------------------------------------------------------------------
std::ostream & ROOT::Reflex::operator<<( std::ostream & s,
                                         const PropertyList & p ) {
//-------------------------------------------------------------------------------
// Operator to put a property list on the ostream.
   if ( p.fPropertyListImpl ) s << *(p.fPropertyListImpl); 
   return s;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::StdString_Iterator ROOT::Reflex::PropertyList::Key_Begin() {
//-------------------------------------------------------------------------------
   return PropertyListImpl::Key_Begin();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::StdString_Iterator ROOT::Reflex::PropertyList::Key_End() {
//-------------------------------------------------------------------------------
   return PropertyListImpl::Key_End();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::PropertyList::Key_RBegin() {
//-------------------------------------------------------------------------------
   return PropertyListImpl::Key_RBegin();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::PropertyList::Key_REnd() {
//-------------------------------------------------------------------------------
   return PropertyListImpl::Key_REnd();
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::PropertyList::KeysAsString() {
//-------------------------------------------------------------------------------
   return PropertyListImpl::KeysAsString();
}


//-------------------------------------------------------------------------------
const std::string & ROOT::Reflex::PropertyList::KeyAt( size_t nth ) {
//-------------------------------------------------------------------------------
   return PropertyListImpl::KeyAt( nth );
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::PropertyList::KeyByName( const std::string & key,
                                              bool allocateNew ) {
//-------------------------------------------------------------------------------
   return PropertyListImpl::KeyByName( key, allocateNew );
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::PropertyList::KeySize() {
//-------------------------------------------------------------------------------
   return PropertyListImpl::KeySize();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Any &
ROOT::Reflex::PropertyList::PropertyValue( const std::string & key ) const {
//-------------------------------------------------------------------------------
// Get the value of a property as Any object.
   if ( fPropertyListImpl ) return fPropertyListImpl->PropertyValue( key );
   return Dummy::Any();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Any &
ROOT::Reflex::PropertyList::PropertyValue( size_t key ) const {
//-------------------------------------------------------------------------------
// Get the value of a property as Any object.
   if ( fPropertyListImpl ) return fPropertyListImpl->PropertyValue( key );
   return Dummy::Any();
}

