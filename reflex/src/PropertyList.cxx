// @(#)root/reflex:$Name:  $:$Id: PropertyList.cxx,v 1.11 2006/09/05 17:13:15 roiser Exp $
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
// Return the begin iterator of the keys container.
   return PropertyListImpl::Key_Begin();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::StdString_Iterator ROOT::Reflex::PropertyList::Key_End() {
//-------------------------------------------------------------------------------
// Return the end iterator of the keys container.
   return PropertyListImpl::Key_End();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::PropertyList::Key_RBegin() {
//-------------------------------------------------------------------------------
// Return the rbegin iterator of the keys container.
   return PropertyListImpl::Key_RBegin();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::PropertyList::Key_REnd() {
//-------------------------------------------------------------------------------
// Return the rend iterator of the keys container.
   return PropertyListImpl::Key_REnd();
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::PropertyList::KeysAsString() {
//-------------------------------------------------------------------------------
// Return all keys as one string concatenation.
   return PropertyListImpl::KeysAsString();
}


//-------------------------------------------------------------------------------
const std::string & ROOT::Reflex::PropertyList::KeyAt( size_t nth ) {
//-------------------------------------------------------------------------------
// Return key at position nth.
   return PropertyListImpl::KeyAt( nth );
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::PropertyList::KeyByName( const std::string & key,
                                              bool allocateNew ) {
//-------------------------------------------------------------------------------
// Return the position of a Key. If allocateNew is set to true allocate a new key
// if necessary.
   return PropertyListImpl::KeyByName( key, allocateNew );
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::PropertyList::KeySize() {
//-------------------------------------------------------------------------------
// Return the number of all allocated keys.
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

