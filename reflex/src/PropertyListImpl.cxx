// @(#)root/reflex:$Name:  $:$Id: PropertyListImpl.cxx,v 1.12 2006/09/08 20:54:05 roiser Exp $
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

#include "Reflex/internal/PropertyListImpl.h"
#include "Reflex/Any.h"
#include "Reflex/Tools.h"

#include <sstream>

// SOLARIS CC FIX (this include file is needed for a fix for std::distance)
#include "stl_hash.h"

/** the Key container */
typedef std::vector< std::string > Keys_t;

//-------------------------------------------------------------------------------
Keys_t & sKeys() {
//-------------------------------------------------------------------------------
   // Wrapper for static keys container.
   static Keys_t k;
   return k;
}


//-------------------------------------------------------------------------------
std::ostream & ROOT::Reflex::operator<<( std::ostream & s,
                                         const PropertyListImpl & p ) {
//-------------------------------------------------------------------------------
// Operator to put properties on the ostream.
   if ( p.fProperties ) {
      for ( size_t i = 0; i < p.fProperties->size(); ++i ) {
         Any & a = p.PropertyValue( i );
         if ( a ) s << sKeys()[i] << " : " << a << std::endl;
      }
   }
   return s;
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::PropertyListImpl::ClearProperties() {
//-------------------------------------------------------------------------------
// Clear, remove all properties.
   if ( fProperties ) delete fProperties;
   fProperties = 0;
}


//-------------------------------------------------------------------------------
bool ROOT::Reflex::PropertyListImpl::HasProperty( const std::string & key ) const {
//-------------------------------------------------------------------------------
   // Return true if property has key.
   size_t i = KeyByName(key);
   if ( i == NPos() ) return false;
   else               return PropertyValue( i );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::StdString_Iterator ROOT::Reflex::PropertyListImpl::Key_Begin() {
//-------------------------------------------------------------------------------
   // Return begin iterator of key container
   return sKeys().begin();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::StdString_Iterator ROOT::Reflex::PropertyListImpl::Key_End() {
//-------------------------------------------------------------------------------
   // Return end iterator of key container
   return sKeys().end();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::PropertyListImpl::Key_RBegin() {
//-------------------------------------------------------------------------------
   // Return rbegin iterator of key container
   return ((const std::vector<std::string>&)sKeys()).rbegin();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_StdString_Iterator ROOT::Reflex::PropertyListImpl::Key_REnd() {
//-------------------------------------------------------------------------------
   // Return rend iterator of key container
   return ((const std::vector<std::string>&)sKeys()).rend();
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::PropertyListImpl::KeysAsString() {
//-------------------------------------------------------------------------------
// Return a string containing all property keys.
   return Tools::StringVec2String(sKeys());
}


//-------------------------------------------------------------------------------
const std::string & ROOT::Reflex::PropertyListImpl::KeyAt( size_t nth ) {
//-------------------------------------------------------------------------------
   // Return the nth property key.
   return sKeys().at( nth );
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::PropertyListImpl::KeyByName( const std::string & key,
                                                  bool allocateNew ) {
//-------------------------------------------------------------------------------
// Return a key by it's name.
   Keys_t::iterator it = std::find( sKeys().begin(), sKeys().end(), key );
   if ( it != sKeys().end() ) {
      return std::distance(sKeys().begin(), it);
   }
   else if ( allocateNew ) {
      sKeys().push_back(key);
      return sKeys().size() - 1;
   }
   return NPos();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::PropertyListImpl::KeySize() {
//-------------------------------------------------------------------------------
   // Return number of all allocated keys.
   return sKeys().size();
}


//-------------------------------------------------------------------------------
std::string 
ROOT::Reflex::PropertyListImpl::PropertyAsString( const std::string & key ) const {
//-------------------------------------------------------------------------------
// Return a property as a string.
   return PropertyAsString( PropertyKey( key ));
}



//-------------------------------------------------------------------------------
std::string 
ROOT::Reflex::PropertyListImpl::PropertyAsString( size_t key ) const {
//-------------------------------------------------------------------------------
   // Return a string representation of the property with key.
   Any & a = PropertyValue( key );
   if ( a ) {
      std::ostringstream o;
      o << a;
      return o.str();
   }
   return "";
}



//-------------------------------------------------------------------------------
size_t ROOT::Reflex::PropertyListImpl::PropertyKey( const std::string & key,
                                                    bool allocateNew ) const {
//-------------------------------------------------------------------------------
   // return the index of property key, allocate a new one if allocateNew = true
   return KeyByName( key, allocateNew );
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::PropertyListImpl::PropertyKeys() const {
//-------------------------------------------------------------------------------
// Return a string containing all property keys.
   std::vector<std::string> kv;
   for ( size_t i = 0; i < KeySize(); ++i ) {
      if ( PropertyValue(i)) kv.push_back(KeyAt(i));
   }
   return Tools::StringVec2String(kv);
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::PropertyListImpl::PropertyCount() const {
//-------------------------------------------------------------------------------
// Returns the number of properties attached to this item. Don't use the output
// for iteration. Use KeySize() instead.
   size_t s = 0;
   if ( fProperties ) {
      for ( size_t i = 0; i < fProperties->size(); ++i ) {
         if ( PropertyValue( i ) ) ++s;
      }
   }
   return s;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Any &
ROOT::Reflex::PropertyListImpl::PropertyValue( const std::string & key ) const {
//-------------------------------------------------------------------------------
// Return a property as an Any object.
   return PropertyValue( PropertyKey( key ));
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Any &
ROOT::Reflex::PropertyListImpl::PropertyValue( size_t key ) const {
//-------------------------------------------------------------------------------
   // Return property as Any object
   if ( fProperties ) return fProperties->at( key );
   return Dummy::Any();
}

