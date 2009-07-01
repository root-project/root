// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
# define REFLEX_BUILD
#endif

#include "Reflex/internal/PropertyListImpl.h"
#include "Reflex/Any.h"
#include "Reflex/Tools.h"

#include <sstream>

// SOLARIS CC FIX (this include file is needed for a fix for std::distance)
#include "stl_hash.h"

/** the Key container */
typedef std::vector<std::string> Keys_t;

//-------------------------------------------------------------------------------
Keys_t&
sKeys() {
//-------------------------------------------------------------------------------
// Wrapper for static keys container.
   static Keys_t* k = 0;

   if (!k) {
      k = new Keys_t;
   }
   return *k;
}


//-------------------------------------------------------------------------------
std::ostream&
Reflex::operator <<(std::ostream& s,
                    const PropertyListImpl& p) {
//-------------------------------------------------------------------------------
// Operator to put properties on the ostream.
   if (p.fProperties) {
      for (size_t i = 0; i < p.fProperties->size(); ++i) {
         Any& a = p.PropertyValue(i);

         if (a) {
            s << sKeys()[i] << " : " << a << std::endl;
         }
      }
   }
   return s;
}


//-------------------------------------------------------------------------------
Reflex::PropertyListImpl::~PropertyListImpl() {
//-------------------------------------------------------------------------------
// Destruct, deleting our fProperties.
   delete fProperties;
}


//-------------------------------------------------------------------------------
void
Reflex::PropertyListImpl::ClearProperties() {
//-------------------------------------------------------------------------------
// Clear, remove all properties.
   if (fProperties) {
      delete fProperties;
   }
   fProperties = 0;
}


//-------------------------------------------------------------------------------
bool
Reflex::PropertyListImpl::HasProperty(const std::string& key) const {
//-------------------------------------------------------------------------------
// Return true if property has key.
   size_t i = KeyByName(key);

   if (i == NPos()) {
      return false;
   } else { return PropertyValue(i); }
}


//-------------------------------------------------------------------------------
bool
Reflex::PropertyListImpl::HasProperty(size_t key) const {
//-------------------------------------------------------------------------------
// Return true if property has key.
   return PropertyValue(key);
}


//-------------------------------------------------------------------------------
Reflex::StdString_Iterator
Reflex::PropertyListImpl::Key_Begin() {
//-------------------------------------------------------------------------------
// Return begin iterator of key container
   return sKeys().begin();
}


//-------------------------------------------------------------------------------
Reflex::StdString_Iterator
Reflex::PropertyListImpl::Key_End() {
//-------------------------------------------------------------------------------
// Return end iterator of key container
   return sKeys().end();
}


//-------------------------------------------------------------------------------
Reflex::Reverse_StdString_Iterator
Reflex::PropertyListImpl::Key_RBegin() {
//-------------------------------------------------------------------------------
// Return rbegin iterator of key container
   return ((const std::vector<std::string> &)sKeys()).rbegin();
}


//-------------------------------------------------------------------------------
Reflex::Reverse_StdString_Iterator
Reflex::PropertyListImpl::Key_REnd() {
//-------------------------------------------------------------------------------
// Return rend iterator of key container
   return ((const std::vector<std::string> &)sKeys()).rend();
}


//-------------------------------------------------------------------------------
std::string
Reflex::PropertyListImpl::KeysAsString() {
//-------------------------------------------------------------------------------
// Return a string containing all property keys.
   return Tools::StringVec2String(sKeys());
}


//-------------------------------------------------------------------------------
const std::string&
Reflex::PropertyListImpl::KeyAt(size_t nth) {
//-------------------------------------------------------------------------------
// Return the nth property key.
   return sKeys().at(nth);
}


//-------------------------------------------------------------------------------
size_t
Reflex::PropertyListImpl::KeyByName(const std::string& key,
                                    bool allocateNew) {
//-------------------------------------------------------------------------------
// Return a key by it's name.
   Keys_t::iterator it = std::find(sKeys().begin(), sKeys().end(), key);

   if (it != sKeys().end()) {
      return std::distance(sKeys().begin(), it);
   } else if (allocateNew) {
      sKeys().push_back(key);
      return sKeys().size() - 1;
   }
   return NPos();
}


//-------------------------------------------------------------------------------
size_t
Reflex::PropertyListImpl::KeySize() {
//-------------------------------------------------------------------------------
// Return number of all allocated keys.
   return sKeys().size();
}


//-------------------------------------------------------------------------------
std::string
Reflex::PropertyListImpl::PropertyAsString(const std::string& key) const {
//-------------------------------------------------------------------------------
// Return a property as a string.
   return PropertyAsString(PropertyKey(key));
}


//-------------------------------------------------------------------------------
std::string
Reflex::PropertyListImpl::PropertyAsString(size_t key) const {
//-------------------------------------------------------------------------------
// Return a string representation of the property with key.
   Any& a = PropertyValue(key);

   if (a) {
      std::ostringstream o;
      o << a;
      return o.str();
   }
   return "";
}


//-------------------------------------------------------------------------------
size_t
Reflex::PropertyListImpl::PropertyKey(const std::string& key,
                                      bool allocateNew) const {
//-------------------------------------------------------------------------------
// return the index of property key, allocate a new one if allocateNew = true
   return KeyByName(key, allocateNew);
}


//-------------------------------------------------------------------------------
std::string
Reflex::PropertyListImpl::PropertyKeys() const {
//-------------------------------------------------------------------------------
// Return a string containing all property keys.
   std::vector<std::string> kv;

   for (size_t i = 0; i < KeySize(); ++i) {
      if (PropertyValue(i)) {
         kv.push_back(KeyAt(i));
      }
   }
   return Tools::StringVec2String(kv);
}


//-------------------------------------------------------------------------------
size_t
Reflex::PropertyListImpl::PropertyCount() const {
//-------------------------------------------------------------------------------
// Returns the number of properties attached to this item. Don't use the output
// for iteration. Use KeySize() instead.
   size_t s = 0;

   if (fProperties) {
      for (size_t i = 0; i < fProperties->size(); ++i) {
         if (PropertyValue(i)) {
            ++s;
         }
      }
   }
   return s;
}


//-------------------------------------------------------------------------------
Reflex::Any&
Reflex::PropertyListImpl::PropertyValue(const std::string& key) const {
//-------------------------------------------------------------------------------
// Return a property as an Any object.
   return PropertyValue(PropertyKey(key));
}


//-------------------------------------------------------------------------------
Reflex::Any&
Reflex::PropertyListImpl::PropertyValue(size_t key) const {
//-------------------------------------------------------------------------------
// Return property as Any object
   if (fProperties && key < fProperties->size()) {
      return (*fProperties)[key];
   }
   return Dummy::Any();
}
