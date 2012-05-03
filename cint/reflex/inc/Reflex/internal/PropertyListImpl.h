// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_PropertyListImpl
#define Reflex_PropertyListImpl


// Include files
#include "Reflex/Kernel.h"
#include "Reflex/Any.h"
#include <map>
#include <iostream>

#ifdef _WIN32
# pragma warning( push )
# pragma warning( disable : 4251 )
#endif

namespace Reflex {
/**
 * @class PropertyList PropertyList.h Reflex/PropertyList.h
 * @author Stefan Roiser
 * @date 24/11/2003
 * @ingroup Ref
 */
class RFLX_API PropertyListImpl {
   friend RFLX_API std::ostream& operator <<(std::ostream& s,
                                             const PropertyListImpl& p);

public:
   /** default constructor */
   PropertyListImpl();


   /** copy constructor */
   PropertyListImpl(const PropertyListImpl &pl);


   /** destructor */
   virtual ~PropertyListImpl();


   /** assignment op */
   PropertyListImpl& operator=(const PropertyListImpl &pl);


   /**
    * AddProperty will add a key value pair to the PropertyNth lsit
    * @param key the key of the PropertyNth
    * @param value the value of the PropertyNth (as any object)
    * @return the property key of this property
    */
   size_t AddProperty(const std::string& key,
                      const Any& value);


   /**
    * AddProperty will add a property value pair to the property list
    * @param key the key of the property
    * @param value the value of the property (as any object)
    */
   void AddProperty(size_t key,
                    const Any& value);


   /**
    * AddProperty will add a key value pair to the PropertyNth lsit
    * @param key the key of the PropertyNth
    * @param value the value of the PropertyNth (as any object)
    * @return the property key of this property
    */
   size_t AddProperty(const std::string& key,
                      const char* value);


   /**
    * AddProperty will add a key value pair to the property list
    * @param key the key of the property
    * @param value the value of the property (as any object)
    */
   void AddProperty(size_t key,
                    const char* value);


   /**
    * ClearProperties will remove all properties from the list
    */
   void ClearProperties();


   /**
    * HasProperty will return true if the property list contains a key "key" and
    * the property for this key is valid
    * @param  key to look for
    * @return true if a valid property for key exists
    */
   bool HasProperty(const std::string& key) const;


   /**
    * HasProperty will return true if the property list contains a key "key" and
    * the property for this key is valid
    * @param  key to look for
    * @return true if a valid property for key exists
    */
   bool HasProperty(size_t key) const;


   /**
    * Key_Begin will return the begin iterator of the key container
    * @return begin iterator of key container
    */
   static StdString_Iterator Key_Begin();


   /**
    * Key_End will return the end iterator of the key container
    * @return end iterator of key container
    */
   static StdString_Iterator Key_End();


   /**
    * Key_RBegin will return the rbegin iterator of the key container
    * @return rbegin iterator of key container
    */
   static Reverse_StdString_Iterator Key_RBegin();


   /**
    * Key_REnd will return the rend iterator of the key container
    * @return rend iterator of key container
    */
   static Reverse_StdString_Iterator Key_REnd();


   /**
    * KeysAsString will return a space separated list of all keys
    * @return a list of all currently allocated keys
    */
   static std::string KeysAsString();


   /**
    * KeyAt will return the nth key allocated
    * @param nth key currently allocated
    * @return key as a string
    */
   static const std::string& KeyAt(size_t nth);


   /**
    * Key is the static getter function to return the index of a key. If allocateNew is
    * set to true a new key will be allocated if it doesn't exist and it's index returned.
    * Otherwise if the key exists the function returns it's index or 0 if no key exists.
    * @param key the key to look for
    * @param allocateNew allocate a new key if the key doesn't exist
    * @return key index or 0 if no key exists and allocateNew is set to false
    */
   static size_t KeyByName(const std::string& key,
                           bool allocateNew = false);


   /**
    * KeySize will return the number of currently allocated keys
    * @return number of allocated keys
    */
   static size_t KeySize();


   /**
    * propertyNumString will return the nth PropertyNth as a string if printable
    * @param  key the PropertyNth key
    * @return nth PropertyNth value as string
    */
   std::string PropertyAsString(const std::string& key) const;


   /**
    * PropertyAsString will return the property value as a string if it exists
    * The parameter is a property key which can be aquired with the PropertyKey method.
    * @param key property key to look for
    * @return string representation of the property
    */
   std::string PropertyAsString(size_t key) const;


   /**
    * PropertyKey will return the the key value corresponding to the parameter given
    * @param key the string denoting the key value to lookup
    * @param allocateNew if set to true a new key will be allocated if it doesn't exist
      if set to false and the key doesn't exist the function returns 0
    * @return the key value corresponding to the key param
    */
   size_t PropertyKey(const std::string& key,
                      bool allocateNew = false) const;


   /**
    * PropertyKeys will return all keys of this PropertyNth list
    * @return all PropertyNth keys
    */
   std::string PropertyKeys() const;


   /**
    * PropertyCount will return the number of properties attached
    * to this item
    * @return number of properties
    */
   size_t PropertyCount() const;


   /**
    * propertyNumValue will return the nth PropertyNth value
    * @param  key the PropertyNth key
    * @return nth PropertyNth value
    */
   Any& PropertyValue(const std::string& key) const;


   /**
    * PropertyAsString will return the property value as an Any object if it exists
    * The parameter is a property key which can be aquired with the PropertyKey method.
    * @param key property key to look for
    * @return Any representation of the property
    */
   Any& PropertyValue(size_t key) const;


   /**
    * RemoveProperty will remove a key value pair to the PropertyNth lsit
    * @param key the key of the PropertyNth
    */
   void RemoveProperty(const std::string& key);


   /**
    * RemoveProperty will remove a property value from the property list
    * @param key of the property identified by the property key number
    */
   void RemoveProperty(size_t key);

private:
   /**
    * the Property container
    */
   typedef std::vector<Any> Properties;


   /** the properties of the item
    * @label properties
    * @link aggregationByValue
    * @clientCardinality 1
    * @supplierCardinality 0..1
    */
   std::vector<Any>* fProperties;

};    // class PropertyListImpl

/**
 * will put the PropertyNth (key and value) on the ostream if printable
 * @param s the reference to the stream
 * @return the stream
 */
RFLX_API std::ostream& operator <<(std::ostream& s,
                                   const PropertyListImpl& p);

} //namespace Reflex


//-------------------------------------------------------------------------------
inline Reflex::PropertyListImpl::PropertyListImpl()
//-------------------------------------------------------------------------------
   : fProperties(0) {
}


//-------------------------------------------------------------------------------
inline Reflex::PropertyListImpl::PropertyListImpl(const PropertyListImpl& pl)
//-------------------------------------------------------------------------------
   : fProperties(pl.fProperties) {
}


//-------------------------------------------------------------------------------
inline
Reflex::PropertyListImpl&
Reflex::PropertyListImpl::operator=(const PropertyListImpl& pl) {
//-------------------------------------------------------------------------------
   if (&pl != this) {
      fProperties = pl.fProperties;
   }
   return *this;
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::PropertyListImpl::AddProperty(const std::string& key,
                                      const Any& value) {
//-------------------------------------------------------------------------------
   size_t k = PropertyKey(key, true);
   AddProperty(k, value);
   return k;
}


//-------------------------------------------------------------------------------
inline void
Reflex::PropertyListImpl::AddProperty(size_t key,
                                      const Any& value) {
//-------------------------------------------------------------------------------
   if (!fProperties) {
      fProperties = new Properties();
   }

   if (key >= fProperties->size()) {
      fProperties->resize(key + 1, Dummy::Any());
   }
   (*fProperties)[key] = value;
}


//-------------------------------------------------------------------------------
inline size_t
Reflex::PropertyListImpl::AddProperty(const std::string& key,
                                      const char* value) {
//-------------------------------------------------------------------------------
   return AddProperty(key, Any(value));
}


//-------------------------------------------------------------------------------
inline void
Reflex::PropertyListImpl::AddProperty(size_t key,
                                      const char* value) {
//-------------------------------------------------------------------------------
   AddProperty(key, Any(value));
}


//-------------------------------------------------------------------------------
inline void
Reflex::PropertyListImpl::RemoveProperty(const std::string& key) {
//-------------------------------------------------------------------------------
   RemoveProperty(PropertyKey(key));
}


//-------------------------------------------------------------------------------
inline void
Reflex::PropertyListImpl::RemoveProperty(size_t key) {
//-------------------------------------------------------------------------------
   if (fProperties) {
      fProperties->at(key).Swap(Dummy::Any());
   }
}


#ifdef _WIN32
# pragma warning( pop )
#endif

#endif // Reflex_PropertyListImpl
