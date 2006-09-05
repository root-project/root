// @(#)root/reflex:$Name:  $:$Id: PropertyList.h,v 1.11 2006/08/11 06:31:59 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_PropertyList
#define ROOT_Reflex_PropertyList


// Include files
#include "Reflex/Kernel.h"
#include <iostream>

namespace ROOT {
   namespace Reflex {

      // forward declarations
      class PropertyListImpl;
      class Any;

      /**
       * @class PropertyList PropertyList.h Reflex/PropertyList.h
       * @author Stefan Roiser
       * @date 24/11/2003
       * @ingroup Ref
       */
      class RFLX_API PropertyList {

         friend class OwnedPropertyList;
         friend std::ostream & operator << ( std::ostream & s,
                                             const PropertyList & p );

      public:

         /** default constructor */
         PropertyList( PropertyListImpl * propertyListImpl = 0 );


         /** copy constructor */
         PropertyList( const PropertyList & pl );


         /** destructor */
         ~PropertyList();


         /**
          * operator bool will return true if the property list is implemented
          * @return true if property list is not a fake one
          */
         operator bool () const;
       

         /**
          * AddProperty will add a key value pair to the property list
          * @param key the key of the property
          * @param value the value of the property (as any object)
          * @return the property key of this property
          */
         size_t AddProperty( const std::string & key,
                             const Any & value ) const;


         /**
          * AddProperty will add a property value pair to the property list
          * @param key the key of the property
          * @param value the value of the property (as any object)
          */
         void AddProperty( size_t key,
                           const Any & value ) const;


         /**
          * AddProperty will add a key value pair to the property lsit
          * @param key the key of the property
          * @param value the value of the property (as any object)
          * @return the property key of this property
          */
         size_t AddProperty( const std::string & key,
                             const char * value ) const;


         /**
          * AddProperty will add a key value pair to the property list
          * @param key the key of the property
          * @param value the value of the property (as any object)
          */
         void AddProperty( size_t key,
                           const char * value ) const;

 
        /**
          * ClearProperties will remove all properties from the list
          */
         void ClearProperties() const;

      
         /**
          * HasKey will return true if the property list contains the key
          * @param  key the property key
          * @return nth property key
          */
         bool HasKey( const std::string & key ) const;


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
         static const std::string & KeyAt( size_t nth );


         /**
          * Key is the static getter function to return the index of a key. If allocateNew is 
          * set to true a new key will be allocated if it doesn't exist and it's index returned.
          * Otherwise if the key exists the function returns it's index or 0 if no key exists.
          * @param key the key to look for 
          * @param allocateNew allocate a new key if the key doesn't exist
          * @return key index or 0 if no key exists and allocateNew is set to false
          */
         static size_t KeyByName( const std::string & key,
                                  bool allocateNew = false );


         /**
          * KeySize will return the number of currently allocated keys
          * @return number of allocated keys
          */
         static size_t KeySize();


         /**
          * PropertyAsString will return the nth property as a string if printable
          * @param  key the property key
          * @return nth property value as string
          */
         std::string PropertyAsString(const std::string & key) const;

     
         /**
          * PropertyAsString will return the property value as a string if it exists
          * The parameter is a property key which can be aquired with the PropertyKey method.
          * @param key property key to look for
          * @return string representation of the property
          */
         std::string PropertyAsString( size_t key ) const;        
         
         
         /**
          * PropertyKey will return the the key value corresponding to the parameter given
          * @param key the string denoting the key value to lookup
          * @param allocateNew if set to true a new key will be allocated if it doesn't exist
          if set to false and the key doesn't exist the function returns 0
          * @return the key value corresponding to the key param
          */
         size_t PropertyKey( const std::string & key,
                             bool allocateNew = false ) const;
         

         /**
          * PropertyKeys will return all keys of this property list
          * @return all property keys
          */
         std::string PropertyKeys() const;

         
         /**
          * PropertySize will return the number of properties attached
          * to this item
          * @return number of properties
          */
         size_t PropertySize() const;


         /**
          * PropertyValue will return the nth property value 
          * @param  key the property key
          * @return nth property value
          */
         Any & PropertyValue(const std::string & key) const;


         /**
          * PropertyAsString will return the property value as an Any object if it exists
          * The parameter is a property key which can be aquired with the PropertyKey method.
          * @param key property key to look for
          * @return Any representation of the property
          */
         Any & PropertyValue( size_t key ) const;


         /**
          * RemoveProperty will remove property value from the property list
          * @param key of the property identified by the string 
          */
         void RemoveProperty( const std::string & key ) const;


         /**
          * RemoveProperty will remove a property value from the property list 
          * @param key of the property identified by the property key number
          */
         void RemoveProperty( size_t key ) const;
     
      private:

         /** 
          * the properties of the item 
          * @link aggregation
          * @clentCardinality 1
          * @supplierCardinality 0..1
          * @label propertylist impl
          */
         PropertyListImpl * fPropertyListImpl;

      }; // class Propertylist

      /** 
       * will put the property (key and value) on the ostream if printable
       * @param s the reference to the stream
       * @return the stream
       */
      std::ostream & operator << ( std::ostream & s,
                                   const PropertyList & p );

   } //namespace Reflex
} //namespace ROOT

#include "Reflex/internal/PropertyListImpl.h"

//-------------------------------------------------------------------------------
inline ROOT::Reflex::PropertyList::operator bool () const {
//-------------------------------------------------------------------------------
   return 0 != fPropertyListImpl; 
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::PropertyList::PropertyList( PropertyListImpl * propertyListImpl ) 
//-------------------------------------------------------------------------------
   : fPropertyListImpl( propertyListImpl ) {}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::PropertyList::PropertyList( const PropertyList & pl) 
//-------------------------------------------------------------------------------
   : fPropertyListImpl( pl.fPropertyListImpl ) {}

  
//-------------------------------------------------------------------------------
inline ROOT::Reflex::PropertyList::~PropertyList() {
//-------------------------------------------------------------------------------
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::PropertyList::AddProperty( const std::string & key,
                                                       const Any & value ) const {
//-------------------------------------------------------------------------------
   if ( fPropertyListImpl ) return fPropertyListImpl->AddProperty( key, value );
   return 0;
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::PropertyList::AddProperty( size_t key,
                                                     const Any & value ) const {
//-------------------------------------------------------------------------------
   if ( fPropertyListImpl ) return fPropertyListImpl->AddProperty( key, value );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::PropertyList::AddProperty( const std::string & key,
                                                       const char* value ) const {
//-------------------------------------------------------------------------------
   if ( fPropertyListImpl ) return fPropertyListImpl->AddProperty( key, value );
   return 0;
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::PropertyList::AddProperty( size_t key,
                                                     const char* value ) const {
//-------------------------------------------------------------------------------
   if ( fPropertyListImpl ) return fPropertyListImpl->AddProperty( key, value );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::PropertyList::ClearProperties() const {
//-------------------------------------------------------------------------------
   if ( fPropertyListImpl ) fPropertyListImpl->ClearProperties();
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::PropertyList::HasKey(const std::string & key) const {
//-------------------------------------------------------------------------------
   if ( fPropertyListImpl ) return fPropertyListImpl->HasKey( key );
   return false;
}


//-------------------------------------------------------------------------------
inline std::string 
ROOT::Reflex::PropertyList::PropertyAsString( const std::string & key ) const {
//-------------------------------------------------------------------------------
   if ( fPropertyListImpl ) return fPropertyListImpl->PropertyAsString( key );
   return "";
}


//-------------------------------------------------------------------------------
inline std::string 
ROOT::Reflex::PropertyList::PropertyAsString( size_t key ) const {
//-------------------------------------------------------------------------------
   if ( fPropertyListImpl ) return fPropertyListImpl->PropertyAsString( key );
   return "";
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::PropertyList::PropertyKey( const std::string & key,
                                                       bool allocateNew ) const {
//-------------------------------------------------------------------------------
   if ( fPropertyListImpl ) return fPropertyListImpl->PropertyKey( key, allocateNew );
   return 0;
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::PropertyList::PropertyKeys() const {
//-------------------------------------------------------------------------------
   if ( fPropertyListImpl ) return fPropertyListImpl->PropertyKeys();
   return "";
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::PropertyList::PropertySize() const {
//-------------------------------------------------------------------------------
   if ( fPropertyListImpl ) return fPropertyListImpl->PropertySize();
   return 0;
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::PropertyList::RemoveProperty( const std::string & key ) const {
//-------------------------------------------------------------------------------
   if ( fPropertyListImpl ) fPropertyListImpl->RemoveProperty( key );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::PropertyList::RemoveProperty( size_t key ) const {
//-------------------------------------------------------------------------------
   if ( fPropertyListImpl ) fPropertyListImpl->RemoveProperty( key );
}


#endif // ROOT_Reflex_PropertyList
