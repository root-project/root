// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
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
    class PropertyList {

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
       * operator bool will return true if the PropertyNth list is implemented
       * @return true if PropertyNth list is not a fake one
       */
      operator bool () const;
       

      /**
       * AddProperty will add a key value pair to the PropertyNth lsit
       * @param key the key of the PropertyNth
       * @param value the value of the PropertyNth (as any object)
       */
      void AddProperty( const std::string & key,
                        const Any & value );


      /**
       * AddProperty will add a key value pair to the PropertyNth lsit
       * @param key the key of the PropertyNth
       * @param value the value of the PropertyNth (as any object)
       */
      void AddProperty( const std::string & key,
                        const char * value );


      /**
       * ClearProperties will remove all properties from the list
       */
      void ClearProperties();

      
      /**
       * RemoveProperty will remove a key value pair to the PropertyNth lsit
       * @param key the key of the PropertyNth
       */
      void RemoveProperty( const std::string & key );


      /**
       * HasKey will return true if the PropertyNth list contains the key
       * @param  key the PropertyNth key
       * @return nth PropertyNth key
       */
      bool HasKey( const std::string & key ) const;


      /**
       * PropertyCount will return the number of properties attached
       * to this item
       * @return number of properties
       */
      size_t PropertyCount() const;


      /**
       * PropertyKeys will return all keys of this PropertyNth list
       * @return all PropertyNth keys
       */
      std::string PropertyKeys() const;


      /**
       * propertyNumString will return the nth PropertyNth as a string if printable
       * @param  key the PropertyNth key
       * @return nth PropertyNth value as string
       */
      std::string PropertyAsString(const std::string & key) const;


      /**
       * propertyNumValue will return the nth PropertyNth value 
       * @param  key the PropertyNth key
       * @return nth PropertyNth value
       */
      Any PropertyValue(const std::string & key) const;

    private:

      /** the properties of the item 
       * @link aggregation
       * @clientCardinality 1
       * @label propertylist impl*/
      PropertyListImpl * fPropertyListImpl;

    }; // class Propertylist

    /** 
     * will put the PropertyNth (key and value) on the ostream if printable
     * @param s the reference to the stream
     * @return the stream
     */
    std::ostream & operator << ( std::ostream & s,
				 const PropertyList & p );

  } //namespace Reflex
} //namespace ROOT

#include "Reflex/PropertyListImpl.h"

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
inline void ROOT::Reflex::PropertyList::AddProperty( const std::string & key,
                                                     const Any & value ) {
//-------------------------------------------------------------------------------
  if ( fPropertyListImpl ) fPropertyListImpl->AddProperty( key, value );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::PropertyList::AddProperty( const std::string & key,
                                                     const char* value ) {
//-------------------------------------------------------------------------------
  if ( fPropertyListImpl ) fPropertyListImpl->AddProperty( key, value );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::PropertyList::ClearProperties() {
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
inline size_t ROOT::Reflex::PropertyList::PropertyCount() const {
//-------------------------------------------------------------------------------
  if ( fPropertyListImpl ) return fPropertyListImpl->PropertyCount();
  return 0;
}


//-------------------------------------------------------------------------------
inline std::string ROOT::Reflex::PropertyList::PropertyKeys() const {
//-------------------------------------------------------------------------------
  if ( fPropertyListImpl ) return fPropertyListImpl->PropertyKeys();
  return "";
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Any
ROOT::Reflex::PropertyList::PropertyValue(const std::string & key) const {
//-------------------------------------------------------------------------------
  if ( fPropertyListImpl ) return fPropertyListImpl->PropertyValue( key );
  return Any();
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::PropertyList::RemoveProperty( const std::string & key ) {
//-------------------------------------------------------------------------------
  if ( fPropertyListImpl ) fPropertyListImpl->RemoveProperty( key );
}

#endif // ROOT_Reflex_PropertyList
