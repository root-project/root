// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_Object
#define ROOT_Reflex_Object

// Include files
#include "Reflex/Type.h"
#include <string>
#include <vector>

namespace ROOT {
  namespace Reflex {

    // forward declarations

    /** 
     * @class Object Object.h Reflex/Object.h
     * @author Stefan Roiser
     * @date 24/06/2004
     * @ingroup Ref
     */
    class Object {
      
    public:

      /** constructor */
      Object( const Type & TypeNth = Type(), 
              void * mem = 0 );


      /** constructor */
      Object( const Object & );
      

      /** destructor */
      ~Object() {}

      
      /**
       * operator assigment 
       */
      Object operator = ( const Object & obj );

  
      /**
       * operator ==
       */
      bool operator == ( const Object & obj );


      /**
       * inequal operator 
       */
      bool operator != ( const Object & obj );


      /**
       * DynamicType is used to discover runtime TypeNth of the object
       * @return the actual class of the object
       */
      Type DynamicType() const;


      /** 
       * operator bool
       */
      operator bool () const;


      /**
       * TypeNth will return the pointer to the TypeNth of the object
       * @return pointer to object TypeNth
       */
      Type TypeGet() const;


      /** 
       * AddressGet will return the memory AddressGet of the object
       * @return memory AddressGet of object
       */
      void * AddressGet() const;


      /**
       * CastObject an object from this class TypeNth to another one
       * @param  to is the class TypeNth to cast into
       * @return new Object casted to Type
       */
      Object CastObject( const Type & to ) const;

      
      Object Field( const std::string & data ) const ;


      Object Get(const std::string & dm ) const;


      template < class T >
        T GetT(const std::string & dm) const;

      
      //Object Invoke( const std::string & fm, 
      //              std::vector< Object > args = std::vector< Object>() ) const;
      Object Invoke( const std::string & fm, 
                     std::vector< void * > args = std::vector<void*>() ) const;
      
      
      //template < class R >
      //  R Invoke( const std::string & fm, 
      //            std::vector< Object > args = std::vector< Object >() ) const;
      template < class R >
        R InvokeT( const std::string & fm, 
                   const std::vector< void * > args = std::vector<void*>() ) const;
      
      
      template < class R >
        R InvokeT( const std::string & fm ) const;
      
      
      template < class T0 >
        Object Invoke( const std::string & fm,
                       const T0 & p0 ) const ;
      
      
      template < class R, class T0 >
        R InvokeT( const std::string & fm,
		   const T0 & p0 ) const ;
      
      
      template < class T0, class T1 >
        Object Invoke( const std::string & fm,
                       const T0 & p0,
                       const T1 & p1) const ;
      
      
      template < class R, class T0, class T1 >
        R InvokeT( const std::string & fm,
		   const T0 & p0,
		   const T1 & p1) const ;


      //void Set(const std::string & dm,
      //         const Object & value ) const;
      void Set(const std::string & dm,
               const void * value ) const;

      
      template < class T >
        void SetT(const std::string & dm,
                 const T & value ) const;
      
    private:

      /** the TypeNth of the object 
       * @link aggregationByValue
       * @clientCardinality 0..*
       * @supplierCardinality 1
       * @label object TypeNth*/
      Type fType;


      /** the AddressGet of the object */
      void * fAddress;

    }; // class Object

    template < class T > T object_cast( const Object & o );

  } // namespace Reflex
} // namespace ROOT


//-------------------------------------------------------------------------------
template < class T >
inline T ROOT::Reflex::object_cast( const Object & o ) {
//-------------------------------------------------------------------------------
  return *(T*)o.AddressGet();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Object::Object( const Type & TypeNth,
                                     void * AddressGet ) 
//-------------------------------------------------------------------------------
  : fType( TypeNth ),
    fAddress( AddressGet ) {}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Object::Object( const Object & obj )
//-------------------------------------------------------------------------------
  : fType( obj.fType ),
    fAddress( obj.fAddress ) {}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Object ROOT::Reflex::Object::operator = ( const Object & obj ) {
//-------------------------------------------------------------------------------
  fType    = obj.fType;
  fAddress = obj.fAddress;
  return * this;
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Object::operator == ( const Object & obj ) {
//-------------------------------------------------------------------------------
  return ( fType == obj.fType && fAddress == obj.fAddress );
}


//-------------------------------------------------------------------------------
inline bool ROOT::Reflex::Object::operator != ( const Object & obj ) {
//-------------------------------------------------------------------------------
  return ( fType != obj.fType || fAddress != obj.fAddress );
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type ROOT::Reflex::Object::DynamicType() const {
//-------------------------------------------------------------------------------
  return fType.DynamicType(*this);
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Object::operator bool () const {
//-------------------------------------------------------------------------------
  if ( fType && fAddress ) return true;
  return false;
}


//-------------------------------------------------------------------------------
inline void * ROOT::Reflex::Object::AddressGet() const {
//-------------------------------------------------------------------------------
  return fAddress;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Object ROOT::Reflex::Object::CastObject( const Type & to ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fType.CastObject(to, *this);
  return Object();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type ROOT::Reflex::Object::TypeGet() const {
//-------------------------------------------------------------------------------
  return fType;
}


//-------------------------------------------------------------------------------
template < class T >
inline void ROOT::Reflex::Object::SetT( const std::string & dm,
					const T & value ) const {
//-------------------------------------------------------------------------------
  Set( dm, Object( Type::ByTypeInfo( typeid(T).name()), & value ));
}


/*/-------------------------------------------------------------------------------
template < class R > 
inline R ROOT::Reflex::Object::InvokeT( const std::string & fm,
                                        const std::vector< Object > args ) const {
//-------------------------------------------------------------------------------
  return object_cast < R > ( Invoke( fm, args ) );
}
*/


//-------------------------------------------------------------------------------
template < class R > 
inline R ROOT::Reflex::Object::InvokeT( const std::string & fm,
                                        const std::vector< void * > args ) const {
//-------------------------------------------------------------------------------
  return object_cast < R > ( Invoke( fm, args ) );
}


//-------------------------------------------------------------------------------
template < class R > 
inline R ROOT::Reflex::Object::InvokeT( const std::string & fm ) const {
//-------------------------------------------------------------------------------
  return object_cast < R > ( Invoke( fm )) ;
}


//-------------------------------------------------------------------------------
template < class R, class T0 >
inline R ROOT::Reflex::Object::InvokeT( const std::string & fm,
                                        const T0 & p0 ) const {
//-------------------------------------------------------------------------------
  return object_cast < R > ( Invoke( fm, p0 ) );
}


//-------------------------------------------------------------------------------
template < class R, class T0, class T1 >
inline R ROOT::Reflex::Object::InvokeT( const std::string & fm,
                                        const T0 & p0,
                                        const T1 & p1 ) const {
//-------------------------------------------------------------------------------
  return object_cast< R > ( Invoke( fm, p0, p1 ) );
}


#endif // ROOT_Reflex_Object
