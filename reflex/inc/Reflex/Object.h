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
      Object( const Type & type = Type(), 
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
       * operator bool
       */
      operator bool () const;


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


      /**
       * Destruct will call the destructor of the object and remove it 
       * from the heap if possible
       */
      void Destruct() const;

      
      /**
       * DynamicType is used to discover runtime TypeNth of the object
       * @return the actual class of the object
       */
      Type DynamicType() const;


      /**
       * Get will look for a data MemberNth with Name dm and return the value
       * and TypeNth of this data MemberNth as an object
       * @param  dm Name of the data MemberNth
       * @return data MemberNth as object 
       */
      Object Get( const std::string & dm ) const ;


      Object Invoke( const std::string & fm, 
                     std::vector< void * > args = std::vector<void*>()) const;
      
      
      Object Invoke( const std::string & fm, 
                     const Type & sign,
                     std::vector< void * > args = std::vector<void*>()) const;
      
      
      /*
      Object Invoke( const std::string & fm, 
                     std::vector< Object > args = std::vector< Object>()) const;
      Object Invoke( const std::string & fm, 
                     const Type & ft,
                     std::vector< Object > args = std::vector< Object>()) const;
      */


      template < class T0 >
        Object Invoke( const std::string & fm,
                       const T0 & p0 ) const ;
      
      
      template < class T0 >
        Object Invoke( const std::string & fm,
                       const Type & sign,
                       const T0 & p0 ) const ;
      
      
      template < class T0, class T1 >
        Object Invoke( const std::string & fm,
                       const T0 & p0,
                       const T1 & p1 ) const ;
      

      template < class T0, class T1 >
        Object Invoke( const std::string & fm,
                       const Type & sign,
                       const T0 & p0,
                       const T1 & p1 ) const ;
      

      void Set(const std::string & dm,
               const void * value ) const;


      /*
      void Set(const std::string & dm,
               const Object & value ) const;
      */


      template < class T >
        void Set(const std::string & dm,
                 const T & value ) const;
      

      /**
       * TypeNth will return the pointer to the TypeNth of the object
       * @return pointer to object TypeNth
       */
      Type TypeGet() const;


      //
      //  D E P R E C A T E D   M E M B E R   F U N C T I O N S   -   B E G I N
      //

      /** 
       * Trying to consolidate the different versions of Object::Invoke and Object::InvokeT
       * The InvokeT functions are deprecated and will be removed in a future release.
       * As a replacement for a cast on the return TypeNth please use Object_Cast<T>(obj.Invoke(....))
       */

      template < class R >
        R InvokeT( const std::string & fm, 
                   const std::vector< void * > args = std::vector<void*>()) const
#if defined (__GNUC__)
        __attribute__((deprecated))
#endif
        ;
      
      
      template < class R, class T0 >
        R InvokeT( const std::string & fm,
                   const T0 & p0 ) const
#if defined (__GNUC__)
        __attribute__((deprecated))
#endif
        ;
      
      
      template < class R, class T0, class T1 >
        R InvokeT( const std::string & fm,
                   const T0 & p0,
                   const T1 & p1 ) const
#if defined (__GNUC__)
        __attribute__((deprecated))
#endif
        ;
 

      /**
       * GetT is deprecated and will be removed in a future release. 
       * As replacement please use Object_Cast<T>(Get(....))
       */
      template < class T >
        T GetT(const std::string & dm) const
#if defined (__GNUC__)
        __attribute__((deprecated))
#endif
        ;


      /**
       * Field is deprecated and will be removed in a future release
       * please use "Object Get(const std::string&) const" instead
       */
      Object Field( const std::string & data ) const
#if defined (__GNUC__)
        __attribute__((deprecated))
#endif
        ;


      /** 
       * SetT will be depricated in a future release, please use the templated Set function instead
       * (this is done in order to unify all Get/set/invoke functions)
       */
      template < class T >
        void SetT(const std::string & dm,
                  const T & value ) const
#if defined (__GNUC__)
        __attribute__((deprecated))
#endif
        ;

      //
      //  D E P R E C A T E D   M E M B E R   F U N C T I O N S   -   E N D
      //

    private:

      void Set2( const std::string & dm,
                 const void * value ) const;

      /** the TypeNth of the object 
       * @link aggregationByValue
       * @clientCardinality 0..*
       * @supplierCardinality 1
       * @label object TypeNth*/
      Type fType;


      /** the AddressGet of the object */
      mutable
      void * fAddress;

    }; // class Object


    /** 
     * Object_Cast can be used to cast an object into a given TypeNth
     * (no additional checks are performed for the time being)
     * @param o the object to be casted 
     * @return the AddressGet of the object casted into TypeNth T
     */
    template < class T > T Object_Cast( const Object & o );


  } // namespace Reflex
} // namespace ROOT

#include "Reflex/Member.h"
#include "Reflex/Tools.h"

//-------------------------------------------------------------------------------
template < class T >
inline T ROOT::Reflex::Object_Cast( const Object & o ) {
//-------------------------------------------------------------------------------
  return *(T*)o.AddressGet();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Object::Object( const Type & type,
                                     void * mem ) 
//-------------------------------------------------------------------------------
  : fType( type ),
    fAddress( mem ) {}


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
inline void ROOT::Reflex::Object::Destruct() const {
//-------------------------------------------------------------------------------
  if ( * this ) {
    fType.Destruct(fAddress);
    fAddress = 0;
  }
}
   

//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type ROOT::Reflex::Object::DynamicType() const {
//-------------------------------------------------------------------------------
  return fType.DynamicType(*this);
}


//-------------------------------------------------------------------------------
template < class T0 > 
inline ROOT::Reflex::Object
ROOT::Reflex::Object::Invoke( const std::string & fm,
                              const T0 & p0 ) const {
//-------------------------------------------------------------------------------
  return Invoke(fm,Tools::makeVector<void*>(Tools::CheckPointer<T0>::Get(p0)));
/*
  m = TypeGet().FunctionMemberNth( fm );
  if ( m ) {
    std::vector< void* > argList;
    argList.push_back( (void*)&p0 );
    return m.Invoke( * this, argList );
  }
  else throw RuntimeError("No such MemberNth " + fm );
  return Object();
*/
}


//-------------------------------------------------------------------------------
template < class T0 > 
inline ROOT::Reflex::Object
ROOT::Reflex::Object::Invoke( const std::string & fm,
                              const Type & sign,
                              const T0 & p0 ) const {
//-------------------------------------------------------------------------------
  return Invoke(fm,sign,Tools::makeVector<void*>(Tools::CheckPointer<T0>::Get(p0)));
}


//-------------------------------------------------------------------------------
template < class T0, class T1 > 
inline ROOT::Reflex::Object
ROOT::Reflex::Object::Invoke( const std::string & fm,
                              const T0 & p0,
                              const T1 & p1 ) const {
//-------------------------------------------------------------------------------
  return Invoke(fm,Tools::makeVector<void*>(Tools::CheckPointer<T0>::Get(p0), 
                                            Tools::CheckPointer<T1>::Get(p1)));
/*
  m = TypeGet().FunctionMemberNth( fm );
  if ( m ) {
    std::vector< void* > argList;
    argList.push_back( (void*)&p0 );
    argList.push_back( (void*)&p1 );
    return m.Invoke( * this, argList );
  }
  else throw RuntimeError("No such MemberNth " + fm );
  return Object();
*/
}


//-------------------------------------------------------------------------------
template < class T0, class T1 > 
inline ROOT::Reflex::Object
ROOT::Reflex::Object::Invoke( const std::string & fm,
                              const Type & sign, 
                              const T0 & p0,
                              const T1 & p1 ) const {
//-------------------------------------------------------------------------------
  return Invoke(fm,sign,Tools::makeVector<void*>(Tools::CheckPointer<T0>::Get(p0), 
                                                 Tools::CheckPointer<T1>::Get(p1)));
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Object::Set( const std::string & dm,
                                       const void * value ) const {
//-------------------------------------------------------------------------------
  Set2( dm, value );
}


//-------------------------------------------------------------------------------
template < class T >
inline void ROOT::Reflex::Object::Set( const std::string & dm,
                                       const T & value ) const {
//-------------------------------------------------------------------------------
  Set2( dm, & value );
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type ROOT::Reflex::Object::TypeGet() const {
//-------------------------------------------------------------------------------
  return fType;
}


//
// DEPRICATED INLINE FUNCTION DEFINITIONS START HERE
//

//-------------------------------------------------------------------------------
template < class R > 
inline R ROOT::Reflex::Object::InvokeT( const std::string & fm,
                                        const std::vector< void * > args ) const {
//-------------------------------------------------------------------------------
  if (args.size()) return Object_Cast < R > ( Invoke( fm ));
  else             return Object_Cast < R > ( Invoke( fm, args ) );
}


//-------------------------------------------------------------------------------
template < class R, class T0 >
inline R ROOT::Reflex::Object::InvokeT( const std::string & fm,
                                        const T0 & p0 ) const {
//-------------------------------------------------------------------------------
  return Object_Cast < R > ( Invoke( fm, p0 ) );
}


//-------------------------------------------------------------------------------
template < class R, class T0, class T1 >
inline R ROOT::Reflex::Object::InvokeT( const std::string & fm,
                                        const T0 & p0,
                                        const T1 & p1 ) const {
//-------------------------------------------------------------------------------
  return Object_Cast< R > ( Invoke( fm, p0, p1 ) );
}


//-------------------------------------------------------------------------------
template < class T > 
inline T ROOT::Reflex::Object::GetT( const std::string & dm ) const {
//-------------------------------------------------------------------------------
  Member m = TypeGet().MemberNth( dm );
  if ( m ) return Object_Cast< T > ( m.Get( * this ));
  else throw RuntimeError("No such MemberNth " + dm );
  return Object_Cast < T > ( Object() );
}


//-------------------------------------------------------------------------------
template < class T >
inline void ROOT::Reflex::Object::SetT( const std::string & dm,
                                        const T & value ) const {
//-------------------------------------------------------------------------------
  Set2( dm, & value );
}


#endif // ROOT_Reflex_Object
