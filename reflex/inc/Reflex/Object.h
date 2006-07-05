// @(#)root/reflex:$Name: HEAD $:$Id: Object.h,v 1.6 2006/03/13 15:49:50 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
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
      class RFLX_API Object {
      
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
          * Address will return the memory address of the object
          * @return memory address of object
          */
         void * Address() const;


         /**
          * CastObject an object from this class type to another one
          * @param  to is the class type to cast into
          * @param  obj the memory address of the object to be casted
          */
         Object CastObject( const Type & to ) const;


         /**
          * Destruct will call the destructor of a type and remove its memory
          * allocation if desired
          */
         void Destruct() const;

      
         /**
          * DynamicType is used to discover the dynamic type (useful in 
          * case of polymorphism)
          * @return the actual class of the object
          */
         Type DynamicType() const;


         /** 
          * Get the data member value 
          * @param dm name of the data member to get
          * @return member value as object
          */
         Object Get( const std::string & dm ) const ;


         /**
          * Invoke a member function of the object
          * @param fm name of the member function
          * @param args a vector of memory addresses to parameter values
          * @return the return value of the function as object
          */
         Object Invoke( const std::string & fm, 
                        std::vector< void * > args = std::vector<void*>()) const;
      
      
         /**
          * Invoke a member function of the object
          * @param fm name of the member function
          * @param sign the signature of the member function (for overloads)
          * @param args a vector of memory addresses to parameter values
          * @return the return value of the function as object
          */
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


         /**
          * Invoke a member function of the object
          * @param fm name of the member function
          * @param p0 the first argument of the function 
          * @return the return value of the function as object
          */
         template < class T0 >
            Object Invoke( const std::string & fm,
                           const T0 & p0 ) const ;
      
      
         /**
          * Invoke a member function of the object
          * @param fm name of the member function
          * @param sign the signature of the member function (for overloads)
          * @param p0 the first argument of the function 
          * @return the return value of the function as object
          */
         template < class T0 >
            Object Invoke( const std::string & fm,
                           const Type & sign,
                           const T0 & p0 ) const ;
      
      
         /**
          * Invoke a member function of the object
          * @param fm name of the member function
          * @param p0 the first argument of the function 
          * @param p1 the second argument of the function 
          * @return the return value of the function as object
          */
         template < class T0, class T1 >
            Object Invoke( const std::string & fm,
                           const T0 & p0,
                           const T1 & p1 ) const ;
      

         /**
          * Invoke a member function of the object
          * @param fm name of the member function
          * @param sign the signature of the member function (for overloads)
          * @param p0 the first argument of the function 
          * @param p1 the second argument of the function 
          * @return the return value of the function as object
          */
         template < class T0, class T1 >
            Object Invoke( const std::string & fm,
                           const Type & sign,
                           const T0 & p0,
                           const T1 & p1 ) const ;
      

         /**
          * Set will set a data member value of this object
          * @param dm the name of the data member
          * @param value the memory address of the value to set
          */
         void Set(const std::string & dm,
                  const void * value ) const;


         /*
           void Set(const std::string & dm,
           const Object & value ) const;
         */


         /**
          * Set will set a data member value of this object
          * @param dm the name of the data member
          * @param value the memory address of the value to set
          */
         template < class T >
            void Set(const std::string & dm,
                     const T & value ) const;
      

         /**
          * TypeOf will return the type of the object
          * @return type of the object
          */
         Type TypeOf() const;

      private:

         /** */
         void Set2( const std::string & dm,
                    const void * value ) const;

         /** the At of the object 
          * @link aggregationByValue
          * @clientCardinality 0..*
          * @supplierCardinality 1
          * @label object At*/
         Type fType;


         /** the Address of the object */
         mutable
            void * fAddress;

      }; // class Object


      /** 
       * Object_Cast can be used to cast an object into a given type
       * (no additional checks are performed for the time being)
       * @param o the object to be casted 
       * @return the address of the object casted into type T
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
   return *(T*)o.Address();
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
inline void * ROOT::Reflex::Object::Address() const {
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
   return Invoke(fm,Tools::MakeVector<void*>(Tools::CheckPointer<T0>::Get(p0)));
   /*
     m = TypeOf().FunctionMemberAt( fm );
     if ( m ) {
     std::vector< void* > argList;
     argList.push_back( (void*)&p0 );
     return m.Invoke( * this, argList );
     }
     else throw RuntimeError("No such MemberAt " + fm );
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
   return Invoke(fm,sign,Tools::MakeVector<void*>(Tools::CheckPointer<T0>::Get(p0)));
}


//-------------------------------------------------------------------------------
template < class T0, class T1 > 
inline ROOT::Reflex::Object
ROOT::Reflex::Object::Invoke( const std::string & fm,
                              const T0 & p0,
                              const T1 & p1 ) const {
//-------------------------------------------------------------------------------
  return Invoke(fm,Tools::MakeVector<void*>(Tools::CheckPointer<T0>::Get(p0), 
                                            Tools::CheckPointer<T1>::Get(p1)));
/*
  m = TypeOf().FunctionMemberAt( fm );
  if ( m ) {
    std::vector< void* > argList;
    argList.push_back( (void*)&p0 );
    argList.push_back( (void*)&p1 );
    return m.Invoke( * this, argList );
  }
  else throw RuntimeError("No such MemberAt " + fm );
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
  return Invoke(fm,sign,Tools::MakeVector<void*>(Tools::CheckPointer<T0>::Get(p0), 
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
inline ROOT::Reflex::Type ROOT::Reflex::Object::TypeOf() const {
//-------------------------------------------------------------------------------
  return fType;
}


#endif // ROOT_Reflex_Object
