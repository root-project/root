// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/Object.h"

#include "Class.h"
#include "DataMember.h"
#include "FunctionMember.h"


//-------------------------------------------------------------------------------
ROOT::Reflex::Object ROOT::Reflex::Object::Field( const std::string & data ) const {
//-------------------------------------------------------------------------------
  Type t = TypeGet();
  if ( ! t.IsClass() ) throw RuntimeError("Object is not a composite");
  for ( size_t i = 0; i < t.DataMemberCount(); ++i ) {
    Member dm = t.DataMemberNth( i );
    if ( dm.Name() == data ) {
      return Object(dm.TypeGet(), (void*)((char*)AddressGet() + dm.Offset()));
    }
  }
  throw RuntimeError("Data MemberNth not found in class");
}
 

//-------------------------------------------------------------------------------
ROOT::Reflex::Object 
ROOT::Reflex::Object::Get( const std::string & dm ) const {
//-------------------------------------------------------------------------------
  Member m = TypeGet().MemberNth( dm );
  if ( m ) return m.Get( * this );
  else throw RuntimeError("No such MemberNth " + dm );
  return Object();
}


/*/-------------------------------------------------------------------------------
ROOT::Reflex::Object
ROOT::Reflex::Object::Invoke( const std::string & fm,
                              std::vector< Object > args ) const {
//-------------------------------------------------------------------------------
  Member m = TypeGet().FunctionMemberNth( fm );
  if ( m ) {
    if ( args.size() ) return m.Invoke( * this, args );
    else               return m.Invoke( * this );
  }
  else throw RuntimeError("No such MemberNth " + fm );
  return Object();
}
*/


//-------------------------------------------------------------------------------
ROOT::Reflex::Object
ROOT::Reflex::Object::Invoke( const std::string & fm,
                              std::vector < void * > args ) const {
//-------------------------------------------------------------------------------
  return Invoke(fm,Type(),args);
  /*
  m = TypeGet().FunctionMemberNth( fm );
  if ( m ) {
    if ( args.size() ) return m.Invoke( * this, args );
    else               return m.Invoke( * this );
  }
  else throw RuntimeError("No such MemberNth " + fm );
  return Object();
  */
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Object
ROOT::Reflex::Object::Invoke( const std::string & fm,
                              const Type & sign,
                              std::vector < void * > args ) const {
//-------------------------------------------------------------------------------
  Member m = TypeGet().FunctionMemberNth( fm, sign );
  if ( m ) {
    if ( args.size() ) return m.Invoke( * this, args );
    else               return m.Invoke( * this );
  }
  else throw RuntimeError("No such MemberNth " + fm );
  return Object();
}


//-------------------------------------------------------------------------------
//void ROOT::Reflex::Object::Set( const std::string & dm,
//                                const Object & value ) const {
//-------------------------------------------------------------------------------
//  Member m = TypeGet().MemberNth( dm );
//  if ( m ) m.Set( * this, value );
//  else throw RuntimeError("No such MemberNth " + dm );
//}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Object::Set2( const std::string & dm,
                                 const void * value ) const {
//-------------------------------------------------------------------------------
  Member m = TypeGet().MemberNth( dm );
  if ( m ) m.Set( * this, value );
  else throw RuntimeError("No such MemberNth " + dm );
}
