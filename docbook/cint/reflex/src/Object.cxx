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

#include "Reflex/Object.h"

#include "Class.h"
#include "DataMember.h"
#include "FunctionMember.h"


//-------------------------------------------------------------------------------
Reflex::Object
Reflex::Object::Get(const std::string& dm) const {
//-------------------------------------------------------------------------------
// Get a data member value of this object.
   Member m = TypeOf().MemberByName(dm);

   if (!m) {
      throw RuntimeError("No such MemberAt " + dm);
   }
   return m.Get(*this);
}


/*/-------------------------------------------------------------------------------
   Reflex::Object
   Reflex::Object::Invoke( const std::string & fm,
   std::vector< Object > args ) const {
   //-------------------------------------------------------------------------------
   Member m = TypeOf().FunctionMemberAt( fm );
   if ( m ) {
   if ( args.size() ) return m.Invoke( * this, args );
   else               return m.Invoke( * this );
   }
   else throw RuntimeError("No such MemberAt " + fm );
   return Object();
   }
 */


//-------------------------------------------------------------------------------
void
Reflex::Object::Invoke(const std::string& fm,
                       Object* ret,
                       const std::vector<void*>& args) const {
//-------------------------------------------------------------------------------
// Invoke a data member of this object. Put return value(if not void) into ret.
   Invoke(fm, Type(), ret, args);

   /*
      m = TypeOf().FunctionMemberAt( fm );
      if ( m ) {
      if ( args.size() ) return m.Invoke( * this, args );
      else               return m.Invoke( * this );
      }
      else throw RuntimeError("No such MemberAt " + fm );
      return Object();
    */
}


//-------------------------------------------------------------------------------
void
Reflex::Object::Invoke(const std::string& fm,
                       const Type& sign,
                       Object* ret,
                       const std::vector<void*>& args) const {
//-------------------------------------------------------------------------------
// Invoke a data member of this object. Sign can be used for finding overloaded funs.
// Put return value(if not void) into ret.
   Member m = TypeOf().FunctionMemberByName(fm, sign);

   if (!m) {
      throw RuntimeError("No such MemberAt " + fm);
   }

   m.Invoke(*this, ret, args);
}


//-------------------------------------------------------------------------------
//void Reflex::Object::Set( const std::string & dm,
//                                const Object & value ) const {
//-------------------------------------------------------------------------------
//  Member m = TypeOf().MemberAt( dm );
//  if ( m ) m.Set( * this, value );
//  else throw RuntimeError("No such MemberAt " + dm );
//}


//-------------------------------------------------------------------------------
void
Reflex::Object::Set2(const std::string& dm,
                     const void* value) const {
//-------------------------------------------------------------------------------
// Internal set method. Wrapped from Set methods.
   Member m = TypeOf().MemberByName(dm);

   if (m) {
      m.Set(*this, value);
   } else { throw RuntimeError("No such MemberAt " + dm); }
}
