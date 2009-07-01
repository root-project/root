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

#include "Reflex/Member.h"

#include "Reflex/Scope.h"
#include "Reflex/Type.h"
#include "Reflex/Base.h"
#include "Reflex/PropertyList.h"

#include "Reflex/Tools.h"
#include "Class.h"

#include <iostream>


//-------------------------------------------------------------------------------
Reflex::Member::Member(const MemberBase* memberBase)
//-------------------------------------------------------------------------------
   : fMemberBase(const_cast<MemberBase*>(memberBase)) {
   // Construct a member, attaching it to MemberBase.
}


//-------------------------------------------------------------------------------
Reflex::Member::Member(const Member& rh)
//-------------------------------------------------------------------------------
   : fMemberBase(rh.fMemberBase) {
   // Member copy constructor.
}


//-------------------------------------------------------------------------------
Reflex::Member::~Member() {
//-------------------------------------------------------------------------------
// Member desructor.
}


//-------------------------------------------------------------------------------
void
Reflex::Member::Delete() {
//-------------------------------------------------------------------------------
// delete the MemberBase
   delete fMemberBase;
   fMemberBase = 0;
}


//-------------------------------------------------------------------------------
Reflex::Object
Reflex::Member::Get() const {
//-------------------------------------------------------------------------------
// Get the value of a static member.
   if (fMemberBase) {
      return fMemberBase->Get(Object());
   }
   return Object();
}


//-------------------------------------------------------------------------------
Reflex::Object
Reflex::Member::Get(const Object& obj) const {
//-------------------------------------------------------------------------------
// Get the value of a non static data member.
   if (fMemberBase) {
      return fMemberBase->Get(obj);
   }
   return Object();
}


/*//-------------------------------------------------------------------------------
   Reflex::Object
   Reflex::Member::Invoke( const Object & obj,
   const std::vector < Object > & paramList ) const {
   //-------------------------------------------------------------------------------
   if ( fMemberBase ) return fMemberBase->Invoke( obj, paramList );
   return Object();
   }
 */


//-------------------------------------------------------------------------------
void
Reflex::Member::Invoke(const Object& obj,
                       Object* ret,
                       const std::vector<void*>& paramList) const {
//-------------------------------------------------------------------------------
// Invoke a non static data member. Put return value (if not void) into ret.
   if (fMemberBase) {
      fMemberBase->Invoke(obj, ret, paramList);
   }
}


/*/-------------------------------------------------------------------------------
   Reflex::Object
   Reflex::Member::Invoke( const std::vector < Object > & paramList ) const {
   //-------------------------------------------------------------------------------
   if ( fMemberBase ) return fMemberBase->Invoke( paramList );
   return Object();
   }
 */


//-------------------------------------------------------------------------------
void
Reflex::Member::Invoke(Object* ret,
                       const std::vector<void*>& paramList) const {
//-------------------------------------------------------------------------------
// Invoke a static data member. Put return value (if not void) into ret.
   if (fMemberBase) {
      fMemberBase->Invoke(ret, paramList);
   }
}


/*/-------------------------------------------------------------------------------
   void Reflex::Member::Set( const Object & instance,
   const Object & value ) const {
   //--------------------------------------------------------------------------------
   if (fMemberBase ) fMemberBase->Set( instance, value );
   }
 */


//-------------------------------------------------------------------------------
void
Reflex::Member::Set(const Object& instance,
                    const void* value) const {
//-------------------------------------------------------------------------------
// Set a non static data member.
   if (fMemberBase) {
      fMemberBase->Set(instance, value);
   }
}


//-------------------------------------------------------------------------------
void
Reflex::Member::GenerateDict(DictionaryGenerator& generator) const {
//-------------------------------------------------------------------------------
// Generate Dictionary information about itself.
   if (*this) {
      fMemberBase->GenerateDict(generator);
   }
}


#ifdef REFLEX_CINT_MERGE
bool
Reflex::Member::operator &&(const Scope& right) const
{ return operator bool() && (bool) right; }

bool
Reflex::Member::operator &&(const Type& right) const
{ return operator bool() && (bool) right; }

bool
Reflex::Member::operator &&(const Member& right) const
{ return operator bool() && (bool) right; }

bool
Reflex::Member::operator ||(const Scope& right) const
{ return operator bool() && (bool) right; }

bool
Reflex::Member::operator ||(const Type& right) const
{ return operator bool() && (bool) right; }

bool
Reflex::Member::operator ||(const Member& right) const
{ return operator bool() && (bool) right; }

#endif
