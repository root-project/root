// @(#)root/reflex:$Name: HEAD $:$Id: Member.cxx,v 1.7 2006/07/04 15:02:55 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
#define REFLEX_BUILD
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
ROOT::Reflex::Member::Member( const MemberBase * memberBase )
//-------------------------------------------------------------------------------
   : fMemberBase( memberBase ) {
   // Construct a member, attaching it to MemberBase.
}



//-------------------------------------------------------------------------------
ROOT::Reflex::Member::Member( const Member & rh )
//-------------------------------------------------------------------------------
   : fMemberBase( rh.fMemberBase ) {
   // Member copy constructor.
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member::~Member() {
//-------------------------------------------------------------------------------
// Member desructor.
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Object ROOT::Reflex::Member::Get() const {
//-------------------------------------------------------------------------------
// Get the value of a static member.
   if ( fMemberBase ) return fMemberBase->Get( Object());
   return Object();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Object ROOT::Reflex::Member::Get( const Object & obj) const {
//-------------------------------------------------------------------------------
// Get the value of a non static data member.
   if ( fMemberBase ) return fMemberBase->Get( obj );
   return Object();
}


/*//-------------------------------------------------------------------------------
  ROOT::Reflex::Object 
  ROOT::Reflex::Member::Invoke( const Object & obj,
  const std::vector < Object > & paramList ) const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->Invoke( obj, paramList );
  return Object();
  }
*/


//-------------------------------------------------------------------------------
ROOT::Reflex::Object 
ROOT::Reflex::Member::Invoke( const Object & obj,
                              const std::vector < void * > & paramList ) const {
//-------------------------------------------------------------------------------
// Invoke a non static data member.
   if ( fMemberBase ) return fMemberBase->Invoke( obj, paramList );
   return Object();
}


/*/-------------------------------------------------------------------------------
  ROOT::Reflex::Object 
  ROOT::Reflex::Member::Invoke( const std::vector < Object > & paramList ) const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->Invoke( paramList );
  return Object();
  }
*/


//-------------------------------------------------------------------------------
ROOT::Reflex::Object 
ROOT::Reflex::Member::Invoke( const std::vector < void * > & paramList ) const {
//-------------------------------------------------------------------------------
// Invoke a static data member.
   if ( fMemberBase ) return fMemberBase->Invoke( paramList );
   return Object();
}


/*/-------------------------------------------------------------------------------
  void ROOT::Reflex::Member::Set( const Object & instance,
  const Object & value ) const {
//--------------------------------------------------------------------------------
  if (fMemberBase ) fMemberBase->Set( instance, value );
  }
*/


//-------------------------------------------------------------------------------
void ROOT::Reflex::Member::Set( const Object & instance,
                                const void * value ) const {
//-------------------------------------------------------------------------------
// Set a non static data member.
   if (fMemberBase ) fMemberBase->Set( instance, value );
}


