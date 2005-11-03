// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

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
  : fMemberBase( memberBase ) {}



//-------------------------------------------------------------------------------
ROOT::Reflex::Member::Member( const Member & rh )
//-------------------------------------------------------------------------------
  : fMemberBase( rh.fMemberBase ) {}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member::~Member() {
//-------------------------------------------------------------------------------
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Object ROOT::Reflex::Member::Get() const {
//-------------------------------------------------------------------------------
  if ( fMemberBase ) return fMemberBase->Get( Object());
  return Object();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Object ROOT::Reflex::Member::Get( const Object & obj) const {
//-------------------------------------------------------------------------------
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
  if ( fMemberBase ) return fMemberBase->Invoke( paramList );
  return Object();
}


/*/-------------------------------------------------------------------------------
void ROOT::Reflex::Member::Set( const Object & instance,
                                const Object & value ) const {
//-------------------------------------------------------------------------------
  if (fMemberBase ) fMemberBase->Set( instance, value );
}
*/


//-------------------------------------------------------------------------------
void ROOT::Reflex::Member::Set( const Object & instance,
                                const void * value ) const {
//-------------------------------------------------------------------------------
  if (fMemberBase ) fMemberBase->Set( instance, value );
}


