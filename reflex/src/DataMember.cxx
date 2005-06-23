// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "DataMember.h"

#include "Reflex/Scope.h"
#include "Reflex/Object.h"
#include "Reflex/Member.h"

#include "Reflex/Tools.h"
#include "Class.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::DataMember::DataMember( const char *  Name,
                                      const Type &  TypeNth,
                                      size_t        Offset,
                                      unsigned int  modifiers )
//-------------------------------------------------------------------------------
  : MemberBase ( Name, TypeNth, DATAMEMBER, modifiers ),
    fOffset( Offset) { }


//-------------------------------------------------------------------------------
ROOT::Reflex::DataMember::~DataMember() {}
//-------------------------------------------------------------------------------


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::DataMember::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
  std::string s = "";

  if ( 0 != ( mod & ( QUALIFIED | Q ))) {
    if ( IsPublic())          { s += "public ";    }
    if ( IsProtected())       { s += "protected "; }
    if ( IsPrivate())         { s += "private ";   }
    if ( IsExtern())          { s += "extern ";    }
    if ( IsStatic())          { s += "static ";    }
    if ( IsAuto())            { s += "auto ";      }
    if ( IsRegister())        { s += "register ";  }
    if ( IsMutable())         { s += "mutable ";   }
  }

  if ( ScopeGet().IsEnum()) {
    if (ScopeGet().DeclaringScope()) {
      std::string sc = ScopeGet().DeclaringScope().Name(SCOPED);
      if ( sc != "::" ) s += sc + "::";
    }
    s += MemberBase::Name( mod & FINAL & QUALIFIED );
  }
  else {
    s += MemberBase::Name( mod );
  }

  return s;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Object ROOT::Reflex::DataMember::Get( const Object & obj ) const {
//-------------------------------------------------------------------------------
  if (ScopeGet().ScopeType() == ENUM ) {
    return Object(Type::ByName("int"), (void*)&fOffset);
  }
  else {
    void * mem = CalculateBaseObject( obj );
    mem = (char*)mem + Offset();
    return Object(TypeGet(),mem);
  }
}


/*/-------------------------------------------------------------------------------
void ROOT::Reflex::DataMember::Set( const Object & instance,
                                    const Object & value ) const {
//-------------------------------------------------------------------------------
  void * mem = CalculateBaseObject( instance );
  mem = (char*)mem + Offset();
  if (TypeGet().IsClass() ) {
    // Should use the asigment operator if exists (FIX-ME)
    memcpy( mem, value.AddressGet(), TypeGet().SizeOf());
  }
  else {
    memcpy( mem, value.AddressGet(), TypeGet().SizeOf() );
  }
}
*/


//-------------------------------------------------------------------------------
void ROOT::Reflex::DataMember::Set( const Object & instance,
                                    const void * value ) const {
//-------------------------------------------------------------------------------
  void * mem = CalculateBaseObject( instance );
  mem = (char*)mem + Offset();
  if (TypeGet().IsClass() ) {
    // Should use the asigment operator if exists (FIX-ME)
    memcpy( mem, value, TypeGet().SizeOf());
  }
  else {
    memcpy( mem, value, TypeGet().SizeOf() );
  }
}
