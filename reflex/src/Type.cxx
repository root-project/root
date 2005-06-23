// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/Type.h"

#include "Reflex/Object.h"
#include "Reflex/Scope.h"
#include "Reflex/Base.h"
#include "Reflex/MemberTemplate.h"
#include "Reflex/TypeTemplate.h"

#include "Enum.h"
#include "Union.h"
#include "Class.h"
#include "Reflex/Tools.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::Base ROOT::Reflex::Type::BaseNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->BaseNth( nth );
  return Base();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type
ROOT::Reflex::Type::ByName( const std::string & key ) {
//-------------------------------------------------------------------------------
  return TypeName::ByName( key );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type
ROOT::Reflex::Type::ByTypeInfo( const std::type_info & tid ) {
//-------------------------------------------------------------------------------
  return TypeName::ByTypeInfo( tid );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Object
ROOT::Reflex::Type::CastObject( const Type & to,
                                const Object & obj ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->CastObject( to, obj );
  return Object();
}


/*/-------------------------------------------------------------------------------
ROOT::Reflex::Object 
ROOT::Reflex::Type::Construct( const Type & signature,
                               const std::vector < Object > & values, 
                               void * mem ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->Construct( signature, 
                                                          values, 
                                                          mem ); 
  return Object();
}
*/


//-------------------------------------------------------------------------------
ROOT::Reflex::Object 
ROOT::Reflex::Type::Construct( const Type & signature,
                               const std::vector < void * > & values, 
                               void * mem ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->Construct( signature, 
                                                          values, 
                                                          mem ); 
  return Object();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::Type::DataMemberNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->DataMemberNth( nth );
  return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::Type::FunctionMemberNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->FunctionMemberNth( nth );
  return Member();
}


//-------------------------------------------------------------------------------
bool ROOT::Reflex::Type::IsEquivalentTo( const Type & TypeNth ) const {
//-------------------------------------------------------------------------------
  Type t1 = *this;
  Type t2 = TypeNth;
  while (t1.IsTypedef()) t1 = t1.ToType();
  while (t2.IsTypedef()) t2 = t2.ToType();
  switch ( t1.TypeType() ) {
  case CLASS:
    if ( t2.IsClass() )           return ( t1.fTypeName == t2.fTypeName ); 
  case FUNDAMENTAL:
    if ( t2.IsFundamental() )     return ( t1.fTypeName == t2.fTypeName );
  case UNION:
    if ( t2.IsUnion() )           return ( t1.fTypeName == t2.fTypeName ); 
  case ENUM:
    if ( t2.IsEnum() )            return ( t1.fTypeName == t2.fTypeName ); 
  case POINTER:
    if ( t2.IsPointer() )         return ( t1.ToType().IsEquivalentTo(t2.ToType()) );
  case POINTERTOMEMBER:
    if ( t2.IsPointerToMember() ) return ( t1.ToType().IsEquivalentTo(t2.ToType()) );
  case ARRAY:
    if ( t2.IsArray() )           return ( t1.ToType().IsEquivalentTo(t2.ToType()) && t1.Length() == t2.Length() );
  case FUNCTION:
    if ( t2.IsFunction() )        return true; // FIXME 
  default:
    return false;
  }
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::Type::MemberNth( const std::string & Name ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->MemberNth( Name );
  return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::Type::MemberNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->MemberNth( nth );
  return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplate ROOT::Reflex::Type::MemberTemplateNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->MemberTemplateNth( nth );
  return MemberTemplate();
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::Type::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
  std::string s = "";
  std::string cv = "";

  /** apply qualifications if wanted */
  if ( 0 != ( mod & ( QUALIFIED | Q ))) {
    if ( IsConst() && IsVolatile()) cv = "const volatile";
    else if ( IsConst())            cv = "const";
    else if ( IsVolatile())         cv = "volatile";
  }

  /** if TypeNth is not a pointer qualifiers can be put before */
  if ( cv.length() && TypeType() != POINTER ) s += cv + " ";
  
  /** use implemented names if available */
  if ( * this ) s += fTypeName->fTypeBase->Name( mod );
  /** otherwise use the TypeName */
  else {
    if ( fTypeName ) {
      /** unscoped TypeNth Name */
      if ( 0 != ( mod & ( SCOPED | S ))) s += fTypeName->Name();
      else  s += Tools::GetBaseName(fTypeName->Name());
    } 
    else { 
        return ""; 
    }
  }

  /** if TypeNth is a pointer qualifiers have to be after TypeNth */
  if ( cv.length() && TypeType() == POINTER ) s += " " + cv;

  /** apply reference if qualifications wanted */
  if ( ( 0 != ( mod & ( QUALIFIED | Q ))) && IsReference()) s += "&";

  return s;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope ROOT::Reflex::Type::ScopeGet() const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->ScopeGet();
  return Scope();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope ROOT::Reflex::Type::SubScopeNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->SubScopeNth( nth );
  return Scope();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::Type::SubTypeNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->SubTypeNth( nth );
  return Type();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::Type::TypeNth( size_t nth ) {
//-------------------------------------------------------------------------------
  return TypeName::TypeNth( nth );
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Type::TypeCount() {
//-------------------------------------------------------------------------------
  return TypeName::TypeCount();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplate ROOT::Reflex::Type::TypeTemplateNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( * this ) return fTypeName->fTypeBase->TypeTemplateNth( nth );
  return TypeTemplate();
}
