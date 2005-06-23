// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/TypeBase.h"

#include "Reflex/Type.h"
#include "Reflex/PropertyList.h"
#include "Reflex/Object.h"
#include "Reflex/Scope.h"
#include "Reflex/TypeName.h"
#include "Reflex/Base.h"
#include "Reflex/TypeTemplate.h"

#include "Array.h"
#include "Pointer.h"
#include "PointerToMember.h"
#include "Union.h"
#include "Enum.h"
#include "Fundamental.h"
#include "Function.h"
#include "Class.h"
#include "Typedef.h"
#include "ClassTemplateInstance.h"
#include "FunctionMemberTemplateInstance.h"
#include "Reflex/Tools.h"

#include <iostream>

//-------------------------------------------------------------------------------
ROOT::Reflex::TypeBase::TypeBase( const char * Name, 
                                  size_t size,
                                  TYPE TypeType, 
                                  const std::type_info & ti ) 
//-------------------------------------------------------------------------------
  : fScope( Scope::__NIRVANA__ ),
    fSize( size ),
    fTypeInfo( ti ), 
    fTypeType( TypeType ),
    fPropertyList( PropertyList(new PropertyListImpl())),
    fBasePosition(Tools::GetBasePosition( Name)) {

  Type t = TypeName::ByName( Name );
  if ( t.Id() == 0 ) { 
    fTypeName = new TypeName( Name, this, &ti ); 
  }
  else {
    fTypeName = (TypeName*)t.Id();
    if ( t.Id() != TypeName::ByTypeInfo(ti).Id()) fTypeName->SetTypeId( ti );
    if ( fTypeName->fTypeBase ) delete fTypeName->fTypeBase;
    fTypeName->fTypeBase = this;
  }

  if ( TypeType != FUNDAMENTAL && 
       TypeType != FUNCTION &&
       TypeType != POINTER  ) {
    std::string sname = Tools::GetScopeName(Name);
    fScope = Scope::ByName(sname);
    if ( fScope.Id() == 0 ) fScope = (new ScopeName(sname.c_str(), 0))->ScopeGet();
    
    // Set declaring ScopeNth
    if ( fScope ) fScope.AddSubType(TypeGet());
  }
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeBase::~TypeBase( ) {
//-------------------------------------------------------------------------------
  if( fTypeName->fTypeBase == this ) fTypeName->fTypeBase = 0;
  fPropertyList.ClearProperties();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeBase::operator ROOT::Reflex::Type () const {
//-------------------------------------------------------------------------------
  return Type( fTypeName );
}


//-------------------------------------------------------------------------------
void * ROOT::Reflex::TypeBase::Allocate() const {
//-------------------------------------------------------------------------------
  return operator new( fSize );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Base ROOT::Reflex::TypeBase::BaseNth( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
  throw RuntimeError("Type does not represent a Class/Struct");
  return Base();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::TypeBase::BaseCount() const {
//-------------------------------------------------------------------------------
  throw RuntimeError("Type does not represent a Class/Struct");
  return 0;
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::TypeBase::Deallocate( void * instance ) const {
//-------------------------------------------------------------------------------
  operator delete( instance );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Object ROOT::Reflex::TypeBase::CastObject( const Type & /* to */,
                                                         const Object & /* obj */ ) const {
//-------------------------------------------------------------------------------
  throw RuntimeError("This function can only be called on Class/Struct");
  return Object();
}


//-------------------------------------------------------------------------------
//ROOT::Reflex::Object 
//ROOT::Reflex::TypeBase::Construct( const Type &  /*signature*/,
//                                   std::vector < Object > /*values*/, 
//                                   void * /*mem*/ ) const {
//-------------------------------------------------------------------------------
//  return Object(TypeGet(), Allocate());
//}


//-------------------------------------------------------------------------------
ROOT::Reflex::Object 
ROOT::Reflex::TypeBase::Construct( const Type &  /*signature*/,
                                   std::vector < void * > /*values*/, 
                                   void * /*mem*/ ) const {
//-------------------------------------------------------------------------------
  return Object(TypeGet(), Allocate());
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::TypeBase::DataMemberNth( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
  return Member();
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::TypeBase::Destruct( void * instance, 
                                       bool dealloc ) const {
//-------------------------------------------------------------------------------
  if ( dealloc ) Deallocate(instance);
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::TypeBase::DynamicType( const Object & /* obj */ ) const {
//-------------------------------------------------------------------------------
  throw RuntimeError("This function can only be called on Class/Struct");
  return Type();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::TypeBase::FunctionMemberNth( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
  return Member();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::TypeBase::Length() const {
//-------------------------------------------------------------------------------
  return 0;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::TypeBase::MemberNth( const std::string & /* Name */ ) const {
//-------------------------------------------------------------------------------
  return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::TypeBase::MemberNth( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
  return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplate ROOT::Reflex::TypeBase::MemberTemplateNth( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
  return MemberTemplate();
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::TypeBase::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
  if ( 0 != ( mod & ( SCOPED | S ))) return fTypeName->Name();
  return std::string(fTypeName->Name(), fBasePosition);
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::TypeBase::ParameterNth( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
  return Type();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::TypeBase::ParameterCount() const {
//-------------------------------------------------------------------------------
  return 0;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::PropertyList ROOT::Reflex::TypeBase::PropertyListGet() const {
//-------------------------------------------------------------------------------
  return fPropertyList;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::TypeBase::ReturnType() const {
//-------------------------------------------------------------------------------
  return Type();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope ROOT::Reflex::TypeBase::ScopeGet() const {
//-------------------------------------------------------------------------------
  return fScope;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope ROOT::Reflex::TypeBase::SubScopeNth( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
  return Scope();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::TypeBase::SubTypeNth( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
  return Type();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::TypeBase::TemplateArgumentNth( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
  return Type();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::TypeBase::ToType() const {
//-------------------------------------------------------------------------------
  return Type();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::TypeBase::TypeGet() const {
//-------------------------------------------------------------------------------
  return fTypeName->TypeGet();
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::TypeBase::TypeTypeAsString() const {
//-------------------------------------------------------------------------------
  switch ( fTypeType ) {
  case CLASS:
    return "CLASS";
    break;
  case ENUM:
    return "ENUM";
    break;
  case FUNCTION:
    return "FUNCTION";
    break;
  case ARRAY:
    return "ARRAY";
    break;
  case FUNDAMENTAL:
    return "FUNDAMENTAL";
    break;
  case POINTER:
    return "POINTER";
    break;
  case REFERENCE:
    return "REFERENCE";
    break;
  case TYPEDEF:
    return "TYPEDEF";
    break;
  case TYPETEMPLATEINSTANCE:
    return "TYPETEMPLATEINSTANCE";
    break;
  case MEMBERTEMPLATEINSTANCE:
    return "MEMBERTEMPLATEINSTANCE";
    break;
  case UNRESOLVED:
    return "UNRESOLVED";
    break;
  default:
    return "Type " + Name() + "is not assigned to a TYPE";
  }
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplate ROOT::Reflex::TypeBase::TypeTemplateNth( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
  return TypeTemplate();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TYPE ROOT::Reflex::TypeBase::TypeType() const {
//-------------------------------------------------------------------------------
  return fTypeType;
}
