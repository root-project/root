// @(#)root/reflex:$Name:  $:$Id: TypeBase.cxx,v 1.9 2006/04/12 10:21:11 roiser Exp $
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
ROOT::Reflex::TypeBase::TypeBase( const char * nam, 
                                  size_t size,
                                  TYPE typeTyp, 
                                  const std::type_info & ti ) 
   : fScope( Scope::fg__NIRVANA__ ),
     fSize( size ),
     fTypeInfo( ti ), 
     fTypeType( typeTyp ),
     fPropertyList( PropertyList(new PropertyListImpl())),
     fBasePosition(Tools::GetBasePosition( nam)) {
//-------------------------------------------------------------------------------

   Type t = TypeName::ByName( nam );
   if ( t.Id() == 0 ) { 
      fTypeName = new TypeName( nam, this, &ti ); 
   }
   else {
      fTypeName = (TypeName*)t.Id();
      if ( t.Id() != TypeName::ByTypeInfo(ti).Id()) fTypeName->SetTypeId( ti );
      if ( fTypeName->fTypeBase ) delete fTypeName->fTypeBase;
      fTypeName->fTypeBase = this;
   }

   if ( typeTyp != FUNDAMENTAL && 
        typeTyp != FUNCTION &&
        typeTyp != POINTER  ) {
      std::string sname = Tools::GetScopeName(nam);
      fScope = Scope::ByName(sname);
      if ( fScope.Id() == 0 ) fScope = (new ScopeName(sname.c_str(), 0))->ThisScope();
    
      // Set declaring At
      if ( fScope ) fScope.AddSubType(ThisType());
   }
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeBase::~TypeBase( ) {
//-------------------------------------------------------------------------------
   if( fTypeName->fTypeBase == this ) fTypeName->fTypeBase = 0;
   fPropertyList.ClearProperties();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeBase::operator ROOT::Reflex::Scope() const {
//-------------------------------------------------------------------------------
   switch ( fTypeType ) {
   case CLASS:
   case TYPETEMPLATEINSTANCE:
   case UNION:
   case ENUM:
      return *(dynamic_cast<const ScopeBase*>(this));
   default:
      return Scope();
   }
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeBase::operator ROOT::Reflex::Type () const {
//-------------------------------------------------------------------------------
   return Type( fTypeName );
}


//-------------------------------------------------------------------------------
void * ROOT::Reflex::TypeBase::Allocate() const {
//-------------------------------------------------------------------------------
   return malloc( fSize );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Base ROOT::Reflex::TypeBase::BaseAt( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
   throw RuntimeError("Type does not represent a Class/Struct");
   return Base();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::TypeBase::BaseSize() const {
//-------------------------------------------------------------------------------
   throw RuntimeError("Type does not represent a Class/Struct");
   return 0;
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::TypeBase::Deallocate( void * instance ) const {
//-------------------------------------------------------------------------------
   free( instance );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope ROOT::Reflex::TypeBase::DeclaringScope() const {
//-------------------------------------------------------------------------------
   return fScope;
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
//                                   const std::vector < Object > & /*values*/, 
//                                   void * /*mem*/ ) const {
//-------------------------------------------------------------------------------
//  return Object(ThisType(), Allocate());
//}


//-------------------------------------------------------------------------------
ROOT::Reflex::Object 
ROOT::Reflex::TypeBase::Construct( const Type &  /*signature*/,
                                   const std::vector < void * > & /*values*/, 
                                   void * /*mem*/ ) const {
//-------------------------------------------------------------------------------
   return Object(ThisType(), Allocate());
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::TypeBase::DataMemberAt( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
   return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::TypeBase::DataMemberByName( const std::string & /* nam */ ) const {
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
ROOT::Reflex::Member ROOT::Reflex::TypeBase::FunctionMemberAt( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
   return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::TypeBase::FunctionMemberByName( const std::string & /* nam */,
                                                                   const Type & /* signature */ ) const {
//-------------------------------------------------------------------------------
   return Member();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::TypeBase::ArrayLength() const {
//-------------------------------------------------------------------------------
   return 0;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::TypeBase::MemberByName( const std::string & /* nam */,
                                                           const Type & /* signature */) const {
//-------------------------------------------------------------------------------
   return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::TypeBase::MemberAt( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
   return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplate ROOT::Reflex::TypeBase::MemberTemplateAt( size_t /* nth */ ) const {
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
ROOT::Reflex::Type ROOT::Reflex::TypeBase::FunctionParameterAt( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
   return Type();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::TypeBase::FunctionParameterSize() const {
//-------------------------------------------------------------------------------
   return 0;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::PropertyList ROOT::Reflex::TypeBase::Properties() const {
//-------------------------------------------------------------------------------
   return fPropertyList;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::TypeBase::ReturnType() const {
//-------------------------------------------------------------------------------
   return Type();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope ROOT::Reflex::TypeBase::SubScopeAt( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
   return Scope();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::TypeBase::SubTypeAt( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
   return Type();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::TypeBase::TemplateArgumentAt( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
   return Type();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::TypeBase::ToType( unsigned int /* mod */ ) const {
//-------------------------------------------------------------------------------
   return Type();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::TypeBase::ThisType() const {
//-------------------------------------------------------------------------------
   return fTypeName->ThisType();
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
ROOT::Reflex::TypeTemplate ROOT::Reflex::TypeBase::SubTypeTemplateAt( size_t /* nth */ ) const {
//-------------------------------------------------------------------------------
   return TypeTemplate();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TYPE ROOT::Reflex::TypeBase::TypeType() const {
//-------------------------------------------------------------------------------
   return fTypeType;
}
