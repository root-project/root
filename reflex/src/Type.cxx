// @(#)root/reflex:$Name:  $:$Id: Type.cxx,v 1.8 2006/03/20 09:46:18 roiser Exp $
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
ROOT::Reflex::Type::operator ROOT::Reflex::Scope () const {
//-------------------------------------------------------------------------------
   if ( * this ) return *(fTypeName->fTypeBase);
   return Scope();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Base ROOT::Reflex::Type::BaseAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->BaseAt( nth );
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
ROOT::Reflex::Member ROOT::Reflex::Type::DataMemberAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->DataMemberAt( nth );
   return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::Type::DataMemberByName( const std::string & nam ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->DataMemberByName( nam );
   return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::Type::DynamicType( const Object & obj ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->DynamicType( obj );
   return Type();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::Type::FunctionMemberAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->FunctionMemberAt( nth );
   return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::Type::FunctionMemberByName( const std::string & nam,
                                                               const Type & signature ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->FunctionMemberByName( nam, signature );
   return Member();
}


//-------------------------------------------------------------------------------
bool ROOT::Reflex::Type::IsEquivalentTo( const Type & typ ) const {
  //-------------------------------------------------------------------------------
  Type t1 = *this;
  Type t2 = typ;
  if (!t1.Id() & !t2.Id()) return true;

  unsigned int mod1 = t1.fModifiers;
  unsigned int mod2 = t2.fModifiers;

  while (t1.IsTypedef()) { 
    t1 = t1.ToType();
    mod1 |= t1.fModifiers;
  }
  while ( t2.IsTypedef()) {
    t2 = t2.ToType();
    mod2 |= t2.fModifiers;
  }

  if (mod1 == mod2) {

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
      if ( t2.IsArray() )           return ( t1.ToType().IsEquivalentTo(t2.ToType()) && t1.ArrayLength() == t2.ArrayLength() );
    case FUNCTION:
      if ( t2.IsFunction() ) {

        if ( t1.ReturnType().IsEquivalentTo(t2.ReturnType())) {

          if ( t1.FunctionParameterSize() == t2.FunctionParameterSize() ) {

            Type_Iterator pi1;
            Type_Iterator pi2;
            for ( pi1 = t1.FunctionParameter_Begin(), pi2 = t2.FunctionParameter_Begin(); 
                  pi1 != t1.FunctionParameter_End(),  pi2 != t2.FunctionParameter_End(); 
                  ++pi1, ++pi2 ) {

              if ( ! pi1->IsEquivalentTo(*pi2)) return false;

            }
            return true;
          }
        }
        return false;
      }
    default:
      return false;
    }
  }
  else {
    return false;
  }
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::Type::MemberByName( const std::string & nam,
                                                       const Type & signature ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->MemberByName( nam, signature );
   return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member ROOT::Reflex::Type::MemberAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->MemberAt( nth );
   return Member();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplate ROOT::Reflex::Type::MemberTemplateAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->MemberTemplateAt( nth );
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

   /** if At is not a pointer qualifiers can be put before */
   if ( cv.length() && TypeType() != POINTER && TypeType() != FUNCTION ) s += cv + " ";
  
   /** use implemented names if available */
   if ( * this ) s += fTypeName->fTypeBase->Name( mod );
   /** otherwise use the TypeName */
   else {
      if ( fTypeName ) {
         /** unscoped At Name */
         if ( 0 != ( mod & ( SCOPED | S ))) s += fTypeName->Name();
         else  s += Tools::GetBaseName(fTypeName->Name());
      } 
      else { 
         return ""; 
      }
   }

   /** if At is a pointer qualifiers have to be after At */
   if ( cv.length() && ( TypeType() == POINTER || TypeType() == FUNCTION) ) s += " " + cv;

   /** apply reference if qualifications wanted */
   if ( ( 0 != ( mod & ( QUALIFIED | Q ))) && IsReference()) s += "&";

   return s;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope ROOT::Reflex::Type::SubScopeAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->SubScopeAt( nth );
   return Scope();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::Type::SubTypeAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->SubTypeAt( nth );
   return Type();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::Type::TypeAt( size_t nth ) {
//-------------------------------------------------------------------------------
   return TypeName::TypeAt( nth );
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Type::TypeSize() {
//-------------------------------------------------------------------------------
   return TypeName::TypeSize();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplate ROOT::Reflex::Type::SubTypeTemplateAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   if ( * this ) return fTypeName->fTypeBase->SubTypeTemplateAt( nth );
   return TypeTemplate();
}
