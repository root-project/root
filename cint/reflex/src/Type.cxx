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
#define REFLEX_BUILD
#endif

#include "Reflex/Type.h"

#include "Reflex/Object.h"
#include "Reflex/Scope.h"
#include "Reflex/Base.h"
#include "Reflex/MemberTemplate.h"
#include "Reflex/TypeTemplate.h"
#include "Reflex/DictionaryGenerator.h"

#include "Enum.h"
#include "Union.h"
#include "Class.h"
#include "Reflex/Tools.h"


//-------------------------------------------------------------------------------
Reflex::Type::operator Reflex::Scope () const {
//-------------------------------------------------------------------------------
// Conversion operator to Scope.
   if ( * this ) return *(fTypeName->fTypeBase);
   return Dummy::Scope();
}


//-------------------------------------------------------------------------------
Reflex::Base Reflex::Type::BaseAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return nth base info.
   if ( * this ) return fTypeName->fTypeBase->BaseAt( nth );
   return Dummy::Base();
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::Type::ByName( const std::string & key ) {
//-------------------------------------------------------------------------------
// Lookup a type by its fully qualified name.
   return TypeName::ByName( key );
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::Type::ByTypeInfo( const std::type_info & tid ) {
//-------------------------------------------------------------------------------
// Lookup a type by its type_info.
   return TypeName::ByTypeInfo( tid );
}


//-------------------------------------------------------------------------------
Reflex::Object
Reflex::Type::CastObject( const Type & to,
                                const Object & obj ) const {
//-------------------------------------------------------------------------------
// Cast the current type to "to" using the object obj.
   if ( * this ) return fTypeName->fTypeBase->CastObject( to, obj );
   return Dummy::Object();
}


/*/-------------------------------------------------------------------------------
  Reflex::Object 
  Reflex::Type::Construct( const Type & signature,
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
Reflex::Object
Reflex::Type::Construct( const Type & signature,
                               const std::vector < void * > & values, 
                               void * mem ) const {
//-------------------------------------------------------------------------------
// Construct this type and return it as an object. Signature can be used for overloaded
// constructors. Values is a collection of memory addresses of paramters. Mem the memory
// address for in place construction.
   if ( * this ) return fTypeName->fTypeBase->Construct( signature, 
                                                         values, 
                                                         mem ); 
   return Dummy::Object();
}


//-------------------------------------------------------------------------------
Reflex::Member Reflex::Type::DataMemberAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return the nth data member.
   if ( * this ) return fTypeName->fTypeBase->DataMemberAt( nth );
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
Reflex::Member Reflex::Type::DataMemberByName( const std::string & nam ) const {
//-------------------------------------------------------------------------------
// Return a data member by name.
   if ( * this ) return fTypeName->fTypeBase->DataMemberByName( nam );
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
Reflex::Type Reflex::Type::DynamicType( const Object & obj ) const {
//-------------------------------------------------------------------------------
// Return the dynamic type of this type.
   if ( * this ) return fTypeName->fTypeBase->DynamicType( obj );
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
Reflex::Member Reflex::Type::FunctionMemberAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return the nth function member.
   if ( * this ) return fTypeName->fTypeBase->FunctionMemberAt( nth );
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
Reflex::Member Reflex::Type::FunctionMemberByName( const std::string & nam,
                                                   const Type & signature,
                                                   unsigned int modifiers_mask /*= 0*/) const {
//-------------------------------------------------------------------------------
// Return a function member by name. Signature can be used for overloaded functions.
   if ( * this ) return fTypeName->fTypeBase->FunctionMemberByName( nam, signature, modifiers_mask );
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
bool Reflex::Type::HasBase( const Type & cl ) const {
//-------------------------------------------------------------------------------
   // Return base info if type has base cl.
   if ( * this ) return fTypeName->fTypeBase->HasBase( cl );
   return false;
}


//-------------------------------------------------------------------------------
bool Reflex::Type::IsEquivalentTo( const Type & typ, unsigned int modifiers_mask ) const {
//-------------------------------------------------------------------------------
// Check if two types are equivalent. It will compare the information of the type
// depending on the TypeType.
   if ( *this == typ ) return true;

   Type t1 = *this;
   Type t2 = typ;

   unsigned int mod1 = t1.fModifiers | modifiers_mask;
   unsigned int mod2 = t2.fModifiers | modifiers_mask;

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
      case STRUCT:
      case TYPETEMPLATEINSTANCE:
         if ( t2.IsClass() )           return ( t1.fTypeName == t2.fTypeName ); 
      case FUNDAMENTAL:
         if ( t2.IsFundamental() )     return ( t1.fTypeName == t2.fTypeName );
      case UNION:
         if ( t2.IsUnion() )           return ( t1.fTypeName == t2.fTypeName ); 
      case ENUM:
         if ( t2.IsEnum() )            return ( t1.fTypeName == t2.fTypeName ); 
      case POINTER:
         if ( t2.IsPointer() )         return ( t1.ToType().IsEquivalentTo(t2.ToType(),modifiers_mask) );
      case POINTERTOMEMBER:
         if ( t2.IsPointerToMember() ) return ( t1.ToType().IsEquivalentTo(t2.ToType(),modifiers_mask) );
      case ARRAY:
         if ( t2.IsArray() )           return ( t1.ToType().IsEquivalentTo(t2.ToType(),modifiers_mask) && t1.ArrayLength() == t2.ArrayLength() );
      case FUNCTION:
         if ( t2.IsFunction() ) {

            if ( t1.ReturnType().IsEquivalentTo(t2.ReturnType(),modifiers_mask)) {

               if ( t1.FunctionParameterSize() == t2.FunctionParameterSize() ) {

                  Type_Iterator pi1;
                  Type_Iterator pi2;
                  for ( pi1 = t1.FunctionParameter_Begin(), pi2 = t2.FunctionParameter_Begin(); 
                        pi1 != t1.FunctionParameter_End(),  pi2 != t2.FunctionParameter_End(); 
                        ++pi1, ++pi2 ) {

                     if ( ! pi1->IsEquivalentTo(*pi2,modifiers_mask)) return false;

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
bool Reflex::Type::IsSignatureEquivalentTo( const Type & typ, unsigned int modifiers_mask ) const {
//-------------------------------------------------------------------------------
// Check if two function types are equivalent, ignoring the return type
// for functions. It will compare the information of the type depending
// on the TypeType.
   if ( *this == typ ) return true;

   Type t1 = *this;
   Type t2 = typ;

   unsigned int mod1 = t1.fModifiers | modifiers_mask;
   unsigned int mod2 = t2.fModifiers | modifiers_mask;

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
      case STRUCT:
      case TYPETEMPLATEINSTANCE:
         if ( t2.IsClass() )           return ( t1.fTypeName == t2.fTypeName ); 
      case FUNDAMENTAL:
         if ( t2.IsFundamental() )     return ( t1.fTypeName == t2.fTypeName );
      case UNION:
         if ( t2.IsUnion() )           return ( t1.fTypeName == t2.fTypeName ); 
      case ENUM:
         if ( t2.IsEnum() )            return ( t1.fTypeName == t2.fTypeName ); 
      case POINTER:
         if ( t2.IsPointer() )         return ( t1.ToType().IsEquivalentTo(t2.ToType(),modifiers_mask) );
      case POINTERTOMEMBER:
         if ( t2.IsPointerToMember() ) return ( t1.ToType().IsEquivalentTo(t2.ToType(),modifiers_mask) );
      case ARRAY:
         if ( t2.IsArray() )           return ( t1.ToType().IsEquivalentTo(t2.ToType(),modifiers_mask) && t1.ArrayLength() == t2.ArrayLength() );
      case FUNCTION:
         if ( t2.IsFunction() ) {

            if ( t1.FunctionParameterSize() == t2.FunctionParameterSize() ) {

               Type_Iterator pi1;
               Type_Iterator pi2;
               for ( pi1 = t1.FunctionParameter_Begin(), pi2 = t2.FunctionParameter_Begin(); 
                     pi1 != t1.FunctionParameter_End(),  pi2 != t2.FunctionParameter_End(); 
                     ++pi1, ++pi2 ) {

                  if ( ! pi1->IsEquivalentTo(*pi2,modifiers_mask)) return false;

               }
               return true;
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
Reflex::Member Reflex::Type::MemberByName( const std::string & nam,
                                                       const Type & signature ) const {
//-------------------------------------------------------------------------------
// Return a member by name. Signature is optional for overloaded function members.
   if ( * this ) return fTypeName->fTypeBase->MemberByName( nam, signature );
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
Reflex::Member Reflex::Type::MemberAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return the nth member.
   if ( * this ) return fTypeName->fTypeBase->MemberAt( nth );
   return Dummy::Member();
}


//-------------------------------------------------------------------------------
Reflex::MemberTemplate Reflex::Type::MemberTemplateAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return the nth member template.
   if ( * this ) return fTypeName->fTypeBase->MemberTemplateAt( nth );
   return Dummy::MemberTemplate();
}


//-------------------------------------------------------------------------------
std::string Reflex::Type::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
// Return the name of the type (qualified and scoped if requested)

   if (( mod & ( QUALIFIED | Q ))==0 && (* this) ) { 
      // most common case
      return fTypeName->fTypeBase->Name( mod );
   }

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
Reflex::Scope Reflex::Type::PointerToMemberScope() const {
//-------------------------------------------------------------------------------
   // Return the scope of the pointer to member type
   if ( * this ) return fTypeName->fTypeBase->PointerToMemberScope();
   return Dummy::Scope();
}


//-------------------------------------------------------------------------------
Reflex::Scope Reflex::Type::SubScopeAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return the nth sub scope.
   if ( * this ) return fTypeName->fTypeBase->SubScopeAt( nth );
   return Dummy::Scope();
}


//-------------------------------------------------------------------------------
Reflex::Type Reflex::Type::SubTypeAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return the nth sub type.
   if ( * this ) return fTypeName->fTypeBase->SubTypeAt( nth );
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
Reflex::Type Reflex::Type::TypeAt( size_t nth ) {
//-------------------------------------------------------------------------------
// Return the nth type defined in Reflex.
   return TypeName::TypeAt( nth );
}


//-------------------------------------------------------------------------------
size_t Reflex::Type::TypeSize() {
//-------------------------------------------------------------------------------
// Return the number of types defined in Reflex.
   return TypeName::TypeSize();
}


//-------------------------------------------------------------------------------
Reflex::TypeTemplate Reflex::Type::SubTypeTemplateAt( size_t nth ) const {
//-------------------------------------------------------------------------------
// Return the nth sub type template.
   if ( * this ) return fTypeName->fTypeBase->SubTypeTemplateAt( nth );
   return Dummy::TypeTemplate();
}


//------------------------------------------------------------------------------
void Reflex::Type::GenerateDict( DictionaryGenerator & generator) const {
//------------------------------------------------------------------------------
// Generate Dictionary information about itself.
   if ( * this ) fTypeName->fTypeBase->GenerateDict( generator );
}


//-------------------------------------------------------------------------------
void Reflex::Type::Unload() const {
//-------------------------------------------------------------------------------
//  Unload a type, i.e. delete the TypeName's TypeBase object.
   if ( * this ) delete fTypeName->fTypeBase;
}

#ifdef REFLEX_CINT_MERGE
bool Reflex::Type::operator&&(const Scope &right) const
{ return operator bool() && (bool)right; }
bool Reflex::Type::operator&&(const Type &right) const 
{ return operator bool() && (bool)right; }
bool Reflex::Type::operator&&(const Member &right) const 
{ return operator bool() && (bool)right; }
bool Reflex::Type::operator||(const Scope &right) const 
{ return operator bool() && (bool)right; }
bool Reflex::Type::operator||(const Type &right) const 
{ return operator bool() && (bool)right; }
bool Reflex::Type::operator||(const Member &right) const 
{ return operator bool() || (bool)right; }
#endif
