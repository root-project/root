// @(#)root/reflex:$Name:  $:$Id: TypeName.cxx,v 1.2 2005/11/03 15:24:40 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/TypeName.h"

#include "Reflex/Type.h"

#include "stl_hash.h"
#include <vector>


//-------------------------------------------------------------------------------
typedef __gnu_cxx::hash_map<const char *, ROOT::Reflex::TypeName * > Name2Type;
typedef __gnu_cxx::hash_map<const char *, ROOT::Reflex::TypeName * > TypeId2Type;
typedef std::vector< ROOT::Reflex::Type > TypeVec;


//-------------------------------------------------------------------------------
Name2Type & sTypes() {
//-------------------------------------------------------------------------------
  static Name2Type m;
  return m;
}


//-------------------------------------------------------------------------------
TypeId2Type & sTypeInfos() {
//-------------------------------------------------------------------------------
  static TypeId2Type m;
  return m;
}


//-------------------------------------------------------------------------------
TypeVec & sTypeVec() {
//-------------------------------------------------------------------------------
  static TypeVec m;
  return m;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeName::TypeName( const char * nam,
                                  TypeBase * typeBas,
                                  const std::type_info * ti )
//-------------------------------------------------------------------------------
  : fName( nam ),
    fTypeBase( typeBas ) {
      sTypes() [ fName.c_str() ] = this;
      sTypeVec().push_back(Type(this));
      if ( ti ) sTypeInfos() [ ti->name() ] = this;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeName::~TypeName() {
//-------------------------------------------------------------------------------
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::TypeName::SetTypeId( const std::type_info & ti ) const {
//-------------------------------------------------------------------------------
  sTypeInfos() [ ti.name() ] = const_cast<TypeName*>(this);
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type
ROOT::Reflex::TypeName::ByName( const std::string & key ) {
//-------------------------------------------------------------------------------
  Name2Type::iterator it = sTypes().find(key.c_str());
  if( it != sTypes().end() ) return Type( it->second );
  else                       return Type();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type
ROOT::Reflex::TypeName::ByTypeInfo( const std::type_info & ti ) {
//-------------------------------------------------------------------------------
  TypeId2Type::iterator it = sTypeInfos().find(ti.name());
  if( it != sTypeInfos().end() ) return Type( it->second );
  else                           return Type();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::TypeName::ThisType() const {
//-------------------------------------------------------------------------------
  return Type( this );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::TypeName::TypeAt( size_t nth ) {
//-------------------------------------------------------------------------------
  if ( nth < sTypeVec().size()) return sTypeVec()[nth];
  return Type();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::TypeName::TypeSize() {
//-------------------------------------------------------------------------------
  return sTypeVec().size();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type_Iterator ROOT::Reflex::TypeName::Type_Begin() {
//-------------------------------------------------------------------------------
  return sTypeVec().begin();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type_Iterator ROOT::Reflex::TypeName::Type_End() {
//-------------------------------------------------------------------------------
  return sTypeVec().end();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::TypeName::Type_RBegin() {
//-------------------------------------------------------------------------------
  return sTypeVec().rbegin();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_Type_Iterator ROOT::Reflex::TypeName::Type_REnd() {
//-------------------------------------------------------------------------------
  return sTypeVec().rend();
}


