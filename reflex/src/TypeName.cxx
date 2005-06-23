// @(#)root/reflex:$Name:$:$Id:$
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
typedef std::vector< ROOT::Reflex::TypeName * > TypeVec;


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
ROOT::Reflex::TypeName::TypeName( const char * Name,
                                  TypeBase * TypeBaseNth,
                                  const std::type_info * ti )
//-------------------------------------------------------------------------------
  : fName( Name ),
    fTypeBase( TypeBaseNth ) {
      sTypes() [ fName.c_str() ] = this;
      sTypeVec().push_back(this);
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
ROOT::Reflex::Type ROOT::Reflex::TypeName::TypeGet() const {
//-------------------------------------------------------------------------------
  return Type( this );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::TypeName::TypeNth( size_t nth ) {
//-------------------------------------------------------------------------------
  if ( nth < sTypeVec().size()) return Type(sTypeVec()[nth]);
  return Type();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::TypeName::TypeCount() {
//-------------------------------------------------------------------------------
  return sTypeVec().size();
}

