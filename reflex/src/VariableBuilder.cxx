// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/Builder/VariableBuilder.h"

#include "Namespace.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::VariableBuilderImpl::VariableBuilderImpl( const char * Name,
                                                        const Type & TypeNth,
                                                        size_t Offset,
                                                        unsigned int modifiers ) 
//-------------------------------------------------------------------------------
  : fDataMember(Member()) {

  std::string declScope = "";
  std::string memName = std::string(Name);
  size_t pos = memName.rfind( "::" );
  if ( pos != std::string::npos ) {
    declScope = memName.substr( 0, pos );
    memName = memName.substr( pos + 2 );
  }

  Scope ScopeNth = Scope::ByName(declScope);
  
  if ( ! ScopeNth ) {
    ScopeNth = (new Namespace(declScope.c_str()))->ScopeGet();
  }
  
  if ( ! ScopeNth.IsNamespace()) throw RuntimeError("Declaring ScopeNth is not a namespace");

  ScopeNth.AddDataMember( memName.c_str(),
                       TypeNth,
                       Offset,
                       modifiers );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::VariableBuilderImpl::~VariableBuilderImpl() {
  //-------------------------------------------------------------------------------
  FireFunctionCallback( fDataMember );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::VariableBuilderImpl::AddProperty( const char * key, 
                                                     const char * value ) {
  //-------------------------------------------------------------------------------
  fDataMember.PropertyListGet().AddProperty( key , value );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::VariableBuilderImpl::AddProperty( const char * key, 
                                                     Any value ) {
  //-------------------------------------------------------------------------------
  fDataMember.PropertyListGet().AddProperty( key , value );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::VariableBuilder::VariableBuilder( const char * Name, 
                                                const Type & TypeNth,
                                                size_t Offset,
                                                unsigned int modifiers) 
//-------------------------------------------------------------------------------
  : fDataMember( Member()) {

  std::string sname = std::string( Name );
  size_t pos = sname.rfind( "::" );
  std::string declScope = sname.substr( pos + 2 );
  std::string memName = sname.substr( 0, pos );
  
  Scope ScopeNth = Scope::ByName(declScope);
  
  if ( ! ScopeNth ) {
    ScopeNth = (new Namespace(declScope.c_str()))->ScopeGet();
  }
  
  if ( ! ScopeNth.IsNamespace()) throw RuntimeError("Declaring ScopeNth is not a namespace");

  ScopeNth.AddDataMember( memName.c_str(),
                       TypeNth,
                       Offset,
                       modifiers );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::VariableBuilder::~VariableBuilder() {
  //-------------------------------------------------------------------------------
  FireFunctionCallback( fDataMember );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::VariableBuilder & 
ROOT::Reflex::VariableBuilder::AddProperty( const char * key, 
                                            const char * value ) {
  //-------------------------------------------------------------------------------
  fDataMember.PropertyListGet().AddProperty( key , value );
  return * this;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::VariableBuilder & 
ROOT::Reflex::VariableBuilder::AddProperty( const char * key, 
                                            Any value ) {
  //-------------------------------------------------------------------------------
  fDataMember.PropertyListGet().AddProperty( key , value );
  return * this;
}

