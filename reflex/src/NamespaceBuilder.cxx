// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/Builder/NamespaceBuilder.h"

#include "Reflex/PropertyList.h"
#include "Reflex/Scope.h"

#include "Namespace.h"


//-------------------------------------------------------------------------------
ROOT::Reflex::NamespaceBuilder::NamespaceBuilder( const char * Name ) {
//-------------------------------------------------------------------------------
  Scope ScopeNth = Scope::ByName( Name );
  if ( ScopeNth && ScopeNth.IsNamespace() ) {
    fNamespace       = ScopeNth;
  }
  else {
    fNamespace       = (new Namespace( Name ))->ScopeGet();
  }
}


//-------------------------------------------------------------------------------
ROOT::Reflex::NamespaceBuilder & 
ROOT::Reflex::NamespaceBuilder::AddProperty( const char * key, 
                                             const char * value ) {
//-------------------------------------------------------------------------------
  fNamespace.PropertyListGet().AddProperty( key , value );
  return * this;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::NamespaceBuilder & 
ROOT::Reflex::NamespaceBuilder::AddProperty( const char * key, 
                                             Any value ) {
//-------------------------------------------------------------------------------
  fNamespace.PropertyListGet().AddProperty( key , value );
  return * this;
}
    
