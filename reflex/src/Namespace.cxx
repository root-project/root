// @(#)root/reflex:$Name:  $:$Id: Namespace.cxx,v 1.8 2006/07/04 15:02:55 roiser Exp $
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

#include "Namespace.h"
#include "Reflex/internal/OwnedMember.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::Namespace::Namespace( const char * scop ) 
//-------------------------------------------------------------------------------
   : ScopeBase( scop, NAMESPACE ) {
   // Create dictionary info for a namespace scope.
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Namespace::Namespace() 
//-------------------------------------------------------------------------------
   : ScopeBase() {
   // Destructor.
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Namespace::InitGlobalNamespace() {
//-------------------------------------------------------------------------------
// Initialise the global namespace at startup.
   Scope s = Scope::ByName("");
   if ( ! s ) new Namespace();
}
