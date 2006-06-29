// @(#)root/reflex:$Name:  $:$Id: Namespace.cxx,v 1.6 2006/03/20 09:46:18 roiser Exp $
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
#include "Reflex/Member.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::Namespace::Namespace( const char * scop ) 
//-------------------------------------------------------------------------------
   : ScopeBase( scop, NAMESPACE ) {}


//-------------------------------------------------------------------------------
ROOT::Reflex::Namespace::Namespace() 
//-------------------------------------------------------------------------------
   : ScopeBase() {}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Namespace::InitGlobalNamespace() {
//-------------------------------------------------------------------------------
   Scope s = Scope::ByName("");
   if ( ! s ) new Namespace();
}
