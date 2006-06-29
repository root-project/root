// @(#)root/reflex:$Name:  $:$Id: ScopeName.cxx,v 1.13 2006/06/29 15:50:48 roiser Exp $
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

#include "Reflex/ScopeName.h"

#include "Reflex/Scope.h"
#include "Reflex/ScopeBase.h"
#include "Reflex/Type.h"

#include "Reflex/Tools.h"
#include "Reflex/Member.h"

#include "stl_hash.h"
#include <vector>


//-------------------------------------------------------------------------------
typedef __gnu_cxx::hash_map < const char *, ROOT::Reflex::ScopeName * > Name2Scope_t;
typedef std::vector< ROOT::Reflex::Scope > ScopeVec_t;

//-------------------------------------------------------------------------------
static Name2Scope_t & sScopes() {
//-------------------------------------------------------------------------------
   static Name2Scope_t m;
   return m;
}


//-------------------------------------------------------------------------------
static ScopeVec_t & sScopeVec() {
//-------------------------------------------------------------------------------
   static ScopeVec_t m;
   return m;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::ScopeName::ScopeName( const char * name,
                                    ScopeBase * scopeBase ) 
   : fName(name),
     fScopeBase(scopeBase) {
//-------------------------------------------------------------------------------
   sScopes() [ fName.c_str() ] = this;
   sScopeVec().push_back(Scope(this));
   //---Build recursively the declaring scopeNames
   if( fName != "@N@I@R@V@A@N@A@" ) {
      std::string decl_name = Tools::GetScopeName(fName);
      if ( ! Scope::ByName( decl_name ).Id() )  new ScopeName( decl_name.c_str(), 0 );
   }
}


//-------------------------------------------------------------------------------
ROOT::Reflex::ScopeName::~ScopeName() {
//-------------------------------------------------------------------------------
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope ROOT::Reflex::ScopeName::ByName( const std::string & name ) {
//-------------------------------------------------------------------------------
   size_t pos =  name.substr(0,2) == "::" ?  2 : 0;
   Name2Scope_t::iterator it = sScopes().find(name.substr(pos).c_str());
   if (it != sScopes().end() ) return Scope( it->second );
   //else                        return Scope();
   // HERE STARTS AN UGLY HACK WHICH HAS TO BE UNDONE ASAP
   // (also remove inlcude Reflex/Type.h)
   Type t = Type::ByName(name);
   if ( t && t.IsTypedef()) {
      while ( t.IsTypedef()) t = t.ToType();
      if ( t.IsClass() || t.IsEnum() || t.IsUnion() ) return (Scope)t;
   }
   return Scope();
   // END OF UGLY HACK
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope ROOT::Reflex::ScopeName::ThisScope() const {
//-------------------------------------------------------------------------------
   return Scope( this );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope ROOT::Reflex::ScopeName::ScopeAt( size_t nth ) {
//-------------------------------------------------------------------------------
   if ( nth < sScopeVec().size()) return sScopeVec()[nth];
   return Scope();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::ScopeName::ScopeSize() {
//-------------------------------------------------------------------------------
   return sScopeVec().size();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope_Iterator ROOT::Reflex::ScopeName::Scope_Begin() {
//-------------------------------------------------------------------------------
   return sScopeVec().begin();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope_Iterator ROOT::Reflex::ScopeName::Scope_End() {
//-------------------------------------------------------------------------------
   return sScopeVec().end();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_Scope_Iterator ROOT::Reflex::ScopeName::Scope_RBegin() {
//-------------------------------------------------------------------------------
   return sScopeVec().rbegin();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_Scope_Iterator ROOT::Reflex::ScopeName::Scope_REnd() {
//-------------------------------------------------------------------------------
   return sScopeVec().rend();
}


