// @(#)root/reflex:$Name:  $:$Id: ScopeName.cxx,v 1.2 2005/11/03 15:24:40 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/ScopeName.h"

#include "Reflex/Scope.h"
#include "Reflex/ScopeBase.h"

#include "Reflex/Tools.h"

#include "stl_hash.h"
#include <vector>


//-------------------------------------------------------------------------------
typedef __gnu_cxx::hash_map < const char *, ROOT::Reflex::ScopeName * > Name2Scope;
typedef std::vector< ROOT::Reflex::Scope > ScopeVec;

//-------------------------------------------------------------------------------
Name2Scope & sScopes() {
//-------------------------------------------------------------------------------
  static Name2Scope m;
  return m;
}


//-------------------------------------------------------------------------------
ScopeVec & sScopeVec() {
//-------------------------------------------------------------------------------
  static ScopeVec m;
  return m;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::ScopeName::ScopeName( const char * name,
                                    ScopeBase * scopeBase ) 
//-------------------------------------------------------------------------------
  : fName(name),
    fScopeBase(scopeBase) {
      sScopes() [ fName.c_str() ] = this;
      sScopeVec().push_back(Scope(this));
      //---Build recursively the declaring scopeNames
      if( fName != "" ) {
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
  Name2Scope::iterator it = sScopes().find(name.c_str());
  if (it != sScopes().end() ) return Scope( it->second );
  else                        return Scope();
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


