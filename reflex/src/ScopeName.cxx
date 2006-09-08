// @(#)root/reflex:$Name:  $:$Id: ScopeName.cxx,v 1.21 2006/09/05 17:13:15 roiser Exp $
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

#include "Reflex/internal/ScopeName.h"

#include "Reflex/Scope.h"
#include "Reflex/internal/ScopeBase.h"
#include "Reflex/Type.h"

#include "Reflex/Tools.h"
#include "Reflex/internal/OwnedMember.h"

#include "stl_hash.h"
#include <vector>


//-------------------------------------------------------------------------------
typedef __gnu_cxx::hash_map < const std::string *, ROOT::Reflex::Scope > Name2Scope_t;
typedef std::vector< ROOT::Reflex::Scope > ScopeVec_t;

//-------------------------------------------------------------------------------
static Name2Scope_t & sScopes() {
//-------------------------------------------------------------------------------
// Static wrapper around scope map.
   static Name2Scope_t m;
   return m;
}


//-------------------------------------------------------------------------------
static ScopeVec_t & sScopeVec() {
//-------------------------------------------------------------------------------
// Static wrapper around scope vector.
   static ScopeVec_t m;
   return m;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::ScopeName::ScopeName( const char * name,
                                    ScopeBase * scopeBase ) 
   : fName(name),
     fScopeBase(scopeBase) {
//-------------------------------------------------------------------------------
// Create the scope name dictionary info.
   fThisScope = new Scope(this);
   sScopes() [ &fName ] = *fThisScope;
   sScopeVec().push_back(*fThisScope);
   //---Build recursively the declaring scopeNames
   if( fName != "@N@I@R@V@A@N@A@" ) {
      std::string decl_name = Tools::GetScopeName(fName);
      if ( ! Scope::ByName( decl_name ).Id() )  new ScopeName( decl_name.c_str(), 0 );
   }
}


//-------------------------------------------------------------------------------
ROOT::Reflex::ScopeName::~ScopeName() {
//-------------------------------------------------------------------------------
// Destructor.
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Scope & ROOT::Reflex::ScopeName::ByName( const std::string & name ) {
//-------------------------------------------------------------------------------
// Lookup a scope by fully qualified name.
   size_t pos =  name.substr(0,2) == "::" ?  2 : 0;
   const std::string & k = name.substr(pos);
   Name2Scope_t::iterator it = sScopes().find(&k);
   if (it != sScopes().end() ) return it->second;
   //else                        return Dummy::Scope();
   // HERE STARTS AN UGLY HACK WHICH HAS TO BE UNDONE ASAP
   // (also remove inlcude Reflex/Type.h)
   Type t = Type::ByName(name);
   if ( t && t.IsTypedef()) {
      while ( t.IsTypedef()) t = t.ToType();
      if ( t.IsClass() || t.IsEnum() || t.IsUnion() ) return t.operator const Scope &();
   }
   return Dummy::Scope();
   // END OF UGLY HACK
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeName::CleanUp() {
//-------------------------------------------------------------------------------
   // Cleanup memory allocations for scopes.
   ScopeVec_t::iterator it;
   for ( it = sScopeVec().begin(); it != sScopeVec().end(); ++it ) {
      Scope * s = ((ScopeName*)it->Id())->fThisScope;
      if ( *s ) s->Unload();
      delete s;
   }
   for ( it = sScopeVec().begin(); it != sScopeVec().end(); ++it ) {
      delete ((ScopeName*)it->Id());
   }
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeName::DeleteScope() const {
//-------------------------------------------------------------------------------
// Delete the scope base information.
   delete fScopeBase;
   fScopeBase = 0;
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ScopeName::HideName() {
//-------------------------------------------------------------------------------
// Append the string " @HIDDEN@" to a scope name.
   if ( fName[fName.length()] != '@' ) {
      sScopes().erase(&fName);
      fName += " @HIDDEN@";
      sScopes()[&fName] = this;
   }
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Scope & ROOT::Reflex::ScopeName::ThisScope() const {
//-------------------------------------------------------------------------------
// Return the scope corresponding to this scope.
   return *fThisScope;
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Scope & ROOT::Reflex::ScopeName::ScopeAt( size_t nth ) {
//-------------------------------------------------------------------------------
// Return the nth scope defined in Reflex.
   if ( nth < sScopeVec().size()) return sScopeVec()[nth];
   return Dummy::Scope();
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::ScopeName::ScopeSize() {
//-------------------------------------------------------------------------------
// Return the number of scopes defined in Reflex.
   return sScopeVec().size();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope_Iterator ROOT::Reflex::ScopeName::Scope_Begin() {
//-------------------------------------------------------------------------------
// Return the begin iterator of the scope collection.
   return sScopeVec().begin();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope_Iterator ROOT::Reflex::ScopeName::Scope_End() {
//-------------------------------------------------------------------------------
// Return the end iterator of the scope collection.
   return sScopeVec().end();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_Scope_Iterator ROOT::Reflex::ScopeName::Scope_RBegin() {
//-------------------------------------------------------------------------------
// Return the rbegin iterator of the scope collection.
   return ((const std::vector<Scope>&)sScopeVec()).rbegin();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Reverse_Scope_Iterator ROOT::Reflex::ScopeName::Scope_REnd() {
//-------------------------------------------------------------------------------
// Return the rend iterator of the scope collection.
   return ((const std::vector<Scope>&)sScopeVec()).rend();
}


