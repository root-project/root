// @(#)root/reflex:$Name:  $:$Id: NameLookup.cxx,v 1.6 2006/08/01 10:28:45 roiser Exp $
// Author: Stefan Roiser 2006

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

#include "NameLookup.h"
#include "Reflex/Base.h"
#include "Reflex/Scope.h"
#include "Reflex/Type.h"
#include "Reflex/internal/OwnedMember.h"

//-------------------------------------------------------------------------------
const ROOT::Reflex::Type &
ROOT::Reflex::NameLookup::LookupType( const std::string & nam, 
                                      const Scope & current ) {
//-------------------------------------------------------------------------------
// Lookup up a type name.
   if ( nam.find("::") == 0 ) {
      std::set<Scope> lookedAtUsingDir;
      bool partial_success = false;
      return LookupTypeInScope( nam.substr(2), Scope::GlobalScope(),  partial_success, lookedAtUsingDir);
   } else
      return LookupTypeInUnknownScope( nam, current );

}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Type &
ROOT::Reflex::NameLookup::LookupTypeInScope( const std::string & nam,
                                             const Scope & current,
                                             bool &partial_success,
                                             std::set<Scope> & lookedAtUsingDir,
                                             size_t pos_subscope /*= 0*/,
                                             size_t pos_next_scope /*= npos*/ ) {
//-------------------------------------------------------------------------------
// Lookup a type in a scope.
   if (!current) return Dummy::Type();
   if (lookedAtUsingDir.find(current) != lookedAtUsingDir.end()) return Dummy::Type();

   if (pos_next_scope == std::string::npos) {
      pos_next_scope = nam.find("::", pos_subscope);
      if (pos_next_scope == std::string::npos) pos_next_scope = 0;
      else                                     pos_next_scope += 2;
   }
   std::string scopenam = nam.substr(pos_subscope, pos_next_scope == 0 ? std::string::npos : pos_next_scope - 2 - pos_subscope);
   size_t pos_current = pos_subscope;
   pos_subscope = pos_next_scope;
   if (pos_next_scope) {
      pos_next_scope = nam.find("::", pos_next_scope);
      if (pos_next_scope == std::string::npos) pos_next_scope = 0;
      else                                     pos_next_scope += 2;
   }

   for ( Type_Iterator it = current.SubType_Begin(); it != current.SubType_End(); ++it ) {
      if ( (*it).Name() == scopenam ) {
         partial_success = true;
         lookedAtUsingDir.clear();
         if (!pos_subscope) return *it;
         if (it->IsTypedef())
            return LookupTypeInScope(nam, it->FinalType(), partial_success, lookedAtUsingDir, pos_subscope, pos_next_scope);
         else
            return LookupTypeInScope(nam, *it, partial_success, lookedAtUsingDir, pos_subscope, pos_next_scope);
      }
   }

   for ( Scope_Iterator in = current.SubScope_Begin(); in != current.SubScope_End(); ++in ) {
      if (in->IsNamespace() && (*in).Name() == scopenam ) {
         partial_success = true;
         lookedAtUsingDir.clear();
         if (!pos_subscope) return Dummy::Type(); // namespace is no a type
         return LookupTypeInScope(nam, *in, partial_success, lookedAtUsingDir, pos_subscope, pos_next_scope);
      }
   }

   lookedAtUsingDir.insert(current);
   for ( Scope_Iterator si = current.UsingDirective_Begin(); si != current.UsingDirective_End(); ++si ) {
      const Type & t = LookupTypeInScope( nam, *si, partial_success, lookedAtUsingDir, pos_current, pos_subscope);
      if (t || partial_success) return t;
   }

   if (pos_current == 0) // only for "BaseClass::whatever", not for "CurrentScope::BaseClass::whatever"!
      for ( Base_Iterator bi = current.Base_Begin(); bi != current.Base_End(); ++bi ) {
         if ( (*bi).Name() == scopenam ) {
            partial_success = true;
            lookedAtUsingDir.clear();
            if (!pos_subscope) return bi->ToType();
            return LookupTypeInScope(nam, bi->ToType().FinalType(), partial_success, lookedAtUsingDir, pos_subscope, pos_next_scope);
         }
      }

   for ( Base_Iterator bi = current.Base_Begin(); bi != current.Base_End(); ++bi ) {
      const Type & t = LookupTypeInScope( nam, bi->ToScope(), partial_success, lookedAtUsingDir, pos_current, pos_subscope );
      if ( t || partial_success) return t;
   }

   return Dummy::Type();
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Type &
ROOT::Reflex::NameLookup::LookupTypeInUnknownScope( const std::string & nam,
                                                    const Scope & current ) {
//-------------------------------------------------------------------------------
// Lookup a type in an unknown scope.
   bool partial_success = false;
   std::set<Scope> lookedAtUsingDir;
   const Type & t = LookupTypeInScope( nam, current, partial_success, lookedAtUsingDir);
   if ( t || partial_success) return t;
   if ( ! current.IsTopScope() ) return LookupTypeInUnknownScope( nam, current.DeclaringScope() );
   return t;
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Member &
ROOT::Reflex::NameLookup::LookupMember( const std::string & nam, 
                                        const Scope & current ) {
//-------------------------------------------------------------------------------
// Lookup a member.
   if ( Tools::GetBasePosition(nam)) return LookupMemberQualified( nam );
   else                              return LookupMemberUnqualified( nam, current );

}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Member &
ROOT::Reflex::NameLookup::LookupMemberQualified( const std::string & nam ) {
//-------------------------------------------------------------------------------
// Lookup of a qualified member.
   const Scope & bscope = Scope::ByName(Tools::GetScopeName(nam));
   if ( bscope ) {
      return LookupMemberUnqualified( Tools::GetBaseName(nam), bscope);
   }
   else {
      return Dummy::Member();
   }

}



//-------------------------------------------------------------------------------
const ROOT::Reflex::Member &
ROOT::Reflex::NameLookup::LookupMemberUnqualified( const std::string & nam,
                                                   const Scope & current ) {
//-------------------------------------------------------------------------------
// Lookup of an unqualified member.
   const Member & m0 = current.MemberByName(nam);
   if ( m0 ) return m0;
      
   for ( Scope_Iterator si = current.UsingDirective_Begin(); si != current.UsingDirective_End(); ++si ) {
      const Member & m1 = LookupMember( nam, *si );
      if ( m1 ) return m1;
   }

   for ( Base_Iterator bi = current.Base_Begin(); bi != current.Base_End(); ++ bi ) {
      const Member & m2 = LookupMember( nam, bi->ToScope() );
      if ( m2 ) return m2;
   }
         
   if ( ! current.IsTopScope() ) return LookupMember( nam, current.DeclaringScope() );

   return Dummy::Member();

}



//-------------------------------------------------------------------------------
const ROOT::Reflex::Type & ROOT::Reflex::NameLookup::AccessControl( const Type & typ,
                                                                    const Scope & /* current */ ) {
//-------------------------------------------------------------------------------
// Check access .
   
   //if ( typ.IsPublic()) return true;

   //else if ( typ.IsProtected() && current.HasBase(typ.DeclaringScope()) ) return true;

   return typ;

}
