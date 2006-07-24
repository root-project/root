// @(#)root/reflex:$Name: HEAD $:$Id: NameLookup.cxx,v 1.4 2006/07/05 07:09:09 roiser Exp $
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
#include "Reflex/Member.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::Type
ROOT::Reflex::NameLookup::LookupType( const std::string & nam, 
                                      const Scope & current ) {
//-------------------------------------------------------------------------------
   Type t;

   if ( nam.find("::") == 0 ) {
      std::set<Scope> lookedAtUsingDir;
      bool partial_success = false;
      t = LookupTypeInScope( nam.substr(2), Scope::GlobalScope(),  partial_success, lookedAtUsingDir);
   } else
      t = LookupTypeInUnknownScope( nam, current );

   //if ( t && AccessControl(t, current)) return t;
   //else                                 return Type();
   return t;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type
ROOT::Reflex::NameLookup::LookupTypeInScope( const std::string & nam,
                                             const Scope & current,
                                             bool &partial_success,
                                             std::set<Scope> & lookedAtUsingDir,
                                             size_t pos_subscope /*= 0*/,
                                             size_t pos_next_scope /*= npos*/ ) {
//-------------------------------------------------------------------------------

   if (!current) return Type();
   if (lookedAtUsingDir.find(current) != lookedAtUsingDir.end()) return Type();

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
         if (!pos_subscope) return Type(); // namespace is no a type
         return LookupTypeInScope(nam, *in, partial_success, lookedAtUsingDir, pos_subscope, pos_next_scope);
      }
   }

   lookedAtUsingDir.insert(current);
   for ( Scope_Iterator si = current.UsingDirective_Begin(); si != current.UsingDirective_End(); ++si ) {
      Type t = LookupTypeInScope( nam, *si, partial_success, lookedAtUsingDir, pos_current, pos_subscope);
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
      Type t = LookupTypeInScope( nam, bi->ToScope(), partial_success, lookedAtUsingDir, pos_current, pos_subscope );
      if ( t || partial_success) return t;
   }

   return Type();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type
ROOT::Reflex::NameLookup::LookupTypeInUnknownScope( const std::string & nam,
                                                    const Scope & current ) {
//-------------------------------------------------------------------------------

   bool partial_success = false;
   std::set<Scope> lookedAtUsingDir;
   Type t = LookupTypeInScope( nam, current, partial_success, lookedAtUsingDir);
   if ( t || partial_success) return t;
   if ( ! current.IsTopScope() ) t = LookupTypeInUnknownScope( nam, current.DeclaringScope() );
   return t;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member
ROOT::Reflex::NameLookup::LookupMember( const std::string & nam, 
                                        const Scope & current ) {
//-------------------------------------------------------------------------------

   Member m = Member();

   if ( Tools::GetBasePosition(nam)) m = LookupMemberQualified( nam );
   else                              m = LookupMemberUnqualified( nam, current );

   //if ( m && AccessControl(m.TypeOf(), current)) return m;
   //else                                          return Member();
   return m;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member
ROOT::Reflex::NameLookup::LookupMemberQualified( const std::string & nam ) {
//-------------------------------------------------------------------------------

   Scope bscope = Scope::ByName(Tools::GetScopeName(nam));
   if ( bscope ) {
      return LookupMemberUnqualified( Tools::GetBaseName(nam), bscope);
   }
   else {
      return Member();
   }

}



//-------------------------------------------------------------------------------
ROOT::Reflex::Member
ROOT::Reflex::NameLookup::LookupMemberUnqualified( const std::string & nam,
                                                   const Scope & current ) {
//-------------------------------------------------------------------------------

   Member m = Member();

   m = current.MemberByName(nam);
   if ( m ) return m;
      
   for ( Scope_Iterator si = current.UsingDirective_Begin(); si != current.UsingDirective_End(); ++si ) {
      m = LookupMember( nam, *si );
      if ( m ) return m;
   }

   for ( Base_Iterator bi = current.Base_Begin(); bi != current.Base_End(); ++ bi ) {
      m = LookupMember( nam, bi->ToScope() );
      if ( m ) return m;
   }
         
   if ( ! current.IsTopScope() ) m = LookupMember( nam, current.DeclaringScope() );

   return m;

}



//-------------------------------------------------------------------------------
bool ROOT::Reflex::NameLookup::AccessControl( const Type & /* typ */,
                                              const Scope & /* current */ ) {
//-------------------------------------------------------------------------------

   
   //if ( typ.IsPublic()) return true;

   //else if ( typ.IsProtected() && current.HasBase(typ.DeclaringScope()) ) return true;

   return false;

}
