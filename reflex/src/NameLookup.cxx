// @(#)root/reflex:$Name:  $:$Id: NameLookup.cxx,v 1.1 2006/06/08 16:05:14 roiser Exp $
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
   Type t = Type();

   if ( Tools::GetBasePosition(nam)) t = LookupTypeQualified( nam );
   else                              t = LookupTypeUnqualified( nam, current );

   return t;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type
ROOT::Reflex::NameLookup::LookupTypeQualified( const std::string & nam ) {
//-------------------------------------------------------------------------------

   Scope bscope = Scope::ByName(Tools::GetScopeName(nam));
   if ( bscope ) {
      return LookupType( Tools::GetBaseName(nam), bscope);
   }
   else { 
      return Type(); 
   }

}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type
ROOT::Reflex::NameLookup::LookupTypeUnqualified( const std::string & nam,
                                                 const Scope & current ) {
//-------------------------------------------------------------------------------

   Type t = Type();

   for ( Type_Iterator it = current.SubType_Begin(); it != current.SubType_End(); ++it ) {
      if ( (*it).Name() == nam ) return *it;
   }

   for ( Scope_Iterator si = current.UsingDirective_Begin(); si != current.UsingDirective_End(); ++si ) {
      t = LookupType( nam, *si );
      if ( t ) return t;
   }

   for ( Base_Iterator bi = current.Base_Begin(); bi != current.Base_End(); ++bi ) {
      t = LookupType( nam, bi->ToScope() );
      if ( t ) return t;
   }

   if ( ! current.IsTopScope() ) t = LookupType( nam, current.DeclaringScope() );

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
