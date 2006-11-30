// @(#)root/reflex:$Name:  $:$Id: NameLookup.cxx,v 1.14 2006/11/09 08:01:04 roiser Exp $
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
#include "Reflex/Tools.h"
#include "Reflex/internal/OwnedMember.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::NameLookup::NameLookup(const std::string& name, const Scope& current):
   fCurrentScope(current), fLookupName(name), fPartialSuccess(false), 
   fPosNamePart(0), fPosNamePartLen(std::string::npos) {
   // Initialize a NameLookup object used internally to keep track of lookup
   // states.
   }
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
ROOT::Reflex::Type
ROOT::Reflex::NameLookup::LookupType( const std::string & nam, 
                                      const Scope & current ) {
//-------------------------------------------------------------------------------
// Lookup up a (possibly scoped) type name appearing in the scope context 
// current. This is the public interface for type lookup.
   NameLookup lookup(nam, current);
   return lookup.Lookup< Type >();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope 
ROOT::Reflex::NameLookup::LookupScope( const std::string & nam,
                                         const Scope & current ) {
//-------------------------------------------------------------------------------
// Lookup up a (possibly scoped) scope name appearing in the scope context 
// current. This is the public interface for scope lookup.
   NameLookup lookup(nam, current);
   return lookup.Lookup< Scope >();
}

/*
//-------------------------------------------------------------------------------
ROOT::Reflex::Member LookupMember( const std::string & nam,
                                   const Scope & current ) {
//-------------------------------------------------------------------------------
// Lookup up a (possibly scoped) member name appearing in the scope context 
// current. This is the public interface for member lookup.
   NameLookup lookup(nam, current);

   // this will not work, member lookup is too different from type lookup...
   return lookup.Lookup<Member>();
}
*/


//-------------------------------------------------------------------------------
template< class T > T
ROOT::Reflex::NameLookup::Lookup() {
//-------------------------------------------------------------------------------
// Lookup a type using fLookupName, fCurrentScope.
   fPartialSuccess = false;
   fPosNamePart = 0;
   fPosNamePartLen = std::string::npos;
   FindNextScopePos();

   if ( fPosNamePart == 2 ) {
      fLookedAtUsingDir.clear();
      // ::A...
      fCurrentScope = Scope::GlobalScope();
      return LookupInScope< T >();
   } else
      // A...
      return LookupInUnknownScope< T >();
}


//-------------------------------------------------------------------------------
template< class T > T
ROOT::Reflex::NameLookup::LookupInScope() {
//-------------------------------------------------------------------------------
// Lookup a type in fCurrentScope.
// Checks sub-types, sub-scopes, using directives, and base classes for
// the name given by fLookupName, fPosNamePart, and fPosNamePartLen.
// If the name part is found, and another name part follows in fPosNamePart,
// LookupTypeInScope requests the scope found to lookup the next name
// part. fPartialMatch reflexts that the left part of the name was matched;
// even if the trailing part of fLookupName cannot be found, the lookup 
// cannot continue on declaring scopes and has to fail.
// A list of lookups performed in namespaces pulled in via using directives is 
// kept in fLookedAtUsingDir, to prevent infinite loops due to 
//   namespace A{using namespace B;} namespace B{using namespace A;}
// loops.
// The lookup does not take the declaration order into account; the result of
// parts of the lookup algorithm which depend on the order will be unpredictable.

   if (!fCurrentScope) return Dummy::Get< T >();
   if (fLookedAtUsingDir.find(fCurrentScope) != fLookedAtUsingDir.end())
      // prevent inf loop from
      // ns A { using ns B; } ns B {using ns A;}
      return Dummy::Get< T >();

   //Type_Iterator subtype_end( fCurrentScope.SubType_End() );
   int len = fCurrentScope.SubTypeSize();
   int i = 0;
   for ( Type_Iterator it = fCurrentScope.SubType_Begin(); /* it != subtype_end */ i<len; ++it, ++i ) {
      const Type& type = *it;
      const TypeBase *base = type.ToTypeBase();      
      if ( base ) {
         size_t pos;
         const std::string &name( base->SimpleName(pos) );
         if (fLookupName[fPosNamePart]==name[pos] &&
             0 == fLookupName.compare(fPosNamePart, fPosNamePartLen, name, pos, name.length() ) ) {
            fPartialSuccess = true;
            fLookedAtUsingDir.clear();
            FindNextScopePos();
            if (fPosNamePart == std::string::npos) return type;
            if (it->IsTypedef()) fCurrentScope = it->FinalType();
            else fCurrentScope = type;
            return LookupInScope< T >();
         }
      }
   }
   
   Scope_Iterator subscope_end(  fCurrentScope.SubScope_End() );
   for ( Scope_Iterator in = fCurrentScope.SubScope_Begin(); in != subscope_end; ++in ) {
      // only take namespaces into account - classes were checked as part of SubType
      if (in->IsNamespace()){
         const Scope &scope = *in;
         const ScopeBase *base = scope.ToScopeBase();
         if (base) {
            size_t pos;
            const std::string &name( base->SimpleName(pos) );
            
            if (fLookupName[fPosNamePart]==name[pos] && 
                0 == fLookupName.compare(fPosNamePart, fPosNamePartLen, name, pos, name.length() ) ) {
               fPartialSuccess = true;
               fLookedAtUsingDir.clear();
               FindNextScopePos();
               if (fPosNamePart == std::string::npos) return (T) (*in);
               fCurrentScope = (Scope) (*in);
               return LookupInScope< T >();
            }
         }
      }
   }

   if (fCurrentScope.UsingDirectiveSize()) {
      fLookedAtUsingDir.insert(fCurrentScope);
      Scope storeCurrentScope = fCurrentScope;
      Scope_Iterator usingscope_end(  storeCurrentScope.UsingDirective_End() );
      for ( Scope_Iterator si = storeCurrentScope.UsingDirective_Begin(); si != usingscope_end; ++si ) {
         fCurrentScope = *si;
         T t = LookupInScope< T >();
         if (fPartialSuccess) return t;
      }
      fCurrentScope = storeCurrentScope;
   }

   if (fPosNamePart == 0) { // only for "BaseClass...", not for "A::BaseClass..."
      Base_Iterator base_end( fCurrentScope.Base_End() );
      for ( Base_Iterator bi = fCurrentScope.Base_Begin(); bi != base_end; ++bi ) {
         if ( 0 == fLookupName.compare(fPosNamePart, fPosNamePartLen, (*bi).Name() ) ) {
            fPartialSuccess = true;
            fLookedAtUsingDir.clear();
            FindNextScopePos();
            if (fPosNamePart == std::string::npos) return bi->ToType();
            fCurrentScope = bi->ToType().FinalType();
            return LookupInScope< T >();
         }
      }
   }

   if (fCurrentScope.BaseSize()) {
      Scope storeCurrentScope = fCurrentScope;
      Base_Iterator base_end( storeCurrentScope.Base_End() ); 
      for ( Base_Iterator bi = storeCurrentScope.Base_Begin(); bi != base_end; ++bi ) {
         fCurrentScope = bi->ToScope();
         T t = LookupInScope< T >();
         if (fPartialSuccess) return t;
      }
      fCurrentScope = storeCurrentScope;
   }

   return Dummy::Get< T >();
}


//-------------------------------------------------------------------------------
template< class T > T
ROOT::Reflex::NameLookup::LookupInUnknownScope() {
//-------------------------------------------------------------------------------
// Lookup a type in fCurrentScope and its declaring scopes.
   for (fPartialSuccess = false; !fPartialSuccess && fCurrentScope; fCurrentScope = fCurrentScope.DeclaringScope()) {
      fLookedAtUsingDir.clear();
      T t = LookupInScope< T >();
      if (fPartialSuccess) return t;
      if (fCurrentScope.IsTopScope()) break;
   }
   return Dummy::Get< T >();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member
ROOT::Reflex::NameLookup::LookupMember( const std::string & nam, 
                                        const Scope & current ) {
//-------------------------------------------------------------------------------
// Lookup a member.
   if ( Tools::GetBasePosition(nam)) return LookupMemberQualified( nam );
   else                              return LookupMemberUnqualified( nam, current );

}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member
ROOT::Reflex::NameLookup::LookupMemberQualified( const std::string & nam ) {
//-------------------------------------------------------------------------------
// Lookup of a qualified member.
   Scope bscope = Scope::ByName(Tools::GetScopeName(nam));
   if ( bscope ) {
      return LookupMemberUnqualified( Tools::GetBaseName(nam), bscope);
   }
   else {
      return Dummy::Member();
   }

}



//-------------------------------------------------------------------------------
ROOT::Reflex::Member
ROOT::Reflex::NameLookup::LookupMemberUnqualified( const std::string & nam,
                                                   const Scope & current ) {
//-------------------------------------------------------------------------------
// Lookup of an unqualified member.
   Member m0 = current.MemberByName(nam);
   if ( m0 ) return m0;
      
   for ( Scope_Iterator si = current.UsingDirective_Begin(); si != current.UsingDirective_End(); ++si ) {
      Member m1 = LookupMember( nam, *si );
      if ( m1 ) return m1;
   }

   for ( Base_Iterator bi = current.Base_Begin(); bi != current.Base_End(); ++ bi ) {
      Member m2 = LookupMember( nam, bi->ToScope() );
      if ( m2 ) return m2;
   }
         
   if ( ! current.IsTopScope() ) return LookupMember( nam, current.DeclaringScope() );

   return Dummy::Member();

}



//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::NameLookup::AccessControl( const Type & typ,
                                                            const Scope & /* current */ ) {
//-------------------------------------------------------------------------------
// Check access .
   
   //if ( typ.IsPublic()) return true;

   //else if ( typ.IsProtected() && current.HasBase(typ.DeclaringScope()) ) return true;

   return typ;

}

//-------------------------------------------------------------------------------
void ROOT::Reflex::NameLookup::FindNextScopePos() {
//-------------------------------------------------------------------------------
// Move fPosNamePart to point to the next scope in fLookupName, updating 
// fPosNamePartLen. If fPosNamePartLen == std::string::npos, initialize
// fPosNamePart and fPosNamePartLen. If there is no next scope left, fPosNamePart
// will be set to std::string::npos and fPosNamePartLen will be set to 0.
   if (fPosNamePartLen != std::string::npos) {
      // we know the length, so jump
      fPosNamePart += fPosNamePartLen + 2;
      if (fPosNamePart > fLookupName.length()) {
         // past the string's end?
         fPosNamePart = std::string::npos;
         fPosNamePartLen = 0;
         return;
      }
   }
   else {
      // uninitialized
      // set fPosNamePartLen and check that fLookupName doesn't start with '::'
      if (fLookupName.compare(0, 2, "::") == 0)
         fPosNamePart = 2;
      else fPosNamePart = 0;
   }
   fPosNamePartLen = Tools::GetFirstScopePosition(fLookupName.substr(fPosNamePart));
   if (fPosNamePartLen == 0)
      // no next "::"
      fPosNamePartLen = fLookupName.length();
   else fPosNamePartLen -= 2; // no "::"
   // fPosNamePartLen -= fPosNamePart; No! We already look in substr()
}
